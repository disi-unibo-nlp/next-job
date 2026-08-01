import argparse
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from jobhop_eval_common import (
    build_texts,
    cleanup_cuda,
    configure_hf_cache,
    configure_reproducibility,
    evaluate_score_matrix,
    failure_payload,
    gpu_info,
    load_jobhop_eval_data,
    markov_score_matrix,
    model_slug,
    notebook_conversion_changes,
    now_utc,
    package_versions,
    parse_optional_int,
    row_maxnorm,
    row_minmax,
    select_alpha_by_validation_mrr,
    train_markov_dense,
    write_csv,
    write_json,
    write_json_and_key_value_csv,
)


DENSE_MODEL_DEFAULTS = {
    "BAAI/bge-m3": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "batch_size": 32,
    },
    "Qwen/Qwen3-Embedding-8B": {
        "query_prefix": "Instruct: Given a career history, retrieve the most plausible next ESCO occupation.\nQuery: ",
        "passage_prefix": "",
        "batch_size": 4,
    },
    "nvidia/Nemotron-3-Embed-8B-BF16": {
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "batch_size": 2,
    },
}

ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster-ready JobHop v2 dense embedding evaluation.")
    parser.add_argument("--model-name", required=True, help="SentenceTransformer model name or local path.")
    parser.add_argument("--max-eval-cases", default="2000", help="Integer sample size per eval split, or 'none'.")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--retrieve-k", type=int, default=10, help="Ranking depth used to extract metrics; >=10.")
    parser.add_argument("--output-dir", default="outputs/dense_eval")
    parser.add_argument("--cache-dir", default=None, help="Shared Hugging Face cache directory.")
    parser.add_argument("--query-prefix", default=None, help="Override model-specific query prefix.")
    parser.add_argument("--passage-prefix", default=None, help="Override model-specific passage prefix.")
    return parser.parse_args()


def apply_prefixes(texts, prefix):
    return [prefix + text for text in texts] if prefix else texts


def load_sentence_transformer_model(model_name, device):
    try:
        return SentenceTransformer(model_name, device=device, trust_remote_code=True)
    except TypeError:
        return SentenceTransformer(model_name, device=device, model_kwargs={"trust_remote_code": True})


def encode_texts(model, texts, prefix, batch_size):
    formatted = apply_prefixes(texts, prefix)
    embeddings = model.encode(
        formatted,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def main():
    args = parse_args()
    started_at = now_utc()
    started_monotonic = time.monotonic()
    stage = "initialization"
    max_eval_cases = parse_optional_int(args.max_eval_cases)
    cache_dir = configure_hf_cache(args.cache_dir)
    configure_reproducibility(args.random_seed)

    defaults = DENSE_MODEL_DEFAULTS.get(args.model_name, {})
    batch_size = args.batch_size or defaults.get("batch_size", 16)
    query_prefix = args.query_prefix if args.query_prefix is not None else defaults.get("query_prefix", "")
    passage_prefix = args.passage_prefix if args.passage_prefix is not None else defaults.get("passage_prefix", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(args.output_dir) / model_slug(args.model_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "script": "scripts/dense_eval.py",
        "model_name": args.model_name,
        "max_eval_cases": max_eval_cases,
        "random_seed": args.random_seed,
        "batch_size": batch_size,
        "retrieve_k": args.retrieve_k,
        "output_dir": str(run_dir),
        "cache_dir": cache_dir,
        "device": device,
        "alphas": ALPHAS,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "alpha_selection": "validation MRR only; first alpha in sweep order wins ties",
    }
    write_json_and_key_value_csv(run_dir / "run_config.json", run_dir / "run_config.csv", run_config)
    write_json(run_dir / "notebook_conversion_changes.json", notebook_conversion_changes("dense"))
    write_json_and_key_value_csv(run_dir / "gpu_info.json", run_dir / "gpu_info.csv", gpu_info())
    write_json_and_key_value_csv(run_dir / "package_versions.json", run_dir / "package_versions.csv", package_versions())

    runtime = {"started_at_utc": started_at, "status": "running", "stage": stage}
    write_json_and_key_value_csv(run_dir / "runtime.json", run_dir / "runtime.csv", runtime)

    try:
        stage = "data_loading_and_preprocessing"
        data = load_jobhop_eval_data(max_eval_cases=max_eval_cases, seed=args.random_seed)
        write_csv(run_dir / "dataset_statistics.csv", data["stats_rows"])
        write_json(run_dir / "dataset_statistics.json", data["stats_rows"])
        write_json_and_key_value_csv(run_dir / "dataset_summary.json", run_dir / "dataset_summary.csv", data["stats_summary"])
        write_json_and_key_value_csv(
            run_dir / "preprocessing_config.json", run_dir / "preprocessing_config.csv", data["preprocessing_config"]
        )
        write_json(run_dir / "data_files.json", {"parquet_files": data["parquet_files"], "split_files": data["split_files"]})

        texts = build_texts(
            data["metadata_lookup"],
            data["candidate_codes"],
            data["validation_cases"],
            data["test_cases"],
        )

        stage = "markov_baseline"
        transition_probs, global_probs, code_to_index = train_markov_dense(
            data["trajectories"]["train"],
            data["candidate_codes"],
        )
        validation_markov_norm = row_maxnorm(
            markov_score_matrix(data["validation_cases"], data["candidate_codes"], transition_probs, global_probs)
        )
        test_markov_norm = row_maxnorm(
            markov_score_matrix(data["test_cases"], data["candidate_codes"], transition_probs, global_probs)
        )

        stage = "model_loading"
        model = load_sentence_transformer_model(args.model_name, device=device)

        stage = "candidate_encoding"
        candidate_embeddings = encode_texts(model, texts["candidate_texts"], passage_prefix, batch_size)

        stage = "validation_encoding"
        validation_embeddings = encode_texts(model, texts["validation_query_texts"], query_prefix, batch_size)
        validation_embedding_norm = row_minmax(validation_embeddings @ candidate_embeddings.T)

        stage = "validation_sweep"
        validation_rows = []
        for alpha in ALPHAS:
            final_scores = alpha * validation_markov_norm + (1.0 - alpha) * validation_embedding_norm
            scores = evaluate_score_matrix(final_scores, data["validation_cases"], code_to_index, args.retrieve_k)
            row = {
                "model": args.model_name,
                "split": "validation",
                "alpha_markov": alpha,
                "alpha_embedding": 1.0 - alpha,
                "max_eval_cases": max_eval_cases,
                **scores,
            }
            validation_rows.append(row)
            write_csv(run_dir / "validation_sweep.csv", validation_rows)
            write_json(run_dir / "validation_sweep.json", validation_rows)
        selected_alpha, selected_validation_mrr = select_alpha_by_validation_mrr(validation_rows, ALPHAS)
        selected_alpha_payload = {
            "selected_alpha_markov": selected_alpha,
            "selected_alpha_embedding": 1.0 - selected_alpha,
            "validation_MRR": selected_validation_mrr,
            "selection_rule": "maximum validation MRR only; first sweep alpha wins ties",
        }
        write_json_and_key_value_csv(
            run_dir / "selected_alpha.json", run_dir / "selected_alpha.csv", selected_alpha_payload
        )

        stage = "test_encoding_and_final_eval"
        test_embeddings = encode_texts(model, texts["test_query_texts"], query_prefix, batch_size)
        test_embedding_norm = row_minmax(test_embeddings @ candidate_embeddings.T)
        test_scores = selected_alpha * test_markov_norm + (1.0 - selected_alpha) * test_embedding_norm
        final_metrics = evaluate_score_matrix(test_scores, data["test_cases"], code_to_index, args.retrieve_k)
        final_row = {
            "model": args.model_name,
            "split": "test",
            "alpha_markov": selected_alpha,
            "alpha_embedding": 1.0 - selected_alpha,
            "selected_by": "validation_MRR",
            "selected_validation_MRR": selected_validation_mrr,
            "max_eval_cases": max_eval_cases,
            **final_metrics,
        }
        write_csv(run_dir / "final_test_results.csv", [final_row])
        write_json(run_dir / "final_test_results.json", final_row)

        stage = "finalization"
        runtime.update(
            {
                "status": "success",
                "stage": stage,
                "finished_at_utc": now_utc(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
            }
        )
        write_json_and_key_value_csv(run_dir / "runtime.json", run_dir / "runtime.csv", runtime)
        cleanup_cuda()
        return 0

    except Exception as exc:
        payload = failure_payload(exc, stage, started_at)
        payload["elapsed_seconds"] = time.monotonic() - started_monotonic
        payload["run_config"] = run_config
        write_json(run_dir / "failure.json", payload)
        write_csv(run_dir / "failures.csv", [payload])
        runtime.update(payload)
        write_json_and_key_value_csv(run_dir / "runtime.json", run_dir / "runtime.csv", runtime)
        cleanup_cuda()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
