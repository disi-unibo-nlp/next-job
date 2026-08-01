import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from jobhop_eval_common import (
    build_texts,
    cleanup_cuda,
    configure_hf_cache,
    configure_reproducibility,
    evaluate_ranked_lists_at_k,
    failure_payload,
    gpu_info,
    load_jobhop_eval_data,
    model_slug,
    notebook_conversion_changes,
    now_utc,
    package_versions,
    parse_optional_int,
    parse_k_values,
    select_alpha_by_validation_mrr,
    train_markov_ranked,
    write_csv,
    write_json,
    write_json_and_key_value_csv,
)


DEFAULT_MODEL_NAME = "lightonai/LateOn"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
INDEX_NAME = "esco_occupations"


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster-ready JobHop v2 LightOn LateOn evaluation.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="PyLate ColBERT model name or local path.")
    parser.add_argument("--max-eval-cases", default="2000", help="Integer sample size per eval split, or 'none'.")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16, help="Query batch size.")
    parser.add_argument("--doc-batch-size", type=int, default=32, help="Candidate document batch size.")
    parser.add_argument("--retrieve-k", type=int, default=200, help="LateOn and Markov candidate depth.")
    parser.add_argument("--metric-ks", default="1,3,5,10,20,50,100", help="Comma-separated K values for IR metrics.")
    parser.add_argument("--output-dir", default="outputs/lateon_eval")
    parser.add_argument("--cache-dir", default=None, help="Shared Hugging Face cache directory.")
    parser.add_argument("--index-dir", default=None, help="Optional PyLate index directory.")
    return parser.parse_args()


def parse_retrieved_row(row):
    parsed = []
    if isinstance(row, dict):
        ids = row.get("documents_ids") or row.get("document_ids") or row.get("ids")
        scores = row.get("scores") or row.get("score")
        if ids is not None:
            if scores is None:
                scores = [1.0] * len(ids)
            return [(str(doc_id), float(score)) for doc_id, score in zip(ids, scores)]
    if isinstance(row, (list, tuple)):
        for item in row:
            if isinstance(item, dict):
                doc_id = item.get("id") or item.get("document_id") or item.get("documents_id")
                score = item.get("score", item.get("similarity", 0.0))
                if doc_id is not None:
                    parsed.append((str(doc_id), float(score)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                parsed.append((str(item[0]), float(item[1])))
    return parsed


def retrieve_lateon(colbert_model, retriever, query_texts, batch_size, retrieve_k):
    query_embeddings = colbert_model.encode(
        query_texts,
        batch_size=batch_size,
        is_query=True,
        show_progress_bar=True,
    )
    raw_rows = retriever.retrieve(queries_embeddings=query_embeddings, k=retrieve_k)
    parsed = [parse_retrieved_row(row) for row in raw_rows]
    warnings = []
    if parsed and not parsed[0]:
        warnings.append({"row": 0, "message": "retrieval output parser returned an empty first row", "raw": str(raw_rows[0])})
    return parsed, warnings


def top_markov_scores(case, transition_probs, global_rank, k):
    last = case["prefix"][-1]
    ranked = transition_probs.get(last, global_rank)
    return {code: float(score) for code, score in ranked[:k]}


def minmax_scores(score_dict):
    if not score_dict:
        return {}
    values = np.array(list(score_dict.values()), dtype="float32")
    mn, mx = float(values.min()), float(values.max())
    denom = max(mx - mn, 1e-8)
    return {key: (float(value) - mn) / denom for key, value in score_dict.items()}


def hybrid_ranked_lists(cases, lateon_ranked, transition_probs, global_rank, alpha, retrieve_k):
    outputs = []
    for case, late_row in zip(cases, lateon_ranked):
        late_scores = {code: score for code, score in late_row[:retrieve_k]}
        markov_scores = top_markov_scores(case, transition_probs, global_rank, retrieve_k)
        union = set(late_scores) | set(markov_scores)

        late_norm = minmax_scores({code: late_scores.get(code, 0.0) for code in union})
        markov_norm = minmax_scores({code: markov_scores.get(code, 0.0) for code in union})

        combined = []
        for code in union:
            score = alpha * markov_norm.get(code, 0.0) + (1.0 - alpha) * late_norm.get(code, 0.0)
            combined.append((code, score))
        outputs.append(sorted(combined, key=lambda item: item[1], reverse=True))
    return outputs


def main():
    args = parse_args()
    started_at = now_utc()
    started_monotonic = time.monotonic()
    stage = "initialization"
    max_eval_cases = parse_optional_int(args.max_eval_cases)
    metric_ks = parse_k_values(args.metric_ks)
    retrieve_k = max(10, args.retrieve_k, max(metric_ks))
    cache_dir = configure_hf_cache(args.cache_dir)
    configure_reproducibility(args.random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(args.output_dir) / model_slug(args.model_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    index_dir = Path(args.index_dir) if args.index_dir else run_dir / "pylate_index"

    run_config = {
        "script": "scripts/lateon_eval.py",
        "model_name": args.model_name,
        "max_eval_cases": max_eval_cases,
        "random_seed": args.random_seed,
        "batch_size_queries": args.batch_size,
        "batch_size_docs": args.doc_batch_size,
        "retrieve_k": retrieve_k,
        "metric_ks": metric_ks,
        "output_dir": str(run_dir),
        "cache_dir": cache_dir,
        "index_dir": str(index_dir),
        "device": device,
        "alphas": ALPHAS,
        "alpha_selection": "validation MRR only; first alpha in sweep order wins ties",
    }
    write_json_and_key_value_csv(run_dir / "run_config.json", run_dir / "run_config.csv", run_config)
    write_json(run_dir / "notebook_conversion_changes.json", notebook_conversion_changes("lateon"))
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
        transition_probs, global_rank = train_markov_ranked(data["trajectories"]["train"])

        stage = "lateon_import"
        from pylate import indexes, models, retrieve

        stage = "index_preparation"
        if index_dir.exists():
            shutil.rmtree(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        stage = "model_loading"
        colbert_model = models.ColBERT(model_name_or_path=args.model_name)

        stage = "candidate_encoding_and_indexing"
        document_embeddings = colbert_model.encode(
            texts["candidate_texts"],
            batch_size=args.doc_batch_size,
            is_query=False,
            show_progress_bar=True,
        )
        index = indexes.PLAID(index_folder=str(index_dir), index_name=INDEX_NAME, override=True)
        index.add_documents(documents_ids=data["candidate_codes"], documents_embeddings=document_embeddings)
        retriever = retrieve.ColBERT(index=index)

        stage = "validation_retrieval"
        validation_lateon, validation_warnings = retrieve_lateon(
            colbert_model, retriever, texts["validation_query_texts"], args.batch_size, retrieve_k
        )
        write_json(run_dir / "retrieval_parse_warnings.json", {"validation": validation_warnings, "test": []})

        stage = "validation_sweep"
        validation_rows = []
        for alpha in ALPHAS:
            ranked = hybrid_ranked_lists(
                data["validation_cases"], validation_lateon, transition_probs, global_rank, alpha, retrieve_k
            )
            scores = evaluate_ranked_lists_at_k(data["validation_cases"], ranked, metric_ks)
            row = {
                "model": args.model_name,
                "split": "validation",
                "alpha_markov": alpha,
                "alpha_lateon": 1.0 - alpha,
                "retrieve_k": retrieve_k,
                "max_eval_cases": max_eval_cases,
                **scores,
            }
            validation_rows.append(row)
            write_csv(run_dir / "validation_sweep.csv", validation_rows)
            write_json(run_dir / "validation_sweep.json", validation_rows)
        selected_alpha, selected_validation_mrr = select_alpha_by_validation_mrr(validation_rows, ALPHAS)
        selected_alpha_payload = {
            "selected_alpha_markov": selected_alpha,
            "selected_alpha_lateon": 1.0 - selected_alpha,
            "validation_MRR": selected_validation_mrr,
            "selection_rule": "maximum validation MRR only; first sweep alpha wins ties",
        }
        write_json_and_key_value_csv(
            run_dir / "selected_alpha.json", run_dir / "selected_alpha.csv", selected_alpha_payload
        )

        stage = "test_retrieval_and_final_eval"
        test_lateon, test_warnings = retrieve_lateon(
            colbert_model, retriever, texts["test_query_texts"], args.batch_size, retrieve_k
        )
        write_json(run_dir / "retrieval_parse_warnings.json", {"validation": validation_warnings, "test": test_warnings})
        test_ranked = hybrid_ranked_lists(
            data["test_cases"], test_lateon, transition_probs, global_rank, selected_alpha, retrieve_k
        )
        final_metrics = evaluate_ranked_lists_at_k(data["test_cases"], test_ranked, metric_ks)
        final_row = {
            "model": args.model_name,
            "split": "test",
            "alpha_markov": selected_alpha,
            "alpha_lateon": 1.0 - selected_alpha,
            "selected_by": "validation_MRR",
            "selected_validation_MRR": selected_validation_mrr,
            "retrieve_k": retrieve_k,
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
