import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dense_eval import (  # noqa: E402
    ALPHAS,
    DENSE_MODEL_DEFAULTS,
    encode_texts,
    load_sentence_transformer_model,
)
from jobhop_eval_common import (  # noqa: E402
    build_texts,
    cleanup_cuda,
    configure_hf_cache,
    configure_reproducibility,
    evaluate_score_matrix_at_k,
    load_jobhop_eval_data,
    markov_score_matrix,
    now_utc,
    parse_k_values,
    require_cuda,
    row_maxnorm,
    row_minmax,
    select_alpha_by_validation_mrr,
    train_markov_dense,
    write_csv,
    write_json,
)


def rank_rows(score_matrix, cases, code_to_index, metric_ks, retrieve_k):
    top_k = min(max(max(metric_ks), int(retrieve_k)), score_matrix.shape[1])
    target_indices = np.array([code_to_index[c["target"]] for c in cases], dtype=np.int64)
    rows = []

    for start in range(0, len(cases), 512):
        end = min(start + 512, len(cases))
        chunk = score_matrix[start:end]
        order = np.argpartition(-chunk, kth=top_k - 1, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(chunk, order, axis=1)
        sorted_local = np.argsort(-top_scores, axis=1)
        ranked = np.take_along_axis(order, sorted_local, axis=1)

        for local_i, preds in enumerate(ranked):
            i = start + local_i
            target_idx = target_indices[i]
            matches = np.where(preds == target_idx)[0]
            rank = int(matches[0]) + 1 if len(matches) else None
            case = cases[i]
            rows.append({
                "row_id": i,
                "person_id": case.get("person_id"),
                "prefix_length": case.get("prefix_length"),
                "target": case["target"],
                "gold_rank": rank,
                "hit_at_1": int(rank == 1) if rank is not None else 0,
                "hit_at_10": int(rank is not None and rank <= 10),
                "mrr_at_10": (1.0 / rank) if rank is not None and rank <= 10 else 0.0,
                "ndcg_at_10": (1.0 / math.log2(rank + 1)) if rank is not None and rank <= 10 else 0.0,
            })
    return rows


def paired_bootstrap(delta, n_resamples=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(delta[idx].mean())
    observed = float(delta.mean())
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return observed, float(ci_low), float(ci_high), min(1.0, p)


def metric_vectors(rows):
    return {
        "Accuracy@1": np.array([r["hit_at_1"] for r in rows], dtype=np.float64),
        "MRR@10": np.array([r["mrr_at_10"] for r in rows], dtype=np.float64),
        "NDCG@10": np.array([r["ndcg_at_10"] for r in rows], dtype=np.float64),
        "Recall@10": np.array([r["hit_at_10"] for r in rows], dtype=np.float64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="models/BAAI_bge-m3")
    ap.add_argument("--cache-dir", default="/hf-cache")
    ap.add_argument("--output-dir", default="outputs/paper_checks/bge_m3_original_formal_per_query_full_test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--random-seed", type=int, default=7)
    ap.add_argument("--retrieve-k", type=int, default=10)
    ap.add_argument("--metric-ks", default="1,3,5,10,20,50,100")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metric_ks = parse_k_values(args.metric_ks)
    configure_hf_cache(args.cache_dir)
    configure_reproducibility(args.random_seed)
    require_cuda()

    defaults = DENSE_MODEL_DEFAULTS["BAAI/bge-m3"]
    query_prefix = defaults["query_prefix"]
    passage_prefix = defaults["passage_prefix"]

    run_config = {
        "script": "scripts/export_bge_m3_formal_per_query.py",
        "base_script_logic": "scripts/dense_eval.py + scripts/jobhop_eval_common.py",
        "model_name": args.model_name,
        "random_seed": args.random_seed,
        "batch_size": args.batch_size,
        "retrieve_k": args.retrieve_k,
        "metric_ks": metric_ks,
        "cache_dir": args.cache_dir,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "alphas": ALPHAS,
        "started_at_utc": now_utc(),
    }
    write_json(out / "run_config.json", run_config)

    print("loading formal JobHop data...", flush=True)
    data = load_jobhop_eval_data(max_eval_cases=None, seed=args.random_seed)
    texts = build_texts(
        data["metadata_lookup"],
        data["candidate_codes"],
        data["validation_cases"],
        data["test_cases"],
    )

    print("building formal Markov scores...", flush=True)
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

    print(f"loading model: {args.model_name}", flush=True)
    model = load_sentence_transformer_model(args.model_name, device="cuda")

    print("encoding candidates...", flush=True)
    candidate_embeddings = encode_texts(model, texts["candidate_texts"], passage_prefix, args.batch_size)

    print("validation alpha sweep...", flush=True)
    validation_embeddings = encode_texts(model, texts["validation_query_texts"], query_prefix, args.batch_size)
    validation_embedding_norm = row_minmax(validation_embeddings @ candidate_embeddings.T)

    validation_rows = []
    for alpha in ALPHAS:
        scores = alpha * validation_markov_norm + (1.0 - alpha) * validation_embedding_norm
        metrics = evaluate_score_matrix_at_k(scores, data["validation_cases"], code_to_index, metric_ks, args.retrieve_k)
        validation_rows.append({
            "model": args.model_name,
            "split": "validation",
            "alpha_markov": alpha,
            "alpha_embedding": 1.0 - alpha,
            **metrics,
        })
    write_csv(out / "validation_sweep.csv", validation_rows)
    write_json(out / "validation_sweep.json", validation_rows)

    selected_alpha, selected_validation_mrr = select_alpha_by_validation_mrr(validation_rows, ALPHAS)
    selected = {
        "selected_alpha_markov": selected_alpha,
        "selected_alpha_embedding": 1.0 - selected_alpha,
        "validation_MRR": selected_validation_mrr,
        "selection_rule": "maximum validation MRR only; first sweep alpha wins ties",
    }
    write_json(out / "selected_alpha.json", selected)
    print("selected alpha:", selected, flush=True)

    print("test encoding and formal fusion...", flush=True)
    test_embeddings = encode_texts(model, texts["test_query_texts"], query_prefix, args.batch_size)
    test_embedding_norm = row_minmax(test_embeddings @ candidate_embeddings.T)
    test_scores = selected_alpha * test_markov_norm + (1.0 - selected_alpha) * test_embedding_norm

    pure_metrics = evaluate_score_matrix_at_k(test_embedding_norm, data["test_cases"], code_to_index, metric_ks, args.retrieve_k)
    fusion_metrics = evaluate_score_matrix_at_k(test_scores, data["test_cases"], code_to_index, metric_ks, args.retrieve_k)

    summary_rows = [
        {"method": "bge_m3_pure_formal_test", "split": "test", **pure_metrics},
        {
            "method": f"bge_m3_markov_fusion_alpha_{selected_alpha}_formal_test",
            "split": "test",
            "alpha_markov": selected_alpha,
            "alpha_embedding": 1.0 - selected_alpha,
            "selected_validation_MRR": selected_validation_mrr,
            **fusion_metrics,
        },
    ]
    write_csv(out / "bge_m3_formal_summary.csv", summary_rows)
    write_json(out / "bge_m3_formal_summary.json", summary_rows)

    print("ranking per query...", flush=True)
    pure_rows = rank_rows(test_embedding_norm, data["test_cases"], code_to_index, metric_ks, args.retrieve_k)
    fusion_rows = rank_rows(test_scores, data["test_cases"], code_to_index, metric_ks, args.retrieve_k)

    per_query = []
    for p, f in zip(pure_rows, fusion_rows):
        row = {
            "row_id": p["row_id"],
            "person_id": p["person_id"],
            "prefix_length": p["prefix_length"],
            "target": p["target"],
            "pure_gold_rank": p["gold_rank"],
            "fusion_gold_rank": f["gold_rank"],
            "pure_hit_at_1": p["hit_at_1"],
            "fusion_hit_at_1": f["hit_at_1"],
            "pure_hit_at_10": p["hit_at_10"],
            "fusion_hit_at_10": f["hit_at_10"],
            "pure_mrr_at_10": p["mrr_at_10"],
            "fusion_mrr_at_10": f["mrr_at_10"],
            "pure_ndcg_at_10": p["ndcg_at_10"],
            "fusion_ndcg_at_10": f["ndcg_at_10"],
        }
        per_query.append(row)

    write_csv(out / "bge_m3_formal_per_query_ranks.csv", per_query)
    with (out / "bge_m3_formal_per_query_ranks.jsonl").open("w", encoding="utf-8") as fh:
        for row in per_query:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    pure_vec = metric_vectors(pure_rows)
    fusion_vec = metric_vectors(fusion_rows)
    sig_rows = []
    delta_rows = []
    for metric in ["Accuracy@1", "MRR@10", "NDCG@10", "Recall@10"]:
        delta = fusion_vec[metric] - pure_vec[metric]
        obs, lo, hi, p_boot = paired_bootstrap(delta, n_resamples=10000, seed=42)
        sig_rows.append({
            "test": "bge_m3_formal_fusion_vs_pure_test",
            "metric": metric,
            "n": len(delta),
            "baseline": float(pure_vec[metric].mean()),
            "system": float(fusion_vec[metric].mean()),
            "delta": obs,
            "ci95_low": lo,
            "ci95_high": hi,
            "p_boot": p_boot,
        })
        for i, d in enumerate(delta):
            delta_rows.append({"row_id": i, "metric": metric, "delta": float(d)})

    write_csv(out / "bge_m3_formal_significance_summary.csv", sig_rows)
    write_csv(out / "bge_m3_formal_per_query_deltas.csv", delta_rows)

    md = [
        "# BGE-M3 Formal Significance Test",
        "",
        "Paired bootstrap with 10000 resamples, seed=42. Delta = formal BGE-M3+Markov fusion - formal BGE-M3 pure.",
        "",
        "| Metric | N | Pure | Fusion | Delta | 95% CI | p_boot |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in sig_rows:
        md.append(
            f"| {r['metric']} | {r['n']} | {r['baseline']:.6f} | {r['system']:.6f} | "
            f"{r['delta']:.6f} | [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] | {r['p_boot']:.6g} |"
        )
    (out / "bge_m3_formal_significance_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    write_json(out / "runtime.json", {"status": "success", "finished_at_utc": now_utc()})
    cleanup_cuda()

    print("DONE")
    print(json.dumps(summary_rows, indent=2))
    print((out / "bge_m3_formal_significance_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
