import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def clean_rank(v):
    if pd.isna(v) or v == "" or str(v).lower() == "none":
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def metric_from_rank(rank):
    r = clean_rank(rank)
    return {
        "Accuracy@1": 1.0 if r == 1 else 0.0,
        "MRR@10": (1.0 / r) if r is not None and r <= 10 else 0.0,
        "NDCG@10": (1.0 / math.log2(r + 1)) if r is not None and r <= 10 else 0.0,
        "Recall@10": 1.0 if r is not None and r <= 10 else 0.0,
    }


def method_label(x):
    s = str(x)
    return {
        "kg_alone": "KG-alone",
        "wiki_kg_fusion": "Wiki+KG fusion",
        "wiki_kg": "Wiki+KG fusion",
    }.get(s, s)


def add_rank_rows(base, ranks, row_id_col, rank_col, method):
    tmp = base.merge(
        ranks[[row_id_col, rank_col]],
        left_on="case_index",
        right_on=row_id_col,
        how="left",
    )
    rows = []
    for _, r in tmp.iterrows():
        m = metric_from_rank(r[rank_col])
        rows.append({
            "qid": r["qid"],
            "case_index": int(r["case_index"]),
            "method": method,
            "target": r["target"],
            "rank": clean_rank(r[rank_col]),
            **m,
            "history_len": r["history_len"],
            "history_bucket": r["history_bucket"],
            "return_move": r["return_move"],
            "target_train_freq": r["target_train_freq"],
            "target_freq_bucket": r["target_freq_bucket"],
            "transition_seen": r["transition_seen"],
            "last_code": r["last_code"],
        })
    return pd.DataFrame(rows)


def summarize(group):
    return pd.Series({
        "n": int(len(group)),
        "Accuracy@1": float(group["Accuracy@1"].mean()),
        "MRR@10": float(group["MRR@10"].mean()),
        "NDCG@10": float(group["NDCG@10"].mean()),
        "Recall@10": float(group["Recall@10"].mean()),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--error-per-case", default="outputs/paper_checks/error_analysis/error_analysis_per_case.csv")
    ap.add_argument("--bge-ranks", default="outputs/paper_checks/bge_m3_original_formal_per_query_full_test/bge_m3_formal_per_query_ranks.csv")
    ap.add_argument("--trained-st-ranks", default="outputs/paper_checks/trained_st_formal_per_query_full_test/trained_st_formal_per_query_ranks.csv")
    ap.add_argument("--output-dir", default="outputs/paper_checks/error_analysis_plus_bge")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ea = pd.read_csv(args.error_per_case)
    ea["method"] = ea["method"].map(method_label)

    # One row per test case for buckets/features.
    feature_cols = [
        "qid", "case_index", "target", "history_len", "history_bucket",
        "return_move", "target_train_freq", "target_freq_bucket",
        "transition_seen", "last_code",
    ]
    base = ea[feature_cols].drop_duplicates("case_index").sort_values("case_index")
    assert len(base) == 32089, f"Expected 32089 unique cases, got {len(base)}"

    # Existing KG/Wiki rows are already long-format per method.
    existing = ea[[
        "qid", "case_index", "method", "target", "rank",
        "Accuracy@1", "MRR@10", "NDCG@10", "Recall@10",
        "history_len", "history_bucket", "return_move",
        "target_train_freq", "target_freq_bucket", "transition_seen", "last_code",
    ]].copy()

    bge = pd.read_csv(args.bge_ranks)
    bge_rows = [
        add_rank_rows(base, bge, "row_id", "pure_gold_rank", "BGE-M3 pure"),
        add_rank_rows(base, bge, "row_id", "fusion_gold_rank", "BGE-M3+Markov fusion"),
    ]

    all_parts = [existing] + bge_rows

    st_path = Path(args.trained_st_ranks)
    has_st = st_path.exists()
    if has_st:
        st = pd.read_csv(st_path)
        all_parts.extend([
            add_rank_rows(base, st, "row_id", "pure_gold_rank", "trained ST pure"),
            add_rank_rows(base, st, "row_id", "fusion_gold_rank", "trained ST+Markov fusion"),
        ])

    long_df = pd.concat(all_parts, ignore_index=True)
    long_df.to_csv(out / "error_analysis_plus_bge_per_case_long.csv", index=False)

    rows = []

    overall = long_df.groupby("method", sort=False).apply(summarize).reset_index()
    overall.insert(0, "bucket_value", "all")
    overall.insert(0, "bucket_family", "overall")
    rows.append(overall)

    bucket_specs = [
        ("history_length", "history_bucket"),
        ("return_move", "return_move"),
        ("target_frequency", "target_freq_bucket"),
        ("seen_transition", "transition_seen"),
    ]

    for family, col in bucket_specs:
        part = long_df.groupby([col, "method"], dropna=False, sort=False).apply(summarize).reset_index()
        part = part.rename(columns={col: "bucket_value"})
        part.insert(0, "bucket_family", family)
        rows.append(part)

    by_bucket = pd.concat(rows, ignore_index=True)
    by_bucket.to_csv(out / "error_analysis_plus_bge_by_bucket.csv", index=False)

    examples = []
    tmp = long_df.pivot_table(
        index="case_index",
        columns="method",
        values="rank",
        aggfunc="first",
    ).reset_index()
    tmp = base.merge(tmp, on="case_index", how="left")
    if "BGE-M3 pure" in tmp.columns and "BGE-M3+Markov fusion" in tmp.columns:
        for kind, part in [
            ("bge_fusion_fix", tmp[tmp["BGE-M3 pure"].isna() & tmp["BGE-M3+Markov fusion"].notna()].head(50)),
            ("bge_fusion_regression", tmp[tmp["BGE-M3 pure"].notna() & tmp["BGE-M3+Markov fusion"].isna()].head(50)),
        ]:
            for _, r in part.iterrows():
                examples.append({
                    "example_type": kind,
                    "case_index": int(r["case_index"]),
                    "qid": r["qid"],
                    "target": r["target"],
                    "history_bucket": r["history_bucket"],
                    "return_move": r["return_move"],
                    "target_freq_bucket": r["target_freq_bucket"],
                    "transition_seen": r["transition_seen"],
                    "bge_pure_rank": clean_rank(r.get("BGE-M3 pure")),
                    "bge_fusion_rank": clean_rank(r.get("BGE-M3+Markov fusion")),
                })

    with (out / "error_analysis_plus_bge_examples.jsonl").open("w", encoding="utf-8") as fh:
        for row in examples:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = []
    md.append("# Error Analysis Plus BGE")
    md.append("")
    md.append("This table extends the existing KG/Wiki error analysis with formal BGE-M3 and trained ST per-query ranks.")
    md.append("")
    md.append("## Inputs")
    md.append(f"- error_per_case: `{args.error_per_case}`")
    md.append(f"- bge_ranks: `{args.bge_ranks}`")
    md.append(f"- trained_st_ranks: `{args.trained_st_ranks}`" if has_st else f"- trained_st_ranks missing: `{args.trained_st_ranks}`")
    md.append("")
    md.append("## Overall")
    md.append("")
    overall_view = by_bucket[by_bucket["bucket_family"] == "overall"][
        ["method", "n", "Accuracy@1", "MRR@10", "NDCG@10", "Recall@10"]
    ]
    md.append(overall_view.to_markdown(index=False))
    md.append("")
    md.append("## Files Written")
    md.append("- `error_analysis_plus_bge_per_case_long.csv`")
    md.append("- `error_analysis_plus_bge_by_bucket.csv`")
    md.append("- `error_analysis_plus_bge_examples.jsonl`")
    md.append("- `error_analysis_plus_bge_summary.md`")

    (out / "error_analysis_plus_bge_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("WROTE:")
    print(out / "error_analysis_plus_bge_per_case_long.csv")
    print(out / "error_analysis_plus_bge_by_bucket.csv")
    print(out / "error_analysis_plus_bge_examples.jsonl")
    print(out / "error_analysis_plus_bge_summary.md")
    print()
    print((out / "error_analysis_plus_bge_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
