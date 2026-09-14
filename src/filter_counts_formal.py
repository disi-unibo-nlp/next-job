import json
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jobhop_eval_common import (  # noqa: E402
    SPLITS,
    COLLAPSE_CONSECUTIVE_DUPLICATES,
    compact_consecutive,
    date_key,
    list_parquet_files,
    load_split,
    normalize_jobhop_code,
    split_from_path,
    standardize_jobhop_schema,
)


OUT = Path("outputs/paper_checks/filter_counts")


def json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass
    return str(obj)


def ordered_valid_sequences(df):
    seqs_before = OrderedDict()
    seqs_after = OrderedDict()
    valid_row_count = 0

    for person_id, group in df.groupby("person_id", sort=False):
        group = group.copy()
        group["_date_key"] = group["start_date"].apply(date_key)
        group = group.sort_values(["_date_key", "end_date", "matched_code"], kind="mergesort")

        seq = []
        for _, row in group.iterrows():
            code = normalize_jobhop_code(row["matched_code"])
            if code is None:
                continue
            valid_row_count += 1
            seq.append({"code": code, "education": row.get("university_studies")})

        if seq:
            seqs_before[person_id] = seq
            seqs_after[person_id] = compact_consecutive(seq) if COLLAPSE_CONSECUTIVE_DUPLICATES else seq

    return seqs_before, seqs_after, valid_row_count


def last_step_cases_from_sequences(seqs):
    cases = []
    for person_id, seq in seqs.items():
        if len(seq) >= 2:
            cases.append({
                "person_id": person_id,
                "prefix": [x["code"] for x in seq[:-1]],
                "target": seq[-1]["code"],
                "prefix_length": len(seq) - 1,
            })
    return cases


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    parquet_files = list_parquet_files()
    split_files = {split: [] for split in SPLITS}
    for path in parquet_files:
        split = split_from_path(path)
        if split in split_files:
            split_files[split].append(path)

    raw = {
        split: standardize_jobhop_schema(load_split(split, split_files))
        for split in SPLITS
    }

    seqs_before = {}
    seqs_after = {}
    valid_row_counts = {}
    for split in SPLITS:
        before, after, valid_rows = ordered_valid_sequences(raw[split])
        seqs_before[split] = before
        seqs_after[split] = after
        valid_row_counts[split] = valid_rows

    train_eligible = {
        pid: seq for pid, seq in seqs_after["train"].items()
        if len(seq) >= 2
    }
    candidate_codes = sorted({item["code"] for seq in train_eligible.values() for item in seq})
    candidate_set = set(candidate_codes)

    cases_after_len = {
        split: last_step_cases_from_sequences(seqs_after[split])
        for split in ["validation", "test"]
    }
    cases_target_in_train = {
        split: [case for case in cases_after_len[split] if case["target"] in candidate_set]
        for split in ["validation", "test"]
    }

    by_split_rows = []
    stage_rows = []

    for split in SPLITS:
        df = raw[split]
        raw_rows = len(df)
        raw_persons = df["person_id"].nunique()
        unknown_rows = int((df["matched_code"].astype(str).str.lower() == "unknown").sum())
        valid_rows = valid_row_counts[split]
        persons_with_valid = len(seqs_before[split])
        persons_len_ge_2_before = sum(1 for seq in seqs_before[split].values() if len(seq) >= 2)
        events_after_collapse = sum(len(seq) for seq in seqs_after[split].values())
        persons_after_collapse = len(seqs_after[split])
        persons_len_ge_2_after = sum(1 for seq in seqs_after[split].values() if len(seq) >= 2)

        row = {
            "split": split,
            "raw_rows": raw_rows,
            "raw_persons": raw_persons,
            "unknown_code_rows": unknown_rows,
            "valid_code_rows_after_unknown_invalid_removed": valid_rows,
            "dropped_rows_unknown_or_invalid": raw_rows - valid_rows,
            "persons_with_at_least_one_valid_code": persons_with_valid,
            "persons_len_ge_2_before_collapse": persons_len_ge_2_before,
            "events_after_consecutive_duplicate_collapse": events_after_collapse,
            "persons_after_consecutive_duplicate_collapse": persons_after_collapse,
            "persons_len_ge_2_after_collapse": persons_len_ge_2_after,
            "dropped_persons_by_len_after_collapse": persons_after_collapse - persons_len_ge_2_after,
        }

        if split == "train":
            row["train_trajectories_used_for_markov"] = len(train_eligible)
            row["train_candidate_vocab_size"] = len(candidate_codes)
            row["final_train_trajectories"] = len(train_eligible)
        else:
            row["eval_cases_after_len_filter"] = len(cases_after_len[split])
            row["eval_cases_target_in_train_vocab"] = len(cases_target_in_train[split])
            row["dropped_eval_cases_target_not_in_train_vocab"] = (
                len(cases_after_len[split]) - len(cases_target_in_train[split])
            )
            row["final_eval_cases"] = len(cases_target_in_train[split])

        by_split_rows.append(row)

        stages = [
            ("raw_rows", raw_rows, "Rows loaded after schema standardization."),
            ("valid_code_rows", valid_rows, "Rows retained after removing Unknown/invalid ESCO codes."),
            ("persons_with_valid_code", persons_with_valid, "Persons with at least one valid occupation code."),
            ("persons_len_ge_2_before_collapse", persons_len_ge_2_before, "Persons with at least two valid events before duplicate collapse."),
            ("persons_after_collapse", persons_after_collapse, "Persons remaining after consecutive duplicate collapse."),
            ("persons_len_ge_2_after_collapse", persons_len_ge_2_after, "Persons with at least two events after duplicate collapse."),
        ]
        if split == "train":
            stages.extend([
                ("train_trajectories_used_for_markov", len(train_eligible), "Final train trajectories used for candidate vocabulary and Markov transitions."),
                ("train_candidate_vocab_size", len(candidate_codes), "Unique ESCO candidate codes from retained train trajectories."),
            ])
        else:
            stages.extend([
                ("eval_cases_after_len_filter", len(cases_after_len[split]), "Last-step eval cases after len>=2 trajectory filter."),
                ("eval_cases_target_in_train_vocab", len(cases_target_in_train[split]), "Eval cases retained because target is in train candidate vocabulary."),
                ("final_eval_cases", len(cases_target_in_train[split]), "Final evaluation cases."),
            ])

        for stage, count, note in stages:
            stage_rows.append({
                "split": split,
                "stage": stage,
                "count": count,
                "note": note,
            })

    by_split = pd.DataFrame(by_split_rows)
    stages = pd.DataFrame(stage_rows)

    by_split.to_csv(OUT / "filter_counts_by_split.csv", index=False)
    stages.to_csv(OUT / "filter_counts_stages.csv", index=False)

    summary = {
        "preprocessing": {
            "collapse_consecutive_duplicates": bool(COLLAPSE_CONSECUTIVE_DUPLICATES),
            "target_in_train_vocab_filter_for_eval": True,
            "candidate_vocab_source": "retained train trajectories after Unknown/invalid removal and consecutive duplicate collapse",
        },
        "candidate_vocab_size": len(candidate_codes),
        "by_split": by_split_rows,
        "expected_final_counts": {
            "train": int(by_split.loc[by_split["split"] == "train", "final_train_trajectories"].iloc[0]),
            "validation": int(by_split.loc[by_split["split"] == "validation", "final_eval_cases"].iloc[0]),
            "test": int(by_split.loc[by_split["split"] == "test", "final_eval_cases"].iloc[0]),
        },
        "split_files": {k: [str(p) for p in v] for k, v in split_files.items()},
    }

    (OUT / "filter_counts_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    md = []
    md.append("# Formal Preprocessing Filter Counts")
    md.append("")
    md.append("Counts are computed with the same preprocessing utilities used by the formal evaluation scripts.")
    md.append("")
    md.append("## Key Settings")
    md.append("")
    md.append(f"- collapse_consecutive_duplicates: `{COLLAPSE_CONSECUTIVE_DUPLICATES}`")
    md.append("- validation/test targets retained only if the target ESCO code appears in the retained train candidate vocabulary")
    md.append(f"- train candidate vocabulary size: `{len(candidate_codes)}`")
    md.append("")
    md.append("## By Split")
    md.append("")
    show_cols = [
        "split",
        "raw_rows",
        "raw_persons",
        "valid_code_rows_after_unknown_invalid_removed",
        "dropped_rows_unknown_or_invalid",
        "persons_len_ge_2_after_collapse",
    ]
    extra_cols = [
        c for c in [
            "final_train_trajectories",
            "eval_cases_after_len_filter",
            "eval_cases_target_in_train_vocab",
            "dropped_eval_cases_target_not_in_train_vocab",
            "final_eval_cases",
        ]
        if c in by_split.columns
    ]
    md.append(by_split[show_cols + extra_cols].to_markdown(index=False))
    md.append("")
    md.append("## Final Counts")
    md.append("")
    md.append(f"- train retained trajectories: `{summary['expected_final_counts']['train']}`")
    md.append(f"- validation eval cases: `{summary['expected_final_counts']['validation']}`")
    md.append(f"- test eval cases: `{summary['expected_final_counts']['test']}`")
    md.append("")
    md.append("## Files Written")
    md.append("")
    md.append("- `filter_counts_by_split.csv`")
    md.append("- `filter_counts_stages.csv`")
    md.append("- `filter_counts_summary.json`")
    md.append("- `filter_counts_summary.md`")

    (OUT / "filter_counts_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("WROTE:")
    print(OUT / "filter_counts_by_split.csv")
    print(OUT / "filter_counts_stages.csv")
    print(OUT / "filter_counts_summary.json")
    print(OUT / "filter_counts_summary.md")
    print()
    print((OUT / "filter_counts_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
