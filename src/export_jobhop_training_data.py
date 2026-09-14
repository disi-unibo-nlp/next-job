import argparse
import json
import random
import time
from pathlib import Path

from jobhop_eval_common import (
    candidate_text,
    configure_hf_cache,
    configure_reproducibility,
    failure_payload,
    load_jobhop_eval_data,
    make_last_step_cases,
    now_utc,
    parse_optional_int,
    query_text,
    write_csv,
    write_json,
    write_json_and_key_value_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export JobHop v2 data into the canonical retriever-training JSONL format."
    )
    parser.add_argument("--output-dir", default="handoff/data", help="Directory for train/eval JSONL files.")
    parser.add_argument("--cache-dir", default=None, help="Shared Hugging Face cache directory.")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--max-train-samples", default="none", help="Integer cap for train rows, or 'none'.")
    parser.add_argument("--max-eval-cases", default="none", help="Integer cap per validation/test split, or 'none'.")
    parser.add_argument("--num-negatives", type=int, default=8)
    parser.add_argument(
        "--train-example-mode",
        choices=["last-step", "all-prefixes"],
        default="last-step",
        help=(
            "last-step matches the current evaluation protocol most conservatively. "
            "all-prefixes uses every train transition as an explicit training expansion."
        ),
    )
    parser.add_argument(
        "--negative-strategy",
        choices=["random"],
        default="random",
        help="Negative sampling strategy for the first handoff integration.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exported files.")
    return parser.parse_args()


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def make_all_prefix_cases(trajectories_dict):
    cases = []
    for person_id, seq in trajectories_dict.items():
        codes = [item["code"] for item in seq]
        for step in range(1, len(codes)):
            cases.append(
                {
                    "person_id": person_id,
                    "prefix": codes[:step],
                    "target": codes[step],
                    "education": seq[0].get("education"),
                    "prefix_length": step,
                    "train_step_index": step,
                }
            )
    return cases


def sample_negatives(candidate_codes, target, num_negatives, rng):
    pool = [code for code in candidate_codes if code != target]
    if len(pool) <= num_negatives:
        return pool
    return rng.sample(pool, num_negatives)


def build_train_rows(cases, metadata_lookup, candidate_codes, num_negatives, rng):
    rows = []
    skipped = []
    candidate_set = set(candidate_codes)
    for idx, case in enumerate(cases):
        target = case["target"]
        if target not in candidate_set:
            skipped.append({"index": idx, "person_id": str(case["person_id"]), "target": target, "reason": "target_not_in_train_candidate_pool"})
            continue
        negative_codes = sample_negatives(candidate_codes, target, num_negatives, rng)
        if len(negative_codes) < num_negatives:
            skipped.append({"index": idx, "person_id": str(case["person_id"]), "target": target, "reason": "not_enough_negatives"})
            continue
        rows.append(
            {
                "qid": f"train_{len(rows)}",
                "person_id": str(case["person_id"]),
                "target_docid": target,
                "negative_docids": negative_codes,
                "query": query_text(metadata_lookup, case),
                "positive": candidate_text(metadata_lookup, target),
                "negatives": [candidate_text(metadata_lookup, code) for code in negative_codes],
            }
        )
    return rows, skipped


def build_eval_query_rows(split, cases, metadata_lookup):
    return [
        {
            "qid": f"{split}_{idx}",
            "person_id": str(case["person_id"]),
            "query": query_text(metadata_lookup, case),
            "positive_docids": [case["target"]],
        }
        for idx, case in enumerate(cases)
    ]


def guard_outputs(output_dir, overwrite):
    output_dir = Path(output_dir)
    targets = [
        output_dir / "train.jsonl",
        output_dir / "eval_corpus.jsonl",
        output_dir / "eval_queries.jsonl",
        output_dir / "eval_queries_validation.jsonl",
        output_dir / "eval_queries_test.jsonl",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing export files: {names}. Pass --overwrite.")


def main():
    args = parse_args()
    started_at = now_utc()
    started_monotonic = time.monotonic()
    stage = "initialization"
    output_dir = Path(args.output_dir)
    cache_dir = configure_hf_cache(args.cache_dir)
    configure_reproducibility(args.random_seed)
    rng = random.Random(args.random_seed)

    try:
        stage = "output_guard"
        guard_outputs(output_dir, args.overwrite)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_config = {
            "script": "scripts/export_jobhop_training_data.py",
            "output_dir": str(output_dir),
            "cache_dir": cache_dir,
            "random_seed": args.random_seed,
            "max_train_samples": parse_optional_int(args.max_train_samples),
            "max_eval_cases": parse_optional_int(args.max_eval_cases),
            "num_negatives": args.num_negatives,
            "negative_strategy": args.negative_strategy,
            "train_example_mode": args.train_example_mode,
        }
        write_json_and_key_value_csv(output_dir / "export_config.json", output_dir / "export_config.csv", run_config)

        stage = "load_jobhop_data"
        data = load_jobhop_eval_data(max_eval_cases=run_config["max_eval_cases"], seed=args.random_seed)
        candidate_codes = data["candidate_codes"]
        metadata_lookup = data["metadata_lookup"]

        stage = "build_training_cases"
        if args.train_example_mode == "last-step":
            train_cases = make_last_step_cases(data["trajectories"]["train"])
        else:
            train_cases = make_all_prefix_cases(data["trajectories"]["train"])

        train_cases = [case for case in train_cases if case["target"] in set(candidate_codes)]
        if run_config["max_train_samples"] is not None:
            train_cases = rng.sample(train_cases, min(run_config["max_train_samples"], len(train_cases)))

        train_rows, skipped_train_rows = build_train_rows(
            train_cases,
            metadata_lookup,
            candidate_codes,
            args.num_negatives,
            rng,
        )

        stage = "build_eval_files"
        corpus_rows = [
            {"docid": code, "text": candidate_text(metadata_lookup, code)}
            for code in candidate_codes
        ]
        validation_query_rows = build_eval_query_rows("validation", data["validation_cases"], metadata_lookup)
        test_query_rows = build_eval_query_rows("test", data["test_cases"], metadata_lookup)

        stage = "write_outputs"
        write_jsonl(output_dir / "train.jsonl", train_rows)
        write_jsonl(output_dir / "eval_corpus.jsonl", corpus_rows)
        write_jsonl(output_dir / "eval_queries_validation.jsonl", validation_query_rows)
        write_jsonl(output_dir / "eval_queries_test.jsonl", test_query_rows)
        write_jsonl(output_dir / "eval_queries.jsonl", validation_query_rows)
        write_json(output_dir / "skipped_train_rows.json", skipped_train_rows)
        write_csv(output_dir / "dataset_statistics.csv", data["stats_rows"])

        summary = {
            "status": "ok",
            "started_at_utc": started_at,
            "finished_at_utc": now_utc(),
            "runtime_seconds": time.monotonic() - started_monotonic,
            "train_rows": len(train_rows),
            "skipped_train_rows": len(skipped_train_rows),
            "candidate_corpus_rows": len(corpus_rows),
            "validation_queries": len(validation_query_rows),
            "test_queries": len(test_query_rows),
            "protocol_note": (
                "Evaluation remains last-step next-occupation prediction. "
                "Training rows are generated from the train split only using the configured train_example_mode."
            ),
            "preprocessing_config": data["preprocessing_config"],
            "stats_summary": data["stats_summary"],
            "files": {
                "train": str(output_dir / "train.jsonl"),
                "eval_corpus": str(output_dir / "eval_corpus.jsonl"),
                "eval_queries_default_validation": str(output_dir / "eval_queries.jsonl"),
                "eval_queries_validation": str(output_dir / "eval_queries_validation.jsonl"),
                "eval_queries_test": str(output_dir / "eval_queries_test.jsonl"),
            },
        }
        write_json_and_key_value_csv(output_dir / "export_summary.json", output_dir / "export_summary.csv", summary)
        print(json.dumps(summary, indent=2), flush=True)

    except Exception as exc:
        failure = failure_payload(exc, stage, started_at)
        write_json(output_dir / "export_failure.json", failure)
        raise


if __name__ == "__main__":
    main()
