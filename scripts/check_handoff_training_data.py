import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Guard against accidentally training on smoke handoff data.")
    parser.add_argument("--data-dir", default="handoff/data_full")
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--allow-smoke-env", default="ALLOW_SMOKE_DATA")
    return parser.parse_args()


def count_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    summary_path = data_dir / "export_summary.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training file: {train_path}")

    rows = count_jsonl(train_path)
    summary_rows = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_rows = summary.get("train_rows")

    allow_smoke = os.environ.get(args.allow_smoke_env) == "1"
    if rows < args.min_train_rows and not allow_smoke:
        raise RuntimeError(
            f"{train_path} has only {rows} rows"
            f"{'' if summary_rows is None else f' (export_summary train_rows={summary_rows})'}. "
            "This looks like smoke data. Export full data to handoff/data_full first, "
            f"or set {args.allow_smoke_env}=1 deliberately for a smoke run."
        )

    print(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "train_rows": rows,
                "summary_train_rows": summary_rows,
                "min_train_rows": args.min_train_rows,
                "allow_smoke": allow_smoke,
                "status": "ok",
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
