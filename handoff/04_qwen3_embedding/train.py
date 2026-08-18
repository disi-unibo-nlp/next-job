#!/usr/bin/env python3
"""Fine-tune Qwen3-Embedding with ms-swift.

Qwen3-Embedding is a decoder LLM repurposed as an embedder: the text goes
through a causal transformer and the **last token's** hidden state becomes the
embedding. That is the key difference from the other three recipes, and the one
that trips people up at evaluation time — mean-pool a Qwen3 embedder and the
numbers collapse.

Training follows the upstream recipe in
https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md,
which uses ms-swift rather than Sentence Transformers. `train.py` therefore does
two things:

  1. Converts ../data/train.jsonl into swift's `infonce` message format.
  2. Builds and runs the `swift sft` command line.

Because the backbone is an LLM (0.6B / 4B / 8B), LoRA is the practical default
on a single GPU; the upstream example uses full fine-tuning across 8 GPUs with
DeepSpeed ZeRO-3.

Usage:
  python train.py --dry-run                  # convert data, print the swift command
  python train.py
  python train.py --train-type full --deepspeed zero3 --nproc-per-node 8
  python train.py --model Qwen/Qwen3-Embedding-4B --per-device-batch-size 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Qwen3-Embedding on the shared handoff dataset via ms-swift",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model ---
    p.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                   help="0.6B, 4B or 8B.")
    p.add_argument("--model-type", type=str, default="qwen3_emb",
                   help="swift model type. Keep qwen3_emb for this family.")
    p.add_argument("--train-type", type=str, default="lora", choices=["lora", "full"],
                   help="LoRA fits one GPU; full needs DeepSpeed and several. Emitted as "
                        "swift's --tuner_type (the upstream docs still say --train_type, "
                        "which ms-swift 4.4 renamed).")
    p.add_argument("--hub", type=str, default="hf", choices=["hf", "modelscope"],
                   help="Where swift downloads the base model from. ms-swift defaults to "
                        "ModelScope; the rest of this handoff uses HuggingFace.")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--target-modules", type=str, default="all-linear")
    p.add_argument("--max-length", type=int, default=512,
                   help="Tokens per text. Queries and passages share this budget.")

    # --- Data ---
    p.add_argument("--data-dir", type=str, default="../data")
    p.add_argument("--num-negatives", type=int, default=7,
                   help="Negatives written per query. Also sets MAX_NEGATIVE_SAMPLES.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--query-instruction", type=str,
                   default="Instruct: Given a web search query, retrieve relevant passages "
                           "that answer the query\nQuery: ",
                   help="Qwen3-Embedding's retrieval instruction, applied to queries only. "
                        "Pass '' to train without it.")

    # --- Loss ---
    p.add_argument("--loss-type", type=str, default="infonce",
                   choices=["infonce", "cosine_similarity"],
                   help="cosine_similarity needs a per-pair `label`, which the shared data "
                        "does not carry.")
    p.add_argument("--infonce-temperature", type=float, default=0.01)
    p.add_argument("--infonce-use-batch", action="store_true", default=True,
                   help="Use other rows in the batch as extra negatives.")
    p.add_argument("--no-infonce-use-batch", dest="infonce_use_batch", action="store_false")
    p.add_argument("--infonce-mask-fake-negative", action="store_true",
                   help="Drop in-batch negatives that look like false negatives.")

    # --- Optimisation ---
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per-device-batch-size", type=int, default=4,
                   help="Queries per device. See the effective-batch note in the README.")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=6.0e-6)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--quant-method", type=str, default=None,
                   help="Optional swift quantization backend, e.g. bnb for QLoRA.")
    p.add_argument("--quant-bits", type=int, default=None,
                   help="Optional quantization bits, e.g. 4.")
    p.add_argument("--bnb-4bit-compute-dtype", type=str, default="bfloat16",
                   help="Compute dtype when using bitsandbytes 4-bit quantization.")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--deepspeed", type=str, default=None,
                   help="swift preset: zero2 / zero3 / zero3-offload. Needed for --train-type full.")
    p.add_argument("--nproc-per-node", type=int, default=1,
                   help="GPUs. >1 sets NPROC_PER_NODE so swift launches distributed.")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    # --- Output ---
    p.add_argument("--output-dir", type=str, default="../checkpoints/qwen3_embedding")
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=None)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--report-to", type=str, default="none", choices=["none", "wandb"])
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Convert the data and print the swift command, then exit.")
    p.add_argument("--convert-only", action="store_true",
                   help="Write the swift JSONL and stop. Use it to run swift by hand.")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run ../shared/prepare_data.py first.")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_swift_jsonl(rows: list[dict], out_path: Path, num_negatives: int,
                      query_instruction: str) -> int:
    """Convert the canonical JSONL into ms-swift's infonce embedding format.

    swift models every text as a chat message list, so one training row is:

      {"messages":          [{"role": "user", "content": "<query>"}],
       "positive_messages": [[{"role": "user", "content": "<positive>"}]],
       "negative_messages": [[{"role": "user", "content": "<negative>"}], ...]}

    Note the double nesting on the positive/negative fields: each is a *list of
    message lists*, because swift supports several positives per query.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            negatives = row["negatives"][:num_negatives]
            if not negatives:
                continue
            record = {
                "messages": [{"role": "user", "content": query_instruction + row["query"]}],
                "positive_messages": [[{"role": "user", "content": row["positive"]}]],
                "negative_messages": [[{"role": "user", "content": neg}] for neg in negatives],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def build_swift_command(args, dataset_path: Path, output_dir: Path) -> list[str]:
    cmd = [
        "swift", "sft",
        "--model", args.model,
        "--model_type", args.model_type,
        "--task_type", "embedding",
        # ms-swift 4.4 renamed --train_type to --tuner_type. The upstream
        # Qwen3-Embedding doc predates that rename.
        "--tuner_type", args.train_type,
        "--use_hf", "true" if args.hub == "hf" else "false",
        "--loss_type", args.loss_type,
        "--dataset", str(dataset_path),
        "--output_dir", str(output_dir),
        "--num_train_epochs", str(args.epochs),
        "--per_device_train_batch_size", str(args.per_device_batch_size),
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--learning_rate", str(args.lr),
        "--warmup_ratio", str(args.warmup_ratio),
        "--max_length", str(args.max_length),
        "--logging_steps", str(args.logging_steps),
        "--save_steps", str(args.save_steps),
        "--seed", str(args.seed),
        "--dataloader_num_workers", "4",
        "--report_to", args.report_to,
    ]
    if args.save_total_limit is not None:
        cmd += ["--save_total_limit", str(args.save_total_limit)]
    if args.quant_method:
        cmd += ["--quant_method", args.quant_method]
    if args.quant_bits is not None:
        cmd += ["--quant_bits", str(args.quant_bits)]
    if args.quant_method == "bnb" and args.quant_bits == 4:
        cmd += ["--bnb_4bit_compute_dtype", args.bnb_4bit_compute_dtype]
    if args.gradient_checkpointing:
        cmd += ["--gradient_checkpointing", "true"]
    if args.train_type == "lora":
        cmd += [
            "--lora_rank", str(args.lora_rank),
            "--lora_alpha", str(args.lora_alpha),
            "--target_modules", args.target_modules,
        ]
    if args.deepspeed:
        cmd += ["--deepspeed", args.deepspeed]
    if args.max_steps is not None:
        cmd += ["--max_steps", str(args.max_steps)]
    if args.run_name:
        cmd += ["--run_name", args.run_name]
    return cmd


def find_last_checkpoint(output_dir: Path) -> Path | None:
    """Locate the checkpoint swift just wrote.

    swift versions each run as `<output_dir>/v<N>-<timestamp>/checkpoint-<step>`,
    so the path is not predictable up front. Pick the newest one.
    """
    checkpoints = [p for p in output_dir.glob("v*/checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def build_swift_env(args) -> dict[str, str]:
    """InfoNCE behaviour is configured through environment variables, not flags."""
    env = {
        "INFONCE_TEMPERATURE": str(args.infonce_temperature),
        "INFONCE_USE_BATCH": "true" if args.infonce_use_batch else "false",
        "INFONCE_HARD_NEGATIVES": str(args.num_negatives),
        "INFONCE_MASK_FAKE_NEGATIVE": "true" if args.infonce_mask_fake_negative else "false",
        "MAX_POSITIVE_SAMPLES": "1",
        "MAX_NEGATIVE_SAMPLES": str(args.num_negatives),
    }
    if args.nproc_per_node > 1:
        env["NPROC_PER_NODE"] = str(args.nproc_per_node)
    return env


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    rows = read_jsonl(data_dir / "train.jsonl")
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    output_dir = Path(args.output_dir).resolve()
    swift_data = output_dir / "swift_data" / "train.jsonl"
    n_written = write_swift_jsonl(rows, swift_data, args.num_negatives, args.query_instruction)
    print(f"[train] Wrote {n_written:,} rows -> {swift_data}")
    if n_written == 0:
        raise SystemExit("No rows had any negatives. Re-run ../shared/prepare_data.py.")

    if args.loss_type == "cosine_similarity":
        raise SystemExit(
            "--loss-type cosine_similarity needs a float `label` per pair, which the "
            "shared triplet data does not carry. Use infonce, or build a labelled "
            "similarity dataset instead."
        )

    env_overrides = build_swift_env(args)
    cmd = build_swift_command(args, swift_data, output_dir)

    per_device = args.per_device_batch_size * 1 * (1 + args.num_negatives)
    print(f"[train] model={args.model} train_type={args.train_type} loss={args.loss_type}")
    print(f"[train] {per_device} texts per device per step "
          f"({args.per_device_batch_size} queries x (1 positive + {args.num_negatives} negatives))")
    print(f"[train] effective batch = {args.per_device_batch_size} x {args.grad_accum} x "
          f"{args.nproc_per_node} = "
          f"{args.per_device_batch_size * args.grad_accum * args.nproc_per_node} queries")
    print("[train] environment:")
    for key, value in env_overrides.items():
        print(f"           {key}={value}")
    print("[train] command:")
    print("           " + " ".join(cmd))

    if args.convert_only:
        print("[train] --convert-only: data written, nothing launched.")
        return
    if args.dry_run:
        print("[train] --dry-run: nothing launched.")
        return

    if shutil.which("swift") is None:
        raise SystemExit(
            "`swift` is not on PATH. Install the environment first:\n"
            "  uv sync\n"
            "and run through `uv run`, or `bash run.sh`."
        )

    env = os.environ.copy()
    env.update(env_overrides)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)

    meta = {
        "base_model": args.model,
        "train_type": args.train_type,
        "loss_type": args.loss_type,
        "train_rows": n_written,
        "num_negatives": args.num_negatives,
        "query_instruction": args.query_instruction,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "quant_method": args.quant_method,
        "quant_bits": args.quant_bits,
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
        "lr": args.lr,
        "seed": args.seed,
        "swift_env": env_overrides,
        "swift_command": cmd,
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"[train] Saved -> {output_dir}")
    merged_dir = output_dir / "merged"
    if args.train_type == "lora":
        checkpoint = find_last_checkpoint(output_dir)
        print("[train] LoRA adapters must be merged before evaluation:")
        if checkpoint is None:
            print(f"  (no checkpoint found under {output_dir}/v*/checkpoint-*)")
        else:
            hub_flag = "true" if args.hub == "hf" else "false"
            print(f"  uv run swift export --adapters {checkpoint} --merge_lora true "
                  f"--use_hf {hub_flag} --output_dir {merged_dir}")
    else:
        merged_dir = output_dir
    print("[train] Then evaluate with:")
    print(f"  cd ../shared && uv run python evaluate.py --backend dense "
          f"--model-path {merged_dir} \\\n"
          f"      --pooling lasttoken --normalize \\\n"
          f"      --query-prompt {args.query_instruction!r}")


if __name__ == "__main__":
    main()
