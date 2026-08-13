#!/usr/bin/env python3
"""Fine-tune BGE-M3 with FlagEmbedding.

BGE-M3 is a multi-functional retriever: one backbone (XLM-RoBERTa-large, 568M)
producing three retrieval signals at once.

  dense    the CLS vector                       -> cosine similarity
  sparse   a learned weight per vocabulary term -> lexical matching, like BM25
  colbert  one vector per token                 -> MaxSim, like recipe 02

`--unified-finetuning` (on by default) trains all three heads together and adds
a self-distillation term that teaches each head from the ensemble of the other
two. That is what makes BGE-M3 different from the dense recipe: you are not
choosing between lexical and semantic matching, you are training both.

Training runs through FlagEmbedding's own `EncoderOnlyEmbedderM3Runner` rather
than Sentence Transformers, because the sparse and ColBERT heads and the
self-distillation loop live inside it.

Reads ../data/train.jsonl (see ../shared/prepare_data.py) and converts it to the
JSONL layout FlagEmbedding expects.

Usage:
  python train.py --dry-run                    # convert data, print plan, don't train
  python train.py
  python train.py --neg-num 7 --batch-size 2 --gradient-checkpointing
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BGE-M3 on the shared handoff dataset via FlagEmbedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model ---
    p.add_argument("--model", type=str, default="BAAI/bge-m3")

    # --- Data ---
    p.add_argument("--data-dir", type=str, default="../data")
    p.add_argument("--neg-num", type=int, default=7,
                   help="Negatives written per query. --train-group-size follows this "
                        "automatically unless you override it.")
    p.add_argument("--train-group-size", type=int, default=None,
                   help="Passages per query inside the loss. Must equal --neg-num + 1; "
                        "defaults to exactly that.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--query-max-len", type=int, default=128,
                   help="NQ questions are short; passages get --passage-max-len.")
    p.add_argument("--passage-max-len", type=int, default=512)
    p.add_argument("--query-instruction", type=str,
                   default="Represent this sentence for searching relevant passages: ",
                   help="Prefix applied to queries. Pass '' to disable.")

    # --- Loss / heads ---
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--sentence-pooling-method", type=str, default="cls",
                   choices=["cls", "mean", "last_token"])
    p.add_argument("--unified-finetuning", action="store_true", default=True,
                   help="Train dense + sparse + ColBERT heads together.")
    p.add_argument("--dense-only", dest="unified_finetuning", action="store_false",
                   help="Train the dense head only.")
    p.add_argument("--use-self-distill", action="store_true", default=True)
    p.add_argument("--no-self-distill", dest="use_self_distill", action="store_false")
    p.add_argument("--self-distill-start-step", type=int, default=0)
    p.add_argument("--knowledge-distillation", action="store_true",
                   help="Learn from teacher scores. Requires ../data/train.jsonl built "
                        "with `prepare_data.py --with-scores`.")
    p.add_argument("--kd-loss-type", type=str, default="m3_kd_loss",
                   choices=["m3_kd_loss", "kl_div"])
    p.add_argument("--negatives-cross-device", action="store_true", default=True,
                   help="Share negatives across GPUs. Harmless on one GPU.")
    p.add_argument("--no-negatives-cross-device", dest="negatives_cross_device",
                   action="store_false")

    # --- Optimisation ---
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Per device. The real cost is batch-size x train-group-size passages.")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1.0e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--deepspeed", type=str, default="ds_stage0.json",
                   help="DeepSpeed config path. Pass '' to run without DeepSpeed.")
    p.add_argument("--seed", type=int, default=42)

    # --- Output ---
    p.add_argument("--output-dir", type=str, default="../checkpoints/bge_m3")
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--report-to", type=str, default="none", choices=["none", "wandb"])
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Convert the data and print the plan, then exit.")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run ../shared/prepare_data.py first.")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_flagembedding_jsonl(rows: list[dict], out_path: Path, neg_num: int,
                              knowledge_distillation: bool) -> int:
    """Convert the canonical JSONL into FlagEmbedding's training format.

    One object per query:
      {"query": str, "pos": [str], "neg": [str, ...], "prompt": "", "type": "normal"}

    With knowledge distillation two more fields carry the teacher scores:
      "pos_scores": [float], "neg_scores": [float]
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            negatives = row["negatives"][:neg_num]
            if len(negatives) < neg_num:
                continue
            record = {
                "query": row["query"],
                "pos": [row["positive"]],
                "neg": negatives,
                "prompt": "",
                "type": "normal",
            }
            if knowledge_distillation:
                if "positive_score" not in row or "negative_scores" not in row:
                    raise SystemExit(
                        "--knowledge-distillation needs teacher scores, but "
                        "train.jsonl has none. Rebuild the data with:\n"
                        "  cd ../shared && uv run python prepare_data.py --with-scores --overwrite"
                    )
                record["pos_scores"] = [float(row["positive_score"])]
                record["neg_scores"] = [float(s) for s in row["negative_scores"][:neg_num]]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    args = parse_args()

    train_group_size = args.train_group_size
    if train_group_size is None:
        train_group_size = args.neg_num + 1
    elif train_group_size != args.neg_num + 1:
        print(f"[train] WARNING: --train-group-size {train_group_size} != --neg-num "
              f"{args.neg_num} + 1. FlagEmbedding will pad or truncate the group; "
              f"keep them in sync unless you know why you are not.")

    data_dir = Path(args.data_dir).resolve()
    rows = read_jsonl(data_dir / "train.jsonl")
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    output_dir = Path(args.output_dir).resolve()
    ft_data = output_dir / "ft_data" / "training.jsonl"
    n_written = write_flagembedding_jsonl(rows, ft_data, args.neg_num, args.knowledge_distillation)
    print(f"[train] Wrote {n_written:,} rows -> {ft_data}")
    if n_written < len(rows):
        print(f"[train] Skipped {len(rows) - n_written:,} rows with fewer than "
              f"{args.neg_num} negatives.")
    if n_written == 0:
        raise SystemExit(f"No rows had {args.neg_num} negatives. Lower --neg-num, or "
                         f"re-mine with a larger --num-negatives.")

    passages_per_step = args.batch_size * train_group_size
    print(f"[train] model={args.model} heads="
          f"{'dense+sparse+colbert' if args.unified_finetuning else 'dense'}")
    print(f"[train] train_group_size={train_group_size} (1 positive + {args.neg_num} negatives)")
    print(f"[train] {passages_per_step} passages encoded per device per step "
          f"({args.batch_size} queries x {train_group_size})")
    print(f"[train] effective batch = {args.batch_size} x {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} queries")

    deepspeed_path = args.deepspeed or None
    if deepspeed_path and not Path(deepspeed_path).exists():
        raise SystemExit(f"DeepSpeed config not found: {deepspeed_path}. "
                         f"Ship ds_stage0.json alongside train.py, or pass --deepspeed ''.")

    if args.dry_run:
        print(f"[train] output={output_dir}")
        print("[train] --dry-run: data converted, nothing trained.")
        return

    import torch
    from FlagEmbedding.finetune.embedder.encoder_only.m3 import (
        EncoderOnlyEmbedderM3DataArguments,
        EncoderOnlyEmbedderM3ModelArguments,
        EncoderOnlyEmbedderM3Runner,
        EncoderOnlyEmbedderM3TrainingArguments,
    )

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", "handoff-bge-m3")
        if args.run_name:
            os.environ["WANDB_RUN_NAME"] = args.run_name

    # Leave cache_dir unset so the base model comes from the shared HuggingFace
    # cache. Pointing it inside output_dir makes every run re-download several GB
    # of BGE-M3 weights into a fresh directory.
    model_args = EncoderOnlyEmbedderM3ModelArguments(
        model_name_or_path=args.model,
    )
    data_args = EncoderOnlyEmbedderM3DataArguments(
        train_data=[str(ft_data)],
        cache_path=str(output_dir / "dataset_cache"),
        train_group_size=train_group_size,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len,
        pad_to_multiple_of=8,
        query_instruction_for_retrieval=args.query_instruction,
        query_instruction_format="{}{}",
        knowledge_distillation=args.knowledge_distillation,
        same_dataset_within_batch=False,
    )
    training_args = EncoderOnlyEmbedderM3TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        seed=args.seed,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        negatives_cross_device=args.negatives_cross_device,
        temperature=args.temperature,
        sentence_pooling_method=args.sentence_pooling_method,
        normalize_embeddings=True,
        kd_loss_type=args.kd_loss_type,
        unified_finetuning=args.unified_finetuning,
        use_self_distill=args.use_self_distill,
        self_distill_start_step=args.self_distill_start_step,
        deepspeed=deepspeed_path,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=args.report_to,
        run_name=args.run_name,
    )

    # FlagEmbedding's modeling code needs a process group whenever
    # negatives_cross_device is on. Initialising a single-process world lets this
    # run under a plain `python train.py`; see run.sh for the multi-GPU form.
    if args.negatives_cross_device and not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )

    runner = EncoderOnlyEmbedderM3Runner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    runner.run()

    meta = {
        "base_model": args.model,
        "train_rows": n_written,
        "neg_num": args.neg_num,
        "train_group_size": train_group_size,
        "unified_finetuning": args.unified_finetuning,
        "use_self_distill": args.use_self_distill,
        "knowledge_distillation": args.knowledge_distillation,
        "temperature": args.temperature,
        "sentence_pooling_method": args.sentence_pooling_method,
        "query_instruction": args.query_instruction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "lr": args.lr,
        "seed": args.seed,
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"[train] Saved -> {output_dir}")
    print("[train] Evaluate the dense head with:")
    print(f"  cd ../shared && uv run python evaluate.py --backend dense "
          f"--model-path {output_dir} \\\n"
          f"      --query-prompt {args.query_instruction!r}")


if __name__ == "__main__":
    main()
