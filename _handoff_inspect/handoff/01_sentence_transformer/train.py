#!/usr/bin/env python3
"""Fine-tune a dense bi-encoder with Sentence Transformers.

The classic retrieval setup: one encoder maps both queries and passages into a
single vector each, and relevance is cosine similarity. Fast to train, fast to
serve, and the baseline every other recipe here should be compared against.

Reads ../data/train.jsonl (see ../shared/prepare_data.py) and converts it to the
column layout the selected loss expects:

  anchor | positive | negative_1 | ... | negative_N

Training uses `SentenceTransformerTrainer`, with an
`InformationRetrievalEvaluator` over the held-out split running during training
so you can watch NDCG@10 move.

Usage:
  python train.py --dry-run
  python train.py                                     # defaults
  python train.py --loss cmnrl --batch-size 256 --mini-batch-size 32
  python train.py --model BAAI/bge-base-en-v1.5 --num-negatives 4 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# --- Loss registry -----------------------------------------------------------
# Maps --loss to (sentence_transformers.losses class name, how many negative
# columns it consumes). `None` means "every negative available".
LOSS_SPECS = {
    "mnrl": ("MultipleNegativesRankingLoss", None),
    "cmnrl": ("CachedMultipleNegativesRankingLoss", None),
    "mnrl-symmetric": ("MultipleNegativesSymmetricRankingLoss", 0),
    "triplet": ("TripletLoss", 1),
    "gist": ("GISTEmbedLoss", None),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a dense bi-encoder on the shared handoff dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model ---
    p.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Base model. Any SentenceTransformer or plain encoder from the Hub.")
    p.add_argument("--max-seq-length", type=int, default=384)
    p.add_argument("--pooling", type=str, default=None, choices=["mean", "cls", "max", "lasttoken"],
                   help="Force a pooling head. Omit to use the model's own config.")

    # --- Data ---
    p.add_argument("--data-dir", type=str, default="../data")
    p.add_argument("--num-negatives", type=int, default=4,
                   help="Hard negatives per example to actually train on. Capped by what "
                        "prepare_data.py mined, and by the loss (see --loss).")
    p.add_argument("--max-samples", type=int, default=None, help="Cap training rows.")
    p.add_argument("--query-prompt", type=str, default=None,
                   help="Prefix prepended to anchors, e.g. BGE's 'Represent this sentence "
                        "for searching relevant passages: '.")
    p.add_argument("--corpus-prompt", type=str, default=None,
                   help="Prefix prepended to positives and negatives.")

    # --- Loss ---
    p.add_argument("--loss", type=str, default="mnrl", choices=sorted(LOSS_SPECS),
                   help="See the README table for the columns each loss consumes.")
    p.add_argument("--scale", type=float, default=20.0,
                   help="Similarity scale (1/temperature) for the MNRL family.")
    p.add_argument("--mini-batch-size", type=int, default=32,
                   help="GradCache micro-batch. Only used by --loss cmnrl.")
    p.add_argument("--triplet-margin", type=float, default=5.0,
                   help="Only used by --loss triplet.")
    p.add_argument("--guide-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Guide encoder that filters false negatives. Only used by --loss gist.")
    p.add_argument("--gist-temperature", type=float, default=0.01,
                   help="Only used by --loss gist.")

    # --- Optimisation ---
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2.0e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--scheduler", type=str, default="cosine")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Stop after N optimizer steps. Use a small value for smoke runs.")
    p.add_argument("--seed", type=int, default=42)

    # --- Evaluation & output ---
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-queries", type=int, default=500,
                   help="Queries used by the in-training evaluator. Keep it small; the full "
                        "set is for ../shared/evaluate.py.")
    p.add_argument("--no-eval", action="store_true", help="Skip the in-training evaluator.")
    p.add_argument("--output-dir", type=str, default="../checkpoints/sentence_transformer")
    p.add_argument("--no-save", action="store_true", help="Train but write no checkpoint.")
    p.add_argument("--report-to", type=str, default="none", choices=["none", "wandb"])
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve config and data, print the plan, then exit.")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run ../shared/prepare_data.py first.")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_train_dataset(rows: list[dict], num_negatives: int, query_prompt: str | None,
                        corpus_prompt: str | None):
    """Convert the canonical JSONL into anchor / positive / negative_i columns."""
    from datasets import Dataset

    qp = query_prompt or ""
    cp = corpus_prompt or ""
    data: dict[str, list[str]] = {
        "anchor": [qp + r["query"] for r in rows],
        "positive": [cp + r["positive"] for r in rows],
    }
    for i in range(num_negatives):
        data[f"negative_{i + 1}"] = [cp + r["negatives"][i] for r in rows]
    return Dataset.from_dict(data)


def build_evaluator(args, data_dir: Path):
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    corpus_rows = read_jsonl(data_dir / "eval_corpus.jsonl")
    query_rows = read_jsonl(data_dir / "eval_queries.jsonl")[: args.eval_queries]

    queries = {r["qid"]: r["query"] for r in query_rows}
    relevant_docs = {r["qid"]: set(r["positive_docids"]) for r in query_rows}
    needed = {d for docs in relevant_docs.values() for d in docs}
    # Keep every gold document plus enough distractors to make the task non-trivial.
    corpus = {r["docid"]: r["text"] for r in corpus_rows if r["docid"] in needed}
    for r in corpus_rows:
        if len(corpus) >= max(len(needed) * 10, 2000):
            break
        corpus.setdefault(r["docid"], r["text"])

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[10],
        batch_size=args.eval_batch_size,
        name="dev",
        show_progress_bar=False,
        write_csv=False,
        query_prompt=args.query_prompt,
        corpus_prompt=args.corpus_prompt,
    )


def build_loss(args, model):
    from sentence_transformers import SentenceTransformer, losses

    kind = args.loss
    if kind == "mnrl":
        return losses.MultipleNegativesRankingLoss(model, scale=args.scale)
    if kind == "cmnrl":
        return losses.CachedMultipleNegativesRankingLoss(
            model, scale=args.scale, mini_batch_size=args.mini_batch_size
        )
    if kind == "mnrl-symmetric":
        return losses.MultipleNegativesSymmetricRankingLoss(model, scale=args.scale)
    if kind == "triplet":
        return losses.TripletLoss(model, triplet_margin=args.triplet_margin)
    if kind == "gist":
        guide = SentenceTransformer(args.guide_model)
        return losses.GISTEmbedLoss(model, guide=guide, temperature=args.gist_temperature)
    raise ValueError(f"Unhandled loss {kind}")


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    rows = read_jsonl(data_dir / "train.jsonl")
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    available = min(len(r["negatives"]) for r in rows) if rows else 0
    _, loss_cap = LOSS_SPECS[args.loss]
    requested = args.num_negatives
    num_negatives = min(requested, available)
    if loss_cap is not None:
        num_negatives = min(num_negatives, loss_cap)

    columns = ["anchor", "positive"] + [f"negative_{i + 1}" for i in range(num_negatives)]
    print(f"[train] loss={args.loss} ({LOSS_SPECS[args.loss][0]})")
    print(f"[train] {len(rows):,} rows, columns: {', '.join(columns)}")
    if num_negatives < requested:
        reason = "the loss ignores the rest" if loss_cap is not None and loss_cap < available \
            else "prepare_data.py mined fewer"
        print(f"[train] --num-negatives {requested} reduced to {num_negatives} ({reason}).")

    if args.dry_run:
        print(f"[train] model={args.model} batch={args.batch_size} x grad_accum={args.grad_accum} "
              f"epochs={args.epochs} lr={args.lr}")
        print(f"[train] output={Path(args.output_dir).resolve()}")
        print("[train] --dry-run: nothing trained.")
        return

    import torch
    from sentence_transformers import SentenceTransformer, models
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.training_args import BatchSamplers

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = build_train_dataset(rows, num_negatives, args.query_prompt, args.corpus_prompt)

    if args.pooling is None:
        model = SentenceTransformer(args.model)
        model.max_seq_length = args.max_seq_length
    else:
        transformer = models.Transformer(args.model, max_seq_length=args.max_seq_length)
        model = SentenceTransformer(modules=[
            transformer,
            models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=args.pooling),
            models.Normalize(),
        ])

    loss = build_loss(args, model)
    evaluator = None if args.no_eval else build_evaluator(args, data_dir)

    output_dir = Path(args.output_dir).resolve()
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        # Guarantees no sentence appears twice in a batch, which would make it an
        # in-batch negative for itself.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps" if evaluator is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy="no" if args.no_save else "steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        logging_steps=25,
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    if args.no_save:
        print("[train] --no-save: checkpoint discarded.")
        return

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    meta = {
        "base_model": args.model,
        "loss": args.loss,
        "loss_class": LOSS_SPECS[args.loss][0],
        "num_negatives": num_negatives,
        "train_rows": len(rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "lr": args.lr,
        "query_prompt": args.query_prompt,
        "corpus_prompt": args.corpus_prompt,
        "seed": args.seed,
    }
    (final_dir / "training_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[train] Saved -> {final_dir}")
    print(f"[train] Evaluate with:\n"
          f"  cd ../shared && uv run python evaluate.py --backend dense --model-path {final_dir}")


if __name__ == "__main__":
    main()
