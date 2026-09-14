#!/usr/bin/env python3
"""Build the shared retrieval dataset that every recipe in ../ trains on.

Pipeline:
  1. Load a (query, positive) pair dataset from the Hub. The default is
     `sentence-transformers/natural-questions` (100k question -> Wikipedia
     passage pairs, one `train` split).
  2. Hold out `--eval-queries` rows for evaluation. They never appear in
     training, and their positives are never mined as negatives.
  3. Mine hard negatives over the remaining rows with
     `sentence_transformers.util.mine_hard_negatives`: an embedding model
     retrieves near-miss passages for each query, and the false-negative
     filters (`--range-min`, `--max-score`, `--relative-margin`) drop
     candidates that are probably relevant after all.
  4. Write three JSONL files that every trainer and the shared evaluator read.

Output schema (all files land in --output-dir, default ../data):

  train.jsonl        {"query": str, "positive": str, "negatives": [str, ...]}
                     plus "positive_score": float and "negative_scores": [float]
                     when --with-scores is passed.
  eval_corpus.jsonl  {"docid": "d0", "text": str}
  eval_queries.jsonl {"qid": "q0", "query": str, "positive_docids": ["d0"]}

Usage:
  python prepare_data.py                                   # full default build
  python prepare_data.py --train-size 200 --eval-queries 50 --eval-corpus-size 500
  python prepare_data.py --with-scores                     # adds teacher scores
  python prepare_data.py --use-faiss                       # faster mining, needs faiss-cpu
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Column names we recognise without being told, in preference order.
ANCHOR_CANDIDATES = ("query", "question", "anchor", "sentence1")
POSITIVE_CANDIDATES = ("answer", "positive", "document", "passage", "text", "sentence2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mine hard negatives and emit the shared handoff dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Source dataset ---
    p.add_argument("--dataset", type=str, default="sentence-transformers/natural-questions",
                   help="HF dataset of (query, positive) pairs.")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--anchor-column", type=str, default=None,
                   help="Query column. Auto-detected when omitted.")
    p.add_argument("--positive-column", type=str, default=None,
                   help="Positive-passage column. Auto-detected when omitted.")

    # --- Sizes ---
    p.add_argument("--train-size", type=int, default=50000,
                   help="Training pairs to mine negatives for. -1 uses everything left.")
    p.add_argument("--eval-queries", type=int, default=2000,
                   help="Held-out queries for evaluation.")
    p.add_argument("--eval-corpus-size", type=int, default=20000,
                   help="Eval corpus size. Every eval positive is always kept; the "
                        "remainder is filled with distractor passages.")

    # --- Mining ---
    p.add_argument("--mining-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Embedding model used to retrieve negative candidates.")
    p.add_argument("--num-negatives", type=int, default=8,
                   help="Negatives per query. Rows where fewer are found are dropped.")
    p.add_argument("--range-min", type=int, default=10,
                   help="Skip the top-N most similar candidates (they are often true positives).")
    p.add_argument("--range-max", type=int, default=50,
                   help="Consider candidates only down to this rank.")
    p.add_argument("--max-score", type=float, default=0.8,
                   help="Drop candidates scoring above this (too similar to be a negative).")
    p.add_argument("--min-score", type=float, default=None,
                   help="Drop candidates scoring below this (too easy).")
    p.add_argument("--absolute-margin", type=float, default=None,
                   help="Require positive_score - negative_score >= margin.")
    p.add_argument("--relative-margin", type=float, default=0.05,
                   help="Require negative_score <= (1 - margin) * positive_score.")
    p.add_argument("--sampling-strategy", type=str, default="top", choices=["top", "random"],
                   help="'top' keeps the hardest candidates, 'random' samples from the range.")
    p.add_argument("--mining-batch-size", type=int, default=128)
    p.add_argument("--use-faiss", action="store_true",
                   help="Use a faiss index instead of a dense matmul. Needs the 'faiss' extra.")
    p.add_argument("--with-scores", action="store_true",
                   help="Also emit mining similarity scores. Needed for knowledge-distillation "
                        "losses (see ../03_bge_m3_flagembedding).")

    # --- Output ---
    p.add_argument("--output-dir", type=str, default="../data")
    p.add_argument("--overwrite", action="store_true",
                   help="Rebuild even if the output files already exist.")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def pick_column(columns: list[str], explicit: str | None, candidates: tuple[str, ...],
                fallback_index: int, role: str) -> str:
    if explicit is not None:
        if explicit not in columns:
            raise SystemExit(f"--{role}-column '{explicit}' not in dataset columns {columns}")
        return explicit
    for name in candidates:
        if name in columns:
            return name
    if len(columns) > fallback_index:
        return columns[fallback_index]
    raise SystemExit(f"Could not determine the {role} column from {columns}; pass --{role}-column.")


def write_jsonl(path: Path, rows) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    corpus_path = out_dir / "eval_corpus.jsonl"
    queries_path = out_dir / "eval_queries.jsonl"

    existing = [p for p in (train_path, corpus_path, queries_path) if p.exists()]
    if existing and not args.overwrite:
        print(f"[prepare] Already built: {', '.join(p.name for p in existing)}")
        print("[prepare] Pass --overwrite to rebuild.")
        return

    # Imported here so `--help` works without a torch import.
    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import mine_hard_negatives

    print(f"[prepare] Loading {args.dataset} (split={args.split})")
    dataset = load_dataset(args.dataset, split=args.split)
    anchor_col = pick_column(dataset.column_names, args.anchor_column, ANCHOR_CANDIDATES, 0, "anchor")
    positive_col = pick_column(dataset.column_names, args.positive_column, POSITIVE_CANDIDATES, 1, "positive")
    print(f"[prepare] {len(dataset):,} rows; anchor='{anchor_col}' positive='{positive_col}'")

    dataset = dataset.shuffle(seed=args.seed)

    n_eval = min(args.eval_queries, len(dataset))
    eval_ds = dataset.select(range(n_eval))
    remainder = dataset.select(range(n_eval, len(dataset)))
    n_train = len(remainder) if args.train_size < 0 else min(args.train_size, len(remainder))
    train_ds = remainder.select(range(n_train))
    print(f"[prepare] Split: {len(train_ds):,} train pairs / {len(eval_ds):,} eval queries")

    # ── Hard-negative mining ──────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[prepare] Mining {args.num_negatives} negatives per query with "
          f"{args.mining_model} on {device}")
    miner = SentenceTransformer(args.mining_model, device=device)
    mined = mine_hard_negatives(
        train_ds,
        model=miner,
        anchor_column_name=anchor_col,
        positive_column_name=positive_col,
        num_negatives=args.num_negatives,
        range_min=args.range_min,
        range_max=args.range_max,
        max_score=args.max_score,
        min_score=args.min_score,
        absolute_margin=args.absolute_margin,
        relative_margin=args.relative_margin,
        sampling_strategy=args.sampling_strategy,
        batch_size=args.mining_batch_size,
        use_faiss=args.use_faiss,
        output_format="n-tuple",
        output_scores=args.with_scores,
        verbose=True,
    )
    del miner
    if device == "cuda":
        torch.cuda.empty_cache()

    neg_cols = [c for c in (f"negative_{i}" for i in range(1, args.num_negatives + 1))
                if c in mined.column_names]
    if not neg_cols:
        raise SystemExit("Mining produced no negative columns; loosen --max-score / --relative-margin.")

    queries = mined[anchor_col]
    positives = mined[positive_col]
    negatives = list(zip(*(mined[c] for c in neg_cols)))
    scores = mined["scores"] if args.with_scores else None

    def train_rows():
        for i, (q, pos, negs) in enumerate(zip(queries, positives, negatives)):
            row = {"query": q, "positive": pos, "negatives": list(negs)}
            if scores is not None:
                row["positive_score"] = scores[i][0]
                row["negative_scores"] = scores[i][1:]
            yield row

    n_written = write_jsonl(train_path, train_rows())
    print(f"[prepare] Wrote {n_written:,} rows -> {train_path}")
    if n_written < len(train_ds):
        dropped = len(train_ds) - n_written
        print(f"[prepare] Dropped {dropped:,} rows that had fewer than "
              f"{args.num_negatives} valid negatives.")

    # ── Evaluation corpus + queries ───────────────────────────────────────────
    corpus_texts: list[str] = []
    text_to_docid: dict[str, str] = {}

    def add_document(text: str) -> str:
        docid = text_to_docid.get(text)
        if docid is None:
            docid = f"d{len(corpus_texts)}"
            text_to_docid[text] = docid
            corpus_texts.append(text)
        return docid

    query_records = []
    for i, (q, pos) in enumerate(zip(eval_ds[anchor_col], eval_ds[positive_col])):
        query_records.append({"qid": f"q{i}", "query": q, "positive_docids": [add_document(pos)]})

    n_positives = len(corpus_texts)
    if n_positives > args.eval_corpus_size:
        print(f"[prepare] Note: {n_positives:,} distinct eval positives exceed "
              f"--eval-corpus-size {args.eval_corpus_size:,}; keeping all of them so "
              f"every query stays answerable.")

    distractors = list(train_ds[positive_col])
    rng.shuffle(distractors)
    for text in distractors:
        if len(corpus_texts) >= args.eval_corpus_size:
            break
        add_document(text)

    write_jsonl(corpus_path, ({"docid": f"d{i}", "text": t} for i, t in enumerate(corpus_texts)))
    write_jsonl(queries_path, query_records)
    print(f"[prepare] Wrote {len(corpus_texts):,} documents -> {corpus_path}")
    print(f"[prepare] Wrote {len(query_records):,} queries   -> {queries_path}")

    summary = {
        "dataset": args.dataset,
        "anchor_column": anchor_col,
        "positive_column": positive_col,
        "train_rows": n_written,
        "negatives_per_row": len(neg_cols),
        "with_scores": args.with_scores,
        "eval_queries": len(query_records),
        "eval_corpus": len(corpus_texts),
        "eval_positives_in_corpus": n_positives,
        "mining_model": args.mining_model,
        "mining": {
            "range_min": args.range_min,
            "range_max": args.range_max,
            "max_score": args.max_score,
            "min_score": args.min_score,
            "absolute_margin": args.absolute_margin,
            "relative_margin": args.relative_margin,
            "sampling_strategy": args.sampling_strategy,
        },
        "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[prepare] Summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
