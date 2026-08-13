#!/usr/bin/env python3
"""Fine-tune a late-interaction (ColBERT-style) retriever with PyLate.

Instead of collapsing a text into one vector, a late-interaction model keeps one
small vector per token. Relevance is MaxSim: for each query token, take its best
match anywhere in the document, then sum. That preserves term-level signal a
single vector throws away, at the cost of a much larger index.

    MaxSim(q, d) = sum over query tokens t of  max over document tokens s of  q_t . d_s

PyLate ships no trainer of its own — training is `SentenceTransformerTrainer`
with PyLate's `ColBERTCollator`, which tokenizes the query column and the
document columns with different lengths and prefixes.

Reads ../data/train.jsonl (see ../shared/prepare_data.py) and converts it to:

  query | document | negative_0 | ... | negative_{N-1}

Usage:
  python train.py --dry-run
  python train.py --gradient-checkpointing
  python train.py --loss cached_contrastive --batch-size 256 --mini-batch-size 16
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# PyLate's ColBERT wrapper reserves this many extra positions for the mask-token
# query expansion used by the original ColBERT recipe.
EXTRA_LENGTH = 7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a ColBERT-style retriever on the shared handoff dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model ---
    p.add_argument("--model", type=str, default="lightonai/GTE-ModernColBERT-v1",
                   help="A PyLate ColBERT checkpoint, or any encoder (a fresh randomly "
                        "initialised projection head is added when none exists).")
    p.add_argument("--query-length", type=int, default=32,
                   help="Query tokens. Queries are padded up to exactly this length.")
    p.add_argument("--document-length", type=int, default=300,
                   help="Document tokens. Drives both VRAM and index size.")
    p.add_argument("--use-extra-length", action="store_true",
                   help=f"Add {EXTRA_LENGTH} positions for mask-token query expansion "
                        "(original ColBERT recipe).")
    p.add_argument("--use-prompts", action="store_true",
                   help="Prepend 'search_query: ' / 'search_document: '. Only for models "
                        "trained with those prefixes.")
    p.add_argument("--query-prefix", type=str, default="search_query: ")
    p.add_argument("--document-prefix", type=str, default="search_document: ")

    # --- Data ---
    p.add_argument("--data-dir", type=str, default="../data")
    p.add_argument("--num-negatives", type=int, default=4,
                   help="Hard negatives per query. Multiplies the per-step document count.")
    p.add_argument("--max-samples", type=int, default=None)

    # --- Loss ---
    p.add_argument("--loss", type=str, default="contrastive",
                   choices=["contrastive", "cached_contrastive"])
    p.add_argument("--temperature", type=float, default=0.05,
                   help="InfoNCE temperature. Lower sharpens the contrast; 0.05 suits "
                        "quality-mined negatives, 0.2 suits random ones.")
    p.add_argument("--mini-batch-size", type=int, default=16,
                   help="GradCache micro-batch. Only used by --loss cached_contrastive.")
    p.add_argument("--gather-across-devices", action="store_true",
                   help="Share in-batch negatives across GPUs. Turn on for multi-GPU runs; "
                        "on a single GPU it only prints a warning.")

    # --- Optimisation ---
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Keep this at 8 or above; the recipes are tuned around the "
                        "effective batch (batch-size x grad-accum).")
    p.add_argument("--lr", type=float, default=3.0e-6)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--scheduler", type=str, default="cosine")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Strongly recommended. Multi-vector activations are large.")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    # --- Evaluation & output ---
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--eval-queries", type=int, default=200,
                   help="Queries for the in-training MaxSim evaluator. Keep it small; "
                        "MaxSim scoring is quadratic in tokens.")
    p.add_argument("--eval-corpus", type=int, default=2000)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--output-dir", type=str, default="../checkpoints/late_interaction")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--report-to", type=str, default="none", choices=["none", "wandb"])
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run ../shared/prepare_data.py first.")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_train_dataset(rows: list[dict], num_negatives: int):
    """Convert the canonical JSONL into query / document / negative_i columns."""
    from datasets import Dataset

    data: dict[str, list[str]] = {
        "query": [r["query"] for r in rows],
        "document": [r["positive"] for r in rows],
    }
    for i in range(num_negatives):
        data[f"negative_{i}"] = [r["negatives"][i] for r in rows]
    return Dataset.from_dict(data)


def max_sim(queries, documents, query_chunk: int = 8, doc_chunk: int = 256):
    """MaxSim over zero-padded (batch, tokens, dim) tensors, blocked to bound memory."""
    import torch

    queries = queries.float()
    documents = documents.float().to(queries.device)
    out = torch.zeros(queries.size(0), documents.size(0), device=queries.device)
    for qs in range(0, queries.size(0), query_chunk):
        q_block = queries[qs:qs + query_chunk]
        for ds in range(0, documents.size(0), doc_chunk):
            d_block = documents[ds:ds + doc_chunk]
            sim = torch.einsum("qtd,psd->qpts", q_block, d_block)
            out[qs:qs + q_block.size(0), ds:ds + d_block.size(0)] = sim.max(dim=3).values.sum(dim=2)
    return out


def make_maxsim_evaluator(queries: dict[str, str], corpus: dict[str, str],
                          relevant_docs: dict[str, set[str]], ks=(1, 5, 10),
                          batch_size: int = 32):
    """Build a compact in-training evaluator: encode, MaxSim, report recall and NDCG.

    `../shared/evaluate.py` is the real evaluation. This one exists so the
    training curve carries a retrieval signal, and is deliberately small so it
    does not dominate step time.

    The class is defined inside the function only so that `--help` and
    `--dry-run` do not have to import torch.
    """
    import math

    import torch
    from sentence_transformers.evaluation import SentenceEvaluator

    class MaxSimRecallEvaluator(SentenceEvaluator):
        def __init__(self):
            super().__init__()
            self.query_ids = list(queries)
            self.query_texts = [queries[q] for q in self.query_ids]
            self.corpus_ids = list(corpus)
            self.corpus_texts = [corpus[d] for d in self.corpus_ids]
            self.ks = list(ks)
            self.primary_metric = f"dev_maxsim_ndcg@{max(self.ks)}"

        def _encode(self, model, texts, is_query):
            embeddings = model.encode(
                texts, is_query=is_query, batch_size=batch_size,
                show_progress_bar=False, convert_to_numpy=False, convert_to_tensor=True,
                padding=False,
            )
            if not isinstance(embeddings, list):
                embeddings = [embeddings]
            return torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value=0.0)

        def __call__(self, model, output_path=None, epoch=-1, steps=-1, **kwargs):
            was_training = model.training
            model.eval()
            with torch.no_grad():
                query_emb = self._encode(model, self.query_texts, True)
                corpus_emb = self._encode(model, self.corpus_texts, False)
                scores = max_sim(query_emb, corpus_emb)
                top_k = min(max(self.ks), scores.size(1))
                ranked = torch.topk(scores, k=top_k, dim=1).indices.cpu().tolist()
            if was_training:
                model.train()

            metrics = {}
            for k in self.ks:
                recalls, ndcgs = [], []
                for qi, qid in enumerate(self.query_ids):
                    gold = relevant_docs.get(qid, set())
                    if not gold:
                        continue
                    retrieved = [self.corpus_ids[i] for i in ranked[qi][:k]]
                    hits = sum(1 for d in retrieved if d in gold)
                    recalls.append(hits / min(len(gold), k))
                    dcg = sum(1 / math.log2(rank + 2)
                              for rank, d in enumerate(retrieved) if d in gold)
                    idcg = sum(1 / math.log2(rank + 2) for rank in range(min(len(gold), k)))
                    ndcgs.append(dcg / idcg if idcg else 0.0)
                metrics[f"dev_maxsim_recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
                metrics[f"dev_maxsim_ndcg@{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
            return metrics

    return MaxSimRecallEvaluator()


def build_evaluator(args, data_dir: Path):
    corpus_rows = read_jsonl(data_dir / "eval_corpus.jsonl")
    query_rows = read_jsonl(data_dir / "eval_queries.jsonl")[: args.eval_queries]

    queries = {r["qid"]: r["query"] for r in query_rows}
    relevant_docs = {r["qid"]: set(r["positive_docids"]) for r in query_rows}
    needed = {d for docs in relevant_docs.values() for d in docs}

    corpus = {r["docid"]: r["text"] for r in corpus_rows if r["docid"] in needed}
    for r in corpus_rows:
        if len(corpus) >= max(len(needed), args.eval_corpus):
            break
        corpus.setdefault(r["docid"], r["text"])

    return make_maxsim_evaluator(queries, corpus, relevant_docs, batch_size=args.eval_batch_size)


def build_loss(args, pylate_losses, model):
    # Both PyLate losses accept `temperature` and `gather_across_devices`; pass them
    # explicitly so that switching --loss does not silently change the objective.
    if args.loss == "contrastive":
        return pylate_losses.Contrastive(
            model=model,
            temperature=args.temperature,
            gather_across_devices=args.gather_across_devices,
        )
    if args.loss == "cached_contrastive":
        return pylate_losses.CachedContrastive(
            model=model,
            mini_batch_size=args.mini_batch_size,
            temperature=args.temperature,
            gather_across_devices=args.gather_across_devices,
            show_progress_bar=False,
        )
    raise ValueError(f"Unhandled loss {args.loss}")


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    rows = read_jsonl(data_dir / "train.jsonl")
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    available = min(len(r["negatives"]) for r in rows) if rows else 0
    num_negatives = min(args.num_negatives, available)
    columns = ["query", "document"] + [f"negative_{i}" for i in range(num_negatives)]

    query_length = args.query_length + (EXTRA_LENGTH if args.use_extra_length else 0)
    document_length = args.document_length + (EXTRA_LENGTH if args.use_extra_length else 0)

    print(f"[train] model={args.model} loss={args.loss} temperature={args.temperature}")
    print(f"[train] {len(rows):,} rows, columns: {', '.join(columns)}")
    print(f"[train] query_length={query_length} document_length={document_length} "
          f"prompts={'on' if args.use_prompts else 'off'}")
    print(f"[train] effective batch = {args.batch_size} x {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum}; "
          f"{1 + num_negatives} documents per query per step")
    if num_negatives < args.num_negatives:
        print(f"[train] --num-negatives {args.num_negatives} reduced to {num_negatives} "
              f"(prepare_data.py mined fewer).")
    if args.grad_accum < 8:
        print(f"[train] WARNING: --grad-accum {args.grad_accum} is below the recommended "
              f"floor of 8. Raise --batch-size or --num-negatives to fit VRAM instead.")
    if not args.gradient_checkpointing:
        print("[train] WARNING: --gradient-checkpointing is off. Multi-vector activations "
              "are large; expect OOM at anything but tiny batches.")

    if args.dry_run:
        print(f"[train] output={Path(args.output_dir).resolve()}")
        print("[train] --dry-run: nothing trained.")
        return

    import torch
    from pylate import losses as pylate_losses, models as pylate_models, utils as pylate_utils
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = build_train_dataset(rows, num_negatives)

    model = pylate_models.ColBERT(
        model_name_or_path=args.model,
        query_length=query_length,
        document_length=document_length,
    )

    prompts = None
    if args.use_prompts:
        prompts = {"query": args.query_prefix}
        for column in columns[1:]:
            prompts[column] = args.document_prefix

    loss = build_loss(args, pylate_losses, model)
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
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        eval_strategy="steps" if evaluator is not None else "no",
        eval_steps=args.eval_steps,
        save_strategy="no" if args.no_save else "steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        logging_steps=10,
        dataloader_drop_last=True,
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
        # PyLate's collator is what makes queries and documents tokenize differently.
        data_collator=pylate_utils.ColBERTCollator(tokenize_fn=model.tokenize, prompts=prompts),
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
        "temperature": args.temperature,
        "num_negatives": num_negatives,
        "query_length": query_length,
        "document_length": document_length,
        "use_prompts": args.use_prompts,
        "train_rows": len(rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "lr": args.lr,
        "seed": args.seed,
    }
    (final_dir / "training_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[train] Saved -> {final_dir}")
    print(f"[train] Evaluate with:\n"
          f"  cd ../shared && uv run python evaluate.py --backend late-interaction "
          f"--model-path {final_dir}")


if __name__ == "__main__":
    main()
