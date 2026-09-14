#!/usr/bin/env python3
"""Score any handoff checkpoint on the shared evaluation set.

One evaluator for all four recipes. The metric computation is
`sentence_transformers.evaluation.InformationRetrievalEvaluator` in every case;
only the encoder differs:

  --backend dense             a SentenceTransformer, cosine similarity.
                              Used by 01_sentence_transformer,
                              03_bge_m3_flagembedding and 04_qwen3_embedding.

  --backend late-interaction  a PyLate ColBERT model. Token embeddings are
                              zero-padded into a (batch, tokens, dim) tensor and
                              scored with MaxSim, injected through the
                              evaluator's `score_functions` hook. Padding is
                              inert because MaxSim takes a max over document
                              tokens and the padded rows are all-zero.
                              Used by 02_late_interaction.

Checkpoints that ship no SentenceTransformer config (a plain transformers
directory, e.g. what `swift export` writes) must be given an explicit
`--pooling`; otherwise SentenceTransformer silently defaults to mean pooling.
Qwen3-Embedding needs `--pooling lasttoken --normalize`.

Usage:
  # baseline, before any training
  python evaluate.py --backend dense --model-path sentence-transformers/all-MiniLM-L6-v2

  # a trained dense checkpoint
  python evaluate.py --backend dense --model-path ../checkpoints/sentence_transformer/final

  # a trained ColBERT checkpoint (needs the 'late-interaction' extra)
  python evaluate.py --backend late-interaction --model-path ../checkpoints/late_interaction/final

  # a merged Qwen3-Embedding checkpoint
  python evaluate.py --backend dense --model-path ../checkpoints/qwen3_embedding/merged \
      --pooling lasttoken --normalize \
      --query-prompt "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieval evaluation for every handoff recipe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend", type=str, default="dense", choices=["dense", "late-interaction"])
    p.add_argument("--model-path", type=str, required=True,
                   help="Local checkpoint directory or a Hub model id.")
    p.add_argument("--data-dir", type=str, default="../data",
                   help="Directory holding eval_queries.jsonl and eval_corpus.jsonl.")
    p.add_argument("--name", type=str, default=None,
                   help="Label for the run. Defaults to the checkpoint basename.")

    # Dense encoder options
    p.add_argument("--pooling", type=str, default=None,
                   choices=["mean", "cls", "max", "lasttoken"],
                   help="Force a pooling mode by rebuilding the ST module stack. "
                        "Omit to load the checkpoint's own SentenceTransformer config.")
    p.add_argument("--normalize", action="store_true",
                   help="Append a Normalize module. Only applies with --pooling.")
    p.add_argument("--max-seq-length", type=int, default=512, help="Only applies with --pooling.")
    p.add_argument("--query-prompt", type=str, default=None,
                   help="Prefix prepended to every query before encoding.")
    p.add_argument("--corpus-prompt", type=str, default=None,
                   help="Prefix prepended to every document before encoding.")

    # Late-interaction options
    p.add_argument("--query-length", type=int, default=32)
    p.add_argument("--document-length", type=int, default=256)
    p.add_argument("--score-query-chunk", type=int, default=8,
                   help="Queries per MaxSim block. Lower it if scoring OOMs.")
    p.add_argument("--score-doc-chunk", type=int, default=256,
                   help="Documents per MaxSim block. Lower it if scoring OOMs.")

    # Shared
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--corpus-chunk-size", type=int, default=50000)
    p.add_argument("--max-queries", type=int, default=None,
                   help="Evaluate only the first N queries (quick smoke checks).")
    p.add_argument("--output", type=str, default=None,
                   help="Write the metrics to this JSON file.")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run prepare_data.py first.")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_eval_set(data_dir: Path, max_queries: int | None):
    """Return (queries, corpus, relevant_docs) in InformationRetrievalEvaluator form."""
    corpus_rows = read_jsonl(data_dir / "eval_corpus.jsonl")
    query_rows = read_jsonl(data_dir / "eval_queries.jsonl")
    if max_queries is not None:
        query_rows = query_rows[:max_queries]

    corpus = {row["docid"]: row["text"] for row in corpus_rows}
    queries = {row["qid"]: row["query"] for row in query_rows}
    relevant_docs = {row["qid"]: set(row["positive_docids"]) for row in query_rows}

    missing = {d for docs in relevant_docs.values() for d in docs} - corpus.keys()
    if missing:
        raise SystemExit(f"{len(missing)} relevant docids are absent from eval_corpus.jsonl.")
    return queries, corpus, relevant_docs


def build_dense_model(args):
    from sentence_transformers import SentenceTransformer, models

    if args.pooling is None:
        return SentenceTransformer(args.model_path, trust_remote_code=args.trust_remote_code)

    # In sentence-transformers 5.x trust_remote_code is passed through to the
    # underlying auto classes rather than being a Transformer kwarg.
    remote_code = {"trust_remote_code": True} if args.trust_remote_code else {}
    transformer = models.Transformer(
        args.model_path,
        max_seq_length=args.max_seq_length,
        model_args=remote_code,
        tokenizer_args=remote_code,
        config_args=remote_code,
    )
    modules = [
        transformer,
        models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=args.pooling),
    ]
    if args.normalize:
        modules.append(models.Normalize())
    return SentenceTransformer(modules=modules)


class PaddedColBERT:
    """Adapt a PyLate ColBERT model to the dense-encoder interface the evaluator expects.

    `InformationRetrievalEvaluator` calls `model.encode_query(...)` and
    `model.encode_document(...)` and hands the results straight to the configured
    score function. PyLate returns a list of variable-length `(tokens, dim)`
    tensors, so we pad each batch to its own longest sequence and stack. Every
    corpus chunk is scored independently, so chunks never need matching widths.

    Any other attribute access (`model_card_data`, `similarity_fn_name`, ...)
    falls through to the wrapped model.
    """

    def __init__(self, model, query_length: int, document_length: int):
        self.model = model
        self.query_length = query_length
        self.document_length = document_length

    def __getattr__(self, item):
        return getattr(self.model, item)

    def _encode(self, sentences, is_query: bool, prompt: str | None, batch_size: int,
                show_progress_bar: bool):
        import torch

        if prompt:
            sentences = [prompt + s for s in sentences]
        embeddings = self.model.encode(
            sentences,
            is_query=is_query,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=False,
            convert_to_tensor=True,
            padding=False,
        )
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        return torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value=0.0)

    def encode_query(self, sentences, prompt=None, batch_size=32, show_progress_bar=False, **kwargs):
        return self._encode(list(sentences), True, prompt, batch_size, bool(show_progress_bar))

    def encode_document(self, sentences, prompt=None, batch_size=32, show_progress_bar=False, **kwargs):
        return self._encode(list(sentences), False, prompt, batch_size, bool(show_progress_bar))


def make_max_sim(query_chunk: int, doc_chunk: int):
    """MaxSim scorer over zero-padded (batch, tokens, dim) tensors.

    Blocked over both axes: the intermediate is (q_chunk, d_chunk, q_tokens,
    d_tokens), which is what actually dominates memory here.
    """
    import torch

    def max_sim(queries, documents):
        queries = queries.float()
        documents = documents.float().to(queries.device)
        out = torch.zeros(queries.size(0), documents.size(0), device=queries.device)
        for qs in range(0, queries.size(0), query_chunk):
            q_block = queries[qs:qs + query_chunk]
            for ds in range(0, documents.size(0), doc_chunk):
                d_block = documents[ds:ds + doc_chunk]
                # (q, d, q_tokens, d_tokens) -> max over document tokens -> sum over query tokens
                sim = torch.einsum("qtd,psd->qpts", q_block, d_block)
                out[qs:qs + q_block.size(0), ds:ds + d_block.size(0)] = sim.max(dim=3).values.sum(dim=2)
        return out

    return max_sim


def main() -> None:
    args = parse_args()
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    data_dir = Path(args.data_dir).resolve()
    queries, corpus, relevant_docs = load_eval_set(data_dir, args.max_queries)
    name = args.name or Path(args.model_path).name
    print(f"[eval] {len(queries):,} queries against {len(corpus):,} documents "
          f"(backend={args.backend}, model={args.model_path})")

    evaluator_kwargs = dict(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        accuracy_at_k=args.k,
        precision_recall_at_k=args.k,
        mrr_at_k=args.k,
        ndcg_at_k=args.k,
        map_at_k=args.k,
        batch_size=args.batch_size,
        corpus_chunk_size=args.corpus_chunk_size,
        show_progress_bar=True,
        write_csv=False,
        name=name,
        query_prompt=args.query_prompt,
        corpus_prompt=args.corpus_prompt,
    )

    if args.backend == "dense":
        model = build_dense_model(args)
    else:
        try:
            from pylate import models as pylate_models
        except ImportError as exc:  # pragma: no cover - environment guard
            raise SystemExit(
                "The late-interaction backend needs PyLate. Install it with:\n"
                "  uv sync --extra late-interaction"
            ) from exc

        colbert = pylate_models.ColBERT(
            model_name_or_path=args.model_path,
            query_length=args.query_length,
            document_length=args.document_length,
        )
        model = PaddedColBERT(colbert, args.query_length, args.document_length)
        evaluator_kwargs["score_functions"] = {
            "maxsim": make_max_sim(args.score_query_chunk, args.score_doc_chunk)
        }

    evaluator = InformationRetrievalEvaluator(**evaluator_kwargs)
    metrics = evaluator(model)

    prefix = f"{name}_"
    score_fn = "cosine" if args.backend == "dense" else "maxsim"
    print()
    print(f"{'metric':<16}" + "".join(f"{('@' + str(k)):>10}" for k in args.k))
    print("-" * (16 + 10 * len(args.k)))
    for label, key in (("accuracy", "accuracy@"), ("precision", "precision@"),
                       ("recall", "recall@"), ("ndcg", "ndcg@"),
                       ("mrr", "mrr@"), ("map", "map@")):
        cells = []
        for k in args.k:
            value = metrics.get(f"{prefix}{score_fn}_{key}{k}")
            cells.append(f"{value * 100:>10.2f}" if value is not None else f"{'-':>10}")
        print(f"{label:<16}" + "".join(cells))

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "backend": args.backend,
            "model_path": args.model_path,
            "num_queries": len(queries),
            "num_documents": len(corpus),
            "metrics": metrics,
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\n[eval] Metrics -> {out_path}")


if __name__ == "__main__":
    main()
