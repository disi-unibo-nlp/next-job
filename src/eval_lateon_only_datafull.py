import argparse, csv, json, math, time
from pathlib import Path

INDEX_NAME = "esco_occupations"

def read_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def parse_limit(x):
    return None if str(x).lower() in {"none", "all", "-1"} else int(x)

def parse_ks(x):
    return [int(v) for v in str(x).split(",") if v.strip()]

def parse_retrieved_row(row):
    parsed = []
    if isinstance(row, dict):
        iterable = row.items()
    else:
        iterable = enumerate(row)
    for key, value in iterable:
        if isinstance(value, dict):
            docid = value.get("id") or value.get("doc_id") or value.get("document_id") or value.get("corpus_id")
            score = value.get("score", value.get("similarity", 0.0))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            docid, score = value[0], value[1]
        else:
            docid, score = key, value
        if docid is not None:
            parsed.append((str(docid), float(score)))
    return parsed

def summarize(ranks, ks):
    n = len(ranks)
    out = {"n_cases": n}
    for k in ks:
        hits = [r for r in ranks if r is not None and r <= k]
        acc = len(hits) / n
        out[f"Accuracy@{k}"] = acc
        out[f"Precision@{k}"] = acc / k
        out[f"Recall@{k}"] = acc
        out[f"MRR@{k}"] = sum(1.0 / r for r in hits) / n
        out[f"NDCG@{k}"] = sum(1.0 / math.log2(r + 1) for r in hits) / n
        out[f"MAP@{k}"] = out[f"MRR@{k}"]
    out["MRR"] = out.get("MRR@10")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="handoff/data_full")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--model-name", default="lightonai/LateOn")
    ap.add_argument("--index-dir", default="outputs/lateon_eval_compare/lightonai_LateOn/pylate_index")
    ap.add_argument("--max-eval-cases", default="none")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--retrieve-k", type=int, default=200)
    ap.add_argument("--metric-ks", default="1,3,5,10,20,50,100")
    ap.add_argument("--output-dir", default="outputs/paper_checks/lateon_only_datafull")
    args = ap.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    limit = parse_limit(args.max_eval_cases)
    ks = parse_ks(args.metric_ks)
    max_k = max(max(ks), args.retrieve_k)

    queries = read_jsonl(data_dir / f"eval_queries_{args.split}.jsonl", limit=limit)
    corpus = read_jsonl(data_dir / "eval_corpus.jsonl")
    valid_docids = {str(r["docid"]) for r in corpus}

    write_json(out_dir / "run_config.json", {
        "script": "scripts/eval_lateon_only_datafull.py",
        "data_dir": str(data_dir),
        "split": args.split,
        "model_name": args.model_name,
        "index_dir": args.index_dir,
        "max_eval_cases": limit,
        "batch_size": args.batch_size,
        "retrieve_k": args.retrieve_k,
        "metric_ks": ks,
        "alpha_markov": 0.0,
        "alpha_lateon": 1.0,
        "n_queries": len(queries),
        "n_corpus": len(corpus),
        "note": "LateOn-only evaluation on data_full, reusing existing PyLate index.",
    })

    print("loading PyLate model/index...", flush=True)
    from pylate import indexes, models, retrieve

    colbert_model = models.ColBERT(
        model_name_or_path=args.model_name,
        device="cuda",
    )

    index = indexes.PLAID(
        index_folder=str(Path(args.index_dir)),
        index_name=INDEX_NAME,
        override=False,
    )
    retriever = retrieve.ColBERT(index=index)

    print("encoding/retrieving queries:", len(queries), flush=True)
    query_texts = [str(r["query"]) for r in queries]
    query_embeddings = colbert_model.encode(
        query_texts,
        batch_size=args.batch_size,
        is_query=True,
        show_progress_bar=True,
    )
    raw_rows = retriever.retrieve(queries_embeddings=query_embeddings, k=max_k)
    ranked_rows = [parse_retrieved_row(r) for r in raw_rows]

    print("scoring...", flush=True)
    ranks = []
    per_query = []
    warnings = []

    for i, (q, ranked) in enumerate(zip(queries, ranked_rows)):
        target = str((q.get("positive_docids") or [""])[0])
        ranked_ids = [docid for docid, _ in ranked[:max_k]]
        if target not in valid_docids:
            warnings.append({"case_index": i, "qid": q.get("qid"), "target": target, "warning": "target_not_in_corpus"})
        rank = ranked_ids.index(target) + 1 if target in ranked_ids else None
        ranks.append(rank)

        per_query.append({
            "case_index": i,
            "qid": q.get("qid"),
            "person_id": q.get("person_id"),
            "target": target,
            "rank": rank,
            "acc1": 1.0 if rank == 1 else 0.0,
            "mrr10": (1.0 / rank) if rank is not None and rank <= 10 else 0.0,
            "ndcg10": (1.0 / math.log2(rank + 1)) if rank is not None and rank <= 10 else 0.0,
            "recall10": 1.0 if rank is not None and rank <= 10 else 0.0,
            "top100": ranked_ids[:100],
        })

    metrics = summarize(ranks, ks)
    result = {
        "model": args.model_name,
        "split": args.split,
        "alpha_markov": 0.0,
        "alpha_lateon": 1.0,
        "retrieve_k": args.retrieve_k,
        "max_eval_cases": limit,
        **metrics,
    }

    write_csv(out_dir / "lateon_only_datafull_results.csv", [result])
    write_json(out_dir / "lateon_only_datafull_results.json", result)
    write_jsonl(out_dir / "lateon_only_datafull_per_query.jsonl", per_query)
    write_json(out_dir / "warnings.json", warnings)
    write_json(out_dir / "runtime.json", {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "seconds_per_case": (time.time() - started) / max(1, len(queries)),
        "n_cases": len(queries),
    })

    print("DONE", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
