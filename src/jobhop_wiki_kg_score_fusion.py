import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from jobhop_wiki_kg_ablation import (
    parse_limit, read_jsonl, write_json, write_jsonl,
    load_raw_corpus, load_entity_pages, page_blob,
    build_bm25_index, bm25_rank,
    build_train_transition_index, transition_rank,
    target_docids, first_gold_rank, eval_ranked,
)

def normalize(items):
    if not items:
        return {}
    vals = [s for _, s in items]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {docid: 1.0 for docid, _ in items}
    return {docid: (score - lo) / (hi - lo) for docid, score in items}

def fill_with_fallback(ranked, fallback_items, top_k):
    seen = set(ranked)
    out = list(ranked)
    for docid, _ in fallback_items:
        if docid not in seen:
            out.append(docid)
            seen.add(docid)
        if len(out) >= top_k:
            break
    return out[:top_k]

def fusion_rank(kg_items, wiki_items, raw_items, alpha, top_k):
    kg = normalize(kg_items)
    wiki = normalize(wiki_items)

    ids = set(kg) | set(wiki)
    scored = []
    for docid in ids:
        score = alpha * kg.get(docid, 0.0) + (1.0 - alpha) * wiki.get(docid, 0.0)
        scored.append((docid, score))

    ranked = [docid for docid, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    return fill_with_fallback(ranked, raw_items, top_k)

def make_rows(method, queries, rankings, args):
    rows = []
    for i, case in enumerate(queries):
        gold = target_docids(case)
        ranked = rankings[i]
        rows.append({
            "case_index": i,
            "qid": case.get("qid", f"{args.split}_{i}"),
            "target_docids": gold,
            "ranked_ids": ranked,
            "gold_in_top100": bool(set(gold) & set(ranked[:100])),
            "method": method,
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="handoff/data_full")
    ap.add_argument("--wiki-dir", default="outputs/qwen_entity_wiki_full")
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--max-eval-cases", default="1000")
    ap.add_argument("--output-dir", default="outputs/wiki_kg_score_fusion")
    ap.add_argument("--candidate-pool-size", type=int, default=100)
    ap.add_argument("--wiki-top-k", type=int, default=100)
    ap.add_argument("--raw-top-k", type=int, default=100)
    ap.add_argument("--transition-top-k", type=int, default=100)
    ap.add_argument("--alphas", default="0.7,0.8,0.9,0.95")
    ap.add_argument("--history-window", type=int, default=5)
    ap.add_argument("--transition-recency-decay", type=float, default=0.65)
    ap.add_argument("--exact-transition-weight", type=float, default=8.0)
    ap.add_argument("--prefix4-weight", type=float, default=1.0)
    ap.add_argument("--prefix2-weight", type=float, default=1.0)
    ap.add_argument("--prefix1-weight", type=float, default=1.0)
    ap.add_argument("--prior-weight", type=float, default=0.05)
    args = ap.parse_args()

    t0 = time.time()
    data_dir = Path(args.data_dir)
    out = Path(args.output_dir) / args.split
    out.mkdir(parents=True, exist_ok=True)

    raw_corpus = load_raw_corpus(data_dir)
    pages = load_entity_pages(Path(args.wiki_dir))
    valid_docids = set(pages)

    wiki_docs = {docid: page_blob(page, raw_corpus.get(docid, {})) for docid, page in pages.items()}
    raw_docs = {docid: row["text"] for docid, row in raw_corpus.items() if docid in valid_docids}

    wiki_inv, wiki_idf, wiki_lens, wiki_avgdl = build_bm25_index(wiki_docs)
    raw_inv, raw_idf, raw_lens, raw_avgdl = build_bm25_index(raw_docs)
    trans, prefix_trans, prior, transition_stats = build_train_transition_index(data_dir, valid_docids, args)

    queries = read_jsonl(data_dir / f"eval_queries_{args.split}.jsonl", limit=parse_limit(args.max_eval_cases))
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    wiki_items_all = [bm25_rank(q.get("query", ""), wiki_inv, wiki_idf, wiki_lens, wiki_avgdl, args.wiki_top_k) for q in queries]
    raw_items_all = [bm25_rank(q.get("query", ""), raw_inv, raw_idf, raw_lens, raw_avgdl, args.raw_top_k) for q in queries]
    kg_items_all = [transition_rank(q, trans, prefix_trans, prior, valid_docids, args)[:args.transition_top_k] for q in queries]

    metrics = {
        "n_cases": len(queries),
        "n_pages": len(pages),
        "split": args.split,
        "transition_stats": transition_stats,
        "alphas": alphas,
    }

    for alpha in alphas:
        method = f"wiki_kg_fusion_alpha_{alpha:g}"
        rankings = [
            fusion_rank(kg_items_all[i], wiki_items_all[i], raw_items_all[i], alpha, args.candidate_pool_size)
            for i in range(len(queries))
        ]
        rows = make_rows(method, queries, rankings, args)
        write_jsonl(out / f"{method}_parsed_results.jsonl", rows)
        metrics[method] = eval_ranked(rows, [1, 3, 5, 10, 20, 50, 100])
        metrics[method]["candidate_recall@100"] = sum(1 for r in rows if r["gold_in_top100"]) / max(len(rows), 1)

    metrics["elapsed_seconds"] = time.time() - t0
    metrics["seconds_per_case"] = metrics["elapsed_seconds"] / max(len(queries), 1)

    write_json(out / "run_config.json", vars(args))
    write_json(out / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
