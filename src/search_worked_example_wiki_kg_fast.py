import math
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from jobhop_transition_wiki_eval import (
    build_lexical_index,
    build_transition_wiki,
    extract_codes_from_query,
    load_corpus,
    preselect_pages,
    read_jsonl,
    tokenize,
)

DATA_DIR = Path("handoff/data_full")
OUT = Path("outputs/paper_checks/worked_example_search")
OUT.mkdir(parents=True, exist_ok=True)

MAX_CASES = 5000


def title(corpus, docid):
    return corpus.get(str(docid), {}).get("title") or str(docid)


def fmt(corpus, docid):
    return f"{title(corpus, docid)} ({docid})"


def gold_rank(ids, gold):
    ids = [str(x) for x in ids]
    gold = str(gold)
    return ids.index(gold) + 1 if gold in ids else None


def rank_str(rank):
    return "Not retrieved" if rank is None else str(rank)


def build_bm25_cache(corpus):
    doc_tokens = {docid: row["tokens"] for docid, row in corpus.items()}
    doc_len = {docid: max(sum(c.values()), 1) for docid, c in doc_tokens.items()}
    n_docs = len(corpus)
    avgdl = sum(doc_len.values()) / max(n_docs, 1)

    df = Counter()
    postings = defaultdict(list)
    for docid, c in doc_tokens.items():
        for tok, freq in c.items():
            df[tok] += 1
            postings[tok].append((docid, freq))

    idf = {
        tok: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
        for tok, freq in df.items()
    }
    return postings, idf, doc_len, avgdl


def bm25_rank_fast(query, corpus, bm25_cache, top_k=100):
    postings, idf, doc_len, avgdl = bm25_cache
    toks = Counter(tokenize(query))
    k1 = 1.5
    b = 0.75
    scores = defaultdict(float)

    for tok, qf in toks.items():
        tok_idf = idf.get(tok)
        if tok_idf is None:
            continue
        for docid, f in postings.get(tok, []):
            dl = doc_len[docid]
            scores[docid] += tok_idf * (f * (k1 + 1)) / (
                f + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
            )

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def minmax(items):
    if not items:
        return {}
    vals = [v for _, v in items]
    lo, hi = min(vals), max(vals)
    den = max(hi - lo, 1e-9)
    return {k: (v - lo) / den for k, v in items}


def wiki_kg_fusion_rank(kg_items, wiki_items, alpha=0.8, top_k=100):
    kg = minmax(kg_items)
    wiki = minmax(wiki_items)
    keys = set(kg) | set(wiki)
    scored = [(k, alpha * kg.get(k, 0.0) + (1 - alpha) * wiki.get(k, 0.0)) for k in keys]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]


def main():
    print("loading corpus...", flush=True)
    corpus = load_corpus(DATA_DIR)

    print("building lexical index...", flush=True)
    inv, idf, doc_len = build_lexical_index(corpus)

    print("building fast BM25 cache...", flush=True)
    bm25_cache = build_bm25_cache(corpus)

    print("building train transition KG...", flush=True)
    trans, prefix_trans, target_prior, stats = build_transition_wiki(
        DATA_DIR,
        corpus,
        max_train_cases=None,
        history_window=5,
        recency_decay=0.65,
    )

    args = Namespace(
        history_window=5,
        recency_decay=0.65,
        top_transitions_per_source=120,
        top_prefix_transitions=200,
        top_prior=300,
        exact_transition_weight=8.0,
        prefix4_weight=2.5,
        prefix2_weight=1.0,
        prefix1_weight=0.25,
        lexical_weight=0.0,
        prior_weight=0.05,
        page_candidate_k=100,
    )

    cases = read_jsonl(DATA_DIR / "eval_queries_test.jsonl")[:MAX_CASES]
    rows = []

    print(f"scanning {len(cases)} test cases...", flush=True)
    for i, case in enumerate(cases):
        query = str(case.get("query", ""))
        golds = [str(x) for x in case.get("positive_docids", [])]
        if not golds:
            continue
        gold = golds[0]
        history = extract_codes_from_query(query, set(corpus))
        if not history:
            continue
        if gold in history:
            continue

        kg_ranked = preselect_pages(
            query, history, corpus, inv, idf, doc_len,
            trans, prefix_trans, target_prior, args
        )
        kg_items = [(x["docid"], float(x["score"])) for x in kg_ranked]
        kg_ids = [x[0] for x in kg_items]

        wiki_items = bm25_rank_fast(query, corpus, bm25_cache, top_k=100)
        wiki_ids = [x[0] for x in wiki_items]

        fusion_items = wiki_kg_fusion_rank(kg_items, wiki_items, alpha=0.8, top_k=100)
        fusion_ids = [x[0] for x in fusion_items]

        kg_r = gold_rank(kg_ids, gold)
        wiki_r = gold_rank(wiki_ids, gold)
        fusion_r = gold_rank(fusion_ids, gold)

        if fusion_r is not None and fusion_r <= 10 and (kg_r is None or kg_r > 10):
            rows.append({
                "case_index": i,
                "history_len": len(history),
                "last_code": history[-1],
                "last_title": title(corpus, history[-1]),
                "gold": gold,
                "gold_title": title(corpus, gold),
                "kg_rank": rank_str(kg_r),
                "wiki_rank": rank_str(wiki_r),
                "fusion_rank": rank_str(fusion_r),
                "kg_top1": fmt(corpus, kg_ids[0]),
                "kg_top2": fmt(corpus, kg_ids[1]),
                "kg_top3": fmt(corpus, kg_ids[2]),
                "wiki_top1": fmt(corpus, wiki_ids[0]),
                "wiki_top2": fmt(corpus, wiki_ids[1]),
                "wiki_top3": fmt(corpus, wiki_ids[2]),
                "fusion_top1": fmt(corpus, fusion_ids[0]),
                "fusion_top2": fmt(corpus, fusion_ids[1]),
                "fusion_top3": fmt(corpus, fusion_ids[2]),
                "query": query,
            })

        if i % 250 == 0:
            print(f"processed {i}/{len(cases)}; found {len(rows)}", flush=True)
        if len(rows) >= 30:
            print("found enough examples; stopping early", flush=True)
            break

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wiki_kg_improvement_candidates_fast.csv", index=False)

    if not df.empty:
        df2 = df.copy()
        df2["fusion_rank_num"] = df2["fusion_rank"].astype(int)
        df2["kg_rank_score"] = df2["kg_rank"].map(lambda x: 999 if x == "Not retrieved" else int(x))
        df2 = df2.sort_values(["fusion_rank_num", "history_len", "kg_rank_score"], ascending=[True, True, False])
        best = df2.head(20).drop(columns=["fusion_rank_num", "kg_rank_score"])
        best.to_csv(OUT / "wiki_kg_improvement_best20_fast.csv", index=False)

    print("WROTE:", flush=True)
    print(OUT / "wiki_kg_improvement_candidates_fast.csv", flush=True)
    print(OUT / "wiki_kg_improvement_best20_fast.csv", flush=True)
    print("", flush=True)
    if not df.empty:
        print(pd.read_csv(OUT / "wiki_kg_improvement_best20_fast.csv").head(10).to_string(index=False), flush=True)
    else:
        print("No cases found under current criteria.", flush=True)


if __name__ == "__main__":
    main()
