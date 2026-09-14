import json
import math
import re
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from jobhop_transition_wiki_eval import (
    build_lexical_index,
    build_transition_wiki,
    load_corpus,
    preselect_pages,
    read_jsonl,
    tokenize,
)

DATA_DIR = Path("handoff/data_full")
OUT = Path("outputs/paper_checks/worked_example_commissioning")
OUT.mkdir(parents=True, exist_ok=True)

HISTORY_ID = "2149.6"
GOLD_ID = "3119.4"


def title(corpus, docid):
    row = corpus.get(str(docid), {})
    return row.get("title") or str(docid)


def fmt_candidate(corpus, docid):
    return f"{title(corpus, docid)} ({docid})"


def find_case(corpus):
    rows = read_jsonl(DATA_DIR / "eval_queries_test.jsonl")
    for i, row in enumerate(rows):
        q = str(row.get("query", ""))
        positives = [str(x) for x in row.get("positive_docids", [])]
        if HISTORY_ID in q and GOLD_ID in positives:
            row["case_index"] = i
            return row
    raise RuntimeError("Could not find requested test case.")


def gold_rank(ranked_ids):
    ranked_ids = [str(x) for x in ranked_ids]
    return ranked_ids.index(GOLD_ID) + 1 if GOLD_ID in ranked_ids else "Not retrieved"


def bm25_rank(query, corpus, top_k=100):
    toks = Counter(tokenize(query))
    doc_tokens = {docid: row["tokens"] for docid, row in corpus.items()}
    n_docs = len(corpus)
    avgdl = sum(sum(c.values()) for c in doc_tokens.values()) / max(n_docs, 1)
    df = Counter()
    for c in doc_tokens.values():
        for t in c:
            df[t] += 1

    k1 = 1.5
    b = 0.75
    scores = defaultdict(float)

    for tok, qf in toks.items():
        if tok not in df:
            continue
        idf = math.log((n_docs - df[tok] + 0.5) / (df[tok] + 0.5) + 1.0)
        for docid, c in doc_tokens.items():
            f = c.get(tok, 0)
            if not f:
                continue
            dl = sum(c.values())
            scores[docid] += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avgdl, 1e-9)))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


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


def dense_top3(model_name, case_query, corpus, query_prefix="", batch_size=32):
    from sentence_transformers import SentenceTransformer

    device = "cuda"
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    docids = sorted(corpus)
    texts = [corpus[d]["text"] for d in docids]
    q = query_prefix + case_query

    doc_emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    q_emb = model.encode(
        [q],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    scores = doc_emb @ q_emb
    order = np.argsort(-scores)
    ranked_ids = [docids[i] for i in order]
    return ranked_ids


def main():
    corpus = load_corpus(DATA_DIR)
    case = find_case(corpus)
    query = case["query"]

    print("CASE INDEX:", case["case_index"])
    print("QUERY:", query)
    print("GOLD:", fmt_candidate(corpus, GOLD_ID))

    # KG / Wiki / Wiki+KG
    inv, idf, doc_len = build_lexical_index(corpus)
    trans, prefix_trans, target_prior, stats = build_transition_wiki(
        DATA_DIR,
        corpus,
        max_train_cases=None,
        history_window=5,
        recency_decay=0.65,
    )

    kg_args = Namespace(
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

    kg_ranked = preselect_pages(
        query,
        [HISTORY_ID],
        corpus,
        inv,
        idf,
        doc_len,
        trans,
        prefix_trans,
        target_prior,
        kg_args,
    )
    kg_items = [(x["docid"], float(x["score"])) for x in kg_ranked]
    kg_ids = [x[0] for x in kg_items]

    wiki_items = bm25_rank(query, corpus, top_k=100)
    wiki_ids = [x[0] for x in wiki_items]

    fusion_items = wiki_kg_fusion_rank(kg_items, wiki_items, alpha=0.8, top_k=100)
    fusion_ids = [x[0] for x in fusion_items]

    # Dense methods. These use local model paths if available; adjust only if your paths differ.
    bge_ids = dense_top3(
        "models/BAAI_bge-m3",
        query,
        corpus,
        query_prefix="",
        batch_size=32,
    )

    rows = []
    methods = [
        ("BGE-M3", bge_ids),
        ("Transition KG", kg_ids),
        ("Wiki only", wiki_ids),
        ("Wiki+KG fusion", fusion_ids),
    ]

    for method, ranked_ids in methods:
        rows.append({
            "Method": method,
            "Top-1": fmt_candidate(corpus, ranked_ids[0]) if len(ranked_ids) > 0 else "",
            "Top-2": fmt_candidate(corpus, ranked_ids[1]) if len(ranked_ids) > 1 else "",
            "Top-3": fmt_candidate(corpus, ranked_ids[2]) if len(ranked_ids) > 2 else "",
            "Gold rank": gold_rank(ranked_ids),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "worked_example_top3.csv", index=False)

    md = []
    md.append("# Worked Example: Commissioning Engineer -> Commissioning Technician")
    md.append("")
    md.append(f"- case_index: `{case['case_index']}`")
    md.append(f"- history: `{fmt_candidate(corpus, HISTORY_ID)}`")
    md.append(f"- gold: `{fmt_candidate(corpus, GOLD_ID)}`")
    md.append("")
    md.append(df.to_markdown(index=False))
    md.append("")
    md.append("## LaTeX")
    md.append("")
    md.append("```latex")
    md.append(r"\begin{table*}[!t]")
    md.append(r"\centering")
    md.append(r"\small")
    md.append(r"\begin{tabular}{llllr}")
    md.append(r"\toprule")
    md.append(r"\textbf{Method} & \textbf{Top-1} & \textbf{Top-2} & \textbf{Top-3} & \textbf{Gold rank} \\")
    md.append(r"\midrule")
    for _, r in df.iterrows():
        vals = [str(r[c]).replace("&", r"\&").replace("_", r"\_") for c in ["Method", "Top-1", "Top-2", "Top-3", "Gold rank"]]
        md.append(" & ".join(vals) + r" \\")
    md.append(r"\bottomrule")
    md.append(r"\end{tabular}")
    md.append(r"\caption{\textbf{Worked example for an unseen transition.}}")
    md.append(r"\label{tab:worked_example}")
    md.append(r"\end{table*}")
    md.append("```")
    md.append("")

    (OUT / "worked_example_top3.md").write_text("\n".join(md), encoding="utf-8")

    print("WROTE:")
    print(OUT / "worked_example_top3.csv")
    print(OUT / "worked_example_top3.md")
    print()
    print((OUT / "worked_example_top3.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
