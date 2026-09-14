import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "at",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "this", "that",
    "career", "history", "query", "predict", "next", "esco", "occupation", "code",
    "title", "description", "level", "education"
}

def parse_limit(x):
    if x is None or str(x).lower() in {"none", "all", "-1"}:
        return None
    return int(x)

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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in STOP and len(t) > 1]

def parse_title_desc(text):
    m = re.search(r"Occupation title:\s*(.*?)\.\s*Description:\s*(.*)$", str(text), flags=re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", str(text)

def load_raw_corpus(data_dir):
    corpus = {}
    for row in read_jsonl(data_dir / "eval_corpus.jsonl"):
        docid = str(row["docid"])
        title, desc = parse_title_desc(row["text"])
        corpus[docid] = {
            "docid": docid,
            "title": title,
            "desc": desc,
            "text": str(row["text"]),
        }
    return corpus

def load_entity_pages(wiki_dir):
    pages = {}
    for row in read_jsonl(wiki_dir / "entity_pages.jsonl"):
        pages[str(row["docid"])] = row
    return pages

def page_blob(page, raw):
    parts = [
        page.get("title", ""),
        page.get("summary", ""),
        " ".join(map(str, page.get("core_tasks", []))),
        " ".join(map(str, page.get("skills_or_context", []))),
        " ".join(map(str, page.get("career_transition_hints", []))),
        raw.get("title", ""),
        raw.get("desc", ""),
    ]
    return " ".join(parts)

def build_bm25_index(docs):
    inv = defaultdict(list)
    df = Counter()
    lens = {}
    for docid, text in docs.items():
        cnt = Counter(tokenize(text))
        lens[docid] = max(sum(cnt.values()), 1)
        for tok, freq in cnt.items():
            inv[tok].append((docid, freq))
            df[tok] += 1
    n = len(docs)
    avgdl = sum(lens.values()) / max(n, 1)
    idf = {tok: math.log((n + 1) / (freq + 1)) + 1.0 for tok, freq in df.items()}
    return inv, idf, lens, avgdl

def bm25_rank(text, inv, idf, lens, avgdl, top_k, k1=1.5, b=0.75):
    q = Counter(tokenize(text))
    scores = defaultdict(float)
    for tok, qfreq in q.items():
        tok_idf = idf.get(tok)
        if tok_idf is None:
            continue
        for docid, tf in inv.get(tok, []):
            dl = lens[docid]
            denom = tf + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
            scores[docid] += qfreq * tok_idf * (tf * (k1 + 1)) / max(denom, 1e-9)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

def extract_history_codes(query, valid_docids):
    found = []
    for x in re.findall(r"\b\d{4}(?:\.\d+)+\b", str(query)):
        if x in valid_docids and x not in found:
            found.append(x)
    return found

def code_prefixes(code):
    first = str(code).split(".", 1)[0]
    out = []
    if first:
        out.append(("unit4", first))
    if len(first) >= 2:
        out.append(("minor2", first[:2]))
    if len(first) >= 1:
        out.append(("major1", first[:1]))
    return out

def build_train_transition_index(data_dir, valid_docids, args):
    trans = defaultdict(Counter)
    prefix_trans = defaultdict(Counter)
    prior = Counter()

    rows = read_jsonl(data_dir / "train.jsonl")
    used = 0
    no_history = 0

    for row in rows:
        if row.get("positive_docids"):
            targets = [str(x) for x in row.get("positive_docids", []) if str(x) in valid_docids]
        elif row.get("target_docid") is not None:
            x = str(row.get("target_docid"))
            targets = [x] if x in valid_docids else []
        else:
            targets = []

        hist = extract_history_codes(row.get("query", ""), valid_docids)
        if not targets:
            continue
        if not hist:
            no_history += 1
            continue

        used += 1
        gold = targets[0]
        prior[gold] += 1.0

        recent = list(reversed(hist[-args.history_window:]))
        for age, src in enumerate(recent):
            rec = args.transition_recency_decay ** age
            trans[src][gold] += rec
            for kind, pref in code_prefixes(src):
                key = f"{kind}:{pref}"
                if kind == "unit4":
                    prefix_trans[key][gold] += 0.75 * rec
                elif kind == "minor2":
                    prefix_trans[key][gold] += 0.35 * rec
                else:
                    prefix_trans[key][gold] += 0.15 * rec

    stats = {
        "train_rows_loaded": len(rows),
        "train_rows_used": used,
        "train_rows_no_history": no_history,
        "exact_source_pages": len(trans),
        "prefix_pages": len(prefix_trans),
        "target_prior_size": len(prior),
    }
    return trans, prefix_trans, prior, stats

def transition_rank(case, trans, prefix_trans, prior, valid_docids, args):
    scores = defaultdict(float)
    hist = extract_history_codes(case.get("query", ""), valid_docids)
    recent = list(reversed(hist[-args.history_window:]))

    for age, src in enumerate(recent):
        rec = args.transition_recency_decay ** age

        outgoing = trans.get(src, Counter())
        total = sum(outgoing.values())
        for rank, (tgt, w) in enumerate(outgoing.most_common(args.transition_top_k)):
            scores[tgt] += args.exact_transition_weight * rec * (w / max(total, 1e-9) + 1.0 / (rank + 2))

        for kind, pref in code_prefixes(src):
            outgoing2 = prefix_trans.get(f"{kind}:{pref}", Counter())
            total2 = sum(outgoing2.values())
            base = args.prefix4_weight if kind == "unit4" else args.prefix2_weight if kind == "minor2" else args.prefix1_weight
            for rank, (tgt, w) in enumerate(outgoing2.most_common(args.transition_top_k)):
                scores[tgt] += base * rec * (w / max(total2, 1e-9) + 1.0 / (rank + 3))

    if args.prior_weight > 0 and prior:
        mx = max(prior.values())
        for tgt, c in prior.most_common(args.transition_top_k):
            scores[tgt] += args.prior_weight * c / max(mx, 1e-9)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def merge_scored(primary, secondary, top_k):
    out = []
    seen = set()
    for docid, _ in primary:
        if docid not in seen:
            out.append(docid)
            seen.add(docid)
        if len(out) >= top_k:
            return out
    for docid, _ in secondary:
        if docid not in seen:
            out.append(docid)
            seen.add(docid)
        if len(out) >= top_k:
            return out
    return out

def first_gold_rank(gold, ranked):
    gs = set(map(str, gold))
    for i, docid in enumerate(ranked, 1):
        if docid in gs:
            return i
    return None

def eval_ranked(rows, ks):
    out = {"n_cases": len(rows)}
    for k in ks:
        hits = []
        rr = []
        ndcg = []
        for row in rows:
            rank = first_gold_rank(row["target_docids"], row["ranked_ids"][:k])
            hits.append(1.0 if rank is not None else 0.0)
            rr.append(0.0 if rank is None else 1.0 / rank)
            ndcg.append(0.0 if rank is None else 1.0 / math.log2(rank + 1))
        out[f"Accuracy@{k}"] = sum(hits) / max(len(hits), 1)
        out[f"Recall@{k}"] = out[f"Accuracy@{k}"]
        out[f"MRR@{k}"] = sum(rr) / max(len(rr), 1)
        out[f"NDCG@{k}"] = sum(ndcg) / max(len(ndcg), 1)
        out[f"MAP@{k}"] = out[f"MRR@{k}"]
    out["MRR"] = out.get("MRR@10", 0.0)
    return out

def target_docids(case):
    if case.get("positive_docids"):
        return [str(x) for x in case["positive_docids"]]
    if case.get("target_docid") is not None:
        return [str(case["target_docid"])]
    return []

def run_mode(mode, queries, wiki_rankings, raw_rankings, trans, prefix_trans, prior, valid_docids, args):
    rows = []
    for i, case in enumerate(queries):
        gold = target_docids(case)

        wiki = wiki_rankings[i]
        raw = raw_rankings[i]
        kg = transition_rank(case, trans, prefix_trans, prior, valid_docids, args)

        if mode == "wiki_only":
            ranked = merge_scored(wiki, raw, args.candidate_pool_size)
        elif mode == "kg_alone":
            ranked = merge_scored(kg, raw, args.candidate_pool_size)
        elif mode == "wiki_kg":
            ranked = merge_scored(kg, wiki + raw, args.candidate_pool_size)
        else:
            raise ValueError(mode)

        rows.append({
            "case_index": i,
            "qid": case.get("qid", f"{args.split}_{i}"),
            "target_docids": gold,
            "ranked_ids": ranked,
            "gold_in_top100": bool(set(gold) & set(ranked[:100])),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="handoff/data_full")
    ap.add_argument("--wiki-dir", default="outputs/qwen_entity_wiki_full")
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--max-eval-cases", default="1000")
    ap.add_argument("--output-dir", default="outputs/wiki_kg_ablation")
    ap.add_argument("--candidate-pool-size", type=int, default=100)
    ap.add_argument("--wiki-top-k", type=int, default=100)
    ap.add_argument("--raw-top-k", type=int, default=100)
    ap.add_argument("--transition-top-k", type=int, default=100)
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

    wiki_rankings = [bm25_rank(q.get("query", ""), wiki_inv, wiki_idf, wiki_lens, wiki_avgdl, args.wiki_top_k) for q in queries]
    raw_rankings = [bm25_rank(q.get("query", ""), raw_inv, raw_idf, raw_lens, raw_avgdl, args.raw_top_k) for q in queries]

    metrics = {
        "n_cases": len(queries),
        "n_pages": len(pages),
        "split": args.split,
        "elapsed_seconds": None,
        "transition_stats": transition_stats,
    }

    for mode in ["wiki_only", "kg_alone", "wiki_kg"]:
        rows = run_mode(mode, queries, wiki_rankings, raw_rankings, trans, prefix_trans, prior, valid_docids, args)
        write_jsonl(out / f"{mode}_parsed_results.jsonl", rows)
        metrics[mode] = eval_ranked(rows, [1, 3, 5, 10, 20, 50, 100])
        metrics[mode]["candidate_recall@100"] = sum(1 for r in rows if r["gold_in_top100"]) / max(len(rows), 1)

    metrics["elapsed_seconds"] = time.time() - t0
    metrics["seconds_per_case"] = metrics["elapsed_seconds"] / max(len(queries), 1)

    write_json(out / "run_config.json", vars(args))
    write_json(out / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
