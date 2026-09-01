import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on", "at",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "this", "that",
    "career", "history", "query", "predict", "next", "esco", "occupation", "code",
    "title", "description", "level", "education"
}


def parse_limit(x):
    return None if str(x).lower() in {"none", "all", "-1"} else int(x)


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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 1 and t not in STOP]


def trunc(text, n):
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0] + "..."


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
            "raw_text": str(row["text"]),
            "raw_title": title,
            "raw_desc": desc,
        }
    return corpus


def load_entity_pages(wiki_dir):
    pages = {}
    for row in read_jsonl(Path(wiki_dir) / "entity_pages.jsonl"):
        docid = str(row["docid"])
        pages[docid] = row
    return pages


def page_blob(row):
    parts = [
        row.get("docid", ""),
        row.get("title", ""),
        row.get("summary", ""),
        " ".join(map(str, row.get("core_tasks", []))),
        " ".join(map(str, row.get("skills_or_context", []))),
        " ".join(map(str, row.get("career_transition_hints", []))),
    ]
    return " ".join(parts)


def build_index(pages, raw_corpus):
    docs = {}
    inv = defaultdict(list)
    df = Counter()

    for docid, page in pages.items():
        raw = raw_corpus.get(docid, {})
        text = page_blob(page) + " " + raw.get("raw_text", "")
        cnt = Counter(tokenize(text))
        docs[docid] = cnt
        for tok, freq in cnt.items():
            inv[tok].append((docid, freq))
            df[tok] += 1

    n = max(len(docs), 1)
    idf = {tok: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()}
    lens = {docid: max(sum(cnt.values()), 1) for docid, cnt in docs.items()}
    avgdl = sum(lens.values()) / max(len(lens), 1)
    return inv, idf, lens, avgdl


def bm25_rank(text, inv, idf, lens, avgdl, top_k, k1=1.5, b=0.75):
    q = Counter(tokenize(text))
    scores = defaultdict(float)

    for tok, qfreq in q.items():
        if tok not in idf:
            continue
        tok_idf = idf[tok]
        for docid, tf in inv.get(tok, []):
            dl = lens[docid]
            denom = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))
            scores[docid] += qfreq * tok_idf * (tf * (k1 + 1.0)) / max(denom, 1e-9)

    return [docid for docid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def index_line(docid, page, source="index"):
    title = page.get("title", "")
    summary = page.get("summary", "")
    return f"id={docid} | title={title} | summary={trunc(summary, 45)}"


def call_chat(prompt, system, args, max_tokens):
    payload = {
        "model": args.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_tokens,
        "seed": args.seed,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }

    r = requests.post(
        args.llm_base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        json=payload,
        timeout=args.timeout,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1200]}")
    return r.json()["choices"][0]["message"]["content"]


def parse_json_obj(text):
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def valid_unique(ids, valid):
    out, invalid = [], []
    seen = set()
    for x in ids:
        s = str(x).strip()
        if s in valid and s not in seen:
            out.append(s)
            seen.add(s)
        elif s not in valid:
            invalid.append(s)
    return out, invalid



def extract_history_codes(query, valid_docids):
    found = []
    for x in re.findall(r"\b\d{4}(?:\.\d+)+\b", str(query)):
        if x in valid_docids and x not in found:
            found.append(x)
    return found


def code_prefixes(code):
    code = str(code)
    head = code.split(".", 1)[0]
    out = []
    if head:
        out.append(("unit4", head))
    if len(head) >= 2:
        out.append(("minor2", head[:2]))
    if len(head) >= 1:
        out.append(("major1", head[:1]))
    return out


def build_train_transition_index(data_dir, valid_docids, args):
    trans = defaultdict(Counter)
    prefix_trans = defaultdict(Counter)
    prior = Counter()

    rows = read_jsonl(Path(data_dir) / "train.jsonl")
    used = 0

    for row in rows:
        target = row.get("target_docid")
        if target is None:
            continue
        target = str(target)
        if target not in valid_docids:
            continue

        history = extract_history_codes(row.get("query", ""), valid_docids)
        if not history:
            continue

        used += 1
        prior[target] += 1.0

        recent = list(reversed(history[-args.history_window:]))
        for age, src in enumerate(recent):
            rec = args.transition_recency_decay ** age
            trans[src][target] += rec

            for kind, pref in code_prefixes(src):
                key = f"{kind}:{pref}"
                if kind == "unit4":
                    w = 0.75
                elif kind == "minor2":
                    w = 0.35
                else:
                    w = 0.15
                prefix_trans[key][target] += w * rec

    stats = {
        "train_rows_loaded": len(rows),
        "train_rows_used": used,
        "exact_source_pages": len(trans),
        "prefix_pages": len(prefix_trans),
        "target_prior_size": len(prior),
    }
    return trans, prefix_trans, prior, stats


def transition_retrieve(case, trans, prefix_trans, prior, valid_docids, args):
    history = extract_history_codes(case.get("query", ""), valid_docids)
    scores = defaultdict(float)

    recent = list(reversed(history[-args.history_window:]))
    for age, src in enumerate(recent):
        rec = args.transition_recency_decay ** age

        outgoing = trans.get(src, Counter())
        total = sum(outgoing.values())
        for rank, (tgt, w) in enumerate(outgoing.most_common(args.transition_top_k)):
            scores[tgt] += args.exact_transition_weight * rec * (w / max(total, 1e-9) + 1.0 / (rank + 2))

        for kind, pref in code_prefixes(src):
            outgoing2 = prefix_trans.get(f"{kind}:{pref}", Counter())
            total2 = sum(outgoing2.values())
            for rank, (tgt, w) in enumerate(outgoing2.most_common(args.transition_top_k)):
                scores[tgt] += args.prefix_transition_weight * rec * (w / max(total2, 1e-9) + 1.0 / (rank + 3))

    if prior:
        mx = max(prior.values())
        for tgt, c in prior.most_common(args.transition_top_k):
            scores[tgt] += args.prior_transition_weight * c / max(mx, 1e-9)

    return [docid for docid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:args.transition_top_k]]

def route_query(case, pages, route_index_ids, args):
    lines = [index_line(docid, pages[docid]) for docid in route_index_ids if docid in pages]

    prompt = f"""Route this JobHop query to relevant ESCO Wiki entity pages.

Career history query:
{case.get("query", "")}

Candidate Wiki index:
{chr(10).join(lines)}

Return JSON only:
{{
  "selected_page_ids": ["id1", "id2"],
  "search_terms": ["term1", "term2"]
}}

Rules:
- Select at most {args.route_selected_pages} page ids.
- Use only page ids from the Candidate Wiki index.
- Search terms should help grep the frozen Wiki for additional relevant pages.
- Prefer plausible next occupations, not past occupations from the query.
"""
    raw = call_chat(
        prompt,
        "Return JSON only. You route JobHop queries to ESCO Wiki pages.",
        args,
        args.route_max_tokens,
    )
    obj = parse_json_obj(raw)
    selected, invalid = valid_unique(
        obj.get("selected_page_ids") or obj.get("selected_ids") or obj.get("ids") or [],
        set(route_index_ids),
    )
    terms = obj.get("search_terms") or obj.get("terms") or []
    terms = [str(x).strip() for x in terms if str(x).strip()][:args.max_search_terms]
    return selected[:args.route_selected_pages], terms, invalid, raw


def grep_retrieve(case, search_terms, pages, inv, idf, lens, avgdl, top_k):
    query = case.get("query", "") + " " + " ".join(search_terms)
    if not query.strip():
        return []
    return bm25_rank(query, inv, idf, lens, avgdl, top_k)


def raw_fallback_retrieve(case, search_terms, inv, idf, lens, avgdl, top_k):
    query = case.get("query", "") + " " + " ".join(search_terms)
    return bm25_rank(query, inv, idf, lens, avgdl, top_k)


def merge_candidates(selected_ids, transition_ids, grep_ids, raw_ids, candidate_pool_size):
    merged = []
    sources = {}
    for source, ids in [
        ("train_transition_kg", transition_ids),
        ("qwen_selected_pages", selected_ids),
        ("wiki_grep", grep_ids),
        ("raw_corpus_fallback", raw_ids),
    ]:
        for docid in ids:
            if docid not in sources:
                sources[docid] = source
                merged.append(docid)
            if len(merged) >= candidate_pool_size:
                break
        if len(merged) >= candidate_pool_size:
            break
    return merged, sources


def page_context(docid, page, raw_corpus, args):
    raw = raw_corpus.get(docid, {})
    parts = [
        f"id={docid}",
        f"title={page.get('title', raw.get('raw_title', ''))}",
        f"summary={trunc(page.get('summary', ''), args.context_summary_chars)}",
        f"tasks={trunc('; '.join(map(str, page.get('core_tasks', [])[:4])), args.context_list_chars)}",
        f"skills={trunc('; '.join(map(str, page.get('skills_or_context', [])[:4])), args.context_list_chars)}",
        f"hints={trunc('; '.join(map(str, page.get('career_transition_hints', [])[:4])), args.context_list_chars)}",
    ]
    return "\n".join(parts)


def candidate_table_line(rank, docid, page, raw_corpus, source, args):
    raw = raw_corpus.get(docid, {})
    title = page.get("title") or raw.get("raw_title", "")
    summary = page.get("summary") or raw.get("raw_desc", "")
    return f"id={docid} | title={title} | source={source} | summary={trunc(summary, args.candidate_summary_chars)}"


def final_rank(case, candidate_ids, sources, pages, raw_corpus, args):
    context_ids = candidate_ids[:args.final_context_pages]
    contexts = [page_context(docid, pages[docid], raw_corpus, args) for docid in context_ids if docid in pages]

    table = [
        candidate_table_line(i, docid, pages[docid], raw_corpus, sources.get(docid, "retrieved"), args)
        for i, docid in enumerate(candidate_ids, start=1)
        if docid in pages
    ]

    prompt = f"""Rank candidate ESCO occupation IDs for JobHop next-occupation prediction.

Career history query:
{case.get("query", "")}

Retrieved Wiki context:
{chr(10).join(chr(10) + "---" + chr(10) + x for x in contexts)}

Candidate occupation IDs:
{chr(10).join(table)}

Return JSON only:
{{
  "ranked_ids": ["id1", "id2"]
}}

Rules:
- Rank ONLY ESCO ids from the Candidate occupation IDs list.
- ESCO ids look like "2432.9" or "1221.3.2"; do not output row numbers like "1", "2", "47".
- Return up to {args.final_rank_k} ids.
- Prefer plausible next occupations, not occupations already in the career history.
- If uncertain, preserve the candidate order.
"""
    raw = call_chat(
        prompt,
        "Return JSON only. You rank retrieved ESCO candidate IDs for JobHop.",
        args,
        args.answer_max_tokens,
    )
    obj = parse_json_obj(raw)
    ranked, invalid = valid_unique(obj.get("ranked_ids") or obj.get("selected_ids") or obj.get("ids") or [], set(candidate_ids))
    return ranked, invalid, raw



def fuse_ranked_ids(candidate_ids, qwen_ranked_ids, mode, bonus):
    if mode == "hard":
        return qwen_ranked_ids + [x for x in candidate_ids if x not in set(qwen_ranked_ids)]

    qpos = {docid: i for i, docid in enumerate(qwen_ranked_ids, start=1)}
    scored = []
    for i, docid in enumerate(candidate_ids, start=1):
        retrieval_score = 1.0 / i
        qwen_score = bonus * (1.0 / qpos[docid]) if docid in qpos else 0.0
        scored.append((docid, retrieval_score + qwen_score))
    return [docid for docid, _ in sorted(scored, key=lambda x: x[1], reverse=True)]


def first_gold_rank(gold, ranked):
    gs = set(map(str, gold))
    for i, docid in enumerate(ranked, start=1):
        if docid in gs:
            return i
    return None


def eval_ranked(rows, ks):
    out = {"n_cases": len(rows)}
    for k in ks:
        acc, mrr, ndcg = [], [], []
        for row in rows:
            rank = first_gold_rank(row["target_docids"], row["ranked_ids"][:k])
            acc.append(1.0 if rank else 0.0)
            mrr.append(0.0 if not rank else 1.0 / rank)
            ndcg.append(0.0 if not rank else 1.0 / math.log2(rank + 1))
        out[f"Accuracy@{k}"] = sum(acc) / max(len(acc), 1)
        out[f"Recall@{k}"] = out[f"Accuracy@{k}"]
        out[f"MRR@{k}"] = sum(mrr) / max(len(mrr), 1)
        out[f"NDCG@{k}"] = sum(ndcg) / max(len(ndcg), 1)
        out[f"MAP@{k}"] = out[f"MRR@{k}"]
    out["MRR"] = out.get("MRR@10", 0.0)
    return out


def run_one(i, case, pages, raw_corpus, wiki_inv, wiki_idf, wiki_lens, wiki_avgdl, raw_inv, raw_idf, raw_lens, raw_avgdl, trans, prefix_trans, target_prior, args):
    target = case.get("positive_docids") or ([case["target_docid"]] if case.get("target_docid") is not None else [])
    target = [str(x) for x in target]

    route_failed = False
    ranking_failed = False
    route_raw = ""
    answer_raw = ""
    route_invalid = []
    rank_invalid = []

    route_index_ids = bm25_rank(case.get("query", ""), wiki_inv, wiki_idf, wiki_lens, wiki_avgdl, args.route_index_candidates)

    try:
        selected_ids, search_terms, route_invalid, route_raw = route_query(case, pages, route_index_ids, args)
    except Exception as e:
        route_failed = True
        selected_ids = []
        search_terms = tokenize(case.get("query", ""))[:args.max_search_terms]
        route_raw = f"ROUTE_ERROR: {type(e).__name__}: {e}"

    transition_ids = transition_retrieve(case, trans, prefix_trans, target_prior, set(pages), args)
    grep_ids = grep_retrieve(case, search_terms, pages, wiki_inv, wiki_idf, wiki_lens, wiki_avgdl, args.grep_top_k)
    raw_ids = raw_fallback_retrieve(case, search_terms, raw_inv, raw_idf, raw_lens, raw_avgdl, args.raw_fallback_top_k)
    candidate_ids, sources = merge_candidates(selected_ids, transition_ids, grep_ids, raw_ids, args.candidate_pool_size)

    gold_in_selected = bool(set(target) & set(selected_ids))
    gold_in_transition = bool(set(target) & set(transition_ids))
    gold_in_grep = bool(set(target) & set(grep_ids))
    gold_in_raw = bool(set(target) & set(raw_ids))
    gold_in_candidates = bool(set(target) & set(candidate_ids))
    rank_before = first_gold_rank(target, candidate_ids)

    try:
        qwen_ranked_ids, rank_invalid, answer_raw = final_rank(case, candidate_ids, sources, pages, raw_corpus, args)
        ranked_ids = fuse_ranked_ids(candidate_ids, qwen_ranked_ids, args.final_fusion_mode, args.qwen_rank_bonus)
    except Exception as e:
        ranking_failed = True
        qwen_ranked_ids = []
        ranked_ids = candidate_ids
        answer_raw = f"RANK_ERROR: {type(e).__name__}: {e}"

    rank_after = first_gold_rank(target, ranked_ids)

    row = {
        "case_index": i,
        "qid": case.get("qid"),
        "target_docids": target,
        "route_failed": route_failed,
        "ranking_failed": ranking_failed,
        "selected_page_ids": selected_ids,
        "search_terms": search_terms,
        "transition_candidate_ids": transition_ids,
        "grep_page_ids": grep_ids,
        "raw_fallback_ids": raw_ids,
        "candidate_ids": candidate_ids,
        "qwen_ranked_ids": qwen_ranked_ids,
        "ranked_ids": ranked_ids,
        "route_invalid_ids": route_invalid,
        "rank_invalid_ids": rank_invalid,
        "gold_in_selected_pages": gold_in_selected,
        "gold_in_transition_kg": gold_in_transition,
        "gold_in_grep_pages": gold_in_grep,
        "gold_in_raw_fallback": gold_in_raw,
        "gold_in_final_candidate_ids": gold_in_candidates,
        "gold_rank_before_qwen": rank_before,
        "gold_rank_after_qwen": rank_after,
        "avg_marker": 1,
    }

    response = {
        "case_index": i,
        "qid": case.get("qid"),
        "route_response": route_raw,
        "answer_response": answer_raw,
    }

    failure = None
    if route_failed or ranking_failed:
        failure = {
            "case_index": i,
            "qid": case.get("qid"),
            "route_failed": route_failed,
            "ranking_failed": ranking_failed,
            "route_response": route_raw[:1200],
            "answer_response": answer_raw[:1200],
        }

    return row, response, failure


def mean_present(vals):
    vals = [x for x in vals if x is not None]
    return None if not vals else sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="handoff/data_full")
    ap.add_argument("--wiki-dir", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--max-eval-cases", default="5")
    ap.add_argument("--output-dir", default="outputs/strict_llm_wiki")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--route-index-candidates", type=int, default=180)
    ap.add_argument("--route-selected-pages", type=int, default=20)
    ap.add_argument("--max-search-terms", type=int, default=8)
    ap.add_argument("--grep-top-k", type=int, default=60)
    ap.add_argument("--raw-fallback-top-k", type=int, default=60)
    ap.add_argument("--transition-top-k", type=int, default=100)
    ap.add_argument("--history-window", type=int, default=5)
    ap.add_argument("--transition-recency-decay", type=float, default=0.65)
    ap.add_argument("--exact-transition-weight", type=float, default=8.0)
    ap.add_argument("--prefix-transition-weight", type=float, default=1.0)
    ap.add_argument("--prior-transition-weight", type=float, default=0.05)
    ap.add_argument("--candidate-pool-size", type=int, default=100)
    ap.add_argument("--final-context-pages", type=int, default=12)
    ap.add_argument("--final-rank-k", type=int, default=50)
    ap.add_argument("--final-fusion-mode", choices=["hard", "bonus"], default="hard")
    ap.add_argument("--qwen-rank-bonus", type=float, default=0.2)

    ap.add_argument("--candidate-summary-chars", type=int, default=80)
    ap.add_argument("--context-summary-chars", type=int, default=160)
    ap.add_argument("--context-list-chars", type=int, default=120)

    ap.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--route-max-tokens", type=int, default=384)
    ap.add_argument("--answer-max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir) / args.split
    out.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    raw_corpus = load_raw_corpus(data_dir)
    pages = load_entity_pages(args.wiki_dir)

    wiki_inv, wiki_idf, wiki_lens, wiki_avgdl = build_index(pages, raw_corpus)
    raw_inv, raw_idf, raw_lens, raw_avgdl = build_index(
        {docid: {"docid": docid, "title": row["raw_title"], "summary": row["raw_text"]} for docid, row in raw_corpus.items()},
        raw_corpus,
    )

    trans, prefix_trans, target_prior, transition_stats = build_train_transition_index(data_dir, set(pages), args)
    write_json(out / "transition_stats.json", transition_stats)

    queries = read_jsonl(data_dir / f"eval_queries_{args.split}.jsonl", limit=parse_limit(args.max_eval_cases))

    print(f"loaded pages: {len(pages)}", flush=True)
    print(f"transition stats: {transition_stats}", flush=True)
    print(f"loaded queries: {len(queries)}", flush=True)

    rows, responses, failures = [], [], []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(run_one, i, q, pages, raw_corpus, wiki_inv, wiki_idf, wiki_lens, wiki_avgdl, raw_inv, raw_idf, raw_lens, raw_avgdl, trans, prefix_trans, target_prior, args)
            for i, q in enumerate(queries)
        ]
        for n, fut in enumerate(as_completed(futs), start=1):
            row, resp, fail = fut.result()
            rows.append(row)
            responses.append(resp)
            if fail:
                failures.append(fail)
            if n % 10 == 0 or n == len(futs):
                print(f"processed {n}/{len(futs)}", flush=True)

    rows.sort(key=lambda x: x["case_index"])
    responses.sort(key=lambda x: x["case_index"])
    failures.sort(key=lambda x: x["case_index"])

    ks = [1, 3, 5, 10, 20, 50]
    retrieval_rows = [{"target_docids": r["target_docids"], "ranked_ids": r["candidate_ids"]} for r in rows]
    final_rows = [{"target_docids": r["target_docids"], "ranked_ids": r["ranked_ids"]} for r in rows]

    metrics = {
        "n_cases": len(rows),
        "n_pages": len(pages),
        "seed": args.seed,
        "route_failures": sum(1 for r in rows if r["route_failed"]),
        "ranking_failures": sum(1 for r in rows if r["ranking_failed"]),
        "avg_selected_pages": sum(len(r["selected_page_ids"]) for r in rows) / max(len(rows), 1),
        "avg_search_terms": sum(len(r["search_terms"]) for r in rows) / max(len(rows), 1),
        "avg_candidate_ids": sum(len(r["candidate_ids"]) for r in rows) / max(len(rows), 1),
        "avg_ranked_ids": sum(len(r["ranked_ids"]) for r in rows) / max(len(rows), 1),
        "gold_in_selected_pages_rate": sum(1 for r in rows if r["gold_in_selected_pages"]) / max(len(rows), 1),
        "gold_in_transition_kg_rate": sum(1 for r in rows if r["gold_in_transition_kg"]) / max(len(rows), 1),
        "gold_in_grep_pages_rate": sum(1 for r in rows if r["gold_in_grep_pages"]) / max(len(rows), 1),
        "gold_in_raw_fallback_rate": sum(1 for r in rows if r["gold_in_raw_fallback"]) / max(len(rows), 1),
        "gold_in_final_candidate_ids_rate": sum(1 for r in rows if r["gold_in_final_candidate_ids"]) / max(len(rows), 1),
        "candidate_recall@50": sum(1 for r in rows if set(r["target_docids"]) & set(r["candidate_ids"][:50])) / max(len(rows), 1),
        "candidate_recall@100": sum(1 for r in rows if set(r["target_docids"]) & set(r["candidate_ids"][:100])) / max(len(rows), 1),
        "mean_gold_rank_before_qwen": mean_present([r["gold_rank_before_qwen"] for r in rows]),
        "mean_gold_rank_after_qwen": mean_present([r["gold_rank_after_qwen"] for r in rows]),
        "oracle_accuracy@1": sum(1 for r in rows if r["gold_in_final_candidate_ids"]) / max(len(rows), 1),
        "elapsed_seconds": time.time() - t0,
    }
    metrics["seconds_per_case"] = metrics["elapsed_seconds"] / max(len(rows), 1)
    metrics["retrieval_without_final_qwen"] = eval_ranked(retrieval_rows, ks)
    metrics["strict_llm_wiki"] = eval_ranked(final_rows, ks)

    write_json(out / "run_config.json", vars(args))
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "parsed_results.jsonl", rows)
    write_jsonl(out / "responses.jsonl", responses)
    write_jsonl(out / "failures.jsonl", failures)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
