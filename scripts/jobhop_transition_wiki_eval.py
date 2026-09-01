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


def parse_optional_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if str(x).lower() in {"none", "all", "-1"}:
        return None
    return int(x)


def read_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


def parse_title_desc(text):
    title = ""
    desc = text
    m = re.search(r"Occupation title:\s*(.*?)\.\s*Description:\s*(.*)$", text, flags=re.S)
    if m:
        title = m.group(1).strip()
        desc = m.group(2).strip()
    return title, desc


def truncate(text, n):
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0] + "..."


def code_prefixes(code):
    code = str(code)
    first_part = code.split(".", 1)[0]
    prefixes = []
    if first_part:
        prefixes.append(("unit4", first_part))
    if len(first_part) >= 2:
        prefixes.append(("minor2", first_part[:2]))
    if len(first_part) >= 1:
        prefixes.append(("major1", first_part[:1]))
    return prefixes


def extract_codes_from_query(query, valid_docids):
    found = []
    for x in re.findall(r"\b\d{4}(?:\.\d+)+\b", query):
        if x in valid_docids and x not in found:
            found.append(x)
    return found


def load_corpus(data_dir):
    corpus = {}
    for row in read_jsonl(data_dir / "eval_corpus.jsonl"):
        docid = str(row["docid"])
        text = str(row["text"])
        title, desc = parse_title_desc(text)
        corpus[docid] = {
            "docid": docid,
            "text": text,
            "title": title,
            "desc": desc,
            "tokens": Counter(tokenize(text)),
        }
    return corpus


def build_lexical_index(corpus):
    inv = defaultdict(list)
    df = Counter()
    doc_len = {}
    for docid, row in corpus.items():
        c = row["tokens"]
        doc_len[docid] = max(sum(c.values()), 1)
        for tok, freq in c.items():
            inv[tok].append((docid, freq))
            df[tok] += 1
    n = len(corpus)
    idf = {tok: math.log((n + 1) / (freq + 1)) + 1.0 for tok, freq in df.items()}
    return inv, idf, doc_len


def build_transition_wiki(data_dir, corpus, max_train_cases=None, history_window=5, recency_decay=0.65):
    valid_docids = set(corpus)
    trans = defaultdict(Counter)
    prefix_trans = defaultdict(Counter)
    target_prior = Counter()

    train_rows = read_jsonl(data_dir / "train.jsonl", limit=max_train_cases)
    used = 0
    no_history = 0

    for row in train_rows:
        query = str(row.get("query", ""))
        if row.get("positive_docids"):
            positives = [str(x) for x in row.get("positive_docids", []) if str(x) in valid_docids]
        elif row.get("target_docid") is not None:
            tgt = str(row.get("target_docid"))
            positives = [tgt] if tgt in valid_docids else []
        else:
            positives = []
        history = extract_codes_from_query(query, valid_docids)

        if not positives:
            continue
        if not history:
            no_history += 1
            continue

        used += 1
        gold = positives[0]
        target_prior[gold] += 1.0

        recent = list(reversed(history[-history_window:]))
        for age, src in enumerate(recent):
            weight = recency_decay ** age
            trans[src][gold] += weight

            for kind, pref in code_prefixes(src):
                key = f"{kind}:{pref}"
                if kind == "unit4":
                    prefix_trans[key][gold] += 0.75 * weight
                elif kind == "minor2":
                    prefix_trans[key][gold] += 0.35 * weight
                else:
                    prefix_trans[key][gold] += 0.15 * weight

    stats = {
        "train_rows_loaded": len(train_rows),
        "train_rows_used": used,
        "train_rows_no_history": no_history,
        "exact_source_pages": len(trans),
        "prefix_pages": len(prefix_trans),
        "target_prior_size": len(target_prior),
    }
    return trans, prefix_trans, target_prior, stats


def write_wiki_pages(out_dir, corpus, trans, prefix_trans):
    pages_dir = out_dir / "wiki_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    index_lines = ["# JobHop Transition Wiki", ""]
    for docid in sorted(corpus):
        row = corpus[docid]
        outgoing = trans.get(docid, Counter()).most_common(30)

        lines = [
            f"# ESCO {docid}",
            "",
            f"Title: {row['title']}",
            "",
            f"Description: {row['desc']}",
            "",
            "## Observed next occupations from train",
        ]

        if outgoing:
            total = sum(trans[docid].values())
            for tgt, w in outgoing:
                tgt_title = corpus.get(tgt, {}).get("title", "")
                lines.append(f"- {tgt} | {tgt_title} | weight={w:.3f} | p={w / max(total, 1e-9):.4f}")
        else:
            lines.append("- No direct outgoing transition observed in train.")

        (pages_dir / f"{docid}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        index_lines.append(f"- {docid}: {row['title']}")

    (out_dir / "wiki_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def preselect_pages(query, history, corpus, inv, idf, doc_len, trans, prefix_trans, target_prior, args):
    scores = defaultdict(float)
    evidence = defaultdict(list)

    def add(docid, amount, why):
        if docid in corpus:
            scores[docid] += amount
            if len(evidence[docid]) < 4:
                evidence[docid].append(why)

    recent = list(reversed(history[-args.history_window:]))

    for age, src in enumerate(recent):
        rec_weight = args.recency_decay ** age

        outgoing = trans.get(src, Counter()).most_common(args.top_transitions_per_source)
        total = sum(trans.get(src, Counter()).values())
        for rank, (tgt, w) in enumerate(outgoing):
            prob = w / max(total, 1e-9)
            amount = args.exact_transition_weight * rec_weight * (prob + 1.0 / (rank + 2))
            add(tgt, amount, f"exact_transition from {src}: weight={w:.2f}, p={prob:.3f}")

        for kind, pref in code_prefixes(src):
            key = f"{kind}:{pref}"
            outgoing2 = prefix_trans.get(key, Counter()).most_common(args.top_prefix_transitions)
            total2 = sum(prefix_trans.get(key, Counter()).values())

            if kind == "unit4":
                base_w = args.prefix4_weight
            elif kind == "minor2":
                base_w = args.prefix2_weight
            else:
                base_w = args.prefix1_weight

            for rank, (tgt, w) in enumerate(outgoing2):
                prob = w / max(total2, 1e-9)
                amount = base_w * rec_weight * (prob + 1.0 / (rank + 3))
                add(tgt, amount, f"{kind}_transition from {pref}: weight={w:.2f}, p={prob:.3f}")

    qtok = Counter(tokenize(query))
    lex_scores = defaultdict(float)
    for tok, qfreq in qtok.items():
        tok_idf = idf.get(tok)
        if tok_idf is None:
            continue
        for docid, dfreq in inv.get(tok, []):
            lex_scores[docid] += qfreq * dfreq * tok_idf * tok_idf / math.sqrt(doc_len[docid])

    if lex_scores:
        mx = max(lex_scores.values())
        for docid, s in lex_scores.items():
            add(docid, args.lexical_weight * s / max(mx, 1e-9), "lexical_page_match")

    if target_prior:
        mx_prior = max(target_prior.values())
        for docid, c in target_prior.most_common(args.top_prior):
            add(docid, args.prior_weight * c / max(mx_prior, 1e-9), "train_target_prior")

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if len(ranked) < args.page_candidate_k:
        seen = {x for x, _ in ranked}
        for docid in corpus:
            if docid not in seen:
                ranked.append((docid, 0.0))
                if len(ranked) >= args.page_candidate_k:
                    break

    out = []
    for docid, score in ranked[:args.page_candidate_k]:
        out.append({
            "docid": docid,
            "score": float(score),
            "evidence": evidence.get(docid, []),
        })
    return out


def build_prompt(case, history, candidates, corpus, args):
    lines = []
    for i, item in enumerate(candidates, start=1):
        docid = item["docid"]
        row = corpus[docid]
        lines.append(
            f"{i}. id={docid} | title={row['title']} | wiki_score={item['score']:.4f} | "
            f"description={truncate(row['desc'], args.page_desc_chars)}"
        )

    hist = ", ".join(history[-args.history_window:]) if history else "none"

    return f"""We are using a JobHop transition wiki to predict the next ESCO occupation.

Career history query:
{case.get("query", "")}

Extracted recent ESCO history:
{hist}

Candidate wiki pages:
{chr(10).join(lines)}

Task:
Select exactly {args.return_k} candidate ids that are the most plausible next occupations.

Decision rules:
- Transition evidence from train is a strong prior.
- Prefer candidates supported by recent exact transitions.
- Use occupation title and description only to break ties or reject implausible transitions.
- Do not invent ids.
- Use only ids from the candidate wiki pages.
- Return ids in descending relevance order.

Return exactly this JSON schema:
{{"selected_ids": ["id1", "id2"]}}"""


def parse_selected_ids(text):
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))

    ids = obj.get("selected_ids") or obj.get("ids") or obj.get("candidate_ids")
    if not isinstance(ids, list):
        raise ValueError("No selected_ids list in LLM response")
    return [str(x).strip() for x in ids if str(x).strip()]


def unique_valid(ids, valid_set):
    out = []
    seen = set()
    invalid = []
    for x in ids:
        if x in valid_set and x not in seen:
            out.append(x)
            seen.add(x)
        elif x not in valid_set:
            invalid.append(x)
    return out, invalid


def call_llm(prompt, args):
    payload = {
        "model": args.llm_model,
        "messages": [
            {"role": "system", "content": "Return JSON only. You are selecting ESCO occupation ids from a transition wiki."},
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }

    r = requests.post(
        args.llm_base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        json=payload,
        timeout=args.timeout,
    )
    r.raise_for_status()
    raw = r.json()
    return raw["choices"][0]["message"]["content"]


def eval_ranked(rows, ks):
    out = {"n_cases": len(rows)}
    for k in ks:
        acc = []
        mrr = []
        ndcg = []
        for row in rows:
            gold = set(row["target_docids"])
            ranked = row["ranked_ids"][:k]
            hit_rank = None
            for i, docid in enumerate(ranked, start=1):
                if docid in gold:
                    hit_rank = i
                    break
            acc.append(1.0 if hit_rank is not None else 0.0)
            mrr.append(0.0 if hit_rank is None else 1.0 / hit_rank)
            ndcg.append(0.0 if hit_rank is None else 1.0 / math.log2(hit_rank + 1))
        out[f"Accuracy@{k}"] = sum(acc) / max(len(acc), 1)
        out[f"Recall@{k}"] = out[f"Accuracy@{k}"]
        out[f"MRR@{k}"] = sum(mrr) / max(len(mrr), 1)
        out[f"NDCG@{k}"] = sum(ndcg) / max(len(ndcg), 1)
        out[f"MAP@{k}"] = out[f"MRR@{k}"]
    out["MRR"] = out.get("MRR@10", 0.0)
    return out


def run_one(i, case, corpus, inv, idf, doc_len, trans, prefix_trans, target_prior, args):
    valid_docids = set(corpus)
    history = extract_codes_from_query(str(case.get("query", "")), valid_docids)
    candidates = preselect_pages(str(case.get("query", "")), history, corpus, inv, idf, doc_len, trans, prefix_trans, target_prior, args)
    candidate_ids = [x["docid"] for x in candidates]
    llm_candidates = candidates[:args.llm_rerank_k] if args.llm_rerank_k else candidates
    llm_candidate_ids = [x["docid"] for x in llm_candidates]
    target_docids = [str(x) for x in case.get("positive_docids", [])]

    gold_in_candidates = bool(set(target_docids) & set(candidate_ids))

    base_row = {
        "case_index": i,
        "qid": case.get("qid"),
        "target_docids": target_docids,
        "ranked_ids": candidate_ids,
        "gold_in_page_candidates": gold_in_candidates,
    }

    if args.diagnose_only:
        return base_row, None, None, None

    prompt = build_prompt(case, history, llm_candidates, corpus, args)

    try:
        raw = call_llm(prompt, args)
        selected, invalid = unique_valid(parse_selected_ids(raw), set(llm_candidate_ids))
        ranked = selected + [x for x in candidate_ids if x not in set(selected)]

        parsed = {
            "case_index": i,
            "qid": case.get("qid"),
            "target_docids": target_docids,
            "history": history,
            "selected_ids": selected,
            "invalid_ids": invalid,
            "ranked_ids": ranked,
            "gold_in_page_candidates": gold_in_candidates,
            "target_in_selected": bool(set(target_docids) & set(selected)),
            "target_in_ranked": bool(set(target_docids) & set(ranked)),
        }
        response = {"case_index": i, "qid": case.get("qid"), "response": raw}
        return base_row, parsed, response, None

    except Exception as e:
        parsed = {
            "case_index": i,
            "qid": case.get("qid"),
            "target_docids": target_docids,
            "history": history,
            "selected_ids": [],
            "invalid_ids": [],
            "ranked_ids": candidate_ids,
            "gold_in_page_candidates": gold_in_candidates,
            "target_in_selected": False,
            "target_in_ranked": gold_in_candidates,
        }
        failure = {"case_index": i, "qid": case.get("qid"), "error_type": type(e).__name__, "error": str(e)}
        return base_row, parsed, None, failure


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="handoff/data_full")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--max-eval-cases", default="100")
    p.add_argument("--max-train-cases", default="none")
    p.add_argument("--output-dir", default="outputs/jobhop_transition_wiki")

    p.add_argument("--diagnose-only", action="store_true")
    p.add_argument("--diagnose-ks", default="20,50,60,100,200,300,500,1000,2000,2982")

    p.add_argument("--page-candidate-k", type=int, default=100)
    p.add_argument("--return-k", type=int, default=20)
    p.add_argument("--llm-rerank-k", type=int, default=None)
    p.add_argument("--page-desc-chars", type=int, default=160)
    p.add_argument("--history-window", type=int, default=5)
    p.add_argument("--recency-decay", type=float, default=0.65)

    p.add_argument("--top-transitions-per-source", type=int, default=120)
    p.add_argument("--top-prefix-transitions", type=int, default=200)
    p.add_argument("--top-prior", type=int, default=300)

    p.add_argument("--exact-transition-weight", type=float, default=8.0)
    p.add_argument("--prefix4-weight", type=float, default=2.5)
    p.add_argument("--prefix2-weight", type=float, default=1.0)
    p.add_argument("--prefix1-weight", type=float, default=0.25)
    p.add_argument("--lexical-weight", type=float, default=0.4)
    p.add_argument("--prior-weight", type=float, default=0.08)

    p.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=int, default=180)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    max_eval_cases = parse_optional_int(args.max_eval_cases)
    max_train_cases = parse_optional_int(args.max_train_cases)

    print('loading corpus...', flush=True)
    corpus = load_corpus(data_dir)
    print(f'loaded corpus: {len(corpus)}', flush=True)
    print('building lexical index...', flush=True)
    inv, idf, doc_len = build_lexical_index(corpus)
    print('lexical index ready', flush=True)
    print('building transition wiki from train...', flush=True)
    trans, prefix_trans, target_prior, train_stats = build_transition_wiki(
        data_dir, corpus, max_train_cases=max_train_cases,
        history_window=args.history_window,
        recency_decay=args.recency_decay,
    )

    print('writing wiki pages...', flush=True)
    write_wiki_pages(out_dir, corpus, trans, prefix_trans)
    print('wiki pages ready', flush=True)

    print('loading eval queries...', flush=True)
    queries = read_jsonl(data_dir / f"eval_queries_{args.split}.jsonl", limit=max_eval_cases)
    print(f'loaded eval queries: {len(queries)}', flush=True)

    write_json(out_dir / "run_config.json", vars(args))
    write_json(out_dir / "wiki_stats.json", {
        "n_corpus": len(corpus),
        **train_stats,
    })

    base_rows = []
    parsed_rows = []
    response_rows = []
    failure_rows = []

    with ThreadPoolExecutor(max_workers=args.concurrency if not args.diagnose_only else 1) as ex:
        futs = [
            ex.submit(run_one, i, case, corpus, inv, idf, doc_len, trans, prefix_trans, target_prior, args)
            for i, case in enumerate(queries)
        ]
        for n, fut in enumerate(as_completed(futs), start=1):
            base, parsed, response, failure = fut.result()
            base_rows.append(base)
            if parsed is not None:
                parsed_rows.append(parsed)
            if response is not None:
                response_rows.append(response)
            if failure is not None:
                failure_rows.append(failure)
            if n % 100 == 0 or n == len(futs):
                print(f"processed {n}/{len(futs)}", flush=True)

    base_rows.sort(key=lambda x: x["case_index"])
    parsed_rows.sort(key=lambda x: x["case_index"])
    response_rows.sort(key=lambda x: x["case_index"])
    failure_rows.sort(key=lambda x: x["case_index"])

    ks = [int(x) for x in args.diagnose_ks.split(",") if x.strip()]
    eval_ks = [1, 3, 5, 10, 20, 50]

    gold_rate = sum(1 for x in base_rows if x["gold_in_page_candidates"]) / max(len(base_rows), 1)

    metrics = {
        "n_cases": len(base_rows),
        "n_llm_failures": len(failure_rows),
        "avg_valid_ids": None if args.diagnose_only else (
            sum(len(x["selected_ids"]) for x in parsed_rows) / max(len(parsed_rows), 1)
        ),
        "gold_in_page_candidates_rate": gold_rate,
        "elapsed_seconds": time.time() - t0,
    }
    metrics["seconds_per_case"] = metrics["elapsed_seconds"] / max(len(base_rows), 1)
    metrics["page_selector_baseline"] = eval_ranked(base_rows, eval_ks)

    if not args.diagnose_only:
        metrics["llm_wiki"] = eval_ranked(parsed_rows, eval_ks)

    if args.diagnose_only:
        by_k = {}
        for k in ks:
            hit = 0
            for row in base_rows:
                gold = set(row["target_docids"])
                if gold & set(row["ranked_ids"][:k]):
                    hit += 1
            by_k[str(k)] = {"hits": hit, "total": len(base_rows), "rate": hit / max(len(base_rows), 1)}
        metrics["candidate_recall_by_k"] = by_k

    write_jsonl(out_dir / "candidate_cache.jsonl", base_rows)
    write_jsonl(out_dir / "parsed_results.jsonl", parsed_rows)
    write_jsonl(out_dir / "responses.jsonl", response_rows)
    write_jsonl(out_dir / "failures.jsonl", failure_rows)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "runtime.json", {"elapsed_seconds": metrics["elapsed_seconds"]})

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
