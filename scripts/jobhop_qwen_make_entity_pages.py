import argparse, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def read_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows

def parse_title(text):
    m = re.search(r"Occupation title:\s*(.*?)\.\s*Description:", text, flags=re.S)
    return m.group(1).strip() if m else ""

def parse_desc(text):
    m = re.search(r"Description:\s*(.*)$", text, flags=re.S)
    return m.group(1).strip() if m else text

def call_qwen(row, args):
    docid = str(row["docid"])
    text = str(row["text"])
    title = parse_title(text)
    desc = parse_desc(text)

    prompt = f"""Create a concise LLM-Wiki entity page for this ESCO occupation.

Source occupation:
id: {docid}
title: {title}
description: {desc}

Return JSON only with this schema:
{{
  "docid": "{docid}",
  "title": "...",
  "summary": "...",
  "core_tasks": ["...", "..."],
  "skills_or_context": ["...", "..."],
  "career_transition_hints": ["...", "..."]
}}

Rules:
- Do not invent another ESCO id.
- Use only information supported by the source occupation.
- Keep each field concise.
"""

    payload = {
        "model": args.llm_model,
        "messages": [
            {"role": "system", "content": "Return JSON only. You write concise wiki entity pages for ESCO occupations."},
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
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1200]}")
    content = r.json()["choices"][0]["message"]["content"]
    obj = json.loads(content)
    return docid, obj, content

def page_markdown(obj):
    docid = obj.get("docid", "")
    title = obj.get("title", "")
    summary = obj.get("summary", "")
    tasks = obj.get("core_tasks", [])
    skills = obj.get("skills_or_context", [])
    hints = obj.get("career_transition_hints", [])

    lines = [f"# {docid} {title}", "", "## Summary", summary, "", "## Core tasks"]
    lines += [f"- {x}" for x in tasks]
    lines += ["", "## Skills or context"]
    lines += [f"- {x}" for x in skills]
    lines += ["", "## Career transition hints"]
    lines += [f"- {x}" for x in hints]
    return "\n".join(lines).strip() + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="handoff/data_full")
    ap.add_argument("--output-dir", default="outputs/qwen_entity_wiki")
    ap.add_argument("--max-pages", default="20")
    ap.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    limit = None if str(args.max_pages).lower() in {"none", "all", "-1"} else int(args.max_pages)
    out = Path(args.output_dir)
    pages_dir = out / "wiki_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(Path(args.data_dir) / "eval_corpus.jsonl", limit=limit)
    print(f"loaded corpus pages: {len(rows)}", flush=True)

    entity_path = out / "entity_pages.jsonl"
    failure_path = out / "failures.jsonl"
    index_lines = ["# JobHop Qwen Entity Wiki", ""]

    successes, failures = [], []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(call_qwen, row, args): row for row in rows}
        for n, fut in enumerate(as_completed(futs), start=1):
            row = futs[fut]
            docid = str(row["docid"])
            try:
                docid, obj, raw = fut.result()
                (pages_dir / f"{docid}.md").write_text(page_markdown(obj), encoding="utf-8")
                successes.append(obj)
                index_lines.append(f"- {docid}: {obj.get('title', '')} | {obj.get('summary', '')}")
            except Exception as e:
                failures.append({"docid": docid, "error_type": type(e).__name__, "error": str(e)})
            if n % 10 == 0 or n == len(rows):
                print(f"processed {n}/{len(rows)}", flush=True)

    with open(entity_path, "w", encoding="utf-8") as f:
        for obj in successes:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(failure_path, "w", encoding="utf-8") as f:
        for obj in failures:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    (out / "wiki_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    metrics = {
        "n_input_pages": len(rows),
        "n_entity_pages": len(successes),
        "n_failures": len(failures),
        "elapsed_seconds": time.time() - t0,
        "seconds_per_page": (time.time() - t0) / max(len(rows), 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
