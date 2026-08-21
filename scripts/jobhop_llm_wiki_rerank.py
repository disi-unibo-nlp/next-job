import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import requests
import torch
from tqdm.auto import tqdm

from dense_eval import DENSE_MODEL_DEFAULTS, encode_texts, load_sentence_transformer_model
from jobhop_eval_common import (
    build_texts,
    candidate_text,
    cleanup_cuda,
    configure_hf_cache,
    configure_reproducibility,
    evaluate_ranked_lists_at_k,
    failure_payload,
    gpu_info,
    label_for_code,
    load_jobhop_eval_data,
    model_slug,
    now_utc,
    package_versions,
    parse_k_values,
    parse_optional_int,
    write_csv,
    write_json,
    write_json_and_key_value_csv,
)


SYSTEM_PROMPT = """You are an expert career trajectory reranker.
Your task is to choose the most plausible next ESCO occupations from a provided candidate list.
Return JSON only. Do not add explanations."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="JobHop LLM-Wiki-style list reranking smoke test using a local OpenAI-compatible LLM endpoint."
    )
    parser.add_argument("--retriever-model", required=True, help="SentenceTransformer model name or local path.")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--max-eval-cases", default="20", help="Integer sample size per split, or 'none'.")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--return-k", type=int, default=20)
    parser.add_argument("--metric-ks", default="1,3,5,10,20,50,100")
    parser.add_argument("--output-dir", default="outputs/llm_wiki_rerank")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--candidate-cache", default=None, help="JSONL path for saved retriever top-k candidates.")
    parser.add_argument("--save-candidates-only", action="store_true", help="Save retriever top-k candidates and exit.")
    parser.add_argument("--rerank-from-candidates", action="store_true", help="Load retriever top-k candidates from JSONL.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--query-prefix", default=None)
    parser.add_argument("--passage-prefix", default=None)
    parser.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--fusion-mode", default="bonus", choices=["bonus", "reorder"])
    parser.add_argument("--rank-bonus", type=float, default=1.0)
    parser.add_argument("--candidate-description-chars", type=int, default=350)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build candidates and prompts, save them, but do not call the LLM endpoint.",
    )
    return parser.parse_args()


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def reset_jsonl_outputs(run_dir):
    for name in ["requests.jsonl", "responses.jsonl", "parsed_results.jsonl", "failures.jsonl"]:
        path = Path(run_dir) / name
        if path.exists():
            path.unlink()


def apply_prefixes(texts, prefix):
    return [prefix + text for text in texts] if prefix else texts


def resolve_device(device, require_cuda):
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run, but torch.cuda.is_available() is False.")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
    return device


def top_k_ranked_lists(score_matrix, candidate_codes, candidate_k):
    top_k = min(candidate_k, score_matrix.shape[1])
    ranked_lists = []
    for start in tqdm(range(0, score_matrix.shape[0]), desc="top-k candidates"):
        row = score_matrix[start]
        kth = max(0, top_k - 1)
        idx = np.argpartition(-row, kth=kth)[:top_k]
        idx = idx[np.argsort(-row[idx])]
        ranked_lists.append([(candidate_codes[i], float(row[i])) for i in idx])
    return ranked_lists


def save_candidate_cache(path, cases, ranked_lists):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for i, (case, ranked) in enumerate(zip(cases, ranked_lists)):
            row = {
                "case_index": i,
                "case": case,
                "target": case["target"],
                "ranked_candidates": [{"code": code, "score": score} for code, score in ranked],
            }
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def load_candidate_cache(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    cases = [row["case"] for row in rows]
    ranked_lists = [
        [(item["code"], float(item["score"])) for item in row["ranked_candidates"]]
        for row in rows
    ]
    return rows, cases, ranked_lists


def build_history_block(metadata_lookup, case):
    lines = []
    for i, code in enumerate(case["prefix"], start=1):
        lines.append(f"{i}. {code} | {label_for_code(metadata_lookup, code)}")
    education = case.get("education")
    if education is not None and str(education).lower() != "nan":
        lines.append(f"Education: {education}")
    return "\n".join(lines)


def truncate_candidate_text(metadata_lookup, code, max_chars):
    text = candidate_text(metadata_lookup, code)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def build_user_prompt(metadata_lookup, case, ranked_candidates, return_k, max_desc_chars):
    candidates = []
    for rank, (code, score) in enumerate(ranked_candidates, start=1):
        text = truncate_candidate_text(metadata_lookup, code, max_desc_chars)
        candidates.append(f"{rank}. id={code} | retriever_score={score:.6f} | {text}")
    return f"""We are doing JobHop last-step next-occupation prediction.

Career history:
{build_history_block(metadata_lookup, case)}

Candidate ESCO occupations from the retriever:
{chr(10).join(candidates)}

Select exactly {return_k} candidate ids that are the most plausible next occupations.
Use only ids from the candidate list. Preserve descending relevance order.
Return exactly this JSON schema:
{{"selected_ids": ["id1", "id2"]}}"""


def strip_code_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_selected_ids(text):
    cleaned = strip_code_fences(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if isinstance(payload, list):
        ids = payload
    else:
        ids = payload.get("selected_ids") or payload.get("ids") or payload.get("candidate_ids")
    if not isinstance(ids, list):
        raise ValueError("LLM response did not contain a list of selected ids.")
    return [str(item).strip() for item in ids if str(item).strip()]


def call_openai_compatible_chat(args, user_prompt):
    url = args.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": args.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    headers = {"Authorization": f"Bearer {args.llm_api_key}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload, timeout=args.timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"], data


def unique_valid_ids(ids, candidate_ids):
    valid = set(candidate_ids)
    seen = set()
    kept = []
    invalid = []
    for item in ids:
        if item not in valid:
            invalid.append(item)
            continue
        if item in seen:
            continue
        seen.add(item)
        kept.append(item)
    return kept, invalid


def rerank_with_llm(ranked, selected_ids, mode, rank_bonus):
    selected_pos = {code: pos for pos, code in enumerate(selected_ids)}
    if mode == "reorder":
        selected = [(code, score) for code, score in ranked if code in selected_pos]
        selected.sort(key=lambda item: selected_pos[item[0]])
        rest = [(code, score) for code, score in ranked if code not in selected_pos]
        return selected + rest

    if not ranked:
        return ranked
    base_scores = np.array([score for _, score in ranked], dtype="float32")
    mn = float(base_scores.min())
    mx = float(base_scores.max())
    denom = max(mx - mn, 1e-8)
    llm_count = max(len(selected_ids), 1)
    rescored = []
    for code, score in ranked:
        base = (score - mn) / denom
        bonus = 0.0
        if code in selected_pos:
            bonus = rank_bonus * (llm_count - selected_pos[code]) / llm_count
        rescored.append((code, float(base + bonus)))
    rescored.sort(key=lambda item: item[1], reverse=True)
    return rescored


def main():
    args = parse_args()
    started_at = now_utc()
    started_monotonic = time.monotonic()
    stage = "initialization"
    max_eval_cases = parse_optional_int(args.max_eval_cases)
    metric_ks = parse_k_values(args.metric_ks)
    cache_dir = configure_hf_cache(args.cache_dir)
    configure_reproducibility(args.random_seed)

    defaults = DENSE_MODEL_DEFAULTS.get(args.retriever_model, {})
    batch_size = args.batch_size or defaults.get("batch_size", 16)
    query_prefix = args.query_prefix if args.query_prefix is not None else defaults.get("query_prefix", "")
    passage_prefix = args.passage_prefix if args.passage_prefix is not None else defaults.get("passage_prefix", "")
    run_dir = Path(args.output_dir) / model_slug(args.retriever_model) / args.split
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "script": "scripts/jobhop_llm_wiki_rerank.py",
        "adapted_from": "LLM_Wiki local OpenAI-compatible JSON-calling pattern",
        "protocol": "JobHop last-step next-occupation prediction; LLM reranks retriever top-k candidates.",
        "retriever_model": args.retriever_model,
        "split": args.split,
        "max_eval_cases": max_eval_cases,
        "random_seed": args.random_seed,
        "batch_size": batch_size,
        "candidate_k": args.candidate_k,
        "return_k": args.return_k,
        "metric_ks": metric_ks,
        "output_dir": str(run_dir),
        "cache_dir": cache_dir,
        "candidate_cache": args.candidate_cache,
        "save_candidates_only": args.save_candidates_only,
        "rerank_from_candidates": args.rerank_from_candidates,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "temperature": args.temperature,
        "fusion_mode": args.fusion_mode,
        "rank_bonus": args.rank_bonus,
        "dry_run": args.dry_run,
    }
    write_json_and_key_value_csv(run_dir / "run_config.json", run_dir / "run_config.csv", run_config)
    reset_jsonl_outputs(run_dir)
    write_json_and_key_value_csv(run_dir / "gpu_info.json", run_dir / "gpu_info.csv", gpu_info())
    write_json_and_key_value_csv(run_dir / "package_versions.json", run_dir / "package_versions.csv", package_versions())
    runtime = {"started_at_utc": started_at, "status": "running", "stage": stage}
    write_json_and_key_value_csv(run_dir / "runtime.json", run_dir / "runtime.csv", runtime)

    try:
        stage = "device_check"
        device = resolve_device(args.device, args.require_cuda)
        run_config["device"] = device
        write_json_and_key_value_csv(run_dir / "run_config.json", run_dir / "run_config.csv", run_config)

        stage = "data_loading_and_preprocessing"
        data = load_jobhop_eval_data(max_eval_cases=max_eval_cases, seed=args.random_seed)
        cases = data[f"{args.split}_cases"]
        texts = build_texts(data["metadata_lookup"], data["candidate_codes"], data["validation_cases"], data["test_cases"])
        query_texts = texts[f"{args.split}_query_texts"]
        write_csv(run_dir / "dataset_statistics.csv", data["stats_rows"])
        write_json_and_key_value_csv(run_dir / "dataset_summary.json", run_dir / "dataset_summary.csv", data["stats_summary"])
        write_json_and_key_value_csv(
            run_dir / "preprocessing_config.json", run_dir / "preprocessing_config.csv", data["preprocessing_config"]
        )

        if args.rerank_from_candidates:
            if not args.candidate_cache:
                raise ValueError("--rerank-from-candidates requires --candidate-cache.")
            stage = "candidate_cache_loading"
            _, cases, base_ranked_lists = load_candidate_cache(args.candidate_cache)
        else:
            stage = "retriever_model_loading"
            model = load_sentence_transformer_model(args.retriever_model, device=device)

            stage = "retriever_encoding"
            candidate_embeddings = encode_texts(model, texts["candidate_texts"], passage_prefix, batch_size)
            query_embeddings = encode_texts(model, query_texts, query_prefix, batch_size)
            score_matrix = query_embeddings @ candidate_embeddings.T
            base_ranked_lists = top_k_ranked_lists(score_matrix, data["candidate_codes"], args.candidate_k)
            if args.candidate_cache:
                save_candidate_cache(args.candidate_cache, cases, base_ranked_lists)
            cleanup_cuda()

        if args.save_candidates_only:
            if not args.candidate_cache:
                raise ValueError("--save-candidates-only requires --candidate-cache.")
            runtime.update(
                {
                    "status": "success",
                    "stage": "candidate_cache_saved",
                    "candidate_cache": args.candidate_cache,
                    "finished_at_utc": now_utc(),
                    "elapsed_seconds": time.monotonic() - started_monotonic,
                }
            )
            write_json_and_key_value_csv(run_dir / "runtime.json", run_dir / "runtime.csv", runtime)
            return 0

        base_metrics = evaluate_ranked_lists_at_k(cases, base_ranked_lists, metric_ks)
        write_json(run_dir / "base_retriever_metrics.json", base_metrics)
        write_csv(run_dir / "base_retriever_metrics.csv", [{"setting": "base_retriever", **base_metrics}])

        stage = "llm_reranking"
        reranked_lists = []
        parsed_rows = []
        failure_rows = []
        for i, (case, ranked) in enumerate(tqdm(list(zip(cases, base_ranked_lists)), desc="llm rerank cases")):
            user_prompt = build_user_prompt(
                data["metadata_lookup"], case, ranked, args.return_k, args.candidate_description_chars
            )
            request_row = {
                "case_index": i,
                "target": case["target"],
                "prompt": user_prompt,
                "candidate_ids": [code for code, _ in ranked],
            }
            append_jsonl(run_dir / "requests.jsonl", request_row)
            if args.dry_run:
                reranked_lists.append(ranked)
                parsed_rows.append(
                    {
                        "case_index": i,
                        "target": case["target"],
                        "selected_ids": [],
                        "valid_selected_ids": [],
                        "invalid_selected_ids": [],
                        "dry_run": True,
                    }
                )
                continue
            try:
                content, raw_response = call_openai_compatible_chat(args, user_prompt)
                selected_ids = parse_selected_ids(content)
                valid_ids, invalid_ids = unique_valid_ids(selected_ids, [code for code, _ in ranked])
                reranked = rerank_with_llm(ranked, valid_ids[: args.return_k], args.fusion_mode, args.rank_bonus)
                reranked_lists.append(reranked)
                response_row = {
                    "case_index": i,
                    "target": case["target"],
                    "raw_content": content,
                    "raw_response": raw_response,
                }
                parsed_row = {
                    "case_index": i,
                    "target": case["target"],
                    "selected_ids": selected_ids,
                    "valid_selected_ids": valid_ids,
                    "invalid_selected_ids": invalid_ids,
                    "n_valid": len(valid_ids),
                    "n_invalid": len(invalid_ids),
                    "target_in_llm_list": case["target"] in valid_ids,
                }
                append_jsonl(run_dir / "responses.jsonl", response_row)
                append_jsonl(run_dir / "parsed_results.jsonl", parsed_row)
                parsed_rows.append(parsed_row)
            except Exception as exc:
                reranked_lists.append(ranked)
                failure = {
                    "case_index": i,
                    "target": case["target"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                append_jsonl(run_dir / "failures.jsonl", failure)
                failure_rows.append(failure)

        stage = "metrics"
        llm_metrics = evaluate_ranked_lists_at_k(cases, reranked_lists, metric_ks)
        summary = {
            "base_retriever": base_metrics,
            "llm_reranked": llm_metrics,
            "n_cases": len(cases),
            "n_llm_failures": len(failure_rows),
            "avg_valid_ids": float(np.mean([row.get("n_valid", 0) for row in parsed_rows])) if parsed_rows else 0.0,
            "fusion_mode": args.fusion_mode,
            "rank_bonus": args.rank_bonus,
        }
        metric_rows = [
            {"setting": "base_retriever", **base_metrics},
            {"setting": "llm_reranked", **llm_metrics},
        ]
        write_json(run_dir / "metrics.json", summary)
        write_csv(run_dir / "metrics.csv", metric_rows)
        write_json(run_dir / "parsed_results.json", parsed_rows)
        write_csv(run_dir / "parsed_results.csv", parsed_rows)
        write_csv(run_dir / "failures.csv", failure_rows)

        if not args.dry_run and len(cases) > 0 and len(failure_rows) == len(cases):
            raise RuntimeError(
                "All LLM reranking calls failed. Check the LLM endpoint and failures.jsonl; "
                "metrics were written with retriever fallback only."
            )

        stage = "finalization"
        runtime.update(
            {
                "status": "success",
                "stage": stage,
                "finished_at_utc": now_utc(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
            }
        )
        write_json_and_key_value_csv(run_dir / "runtime.json", run_dir / "runtime.csv", runtime)
        cleanup_cuda()
        return 0

    except Exception as exc:
        payload = failure_payload(exc, stage, started_at)
        payload["elapsed_seconds"] = time.monotonic() - started_monotonic
        payload["run_config"] = run_config
        write_json(run_dir / "failure.json", payload)
        write_csv(run_dir / "failures.csv", [payload])
        runtime.update(payload)
        write_json_and_key_value_csv(run_dir / "runtime.json", run_dir / "runtime.csv", runtime)
        cleanup_cuda()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
