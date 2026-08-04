import gc
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from datasets import load_dataset
from tqdm.auto import tqdm


DATASET_ID = "aida-ugent/JobHop"
HF_API = f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/main"
SPLITS = ("train", "validation", "test")

REMOVE_UNKNOWN = True
COLLAPSE_CONSECUTIVE_DUPLICATES = True
RANKING_CHUNK_SIZE = 512


def parse_optional_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text == "none":
        return None
    parsed = int(text)
    if parsed < 0:
        raise ValueError("Expected a non-negative integer or 'none'.")
    return parsed


def parse_k_values(value):
    if value is None:
        return [1, 3, 5, 10, 20, 50, 100]
    if isinstance(value, (list, tuple)):
        values = [int(item) for item in value]
    else:
        values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    values = sorted(set(k for k in values if k > 0))
    if not values:
        raise ValueError("Expected at least one positive K value.")
    return values


def model_slug(model_name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows_or_df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows_or_df, pd.DataFrame):
        df = rows_or_df
    else:
        df = pd.DataFrame(rows_or_df)
    df.to_csv(path, index=False)
    return df


def flatten_for_csv(payload, prefix=""):
    rows = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                rows.extend(flatten_for_csv(value, name))
            else:
                rows.append({"key": name, "value": json.dumps(value, default=json_default) if isinstance(value, list) else value})
    else:
        rows.append({"key": prefix or "value", "value": payload})
    return rows


def write_json_and_key_value_csv(json_path, csv_path, payload):
    write_json(json_path, payload)
    write_csv(csv_path, flatten_for_csv(payload))


def configure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_hf_cache(cache_dir):
    if not cache_dir:
        return None
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(cache_path / "datasets")
    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")
    return str(cache_path)


def package_versions():
    packages = [
        "accelerate",
        "datasets",
        "numpy",
        "pandas",
        "pyarrow",
        "pylate",
        "requests",
        "sentence-transformers",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "tqdm",
    ]
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def notebook_conversion_changes(script_name):
    shared = [
        "Removed Colab-only shell installs such as !pip; dependencies are declared in requirements.txt.",
        "Removed display()/Styler notebook rendering; all tabular outputs are saved as CSV and key metadata as JSON.",
        "Added argparse for model name, max evaluation cases with 'none' support, random seed, batch size, retrieve-k, output directory, and cache directory.",
        "Runs one model per process/job instead of iterating over multiple models inside one notebook.",
        "Uses a shared Hugging Face cache directory when --cache-dir is supplied and sets non-interactive cache environment variables.",
        "Persists validation sweeps, selected alpha, final test results, runtime, GPU information, package versions, dataset statistics, preprocessing configuration, data-file discovery, and failures.",
        "Adds explicit failure handling so partial CSV/JSON artifacts are kept when a model or retrieval stage fails.",
        "Keeps JobHop v2 preprocessing: schema standardization, unknown-code removal, start-date/end-date/code stable ordering, consecutive duplicate collapse, length>=2 trajectories, train-only candidate pool, and last-step prediction.",
        "Keeps ESCO metadata loading from danieldux/ESCO and the same candidate/query text templates from the notebooks.",
        "Changes alpha protocol intentionally: alpha is selected using validation MRR only, then frozen for one final test evaluation. Test metrics are not used for alpha selection.",
        "Tie handling is explicit and validation-only: the first alpha in the notebook sweep order wins equal validation MRR ties.",
        "Adds multi-K information retrieval metrics: Accuracy@K, Precision@K, Recall@K, MRR@K, NDCG@K, and MAP@K.",
    ]
    dense = [
        "Dense script keeps the notebook dense alpha grid [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0].",
        "Dense script keeps row min-max normalization for embedding scores and row max normalization for Markov scores.",
        "Dense script preserves full candidate scoring; --retrieve-k controls ranking extraction depth for metrics and is clamped to at least 10.",
        "Dense script runs sentence_transformers.evaluation.InformationRetrievalEvaluator for pure dense retrieval statistics.",
    ]
    lateon = [
        "LateOn script keeps the notebook LateOn alpha grid [0.0, 0.25, 0.5, 0.75, 0.9, 1.0].",
        "LateOn script keeps PyLate ColBERT encoding, PLAID indexing, and ColBERT retrieval.",
        "LateOn script uses --retrieve-k for both LateOn retrieval depth and Markov candidate depth, matching the notebook's shared default of 200.",
        "LateOn script keeps per-case union of LateOn and Markov candidates with per-row min-max normalization before hybrid ranking.",
    ]
    specific = dense if script_name == "dense" else lateon
    return {"shared_changes": shared, "script_specific_changes": specific}


def gpu_info():
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
    return info


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available inside this job. Refusing to run on CPU. "
            f"torch_version={getattr(torch, '__version__', 'unknown')}; "
            f"torch_cuda_version={torch.version.cuda}; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}."
        )
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("CUDA reports available but no CUDA devices are visible.")
    torch.cuda.init()
    return "cuda"


def hf_tree(path=""):
    url = HF_API if not path else f"{HF_API}/{path}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def split_from_path(path):
    lower = path.lower()
    for split in SPLITS:
        if re.search(rf"(^|/|_|-)({split})(/|_|-|\.|$)", lower):
            return split
    if "val" in lower:
        return "validation"
    return None


def list_parquet_files():
    files = []
    stack = [""]
    while stack:
        path = stack.pop()
        try:
            entries = hf_tree(path)
        except Exception:
            continue
        for item in entries:
            item_path = item.get("path") or item.get("name")
            if not item_path:
                continue
            if item_path.endswith(".parquet"):
                files.append(item_path)
            elif item.get("type") == "directory":
                stack.append(item_path)
    return sorted(set(files))


def hf_resolve_url(path):
    return f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/{path}"


def load_split(split, split_files):
    paths = split_files.get(split, [])
    if not paths:
        raise RuntimeError(f"No parquet files found for split={split}.")
    frames = []
    for path in tqdm(paths, desc=f"loading {split}"):
        frames.append(pd.read_parquet(hf_resolve_url(path)))
    df = pd.concat(frames, ignore_index=True)
    df["split"] = split
    return df


def standardize_jobhop_schema(df):
    df = df.copy()
    if "person_id" not in df.columns:
        if "resume_id" in df.columns:
            df["person_id"] = df["resume_id"]
        else:
            raise KeyError(f"No person/resume identifier column found. Columns: {df.columns.tolist()}")

    if "university_studies" not in df.columns:
        if "university_level" in df.columns:
            df["university_studies"] = df["university_level"]
        else:
            df["university_studies"] = pd.NA

    df["matched_code"] = df["matched_code"].astype(str).str.strip()
    return df


def clean_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\n", " ").strip()


def normalize_esco_code(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().rstrip("/").split("/")[-1]


def add_meta(lookup, code, label="", description="", source=""):
    code = normalize_esco_code(code)
    if not code:
        return
    label = clean_text(label)
    description = clean_text(description)
    if code not in lookup or (not lookup[code].get("description") and description):
        lookup[code] = {
            "label": label or lookup.get(code, {}).get("label", ""),
            "description": description or lookup.get(code, {}).get("description", ""),
            "source": source,
        }


def load_danieldux_esco_metadata():
    lookup = {}
    frames = []
    parquet_urls = [
        "https://huggingface.co/datasets/danieldux/ESCO/resolve/main/data/en/train.parquet",
        "https://huggingface.co/datasets/danieldux/ESCO/resolve/main/data/en/test.parquet",
    ]
    for url in parquet_urls:
        try:
            frames.append(pd.read_parquet(url))
            print("Loaded danieldux parquet:", url, flush=True)
        except Exception as exc:
            print("Could not read danieldux parquet:", url, exc, flush=True)

    if not frames:
        ds = load_dataset("danieldux/ESCO", split="train")
        frames.append(ds.to_pandas())
        print("Loaded danieldux/ESCO through datasets", flush=True)

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ESCO_CODE"])
    for _, row in df.iterrows():
        add_meta(
            lookup,
            row.get("ESCO_CODE"),
            row.get("ESCO_OCCUPATION", ""),
            row.get("ESCO_DESCRIPTION", ""),
            "hf_danieldux_esco",
        )
    return lookup


def date_key(value):
    if pd.isna(value):
        return (9999, 9)
    text = str(value).strip()
    if not text or text in {"-", "nan", "None"}:
        return (9999, 9)
    match = re.search(r"(19|20)\d{2}", text)
    year = int(match.group(0)) if match else 9999
    quarter = 9
    qmatch = re.search(r"[Qq][-_ ]?([1-4])", text)
    if qmatch:
        quarter = int(qmatch.group(1))
    return (year, quarter)


def normalize_jobhop_code(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    if REMOVE_UNKNOWN and text.lower() == "unknown":
        return None
    return text


def compact_consecutive(seq):
    out = []
    for item in seq:
        if not out or item["code"] != out[-1]["code"]:
            out.append(item)
    return out


def make_trajectories(df):
    trajectories = {}
    for person_id, group in tqdm(df.groupby("person_id", sort=False), desc="trajectories"):
        group = group.copy()
        group["_date_key"] = group["start_date"].apply(date_key)
        group = group.sort_values(["_date_key", "end_date", "matched_code"], kind="mergesort")
        seq = []
        for _, row in group.iterrows():
            code = normalize_jobhop_code(row["matched_code"])
            if code is None:
                continue
            seq.append({"code": code, "education": row.get("university_studies")})
        if COLLAPSE_CONSECUTIVE_DUPLICATES:
            seq = compact_consecutive(seq)
        if len(seq) >= 2:
            trajectories[person_id] = seq
    return trajectories


def make_last_step_cases(trajectories_dict):
    cases = []
    for person_id, seq in trajectories_dict.items():
        cases.append(
            {
                "person_id": person_id,
                "prefix": [item["code"] for item in seq[:-1]],
                "target": seq[-1]["code"],
                "education": seq[0].get("education"),
                "prefix_length": len(seq) - 1,
            }
        )
    return cases


def dataset_statistics(raw, trajectories, candidate_codes, validation_cases, test_cases, metadata_lookup):
    rows = []
    for split, df in raw.items():
        rows.append(
            {
                "split": split,
                "rows": len(df),
                "resumes": df["person_id"].nunique(),
                "unknown_code_rows": int((df["matched_code"].str.lower() == "unknown").sum()),
                "missing_start_date": int(df["start_date"].isna().sum()) if "start_date" in df.columns else None,
                "missing_end_date": int(df["end_date"].isna().sum()) if "end_date" in df.columns else None,
                "trajectories_len_ge_2": len(trajectories.get(split, {})),
            }
        )
    metadata_coverage = sum(1 for code in candidate_codes if code in metadata_lookup)
    description_coverage = sum(1 for code in candidate_codes if metadata_lookup.get(code, {}).get("description"))
    summary = {
        "candidate_occupations": len(candidate_codes),
        "candidate_metadata_coverage": metadata_coverage,
        "candidate_description_coverage": description_coverage,
        "validation_eval_cases": len(validation_cases),
        "test_eval_cases": len(test_cases),
    }
    return rows, summary


def load_jobhop_eval_data(max_eval_cases, seed):
    parquet_files = list_parquet_files()
    split_files = {split: [] for split in SPLITS}
    for path in parquet_files:
        split = split_from_path(path)
        if split in split_files:
            split_files[split].append(path)

    raw = {split: standardize_jobhop_schema(load_split(split, split_files)) for split in SPLITS}
    metadata_lookup = load_danieldux_esco_metadata()
    trajectories = {split: make_trajectories(raw[split]) for split in SPLITS}

    validation_cases = make_last_step_cases(trajectories["validation"])
    test_cases = make_last_step_cases(trajectories["test"])
    candidate_codes = sorted({item["code"] for seq in trajectories["train"].values() for item in seq})
    candidate_set = set(candidate_codes)

    validation_cases = [case for case in validation_cases if case["target"] in candidate_set]
    test_cases = [case for case in test_cases if case["target"] in candidate_set]

    if max_eval_cases is not None:
        rng = random.Random(seed)
        validation_cases = rng.sample(validation_cases, min(max_eval_cases, len(validation_cases)))
        test_cases = rng.sample(test_cases, min(max_eval_cases, len(test_cases)))

    stats_rows, stats_summary = dataset_statistics(
        raw, trajectories, candidate_codes, validation_cases, test_cases, metadata_lookup
    )
    preprocessing_config = {
        "dataset_id": DATASET_ID,
        "splits": list(SPLITS),
        "remove_unknown": REMOVE_UNKNOWN,
        "collapse_consecutive_duplicates": COLLAPSE_CONSECUTIVE_DUPLICATES,
        "date_sort_keys": ["start_date parsed as year/quarter", "end_date", "matched_code"],
        "min_trajectory_length": 2,
        "prediction_protocol": "last-step next-occupation prediction",
        "candidate_pool": "unique ESCO codes observed in train trajectories",
        "validation_test_filter": "cases whose target is in train candidate pool",
        "max_eval_cases": max_eval_cases,
        "random_seed": seed,
    }
    return {
        "raw": raw,
        "trajectories": trajectories,
        "metadata_lookup": metadata_lookup,
        "candidate_codes": candidate_codes,
        "validation_cases": validation_cases,
        "test_cases": test_cases,
        "stats_rows": stats_rows,
        "stats_summary": stats_summary,
        "preprocessing_config": preprocessing_config,
        "parquet_files": parquet_files,
        "split_files": split_files,
    }


def label_for_code(metadata_lookup, code):
    meta = metadata_lookup.get(code, {})
    return meta.get("label") or f"ESCO code {code}"


def description_for_code(metadata_lookup, code):
    return metadata_lookup.get(code, {}).get("description", "")


def compact_desc(text, max_chars=500):
    text = clean_text(text)
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def candidate_text(metadata_lookup, code):
    label = label_for_code(metadata_lookup, code)
    desc = compact_desc(description_for_code(metadata_lookup, code))
    pieces = [f"ESCO occupation code: {code}.", f"Occupation title: {label}."]
    if desc:
        pieces.append(f"Description: {desc}.")
    return " ".join(pieces)


def query_text(metadata_lookup, case):
    labels = [label_for_code(metadata_lookup, code) for code in case["prefix"]]
    history = " -> ".join(labels)
    codes = " -> ".join(case["prefix"])
    edu = case.get("education")
    edu_text = "" if pd.isna(edu) else f" Education level: {edu}."
    return (
        f"Career history occupations: {history}. "
        f"Career history ESCO codes: {codes}."
        f"{edu_text} Predict the next ESCO occupation."
    )


def build_texts(metadata_lookup, candidate_codes, validation_cases, test_cases):
    return {
        "candidate_texts": [candidate_text(metadata_lookup, code) for code in candidate_codes],
        "validation_query_texts": [query_text(metadata_lookup, case) for case in validation_cases],
        "test_query_texts": [query_text(metadata_lookup, case) for case in test_cases],
    }


def train_markov_dense(trajectories_dict, candidate_codes):
    code_to_index = {code: i for i, code in enumerate(candidate_codes)}
    transitions = defaultdict(Counter)
    global_next = Counter()
    for seq in trajectories_dict.values():
        codes = [item["code"] for item in seq]
        for current, nxt in zip(codes[:-1], codes[1:]):
            transitions[current][nxt] += 1
            global_next[nxt] += 1

    global_total = sum(global_next.values())
    global_probs = np.zeros(len(candidate_codes), dtype="float32")
    if global_total:
        for code, count in global_next.items():
            if code in code_to_index:
                global_probs[code_to_index[code]] = count / global_total

    transition_probs = {}
    for prev, counter in transitions.items():
        total = sum(counter.values())
        indices = []
        probs = []
        for nxt, count in counter.items():
            if nxt in code_to_index:
                indices.append(code_to_index[nxt])
                probs.append(count / total)
        transition_probs[prev] = (np.array(indices, dtype=np.int32), np.array(probs, dtype="float32"))
    return transition_probs, global_probs, code_to_index


def markov_score_matrix(cases, candidate_codes, transition_probs, global_probs):
    scores = np.zeros((len(cases), len(candidate_codes)), dtype="float32")
    for i, case in enumerate(tqdm(cases, desc="markov score rows")):
        last = case["prefix"][-1]
        if last in transition_probs:
            idx, probs = transition_probs[last]
            scores[i, idx] = probs
        else:
            scores[i, :] = global_probs
    return scores


def train_markov_ranked(trajectories_dict):
    transitions = defaultdict(Counter)
    global_next = Counter()
    for seq in trajectories_dict.values():
        codes = [item["code"] for item in seq]
        for current, nxt in zip(codes[:-1], codes[1:]):
            transitions[current][nxt] += 1
            global_next[nxt] += 1

    global_total = sum(global_next.values())
    global_rank = [(code, count / global_total) for code, count in global_next.most_common()] if global_total else []
    transition_probs = {}
    for prev, counter in transitions.items():
        total = sum(counter.values())
        transition_probs[prev] = [(code, count / total) for code, count in counter.most_common()]
    return transition_probs, global_rank


def row_minmax(x):
    if x.size == 0:
        return x.astype("float32")
    mn = x.min(axis=1, keepdims=True)
    mx = x.max(axis=1, keepdims=True)
    denom = np.maximum(mx - mn, 1e-8)
    return ((x - mn) / denom).astype("float32")


def row_maxnorm(x):
    if x.size == 0:
        return x.astype("float32")
    mx = x.max(axis=1, keepdims=True)
    return (x / np.maximum(mx, 1e-8)).astype("float32")


def evaluate_score_matrix(score_matrix, cases, code_to_index, retrieve_k=10):
    return evaluate_score_matrix_at_k(score_matrix, cases, code_to_index, parse_k_values([5, 10]), retrieve_k)


def _empty_metric_result(metric_ks, n_cases=0):
    result = {"n_cases": n_cases}
    for k in metric_ks:
        result[f"Accuracy@{k}"] = 0.0
        result[f"Precision@{k}"] = 0.0
        result[f"Recall@{k}"] = 0.0
        result[f"MRR@{k}"] = 0.0
        result[f"NDCG@{k}"] = 0.0
        result[f"MAP@{k}"] = 0.0
    result["MRR"] = result.get("MRR@10", 0.0)
    result["Recall@5"] = result.get("Recall@5", 0.0)
    result["Recall@10"] = result.get("Recall@10", 0.0)
    return result


def _single_relevant_metrics_from_ranks(ranks, metric_ks):
    n_cases = len(ranks)
    if n_cases == 0:
        return _empty_metric_result(metric_ks, n_cases=0)
    result = {"n_cases": n_cases}
    for k in metric_ks:
        hits = [rank for rank in ranks if rank is not None and rank <= k]
        accuracy = len(hits) / n_cases
        result[f"Accuracy@{k}"] = float(accuracy)
        result[f"Precision@{k}"] = float(accuracy / k)
        result[f"Recall@{k}"] = float(accuracy)
        result[f"MRR@{k}"] = float(sum(1.0 / rank for rank in hits) / n_cases)
        result[f"NDCG@{k}"] = float(sum(1.0 / np.log2(rank + 1) for rank in hits) / n_cases)
        result[f"MAP@{k}"] = result[f"MRR@{k}"]
    result["MRR"] = result.get("MRR@10", result[f"MRR@{max(metric_ks)}"])
    result["Recall@5"] = result.get("Recall@5", 0.0)
    result["Recall@10"] = result.get("Recall@10", result.get(f"Recall@{max(metric_ks)}", 0.0))
    return result


def evaluate_score_matrix_at_k(score_matrix, cases, code_to_index, metric_ks, retrieve_k=10):
    metric_ks = parse_k_values(metric_ks)
    if not cases:
        return _empty_metric_result(metric_ks, n_cases=0)
    top_k = max(max(metric_ks), int(retrieve_k))
    top_k = min(top_k, score_matrix.shape[1])
    target_indices = np.array([code_to_index[case["target"]] for case in cases], dtype=np.int64)
    ranks = []
    for start in range(0, len(cases), RANKING_CHUNK_SIZE):
        end = min(start + RANKING_CHUNK_SIZE, len(cases))
        chunk = score_matrix[start:end]
        kth = max(0, top_k - 1)
        order = np.argpartition(-chunk, kth=kth, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(chunk, order, axis=1)
        sorted_local = np.argsort(-top_scores, axis=1)
        ranked = np.take_along_axis(order, sorted_local, axis=1)
        for row_idx, preds in enumerate(ranked):
            target = target_indices[start + row_idx]
            matches = np.where(preds == target)[0]
            ranks.append(int(matches[0]) + 1 if len(matches) else None)
    return _single_relevant_metrics_from_ranks(ranks, metric_ks)


def evaluate_ranked_lists(cases, ranked_lists):
    return evaluate_ranked_lists_at_k(cases, ranked_lists, parse_k_values([5, 10]))


def evaluate_ranked_lists_at_k(cases, ranked_lists, metric_ks):
    metric_ks = parse_k_values(metric_ks)
    if not cases:
        return _empty_metric_result(metric_ks, n_cases=0)
    max_k = max(metric_ks)
    ranks = []
    for case, ranked in zip(cases, ranked_lists):
        preds = [code for code, _ in ranked[:max_k]]
        target = case["target"]
        ranks.append(preds.index(target) + 1 if target in preds else None)
    return _single_relevant_metrics_from_ranks(ranks, metric_ks)


def select_alpha_by_validation_mrr(validation_rows, alphas):
    best_alpha = None
    best_mrr = -1.0
    for row in validation_rows:
        mrr = float(row["MRR"])
        if mrr > best_mrr:
            best_mrr = mrr
            best_alpha = float(row["alpha_markov"])
    if best_alpha is None:
        best_alpha = float(alphas[0])
    return best_alpha, best_mrr


def failure_payload(exc, stage, started_at):
    return {
        "status": "failed",
        "stage": stage,
        "started_at_utc": started_at,
        "failed_at_utc": now_utc(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
