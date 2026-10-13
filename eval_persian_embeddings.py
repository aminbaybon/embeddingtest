"""Evaluate BGE-M3 and EmbeddingGemma-2 on Persian embedding benchmarks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
VIEWER = "https://datasets-server.huggingface.co"

MODELS = {
    "bge-m3": {
        "id": "BAAI/bge-m3",
        "folder": "bge-m3",
        "kind": "bge",
        "max_seq_length": 512,
        "url": "https://huggingface.co/BAAI/bge-m3",
    },
    "embeddinggemma-2": {
        "id": "google/embeddinggemma-2",
        "folder": "embeddinggemma-2",
        "local_path": str(ROOT / "models" / "embeddinggemma-2"),
        "kind": "gemma2",
        "max_seq_length": 512,
        "url": "https://huggingface.co/google/embeddinggemma-2",
        "text_only": True,
    },
}

# Huge full corpora. Evaluated only with a reduced gold+sample corpus unless --include-large.
LARGE_RETRIEVAL_MARKERS = (
    "msmarco-fa",
    "nq-fa",
    "hotpotqa-fa",
    "fever-fa",
    "dbpedia-fa",
    "persian-web-document-retrieval",
    "climate-fever-fa",
    "cqadupstack",
    "touche2020",
    "webis-touche",
)

TASKS: list[dict[str, Any]] = [
    # STS
    {"name": "FarSick", "hf": "MCINext/farsick-sts", "type": "sts"},
    {"name": "SynPerSTS", "hf": "MCINext/synthetic-persian-sts", "type": "sts"},
    {"name": "Query2Query", "hf": "MCINext/query-to-query-sts", "type": "sts"},
    # Pair classification
    {"name": "SynPerQAPC", "hf": "MCINext/synthetic-persian-qa-pair-classification", "type": "pair"},
    {"name": "SynPerTextKeywordsPC", "hf": "MCINext/synthetic-persian-text-keyword-pair-classification", "type": "pair"},
    {"name": "SynPerChatbotRAGFAQPC", "hf": "MCINext/synthetic-persian-chatbot-rag-faq-pair-classification", "type": "pair"},
    {"name": "FarsTail", "hf": "mteb/FarsTail", "type": "pair"},
    {"name": "FarsiParaphraseDetection", "hf": "alighasemi/farsi_paraphrase_detection", "type": "pair"},
    {"name": "ParsinluEntail", "hf": "mteb/ParsinluEntail", "type": "pair"},
    {"name": "ParsinluQueryParaphPC", "hf": "mteb/ParsinluQueryParaphPC", "type": "pair"},
    # Classification
    {"name": "SIDClassification", "hf": "MCINext/sid-classification", "type": "classification"},
    {"name": "DigiMagClassification", "hf": "MCINext/DigiMag", "type": "classification"},
    {"name": "FaIntent", "hf": "MCINext/FaIntent", "type": "classification"},
    {"name": "PerSHOPClassification", "hf": "MCINext/pershop-classification", "type": "classification"},
    {"name": "StyleClassification", "hf": "MCINext/style-classification", "type": "classification"},
    {"name": "SynPerTextToneClassification", "hf": "MCINext/synthetic-persian-text-tone-classification", "type": "classification"},
    {"name": "SynPerChatbotSatisfaction", "hf": "MCINext/synthetic-persian-chatbot-satisfaction-level-classification", "type": "classification"},
    {"name": "SynPerChatbotToneUser", "hf": "MCINext/synthetic-persian-chatbot-tone-user-classification", "type": "classification"},
    {"name": "SynPerChatbotToneChatbot", "hf": "MCINext/synthetic-persian-chatbot-tone-chatbot-classification", "type": "classification"},
    {"name": "SynPerChatbotRAGToneUser", "hf": "MCINext/synthetic-persian-chatbot-rag-tone-user-classification", "type": "classification"},
    {"name": "SynPerChatbotRAGToneChatbot", "hf": "MCINext/synthetic-persian-chatbot-rag-tone-chatbot-classification", "type": "classification"},
    {"name": "SynPerChatbotConvSAHappiness", "hf": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-happiness", "type": "classification"},
    {"name": "SynPerChatbotConvSASadness", "hf": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-sadness", "type": "classification"},
    {"name": "SynPerChatbotConvSAAnger", "hf": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-anger", "type": "classification"},
    {"name": "SynPerChatbotConvSAFear", "hf": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-fear", "type": "classification"},
    {"name": "SynPerChatbotConvSALove", "hf": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-love", "type": "classification"},
    {"name": "SynPerChatbotConvSASurprise", "hf": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-surprise", "type": "classification"},
    {"name": "SynPerChatbotConvSASatisfaction", "hf": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-satisfaction", "type": "classification"},
    {"name": "SynPerChatbotConvSAFriendship", "hf": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-friendship", "type": "classification"},
    {"name": "SynPerChatbotConvSAJealousy", "hf": "MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-jealousy", "type": "classification"},
    # Clustering
    {"name": "BeytooteClustering", "hf": "MCINext/beytoote-clustering", "type": "clustering"},
    {"name": "SIDClustering", "hf": "MCINext/sid-clustering", "type": "clustering"},
    # Summary retrieval / retrieval
    {"name": "SAMSumFa", "hf": "MCINext/samsum-fa", "type": "beir", "prompt": "summary"},
    {"name": "SynPerChatbotSumSRetrieval", "hf": "MCINext/synthetic-persian-chatbot-summary-retrieval", "type": "beir", "prompt": "summary"},
    {"name": "SynPerChatbotRAGSumSRetrieval", "hf": "MCINext/synthetic-persian-chatbot-rag-summary-retrieval", "type": "beir", "prompt": "summary"},
    {"name": "SciFact-Fa", "hf": "MCINext/scifact-fa-v2", "type": "beir"},
    {"name": "ArguAna-Fa", "hf": "MCINext/arguana-fa-v2", "type": "beir"},
    {"name": "FiQA2018-Fa", "hf": "MCINext/fiqa-fa-v2", "type": "beir"},
    {"name": "Quora-Fa", "hf": "MCINext/quora-fa-v2", "type": "beir"},
    {"name": "SCIDOCS-Fa", "hf": "MCINext/scidocs-fa-v2", "type": "beir"},
    {"name": "TRECCOVID-Fa", "hf": "MCINext/trec-covid-fa-v2", "type": "beir"},
    {"name": "NFCorpus-Fa", "hf": "MCINext/nfcorpus-fa", "type": "beir"},
    {"name": "SynPerQARetrieval", "hf": "MCINext/synthetic-persian-qa-retrieval", "type": "beir"},
    {"name": "SynPerChatbotTopicsRetrieval", "hf": "MCINext/synthetic-persian-chatbot-topics-retrieval", "type": "beir"},
    {"name": "SynPerChatbotRAGTopicsRetrieval", "hf": "MCINext/synthetic-persian-chatbot-rag-topics-retrieval", "type": "beir"},
    {"name": "SynPerChatbotRAGFAQRetrieval", "hf": "MCINext/synthetic-persian-chatbot-rag-faq-retrieval", "type": "beir"},
    {"name": "PersianWebDocumentRetrieval", "hf": "MCINext/persian-web-document-retrieval", "type": "beir", "large": True},
    {"name": "MSMARCO-Fa", "hf": "MCINext/msmarco-fa", "type": "beir", "large": True},
    {"name": "NQ-Fa", "hf": "MCINext/nq-fa", "type": "beir", "large": True},
]


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve_model_source(spec: dict[str, Any]) -> str:
    folder = spec.get("folder") or spec["id"].split("/")[-1]
    candidates: list[Path] = []
    if spec.get("local_path"):
        candidates.append(Path(spec["local_path"]))
    candidates.extend(
        [
            ROOT.parent / folder,
            ROOT / "models" / folder,
            ROOT / folder,
        ]
    )
    for path in candidates:
        if (path / "config.json").exists() and (
            (path / "pytorch_model.bin").exists()
            or (path / "model.safetensors").exists()
            or (path / "model.safetensors.index.json").exists()
        ):
            return str(path.resolve())
    return spec["id"]


def select_torch_dtype(torch: Any, spec: dict[str, Any]) -> Any:
    # EmbeddingGemma-2 must not run in float16; use bfloat16 when available.
    if spec.get("kind") == "gemma2":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def auto_batch_size(spec: dict[str, Any], requested: int) -> int:
    if requested > 0:
        return requested
    import torch

    if not torch.cuda.is_available():
        return 4
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb < 5:
        return 4 if spec.get("kind") == "gemma2" else 8
    if vram_gb < 8:
        return 8
    return 32


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return x / denom


class Embedder:
    def __init__(self, spec: dict[str, Any], batch_size: int):
        import torch
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.kind = spec["kind"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = auto_batch_size(spec, batch_size)
        load_id = resolve_model_source(spec)
        kwargs: dict[str, Any] = {"trust_remote_code": True, "device": self.device}
        dtype = select_torch_dtype(torch, spec)
        model_kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        if spec.get("text_only"):
            kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
        kwargs["model_kwargs"] = model_kwargs
        log(f"loading {load_id} on {self.device} batch={self.batch_size}")
        try:
            self.model = SentenceTransformer(load_id, local_files_only=True, **kwargs)
        except Exception as local_exc:
            log(f"local load failed ({local_exc}); trying hub/cache")
            try:
                self.model = SentenceTransformer(load_id, **kwargs)
            except Exception as exc:
                if spec.get("text_only"):
                    log(f"text-only load failed ({exc}); retrying full model")
                    kwargs.pop("config_kwargs", None)
                    self.model = SentenceTransformer(load_id, **kwargs)
                else:
                    raise
        self.model.max_seq_length = int(spec.get("max_seq_length") or 512)
        self.model.eval()
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            free, total = torch.cuda.mem_get_info()
            log(
                f"ready {spec['id']} device={self.device} dtype={dtype} "
                f"vram_free={free / 1e9:.2f}/{total / 1e9:.2f} GB"
            )
        else:
            log(f"ready {spec['id']} device={self.device} dtype={dtype}")

    def _prompt(self, task: str, is_query: bool) -> dict[str, Any]:
        if self.kind != "gemma2":
            return {}
        mapping = {
            "sts": "SentenceSimilarity",
            "pair": "SentenceSimilarity",
            "classification": "Classification",
            "clustering": "Clustering",
            "retrieval": "SearchQuery" if is_query else "Document",
            "summary": "SearchQuery" if is_query else "Document",
            "smoke": "SentenceSimilarity",
        }
        name = mapping.get(task, "SentenceSimilarity")
        return {"prompt_name": name}

    def encode(self, texts: list[str], task: str, is_query: bool = True) -> np.ndarray:
        import torch

        clean = [("" if t is None else str(t)) for t in texts]
        prompt = self._prompt(task, is_query)
        while True:
            try:
                vecs = self.model.encode(
                    clean,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=len(clean) > 32,
                    **prompt,
                )
                return np.asarray(vecs, dtype=np.float32)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size <= 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                log(f"CUDA OOM; retrying with batch_size={self.batch_size}")

    def close(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def pick_split(available: list[str]) -> str:
    for name in ("test", "validation", "dev", "train"):
        if name in available:
            return name
    return available[0]


def viewer_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{VIEWER}/{path}?{urlencode(params)}"
    last_exc: Exception | None = None
    for attempt in range(8):
        try:
            req = Request(url, headers={"User-Agent": "persian-embedding-eval/1.0"})
            with urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            wait = 45 if "429" in str(exc) else min(2 ** attempt, 20)
            time.sleep(wait)
    raise RuntimeError(f"viewer failed {url}: {last_exc}") from last_exc


def list_splits(dataset_id: str) -> list[dict[str, str]]:
    try:
        data = viewer_get("splits", {"dataset": dataset_id})
        splits = data.get("splits") or []
        if splits:
            return splits
    except Exception as exc:
        log(f"split listing via viewer failed for {dataset_id}: {exc}")
    from datasets import get_dataset_config_names, get_dataset_split_names

    splits: list[dict[str, str]] = []
    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception:
        configs = ["default"]
    for config in configs:
        try:
            for split in get_dataset_split_names(dataset_id, config):
                splits.append({"config": config, "split": split})
        except Exception:
            continue
    if not splits:
        splits = [{"config": "default", "split": "test"}]
    return splits


def fetch_rows_via_datasets(dataset_id: str, config: str, split: str, max_n: int = 0) -> list[dict[str, Any]]:
    ds = _load_split_dataset(dataset_id, config, split)
    limit = len(ds) if max_n <= 0 else min(len(ds), max_n)
    return [dict(ds[i]) for i in range(limit)]


def fetch_rows_via_viewer(dataset_id: str, config: str, split: str, max_n: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page = 100
    while True:
        length = page if max_n <= 0 else min(page, max_n - len(rows))
        if length <= 0:
            break
        data = viewer_get(
            "rows",
            {
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
        )
        batch = [item.get("row") or item for item in data.get("rows") or []]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        total = data.get("num_rows_total")
        if len(batch) < length or (total and offset >= total) or (max_n > 0 and len(rows) >= max_n):
            break
        time.sleep(0.4)
    return rows


def _load_split_dataset(dataset_id: str, config: str, split: str):
    load_kwargs: dict[str, Any] = {"split": split}
    if config and config != "default":
        return load_dataset(dataset_id, config, **load_kwargs)
    try:
        return load_dataset(dataset_id, **load_kwargs)
    except Exception:
        return load_dataset(dataset_id, config or "default", **load_kwargs)


def fetch_rows(dataset_id: str, config: str, split: str, max_n: int = 0) -> list[dict[str, Any]]:
    cache_dir = RESULTS_DIR / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_id.replace('/', '_')}__{config}__{split}__{max_n}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    log(f"fetch {dataset_id} config={config} split={split} max_n={max_n}")
    unbounded = max_n <= 0
    try:
        if unbounded:
            rows = fetch_rows_via_datasets(dataset_id, config, split, max_n=max_n)
        else:
            rows = fetch_rows_via_viewer(dataset_id, config, split, max_n=max_n)
    except Exception as exc:
        fallback = "viewer" if unbounded else "datasets.load_dataset"
        log(f"primary fetch failed for {dataset_id}: {exc}; falling back to {fallback}")
        rows = (
            fetch_rows_via_viewer(dataset_id, config, split, max_n=max_n)
            if unbounded
            else fetch_rows_via_datasets(dataset_id, config, split, max_n=max_n)
        )
    cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    log(f"fetched {len(rows)} rows for {dataset_id}/{config}/{split}")
    return rows


def fetch_id_text_map(
    dataset_id: str,
    config: str,
    split: str,
    wanted: set[str],
    id_keys: tuple[str, ...],
    text_keys: tuple[str, ...],
) -> dict[str, str]:
    """Load only the rows whose ids are in `wanted`, without materializing the full split."""
    if not wanted:
        return {}
    cache_dir = RESULTS_DIR / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(",".join(sorted(wanted)).encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"{dataset_id.replace('/', '_')}__{config}__{split}__match-{digest}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    found: dict[str, str] = {}
    log(f"lookup {len(wanted)} ids in {dataset_id} {config}/{split}")

    def pick_cols(cols: list[str]) -> tuple[str, str]:
        idc = next((c for c in id_keys if c in cols), cols[0])
        tc = next((c for c in text_keys if c in cols), cols[-1])
        return idc, tc

    try:
        ds = _load_split_dataset(dataset_id, config, split)
        idc, tc = pick_cols(list(ds.column_names))
        total = len(ds)
        for i, row in enumerate(ds):
            rid = str(row[idc])
            if rid in wanted and rid not in found:
                found[rid] = str(row[tc] or "")
                if len(found) >= len(wanted):
                    log(f"  found all {len(wanted)} ids at row {i + 1}/{total}")
                    break
            if i > 0 and i % 25000 == 0:
                log(f"  scanned {i}/{total}, found {len(found)}/{len(wanted)}")
    except Exception as exc:
        log(f"datasets id lookup failed for {dataset_id}: {exc}; falling back to viewer pages")
        offset = 0
        page = 100
        idc: str | None = None
        tc: str | None = None
        while len(found) < len(wanted):
            data = viewer_get(
                "rows",
                {
                    "dataset": dataset_id,
                    "config": config,
                    "split": split,
                    "offset": offset,
                    "length": page,
                },
            )
            batch = [item.get("row") or item for item in data.get("rows") or []]
            if not batch:
                break
            if idc is None:
                idc, tc = pick_cols(list(batch[0].keys()))
            for row in batch:
                rid = str(row[idc])
                if rid in wanted and rid not in found:
                    found[rid] = str(row[tc] or "")
            offset += len(batch)
            if offset % 1000 == 0:
                log(f"  viewer offset {offset}, found {len(found)}/{len(wanted)}")
            time.sleep(0.15)

    cache_path.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
    log(f"resolved {len(found)}/{len(wanted)} ids")
    return found


def load_hf(dataset_id: str, config: str | None = None, split: str | None = None, max_n: int = 0):
    splits = list_splits(dataset_id)
    if not splits:
        raise RuntimeError(f"no splits for {dataset_id}")
    if config:
        splits = [s for s in splits if s.get("config") == config] or splits
    configs = {s.get("config") for s in splits}
    chosen_config = config or ("default" if "default" in configs else splits[0]["config"])
    config_splits = [s.get("split") for s in splits if s.get("config") == chosen_config]
    chosen_split = split or pick_split(config_splits)
    return fetch_rows(dataset_id, chosen_config, chosen_split, max_n=max_n)


def col(example: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in example and example[name] is not None:
            return example[name]
    return None


def text_label_columns(columns: list[str]) -> tuple[str, str]:
    text_opts = ("text", "sentence", "content", "input", "tweet", "review", "query", "document")
    label_opts = ("label", "labels", "class", "category", "intent", "target", "tone", "sentiment")
    text_col = next((c for c in text_opts if c in columns), None)
    label_col = next((c for c in label_opts if c in columns), None)
    if text_col and label_col:
        return text_col, label_col
    leftover = [c for c in columns if c not in {"id", "_id", "__index_level_0__", "idx"}]
    if len(leftover) >= 2:
        return leftover[0], leftover[1]
    raise KeyError(f"cannot infer text/label columns from {columns}")


def pair_columns(columns: list[str]) -> tuple[str, str, str]:
    a_opts = ("sentence1", "text1", "premise", "query", "sent1", "s1", "question")
    b_opts = ("sentence2", "text2", "hypothesis", "document", "sent2", "s2", "answer", "keyword")
    y_opts = ("score", "label", "labels", "similarity", "gold_label", "target")
    a = next((c for c in a_opts if c in columns), None)
    b = next((c for c in b_opts if c in columns), None)
    y = next((c for c in y_opts if c in columns), None)
    leftover = [c for c in columns if c not in {"id", "_id", "__index_level_0__", "idx"}]
    if not a or not b:
        if len(leftover) >= 2:
            a, b = leftover[0], leftover[1]
    if not y:
        y = leftover[2] if len(leftover) >= 3 else leftover[-1]
    if not a or not b or not y:
        raise KeyError(f"cannot infer pair columns from {columns}")
    return a, b, y


def subsample_indices(n: int, max_n: int, seed: int = 13) -> np.ndarray:
    if max_n <= 0 or n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def encode_sts(embedder: Embedder, rows: list[dict[str, Any]], max_n: int) -> dict[str, Any]:
    from scipy.stats import spearmanr

    a_col, b_col, y_col = pair_columns(list(rows[0].keys()))
    idx = subsample_indices(len(rows), max_n)
    s1 = [str(rows[i][a_col]) for i in idx]
    s2 = [str(rows[i][b_col]) for i in idx]
    gold = np.array([float(rows[i][y_col]) for i in idx], dtype=np.float64)
    e1 = embedder.encode(s1, task="sts", is_query=True)
    e2 = embedder.encode(s2, task="sts", is_query=True)
    sims = np.sum(e1 * e2, axis=1)
    rho = float(spearmanr(gold, sims).correlation)
    return {"metric": "spearman", "score": rho, "n": int(len(idx)), "columns": [a_col, b_col, y_col]}


def encode_pair(embedder: Embedder, rows: list[dict[str, Any]], max_n: int) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    a_col, b_col, y_col = pair_columns(list(rows[0].keys()))
    idx = subsample_indices(len(rows), max_n)
    labels_raw = [rows[i][y_col] for i in idx]
    unique = sorted({str(v).lower() for v in labels_raw})
    positive = {"1", "true", "yes", "entailment", "entail", "duplicate", "paraphrase", "positive"}
    if len(unique) <= 2:
        y = np.array([1 if str(v).lower() in positive or str(v) in {"1", "1.0"} else int(float(v) > 0) for v in labels_raw], dtype=np.int32)
        if len(set(y.tolist())) < 2:
            mapping = {k: i for i, k in enumerate(unique)}
            y = np.array([mapping[str(v).lower()] for v in labels_raw], dtype=np.int32)
    else:
        pos_name = next((u for u in unique if u in positive), unique[-1])
        y = np.array([1 if str(v).lower() == pos_name else 0 for v in labels_raw], dtype=np.int32)
    s1 = [str(rows[i][a_col]) for i in idx]
    s2 = [str(rows[i][b_col]) for i in idx]
    e1 = embedder.encode(s1, task="pair", is_query=True)
    e2 = embedder.encode(s2, task="pair", is_query=True)
    sims = np.sum(e1 * e2, axis=1)
    ap = float(average_precision_score(y, sims)) if len(set(y.tolist())) > 1 else float("nan")
    try:
        auc = float(roc_auc_score(y, sims)) if len(set(y.tolist())) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {"metric": "average_precision", "score": ap, "auc": auc, "n": int(len(idx)), "n_pos": int(y.sum())}


def encode_classification(embedder: Embedder, rows: list[dict[str, Any]], max_n: int) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    text_col, label_col = text_label_columns(list(rows[0].keys()))
    idx = subsample_indices(len(rows), max_n)
    texts = [str(rows[i][text_col]) for i in idx]
    labels = LabelEncoder().fit_transform([str(rows[i][label_col]) for i in idx])
    if len(set(labels.tolist())) < 2:
        return {"metric": "accuracy", "score": float("nan"), "error": "single class", "n": int(len(idx))}
    embs = embedder.encode(texts, task="classification", is_query=True)
    n_splits = 5 if len(idx) >= 50 else 2
    n_splits = min(n_splits, int(np.min(np.bincount(labels))))
    n_splits = max(n_splits, 2) if len(idx) >= 20 else 0
    if n_splits < 2:
        return {"metric": "accuracy", "score": float("nan"), "error": "too few samples", "n": int(len(idx))}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=13)
    accs, f1s = [], []
    for train_i, test_i in skf.split(embs, labels):
        clf = LogisticRegression(max_iter=1000, n_jobs=1)
        clf.fit(embs[train_i], labels[train_i])
        pred = clf.predict(embs[test_i])
        accs.append(accuracy_score(labels[test_i], pred))
        f1s.append(f1_score(labels[test_i], pred, average="macro"))
    return {
        "metric": "accuracy",
        "score": float(np.mean(accs)),
        "macro_f1": float(np.mean(f1s)),
        "n": int(len(idx)),
        "n_classes": int(len(set(labels.tolist()))),
        "folds": n_splits,
    }


def encode_clustering(embedder: Embedder, rows: list[dict[str, Any]], max_n: int) -> dict[str, Any]:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import v_measure_score
    from sklearn.preprocessing import LabelEncoder

    text_col, label_col = text_label_columns(list(rows[0].keys()))
    idx = subsample_indices(len(rows), max_n)
    texts = [str(rows[i][text_col]) for i in idx]
    labels = LabelEncoder().fit_transform([str(rows[i][label_col]) for i in idx])
    n_clusters = int(len(set(labels.tolist())))
    if n_clusters < 2:
        return {"metric": "v_measure", "score": float("nan"), "error": "single cluster", "n": int(len(idx))}
    embs = embedder.encode(texts, task="clustering", is_query=True)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=13, batch_size=256, n_init=10)
    pred = km.fit_predict(embs)
    return {"metric": "v_measure", "score": float(v_measure_score(labels, pred)), "n": int(len(idx)), "n_clusters": n_clusters}


def dcg_at_k(rels: list[float], k: int) -> float:
    rels = rels[:k]
    return float(sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels)))


def ndcg_at_k(gains: list[float], k: int) -> float:
    ideal = sorted(gains, reverse=True)
    denom = dcg_at_k(ideal, k)
    if denom == 0:
        return 0.0
    return dcg_at_k(gains, k) / denom


def encode_beir(
    embedder: Embedder,
    dataset_id: str,
    max_queries: int,
    max_corpus: int,
    prompt: str,
    include_large: bool,
    large: bool,
) -> dict[str, Any]:
    splits = list_splits(dataset_id)
    configs = {s.get("config") for s in splits}
    default_split = pick_split([s.get("split") for s in splits if s.get("config") in {None, "default", "qrels"}] or ["test"])
    qrels_config = "qrels" if "qrels" in configs else "default" if "default" in configs else splits[0]["config"]
    qrels_split = fetch_rows(dataset_id, qrels_config, default_split, max_n=0)
    if not qrels_split:
        qrels_split = load_hf(dataset_id)

    qrels_cols = list(qrels_split[0].keys())
    qid_col = next((c for c in ("query-id", "query_id", "qid") if c in qrels_cols), qrels_cols[0])
    cid_col = next((c for c in ("corpus-id", "corpus_id", "doc_id", "document_id") if c in qrels_cols), qrels_cols[1])
    score_col = next((c for c in ("score", "label", "relevance") if c in qrels_cols), None)

    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in qrels_split:
        qid = str(row[qid_col])
        cid = str(row[cid_col])
        rel = float(row[score_col]) if score_col else 1.0
        if rel > 0:
            qrels[qid][cid] = rel

    query_ids = list(qrels.keys())
    query_ids = [query_ids[i] for i in subsample_indices(len(query_ids), max_queries)]
    gold_doc_ids = {cid for qid in query_ids for cid in qrels[qid]}

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}

    if "queries" in configs:
        queries = fetch_id_text_map(
            dataset_id,
            "queries",
            "queries",
            wanted=set(query_ids),
            id_keys=("_id", "id", "query-id", "query_id"),
            text_keys=("text", "query", "question"),
        )
    else:
        qtext_col = next((c for c in ("query", "question", "text", "query_text") if c in qrels_cols), None)
        if qtext_col:
            for row in qrels_split:
                queries[str(row[qid_col])] = str(row[qtext_col])

    if "corpus" in configs:
        corpus_cap = 0 if include_large and not large else max_corpus + len(gold_doc_ids)
        crows = fetch_rows(dataset_id, "corpus", "corpus", max_n=corpus_cap)
        ccols = list(crows[0].keys())
        idc = next((c for c in ("_id", "id", "corpus-id", "corpus_id", "doc_id") if c in ccols), ccols[0])
        title_c = next((c for c in ("title",) if c in ccols), None)
        text_c = next((c for c in ("text", "document", "content", "passage") if c in ccols), None)
        if text_c is None:
            text_c = [c for c in ccols if c != idc][-1]
        extra = []
        for row in crows:
            cid = str(row[idc])
            title = str(row[title_c]) if title_c else ""
            body = str(row[text_c])
            text = (title + " " + body).strip() if title else body
            if cid in gold_doc_ids or include_large:
                corpus[cid] = text
            elif len(extra) < max(0, max_corpus - len(gold_doc_ids)):
                extra.append((cid, text))
        for cid, text in extra[: max(0, max_corpus - len(corpus))]:
            corpus[cid] = text
    else:
        dtext_col = next((c for c in ("document", "passage", "corpus", "text", "summary") if c in qrels_cols and c != qid_col), None)
        if dtext_col:
            for row in qrels_split:
                cid = str(row[cid_col])
                corpus[cid] = str(row[dtext_col])
                queries.setdefault(str(row[qid_col]), str(row.get("query") or row.get("question") or row.get(qid_col)))

    query_ids = [qid for qid in query_ids if qid in queries]
    if not query_ids or not corpus:
        raise RuntimeError(f"empty retrieval data for {dataset_id}: queries={len(queries)} corpus={len(corpus)}")

    q_texts = [queries[qid] for qid in query_ids]
    doc_ids = list(corpus.keys())
    d_texts = [corpus[did] for did in doc_ids]
    q_emb = embedder.encode(q_texts, task=prompt, is_query=True)
    d_emb = embedder.encode(d_texts, task=prompt, is_query=False)
    scores = q_emb @ d_emb.T
    ks = (1, 5, 10)
    metrics = {f"ndcg@{k}": [] for k in ks}
    metrics.update({f"recall@{k}": [] for k in ks})
    doc_index = {did: i for i, did in enumerate(doc_ids)}
    for qi, qid in enumerate(query_ids):
        order = np.argsort(-scores[qi])
        ranked = [doc_ids[j] for j in order]
        gold = qrels[qid]
        for k in ks:
            gains = [gold.get(did, 0.0) for did in ranked[:k]]
            metrics[f"ndcg@{k}"].append(ndcg_at_k(gains, k))
            hit = sum(1 for did in ranked[:k] if did in gold)
            metrics[f"recall@{k}"].append(hit / max(len(gold), 1))
    reduced = (not include_large) and large
    return {
        "metric": "ndcg@10",
        "score": float(np.mean(metrics["ndcg@10"])),
        "ndcg@1": float(np.mean(metrics["ndcg@1"])),
        "ndcg@5": float(np.mean(metrics["ndcg@5"])),
        "recall@10": float(np.mean(metrics["recall@10"])),
        "n_queries": int(len(query_ids)),
        "n_corpus": int(len(doc_ids)),
        "reduced_corpus": reduced,
    }


def rows_from_dataset(ds) -> list[dict[str, Any]]:
    if isinstance(ds, list):
        return ds
    if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
        split = pick_split(list(ds.keys()))
        ds = ds[split]
    return [dict(row) for row in ds]


def run_task(embedder: Embedder, task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    ttype = task["type"]
    result: dict[str, Any] = {"name": task["name"], "hf": task["hf"], "type": ttype}
    try:
        if ttype == "sts":
            ds = load_hf(task["hf"], max_n=args.max_sts)
            metrics = encode_sts(embedder, rows_from_dataset(ds), args.max_sts)
        elif ttype == "pair":
            ds = load_hf(task["hf"], max_n=args.max_pair)
            metrics = encode_pair(embedder, rows_from_dataset(ds), args.max_pair)
        elif ttype == "classification":
            ds = load_hf(task["hf"], max_n=args.max_clf)
            metrics = encode_classification(embedder, rows_from_dataset(ds), args.max_clf)
        elif ttype == "clustering":
            ds = load_hf(task["hf"], max_n=args.max_cluster)
            metrics = encode_clustering(embedder, rows_from_dataset(ds), args.max_cluster)
        elif ttype == "beir":
            large = bool(task.get("large")) or any(m in task["hf"].lower() for m in LARGE_RETRIEVAL_MARKERS)
            if large and not args.include_large:
                return {**result, "skipped": True, "reason": "large retrieval corpus"}
            metrics = encode_beir(
                embedder,
                task["hf"],
                max_queries=args.max_queries,
                max_corpus=args.max_corpus,
                prompt=task.get("prompt") or "retrieval",
                include_large=args.include_large,
                large=large,
            )
        else:
            raise ValueError(ttype)
        result.update(metrics)
        result["seconds"] = round(time.time() - started, 1)
        result["ok"] = True
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-2000:]
        result["seconds"] = round(time.time() - started, 1)
        log(f"FAILED {task['name']}: {result['error']}")
    return result


def smoke_test(embedder: Embedder) -> dict[str, Any]:
    queries = [
        "پایتخت ایران کجاست؟",
        "بهترین روش یادگیری زبان فارسی چیست؟",
        "قیمت گوشی سامسونگ چقدر است؟",
    ]
    docs = [
        "تهران پایتخت ایران است و پرجمعیت‌ترین شهر کشور محسوب می‌شود.",
        "برای یادگیری زبان فارسی باید هر روز واژه جدید بخوانید و با فارسی‌زبان‌ها گفتگو کنید.",
        "امروز هوا آفتابی است و تیم فوتبال در ورزشگاه بازی می‌کند.",
    ]
    q = embedder.encode(queries, task="retrieval", is_query=True)
    d = embedder.encode(docs, task="retrieval", is_query=False)
    sims = q @ d.T
    correct = int(np.argmax(sims[0]) == 0) + int(np.argmax(sims[1]) == 1)
    a = embedder.encode(
        ["هوا خیلی گرم است.", "من به مدرسه می‌روم.", "قیمت دلار بالا رفت."],
        task="sts",
        is_query=True,
    )
    b = embedder.encode(
        ["دمای هوا بسیار زیاد است.", "گربه روی فرش خوابیده است.", "ارزش دلار افزایش یافت."],
        task="sts",
        is_query=True,
    )
    sts_sims = np.sum(a * b, axis=1).tolist()
    return {
        "retrieval_hits_2": correct,
        "similarity_paraphrase": float(sts_sims[0]),
        "similarity_unrelated": float(sts_sims[1]),
        "similarity_related_news": float(sts_sims[2]),
        "matrix": sims.tolist(),
    }


def summarize(all_results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": {}}
    for model_key, payload in all_results["models"].items():
        by_type: dict[str, list[float]] = defaultdict(list)
        wins = 0
        for task in payload.get("tasks", []):
            if task.get("ok") and isinstance(task.get("score"), (int, float)) and np.isfinite(task["score"]):
                by_type[task["type"]].append(float(task["score"]))
        summary["models"][model_key] = {
            "n_ok": sum(1 for t in payload.get("tasks", []) if t.get("ok")),
            "n_fail": sum(1 for t in payload.get("tasks", []) if not t.get("ok") and not t.get("skipped")),
            "n_skip": sum(1 for t in payload.get("tasks", []) if t.get("skipped")),
            "task_type_means": {k: float(np.mean(v)) for k, v in by_type.items()},
        }
    names = [t["name"] for t in TASKS]
    head_to_head = []
    model_keys = list(all_results["models"].keys())
    if len(model_keys) == 2:
        a, b = model_keys
        map_a = {t["name"]: t for t in all_results["models"][a].get("tasks", [])}
        map_b = {t["name"]: t for t in all_results["models"][b].get("tasks", [])}
        for name in names:
            ta, tb = map_a.get(name, {}), map_b.get(name, {})
            if ta.get("ok") and tb.get("ok") and np.isfinite(ta.get("score", np.nan)) and np.isfinite(tb.get("score", np.nan)):
                winner = a if ta["score"] > tb["score"] else b if tb["score"] > ta["score"] else "tie"
                head_to_head.append(
                    {"task": name, "type": ta.get("type"), a: ta["score"], b: tb["score"], "winner": winner}
                )
        summary["head_to_head"] = head_to_head
        summary["wins"] = {
            a: sum(1 for row in head_to_head if row["winner"] == a),
            b: sum(1 for row in head_to_head if row["winner"] == b),
            "tie": sum(1 for row in head_to_head if row["winner"] == "tie"),
        }
    return summary


MTEB_SKIP_TASKS = {
    "PersianWebDocumentRetrieval",
    "WikipediaRetrievalMultilingual",
    "MIRACLRetrieval",
    "MSMARCO-Fa",
    "NQ-Fa",
    "HotpotQA-Fa",
    "DBPedia-Fa",
    "ClimateFEVER-Fa",
    "CQADupstackRetrieval-Fa",
    "Touche2020-Fa",
    "NeuCLIR2022Retrieval",
    "NeuCLIR2023Retrieval",
}
MTEB_SKIP_TYPES = {
    "Any2AnyRetrieval",
    "Any2AnyMultilingualRetrieval",
    "VideoClassification",
    "DocumentUnderstanding",
}


def run_mteb_official(model_id: str, batch_size: int, out_dir: Path) -> dict[str, Any]:
    import mteb

    names = ["MTEB(fas, v2)", "MTEB(fas, v1)", "FaMTEB(fas, beta)"]
    benchmark = None
    used = None
    for name in names:
        try:
            benchmark = mteb.get_benchmark(name)
            used = name
            break
        except Exception:
            continue
    if benchmark is None:
        tasks = list(mteb.get_tasks(languages=["fas"]))
        used = "get_tasks(languages=['fas'])"
    else:
        tasks = list(benchmark.tasks)
    selected = []
    for task in tasks:
        name = task.metadata.name
        ttype = task.metadata.type
        if name in MTEB_SKIP_TASKS or ttype in MTEB_SKIP_TYPES:
            log(f"skip {name} ({ttype})")
            continue
        selected.append(task)
    log(f"official MTEB source: {used}; {len(selected)} tasks")
    model = mteb.get_model(model_id)
    results = mteb.evaluate(
        model,
        tasks=selected,
        encode_kwargs={"batch_size": batch_size or 8, "show_progress_bar": True},
        overwrite_strategy="only-missing",
        output_folder=str(out_dir / "mteb_raw"),
    )
    parsed = []
    for item in results:
        try:
            parsed.append(
                {
                    "task": getattr(item, "task_name", None) or str(item),
                    "scores": getattr(item, "scores", None) or getattr(item, "to_dict", lambda: str(item))(),
                }
            )
        except Exception:
            parsed.append({"raw": str(item)})
    return {"source": used, "results": parsed}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Persian embedding benchmarks for BGE-M3 and EmbeddingGemma-2")
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    p.add_argument("--batch-size", type=int, default=0, help="0 = auto from GPU VRAM")
    p.add_argument("--max-sts", type=int, default=8000)
    p.add_argument("--max-pair", type=int, default=8000)
    p.add_argument("--max-clf", type=int, default=4000)
    p.add_argument("--max-cluster", type=int, default=4000)
    p.add_argument("--max-queries", type=int, default=400)
    p.add_argument("--max-corpus", type=int, default=12000)
    p.add_argument("--include-large", action="store_true")
    p.add_argument("--reduced-large", action="store_true", default=True)
    p.add_argument("--no-reduced-large", action="store_false", dest="reduced_large")
    p.add_argument("--only-types", nargs="+", default=None)
    p.add_argument("--only-tasks", nargs="+", default=None)
    p.add_argument("--skip-mteb", action="store_true")
    p.add_argument("--mteb-only", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--summarize", action="store_true", help="print the saved comparison table and exit")
    return p.parse_args()


def print_comparison() -> None:
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run the eval first")
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary") or data
    models = summary.get("models") or {}
    print("category means")
    print(f"{'type':<16} {'bge-m3':>10} {'embeddinggemma-2':>18}")
    types = sorted({k for m in models.values() for k in (m.get("task_type_means") or {})})
    for ttype in types:
        b = (models.get("bge-m3") or {}).get("task_type_means", {}).get(ttype)
        g = (models.get("embeddinggemma-2") or {}).get("task_type_means", {}).get(ttype)
        bs = f"{b:.4f}" if isinstance(b, float) else "-"
        gs = f"{g:.4f}" if isinstance(g, float) else "-"
        print(f"{ttype:<16} {bs:>10} {gs:>18}")
    wins = summary.get("wins") or {}
    if wins:
        print("wins", json.dumps(wins))
    rows = summary.get("head_to_head") or []
    if not rows:
        return
    print()
    print(f"{'task':<40} {'type':<14} {'bge-m3':>10} {'gemma-2':>10} {'winner':>18}")
    for row in rows:
        print(
            f"{row.get('task', ''):<40} {str(row.get('type') or ''):<14} "
            f"{row.get('bge-m3', 0):10.4f} {row.get('embeddinggemma-2', 0):10.4f} "
            f"{str(row.get('winner') or '-'):>18}"
        )


def load_existing_payload(model_key: str) -> dict[str, Any]:
    path = RESULTS_DIR / f"{model_key}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.summarize:
        print_comparison()
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = []
    for task in TASKS:
        if args.only_types and task["type"] not in args.only_types:
            continue
        if args.only_tasks and task["name"] not in args.only_tasks:
            continue
        selected.append(task)
    log(f"planned tasks: {len(selected)}")

    import torch

    hardware = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    if hardware["device"] == "cuda":
        hardware["gpu"] = torch.cuda.get_device_name(0)
        hardware["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    all_results: dict[str, Any] = {
        "models": {},
        "hardware": hardware,
        "notes": {
            "benchmark": "FaMTEB / MTEB(fas) + MCINext Persian embedding datasets",
            "paper": "https://arxiv.org/abs/2502.11571",
            "bge": "https://huggingface.co/BAAI/bge-m3",
            "gemma": "https://huggingface.co/google/embeddinggemma-2",
        },
    }

    for model_key in args.models:
        spec = MODELS[model_key]
        embedder = Embedder(spec, batch_size=args.batch_size)
        payload: dict[str, Any] = {"model_id": spec["id"], "url": spec["url"], "tasks": []}
        existing = {} if args.no_resume else load_existing_payload(model_key)
        if existing:
            payload.update({k: v for k, v in existing.items() if k != "tasks"})
            payload["tasks"] = [
                t for t in (existing.get("tasks") or []) if t.get("ok") or t.get("skipped")
            ]
            payload["model_id"] = spec["id"]
            payload["url"] = spec["url"]
        done_names = {t["name"] for t in payload["tasks"] if t.get("name")}
        if "smoke" not in payload:
            payload["smoke"] = smoke_test(embedder)
            save_json(RESULTS_DIR / f"{model_key}.json", payload)
            log(f"smoke {model_key}: {json.dumps(payload['smoke'], ensure_ascii=False)}")
        else:
            log(f"resume {model_key}: smoke kept, {len(done_names)} tasks already done")
        if not args.smoke_only and not args.mteb_only:
            for i, task in enumerate(selected, 1):
                if task["name"] in done_names:
                    log(f"[{model_key} {i}/{len(selected)}] {task['name']} already done")
                    continue
                log(f"[{model_key} {i}/{len(selected)}] {task['name']}")
                row = run_task(embedder, task, args)
                payload["tasks"].append(row)
                save_json(RESULTS_DIR / f"{model_key}.json", payload)
                if row.get("ok"):
                    log(f"  {row.get('metric')}={row.get('score')} n={row.get('n', row.get('n_queries'))} t={row.get('seconds')}s")
        if args.mteb_only and not args.skip_mteb:
            try:
                payload["mteb"] = run_mteb_official(spec["id"], args.batch_size, RESULTS_DIR / model_key)
            except Exception as exc:
                payload["mteb_error"] = f"{type(exc).__name__}: {exc}"
        embedder.close()
        all_results["models"][model_key] = payload
        save_json(RESULTS_DIR / f"{model_key}.json", payload)

    for other_key in MODELS:
        if other_key in all_results["models"]:
            continue
        other = load_existing_payload(other_key)
        if other.get("tasks"):
            all_results["models"][other_key] = other

    all_results["summary"] = summarize(all_results)
    save_json(RESULTS_DIR / "summary.json", all_results)
    log(json.dumps(all_results["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
