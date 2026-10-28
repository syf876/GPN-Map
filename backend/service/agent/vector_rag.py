import csv
import hashlib
import json
import os
from threading import RLock
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import numpy as np

from backend.service.hexi_poetry import PLACE_GAZETTEER, SPACE_TYPE_RULES, THEME_RULES


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_FILE = REPO_ROOT / "hexi_poems_clean_simplified.json"
DEFAULT_CACHE_DIR = REPO_ROOT / "artifacts" / "rag_vector_index"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
INDEX_VERSION = "bge_m3_numpy_v1"
_INDEX_LOCK = RLock()
_INDEXES: Dict[tuple, Any] = {}


class VectorIndexUnavailable(RuntimeError):
    pass


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class SearchResult:
    record: Dict[str, Any]
    similarity_score: float


class BgeM3Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._backend = ""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load_model()
        values = [str(text or "") for text in texts]
        if self._backend == "FlagEmbedding":
            encoded = model.encode(values, return_dense=True)
            dense = encoded.get("dense_vecs") if isinstance(encoded, dict) else encoded
            return np.asarray(dense, dtype=np.float32)
        return np.asarray(
            model.encode(values, normalize_embeddings=False, show_progress_bar=False),
            dtype=np.float32,
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        local_files_only = os.getenv("BGE_M3_LOCAL_FILES_ONLY", "1").strip().lower() not in {"0", "false", "no"}
        batch_size = int(os.getenv("BGE_M3_BATCH_SIZE", "32") or 32)
        use_fp16 = os.getenv("BGE_M3_USE_FP16", "1").strip().lower() not in {"0", "false", "no"}
        model_path = self.model_name
        if local_files_only and not Path(model_path).is_dir():
            # FlagEmbedding versions do not consistently forward
            # local_files_only to tokenizer / auxiliary-weight downloads.
            try:
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(repo_id=model_path, local_files_only=True)
            except Exception as exc:
                raise VectorIndexUnavailable(f"BGE-M3 local snapshot unavailable: {self.model_name}") from exc
        try:
            from FlagEmbedding import BGEM3FlagModel

            try:
                self._model = BGEM3FlagModel(
                    model_path,
                    use_fp16=use_fp16,
                    local_files_only=local_files_only,
                    batch_size=batch_size,
                    devices=os.getenv("BGE_M3_DEVICE") or None,
                )
            except TypeError as exc:
                if local_files_only and not Path(model_path).exists():
                    raise exc
                self._model = BGEM3FlagModel(model_path, use_fp16=use_fp16, batch_size=batch_size)
            self._backend = "FlagEmbedding"
            return self._model
        except Exception as flag_exc:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_path, local_files_only=local_files_only)
                self._backend = "sentence_transformers"
                return self._model
            except Exception as st_exc:
                raise VectorIndexUnavailable(
                    "BGE-M3 embedder unavailable; install FlagEmbedding or sentence-transformers "
                    f"and ensure model '{self.model_name}' is available locally. "
                    f"FlagEmbedding error: {flag_exc}; sentence-transformers error: {st_exc}"
                ) from st_exc


class BgeM3PoetryVectorIndex:
    def __init__(
        self,
        *,
        data_file: Path = DEFAULT_DATA_FILE,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        model_name: Optional[str] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.data_file = Path(data_file)
        self.cache_dir = Path(cache_dir)
        self.model_name = str(model_name or os.getenv("BGE_M3_MODEL_PATH") or DEFAULT_MODEL_NAME).strip()
        self.embedder = embedder
        self.records_path = self.cache_dir / "records.json"
        self.vectors_path = self.cache_dir / "vectors.npy"
        self._records: List[Dict[str, Any]] = []
        self._vectors: Optional[np.ndarray] = None

    def search(self, query: str, *, top_k: int = 5, rebuild: bool = False) -> List[SearchResult]:
        self.load_or_build(rebuild=rebuild)
        if self._vectors is None or not self._records:
            raise VectorIndexUnavailable("vector index is empty")
        query_vector = self._encode_texts([query])
        query_vector = _normalize_vectors(query_vector)[0]
        # Avoid loading a second BLAS runtime after torch/FlagEmbedding on
        # Windows (MKL delay-load failure can terminate the whole server).
        scores = np.einsum("ij,j->i", self._vectors, query_vector, optimize=False)
        limit = max(1, min(int(top_k), len(self._records)))
        top_indices = np.argsort(-scores)[:limit]
        return [
            SearchResult(record=self._records[int(index)], similarity_score=float(scores[int(index)]))
            for index in top_indices
        ]

    def load_or_build(self, *, rebuild: bool = False) -> None:
        if not rebuild and self._load_cache():
            return
        self.rebuild()

    def rebuild(self) -> None:
        records = build_poetry_knowledge_records(self.data_file)
        if not records:
            raise VectorIndexUnavailable(f"no poetry records found: {self.data_file}")
        vectors = self._encode_texts([record["retrieval_text"] for record in records])
        vectors = _normalize_vectors(vectors)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.vectors_path, vectors)
        payload = {
            "index_version": INDEX_VERSION,
            "model_name": self.model_name,
            "source_fingerprint": _source_fingerprint(_source_files()),
            "record_count": len(records),
            "records": records,
        }
        self.records_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._records = records
        self._vectors = vectors

    def _load_cache(self) -> bool:
        if not self.records_path.exists() or not self.vectors_path.exists():
            return False
        try:
            payload = json.loads(self.records_path.read_text(encoding="utf-8"))
            if payload.get("index_version") != INDEX_VERSION:
                return False
            if payload.get("model_name") != self.model_name:
                return False
            if payload.get("source_fingerprint") != _source_fingerprint(_source_files()):
                return False
            records = payload.get("records")
            vectors = np.load(self.vectors_path)
        except Exception:
            return False
        if not isinstance(records, list) or len(records) != int(payload.get("record_count") or -1):
            return False
        if vectors.ndim != 2 or vectors.shape[0] != len(records):
            return False
        self._records = records
        self._vectors = np.asarray(vectors, dtype=np.float32)
        return True

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if self.embedder is None:
            self.embedder = BgeM3Embedder(self.model_name)
        vectors = np.asarray(self.embedder.encode(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise VectorIndexUnavailable("embedder returned invalid vector shape")
        return vectors


def search_poetry_vectors(
    query: str,
    *,
    top_k: int = 5,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    embedder: Optional[Embedder] = None,
    rebuild: bool = False,
) -> List[Dict[str, Any]]:
    # A request must not allocate another multi-gigabyte model, nor rebuild
    # the same on-disk index concurrently with another worker.
    with _INDEX_LOCK:
        if embedder is not None:
            index = BgeM3PoetryVectorIndex(cache_dir=cache_dir, embedder=embedder)
        else:
            model_name = os.getenv("BGE_M3_MODEL_PATH") or DEFAULT_MODEL_NAME
            key = (str(Path(cache_dir).resolve()), model_name)
            if key not in _INDEXES:
                _INDEXES[key] = BgeM3PoetryVectorIndex(cache_dir=cache_dir, model_name=model_name)
            index = _INDEXES[key]
        results = index.search(query, top_k=top_k, rebuild=rebuild)
    return [_result_payload(result) for result in results]


def build_poetry_knowledge_records(data_file: Path = DEFAULT_DATA_FILE) -> List[Dict[str, Any]]:
    try:
        poems = json.loads(Path(data_file).read_text(encoding="utf-8"))
    except Exception as exc:
        raise VectorIndexUnavailable(f"poetry knowledge base unavailable: {data_file}: {exc}") from exc
    if not isinstance(poems, list):
        raise VectorIndexUnavailable(f"poetry knowledge base must be a list: {data_file}")
    master_by_key = _load_master_metadata()
    label_by_key = _load_dataset_label_metadata()
    imagery_lookup = _load_label_lookup(REPO_ROOT / "labels" / "imagery_labels.json")
    emotion_lookup = _load_label_lookup(REPO_ROOT / "labels" / "emotion_labels.json")
    place_lookup = _load_label_lookup(REPO_ROOT / "labels" / "place_labels.json")

    records: List[Dict[str, Any]] = []
    for index, poem in enumerate(poems, start=1):
        title = _clean(poem.get("title"))
        author = _clean(poem.get("author"))
        dynasty = _clean(poem.get("dynasty"))
        lines = [_clean(line) for line in poem.get("text", []) if _clean(line)]
        content = " ".join(lines)
        keywords = [_clean(item) for item in poem.get("keywords", []) if _clean(item)]
        key = _record_key(title, author, dynasty, content)
        master = master_by_key.get(key, {})
        labels = label_by_key.get(key, {})
        historical_places, modern_places = _resolve_places(
            keywords=keywords,
            labeled_place=_label_text(labels.get("label_place"), place_lookup),
            master_location=_clean(master.get("location")),
            content=" ".join([title, content]),
        )
        imagery = _label_texts(labels.get("label_imagery"), imagery_lookup) or _infer_imagery(content, keywords)
        emotion = _label_text(labels.get("label_emotion"), emotion_lookup) or _clean(master.get("emotion"))
        theme = _match_rule_label(" ".join([title, content, " ".join(keywords)]), THEME_RULES)
        space_type = _match_rule_label(" ".join([title, content, " ".join(keywords)]), SPACE_TYPE_RULES)
        cultural_context = "；".join(
            item for item in [
                f"主题：{theme}" if theme else "",
                f"空间类型：{space_type}" if space_type else "",
                f"地名线索：{'、'.join(historical_places)}" if historical_places else "",
            ] if item
        )
        record_id = _clean(master.get("id")) or f"hexi_{index:04d}"
        retrieval_text = _join_retrieval_fields(
            title=title,
            author=author,
            dynasty=dynasty,
            content=content,
            historical_places=historical_places,
            modern_places=modern_places,
            imagery=imagery,
            emotion=emotion,
            theme=theme,
            cultural_context=cultural_context,
        )
        records.append(
            {
                "record_id": record_id,
                "title": title,
                "author": author,
                "dynasty": dynasty,
                "content": content,
                "historical_places": historical_places,
                "modern_places": modern_places,
                "imagery": imagery,
                "emotion": emotion,
                "theme": theme,
                "cultural_context": cultural_context,
                "source_file": _clean(poem.get("source_file")),
                "retrieval_text": retrieval_text,
            }
        )
    return records


def _result_payload(result: SearchResult) -> Dict[str, Any]:
    record = dict(result.record)
    record["similarity_score"] = result.similarity_score
    return record


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return values / norms


def _load_master_metadata() -> Dict[str, Dict[str, str]]:
    path = REPO_ROOT / "poems_master.csv"
    return _load_csv_by_record_key(path)


def _load_dataset_label_metadata() -> Dict[str, Dict[str, str]]:
    path = REPO_ROOT / "dataset" / "poems_dataset_draft.csv"
    return _load_csv_by_record_key(path)


def _load_csv_by_record_key(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    result: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        for row in csv.DictReader(file_handle):
            key = _record_key(row.get("title"), row.get("author"), row.get("dynasty"), row.get("text"))
            if key:
                result[key] = row
    return result


def _load_label_lookup(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels", []) if isinstance(payload, dict) else []
    return {
        _clean(item.get("id")): _clean(item.get("zh") or item.get("name") or item.get("id"))
        for item in labels
        if isinstance(item, dict) and _clean(item.get("id"))
    }


def _resolve_places(
    *,
    keywords: Sequence[str],
    labeled_place: str,
    master_location: str,
    content: str,
) -> tuple[List[str], List[str]]:
    historical: List[str] = []
    modern: List[str] = []
    candidates = [labeled_place, master_location, *keywords]
    for entry in PLACE_GAZETTEER:
        name = _clean(entry.get("name"))
        aliases = [_clean(alias) for alias in entry.get("aliases", ()) if _clean(alias)]
        if any(candidate and (candidate == name or candidate in aliases) for candidate in candidates) or any(
            alias and alias in content for alias in aliases
        ):
            for alias in aliases:
                if alias in candidates or alias in content:
                    _append_unique(historical, alias)
            _append_unique(modern, name)
    if labeled_place:
        _append_unique(historical, labeled_place)
    if master_location:
        _append_unique(historical, master_location)
    return historical, modern


def _infer_imagery(content: str, keywords: Sequence[str]) -> List[str]:
    text = " ".join([content, " ".join(keywords)])
    result = []
    if any(token in text for token in ("边", "塞", "关", "戍", "征", "战", "烽")):
        result.append("边塞军旅")
    if any(token in text for token in ("沙", "漠", "瀚海", "流沙", "黄尘")):
        result.append("大漠风光")
    if any(token in text for token in ("月", "雁", "云", "雪", "风", "柳", "花", "水")):
        result.append("自然意象")
    return result


def _label_text(label_id: Any, lookup: Dict[str, str]) -> str:
    values = _label_texts(label_id, lookup)
    return values[0] if values else ""


def _label_texts(label_ids: Any, lookup: Dict[str, str]) -> List[str]:
    raw = _clean(label_ids)
    if not raw:
        return []
    result = []
    for item in raw.replace(",", "|").split("|"):
        key = _clean(item)
        if key:
            _append_unique(result, lookup.get(key, key))
    return result


def _match_rule_label(text: str, rules: Sequence[tuple[str, tuple[str, ...]]]) -> str:
    for label, keywords in rules:
        if any(keyword and keyword in text for keyword in keywords):
            return label
    return ""


def _join_retrieval_fields(**fields: Any) -> str:
    lines = []
    for key, value in fields.items():
        if isinstance(value, list):
            text = "、".join(_clean(item) for item in value if _clean(item))
        else:
            text = _clean(value)
        if text:
            lines.append(f"{key}: {text}")
    return "\n".join(lines)


def _record_key(title: Any, author: Any, dynasty: Any, content: Any) -> str:
    text = "|".join([_clean(title), _clean(author), _clean(dynasty), _normalize_content(content)])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _normalize_content(value: Any) -> str:
    return " ".join(_clean(value).replace("\r", "\n").split())


def _append_unique(items: List[str], value: str) -> None:
    text = _clean(value)
    if text and text not in items:
        items.append(text)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _source_files() -> List[Path]:
    return [
        DEFAULT_DATA_FILE,
        REPO_ROOT / "poems_master.csv",
        REPO_ROOT / "dataset" / "poems_dataset_draft.csv",
        REPO_ROOT / "labels" / "place_labels.json",
        REPO_ROOT / "labels" / "imagery_labels.json",
        REPO_ROOT / "labels" / "emotion_labels.json",
    ]


def _source_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha1()
    for path in paths:
        if not path.exists():
            digest.update(f"{path}:missing".encode("utf-8"))
            continue
        stat = path.stat()
        digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()
