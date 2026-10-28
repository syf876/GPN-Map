from backend.admin_auth import require_generation_access
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from backend.types.schema import VeoGenerationRequest

from .env import WAN_API_KEY_ENV_NAMES, env_presence, env_value, first_env_value


DEFAULT_TEST_AUDIO_URL = "https://samplelib.com/mp3/sample-10s.mp3"
REPO_ROOT = Path(__file__).resolve().parents[3]
POEM_DATA_FILE = REPO_ROOT / "hexi_poems_clean_simplified.json"
TTS_CACHE_DIR = REPO_ROOT / "artifacts" / "poetry_tts"
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ResolvedPoemMatch:
    input_text: str
    poem_text: str
    matched: bool
    match_type: str
    title: str
    author: str
    dynasty: str
    source_file: str
    source_type: str = "experiment_dataset"
    included_in_experiment_dataset: bool = True


@dataclass(frozen=True)
class SynthesizedPoetryAudio:
    audio_url: str
    source: str
    model: str
    voice: str
    format: str
    expires_at: int
    cache_hit: bool
    request_id: str


@dataclass(frozen=True)
class PreparedPoetryVideoRequest:
    request: VeoGenerationRequest
    poem_match: ResolvedPoemMatch
    synthesized_audio: SynthesizedPoetryAudio | None
    audio_source: str


@dataclass(frozen=True)
class IndexedPoem:
    poem_id: int
    title: str
    author: str
    dynasty: str
    poem_text: str
    source_file: str
    lines: Tuple[str, ...]
    normalized_title: str
    normalized_author: str
    normalized_dynasty: str
    normalized_full_text: str
    normalized_lines: Tuple[str, ...]


@dataclass(frozen=True)
class PoemSearchIndex:
    poems: Tuple[IndexedPoem, ...]
    title_exact_map: Dict[str, Tuple[int, ...]]
    author_exact_map: Dict[str, Tuple[int, ...]]
    dynasty_exact_map: Dict[str, Tuple[int, ...]]
    full_exact_map: Dict[str, Tuple[int, ...]]
    line_exact_map: Dict[str, Tuple[Tuple[int, int], ...]]


def prepare_poetry_video_request(req: VeoGenerationRequest) -> PreparedPoetryVideoRequest:
    input_text = _clean_text(req.text or req.narration)
    poem_match = resolve_full_poem(input_text)
    narration_text = poem_match.poem_text or input_text
    original_audio_url = _clean_text(req.audio_url)
    should_generate_audio = not original_audio_url or original_audio_url == DEFAULT_TEST_AUDIO_URL

    synthesized_audio = None
    audio_url = original_audio_url
    audio_source = "request"
    if should_generate_audio:
        synthesized_audio = synthesize_poetry_audio(narration_text)
        audio_url = synthesized_audio.audio_url
        audio_source = synthesized_audio.source

    prepared_request = req.copy(
        update={
            "audio_url": audio_url,
            "narration": narration_text,
        }
    )
    summary = {
        "input_text": input_text,
        "matched": poem_match.matched,
        "match_type": poem_match.match_type,
        "title": poem_match.title,
        "author": poem_match.author,
        "dynasty": poem_match.dynasty,
        "audio_source": audio_source,
        "audio_url": audio_url,
        "tts_cache_hit": synthesized_audio.cache_hit if synthesized_audio else False,
    }
    print(f"[poetry_video_prepare] {_json_log(summary)}", flush=True)
    return PreparedPoetryVideoRequest(
        request=prepared_request,
        poem_match=poem_match,
        synthesized_audio=synthesized_audio,
        audio_source=audio_source,
    )


def resolve_full_poem(input_text: str) -> ResolvedPoemMatch:
    clean_input = _clean_text(input_text)
    normalized_input = _normalize_poem_text(clean_input)
    if not normalized_input:
        return ResolvedPoemMatch(
            input_text=clean_input,
            poem_text=clean_input,
            matched=False,
            match_type="fallback_empty",
            title="",
            author="",
            dynasty="",
            source_file="",
            source_type="fallback",
            included_in_experiment_dataset=False,
        )

    poem_index = _load_poem_search_index()
    best_match = _find_exact_poem_match(poem_index, normalized_input)
    if best_match is None:
        curated_match = _find_curated_featured_case_match(clean_input, normalized_input)
        if curated_match is not None:
            return curated_match
        best_match = _find_legacy_poem_match(poem_index, normalized_input)
    if best_match is None:
        return ResolvedPoemMatch(
            input_text=clean_input,
            poem_text=clean_input,
            matched=False,
            match_type="fallback_input_line",
            title="",
            author="",
            dynasty="",
            source_file="",
            source_type="fallback",
            included_in_experiment_dataset=False,
        )

    match_type, poem = best_match
    return ResolvedPoemMatch(
        input_text=clean_input,
        poem_text=poem.poem_text,
        matched=True,
        match_type=match_type,
        title=poem.title,
        author=poem.author,
        dynasty=poem.dynasty,
        source_file=poem.source_file,
        source_type="experiment_dataset",
        included_in_experiment_dataset=True,
    )


def synthesize_poetry_audio(text: str) -> SynthesizedPoetryAudio:
    require_generation_access()
    try:
        return _synthesize_poetry_audio_via_dashscope(text)
    except Exception as exc:
        print(f"[poetry_tts_dashscope_fallback] {_json_log({'error': str(exc)})}", flush=True)
        return _synthesize_poetry_audio_via_windows(text)


def _synthesize_poetry_audio_via_dashscope(text: str) -> SynthesizedPoetryAudio:
    require_generation_access()
    api_key, api_key_source = first_env_value(WAN_API_KEY_ENV_NAMES, "")
    print(
        f"[poetry_tts_env] {_json_log({'api_key_present': bool(api_key), 'api_key_source': api_key_source, 'checked_variables': env_presence(WAN_API_KEY_ENV_NAMES)})}",
        flush=True,
    )
    if not api_key:
        raise RuntimeError(
            "missing Wanxiang API key for poetry TTS; checked DASHSCOPE_API_KEY, QWEN_VL_API_KEY, WAN_API_KEY, WAN_I2V_API_KEY and compatible aliases"
        )

    model = env_value("POETRY_TTS_MODEL", "cosyvoice-v3-flash") or "cosyvoice-v3-flash"
    voice = env_value("POETRY_TTS_VOICE", "longanyang") or "longanyang"
    audio_format = _normalize_audio_format(env_value("POETRY_TTS_FORMAT", "mp3") or "mp3")
    endpoint = env_value(
        "POETRY_TTS_ENDPOINT",
        "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer",
    )
    sample_rate = _int_env("POETRY_TTS_SAMPLE_RATE", 24000, minimum=8000)
    rate = _float_env("POETRY_TTS_RATE", 0.92, minimum=0.5)
    instruction = env_value(
        "POETRY_TTS_INSTRUCTION",
        "请用普通话沉稳朗诵这首古诗，吐字清晰，节奏舒缓，保留古典停顿感和边塞苍茫感。",
    )
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "text": text,
                "model": model,
                "voice": voice,
                "format": audio_format,
                "sample_rate": sample_rate,
                "rate": rate,
                "instruction": instruction,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_path = TTS_CACHE_DIR / f"{cache_key}.json"
    cached = _load_cached_tts(cache_path)
    if cached is not None:
        return cached

    payload = {
        "model": model,
        "input": {
            "text": text,
            "voice": voice,
            "format": audio_format,
            "sample_rate": sample_rate,
            "rate": rate,
            "language_hints": ["zh"],
            "instruction": instruction,
        },
    }
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    audio = data.get("output", {}).get("audio", {}) if isinstance(data, dict) else {}
    audio_url = _clean_text(audio.get("url"))
    if not audio_url:
        raise RuntimeError(_clean_text(data.get("message")) or "dashscope poetry TTS did not return audio url")
    synthesized = SynthesizedPoetryAudio(
        audio_url=audio_url,
        source="dashscope_tts",
        model=model,
        voice=voice,
        format=audio_format,
        expires_at=int(audio.get("expires_at") or 0),
        cache_hit=False,
        request_id=_clean_text(data.get("request_id")),
    )
    cache_payload = {
        "audio_url": synthesized.audio_url,
        "source": synthesized.source,
        "model": synthesized.model,
        "voice": synthesized.voice,
        "format": synthesized.format,
        "expires_at": synthesized.expires_at,
        "request_id": synthesized.request_id,
        "text": text,
    }
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[poetry_tts] {_json_log({'model': model, 'voice': voice, 'audio_url': audio_url, 'request_id': synthesized.request_id})}",
        flush=True,
    )
    return synthesized


def _synthesize_poetry_audio_via_windows(text: str) -> SynthesizedPoetryAudio:
    voice = env_value("POETRY_LOCAL_TTS_VOICE", "Microsoft Huihui Desktop") or "Microsoft Huihui Desktop"
    rate = _int_env("POETRY_LOCAL_TTS_RATE", 0, minimum=-10)
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "text": text,
                "voice": voice,
                "rate": rate,
                "engine": "windows_system_speech",
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    audio_path = TTS_CACHE_DIR / f"{cache_key}.wav"
    meta_path = TTS_CACHE_DIR / f"{cache_key}.windows.json"
    if audio_path.exists():
        return SynthesizedPoetryAudio(
            audio_url=str(audio_path),
            source="windows_tts",
            model="windows_system_speech",
            voice=voice,
            format="wav",
            expires_at=0,
            cache_hit=True,
            request_id=_clean_text(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else "",
        )

    text_path = TTS_CACHE_DIR / f"{cache_key}.txt"
    script_path = TTS_CACHE_DIR / f"{cache_key}.ps1"
    text_path.write_text(text, encoding="utf-8")
    script_path.write_text(
        "\n".join(
            [
                "param(",
                "  [string]$TextPath,",
                "  [string]$VoiceName,",
                "  [string]$WavPath,",
                "  [int]$Rate",
                ")",
                "Add-Type -AssemblyName System.Speech",
                "$text = [System.IO.File]::ReadAllText($TextPath, [System.Text.Encoding]::UTF8)",
                "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
                "$synth.SelectVoice($VoiceName)",
                "$synth.Rate = $Rate",
                "$synth.SetOutputToWaveFile($WavPath)",
                "$synth.Speak($text)",
                "$synth.SetOutputToNull()",
                "$synth.Dispose()",
            ]
        ),
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                str(text_path),
                voice,
                str(audio_path),
                str(rate),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        script_path.unlink(missing_ok=True)
    if not audio_path.exists() or audio_path.stat().st_size <= 0:
        raise RuntimeError("windows system speech did not produce a wav file")
    meta_path.write_text("windows_tts", encoding="utf-8")
    print(
        f"[poetry_tts_windows] {_json_log({'voice': voice, 'audio_path': str(audio_path)})}",
        flush=True,
    )
    return SynthesizedPoetryAudio(
        audio_url=str(audio_path),
        source="windows_tts",
        model="windows_system_speech",
        voice=voice,
        format="wav",
        expires_at=0,
        cache_hit=False,
        request_id="windows_tts",
    )


@lru_cache(maxsize=1)
def _load_poems() -> List[Dict[str, Any]]:
    if not POEM_DATA_FILE.exists():
        return []
    payload = json.loads(POEM_DATA_FILE.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


@lru_cache(maxsize=1)
def _load_poem_search_index() -> PoemSearchIndex:
    raw_poems = _load_poems()
    poems: List[IndexedPoem] = []
    title_exact_map: Dict[str, List[int]] = {}
    author_exact_map: Dict[str, List[int]] = {}
    dynasty_exact_map: Dict[str, List[int]] = {}
    full_exact_map: Dict[str, List[int]] = {}
    line_exact_map: Dict[str, List[Tuple[int, int]]] = {}

    for poem in raw_poems:
        lines = tuple(_extract_poem_lines(poem))
        if not lines:
            continue
        normalized_lines = tuple(_normalize_poem_text(line) for line in lines)
        if not any(normalized_lines):
            continue

        poem_id = len(poems)
        indexed_poem = IndexedPoem(
            poem_id=poem_id,
            title=_clean_text(poem.get("title")),
            author=_clean_text(poem.get("author")),
            dynasty=_clean_text(poem.get("dynasty")),
            poem_text="\n".join(lines),
            source_file=_clean_text(poem.get("source_file")),
            lines=lines,
            normalized_title=_normalize_poem_text(poem.get("title")),
            normalized_author=_normalize_poem_text(poem.get("author")),
            normalized_dynasty=_normalize_poem_text(poem.get("dynasty")),
            normalized_full_text="".join(normalized_lines),
            normalized_lines=normalized_lines,
        )
        poems.append(indexed_poem)
        _append_exact_match(title_exact_map, indexed_poem.normalized_title, poem_id)
        _append_exact_match(author_exact_map, indexed_poem.normalized_author, poem_id)
        _append_exact_match(dynasty_exact_map, indexed_poem.normalized_dynasty, poem_id)
        _append_exact_match(full_exact_map, indexed_poem.normalized_full_text, poem_id)
        for line_index, normalized_line in enumerate(indexed_poem.normalized_lines):
            if normalized_line:
                line_exact_map.setdefault(normalized_line, []).append((poem_id, line_index))

    index = PoemSearchIndex(
        poems=tuple(poems),
        title_exact_map={key: tuple(value) for key, value in title_exact_map.items()},
        author_exact_map={key: tuple(value) for key, value in author_exact_map.items()},
        dynasty_exact_map={key: tuple(value) for key, value in dynasty_exact_map.items()},
        full_exact_map={key: tuple(value) for key, value in full_exact_map.items()},
        line_exact_map={key: tuple(value) for key, value in line_exact_map.items()},
    )
    summary = {
        "poem_count": len(index.poems),
        "title_keys": len(index.title_exact_map),
        "author_keys": len(index.author_exact_map),
        "dynasty_keys": len(index.dynasty_exact_map),
        "full_keys": len(index.full_exact_map),
        "line_keys": len(index.line_exact_map),
        "source_file": POEM_DATA_FILE.name,
    }
    print(f"[poetry_poem_index] {_json_log(summary)}", flush=True)
    return index


def _load_cached_tts(cache_path: Path) -> SynthesizedPoetryAudio | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    expires_at = int(payload.get("expires_at") or 0)
    if expires_at and expires_at > int(time.time()) + 60:
        return SynthesizedPoetryAudio(
            audio_url=_clean_text(payload.get("audio_url")),
            source=_clean_text(payload.get("source")) or "dashscope_tts",
            model=_clean_text(payload.get("model")),
            voice=_clean_text(payload.get("voice")),
            format=_clean_text(payload.get("format")),
            expires_at=expires_at,
            cache_hit=True,
            request_id=_clean_text(payload.get("request_id")),
        )
    return None


def _find_exact_poem_match(index: PoemSearchIndex, normalized_input: str) -> Tuple[str, IndexedPoem] | None:
    best_rank: Tuple[int, int, int, int] | None = None
    best_match: Tuple[str, IndexedPoem] | None = None

    def consider(match_type: str, poem: IndexedPoem, priority: int, line_index: int = 0) -> None:
        nonlocal best_rank, best_match
        rank = (priority, len(poem.normalized_full_text), line_index, poem.poem_id)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_match = (match_type, poem)

    for poem_id, line_index in index.line_exact_map.get(normalized_input, ()):
        consider("exact_line", index.poems[poem_id], 0, line_index)
    for poem_id in index.full_exact_map.get(normalized_input, ()):
        consider("exact_full_poem", index.poems[poem_id], 1)
    for poem_id in index.title_exact_map.get(normalized_input, ()):
        consider("exact_title", index.poems[poem_id], 2)
    for poem_id in index.author_exact_map.get(normalized_input, ()):
        consider("exact_author", index.poems[poem_id], 3)
    for poem_id in index.dynasty_exact_map.get(normalized_input, ()):
        consider("exact_dynasty", index.poems[poem_id], 4)
    return best_match


def _find_legacy_poem_match(index: PoemSearchIndex, normalized_input: str) -> Tuple[str, IndexedPoem] | None:
    best_rank: Tuple[int, int, int, int] | None = None
    best_match: Tuple[str, IndexedPoem] | None = None

    def consider(match_type: str, poem: IndexedPoem, priority: int, line_index: int = 0) -> None:
        nonlocal best_rank, best_match
        rank = (priority, len(poem.normalized_full_text), line_index, poem.poem_id)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_match = (match_type, poem)

    best_match = _find_exact_poem_match(index, normalized_input)
    if best_match is not None:
        return best_match

    for poem in index.poems:
        for line_index, normalized_line in enumerate(poem.normalized_lines):
            if normalized_input in normalized_line:
                consider("line_contains_input", poem, 5, line_index)
                break
        if normalized_input in poem.normalized_full_text:
            consider("full_poem_contains_input", poem, 6)
        if normalized_input in poem.normalized_title:
            consider("title_contains_input", poem, 7)
        if normalized_input in poem.normalized_author:
            consider("author_contains_input", poem, 8)
        if normalized_input in poem.normalized_dynasty:
            consider("dynasty_contains_input", poem, 9)
    return best_match


def _find_curated_featured_case_match(clean_input: str, normalized_input: str) -> ResolvedPoemMatch | None:
    best_rank: Tuple[int, int] | None = None
    best_case: Dict[str, Any] | None = None
    best_match_type = ""
    for case in _load_curated_featured_cases():
        aliases = [_normalize_poem_text(value) for value in case.get("match_inputs", [])]
        if normalized_input in aliases:
            rank = (0, len(str(case.get("content") or "")))
            match_type = "curated_exact_alias"
        elif normalized_input == _normalize_poem_text(case.get("content")):
            rank = (1, len(str(case.get("content") or "")))
            match_type = "curated_exact_full_poem"
        else:
            rank = None
            match_type = ""
        if rank is not None and (best_rank is None or rank < best_rank):
            best_rank = rank
            best_case = case
            best_match_type = match_type
    if best_case is None:
        return None
    return ResolvedPoemMatch(
        input_text=clean_input,
        poem_text=_format_curated_poem_text(best_case),
        matched=True,
        match_type=best_match_type,
        title=_clean_text(best_case.get("title")),
        author=_clean_text(best_case.get("author")),
        dynasty=_clean_text(best_case.get("dynasty")),
        source_file=_clean_text(best_case.get("source_file")),
        source_type="curated_featured_case",
        included_in_experiment_dataset=False,
    )


@lru_cache(maxsize=1)
def _load_curated_featured_cases() -> Tuple[Dict[str, Any], ...]:
    cases_file = REPO_ROOT / "frontend" / "data" / "demo_video_cases.json"
    cases: List[Dict[str, Any]] = []
    try:
        payload = json.loads(cases_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                cases.append(dict(item))
    cases = [_normalize_curated_featured_case(case, cases_file) for case in cases]
    return tuple(case for case in cases if case.get("id") and case.get("content"))


def _normalize_curated_featured_case(case: Dict[str, Any], cases_file: Path) -> Dict[str, Any]:
    result = dict(case)
    if result.get("id") == "liangzhouci-yumenguan":
        content = "黄河远上白云间，一片孤城万仞山。\n羌笛何须怨杨柳，春风不度玉门关。"
        aliases = [
            "凉州词二首·其一",
            "凉州词",
            "出塞",
            "凉州词 王之涣",
            "王之涣 凉州词",
            "黄河远上白云间，一片孤城万仞山。",
            "羌笛何须怨杨柳，春风不度玉门关",
            "羌笛何须怨杨柳春风不度玉门关",
            "黄河远上白云间，一片孤城万仞山。羌笛何须怨杨柳，春风不度玉门关。",
        ]
        result.update(
            {
                "title": "凉州词二首·其一",
                "author": "王之涣",
                "dynasty": "唐",
                "content": content,
                "source_type": "curated_featured_case",
                "included_in_experiment_dataset": False,
                "source_file": cases_file.name,
            }
        )
        merged_aliases = list(result.get("match_inputs") or []) + aliases
        result["match_inputs"] = _unique_clean_texts(merged_aliases)
    return result


def _format_curated_poem_text(case: Dict[str, Any]) -> str:
    content = _clean_text(case.get("content"))
    if "\n" in content:
        return content
    return "\n".join(line for line in re.split(r"(?<=[。！？!?])", content) if _clean_text(line))


def _unique_clean_texts(values: List[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in result:
            result.append(text)
    return result


def _extract_poem_lines(poem: Dict[str, Any]) -> List[str]:
    candidates = poem.get("text")
    if not isinstance(candidates, list):
        for key in ("paragraphs", "content", "contents"):
            value = poem.get(key)
            if isinstance(value, list):
                candidates = value
                break
    if not isinstance(candidates, list):
        return []

    lines: List[str] = []
    for line in candidates:
        cleaned = _clean_text(line)
        if cleaned:
            lines.append(cleaned)
    return lines


def _append_exact_match(mapping: Dict[str, List[int]], key: str, poem_id: int) -> None:
    if not key:
        return
    mapping.setdefault(key, []).append(poem_id)


_POEM_TEXT_NORMALIZE_RE = re.compile(r"[，。！？；：、“”‘’（）()《》〈〉【】〔〕—…·,.!?;:\"'`\-\s]+")


def _normalize_poem_text(text: Any) -> str:
    cleaned = _clean_text(text).lower()
    if not cleaned:
        return ""
    return _POEM_TEXT_NORMALIZE_RE.sub("", cleaned)


def _normalize_audio_format(value: str) -> str:
    text = _clean_text(value).lower()
    return text if text in {"mp3", "wav", "pcm", "opus"} else "mp3"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = env_value(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = env_value(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _json_log(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)
