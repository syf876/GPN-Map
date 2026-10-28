import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from backend.types.schema import VideoCandidate, VideoSearchRequest, VideoSearchResponse

from . import bilibili_provider, douyin_provider, xiaohongshu_provider
try:
    from .bilibili_rules import is_playable_bilibili_candidate
except ModuleNotFoundError:
    def is_playable_bilibili_candidate(candidate: VideoCandidate) -> bool:
        return (
            candidate.platform == "bilibili"
            and bool(str(candidate.bvid or "").strip())
            and bool(str(candidate.embed_url or "").strip())
        )
from .provider_runtime import ProviderRunMode, normalize_provider_run_mode
from .ranker import RankedCandidates, priority_order, priority_order_label, rank_candidates_with_diagnostics


Provider = Callable[[VideoSearchRequest, str, Optional[str]], List[VideoCandidate]]


@dataclass(frozen=True)
class ProviderSpec:
    platform: str
    search: Provider


@dataclass(frozen=True)
class ProviderRunTrace:
    platform: str
    requested_mode: str
    result_count: int
    fallback_triggered: bool
    fallback_target: str
    fallback_reason: str
    priority_hit: bool


@dataclass(frozen=True)
class AggregationTrace:
    query: str
    provider_order: List[str]
    priority_order: List[str]
    provider_runs: List[ProviderRunTrace]
    candidates: List[VideoCandidate]
    best: Optional[VideoCandidate]
    best_reason: str
    priority_hit: str
    candidate_source_distribution: Dict[str, int]


PROVIDERS: List[ProviderSpec] = [
    ProviderSpec(platform="bilibili", search=bilibili_provider.search),
    ProviderSpec(platform="douyin", search=douyin_provider.search),
    ProviderSpec(platform="xiaohongshu", search=xiaohongshu_provider.search),
]

GENERIC_TERMS = {
    "河西",
    "河西走廊",
    "中国",
    "诗词",
    "意境视频",
    "自然意象",
}

POEM_CUE_TERMS = [
    "大漠",
    "孤烟",
    "落日",
    "长河",
    "黄河",
    "玉门关",
    "羌笛",
    "边塞",
    "春风",
    "杨柳",
    "月色",
    "明月",
    "雪",
    "霜",
]


def build_video_query(req: VideoSearchRequest) -> str:
    if req.query:
        return req.query
    if req.query_variants:
        return req.query_variants[0].query

    query_parts: List[str] = []
    for item in (
        *req.scene_tags,
        *req.keywords,
        *req.imagery,
        *_extract_poem_cues(req.poem),
        req.emotion,
        req.location,
        req.poem,
    ):
        text = str(item or "").strip()
        if text and text not in query_parts and text not in GENERIC_TERMS:
            query_parts.append(text)
    return " ".join(query_parts) if query_parts else "河西走廊 纪录片 风光"


def search_videos(req: VideoSearchRequest) -> VideoSearchResponse:
    trace = search_videos_with_trace(req)
    return VideoSearchResponse(
        ok=True,
        query=trace.query,
        best=trace.best,
        candidates=trace.candidates,
        best_reason=trace.best_reason,
        priority_hit=trace.priority_hit,
        candidate_source_distribution=trace.candidate_source_distribution,
    )


def search_videos_with_trace(
    req: VideoSearchRequest,
    platform_modes: Optional[Dict[str, str]] = None,
) -> AggregationTrace:
    query = build_video_query(req)
    return _search_videos_with_trace(req=req, query=query, platform_modes=platform_modes)


def search_videos_with_trace_from_provider_results(
    req: VideoSearchRequest,
    query: str,
    platform_modes: Optional[Dict[str, str]] = None,
    provider_result_overrides: Optional[Dict[str, List[VideoCandidate]]] = None,
) -> AggregationTrace:
    return _search_videos_with_trace(
        req=req,
        query=query,
        platform_modes=platform_modes,
        provider_result_overrides=provider_result_overrides,
    )


def _search_videos_with_trace(
    req: VideoSearchRequest,
    query: str,
    platform_modes: Optional[Dict[str, str]] = None,
    provider_result_overrides: Optional[Dict[str, List[VideoCandidate]]] = None,
) -> AggregationTrace:
    candidates: List[VideoCandidate] = []
    provider_runs: List[ProviderRunTrace] = []

    for provider in PROVIDERS:
        requested_mode = None
        requested_mode_label = "default"
        if provider_result_overrides and provider.platform in provider_result_overrides:
            provider_candidates = provider_result_overrides[provider.platform]
            requested_mode_label = _provider_mode_label(
                provider.platform,
                platform_modes,
                fallback="result_replay",
            )
        else:
            requested_mode = _requested_mode_for_platform(provider.platform, platform_modes)
            requested_mode_label = requested_mode or "default"
            provider_candidates = provider.search(req, query, mode=requested_mode)

        enriched_candidates = _enrich_candidates(
            req=req,
            platform=provider.platform,
            requested_mode=requested_mode_label,
            candidates=provider_candidates,
        )
        candidates.extend(enriched_candidates)
        provider_runs.append(
            ProviderRunTrace(
                platform=provider.platform,
                requested_mode=requested_mode_label,
                result_count=len(enriched_candidates),
                fallback_triggered=False,
                fallback_target="",
                fallback_reason="",
                priority_hit=False,
            )
        )

    ranked = rank_candidates_with_diagnostics(
        candidates,
        platform_hints=req.platform_hints or ["bilibili"],
        preferred_tags=_preferred_tags(req),
    )
    return AggregationTrace(
        query=query,
        provider_order=[item.platform for item in PROVIDERS],
        priority_order=priority_order(),
        provider_runs=_mark_priority_hit(provider_runs, ranked),
        candidates=ranked.candidates,
        best=ranked.best,
        best_reason=ranked.best_reason,
        priority_hit=ranked.priority_hit,
        candidate_source_distribution=ranked.candidate_source_distribution,
    )


def build_aggregation_report(
    req: VideoSearchRequest,
    platform_modes: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    trace = search_videos_with_trace(req, platform_modes=platform_modes)
    return _build_aggregation_report_from_trace(trace)


def build_aggregation_report_from_provider_results(
    req: VideoSearchRequest,
    query: str,
    platform_modes: Optional[Dict[str, str]] = None,
    provider_result_overrides: Optional[Dict[str, List[VideoCandidate]]] = None,
) -> Dict[str, object]:
    trace = search_videos_with_trace_from_provider_results(
        req=req,
        query=query,
        platform_modes=platform_modes,
        provider_result_overrides=provider_result_overrides,
    )
    return _build_aggregation_report_from_trace(trace)


def _build_aggregation_report_from_trace(trace: AggregationTrace) -> Dict[str, object]:
    provider_runs = [asdict(item) for item in trace.provider_runs]
    provider_result_counts = {item["platform"]: item["result_count"] for item in provider_runs}
    best = _candidate_to_dict(trace.best)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": trace.query,
        "provider_order": trace.provider_order,
        "priority_order": trace.priority_order,
        "priority_rule": priority_order_label(),
        "provider_modes": {item.platform: item.requested_mode for item in trace.provider_runs},
        "provider_result_counts": provider_result_counts,
        "provider_runs": provider_runs,
        "fallback_summary": {
            "triggered": False,
            "count": 0,
            "platforms": [],
            "transitions": [],
        },
        "priority_hit": trace.priority_hit,
        "best_reason": trace.best_reason,
        "best_platform": best["platform"] if best else "",
        "best": best,
        "candidate_source_distribution": trace.candidate_source_distribution,
        "candidate_count": len(trace.candidates),
        "candidates": [_candidate_to_dict(item) for item in trace.candidates],
    }


def compare_aggregation_reports(
    current: Dict[str, object],
    baseline: Dict[str, object],
) -> Dict[str, object]:
    query_change = _value_change_entry(baseline.get("query", ""), current.get("query", ""))
    provider_mode_changes = _dict_change_entry(
        _string_dict(baseline.get("provider_modes")),
        _string_dict(current.get("provider_modes")),
    )
    priority_hit_change = _value_change_entry(
        baseline.get("priority_hit", ""),
        current.get("priority_hit", ""),
    )
    best_platform_change = _value_change_entry(
        baseline.get("best_platform", ""),
        current.get("best_platform", ""),
    )
    distribution_change = _distribution_change_entry(
        _int_dict(baseline.get("candidate_source_distribution")),
        _int_dict(current.get("candidate_source_distribution")),
    )
    fallback_change = _fallback_change_entry(
        _dict_object(baseline.get("fallback_summary")),
        _dict_object(current.get("fallback_summary")),
    )
    result_count_changes = _distribution_change_entry(
        _int_dict(baseline.get("provider_result_counts")),
        _int_dict(current.get("provider_result_counts")),
    )
    best_reason_change = _value_change_entry(
        baseline.get("best_reason", ""),
        current.get("best_reason", ""),
    )
    passed = not any(
        (
            query_change["changed"],
            provider_mode_changes["changed"],
            priority_hit_change["changed"],
            best_platform_change["changed"],
            distribution_change["changed"],
            fallback_change["changed"],
        )
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "query_change": query_change,
        "provider_mode_changes": provider_mode_changes,
        "priority_hit_change": priority_hit_change,
        "best_platform_change": best_platform_change,
        "best_reason_change": best_reason_change,
        "candidate_source_distribution_change": distribution_change,
        "provider_result_count_changes": result_count_changes,
        "fallback_behavior_change": fallback_change,
        "baseline": _diff_summary_snapshot(baseline),
        "current": _diff_summary_snapshot(current),
    }


def _preferred_tags(req: VideoSearchRequest) -> List[str]:
    tags: List[str] = []
    for item in (
        *req.scene_tags,
        *req.keywords,
        *req.imagery,
        *_extract_poem_cues(req.poem),
        req.emotion,
        req.location,
    ):
        text = str(item or "").strip()
        if text and text not in tags and text not in GENERIC_TERMS:
            tags.append(text)
    return tags


def _enrich_candidates(
    req: VideoSearchRequest,
    platform: str,
    requested_mode: str,
    candidates: List[VideoCandidate],
) -> List[VideoCandidate]:
    default_source_mode = _default_source_mode(platform, requested_mode)
    preferred_tags = _preferred_tags(req)
    negative_terms = [item.lower() for item in req.negative_terms if str(item).strip()]
    enriched: List[VideoCandidate] = []
    for candidate in candidates:
        existing_tags = [tag for tag in candidate.tags if str(tag).strip()]
        matched_tags = _matched_tags(candidate, preferred_tags)
        tags = _dedupe(existing_tags + matched_tags)[:8]
        source_mode = candidate.source_mode or default_source_mode
        playability_score = _resolve_playability_score(candidate)
        duplicate_key = _resolve_duplicate_key(candidate)
        source_note = candidate.source_note or requested_mode
        reason = candidate.reason or _build_candidate_reason(
            platform=platform,
            source_mode=source_mode,
            matched_tags=matched_tags,
            title=candidate.title,
        )
        score = _apply_negative_term_penalty(candidate.score, candidate, negative_terms)
        enriched.append(
            _candidate_clone(
                candidate,
                score=score,
                tags=tags,
                reason=reason,
                source_mode=source_mode,
                source_note=source_note,
                media_playability_score=playability_score,
                duplicate_key=duplicate_key,
            )
        )
    return enriched


def _candidate_clone(candidate: VideoCandidate, **updates: object) -> VideoCandidate:
    if hasattr(candidate, "model_dump"):
        data = candidate.model_dump()
    else:
        data = candidate.dict()
    data.update(updates)
    return VideoCandidate(**data)


def _dedupe(values: List[str]) -> List[str]:
    result: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _matched_tags(candidate: VideoCandidate, preferred_tags: List[str]) -> List[str]:
    if not preferred_tags:
        return []
    haystacks = [
        str(candidate.title or "").lower(),
        str(candidate.author or "").lower(),
        str(candidate.reason or "").lower(),
    ]
    candidate_tags = {str(item).strip().lower() for item in candidate.tags if str(item).strip()}
    matched: List[str] = []
    for tag in preferred_tags:
        lowered = tag.lower()
        if lowered in candidate_tags or any(lowered in hay for hay in haystacks):
            matched.append(tag)
    return matched


def _build_candidate_reason(
    platform: str,
    source_mode: str,
    matched_tags: List[str],
    title: str,
) -> str:
    if source_mode == "substitute_link":
        return "B站结果仅提供跳转链接，未命中稳定可播 iframe。"
    if matched_tags:
        return f"B站结果命中标签：{' / '.join(matched_tags[:4])}"
    if title:
        return f"B站结果已纳入排序：{title}"
    return "B站结果已纳入排序。"


def _default_source_mode(platform: str, requested_mode: str) -> str:
    requested = str(requested_mode or "").strip().lower()
    env_name = f"{platform.upper()}_FIXTURE_MODE"
    global_mode = os.getenv("VIDEO_SEARCH_MODE", "mock").strip().lower()
    if requested == "mock":
        return "mock"
    if requested in {"default", ""}:
        if global_mode == "real":
            return "fixture" if os.getenv(env_name, "").strip().lower() in {"1", "true", "yes", "on"} else "real"
        return "mock"
    if os.getenv(env_name, "").strip().lower() in {"1", "true", "yes", "on"}:
        return "fixture"
    return "real"


def _extract_poem_cues(poem: str) -> List[str]:
    text = str(poem or "").strip()
    if not text:
        return []
    return [item for item in POEM_CUE_TERMS if item in text]


def _resolve_playability_score(candidate: VideoCandidate) -> float:
    if candidate.media_playability_score:
        return float(candidate.media_playability_score)
    if is_playable_bilibili_candidate(candidate):
        return 1.0
    if candidate.bvid:
        return 0.75
    if candidate.cover:
        return 0.4
    if candidate.external_url:
        return 0.25
    return 0.0


def _resolve_duplicate_key(candidate: VideoCandidate) -> str:
    if candidate.duplicate_key:
        return candidate.duplicate_key
    for value in (candidate.bvid, candidate.external_url, candidate.embed_url, candidate.url):
        if value:
            return str(value).strip().lower()
    return candidate.title.strip().lower()


def _apply_negative_term_penalty(
    score: float,
    candidate: VideoCandidate,
    negative_terms: List[str],
) -> float:
    if not negative_terms:
        return float(score)
    haystack = " ".join(
        [
            str(candidate.title or "").lower(),
            str(candidate.reason or "").lower(),
            " ".join(str(tag).lower() for tag in candidate.tags),
        ]
    )
    penalty = 0.0
    for term in negative_terms:
        if term and term in haystack:
            penalty += 0.08
    return max(0.0, min(1.0, float(score) - penalty))


def _requested_mode_for_platform(
    platform: str,
    platform_modes: Optional[Dict[str, str]],
) -> Optional[ProviderRunMode]:
    if not platform_modes or platform not in platform_modes:
        return None
    return normalize_provider_run_mode(platform_modes[platform])


def _provider_mode_label(
    platform: str,
    platform_modes: Optional[Dict[str, str]],
    fallback: str,
) -> str:
    if not platform_modes or platform not in platform_modes:
        return fallback
    return str(platform_modes[platform]).strip() or fallback


def _mark_priority_hit(
    provider_runs: List[ProviderRunTrace],
    ranked: RankedCandidates,
) -> List[ProviderRunTrace]:
    return [
        ProviderRunTrace(
            platform=item.platform,
            requested_mode=item.requested_mode,
            result_count=item.result_count,
            fallback_triggered=item.fallback_triggered,
            fallback_target=item.fallback_target,
            fallback_reason=item.fallback_reason,
            priority_hit=bool(ranked.priority_hit and item.platform == ranked.priority_hit),
        )
        for item in provider_runs
    ]


def _candidate_to_dict(candidate: Optional[VideoCandidate]) -> Optional[Dict[str, object]]:
    if candidate is None:
        return None
    return {
        "platform": candidate.platform,
        "title": candidate.title,
        "url": candidate.url,
        "cover": candidate.cover,
        "author": candidate.author,
        "score": candidate.score,
        "bvid": candidate.bvid,
        "video_url": candidate.video_url,
        "embed_url": candidate.embed_url,
        "external_url": candidate.external_url,
        "playback_type": candidate.playback_type,
        "reason": candidate.reason,
        "tags": list(candidate.tags),
        "source_mode": candidate.source_mode,
        "source_note": candidate.source_note,
        "media_playability_score": candidate.media_playability_score,
        "duplicate_key": candidate.duplicate_key,
    }


def _value_change_entry(baseline: object, current: object) -> Dict[str, object]:
    return {"changed": baseline != current, "baseline": baseline, "current": current}


def _dict_change_entry(baseline: Dict[str, str], current: Dict[str, str]) -> Dict[str, object]:
    keys = sorted(set(baseline) | set(current))
    changes = {
        key: {
            "baseline": baseline.get(key, ""),
            "current": current.get(key, ""),
            "changed": baseline.get(key, "") != current.get(key, ""),
        }
        for key in keys
        if baseline.get(key, "") != current.get(key, "")
    }
    return {"changed": bool(changes), "baseline": baseline, "current": current, "changes": changes}


def _distribution_change_entry(baseline: Dict[str, int], current: Dict[str, int]) -> Dict[str, object]:
    keys = sorted(set(baseline) | set(current))
    delta = {key: int(current.get(key, 0)) - int(baseline.get(key, 0)) for key in keys}
    changed_keys = {
        key: {
            "baseline": int(baseline.get(key, 0)),
            "current": int(current.get(key, 0)),
            "delta": delta[key],
        }
        for key in keys
        if delta[key] != 0
    }
    return {
        "changed": bool(changed_keys),
        "baseline": baseline,
        "current": current,
        "delta": delta,
        "changes": changed_keys,
    }


def _fallback_change_entry(baseline: Dict[str, object], current: Dict[str, object]) -> Dict[str, object]:
    return {
        "changed": baseline != current,
        "baseline": baseline,
        "current": current,
        "transition_changes": {},
    }


def _diff_summary_snapshot(report: Dict[str, object]) -> Dict[str, object]:
    return {
        "query": report.get("query", ""),
        "provider_modes": _string_dict(report.get("provider_modes")),
        "priority_hit": report.get("priority_hit", ""),
        "best_platform": report.get("best_platform", ""),
        "best_reason": report.get("best_reason", ""),
        "candidate_source_distribution": _int_dict(report.get("candidate_source_distribution")),
        "provider_result_counts": _int_dict(report.get("provider_result_counts")),
        "fallback_summary": _dict_object(report.get("fallback_summary")),
    }


def _string_dict(value: object) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _int_dict(value: object) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, int] = {}
    for key, item in value.items():
        try:
            result[str(key)] = int(item)
        except (TypeError, ValueError):
            result[str(key)] = 0
    return result


def _dict_object(value: object) -> Dict[str, object]:
    return value if isinstance(value, dict) else {}
