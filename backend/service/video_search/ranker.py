from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.types.schema import VideoCandidate

from .bilibili_rules import has_playable_bilibili_embed, normalize_bvid


PLATFORM_PRIORITY: Dict[str, int] = {
    "bilibili": 0,
}

PLATFORM_BASE_BOOST: Dict[str, float] = {
    "bilibili": 0.08,
}

SOURCE_MODE_BOOST: Dict[str, float] = {
    "real_http": 0.08,
    "fixture": 0.03,
    "mock": 0.0,
    "real": 0.02,
    "substitute_link": -0.10,
}

MIN_RELEVANCE_SCORE = 0.64
HEXI_TERMS = {
    "河西走廊",
    "敦煌",
    "玉门关",
    "阳关",
    "嘉峪关",
    "酒泉",
    "张掖",
    "凉州",
    "祁连山",
    "边塞",
    "大漠",
    "长河",
    "黄河",
    "落日",
    "羌笛",
    "孤烟",
    "孤城",
    "丝路",
}
LOW_QUALITY_TERMS = {
    "概念设计",
    "概念片",
    "纯图片",
    "图文",
    "壁纸",
    "ppt",
    "ai绘画",
    "配音",
    "手书",
    "混剪素材",
    "概念图",
    "素材包",
    "纯享bgm",
}


@dataclass(frozen=True)
class RankedCandidates:
    candidates: List[VideoCandidate]
    best: Optional[VideoCandidate]
    priority_hit: str
    best_reason: str
    candidate_source_distribution: Dict[str, int]


def rank_candidates(
    candidates: List[VideoCandidate],
    platform_hints: Optional[List[str]] = None,
    preferred_tags: Optional[List[str]] = None,
) -> List[VideoCandidate]:
    return rank_candidates_with_diagnostics(
        candidates,
        platform_hints=platform_hints,
        preferred_tags=preferred_tags,
    ).candidates


def rank_candidates_with_diagnostics(
    candidates: List[VideoCandidate],
    platform_hints: Optional[List[str]] = None,
    preferred_tags: Optional[List[str]] = None,
) -> RankedCandidates:
    hint_boosts = _platform_hint_boosts(platform_hints)
    normalized_tags = _normalize_tags(preferred_tags)
    duplicate_positions = _duplicate_positions(candidates)
    ranked_candidates = sorted(
        candidates,
        key=lambda item: (
            -_blended_score(item, hint_boosts, normalized_tags, duplicate_positions),
            PLATFORM_PRIORITY.get(item.platform, len(PLATFORM_PRIORITY)),
            -item.score,
        ),
    )

    candidate_source_distribution = _candidate_source_distribution(ranked_candidates)
    if not ranked_candidates:
        return RankedCandidates(
            candidates=[],
            best=None,
            priority_hit="",
            best_reason="no_candidates_after_aggregation",
            candidate_source_distribution=candidate_source_distribution,
        )

    best: Optional[VideoCandidate] = None
    priority_hit = ""
    best_reason = ""
    for candidate in ranked_candidates:
        blended_score = _blended_score(candidate, hint_boosts, normalized_tags, duplicate_positions)
        if blended_score < MIN_RELEVANCE_SCORE:
            continue
        best = candidate
        priority_hit = candidate.platform
        overlap = _tag_overlap(candidate, normalized_tags)
        best_reason = (
            f"selected {candidate.platform}; raw={candidate.score:.2f}, blended={blended_score:.2f}, "
            f"tag_overlap={overlap}, playability={candidate.media_playability_score:.2f}, "
            f"has_bvid={bool(normalize_bvid(candidate.bvid))}, "
            f"playable_embed={_has_playable_embed(candidate)}, "
            f"duplicate_penalty={_duplicate_penalty(candidate, duplicate_positions):.2f}"
        )
        break

    if best is None:
        top = ranked_candidates[0]
        best_reason = (
            f"no candidates meet minimum blended threshold ({MIN_RELEVANCE_SCORE:.2f}); "
            f"highest candidate is {top.platform}:{top.title} with blended score "
            f"{_blended_score(top, hint_boosts, normalized_tags, duplicate_positions):.2f}"
        )

    return RankedCandidates(
        candidates=ranked_candidates,
        best=best,
        priority_hit=priority_hit,
        best_reason=best_reason,
        candidate_source_distribution=candidate_source_distribution,
    )


def priority_order() -> List[str]:
    return [
        platform
        for platform, _ in sorted(
            PLATFORM_PRIORITY.items(),
            key=lambda item: item[1],
        )
    ]


def priority_order_label() -> str:
    return ">".join(priority_order())


def _candidate_source_distribution(candidates: List[VideoCandidate]) -> Dict[str, int]:
    distribution: Dict[str, int] = {platform: 0 for platform in priority_order()}
    for item in candidates:
        distribution[item.platform] = distribution.get(item.platform, 0) + 1
    return distribution


def _platform_hint_boosts(platform_hints: Optional[List[str]]) -> Dict[str, float]:
    boosts = {platform: 0.0 for platform in PLATFORM_PRIORITY}
    if not platform_hints:
        return boosts

    normalized: List[str] = []
    for item in platform_hints:
        platform = str(item).strip().lower()
        if platform in PLATFORM_PRIORITY and platform not in normalized:
            normalized.append(platform)

    hint_values = [0.08, 0.02]
    for index, platform in enumerate(normalized):
        boosts[platform] = hint_values[index] if index < len(hint_values) else 0.01
    return boosts


def _normalize_tags(value: Optional[List[str]]) -> List[str]:
    if not value:
        return []
    result: List[str] = []
    for item in value:
        tag = str(item).strip().lower()
        if tag and tag not in result:
            result.append(tag)
    return result


def _tag_overlap(candidate: VideoCandidate, preferred_tags: List[str]) -> int:
    if not preferred_tags:
        return 0
    haystacks = [str(candidate.title or "").lower(), str(candidate.reason or "").lower()]
    candidate_tags = _normalize_tags(candidate.tags)
    overlap = 0
    for tag in preferred_tags:
        if tag in candidate_tags or any(tag in hay for hay in haystacks):
            overlap += 1
    return overlap


def _tag_boost(candidate: VideoCandidate, preferred_tags: List[str]) -> float:
    if not preferred_tags:
        return 0.0
    matched_tags: List[str] = []
    haystacks = [str(candidate.title or "").lower(), str(candidate.reason or "").lower()]
    candidate_tags = _normalize_tags(candidate.tags)
    for tag in preferred_tags:
        if tag in candidate_tags or any(tag in hay for hay in haystacks):
            matched_tags.append(tag)
    if not matched_tags:
        return -0.08
    weight = sum(_tag_weight(tag) for tag in matched_tags)
    return min(0.20, weight)


def _tag_weight(tag: str) -> float:
    if tag in {"玉门关", "羌笛", "大漠", "黄河", "长河", "明月", "月色", "边塞"}:
        return 0.05
    if tag in {"春风", "杨柳", "落日", "孤烟", "敦煌", "河西走廊"}:
        return 0.04
    if len(tag) >= 3:
        return 0.03
    return 0.015


def _duplicate_positions(candidates: List[VideoCandidate]) -> Dict[str, int]:
    seen: Dict[str, int] = {}
    result: Dict[str, int] = {}
    for item in candidates:
        duplicate_key = _duplicate_key(item)
        if not duplicate_key:
            continue
        position = seen.get(duplicate_key, 0)
        result[f"{duplicate_key}|{id(item)}"] = position
        seen[duplicate_key] = position + 1
    return result


def _duplicate_key(candidate: VideoCandidate) -> str:
    if candidate.duplicate_key:
        return candidate.duplicate_key
    if candidate.bvid:
        return candidate.bvid.strip().lower()
    if candidate.external_url:
        return candidate.external_url.strip().lower()
    return candidate.title.strip().lower()


def _duplicate_penalty(candidate: VideoCandidate, duplicate_positions: Dict[str, int]) -> float:
    key = _duplicate_key(candidate)
    position = duplicate_positions.get(f"{key}|{id(candidate)}", 0)
    return 0.08 * position


def _playability_boost(candidate: VideoCandidate) -> float:
    if _has_playable_embed(candidate):
        return 0.28
    if normalize_bvid(candidate.bvid):
        return 0.14
    return min(0.10, max(0.0, float(candidate.media_playability_score)) * 0.10)


def _bilibili_primary_boost(candidate: VideoCandidate) -> float:
    if candidate.platform != "bilibili":
        return -0.50
    boost = 0.10
    if candidate.source_mode == "real_http":
        boost += 0.08
    if normalize_bvid(candidate.bvid):
        boost += 0.16
    if _has_playable_embed(candidate):
        boost += 0.24
    return boost


def _has_playable_embed(candidate: VideoCandidate) -> bool:
    return has_playable_bilibili_embed(candidate.bvid, candidate.embed_url)


def _hexi_relevance_boost(candidate: VideoCandidate, preferred_tags: List[str]) -> float:
    haystacks = [
        str(candidate.title or "").lower(),
        str(candidate.reason or "").lower(),
        " ".join(str(tag).lower() for tag in candidate.tags),
    ]
    matched = 0
    for term in HEXI_TERMS:
        lowered = term.lower()
        if any(lowered in hay for hay in haystacks):
            matched += 1
    for term in preferred_tags:
        lowered = term.lower()
        if any(lowered in hay for hay in haystacks):
            matched += 1
    return min(0.20, matched * 0.03)


def _low_quality_penalty(candidate: VideoCandidate) -> float:
    haystack = " ".join(
        [
            str(candidate.title or "").lower(),
            str(candidate.reason or "").lower(),
            " ".join(str(tag).lower() for tag in candidate.tags),
        ]
    )
    penalty = 0.0
    for term in LOW_QUALITY_TERMS:
        if term and term in haystack:
            penalty += 0.18
    if candidate.platform == "bilibili" and not normalize_bvid(candidate.bvid):
        penalty += 0.10
    if candidate.platform == "bilibili" and not _has_playable_embed(candidate):
        penalty += 0.06
    return min(0.42, penalty)


def _non_video_penalty(candidate: VideoCandidate) -> float:
    if candidate.video_url:
        return 0.0
    if _has_playable_embed(candidate):
        return 0.0
    if normalize_bvid(candidate.bvid):
        return 0.04
    if candidate.cover and candidate.external_url:
        return 0.16
    return 0.22


def _blended_score(
    candidate: VideoCandidate,
    hint_boosts: Dict[str, float],
    preferred_tags: List[str],
    duplicate_positions: Dict[str, int],
) -> float:
    return (
        float(candidate.score)
        + PLATFORM_BASE_BOOST.get(candidate.platform, 0.0)
        + hint_boosts.get(candidate.platform, 0.0)
        + SOURCE_MODE_BOOST.get(candidate.source_mode, 0.0)
        + _playability_boost(candidate)
        + _bilibili_primary_boost(candidate)
        + _hexi_relevance_boost(candidate, preferred_tags)
        + _tag_boost(candidate, preferred_tags)
        - _low_quality_penalty(candidate)
        - _non_video_penalty(candidate)
        - _duplicate_penalty(candidate, duplicate_positions)
    )
