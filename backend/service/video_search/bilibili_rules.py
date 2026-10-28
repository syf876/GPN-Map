import re
from typing import Any

from backend.types.schema import VideoCandidate


BVID_RE = re.compile(r"(BV[0-9A-Za-z]{10})(?![0-9A-Za-z])", re.IGNORECASE)
STRICT_BVID_RE = re.compile(r"^BV[0-9A-Za-z]{10}$", re.IGNORECASE)
EMBED_RE = re.compile(
    r"^https://player\.bilibili\.com/player\.html\?bvid=(BV[0-9A-Za-z]{10})$",
    re.IGNORECASE,
)


def normalize_bvid(value: Any) -> str:
    text = str(value or "").strip()
    if STRICT_BVID_RE.fullmatch(text):
        return text
    return ""


def extract_bvid(value: Any) -> str:
    text = str(value or "").strip()
    matched = BVID_RE.search(text)
    if not matched:
        return ""
    return matched.group(1)


def build_embed_url(bvid: str) -> str:
    normalized = normalize_bvid(bvid)
    if not normalized:
        return ""
    return f"https://player.bilibili.com/player.html?bvid={normalized}"


def has_valid_embed_url(embed_url: Any) -> bool:
    return bool(EMBED_RE.match(str(embed_url or "").strip()))


def has_playable_bilibili_embed(bvid: Any, embed_url: Any) -> bool:
    normalized_bvid = normalize_bvid(bvid)
    normalized_embed = str(embed_url or "").strip()
    if not normalized_bvid or not has_valid_embed_url(normalized_embed):
        return False
    matched = EMBED_RE.match(normalized_embed)
    if not matched:
        return False
    return matched.group(1).lower() == normalized_bvid.lower()


def is_playable_bilibili_candidate(candidate: VideoCandidate) -> bool:
    return (
        candidate.platform == "bilibili"
        and has_playable_bilibili_embed(candidate.bvid, candidate.embed_url)
    )
