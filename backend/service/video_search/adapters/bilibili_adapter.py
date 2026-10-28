import html
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
    from requests import RequestException
except ModuleNotFoundError:  # pragma: no cover
    requests = None
    RequestException = Exception

from backend.service.video_search.bilibili_rules import build_embed_url, extract_bvid, normalize_bvid
from backend.types.schema import QueryVariant, VideoSearchRequest

from .base import AdapterVideo, BasePlatformAdapter


logger = logging.getLogger(__name__)

HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
QUERY_SPLIT_RE = re.compile(r"[\s,，。！？、；;:：/\\]+")
LOW_QUALITY_RE = re.compile(r"(概念|图文|壁纸|PPT|AI绘画|素材|配音|混剪)", re.IGNORECASE)

SCENE_GROUPS: Dict[str, List[str]] = {
    "desert": ["大漠", "沙漠", "孤烟", "戈壁", "落日", "边塞"],
    "frontier": ["玉门关", "羌笛", "边塞", "关山", "征人", "胡天"],
    "water": ["长河", "黄河", "江", "湖", "水", "泪"],
    "spring": ["春风", "杨柳", "春", "柳", "花", "芳草"],
    "moon": ["明月", "月色", "月夜", "月"],
    "history": ["丝路", "古城", "遗址", "石窟", "汉", "唐"],
}


@dataclass(frozen=True)
class BilibiliRealConfig:
    endpoint: str
    timeout_seconds: float
    headers: Dict[str, str]
    fixture_mode: bool
    fixture_json_path: str
    fixture_json_raw: str
    max_results: int
    page: int
    page_size: int
    fallback_to_fixture: bool
    fallback_to_mock: bool


@dataclass(frozen=True)
class FixtureRegressionResult:
    file: str
    status: str
    result_count: int
    first_video: Optional[Dict[str, Any]]
    reason: str


class BilibiliAdapter(BasePlatformAdapter):
    platform = "bilibili"

    def search_mock(self, req: VideoSearchRequest, query: str) -> List[AdapterVideo]:
        matched_tags = self._preferred_tags(req, query)
        theme = self._mock_theme(matched_tags)
        title = f"{(req.poem or '河西诗词')[:12]} · {theme['title']}"
        bvid = "BV1HeXiMock01"
        return [
            AdapterVideo(
                platform=self.platform,
                title=title,
                url=f"https://www.bilibili.com/video/{bvid}",
                cover=theme["cover"],
                author="河西走廊纪录片",
                score=theme["score"],
                bvid=bvid,
                embed_url=build_embed_url(bvid),
                external_url=f"https://www.bilibili.com/video/{bvid}",
                playback_type="embed_url",
                reason=self._build_reason("mock", matched_tags, title),
                tags=matched_tags[:4] or list(theme["tags"]),
                source_mode="mock",
                source_note="mock_theme_match",
                media_playability_score=1.0,
                duplicate_key=bvid.lower(),
            )
        ]

    def search_real(self, req: VideoSearchRequest, query: str) -> List[AdapterVideo]:
        config = self._load_real_config()

        if config.fixture_mode:
            fixture_videos = self._search_from_fixture(req, query, config)
            if fixture_videos:
                return fixture_videos
            if config.fallback_to_mock:
                return self._mock_with_fallback_reason(req, query, "fixture_empty")
            return []

        real_videos = self._search_from_http(req, query, config)
        if real_videos:
            return real_videos

        if config.fallback_to_fixture:
            fixture_videos = self._search_from_fixture(
                req,
                query,
                config,
                source_note="real_http_fallback_fixture",
            )
            if fixture_videos:
                return fixture_videos

        if config.fallback_to_mock:
            return self._mock_with_fallback_reason(req, query, "real_http_empty")
        return []

    def build_fixture_regression_report(
        self,
        fixture_dir: str,
        query: str = "bilibili fixture regression",
    ) -> Dict[str, Any]:
        results = self.run_fixture_regression(fixture_dir=fixture_dir, query=query)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fixture_dir": fixture_dir,
            "query": query,
            "total": len(results),
            "success": sum(1 for item in results if item.status == "success"),
            "empty": sum(1 for item in results if item.status == "empty"),
            "failure": sum(1 for item in results if item.status == "failure"),
            "fixtures": [
                {
                    "file": item.file,
                    "file_name": Path(item.file).name,
                    "status": item.status,
                    "reason": item.reason,
                    "result_count": item.result_count,
                    "first_video": item.first_video,
                }
                for item in results
            ],
        }

    def run_fixture_regression(
        self,
        fixture_dir: str,
        query: str = "bilibili fixture regression",
    ) -> List[FixtureRegressionResult]:
        fixture_root = Path(fixture_dir)
        if not fixture_root.is_dir():
            return [
                FixtureRegressionResult(
                    file=str(fixture_root),
                    status="failure",
                    result_count=0,
                    first_video=None,
                    reason="directory_not_found",
                )
            ]

        results: List[FixtureRegressionResult] = []
        for fixture_file in sorted(fixture_root.glob("*.json")):
            try:
                payload = json.loads(fixture_file.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                results.append(
                    FixtureRegressionResult(
                        file=str(fixture_file),
                        status="failure",
                        result_count=0,
                        first_video=None,
                        reason="load_failed",
                    )
                )
                continue

            items = self._extract_result_items(payload)
            if not items:
                results.append(
                    FixtureRegressionResult(
                        file=str(fixture_file),
                        status="empty",
                        result_count=0,
                        first_video=None,
                        reason="no_result_items",
                    )
                )
                continue

            videos = [
                self._map_result_item(
                    item=item,
                    req=VideoSearchRequest(),
                    query=query,
                    source_mode="fixture",
                    source_note="fixture_file",
                    round_name="fixture",
                )
                for item in items
            ]
            videos = [item for item in videos if item is not None]
            if not videos:
                results.append(
                    FixtureRegressionResult(
                        file=str(fixture_file),
                        status="empty",
                        result_count=0,
                        first_video=None,
                        reason="no_adapter_videos",
                    )
                )
                continue

            first = videos[0]
            results.append(
                FixtureRegressionResult(
                    file=str(fixture_file),
                    status="success",
                    result_count=len(videos),
                    first_video={
                        "platform": first.platform,
                        "title": first.title,
                        "url": first.url,
                        "author": first.author,
                        "score": first.score,
                        "source_mode": first.source_mode,
                        "source_note": first.source_note,
                    },
                    reason="ok",
                )
            )
        return results

    def _load_real_config(self) -> BilibiliRealConfig:
        return BilibiliRealConfig(
            endpoint=os.getenv(
                "BILIBILI_SEARCH_ENDPOINT",
                "https://api.bilibili.com/x/web-interface/search/type",
            ).strip(),
            timeout_seconds=self._float_env("BILIBILI_SEARCH_TIMEOUT_SECONDS", 6.0, minimum=1.0),
            headers=self._parse_headers(
                os.getenv(
                    "BILIBILI_SEARCH_HEADERS",
                    "User-Agent: Mozilla/5.0; Referer: https://www.bilibili.com/",
                )
            ),
            fixture_mode=self._bool_env("BILIBILI_FIXTURE_MODE", False),
            fixture_json_path=os.getenv(
                "BILIBILI_FIXTURE_JSON_PATH",
                str(self._default_fixture_path()),
            ).strip(),
            fixture_json_raw=os.getenv("BILIBILI_FIXTURE_JSON", "").strip(),
            max_results=self._int_env("BILIBILI_SEARCH_MAX_RESULTS", 12, minimum=1),
            page=self._int_env("BILIBILI_SEARCH_PAGE", 1, minimum=1),
            page_size=self._int_env("BILIBILI_SEARCH_PAGE_SIZE", 10, minimum=1),
            fallback_to_fixture=self._bool_env("BILIBILI_REAL_FALLBACK_TO_FIXTURE", True),
            fallback_to_mock=self._bool_env("BILIBILI_REAL_FALLBACK_TO_MOCK", True),
        )

    def _search_from_http(
        self,
        req: VideoSearchRequest,
        query: str,
        config: BilibiliRealConfig,
    ) -> List[AdapterVideo]:
        videos: List[AdapterVideo] = []
        for round_name, round_query in self._build_real_queries(req, query):
            payload = self._request_payload(config, round_query)
            if not payload:
                continue
            items = self._extract_result_items(payload)
            if not items:
                logger.info("bilibili_real_http_empty_result round=%s query=%r", round_name, round_query)
                continue

            mapped = self._map_items(
                items=items,
                req=req,
                query=round_query,
                source_mode="real_http",
                source_note=f"bilibili_real_http:{round_name}",
                round_name=round_name,
                max_results=config.max_results,
            )
            if not mapped:
                logger.info("bilibili_real_http_all_results_dropped round=%s query=%r", round_name, round_query)
                continue

            videos = self._merge_videos(videos, mapped, config.max_results)
            logger.info(
                "bilibili_real_http_round_merged round=%s query=%r merged_count=%d",
                round_name,
                round_query,
                len(videos),
            )
            if len(videos) >= config.max_results:
                break
        return videos

    def _search_from_fixture(
        self,
        req: VideoSearchRequest,
        query: str,
        config: BilibiliRealConfig,
        source_note: str = "fixture_file",
    ) -> List[AdapterVideo]:
        payload = self._load_fixture_payload(config)
        if not payload:
            return []
        items = self._extract_result_items(payload)
        return self._map_items(
            items=items,
            req=req,
            query=query,
            source_mode="fixture",
            source_note=source_note,
            round_name="fixture",
            max_results=config.max_results,
        )

    def _mock_with_fallback_reason(
        self,
        req: VideoSearchRequest,
        query: str,
        fallback_reason: str,
    ) -> List[AdapterVideo]:
        videos = self.search_mock(req, query)
        return [
            AdapterVideo(
                platform=item.platform,
                title=item.title,
                url=item.url,
                cover=item.cover,
                author=item.author,
                score=item.score,
                bvid=item.bvid,
                video_url=item.video_url,
                embed_url=item.embed_url,
                external_url=item.external_url,
                playback_type=item.playback_type,
                reason=item.reason,
                tags=list(item.tags),
                source_mode="mock",
                source_note=f"real_fallback:{fallback_reason}",
                media_playability_score=item.media_playability_score,
                duplicate_key=item.duplicate_key,
            )
            for item in videos
        ]

    def _request_payload(
        self,
        config: BilibiliRealConfig,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        if requests is None or not config.endpoint:
            logger.info("bilibili_real_http_unavailable query=%r reason=no_requests_or_endpoint", query)
            return None
        try:
            response = requests.get(
                config.endpoint,
                params={
                    "keyword": query,
                    "search_type": "video",
                    "page": config.page,
                    "page_size": config.page_size,
                },
                headers=config.headers,
                timeout=config.timeout_seconds,
            )
            response.raise_for_status()
        except RequestException:
            logger.warning("bilibili_real_http_request_failed query=%r", query, exc_info=True)
            return None

        try:
            payload = response.json()
        except ValueError:
            logger.warning("bilibili_real_http_json_failed query=%r", query, exc_info=True)
            return None

        if not isinstance(payload, dict):
            logger.info("bilibili_real_http_invalid_root query=%r root=%s", query, type(payload).__name__)
            return None
        code = payload.get("code")
        if code not in (None, 0, "0", True):
            logger.info("bilibili_real_http_non_success query=%r code=%r", query, code)
            return None
        return payload

    def _load_fixture_payload(self, config: BilibiliRealConfig) -> Optional[Dict[str, Any]]:
        try:
            if config.fixture_json_raw:
                payload = json.loads(config.fixture_json_raw)
            else:
                payload = json.loads(Path(config.fixture_json_path).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            logger.warning("bilibili_fixture_load_failed", exc_info=True)
            return None
        return payload if isinstance(payload, dict) else None

    def _extract_result_items(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key_path in (
            ("data", "result"),
            ("result",),
            ("data", "items"),
            ("items",),
        ):
            current: Any = payload
            for key in key_path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if isinstance(current, list):
                return [item for item in current if isinstance(item, dict)]
        return []

    def _map_items(
        self,
        items: List[Dict[str, Any]],
        req: VideoSearchRequest,
        query: str,
        source_mode: str,
        source_note: str,
        round_name: str,
        max_results: int,
    ) -> List[AdapterVideo]:
        videos: List[AdapterVideo] = []
        seen_keys = set()
        for item in items:
            mapped = self._map_result_item(
                item=item,
                req=req,
                query=query,
                source_mode=source_mode,
                source_note=source_note,
                round_name=round_name,
            )
            if mapped is None:
                continue
            unique_key = self._video_identity(mapped)
            if unique_key in seen_keys:
                continue
            seen_keys.add(unique_key)
            videos.append(mapped)
            if len(videos) >= max_results:
                break
        videos.sort(key=self._video_quality, reverse=True)
        return videos

    def _map_result_item(
        self,
        item: Dict[str, Any],
        req: VideoSearchRequest,
        query: str,
        source_mode: str,
        source_note: str,
        round_name: str,
    ) -> Optional[AdapterVideo]:
        title = self._normalize_text(item.get("title"))
        if not title:
            return None

        raw_tags = self._parse_tags(item)
        if self._is_low_quality_non_video(title, raw_tags):
            return None

        author = self._normalize_text(item.get("author") or item.get("author_name") or item.get("up_name")) or "未知作者"
        url = self._clean_url(item.get("arcurl") or item.get("url") or item.get("share_url"))
        cover = self._clean_url(item.get("pic") or item.get("cover") or item.get("cover_url"))
        bvid = self._extract_bvid(item, url)
        if not url and bvid:
            url = f"https://www.bilibili.com/video/{bvid}"
        embed_url = build_embed_url(bvid)
        preferred_tags = self._preferred_tags(req, query)
        tags = self._derive_tags(title=title, raw_tags=raw_tags, preferred_tags=preferred_tags)
        score = self._derive_score(item.get("rank_score"), tags, title, bvid, round_name)
        playback_type = "embed_url" if embed_url else ("external" if url else "cover")
        playability = 1.0 if embed_url else (0.72 if bvid else (0.35 if cover else 0.20))
        return AdapterVideo(
            platform=self.platform,
            title=title,
            url=url or "https://www.bilibili.com",
            cover=cover,
            author=author,
            score=score,
            bvid=bvid,
            embed_url=embed_url,
            external_url=url or "https://www.bilibili.com",
            playback_type=playback_type,
            reason=self._build_reason(source_mode, tags, title),
            tags=tags,
            source_mode=source_mode,
            source_note=source_note,
            media_playability_score=playability,
            duplicate_key=(bvid or url or title).strip().lower(),
        )

    def _build_real_queries(
        self,
        req: VideoSearchRequest,
        query: str,
    ) -> List[Tuple[str, str]]:
        variants = self._request_query_variants(req, query)
        result: List[Tuple[str, str]] = []
        seen = set()
        for item in variants:
            normalized_query = self._normalize_search_query(item.query)
            if not normalized_query or normalized_query in seen:
                continue
            seen.add(normalized_query)
            result.append((item.name, normalized_query))
        return result

    def _request_query_variants(self, req: VideoSearchRequest, query: str) -> List[QueryVariant]:
        if req.query_variants:
            return [QueryVariant(name=item.name, query=item.query) for item in req.query_variants if item.query]

        preferred_tags = self._preferred_tags(req, query)
        poem_text = self._normalize_search_query(req.poem)
        location = self._normalize_text(req.location) or "河西走廊"
        theme_terms = self._dedupe_texts(preferred_tags + [req.location, req.emotion])[:4]
        fallback_variants = [
            QueryVariant(name="original", query=poem_text or self._normalize_search_query(query)),
            QueryVariant(name="original_plus_imagery", query=self._join_query_terms([poem_text] + theme_terms[:3])),
            QueryVariant(name="theme", query=self._join_query_terms(theme_terms + ["纪录片", "风光", "航拍"])),
            QueryVariant(name="geo", query=self._join_query_terms([location] + theme_terms[:2] + ["纪录片", "风光"])),
        ]
        return [item for item in fallback_variants if item.query]

    def _merge_videos(
        self,
        current: List[AdapterVideo],
        incoming: List[AdapterVideo],
        max_results: int,
    ) -> List[AdapterVideo]:
        merged: List[AdapterVideo] = list(current)
        positions = {self._video_identity(item): index for index, item in enumerate(merged)}
        for item in incoming:
            key = self._video_identity(item)
            existing_index = positions.get(key)
            if existing_index is None:
                positions[key] = len(merged)
                merged.append(item)
                continue
            if self._video_quality(item) > self._video_quality(merged[existing_index]):
                merged[existing_index] = item
        merged.sort(key=self._video_quality, reverse=True)
        return merged[:max_results]

    def _video_identity(self, item: AdapterVideo) -> str:
        for value in (item.bvid, item.external_url, item.url):
            normalized = self._normalize_text(value).lower()
            if normalized:
                return normalized
        return item.title.strip().lower()

    def _video_quality(self, item: AdapterVideo) -> float:
        score = float(item.score)
        if normalize_bvid(item.bvid):
            score += 1.2
        if item.embed_url:
            score += 1.0
        if item.cover:
            score += 0.08
        if LOW_QUALITY_RE.search(item.title):
            score -= 0.8
        return score

    def _dedupe_texts(self, values: List[Any]) -> List[str]:
        result: List[str] = []
        for item in values:
            text = self._normalize_text(item)
            if text and text not in result:
                result.append(text)
        return result

    def _join_query_terms(self, values: List[str]) -> str:
        return " ".join(item for item in self._dedupe_texts(values) if item)

    def _normalize_search_query(self, value: Any) -> str:
        text = self._normalize_text(value)
        if not text:
            return ""
        return QUERY_SPLIT_RE.sub(" ", text).strip()

    def _preferred_tags(self, req: VideoSearchRequest, query: str) -> List[str]:
        values = [
            *req.scene_tags,
            *req.keywords,
            *req.imagery,
            req.location,
            req.emotion,
            query,
        ]
        result: List[str] = []
        for item in values:
            text = self._normalize_text(item)
            if not text:
                continue
            if text == query:
                for group in SCENE_GROUPS.values():
                    for keyword in group:
                        if keyword in text and keyword not in result:
                            result.append(keyword)
                continue
            if text not in result:
                result.append(text)
        return result

    def _derive_tags(self, title: str, raw_tags: List[str], preferred_tags: List[str]) -> List[str]:
        haystack = f"{title} {' '.join(raw_tags)}"
        tags: List[str] = []
        for tag in raw_tags:
            cleaned = self._normalize_text(tag)
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
        for tag in preferred_tags:
            if tag and tag in haystack and tag not in tags:
                tags.append(tag)
        for group in SCENE_GROUPS.values():
            for keyword in group:
                if keyword in haystack and keyword not in tags:
                    tags.append(keyword)
        return tags[:8]

    def _derive_score(
        self,
        raw_value: Any,
        tags: List[str],
        title: str,
        bvid: str,
        round_name: str,
    ) -> float:
        fallback = min(0.90, 0.52 + len(tags) * 0.05)
        try:
            score = float(raw_value)
        except (TypeError, ValueError):
            score = fallback
        if not math.isfinite(score):
            score = fallback
        elif not 0.0 <= score <= 1.0:
            score = 0.55 + min(0.25, math.log10(abs(score) + 1.0) * 0.08)

        if "纪录" in title or "航拍" in title:
            score += 0.05
        if round_name == "geo":
            score += 0.03
        if normalize_bvid(bvid):
            score += 0.08
        if self._is_low_quality_non_video(title, tags):
            score -= 0.22
        return max(0.0, min(0.99, max(score, fallback)))

    def _mock_theme(self, matched_tags: List[str]) -> Dict[str, Any]:
        tag_text = " ".join(matched_tags)
        if any(keyword in tag_text for keyword in SCENE_GROUPS["desert"]):
            return {
                "title": "河西走廊大漠航拍",
                "cover": "https://picsum.photos/seed/bili-desert/640/360",
                "score": 0.86,
                "tags": ["大漠", "落日"],
            }
        if any(keyword in tag_text for keyword in SCENE_GROUPS["frontier"] + SCENE_GROUPS["history"]):
            return {
                "title": "边塞与丝路人文纪录片",
                "cover": "https://picsum.photos/seed/bili-history/640/360",
                "score": 0.90,
                "tags": ["边塞", "丝路"],
            }
        if any(keyword in tag_text for keyword in SCENE_GROUPS["spring"] + SCENE_GROUPS["moon"]):
            return {
                "title": "春风杨柳意境短片",
                "cover": "https://picsum.photos/seed/bili-spring/640/360",
                "score": 0.80,
                "tags": ["春风", "杨柳"],
            }
        return {
            "title": "河西诗词影像",
            "cover": "https://picsum.photos/seed/bili-default/640/360",
            "score": 0.68,
            "tags": ["诗词"],
        }

    def _build_reason(self, source_mode: str, tags: List[str], title: str) -> str:
        prefix = {
            "real_http": "B站实时搜索命中",
            "fixture": "B站 fixture 命中",
            "mock": "B站 mock 命中",
        }.get(source_mode, "B站结果命中")
        if tags:
            return f"{prefix}：{' / '.join(tags[:4])}"
        return f"{prefix}：{title}"

    def _parse_tags(self, item: Dict[str, Any]) -> List[str]:
        values: List[str] = []
        raw_tag = item.get("tag") or item.get("tags") or item.get("typename")
        if isinstance(raw_tag, list):
            values.extend(self._normalize_text(tag) for tag in raw_tag)
        else:
            values.extend(
                token.strip()
                for token in self._normalize_text(raw_tag).replace("，", ",").split(",")
                if token.strip()
            )
        hit_columns = item.get("hit_columns")
        if isinstance(hit_columns, list):
            values.extend(self._normalize_text(tag) for tag in hit_columns)
        return [item for item in values if item]

    def _extract_bvid(self, item: Dict[str, Any], url: str) -> str:
        direct = normalize_bvid(item.get("bvid"))
        if direct:
            return direct
        for field_name in (
            "arcurl",
            "url",
            "share_url",
            "uri",
            "goto_url",
            "short_link",
            "short_link_v2",
        ):
            matched = extract_bvid(self._clean_url(item.get(field_name)))
            if matched:
                return matched
        matched = extract_bvid(url)
        if matched:
            return matched
        return extract_bvid(json.dumps(item, ensure_ascii=False))

    def _is_low_quality_non_video(self, title: str, tags: List[str]) -> bool:
        haystack = " ".join([title, *tags]).lower()
        return bool(LOW_QUALITY_RE.search(haystack))

    @staticmethod
    def _bool_env(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _int_env(name: str, default: int, minimum: int = 0) -> int:
        raw = os.getenv(name, str(default)).strip()
        try:
            value = int(raw)
        except ValueError:
            value = default
        return max(minimum, value)

    @staticmethod
    def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
        raw = os.getenv(name, str(default)).strip()
        try:
            value = float(raw)
        except ValueError:
            value = default
        return max(minimum, value)

    @staticmethod
    def _parse_headers(value: str) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for entry in value.split(";"):
            if ":" not in entry:
                continue
            key, item = entry.split(":", 1)
            key = key.strip()
            item = item.strip()
            if key and item:
                headers[key] = item
        return headers

    @staticmethod
    def _default_fixture_path() -> Path:
        return Path(__file__).resolve().parents[1] / "fixtures" / "bilibili" / "default.json"

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        text = html.unescape(str(value))
        text = HTML_TAG_RE.sub("", text)
        text = SPACE_RE.sub(" ", text)
        return text.strip()

    @staticmethod
    def _clean_url(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.startswith("//"):
            return f"https:{text}"
        return text


adapter = BilibiliAdapter()
