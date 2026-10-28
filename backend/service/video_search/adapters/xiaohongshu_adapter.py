import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

try:
    import requests
    from requests import RequestException
except ModuleNotFoundError:  # pragma: no cover
    requests = None
    RequestException = Exception

from backend.types.schema import VideoSearchRequest

from .base import AdapterVideo, BasePlatformAdapter


logger = logging.getLogger(__name__)

TRAVEL_TAGS = ["日落", "日出", "旅拍", "旅行", "机位", "风光", "打卡", "徒步", "公路"]
POETRY_TAGS = ["边塞", "大漠", "杨柳", "黄河", "春风", "月色", "羌笛", "丝路", "玉门关"]


@dataclass(frozen=True)
class XiaohongshuRealConfig:
    endpoint: str
    timeout_seconds: float
    headers: Dict[str, str]
    fixture_mode: bool
    fixture_json_path: str
    fixture_json_raw: str
    max_results: int
    http_method: str
    query_param_name: str
    query_in_body: bool
    body_template: Dict[str, Any]
    status_path: str
    success_status_values: str
    empty_status_values: str
    result_list_path: str
    title_path: str
    url_path: str
    video_url_path: str
    embed_url_path: str
    external_url_path: str
    cover_path: str
    author_path: str
    score_path: str
    tags_path: str
    fallback_to_fixture: bool
    fallback_to_substitute: bool
    fallback_to_mock: bool


@dataclass(frozen=True)
class FixtureRegressionResult:
    file: str
    status: str
    result_count: int
    first_video: Optional[Dict[str, Any]]
    reason: str


class XiaohongshuAdapter(BasePlatformAdapter):
    platform = "xiaohongshu"

    def search_mock(self, req: VideoSearchRequest, query: str) -> List[AdapterVideo]:
        matched_tags = self._preferred_tags(req, query)
        theme = self._mock_theme(matched_tags)
        title = f"{(req.poem or '河西诗词')[:8]} · {theme['title']}"
        return [
            AdapterVideo(
                platform=self.platform,
                title=title,
                url="https://www.xiaohongshu.com",
                cover=theme["cover"],
                author="河西旅行手帐",
                score=theme["score"],
                external_url="https://www.xiaohongshu.com",
                reason=self._build_reason("mock", matched_tags or theme["tags"], title),
                tags=matched_tags[:4] or list(theme["tags"]),
                source_mode="mock",
                source_note="mock_theme_match",
                media_playability_score=0.25,
                duplicate_key=f"mock:{title}".lower(),
            )
        ]

    def search_real(self, req: VideoSearchRequest, query: str) -> List[AdapterVideo]:
        config = self._load_real_config()
        if config.fixture_mode:
            fixture_videos = self._search_from_fixture(req, query, config)
            if fixture_videos:
                return fixture_videos
            if config.fallback_to_substitute:
                return [self._build_substitute_result(req, query, "fixture_empty")]
            if config.fallback_to_mock:
                return self._mock_with_fallback(req, query, "fixture_empty")
            return []

        real_videos = self._search_from_http(req, query, config)
        if real_videos:
            return real_videos
        if config.fallback_to_fixture:
            fixture_videos = self._search_from_fixture(req, query, config, source_note="real_http_fallback_fixture")
            if fixture_videos:
                return fixture_videos
        if config.fallback_to_substitute:
            return [self._build_substitute_result(req, query, "real_http_empty")]
        if config.fallback_to_mock:
            return self._mock_with_fallback(req, query, "real_http_empty")
        return []

    def build_fixture_regression_report(
        self,
        fixture_dir: str,
        query: str = "xiaohongshu fixture regression",
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
        query: str = "xiaohongshu fixture regression",
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
            items = self._extract_result_items(payload, self._load_real_config())
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
            videos = self._map_items(
                items=items,
                req=VideoSearchRequest(),
                query=query,
                config=self._load_real_config(),
                source_mode="fixture",
                source_note="fixture_file",
            )
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

    def _load_real_config(self) -> XiaohongshuRealConfig:
        headers = self._parse_headers(
            os.getenv(
                "XIAOHONGSHU_SEARCH_HEADERS",
                "User-Agent: Mozilla/5.0; Referer: https://www.xiaohongshu.com/",
            )
        )
        api_key = os.getenv("XIAOHONGSHU_SEARCH_API_KEY", "").strip()
        auth_header = os.getenv("XIAOHONGSHU_SEARCH_AUTH_HEADER", "Authorization").strip() or "Authorization"
        if api_key and auth_header not in headers:
            headers[auth_header] = f"Bearer {api_key}"
        return XiaohongshuRealConfig(
            endpoint=os.getenv("XIAOHONGSHU_SEARCH_ENDPOINT", "").strip(),
            timeout_seconds=self._float_env("XIAOHONGSHU_SEARCH_TIMEOUT_SECONDS", 6.0, minimum=1.0),
            headers=headers,
            fixture_mode=self._bool_env("XIAOHONGSHU_FIXTURE_MODE", False),
            fixture_json_path=os.getenv(
                "XIAOHONGSHU_FIXTURE_JSON_PATH",
                str(self._default_fixture_path()),
            ).strip(),
            fixture_json_raw=os.getenv("XIAOHONGSHU_FIXTURE_JSON", "").strip(),
            max_results=self._int_env("XIAOHONGSHU_SEARCH_MAX_RESULTS", 8, minimum=1),
            http_method=os.getenv("XIAOHONGSHU_SEARCH_HTTP_METHOD", "GET").strip().upper() or "GET",
            query_param_name=os.getenv("XIAOHONGSHU_SEARCH_QUERY_PARAM_NAME", "keyword").strip() or "keyword",
            query_in_body=self._bool_env("XIAOHONGSHU_QUERY_IN_BODY", False),
            body_template=self._json_env("XIAOHONGSHU_SEARCH_BODY_TEMPLATE", {}),
            status_path=os.getenv("XIAOHONGSHU_STATUS_FIELD", "code|status|status_code|success").strip(),
            success_status_values=os.getenv("XIAOHONGSHU_SUCCESS_STATUS_VALUES", "0|200|ok|success|true").strip(),
            empty_status_values=os.getenv("XIAOHONGSHU_EMPTY_STATUS_VALUES", "204|empty|no_result|no_results|not_found").strip(),
            result_list_path=os.getenv("XIAOHONGSHU_RESULT_LIST_PATH", "data.items|data.notes|data.note_card_list|data.feeds|items|notes|data").strip(),
            title_path=os.getenv("XIAOHONGSHU_TITLE_FIELD", "title|display_title|desc|note.title|note_card.display_title|note_card.title").strip(),
            url_path=os.getenv("XIAOHONGSHU_URL_FIELD", "url|share_url|web_url|link|note_url|note_card.share_url|note_card.url").strip(),
            video_url_path=os.getenv("XIAOHONGSHU_VIDEO_URL_FIELD", "video_url|video.url|note.video.url|note_card.video.url|note_card.video.media.stream.h264.0.master_url").strip(),
            embed_url_path=os.getenv("XIAOHONGSHU_EMBED_URL_FIELD", "embed_url|player_url|iframe_url").strip(),
            external_url_path=os.getenv("XIAOHONGSHU_EXTERNAL_URL_FIELD", "share_url|url|web_url|link|note_url|note_card.share_url|note_card.url").strip(),
            cover_path=os.getenv("XIAOHONGSHU_COVER_FIELD", "cover|cover_url|image|image_url|images.0.url|note_card.image_list.0.url_default|note_card.cover.url_default").strip(),
            author_path=os.getenv("XIAOHONGSHU_AUTHOR_FIELD", "author|author_name|nickname|user.nickname|user.name|note_card.user.nickname").strip(),
            score_path=os.getenv("XIAOHONGSHU_SCORE_FIELD", "score|rank_score|like_count|liked_count").strip(),
            tags_path=os.getenv("XIAOHONGSHU_TAGS_FIELD", "tag_list|tags|topics|topic_names|note_card.tag_list").strip(),
            fallback_to_fixture=self._bool_env("XIAOHONGSHU_REAL_FALLBACK_TO_FIXTURE", True),
            fallback_to_substitute=self._bool_env("XIAOHONGSHU_REAL_FALLBACK_TO_SUBSTITUTE", True),
            fallback_to_mock=self._bool_env("XIAOHONGSHU_REAL_FALLBACK_TO_MOCK", True),
        )

    def _search_from_http(self, req: VideoSearchRequest, query: str, config: XiaohongshuRealConfig) -> List[AdapterVideo]:
        payload = self._request_payload(config, query)
        if payload is None:
            return []
        normalized = self._normalize_response(payload, config)
        if normalized is None:
            return []
        items = self._extract_result_items(normalized, config)
        return self._map_items(items, req, query, config, "real_http", "xiaohongshu_real_http")

    def _search_from_fixture(
        self,
        req: VideoSearchRequest,
        query: str,
        config: XiaohongshuRealConfig,
        source_note: str = "fixture_file",
    ) -> List[AdapterVideo]:
        payload = self._load_fixture_payload(config)
        if payload is None:
            return []
        normalized = self._normalize_response(payload, config, allow_missing_status=True)
        if normalized is None:
            return []
        items = self._extract_result_items(normalized, config)
        return self._map_items(items, req, query, config, "fixture", source_note)

    def _request_payload(self, config: XiaohongshuRealConfig, query: str) -> Optional[Any]:
        if requests is None or not config.endpoint:
            return None
        params: Dict[str, Any] = {}
        json_body: Optional[Dict[str, Any]] = None
        if config.query_in_body:
            json_body = dict(config.body_template)
            json_body[config.query_param_name] = query
        elif config.http_method == "GET":
            params[config.query_param_name] = query
        else:
            json_body = dict(config.body_template)
            if config.query_param_name not in json_body:
                json_body[config.query_param_name] = query

        try:
            if config.http_method == "POST":
                response = requests.post(
                    config.endpoint,
                    params=params,
                    json=json_body,
                    headers=config.headers,
                    timeout=config.timeout_seconds,
                )
            else:
                response = requests.get(
                    config.endpoint,
                    params=params or {config.query_param_name: query},
                    headers=config.headers,
                    timeout=config.timeout_seconds,
                )
            response.raise_for_status()
        except RequestException:
            logger.warning("xiaohongshu_real_http_request_failed query=%r", query, exc_info=True)
            return None

        try:
            return response.json()
        except ValueError:
            logger.warning("xiaohongshu_real_http_json_failed query=%r", query, exc_info=True)
            return None

    def _load_fixture_payload(self, config: XiaohongshuRealConfig) -> Optional[Any]:
        try:
            if config.fixture_json_raw:
                payload = json.loads(config.fixture_json_raw)
            else:
                payload = json.loads(Path(config.fixture_json_path).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            logger.warning("xiaohongshu_fixture_load_failed", exc_info=True)
            return None
        return payload

    def _normalize_response(
        self,
        payload: Any,
        config: XiaohongshuRealConfig,
        allow_missing_status: bool = False,
    ) -> Optional[Any]:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return None
        status_value = self._first_value_by_dot_paths(payload, config.status_path)
        if status_value is None:
            return payload if allow_missing_status or True else None
        status_text = self._normalize_status_value(status_value)
        if status_text in self._normalized_value_options(config.success_status_values):
            return payload
        if status_text in self._normalized_value_options(config.empty_status_values):
            return None
        return None

    def _extract_result_items(self, payload: Any, config: XiaohongshuRealConfig) -> List[Dict[str, Any]]:
        configured = self._first_value_by_dot_paths(payload, config.result_list_path)
        if isinstance(configured, list):
            return [item for item in configured if isinstance(item, dict)]
        if isinstance(configured, dict):
            return [configured]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _map_items(
        self,
        items: List[Dict[str, Any]],
        req: VideoSearchRequest,
        query: str,
        config: XiaohongshuRealConfig,
        source_mode: str,
        source_note: str,
    ) -> List[AdapterVideo]:
        videos: List[AdapterVideo] = []
        seen_keys = set()
        for item in items:
            mapped = self._map_result_item(item, req, query, config, source_mode, source_note)
            if mapped is None:
                continue
            if mapped.duplicate_key and mapped.duplicate_key in seen_keys:
                continue
            seen_keys.add(mapped.duplicate_key)
            videos.append(mapped)
            if len(videos) >= config.max_results:
                break
        return videos

    def _map_result_item(
        self,
        item: Dict[str, Any],
        req: VideoSearchRequest,
        query: str,
        config: XiaohongshuRealConfig,
        source_mode: str,
        source_note: str,
    ) -> Optional[AdapterVideo]:
        title = self._first_text_by_paths(item, config.title_path)
        if not title:
            return None
        video_url = self._normalize_url(self._first_text_by_paths(item, config.video_url_path))
        embed_url = self._normalize_url(self._first_text_by_paths(item, config.embed_url_path))
        url = self._normalize_url(self._first_text_by_paths(item, config.url_path)) or video_url or embed_url
        external_url = self._normalize_url(self._first_text_by_paths(item, config.external_url_path)) or url
        if not video_url and self._looks_like_media_url(url):
            video_url = url
        if not video_url and self._looks_like_media_url(external_url):
            video_url = external_url
        cover = self._normalize_url(self._first_text_by_paths(item, config.cover_path))
        author = self._first_text_by_paths(item, config.author_path) or "未知作者"
        tags = self._extract_tags(item, config.tags_path, req, query, title)
        score = self._derive_score(self._first_value_by_dot_paths(item, config.score_path), tags)
        playability = 1.0 if video_url else 0.78 if embed_url else 0.45 if cover else 0.25 if external_url else 0.0
        playback_type = "video_url" if video_url else "embed_url" if embed_url else "cover" if cover else "external"
        duplicate_key = (external_url or video_url or url or title).strip().lower()
        return AdapterVideo(
            platform=self.platform,
            title=title,
            url=url or external_url,
            cover=cover,
            author=author,
            score=score,
            video_url=video_url,
            embed_url=embed_url,
            external_url=external_url,
            playback_type=playback_type,
            reason=self._build_reason(source_mode, tags, title),
            tags=tags,
            source_mode=source_mode,
            source_note=source_note,
            media_playability_score=playability,
            duplicate_key=duplicate_key,
        )

    def _build_substitute_result(self, req: VideoSearchRequest, query: str, reason: str) -> AdapterVideo:
        text = query or req.poem or "河西走廊诗词"
        search_url = f"https://www.xiaohongshu.com/search_result?keyword={quote(text)}"
        tags = self._preferred_tags(req, query)[:4]
        return AdapterVideo(
            platform=self.platform,
            title=f"小红书检索页 · {text[:16]}",
            url=search_url,
            cover="https://picsum.photos/seed/xhs-substitute/640/360",
            author="小红书检索替代入口",
            score=0.46,
            external_url=search_url,
            playback_type="external",
            reason="当前未接稳定开放搜索接口，先返回小红书检索页作为替代方案。",
            tags=tags,
            source_mode="substitute_link",
            source_note=f"web_search_page:{reason}",
            media_playability_score=0.25,
            duplicate_key=search_url.lower(),
        )

    def _mock_with_fallback(self, req: VideoSearchRequest, query: str, reason: str) -> List[AdapterVideo]:
        return [
            AdapterVideo(
                platform=item.platform,
                title=item.title,
                url=item.url,
                cover=item.cover,
                author=item.author,
                score=item.score,
                video_url=item.video_url,
                embed_url=item.embed_url,
                external_url=item.external_url,
                playback_type=item.playback_type,
                reason=item.reason,
                tags=list(item.tags),
                source_mode="mock",
                source_note=f"real_fallback:{reason}",
                media_playability_score=item.media_playability_score,
                duplicate_key=item.duplicate_key,
            )
            for item in self.search_mock(req, query)
        ]

    def _preferred_tags(self, req: VideoSearchRequest, query: str) -> List[str]:
        result: List[str] = []
        for item in (*req.scene_tags, *req.keywords, *req.imagery, req.emotion, req.poem, query):
            text = self._clean_text(item)
            if not text:
                continue
            if text not in result:
                result.append(text)
        return result

    def _extract_tags(
        self,
        item: Dict[str, Any],
        tags_path: str,
        req: VideoSearchRequest,
        query: str,
        title: str,
    ) -> List[str]:
        tags: List[str] = []
        raw = self._first_value_by_dot_paths(item, tags_path)
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict):
                    for key in ("name", "tag_name", "topic_name", "title"):
                        text = self._clean_text(entry.get(key))
                        if text and text not in tags:
                            tags.append(text)
                else:
                    text = self._clean_text(entry)
                    if text and text not in tags:
                        tags.append(text)
        elif raw:
            for token in self._clean_text(raw).replace("，", ",").split(","):
                token = token.strip()
                if token and token not in tags:
                    tags.append(token)
        haystack = f"{title} {' '.join(tags)}"
        for tag in self._preferred_tags(req, query):
            if tag and tag in haystack and tag not in tags:
                tags.append(tag)
        for tag in TRAVEL_TAGS + POETRY_TAGS:
            if tag in haystack and tag not in tags:
                tags.append(tag)
        return tags[:6]

    def _derive_score(self, raw_value: Any, tags: List[str]) -> float:
        fallback = min(0.9, 0.58 + len(tags) * 0.05)
        try:
            score = float(raw_value)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(score):
            return fallback
        if 0.0 <= score <= 1.0:
            return max(score, fallback)
        normalized = 0.52 + min(0.22, math.log10(abs(score) + 1.0) * 0.07)
        return max(0.0, min(0.95, max(normalized, fallback)))

    def _mock_theme(self, matched_tags: List[str]) -> Dict[str, Any]:
        text = " ".join(matched_tags)
        if any(tag in text for tag in ["日落", "日出", "机位", "旅拍"]):
            return {
                "title": "日落旅拍机位",
                "cover": "https://picsum.photos/seed/xhs-sunset/640/360",
                "score": 0.82,
                "tags": ["日落", "旅拍"],
            }
        if any(tag in text for tag in ["春风", "杨柳", "花", "月色"]):
            return {
                "title": "诗意旅拍氛围",
                "cover": "https://picsum.photos/seed/xhs-poetry/640/360",
                "score": 0.76,
                "tags": ["春风", "月色"],
            }
        return {
            "title": "河西旅行手帐",
            "cover": "https://picsum.photos/seed/xhs-default/640/360",
            "score": 0.63,
            "tags": ["旅行"],
        }

    def _build_reason(self, source_mode: str, tags: List[str], title: str) -> str:
        prefix = {
            "real_http": "小红书真实搜索命中",
            "fixture": "小红书 fixture 命中",
            "mock": "小红书 mock 命中",
            "substitute_link": "小红书替代入口",
        }.get(source_mode, "小红书结果命中")
        if tags:
            return f"{prefix}：{' / '.join(tags[:4])}。"
        return f"{prefix}：{title}。"

    @staticmethod
    def _default_fixture_path() -> Path:
        return Path(__file__).resolve().parents[1] / "fixtures" / "xiaohongshu" / "default.json"

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
    def _json_env(name: str, default: Dict[str, Any]) -> Dict[str, Any]:
        raw = os.getenv(name, "").strip()
        if not raw:
            return dict(default)
        try:
            value = json.loads(raw)
        except ValueError:
            return dict(default)
        return value if isinstance(value, dict) else dict(default)

    @staticmethod
    def _parse_headers(value: str) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for pair in value.split(";"):
            if ":" not in pair:
                continue
            key, item = pair.split(":", 1)
            key = key.strip()
            item = item.strip()
            if key and item:
                headers[key] = item
        return headers

    @staticmethod
    def _normalize_url(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.startswith("//"):
            return f"https:{text}"
        if text.startswith("/"):
            return f"https://www.xiaohongshu.com{text}"
        return text

    @staticmethod
    def _looks_like_media_url(value: str) -> bool:
        text = str(value or "").strip().lower()
        return text.startswith(("http://", "https://")) and any(
            token in text for token in (".mp4", ".m3u8", ".mov", ".webm", ".flv")
        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _normalize_status_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and math.isfinite(value):
            if float(value).is_integer():
                return str(int(value))
        return str(value).strip().lower()

    def _normalized_value_options(self, value: str) -> set:
        return {self._normalize_status_value(item) for item in value.split("|") if item.strip()}

    def _first_value_by_dot_paths(self, item: Any, paths: str) -> Any:
        for path in [part.strip() for part in paths.split("|") if part.strip()]:
            value = self._value_at_dot_path(item, path)
            if value is not None:
                return value
        return None

    def _first_text_by_paths(self, item: Any, paths: str) -> str:
        value = self._first_value_by_dot_paths(item, paths)
        return self._clean_text(value)

    def _value_at_dot_path(self, payload: Any, path: str) -> Any:
        current = payload
        for part in path.split("."):
            part = part.strip()
            if not part:
                return None
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                if index >= len(current):
                    return None
                current = current[index]
            else:
                return None
            if current is None:
                return None
        return current


adapter = XiaohongshuAdapter()
