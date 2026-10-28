import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from backend.types.schema import QueryPlan, QueryVariant

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover
    requests = None


SCENE_KEYWORDS: Dict[str, List[str]] = {
    "desert": ["大漠", "沙漠", "孤烟", "戈壁", "落日", "边塞"],
    "frontier": ["玉门关", "阳关", "嘉峪关", "边关", "关山", "羌笛", "河西走廊"],
    "spring": ["春风", "杨柳", "花", "芳草", "草色"],
    "water": ["黄河", "长河", "江", "河", "水", "泪"],
    "moon": ["明月", "月色", "月夜", "月"],
    "snow": ["雪", "霜", "寒", "冬"],
    "travel": ["旅拍", "旅行", "远行", "打卡", "vlog", "Vlog", "风光"],
    "history": ["丝路", "古城", "遗址", "石窟", "汉唐", "敦煌"],
}

GENERIC_STOPWORDS = {
    "中国",
    "诗词",
    "自然意象",
    "视频",
    "短视频",
}

DEFAULT_NEGATIVE_TERMS = [
    "带货",
    "口播",
    "广告",
    "纯字幕",
    "游戏",
    "直播切片",
]

HEXI_GEO_TERMS = [
    "河西走廊",
    "敦煌",
    "玉门关",
    "阳关",
    "嘉峪关",
    "酒泉",
    "张掖",
    "凉州",
    "祁连山",
    "鸣沙山",
    "月牙泉",
]

HEXI_THEME_TERMS = [
    "边塞",
    "大漠",
    "长河",
    "黄河",
    "落日",
    "羌笛",
    "孤烟",
    "关塞",
    "丝路",
    "戈壁",
    "孤城",
]

BILIBILI_MEDIA_TERMS = ["纪录片", "风光", "航拍", "旅行", "人文"]
BILIBILI_NEGATIVE_TERMS = ["概念设计", "概念片", "纯图片", "图文", "壁纸", "PPT", "AI绘画"]
ALLOWED_PLATFORMS = {"bilibili"}


class QwenQueryPlanner:
    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "",
        model: str = "",
        timeout_seconds: float = 15.0,
        disable_remote: bool = False,
    ):
        self.api_key = api_key.strip()
        self.endpoint = endpoint.strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.model = model.strip() or "qwen-plus"
        self.timeout_seconds = timeout_seconds
        self.disable_remote = disable_remote

    def plan(self, prediction: Dict[str, Any]) -> QueryPlan:
        remote_plan = self._plan_remote(prediction)
        if remote_plan is not None:
            return remote_plan
        return self._plan_fallback(prediction)

    def _plan_remote(self, prediction: Dict[str, Any]) -> Optional[QueryPlan]:
        from backend.admin_auth import SESSION, session_valid, require_generation_access
        if not session_valid(SESSION.get()):
            return None  # Visitors use the existing local search-query rules.
        if self.disable_remote or not self.api_key or requests is None:
            return None

        try:
            require_generation_access("query_planner")
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是河西走廊诗词视频搜索 query planner。"
                                "只返回 JSON。"
                                "字段必须包含 query, query_variants, keywords, negative_terms, "
                                "platform_hints, scene_tags, reasoning。"
                                "query_variants 必须按 original, original_plus_imagery, theme, geo 四轮输出。"
                                "platform_hints 只允许 bilibili。"
                                "优先产出适合 B站实时搜索的中文视频查询词。"
                            ),
                        },
                        {"role": "user", "content": self._build_prompt(prediction)},
                    ],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not isinstance(content, str):
                return None
            data = json.loads(content)
            if not isinstance(data, dict):
                return None
            return self._normalize_remote_plan(data, prediction)
        except Exception:
            return None

    def _normalize_remote_plan(self, data: Dict[str, Any], prediction: Dict[str, Any]) -> QueryPlan:
        text = str(prediction.get("text", "")).strip()
        emotion = _prediction_emotion(prediction)
        imagery_labels = _prediction_imagery(prediction)
        scene_tags = _infer_scene_tags(text, imagery_labels)
        geo_terms = _extract_hits([text, *imagery_labels, *scene_tags], HEXI_GEO_TERMS)
        theme_terms = _extract_hits([text, *imagery_labels, *scene_tags], HEXI_THEME_TERMS)
        keywords = _dedupe_texts(geo_terms + theme_terms + data.get("keywords", []))
        negative_terms = _dedupe_texts(data.get("negative_terms", [])) or list(DEFAULT_NEGATIVE_TERMS)
        negative_terms = _dedupe_texts(negative_terms + BILIBILI_NEGATIVE_TERMS)
        remote_variants = _coerce_query_variants(data.get("query_variants"))
        query_variants = remote_variants or _build_query_variants(
            text=text,
            location=_prediction_location(prediction),
            imagery_labels=imagery_labels,
            emotion=emotion,
            scene_tags=scene_tags,
            keywords=keywords,
            geo_terms=geo_terms,
            theme_terms=theme_terms,
        )
        primary_query = str(data.get("query", "")).strip() or (query_variants[0].query if query_variants else "")
        platform_hints = _normalize_platform_hints(data.get("platform_hints", [])) or ["bilibili"]
        return QueryPlan(
            provider="qwen",
            mode="remote",
            query=primary_query,
            query_variants=query_variants,
            keywords=keywords,
            negative_terms=negative_terms,
            platform_hints=platform_hints,
            scene_tags=_dedupe_texts(scene_tags + data.get("scene_tags", [])),
            reasoning=str(data.get("reasoning", "")).strip()
            or "Qwen remote planner generated multi-round Bilibili queries.",
        )

    def _plan_fallback(self, prediction: Dict[str, Any]) -> QueryPlan:
        text = str(prediction.get("text", "")).strip()
        location = _prediction_location(prediction)
        emotion = _prediction_emotion(prediction)
        imagery_labels = _prediction_imagery(prediction)
        scene_tags = _infer_scene_tags(text, imagery_labels)
        geo_terms = _extract_hits([text, *imagery_labels, *scene_tags, location], HEXI_GEO_TERMS)
        theme_terms = _extract_hits([text, *imagery_labels, *scene_tags], HEXI_THEME_TERMS)
        keywords = _dedupe_texts(
            geo_terms
            + theme_terms
            + scene_tags
            + _extract_text_keywords(text)
            + [item for item in imagery_labels if item not in GENERIC_STOPWORDS]
            + ([location] if location else [])
            + ([emotion] if emotion and emotion not in GENERIC_STOPWORDS else [])
        )
        negative_terms = _infer_negative_terms(scene_tags + geo_terms + theme_terms, emotion)
        query_variants = _build_query_variants(
            text=text,
            location=location,
            imagery_labels=imagery_labels,
            emotion=emotion,
            scene_tags=scene_tags,
            keywords=keywords,
            geo_terms=geo_terms,
            theme_terms=theme_terms,
        )
        primary_query = query_variants[0].query if query_variants else _normalize_query(text)
        return QueryPlan(
            provider="qwen-fallback",
            mode="fallback",
            query=primary_query,
            query_variants=query_variants,
            keywords=keywords,
            negative_terms=negative_terms,
            platform_hints=["bilibili"],
            scene_tags=scene_tags,
            reasoning=(
                "fallback planner emits four stable Bilibili rounds: original, original_plus_imagery, "
                "theme, and geo."
            ),
        )

    @staticmethod
    def _build_prompt(prediction: Dict[str, Any]) -> str:
        return (
            "请根据下列诗句语义分析结果，生成适合 B站视频实时搜索的 query plan。\n"
            "要求：\n"
            "1. query_variants 必须输出四轮：original、original_plus_imagery、theme、geo；\n"
            "2. query 要保留主检索词，不要空泛；\n"
            "3. keywords 控制在 4 到 10 个；\n"
            "4. negative_terms 给出应压低的概念图、图文、壁纸、AI绘画等过滤词；\n"
            "5. platform_hints 只能为 bilibili；\n"
            "6. 优先补足河西走廊、敦煌、玉门关、阳关、嘉峪关、大漠、边塞、羌笛、长河等 B站可搜词；\n"
            "7. 严格返回 JSON。\n"
            f"{json.dumps(prediction, ensure_ascii=False)}"
        )


@lru_cache(maxsize=1)
def get_query_planner() -> QwenQueryPlanner:
    return QwenQueryPlanner(
        api_key=os.getenv("QWEN_API_KEY", "") if os.getenv("ENABLE_PAID_APIS", "false").lower() == "true" else "",
        endpoint=os.getenv("QWEN_API_BASE", ""),
        model=os.getenv("QWEN_MODEL", ""),
        timeout_seconds=max(5.0, _float_env("QWEN_TIMEOUT_SECONDS", 15.0)),
        disable_remote=_bool_env("QWEN_DISABLE_REMOTE", False),
    )


def build_query_plan(prediction: Dict[str, Any]) -> QueryPlan:
    return get_query_planner().plan(prediction)


def _coerce_query_variants(value: Any) -> List[QueryVariant]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    variants: List[QueryVariant] = []
    for index, item in enumerate(value):
        if isinstance(item, QueryVariant):
            query = item.query.strip()
            name = item.name.strip()
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("round") or f"round_{index + 1}").strip()
            query = str(item.get("query") or item.get("text") or "").strip()
        else:
            name = f"round_{index + 1}"
            query = str(item).strip()
        if query:
            variants.append(QueryVariant(name=name or f"round_{index + 1}", query=query))
    return variants


def _prediction_location(prediction: Dict[str, Any]) -> str:
    return str(
        prediction.get("place", {}).get("label")
        or prediction.get("place", {}).get("id")
        or ""
    ).strip()


def _prediction_emotion(prediction: Dict[str, Any]) -> str:
    return str(
        prediction.get("emotion", {}).get("label")
        or prediction.get("emotion", {}).get("id")
        or ""
    ).strip()


def _prediction_imagery(prediction: Dict[str, Any]) -> List[str]:
    return [
        str(item.get("label") or item.get("id") or "").strip()
        for item in prediction.get("imagery", {}).get("labels", [])
        if str(item.get("label") or item.get("id") or "").strip()
    ]


def _infer_scene_tags(text: str, imagery_labels: List[str]) -> List[str]:
    haystacks = [text, *imagery_labels]
    tags: List[str] = []
    if _contains_any(haystacks, SCENE_KEYWORDS["frontier"]):
        tags.extend(["玉门关", "羌笛", "边塞"])
    if _contains_any(haystacks, SCENE_KEYWORDS["desert"]):
        tags.extend(["大漠", "孤烟", "落日"])
    if _contains_any(haystacks, SCENE_KEYWORDS["spring"]):
        tags.extend(["春风", "杨柳"])
    if _contains_any(haystacks, SCENE_KEYWORDS["water"]):
        tags.extend(["黄河", "长河"])
    if _contains_any(haystacks, SCENE_KEYWORDS["moon"]):
        tags.extend(["明月", "月色"])
    if _contains_any(haystacks, SCENE_KEYWORDS["snow"]):
        tags.append("雪景")
    if _contains_any(haystacks, SCENE_KEYWORDS["travel"]):
        tags.extend(["旅拍", "远行"])
    if _contains_any(haystacks, SCENE_KEYWORDS["history"]):
        tags.extend(["丝路", "古迹"])
    tags.extend(_extract_hits(haystacks, HEXI_GEO_TERMS))
    tags.extend(_extract_hits(haystacks, HEXI_THEME_TERMS))
    return _dedupe_texts(tags)


def _extract_text_keywords(text: str) -> List[str]:
    hits: List[str] = []
    for keywords in SCENE_KEYWORDS.values():
        for keyword in keywords:
            if keyword and keyword in text and keyword not in hits:
                hits.append(keyword)
    for keyword in HEXI_GEO_TERMS + HEXI_THEME_TERMS:
        if keyword and keyword in text and keyword not in hits:
            hits.append(keyword)
    return hits


def _infer_negative_terms(scene_tags: List[str], emotion: str) -> List[str]:
    negative_terms = list(DEFAULT_NEGATIVE_TERMS)
    tag_set = set(scene_tags)
    if tag_set & {"玉门关", "阳关", "边塞", "丝路", "古迹"}:
        negative_terms.extend(["搞笑配音", "二次元", "鬼畜"])
    if tag_set & {"春风", "杨柳", "明月", "月色", "旅拍"}:
        negative_terms.extend(["商品橱窗", "美妆口播"])
    if tag_set & set(HEXI_GEO_TERMS + HEXI_THEME_TERMS):
        negative_terms.extend(BILIBILI_NEGATIVE_TERMS)
    if emotion in {"豪迈", "苍凉", "悲壮", "思乡"}:
        negative_terms.append("治愈系vlog")
    return _dedupe_texts(negative_terms)


def _build_query_variants(
    text: str,
    location: str,
    imagery_labels: List[str],
    emotion: str,
    scene_tags: List[str],
    keywords: List[str],
    geo_terms: List[str],
    theme_terms: List[str],
) -> List[QueryVariant]:
    normalized_text = _normalize_query(text)
    location_anchor = location or (geo_terms[0] if geo_terms else "河西走廊")
    imagery_terms = _dedupe_texts(imagery_labels + scene_tags + theme_terms)[:3]
    theme_core = _dedupe_texts(theme_terms + scene_tags + keywords)[:4]
    emotion_term = emotion if emotion and emotion not in GENERIC_STOPWORDS else ""
    variants = [
        QueryVariant(name="original", query=normalized_text or location_anchor),
        QueryVariant(
            name="original_plus_imagery",
            query=_join_query_terms([normalized_text, *imagery_terms]),
        ),
        QueryVariant(
            name="theme",
            query=_join_query_terms(theme_core + [emotion_term, *BILIBILI_MEDIA_TERMS[:3]]),
        ),
        QueryVariant(
            name="geo",
            query=_join_query_terms([location_anchor, *theme_core[:2], *BILIBILI_MEDIA_TERMS[:3]]),
        ),
    ]
    deduped: List[QueryVariant] = []
    seen = set()
    for item in variants:
        normalized_query = _normalize_query(item.query)
        if not normalized_query or normalized_query in seen:
            continue
        seen.add(normalized_query)
        deduped.append(QueryVariant(name=item.name, query=normalized_query))
    return deduped


def _normalize_query(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split()).strip()


def _join_query_terms(values: List[str]) -> str:
    cleaned = _dedupe_texts(values)
    return _normalize_query(" ".join(cleaned))


def _extract_hits(haystacks: List[str], keywords: List[str]) -> List[str]:
    hits: List[str] = []
    for keyword in keywords:
        if keyword and any(keyword in hay for hay in haystacks) and keyword not in hits:
            hits.append(keyword)
    return hits


def _normalize_platform_hints(value: Any) -> List[str]:
    items = _dedupe_texts(value)
    normalized = [item for item in items if item in ALLOWED_PLATFORMS]
    return normalized or ["bilibili"]


def _contains_any(haystacks: List[str], keywords: List[str]) -> bool:
    return any(keyword and keyword in hay for hay in haystacks for keyword in keywords)


def _dedupe_texts(value: Any) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    result: List[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result and text not in GENERIC_STOPWORDS:
            result.append(text)
    return result


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return default
