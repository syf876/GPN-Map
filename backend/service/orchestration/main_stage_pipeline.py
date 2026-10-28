import os
from copy import deepcopy
from functools import lru_cache
from threading import RLock
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.service.image_generation.nano_banana import build_nano_banana_prompt
from backend.service.query_planner import build_query_plan
from backend.service.video_generation.poetry_audio import resolve_full_poem
from backend.service.video_generation.prompt_builder import build_storyboard_preview, build_wan_prompt
from backend.service.video_search import search_videos_with_trace
from backend.types.schema import (
    MainStageComposeRequest,
    MainStageComposeResponse,
    MainStageVideoResponse,
    PoetryImageRequest,
    QueryPlan,
    StoryContext,
    StoryNarrativeStep,
    StoryPlace,
    StoryRouteNode,
    StoryRouteTheme,
    VeoGenerationRequest,
    VideoCandidate,
    VideoSearchRequest,
)
from map_round2 import load_place_coordinates, prediction_to_geojson

_PREDICTION_LOCK = RLock()


ROUTE_THEME_PRESETS: Dict[str, Dict[str, Any]] = {
    "silk-road-trade": {
        "name": "丝路商旅路线",
        "summary": "以城镇、关隘与绿洲节点串联河西走廊的商旅流动空间。",
        "nodes": [
            {"name": "兰州", "lng": 103.8343, "lat": 36.0611, "description": "黄河都会，是进入河西走廊的重要东端门户。"},
            {"name": "武威", "lng": 102.6347, "lat": 37.9282, "description": "凉州故地，承担商旅驻泊与转运功能。"},
            {"name": "张掖", "lng": 100.4498, "lat": 38.9259, "description": "绿洲中继地，连接河西中段交通线。"},
            {"name": "酒泉", "lng": 98.5174, "lat": 39.7432, "description": "泉脉绿洲，是商旅西行的重要补给节点。"},
            {"name": "嘉峪关", "lng": 98.2773, "lat": 39.7865, "description": "长城西端重关，是东来西往的重要门槛。"},
            {"name": "敦煌", "lng": 94.6619, "lat": 40.1421, "description": "丝路名城，商旅、使节与文化交流在此汇聚。"},
            {"name": "阳关", "lng": 94.6767, "lat": 39.9398, "description": "古道关津，常作为西行叙事的离别节点。"},
            {"name": "玉门关", "lng": 93.8631, "lat": 40.3603, "description": "通往西域的关键关隘，承接边地与商路想象。"},
        ],
    },
    "frontier-military": {
        "name": "边塞军旅路线",
        "summary": "以关城、军镇与防线节点展开河西走廊的边塞征行叙事。",
        "nodes": [
            {"name": "武威", "lng": 102.6347, "lat": 37.9282, "description": "凉州军镇，自古是河西防务重地。"},
            {"name": "张掖", "lng": 100.4498, "lat": 38.9259, "description": "河西中段控制点，兼具驻防与补给作用。"},
            {"name": "酒泉", "lng": 98.5174, "lat": 39.7432, "description": "西向防线的重要节点，承接军旅交通。"},
            {"name": "嘉峪关", "lng": 98.2773, "lat": 39.7865, "description": "关城雄踞西陲，是边塞诗中的经典坐标。"},
            {"name": "玉门关", "lng": 93.8631, "lat": 40.3603, "description": "古关遗址，承载远征与边塞寒意。"},
            {"name": "阳关", "lng": 94.6767, "lat": 39.9398, "description": "送别与西征叙事常在此展开。"},
        ],
    },
    "oasis-culture": {
        "name": "绿洲人文路线",
        "summary": "串联绿洲城郡、石窟与文化景观，强调河西走廊的人文沉积。",
        "nodes": [
            {"name": "武威", "lng": 102.6347, "lat": 37.9282, "description": "多民族与多文化交汇的走廊城市。"},
            {"name": "张掖", "lng": 100.4498, "lat": 38.9259, "description": "绿洲农业与城镇生活共同塑造日常景观。"},
            {"name": "酒泉", "lng": 98.5174, "lat": 39.7432, "description": "依水而生的城郡，是走廊居民长期生活空间。"},
            {"name": "敦煌", "lng": 94.6619, "lat": 40.1421, "description": "丝路人文汇聚点，宗教、艺术与商旅共存。"},
            {"name": "莫高窟", "lng": 94.8081, "lat": 40.0413, "description": "石窟艺术宝库，体现河西文明深度。"},
            {"name": "月牙泉", "lng": 94.6769, "lat": 40.0877, "description": "沙泉相依，是敦煌绿洲意象的重要景观。"},
        ],
    },
}


FRONTIER_ROUTE_PLACES = {"嘉峪关", "玉门关", "阳关"}
OASIS_ROUTE_PLACES = {"敦煌", "莫高窟", "月牙泉"}

LIANGZHOUCI_DEMO_TRIGGERS = ("羌笛何须怨杨柳", "春风不度玉门关", "玉门关")
JIUQUANZI_DEMO_TITLE = "酒泉子·花映柳条"
JIUQUANZI_DEMO_TITLE_ALIASES = (JIUQUANZI_DEMO_TITLE, "酒泉子・花映柳条")

DEMO_SEMANTIC_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "liangzhouci": {
        "place": {"id": "PLC_YUMENGUAN", "label": "玉门关"},
        "emotion": {"id": "EMO_HOMESICK", "label": "思乡 / 苍凉 / 边塞"},
        "imagery": [
            {"id": "IMG_FRONTIER", "label": "羌笛"},
            {"id": "IMG_NATURE", "label": "杨柳"},
            {"id": "IMG_NATURE", "label": "春风"},
            {"id": "IMG_PASS", "label": "玉门关"},
        ],
    },
    "jiuquanzi_huayingliutiao": {
        "place": {"id": "PLC_JIUQUAN", "label": "酒泉"},
        "emotion": {"id": "EMO_HOMESICK", "label": "思乡 / 春景 / 离愁"},
        "imagery": [
            {"id": "IMG_NATURE", "label": "花"},
            {"id": "IMG_NATURE", "label": "柳条"},
            {"id": "IMG_NATURE", "label": "绿萍"},
            {"id": "IMG_NATURE", "label": "春水"},
        ],
    },
}


@lru_cache(maxsize=1)
def get_predictor():
    if os.getenv("HEXI_SEMANTIC_BACKEND", "corpus") == "transformer":
        from inference_round2 import Round2PoetryPredictor
        return Round2PoetryPredictor()
    from backend.local_predictor import LocalPredictor
    return LocalPredictor()


@lru_cache(maxsize=1)
def get_place_coordinates():
    return load_place_coordinates()


def predict_poem(text: str, imagery_threshold: float) -> Dict[str, object]:
    with _PREDICTION_LOCK:
        return get_predictor().predict(text, imagery_threshold=imagery_threshold)


def build_map_prediction(prediction: Dict[str, object]) -> Dict[str, object]:
    return prediction_to_geojson(prediction, get_place_coordinates())


def compose_main_stage(req: MainStageComposeRequest) -> MainStageComposeResponse:
    prediction = predict_poem(req.text, imagery_threshold=req.imagery_threshold)
    prediction = _apply_demo_semantic_override(req.text, prediction)
    map_json = build_map_prediction(prediction)
    # Real-world reference retrieval uses local query rules and free public sources.
    from backend.service.query_planner.qwen_planner import QwenQueryPlanner
    from backend.service.video_search.reality import search_reality

    planner = QwenQueryPlanner(disable_remote=True).plan(prediction)
    video_request = _build_video_request(prediction, planner)
    reference = search_reality(video_request)
    video = MainStageVideoResponse(**reference)
    story_context = _build_story_context(req, prediction, map_json, planner, reference["query"])
    for step in story_context.narrative_steps:
        if step.id == "reference_video":
            step.status = "available" if reference["ok"] else reference["status"]
            step.summary = reference["message"]
    return MainStageComposeResponse(
        ok=True,
        prediction=prediction,
        map=map_json,
        planner=planner,
        video=video,
        storyContext=story_context,
    )


def _apply_demo_semantic_override(input_text: str, prediction: Dict[str, object]) -> Dict[str, object]:
    override_key = _demo_semantic_override_key(input_text)
    if not override_key:
        return prediction
    override = DEMO_SEMANTIC_OVERRIDES[override_key]
    result = deepcopy(prediction)
    result["place"] = _override_single_label(
        result.get("place"),
        str(override["place"]["id"]),
        str(override["place"]["label"]),
    )
    result["emotion"] = _override_single_label(
        result.get("emotion"),
        str(override["emotion"]["id"]),
        str(override["emotion"]["label"]),
    )
    result["imagery"] = _override_imagery(result.get("imagery"), list(override["imagery"]))
    confidence = result.get("confidence")
    if isinstance(confidence, dict):
        confidence["place"] = 1.0
        confidence["emotion"] = 1.0
        confidence["imagery"] = 1.0
    result["demo_semantic_override"] = override_key
    return result


def _demo_semantic_override_key(input_text: str) -> str:
    text = str(input_text or "").strip()
    if any(token in text for token in LIANGZHOUCI_DEMO_TRIGGERS):
        return "liangzhouci"
    if any(title in text for title in JIUQUANZI_DEMO_TITLE_ALIASES):
        return "jiuquanzi_huayingliutiao"
    poem_match = resolve_full_poem(text)
    if poem_match.title and any(title == poem_match.title for title in JIUQUANZI_DEMO_TITLE_ALIASES):
        return "jiuquanzi_huayingliutiao"
    return ""


def _override_single_label(node: Any, label_id: str, label: str) -> Dict[str, Any]:
    base = dict(node) if isinstance(node, dict) else {}
    probabilities = base.get("probabilities")
    if not isinstance(probabilities, dict):
        probabilities = {}
    probabilities = {str(key): float(value or 0.0) for key, value in probabilities.items()}
    probabilities[label_id] = 1.0
    base.update(
        {
            "id": label_id,
            "label": label,
            "confidence": 1.0,
            "probabilities": probabilities,
        }
    )
    return base


def _override_imagery(node: Any, labels: List[Dict[str, str]]) -> Dict[str, Any]:
    base = dict(node) if isinstance(node, dict) else {}
    probabilities = base.get("probabilities")
    if not isinstance(probabilities, dict):
        probabilities = {}
    override_labels = []
    for index, item in enumerate(labels):
        label_id = str(item.get("id") or "").strip()
        label = str(item.get("label") or label_id).strip()
        confidence = round(max(0.99 - index * 0.01, 0.9), 4)
        if label_id:
            probabilities[label_id] = max(float(probabilities.get(label_id) or 0.0), confidence)
        override_labels.append({"id": label_id, "label": label, "confidence": confidence})
    base.update(
        {
            "labels": override_labels,
            "confidence": 1.0,
            "probabilities": probabilities,
        }
    )
    return base


def _build_video_request(prediction: Dict[str, object], planner: QueryPlan) -> VideoSearchRequest:
    imagery_labels = [
        str(item.get("label") or item.get("id") or "").strip()
        for item in prediction.get("imagery", {}).get("labels", [])
        if str(item.get("label") or item.get("id") or "").strip()
    ]
    return VideoSearchRequest(
        poem=str(prediction.get("text", "")).strip(),
        location=str(
            prediction.get("place", {}).get("label")
            or prediction.get("place", {}).get("id")
            or ""
        ).strip(),
        imagery=imagery_labels,
        emotion=str(
            prediction.get("emotion", {}).get("label")
            or prediction.get("emotion", {}).get("id")
            or ""
        ).strip(),
        keywords=list(planner.keywords),
        scene_tags=list(planner.scene_tags),
        negative_terms=list(planner.negative_terms),
        query=planner.query,
        query_variants=list(planner.query_variants),
        platform_hints=list(planner.platform_hints),
    )


def _selected_index(
    candidates: List[VideoCandidate],
    best: Optional[VideoCandidate],
) -> int:
    if not candidates or best is None:
        return 0
    for index, item in enumerate(candidates):
        if item.platform == best.platform and item.title == best.title and item.url == best.url:
            return index
    return 0


def _main_stage_platform_modes() -> Dict[str, str]:
    return {
        "bilibili": "fixture" if _env_truthy("BILIBILI_FIXTURE_MODE") else "real",
    }


def _build_story_context(
    req: MainStageComposeRequest,
    prediction: Dict[str, object],
    map_json: Dict[str, object],
    planner: QueryPlan,
    reference_query: str,
) -> StoryContext:
    poem_match = resolve_full_poem(str(req.text or "").strip())
    input_text = str(req.text or "").strip()
    full_poem = poem_match.poem_text or input_text
    title = poem_match.title
    author = poem_match.author
    dynasty = poem_match.dynasty or _prediction_label(prediction.get("dynasty"))
    imagery = _prediction_imagery(prediction)
    emotion = _prediction_label(prediction.get("emotion"))
    primary_place = _primary_place(prediction, map_json)
    candidate_places = _candidate_places(prediction, primary_place)
    route_theme = _resolve_route_theme(primary_place, imagery, planner)
    route_nodes = _build_route_nodes(route_theme.id)
    image_request = PoetryImageRequest(
        poem=full_poem or input_text,
        place=primary_place.label,
        dynasty=dynasty,
        emotion=emotion,
        imagery=imagery,
    )
    video_request = VeoGenerationRequest(
        text=full_poem or input_text,
        location=primary_place.label,
        dynasty=dynasty,
        emotion=emotion,
        imagery=imagery,
        narration=full_poem or input_text,
        prediction=prediction,
    )
    narrative_steps = _build_narrative_steps(
        poem_match=poem_match,
        dynasty=dynasty,
        primary_place=primary_place,
        route_theme=route_theme,
        route_nodes=route_nodes,
        imagery=imagery,
        emotion=emotion,
        reference_query=reference_query,
    )
    return StoryContext(
        input_text=input_text,
        full_poem=full_poem or input_text,
        title=title,
        author=author,
        dynasty=dynasty,
        primary_place=primary_place,
        candidate_places=candidate_places,
        imagery=imagery,
        emotion=emotion,
        route_theme=route_theme,
        route_nodes=route_nodes,
        narrative_steps=narrative_steps,
        image_prompt_context=build_nano_banana_prompt(image_request),
        video_prompt_context=build_wan_prompt(video_request),
        storyboard_preview=_build_storyboard_preview_payload(video_request),
    )


def _primary_place(prediction: Dict[str, object], map_json: Dict[str, object]) -> StoryPlace:
    place = prediction.get("place", {}) if isinstance(prediction, dict) else {}
    features = map_json.get("features", []) if isinstance(map_json, dict) else []
    first_feature = features[0] if isinstance(features, list) and features else {}
    coords = first_feature.get("properties", {}).get("place", {}).get("coordinates", {}) if isinstance(first_feature, dict) else {}
    return StoryPlace(
        id=str(place.get("id") or "").strip(),
        label=str(place.get("label") or place.get("id") or "").strip(),
        lng=float(coords.get("lon") or 0.0),
        lat=float(coords.get("lat") or 0.0),
        confidence=float(place.get("confidence") or 0.0),
    )


def _candidate_places(prediction: Dict[str, object], primary_place: StoryPlace, limit: int = 4) -> List[StoryPlace]:
    place = prediction.get("place", {}) if isinstance(prediction, dict) else {}
    probabilities = place.get("probabilities", {}) if isinstance(place, dict) else {}
    coords_by_place = get_place_coordinates()
    place_meta = get_predictor().place_meta
    items: List[StoryPlace] = []
    for place_id, confidence in sorted(probabilities.items(), key=lambda item: float(item[1] or 0.0), reverse=True):
        label = str(place_meta.get(place_id, {}).get("zh") or place_id).strip()
        coords = coords_by_place.get(label, {})
        items.append(
            StoryPlace(
                id=str(place_id).strip(),
                label=label,
                lng=float(coords.get("lon") or 0.0),
                lat=float(coords.get("lat") or 0.0),
                confidence=float(confidence or 0.0),
            )
        )
        if len(items) >= limit:
            break
    if not items and primary_place.label:
        items.append(primary_place)
    return items


def _resolve_route_theme(
    primary_place: StoryPlace,
    imagery: List[str],
    planner: QueryPlan,
) -> StoryRouteTheme:
    keywords = " ".join(
        [
            primary_place.label,
            *imagery,
            *list(planner.scene_tags),
            *list(planner.keywords),
        ]
    )
    if primary_place.label in FRONTIER_ROUTE_PLACES or any(token in keywords for token in ("边塞", "关", "羌笛", "军", "征")):
        theme_id = "frontier-military"
        reason = f"基于主地点 {primary_place.label or '未识别'} 与边塞/关隘语义线索匹配边塞军旅路线。"
    elif primary_place.label in OASIS_ROUTE_PLACES or any(token in keywords for token in ("石窟", "飞天", "绿洲", "月牙泉", "敦煌")):
        theme_id = "oasis-culture"
        reason = f"基于主地点 {primary_place.label or '未识别'} 与绿洲/人文线索匹配绿洲人文路线。"
    else:
        theme_id = "silk-road-trade"
        reason = "基于当前地点、场景标签与检索关键词，优先采用丝路商旅路线承接空间叙事。"
    preset = ROUTE_THEME_PRESETS[theme_id]
    return StoryRouteTheme(
        id=theme_id,
        name=str(preset["name"]),
        summary=str(preset["summary"]),
        reason=reason,
    )


def _build_route_nodes(theme_id: str) -> List[StoryRouteNode]:
    preset = ROUTE_THEME_PRESETS.get(theme_id, {})
    nodes = preset.get("nodes", [])
    return [
        StoryRouteNode(
            name=str(node.get("name") or "").strip(),
            lng=float(node.get("lng") or 0.0),
            lat=float(node.get("lat") or 0.0),
            description=str(node.get("description") or "").strip(),
        )
        for node in nodes
    ]


def _build_narrative_steps(
    *,
    poem_match,
    dynasty: str,
    primary_place: StoryPlace,
    route_theme: StoryRouteTheme,
    route_nodes: List[StoryRouteNode],
    imagery: List[str],
    emotion: str,
    reference_query: str,
) -> List[StoryNarrativeStep]:
    poem_summary = (
        f"已匹配《{poem_match.title}》 {poem_match.author}，可沿完整诗作展开。"
        if poem_match.matched and poem_match.title
        else "未匹配到整首诗，当前按输入诗句继续后续叙事。"
    )
    dynasty_summary = (
        f"当前故事以 {dynasty} 语境推进。"
        if dynasty
        else "当前仅保留模型输出的语义结果，暂缺完整朝代背景。"
    )
    place_summary = (
        f"主地点定位到 {primary_place.label}。"
        if primary_place.label
        else "当前未获得明确主地点，只能保留河西走廊范围叙事。"
    )
    route_summary = (
        f"推荐路线主题为“{route_theme.name}”，节点数 {len(route_nodes)}。"
        if route_theme.id
        else "当前未分配路线主题。"
    )
    imagery_summary = (
        f"意象 {', '.join(imagery[:4]) or '暂缺'}，情绪主调为 {emotion or '暂缺'}。"
    )
    reference_summary = (
        f"现实影像参考检索 query: {reference_query}。"
        if reference_query
        else "现实影像参考将沿已有检索策略继续执行。"
    )
    return [
        StoryNarrativeStep(id="poem_identification", title="诗词识别", summary=poem_summary, status="resolved" if poem_match.matched else "input_only"),
        StoryNarrativeStep(id="poet_dynasty_context", title="诗人/朝代背景", summary=dynasty_summary, status="available" if dynasty else "input_only"),
        StoryNarrativeStep(id="hexi_spatial_locating", title="河西走廊空间定位", summary=place_summary, status="available" if primary_place.label else "input_only"),
        StoryNarrativeStep(id="route_expansion", title="地点与路线展开", summary=route_summary, status="available" if route_theme.id else "input_only"),
        StoryNarrativeStep(id="imagery_emotion", title="意象与情感解释", summary=imagery_summary, status="available"),
        StoryNarrativeStep(id="image_generation", title="意境图生成", summary="已生成统一图像提示词上下文，沿当前故事语境生成意境图。", status="available"),
        StoryNarrativeStep(id="ai_video_generation", title="AI叙事视频生成", summary="已生成统一视频提示词上下文，沿完整诗句与空间叙事生成视频。", status="available"),
        StoryNarrativeStep(id="reference_video", title="现实影像参考", summary=reference_summary, status="available"),
    ]


def _prediction_label(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    return str(node.get("label") or node.get("id") or "").strip()


def _build_storyboard_preview_payload(video_request: VeoGenerationRequest) -> Dict[str, Any]:
    try:
        return build_storyboard_preview(video_request)
    except Exception as exc:
        return {
            "narration_text": "",
            "shots": [],
            "source": f"storyContext fallback: storyboard preview unavailable ({exc})",
        }


def _prediction_imagery(prediction: Dict[str, object]) -> List[str]:
    imagery = prediction.get("imagery", {}) if isinstance(prediction, dict) else {}
    labels = imagery.get("labels", []) if isinstance(imagery, dict) else []
    result: List[str] = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("id") or "").strip()
        if label and label not in result:
            result.append(label)
    return result


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
