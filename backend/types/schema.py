from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in (_clean_text(item) for item in value) if item]

    text = _clean_text(value)
    if not text:
        return []

    for sep in (",", chr(0xFF0C), chr(0x3001), ";", chr(0xFF1B), "|"):
        text = text.replace(sep, ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def _coerce_query_variants(value: Any) -> List[Dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]

    result: List[Dict[str, str]] = []
    for index, item in enumerate(value):
        if hasattr(item, "model_dump"):
            payload = item.model_dump()
            name = _clean_text(payload.get("name") or payload.get("round") or payload.get("label") or f"round_{index + 1}")
            query = _clean_text(payload.get("query") or payload.get("text") or payload.get("value"))
        elif hasattr(item, "dict"):
            payload = item.dict()
            name = _clean_text(payload.get("name") or payload.get("round") or payload.get("label") or f"round_{index + 1}")
            query = _clean_text(payload.get("query") or payload.get("text") or payload.get("value"))
        elif hasattr(item, "name") or hasattr(item, "query"):
            name = _clean_text(getattr(item, "name", "") or f"round_{index + 1}")
            query = _clean_text(getattr(item, "query", ""))
        elif isinstance(item, dict):
            name = _clean_text(
                item.get("name")
                or item.get("round")
                or item.get("label")
                or f"round_{index + 1}"
            )
            query = _clean_text(item.get("query") or item.get("text") or item.get("value"))
        else:
            name = f"round_{index + 1}"
            query = _clean_text(item)
        if query:
            result.append({"name": name or f"round_{index + 1}", "query": query})
    return result


class QueryVariant(BaseModel):
    name: str = Field(..., description="query round name")
    query: str = Field(..., description="query text")

    @validator("name", "query", pre=True)
    def normalize_variant_text(cls, value: Any) -> str:
        return _clean_text(value)


class VideoSearchRequest(BaseModel):
    poem: str = Field("", description="input poem line")
    location: str = Field("", description="predicted location")
    imagery: List[str] = Field(default_factory=list, description="imagery labels")
    emotion: str = Field("", description="emotion label")
    keywords: List[str] = Field(default_factory=list, description="search keywords")
    scene_tags: List[str] = Field(default_factory=list, description="scene tags")
    negative_terms: List[str] = Field(default_factory=list, description="negative terms")
    query: str = Field("", description="primary query")
    query_variants: List[QueryVariant] = Field(default_factory=list, description="multi-round queries")
    platform_hints: List[str] = Field(default_factory=list, description="platform order hints")

    @validator("poem", "location", "emotion", "query", pre=True)
    def normalize_text(cls, value: Any) -> str:
        return _clean_text(value)

    @validator("imagery", "keywords", "scene_tags", "negative_terms", "platform_hints", pre=True)
    def normalize_list(cls, value: Any) -> List[str]:
        return _coerce_text_list(value)

    @validator("query_variants", pre=True)
    def normalize_query_variants(cls, value: Any) -> List[Dict[str, str]]:
        return _coerce_query_variants(value)


class VideoCandidate(BaseModel):
    platform: str = Field(..., description="video platform")
    title: str
    url: str
    cover: str
    author: str
    score: float = Field(..., ge=0.0, le=1.0)
    bvid: str = Field("", description="bilibili BV id")
    video_url: str = Field("", description="direct playable media url")
    embed_url: str = Field("", description="iframe embed url")
    external_url: str = Field("", description="external page url")
    playback_type: str = Field("", description="video_url/embed_url/cover/external")
    reason: str = Field("", description="candidate reason")
    tags: List[str] = Field(default_factory=list, description="matched tags")
    source_mode: str = Field("", description="real_http/fixture/mock")
    license: str = ""
    location: str = ""
    source_note: str = Field("", description="extra source note")
    media_playability_score: float = Field(0.0, ge=0.0, le=1.0)
    duplicate_key: str = Field("", description="dedupe key")

    @validator(
        "platform",
        "title",
        "url",
        "cover",
        "author",
        "bvid",
        "video_url",
        "embed_url",
        "external_url",
        "playback_type",
        "reason",
        "source_mode",
        "source_note",
        "duplicate_key",
        pre=True,
    )
    def normalize_candidate_text(cls, value: Any) -> str:
        return _clean_text(value)

    @validator("tags", pre=True)
    def normalize_candidate_tags(cls, value: Any) -> List[str]:
        return _coerce_text_list(value)


class VideoSearchResponse(BaseModel):
    ok: bool
    status: str = ""
    message: str = ""
    warnings: List[str] = Field(default_factory=list)
    query: str
    best: Optional[VideoCandidate]
    candidates: List[VideoCandidate]
    best_reason: str = ""
    priority_hit: str = ""
    candidate_source_distribution: Dict[str, int] = Field(default_factory=dict)


class MainStageComposeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="input poem text")
    imagery_threshold: float = Field(0.5, ge=0.0, le=1.0)

    @validator("text", pre=True)
    def normalize_stage_text(cls, value: Any) -> str:
        return _clean_text(value)


class QueryPlan(BaseModel):
    provider: str = Field("", description="query planner provider")
    mode: str = Field("", description="remote/fallback")
    query: str = Field("", description="primary query")
    query_variants: List[QueryVariant] = Field(default_factory=list, description="ordered multi-round queries")
    keywords: List[str] = Field(default_factory=list, description="extracted keywords")
    negative_terms: List[str] = Field(default_factory=list, description="negative terms")
    platform_hints: List[str] = Field(default_factory=list, description="recommended platform order")
    scene_tags: List[str] = Field(default_factory=list, description="scene tags")
    reasoning: str = Field("", description="planner reasoning")

    @validator("provider", "mode", "query", "reasoning", pre=True)
    def normalize_plan_text(cls, value: Any) -> str:
        return _clean_text(value)

    @validator("keywords", "negative_terms", "platform_hints", "scene_tags", pre=True)
    def normalize_plan_list(cls, value: Any) -> List[str]:
        return _coerce_text_list(value)

    @validator("query_variants", pre=True)
    def normalize_plan_query_variants(cls, value: Any) -> List[Dict[str, str]]:
        return _coerce_query_variants(value)


class MainStageVideoResponse(BaseModel):
    ok: bool
    status: str = ""
    message: str = ""
    warnings: List[str] = Field(default_factory=list)
    query: str
    best: Optional[VideoCandidate]
    candidates: List[VideoCandidate]
    best_reason: str = ""
    priority_hit: str = ""
    candidate_source_distribution: Dict[str, int] = Field(default_factory=dict)
    candidate_count: int = 0
    selected_index: int = 0


class StoryPlace(BaseModel):
    id: str = Field("", description="place label id")
    label: str = Field("", description="place display label")
    lng: float = Field(0.0, description="longitude")
    lat: float = Field(0.0, description="latitude")
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    @validator("id", "label", pre=True)
    def normalize_story_place_text(cls, value: Any) -> str:
        return _clean_text(value)


class StoryRouteTheme(BaseModel):
    id: str = Field("", description="route theme id")
    name: str = Field("", description="route theme name")
    summary: str = Field("", description="route theme summary")
    reason: str = Field("", description="route theme selection reason")

    @validator("id", "name", "summary", "reason", pre=True)
    def normalize_story_route_theme_text(cls, value: Any) -> str:
        return _clean_text(value)


class StoryRouteNode(BaseModel):
    name: str = Field("", description="route node name")
    lng: float = Field(0.0, description="longitude")
    lat: float = Field(0.0, description="latitude")
    description: str = Field("", description="route node description")

    @validator("name", "description", pre=True)
    def normalize_story_route_node_text(cls, value: Any) -> str:
        return _clean_text(value)


class StoryNarrativeStep(BaseModel):
    id: str = Field("", description="story step id")
    title: str = Field("", description="story step title")
    summary: str = Field("", description="story step summary")
    status: str = Field("", description="resolved/input_only/available")

    @validator("id", "title", "summary", "status", pre=True)
    def normalize_story_step_text(cls, value: Any) -> str:
        return _clean_text(value)


class StoryboardPreviewShot(BaseModel):
    order: int = Field(0, ge=0, description="shot order")
    scene: str = Field("", description="shot scene summary")
    visual_prompt: str = Field("", description="resolved per-shot visual prompt")
    duration: float = Field(0.0, ge=0.0, description="estimated shot duration in seconds")
    place: str = Field("", description="shot place anchor")
    imagery: List[str] = Field(default_factory=list, description="shot imagery labels")
    emotion: str = Field("", description="shot emotion")

    @validator("scene", "visual_prompt", "place", "emotion", pre=True)
    def normalize_storyboard_preview_shot_text(cls, value: Any) -> str:
        return _clean_text(value)

    @validator("imagery", pre=True)
    def normalize_storyboard_preview_shot_imagery(cls, value: Any) -> List[str]:
        return _coerce_text_list(value)


class StoryboardPreview(BaseModel):
    narration_text: str = Field("", description="narration text used for preview")
    shots: List[StoryboardPreviewShot] = Field(default_factory=list)
    source: str = Field("", description="preview source descriptor")

    @validator("narration_text", "source", pre=True)
    def normalize_storyboard_preview_text(cls, value: Any) -> str:
        return _clean_text(value)


class StoryContext(BaseModel):
    input_text: str = Field("", description="original input poem text")
    full_poem: str = Field("", description="resolved full poem text")
    title: str = Field("", description="resolved poem title")
    author: str = Field("", description="resolved poem author")
    dynasty: str = Field("", description="resolved or predicted dynasty")
    primary_place: StoryPlace = Field(default_factory=StoryPlace)
    candidate_places: List[StoryPlace] = Field(default_factory=list)
    imagery: List[str] = Field(default_factory=list, description="imagery labels")
    emotion: str = Field("", description="emotion label")
    route_theme: StoryRouteTheme = Field(default_factory=StoryRouteTheme)
    route_nodes: List[StoryRouteNode] = Field(default_factory=list)
    narrative_steps: List[StoryNarrativeStep] = Field(default_factory=list)
    image_prompt_context: str = Field("", description="resolved image prompt context")
    video_prompt_context: str = Field("", description="resolved video prompt context")
    storyboard_preview: StoryboardPreview = Field(default_factory=StoryboardPreview)

    @validator(
        "input_text",
        "full_poem",
        "title",
        "author",
        "dynasty",
        "emotion",
        "image_prompt_context",
        "video_prompt_context",
        pre=True,
    )
    def normalize_story_context_text(cls, value: Any) -> str:
        return _clean_text(value)

    @validator("imagery", pre=True)
    def normalize_story_context_imagery(cls, value: Any) -> List[str]:
        return _coerce_text_list(value)


class MainStageComposeResponse(BaseModel):
    ok: bool
    prediction: Dict[str, Any]
    map: Dict[str, Any]
    planner: QueryPlan
    video: MainStageVideoResponse
    storyContext: StoryContext

    @validator("storyContext", pre=True)
    def normalize_story_context_alias(cls, value: Any) -> Any:
        return value or {}


class VeoGenerationRequest(BaseModel):
    text: str = Field("", description="poem text")
    location: str = Field("", description="predicted or requested location")
    dynasty: str = Field("", description="predicted dynasty")
    imagery: List[str] = Field(default_factory=list, description="imagery labels")
    emotion: str = Field("", description="predicted emotion")
    image_url: str = Field("", description="generated poetry image url for image-to-video")
    audio_url: str = Field("", description="optional driving audio url for audio-driven video models")
    narration: str = Field("", description="optional narration text for future text-to-speech bridge")
    resolution: str = Field("", description="requested output resolution, e.g. 480P/720P")
    prediction: Dict[str, Any] = Field(default_factory=dict, description="optional compose prediction payload")

    @validator("text", "location", "dynasty", "emotion", "image_url", "audio_url", "narration", "resolution", pre=True)
    def normalize_veo_text(cls, value: Any) -> str:
        return _clean_text(value)

    @validator("imagery", pre=True)
    def normalize_veo_imagery(cls, value: Any) -> List[str]:
        return _coerce_text_list(value)


class VeoGenerationResponse(BaseModel):
    status: str = Field("", description="ready/failed")
    provider: str = Field("", description="jimeng/veo/wan")
    id: str = Field("", description="generation task id")
    video_url: str = Field("", description="generated video url")
    normalized_video_url: str = Field("", description="browser-compatible normalized video url")
    progress: float = Field(0.0, ge=0.0, le=100.0)
    prompt: str = Field("", description="resolved veo prompt")
    error: str = Field("", description="failure reason")
    error_detail: Dict[str, Any] = Field(default_factory=dict, description="structured provider error detail")
    raw_status: str = Field("", description="provider raw status")

    @validator("status", "provider", "id", "video_url", "normalized_video_url", "prompt", "error", "raw_status", pre=True)
    def normalize_veo_response_text(cls, value: Any) -> str:
        return _clean_text(value)


class PoetryImageRequest(BaseModel):
    poem: str = Field(..., min_length=1, description="input poem text")
    title: str = ""
    author: str = ""
    content: str = ""
    places: List[str] = Field(default_factory=list)
    analysis_text: str = ""
    place: str = Field("", description="predicted or requested place")
    dynasty: str = Field("", description="predicted dynasty")
    emotion: str = Field("", description="predicted emotion")
    imagery: List[str] = Field(default_factory=list, description="imagery labels")
    mode: str = Field("", description="optional image prompt mode, e.g. safe_i2v_first_frame")

    @validator("poem", "place", "dynasty", "emotion", "mode", pre=True)
    def normalize_poetry_image_text(cls, value: Any) -> str:
        return _clean_text(value)

    @validator("imagery", pre=True)
    def normalize_poetry_image_imagery(cls, value: Any) -> List[str]:
        return _coerce_text_list(value)


class PoetryImageResponse(BaseModel):
    image_url: str = Field("", description="generated image url")
    normalized_image_url: str = Field("", description="persistent local image url")
    prompt: str = Field("", description="resolved prompt")
    provider: str = Field("", description="image provider name")
    model: str = Field("", description="image generation model")
    status: str = Field("", description="succeeded/failed/processing")
    error: str = Field("", description="failure reason")

    @validator("image_url", "prompt", "provider", "model", "status", "error", pre=True)
    def normalize_poetry_image_response_text(cls, value: Any) -> str:
        return _clean_text(value)


class HexiPoetryFilterOption(BaseModel):
    value: str = Field("", description="filter option value")
    count: int = Field(0, ge=0, description="matching poem count")

    @validator("value", pre=True)
    def normalize_filter_option_value(cls, value: Any) -> str:
        return _clean_text(value)


class HexiPoetryFiltersResponse(BaseModel):
    total_poems: int = Field(0, ge=0)
    dynasties: List[HexiPoetryFilterOption] = Field(default_factory=list)
    authors: List[HexiPoetryFilterOption] = Field(default_factory=list)
    themes: List[HexiPoetryFilterOption] = Field(default_factory=list)
    space_types: List[HexiPoetryFilterOption] = Field(default_factory=list)


class HexiPoetryPoemItem(BaseModel):
    title: str = Field("", description="poem title")
    author: str = Field("", description="poem author")
    dynasty: str = Field("", description="poem dynasty")
    content: str = Field("", description="poem content")
    matched_theme: str = Field("", description="keyword rule matched theme")
    matched_space_type: str = Field("", description="keyword rule matched space type")

    @validator("title", "author", "dynasty", "content", "matched_theme", "matched_space_type", pre=True)
    def normalize_poem_item_text(cls, value: Any) -> str:
        return _clean_text(value)


class HexiPoetryPoemsResponse(BaseModel):
    total: int = Field(0, ge=0)
    limit: int = Field(0, ge=0)
    offset: int = Field(0, ge=0)
    items: List[HexiPoetryPoemItem] = Field(default_factory=list)


class HexiPoetryPlaceHeatmapItem(BaseModel):
    name: str = Field("", description="canonical place name")
    lng: float = Field(0.0, description="longitude")
    lat: float = Field(0.0, description="latitude")
    count: int = Field(0, ge=0, description="total place mention count")
    poems_count: int = Field(0, ge=0, description="number of poems mentioning the place")

    @validator("name", pre=True)
    def normalize_heatmap_item_name(cls, value: Any) -> str:
        return _clean_text(value)


class HexiPoetryPlaceHeatmapResponse(BaseModel):
    total_poems: int = Field(0, ge=0)
    skipped_count: int = Field(0, ge=0)
    items: List[HexiPoetryPlaceHeatmapItem] = Field(default_factory=list)
