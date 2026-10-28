import json
from importlib import import_module

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from backend.deployment_security import generation_access_allowed, normalized_video_path_allowed
from backend.admin_auth import authenticated_write, require_generation_access, SESSION
from backend.service.agent.tasks import (
    AgentTaskCreateRequest,
    AgentTaskCreateResponse,
    AgentTaskResponse,
    AgentTraceResponse,
    agent_task_service,
)
from backend.service.video_generation.logging_utils import safe_url_preview
from backend.types.schema import (
    HexiPoetryFiltersResponse,
    HexiPoetryPlaceHeatmapResponse,
    HexiPoetryPoemsResponse,
    MainStageComposeRequest,
    MainStageComposeResponse,
    PoetryImageRequest,
    PoetryImageResponse,
    VeoGenerationRequest,
    VeoGenerationResponse,
    VideoSearchRequest,
    VideoSearchResponse,
)


router = APIRouter(prefix="/api")
from backend.reality_runtime import metadata as reality_runtime_metadata
REALITY_RUNTIME = reality_runtime_metadata("backend")


@router.get("/reality/runtime")
def reality_runtime():
    return REALITY_RUNTIME


GENERATION_TOKEN_DESCRIPTION = (
    "Required for realtime generation unless explicitly enabled by deployment environment policy."
)


def _load_service_attr(module_path: str, attr_name: str):
    try:
        module = import_module(module_path)
    except (ModuleNotFoundError, ImportError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"optional service unavailable: {module_path}: {exc}",
        ) from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise HTTPException(status_code=503, detail=f"optional service unavailable: {module_path}.{attr_name}") from exc


@router.post("/video/search", response_model=VideoSearchResponse)
def video_search(req: VideoSearchRequest):
    search_videos = _load_service_attr("backend.service.video_search.reality", "search_reality")
    return search_videos(req)


@router.post("/reality/search")
def reality_search(req: VideoSearchRequest):
    search = _load_service_attr("backend.service.video_search.reality", "search_reality")
    return search(req)


@router.get("/hexi-poetry/filters", response_model=HexiPoetryFiltersResponse)
def hexi_poetry_filters():
    get_hexi_poetry_filters = _load_service_attr("backend.service.hexi_poetry", "get_hexi_poetry_filters")
    return get_hexi_poetry_filters()


@router.get("/hexi-poetry/poems", response_model=HexiPoetryPoemsResponse)
def hexi_poetry_poems(
    dynasty: str = Query(""),
    author: str = Query(""),
    theme: str = Query(""),
    space_type: str = Query(""),
    q: str = Query(""),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    list_hexi_poems = _load_service_attr("backend.service.hexi_poetry", "list_hexi_poems")
    return list_hexi_poems(
        dynasty=dynasty,
        author=author,
        theme=theme,
        space_type=space_type,
        q=q,
        limit=limit,
        offset=offset,
    )


@router.get("/hexi-poetry/place-heatmap", response_model=HexiPoetryPlaceHeatmapResponse)
def hexi_poetry_place_heatmap(
    dynasty: str = Query(""),
    author: str = Query(""),
    theme: str = Query(""),
    space_type: str = Query(""),
    q: str = Query(""),
):
    get_hexi_poetry_place_heatmap = _load_service_attr("backend.service.hexi_poetry", "get_hexi_poetry_place_heatmap")
    return get_hexi_poetry_place_heatmap(
        dynasty=dynasty,
        author=author,
        theme=theme,
        space_type=space_type,
        q=q,
    )


@router.post("/main-stage/compose", response_model=MainStageComposeResponse)
def main_stage_compose(req: MainStageComposeRequest):
    from backend.demo_catalog import find_case, saved_result
    case = find_case(req.text)
    if case and not generation_access_allowed("image"):
        return saved_result(case)
    compose_main_stage = _load_service_attr("backend.service.orchestration", "compose_main_stage")
    return compose_main_stage(req)


@router.post("/agent/tasks", response_model=AgentTaskCreateResponse)
def create_agent_task(req: AgentTaskCreateRequest, request: Request):
    from backend.service.agent.goal_interpreter import interpret_goal
    goal = interpret_goal(requirement=req.task_requirement, requested_target=req.target, poem_text=req.poetry_text)
    if goal.target in {"image", "video"}:
        authenticated_write(request)
    try:
        task = agent_task_service.create_task(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AgentTaskCreateResponse(
        task_id=task.task_id,
        status=task.status,
        target=task.target,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@router.get("/agent/tasks/{task_id}", response_model=AgentTaskResponse)
def get_agent_task(task_id: str):
    task = agent_task_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="agent task not found")
    if task.target in {"image", "video"}:
        require_generation_access(task.target)
    return task


@router.get("/agent/tasks/{task_id}/trace", response_model=AgentTraceResponse)
def get_agent_task_trace(task_id: str):
    task = agent_task_service.get_task(task_id)
    if task and task.target in {"image", "video"}:
        require_generation_access(task.target)
    trace = agent_task_service.get_trace(task_id)
    if not trace:
        raise HTTPException(status_code=404, detail="agent task not found")
    return trace


def _require_image_generation_access(request: Request) -> None:
    authenticated_write(request)
    require_generation_access("image")


def _require_video_generation_access(request: Request) -> None:
    authenticated_write(request)
    require_generation_access("video")


@router.post(
    "/poetry-image/nano-banana",
    response_model=PoetryImageResponse,
    dependencies=[Depends(_require_image_generation_access)],
)
def poetry_image_nano_banana(req: PoetryImageRequest):
    generate_nano_banana_poetry_image = _load_service_attr(
        "backend.service.image_generation",
        "generate_nano_banana_poetry_image",
    )
    return generate_nano_banana_poetry_image(req)


@router.post(
    "/video-generation/poetry-video",
    response_model=VeoGenerationResponse,
    dependencies=[Depends(_require_video_generation_access)],
)
def video_generation_poetry_video(req: VeoGenerationRequest):
    _log_video_route_request("poetry-video", req)
    generate_poetry_video = _load_service_attr("backend.service.video_generation", "generate_poetry_video")
    return generate_poetry_video(req)


@router.post(
    "/video-generation/veo",
    response_model=VeoGenerationResponse,
    dependencies=[Depends(_require_video_generation_access)],
)
def video_generation_veo(req: VeoGenerationRequest):
    _log_video_route_request("veo", req)
    generate_veo_video = _load_service_attr("backend.service.video_generation", "generate_veo_video")
    return generate_veo_video(req)


@router.api_route("/video-generation/proxy", methods=["GET", "HEAD"])
def video_generation_proxy(
    request: Request,
    url: str = Query(..., min_length=1),
    range_header: str = Header("", alias="Range"),
):
    require_generation_access("video")
    proxy_video_request = _load_service_attr("backend.service.video_generation.proxy", "proxy_video_request")
    return proxy_video_request(url, request.method, range_header)


@router.api_route("/video-generation/normalized/{folder_name}/{file_name}", methods=["GET", "HEAD"])
def video_generation_normalized(folder_name: str, file_name: str):
    request_path = f"/api/video-generation/normalized/{folder_name}/{file_name}"
    if not normalized_video_path_allowed(request_path):
        raise HTTPException(status_code=404, detail="normalized video not found")
    serve_normalized_video = _load_service_attr("backend.service.video_generation", "serve_normalized_video")
    return serve_normalized_video(folder_name, file_name)


def _log_video_route_request(endpoint: str, req: VeoGenerationRequest) -> None:
    prediction = req.prediction if isinstance(req.prediction, dict) else {}
    image_url = str(req.image_url or "").strip()
    audio_url = str(req.audio_url or "").strip()
    summary = {
        "endpoint": endpoint,
        "text_present": bool(str(req.text or "").strip()),
        "content_preview": str(req.text or req.narration or "").strip()[:120],
        "image_url_present": bool(image_url),
        "image_url_preview": safe_url_preview(image_url),
        "model": str(prediction.get("model") or prediction.get("video_model") or "").strip(),
        "audio": bool(audio_url) or bool(prediction.get("audio", False)),
        "audio_url_present": bool(audio_url),
        "audio_url_preview": safe_url_preview(audio_url),
        "resolution": str(req.resolution or "").strip(),
        "duration": prediction.get("duration", ""),
    }
    print(f"[video_route] {json.dumps(summary, ensure_ascii=False, default=str)}", flush=True)


def _require_generation_access(kind: str, supplied_token: str) -> None:
    if generation_access_allowed(kind, supplied_token):
        return
    raise HTTPException(
        status_code=403,
        detail="公开演示版禁止实时 AI 生成；请使用精选缓存案例或提供有效管理员令牌。",
    )
