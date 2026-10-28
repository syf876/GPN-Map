from backend.admin_auth import require_generation_access
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from backend.types.schema import PoetryImageRequest, PoetryImageResponse


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
GRSAI_DEFAULT_HOST = "https://grsai.dakka.com.cn"
GRSAI_NANO_BANANA_MODEL = "nano-banana-pro"
GRSAI_NANO_BANANA_SUBMIT_PATH = "/v1/draw/nano-banana"
GRSAI_NANO_BANANA_RESULT_PATH = "/v1/draw/result"
GRSAI_HOST_ENV_NAMES = (
    "GRSAI_NANO_BANANA_BASE_URL",
    "GRSAI_NANO_BANANA_HOST",
    "NANO_BANANA_BASE_URL",
    "NANO_BANANA_HOST",
    "GRSAI_BASE_URL",
    "GRSAI_API_BASE",
)
GRSAI_API_KEY_ENV_NAMES = (
    "GRSAI_NANO_BANANA_API_KEY",
    "NANO_BANANA_API_KEY",
    "GRSAI_API_KEY",
)
SUCCESS_STATUSES = {"succeeded", "success", "completed", "complete", "done", "finished", "ready"}
FAILURE_STATUSES = {"failed", "failure", "error", "cancelled", "canceled", "timeout", "rejected"}
WAITING_STATUSES = {"pending", "running", "processing", "queued", "queueing", "in_queue", "created", "submitted", "generating"}
SUCCESS_CODES = {"0", "200", "success", "succeeded", "completed", "complete", "done", "finished", "ready", "ok", "true"}
NANO_BANANA_NEGATIVE_PROMPT = (
    "文字，标题排版，水印，logo，地图，边界线，坐标轴，图例，信息图，UI界面，现代广告海报，"
    "vertical poster, portrait layout, 9:16 poster, single person poster, phone wallpaper, narrow vertical composition, "
    "竖版海报，9:16，单人物竖屏海报，手机壁纸构图，狭窄纵向构图，真实摄影感，3D渲染，"
    "正面人物，士兵，武器，盔甲，战争，血腥，冲突，旗帜，火焰，攻击动作，低清晰度，模糊，肢体畸形，手指异常，构图混乱。"
)


@dataclass(frozen=True)
class NanoBananaConfig:
    provider: str
    host: str
    api_key: str
    model: str
    generator: str
    image_size: str
    aspect_ratio: str
    dashscope_api_root: str
    env_file: str
    host_source: str
    api_key_source: str
    poll_max_attempts: int
    poll_interval_seconds: float
    poll_timeout_seconds: float


def generate_nano_banana_poetry_image(req: PoetryImageRequest) -> PoetryImageResponse:
    require_generation_access()
    prompt = build_nano_banana_prompt(req)
    negative_prompt = build_nano_banana_negative_prompt()
    config = _load_config()
    _log_image_event(
        "start",
        provider=config.provider,
        model=config.model,
        generator=config.generator,
        host=config.host,
        env_file=config.env_file,
        host_source=config.host_source,
        api_key_present=bool(config.api_key),
        api_key_source=config.api_key_source,
        task_id="",
        status="",
        raw_result_keys=[],
    )

    try:
        result = _generate_real_image(prompt, negative_prompt, config)
        if result.image_url and _normalize_status(result.status) in SUCCESS_STATUSES:
            from .storage import persist_image
            try:
                local_url = persist_image(result.image_url, req, result)
                result = result.copy(update={"normalized_image_url": local_url})
            except Exception as exc:
                _log_image_event("save_failed", error_type=type(exc).__name__, error=_redact_secret(str(exc), config))
            return result
        return _failure_response(
            prompt,
            config,
            result.error or f"no image url returned by provider; status={result.status or 'unknown'}",
            status=result.status or "failed",
        )
    except Exception as exc:
        return _failure_response(prompt, config, f"{exc.__class__.__name__}: {exc}", status="failed")


def build_nano_banana_prompt(req: PoetryImageRequest) -> str:
    poem = _request_text(req, "poem")
    title = _request_text(req, "title")
    author = _request_text(req, "author")
    dynasty = _request_text(req, "dynasty")
    content = _request_text(req, "content") or poem
    places = _request_text(req, "places") or _request_text(req, "place")
    emotion = _request_text(req, "emotion")
    analysis_text = _request_text(req, "analysis_text", _request_text(req, "analysisText"))
    imagery = [str(item).strip() for item in req.imagery if str(item).strip()]
    imagery_text = "、".join(imagery[:6])
    mode = _request_text(req, "mode").lower()
    if mode == "safe_i2v_first_frame":
        return _build_safe_i2v_first_frame_prompt(
            title=title,
            author=author,
            dynasty=dynasty,
            content=content,
            places=places,
            imagery_text=imagery_text,
            emotion=emotion,
            analysis_text=analysis_text,
        )

    lines = [
        "请根据以下诗词分析生成一张横版“诗词意境图”主舞台画面。",
        "16:9 horizontal landscape composition, wide cinematic frame, suitable for website hero stage, not vertical poster, not portrait layout, no text, no watermark.",
        "风格要求：国风、水墨与淡彩结合、诗意意境插画、非写实摄影、留白克制、适合网页宽屏主舞台预览。",
        "构图要求：横版宽屏构图，16:9，宽幅电影画面，铺满主舞台，主体清晰，远中近景完整，河西走廊边塞空间横向展开；不要竖版人物海报、不要肖像/竖屏布局、不要手机壁纸构图、不要标题排版、不要文字说明、不要地图、不要边界线、不要图例、不要 UI。",
        f"title: {title}",
        f"author: {author}",
        f"dynasty: {dynasty}",
        f"content: {content}",
        f"places: {places}",
        f"imagery: {imagery_text}",
        f"emotion: {emotion}",
        f"analysis_text: {analysis_text}",
        "请把地点、意象和情绪融合为同一幅中国古典诗意画面，突出河西走廊相关自然地貌与边塞气质；关隘、沙漠、远山、古道、云影、风沙等横向空间层次要清晰，画面适配网页主舞台宽屏展示，但不要做成地图或海报。",
        "边塞诗安全构图要求：即使是《凉州词》《从军行》或玉门关相关诗句，也不要画士兵、盔甲、武器、战争、血腥、冲突、旗帜、火焰或攻击动作；改画远处关城、荒漠古道、柳树、山影、旅人远景背影或无人物的安全画面。",
    ]
    return "\n".join(lines)


def build_nano_banana_negative_prompt() -> str:
    return NANO_BANANA_NEGATIVE_PROMPT


def _build_safe_i2v_first_frame_prompt(
    *,
    title: str,
    author: str,
    dynasty: str,
    content: str,
    places: str,
    imagery_text: str,
    emotion: str,
    analysis_text: str,
) -> str:
    return "\n".join(
        [
            "safe_i2v_first_frame mode: generate one safe opening first frame for Wan image-to-video.",
            "16:9 horizontal landscape composition, wide cinematic frame, pure landscape or very distant human silhouette only.",
            "No frontal person, no close portrait, no soldier, no weapon, no armor, no war, no battle, no violence, no blood, no conflict, no flag, no flame, no attack action.",
            "no text, no watermark, no weapon, no battle, no violence, no soldier, no armor.",
            "画面必须是河西走廊自然风景、关隘远景、山川、柳树、沙洲、云影、古道；可以无人物，或只有旅人远景背影/剪影。",
            "不要正面人物、士兵、武器、盔甲、战争场面、血腥、冲突、旗帜、火焰或攻击动作。",
            "对《凉州词》、玉门关、羌笛、杨柳等边塞诗意，只表现远处关城、荒漠古道、柳树、山影、沙洲、云影、晨昏光线和旅人远景背影。",
            "Style: Chinese guofeng ink-wash and pale color illustration, quiet, natural, cinematic, suitable as the exact opening first frame of a 15-second video.",
            f"title: {title}",
            f"author: {author}",
            f"dynasty: {dynasty}",
            f"content: {content}",
            f"places: {places}",
            f"imagery: {imagery_text}",
            f"emotion: {emotion}",
            f"analysis_text: {analysis_text}",
        ]
    )


def _request_text(req: PoetryImageRequest, field_name: str, default: str = "") -> str:
    value = getattr(req, field_name, default)
    if value is None:
        value = default
    if isinstance(value, (list, tuple, set)):
        return "、".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _load_config() -> NanoBananaConfig:
    grsai_host_raw, grsai_host_source = _first_env_value(GRSAI_HOST_ENV_NAMES, GRSAI_DEFAULT_HOST)
    grsai_api_key, grsai_api_key_source = _first_env_value(GRSAI_API_KEY_ENV_NAMES, "")
    grsai_host = _resolve_api_root(
        grsai_host_raw or GRSAI_DEFAULT_HOST
    )
    configured_provider = _env_value("IMAGE_GENERATION_PROVIDER", "")
    provider = configured_provider or "nano_banana"
    normalized_provider = _normalize_provider(provider)
    if normalized_provider in {"dashscope", "qwen", "qwen_image"}:
        _log_image_event(
            "config_warning",
            provider="nano_banana",
            configured_provider=normalized_provider,
            model=GRSAI_NANO_BANANA_MODEL,
            error="poetry image endpoint is forced to Grsai nano-banana-pro; ignoring non-Grsai IMAGE_GENERATION_PROVIDER",
        )
        normalized_provider = "nano_banana"

    api_key = grsai_api_key
    configured_model = (
        _env_value("GRSAI_NANO_BANANA_MODEL", "")
        or _env_value("NANO_BANANA_MODEL", "")
        or _env_value("GRSAI_IMAGE_MODEL", "")
    )
    if configured_model and configured_model != GRSAI_NANO_BANANA_MODEL:
        _log_image_event(
            "config_warning",
            provider=normalized_provider,
            model=GRSAI_NANO_BANANA_MODEL,
            configured_model=configured_model,
            error=f"ignoring configured Nano Banana model; forcing {GRSAI_NANO_BANANA_MODEL}",
        )
    model = GRSAI_NANO_BANANA_MODEL
    configured_aspect_ratio = (
        _env_value("GRSAI_NANO_BANANA_ASPECT_RATIO", "")
        or _env_value("NANO_BANANA_ASPECT_RATIO", "")
    ).strip()
    if configured_aspect_ratio.lower() in {"", "auto", "default"}:
        configured_aspect_ratio = "16:9"
    return NanoBananaConfig(
        provider=normalized_provider,
        host=grsai_host,
        api_key=api_key,
        model=model,
        generator=GRSAI_NANO_BANANA_MODEL,
        image_size=_env_value("GRSAI_NANO_BANANA_IMAGE_SIZE", "") or _env_value("NANO_BANANA_IMAGE_SIZE", "1K"),
        aspect_ratio=configured_aspect_ratio,
        dashscope_api_root=_resolve_api_root(
            _env_value("DASHSCOPE_IMAGE_API_BASE", "")
            or _env_value("DASHSCOPE_API_BASE", "")
            or _env_value("QWEN_VL_BASE_URL", "")
            or "https://dashscope.aliyuncs.com"
        ),
        env_file=str(ENV_FILE),
        host_source=grsai_host_source,
        api_key_source=grsai_api_key_source,
        poll_max_attempts=_int_env("IMAGE_GENERATION_POLL_MAX_ATTEMPTS", 180, minimum=1),
        poll_interval_seconds=_float_env("IMAGE_GENERATION_POLL_INTERVAL_SECONDS", 2.0, minimum=0.1),
        poll_timeout_seconds=_float_env("IMAGE_GENERATION_POLL_TIMEOUT_SECONDS", 900.0, minimum=5.0),
    )


def _generate_real_image(prompt: str, negative_prompt: str, config: NanoBananaConfig) -> PoetryImageResponse:
    if _normalize_provider(config.provider) == "mock":
        raise RuntimeError("mock image provider is disabled; configure a real image provider")
    if not config.api_key:
        checked_names = ", ".join(GRSAI_API_KEY_ENV_NAMES)
        raise RuntimeError(
            f"{config.provider} image generation missing API key; checked process env and {config.env_file}: {checked_names}"
        )
    if not config.host:
        raise RuntimeError("nano_banana image generation missing GRSAI_NANO_BANANA_HOST")
    return _generate_grsai_nano_banana_image(prompt, negative_prompt, config)


def _generate_grsai_nano_banana_image(prompt: str, negative_prompt: str, config: NanoBananaConfig) -> PoetryImageResponse:
    url = _join_url(config.host, GRSAI_NANO_BANANA_SUBMIT_PATH)
    payload = {
        "model": GRSAI_NANO_BANANA_MODEL,
        "prompt": prompt,
        "imageSize": config.image_size,
        "aspectRatio": config.aspect_ratio,
        "webHook": "-1",
        "shutProgress": False,
    }
    if negative_prompt:
        payload["negativePrompt"] = negative_prompt
    data = _post_json(
        url,
        payload,
        _bearer_headers(config.api_key),
        timeout=120,
        config=config,
        stage="submit_http",
    )
    return _resolve_image_task(
        submit_payload=data,
        prompt=prompt,
        config=config,
        poller=lambda task_id, status_url: _poll_grsai_nano_banana_task(config, task_id, status_url),
    )


def _generate_dashscope_image(prompt: str, negative_prompt: str, config: NanoBananaConfig) -> PoetryImageResponse:
    url = f"{config.dashscope_api_root.rstrip('/')}/api/v1/services/aigc/text2image/image-synthesis"
    payload = {
        "model": config.model,
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        },
        "parameters": {
            "size": _dashscope_image_size(config),
            "n": 1,
        },
    }
    headers = _bearer_headers(config.api_key)
    headers["X-DashScope-Async"] = "enable"
    data = _post_json(url, payload, headers, timeout=120, config=config, stage="submit_http")
    return _resolve_image_task(
        submit_payload=data,
        prompt=prompt,
        config=config,
        poller=lambda task_id, status_url: _poll_dashscope_image_task(config, task_id, status_url),
    )


def _resolve_image_task(
    *,
    submit_payload: Any,
    prompt: str,
    config: NanoBananaConfig,
    poller,
) -> PoetryImageResponse:
    task_id = _extract_task_id(submit_payload)
    status_url = _extract_status_url(submit_payload)
    raw_status = _extract_status(submit_payload)
    status = _normalize_status(raw_status)
    image_url, image_url_path = _extract_image_url_with_path(submit_payload, config)
    _log_provider_response_summary("submit", submit_payload, config=config, task_id=task_id, image_url=image_url, image_url_path=image_url_path)
    _log_image_event(
        "submit",
        provider=config.provider,
        model=config.model,
        generator=config.generator,
        task_id=task_id,
        status=raw_status or ("succeeded" if image_url else ""),
        image_url_present=bool(image_url),
        image_url_path=image_url_path,
        raw_result_keys=_raw_result_keys(submit_payload),
    )

    submit_error = _provider_error_message(submit_payload)
    if submit_error:
        _raise_logged_image_error(
            config,
            task_id=task_id,
            status=raw_status or "failed",
            error=submit_error,
            payload=submit_payload,
        )
    if image_url and (not status or status in SUCCESS_STATUSES or _is_success_payload(submit_payload)):
        return PoetryImageResponse(image_url=image_url, prompt=prompt, provider=config.provider, model=config.model, status=raw_status or "succeeded")
    if status in FAILURE_STATUSES:
        _raise_logged_image_error(
            config,
            task_id=task_id,
            status=raw_status,
            error=_extract_failure_message(submit_payload) or f"{config.provider} image generation failed: {raw_status}",
            payload=submit_payload,
        )
    if not task_id and status not in WAITING_STATUSES:
        if _is_success_payload(submit_payload):
            _raise_logged_image_error(
                config,
                task_id=task_id,
                status=raw_status or "success",
                error=_missing_success_image_url_message(submit_payload),
                payload=submit_payload,
            )
        _raise_logged_image_error(
            config,
            task_id=task_id,
            status=raw_status,
            error=_extract_failure_message(submit_payload) or f"{config.provider} image generation returned no image_url or task_id",
            payload=submit_payload,
        )
    if not task_id:
        _raise_logged_image_error(
            config,
            task_id=task_id,
            status=raw_status or "pending",
            error=f"{config.provider} image generation is {raw_status or 'pending'} but no task_id was returned",
            payload=submit_payload,
        )

    started_at = time.monotonic()
    last_status = raw_status or "pending"
    last_payload = submit_payload
    for attempt in range(config.poll_max_attempts):
        elapsed = time.monotonic() - started_at
        if elapsed >= config.poll_timeout_seconds:
            break
        if attempt > 0:
            time.sleep(config.poll_interval_seconds)

        poll_payload = poller(task_id, status_url)
        last_payload = poll_payload
        last_status = _extract_status(poll_payload) or last_status
        normalized_status = _normalize_status(last_status)
        image_url, image_url_path = _extract_image_url_with_path(poll_payload, config)
        _log_provider_response_summary("poll", poll_payload, config=config, task_id=task_id, image_url=image_url, image_url_path=image_url_path)
        _log_image_event(
            "poll",
            provider=config.provider,
            model=config.model,
            generator=config.generator,
            task_id=task_id,
            status=last_status,
            image_url_present=bool(image_url),
            image_url_path=image_url_path,
            raw_result_keys=_raw_result_keys(poll_payload),
        )

        poll_error = _provider_error_message(poll_payload)
        if poll_error:
            _raise_logged_image_error(
                config,
                task_id=task_id,
                status=last_status or "failed",
                error=poll_error,
                payload=poll_payload,
            )
        if image_url and (not normalized_status or normalized_status in SUCCESS_STATUSES or _is_success_payload(poll_payload)):
            return PoetryImageResponse(image_url=image_url, prompt=prompt, provider=config.provider, model=config.model, status=last_status or "succeeded")
        if normalized_status in FAILURE_STATUSES:
            _raise_logged_image_error(
                config,
                task_id=task_id,
                status=last_status,
                error=_extract_failure_message(poll_payload) or f"{config.provider} image generation failed: {last_status}",
                payload=poll_payload,
            )
        if normalized_status in SUCCESS_STATUSES and not image_url:
            break

    elapsed = time.monotonic() - started_at
    if _is_success_payload(last_payload):
        _raise_logged_image_error(
            config,
            task_id=task_id,
            status=last_status or "success",
            error=_missing_success_image_url_message(last_payload, task_id=task_id),
            payload=last_payload,
        )
    if elapsed >= config.poll_timeout_seconds:
        _raise_logged_image_error(
            config,
            task_id=task_id,
            status=last_status,
            error=f"timeout waiting for image generation; task_id={task_id}; status={last_status}; no image url",
            payload=last_payload,
        )
    _raise_logged_image_error(
        config,
        task_id=task_id,
        status=last_status,
        error=_extract_failure_message(last_payload) or f"no image url returned by provider; task_id={task_id}; status={last_status}",
        payload=last_payload,
    )


def _poll_dashscope_image_task(config: NanoBananaConfig, task_id: str, status_url: str = "") -> Any:
    url = status_url or f"{config.dashscope_api_root.rstrip('/')}/api/v1/tasks/{task_id}"
    return _get_json(url, _bearer_headers(config.api_key), timeout=60, config=config, stage="poll_http", task_id=task_id)


def _poll_grsai_nano_banana_task(config: NanoBananaConfig, task_id: str, status_url: str = "") -> Any:
    headers = _bearer_headers(config.api_key)
    if status_url:
        return _post_json(
            status_url,
            {"id": task_id},
            headers,
            timeout=60,
            config=config,
            stage="poll_http",
            task_id=task_id,
        )

    return _post_json(
        _join_url(config.host, GRSAI_NANO_BANANA_RESULT_PATH),
        {"id": task_id},
        headers,
        timeout=60,
        config=config,
        stage="poll_http",
        task_id=task_id,
    )


def _env_value(name: str, default: str = "") -> str:
    value, _source = _env_lookup(name)
    return value or str(default or "").strip()


def _first_env_value(names: Iterable[str], default: str = "") -> Tuple[str, str]:
    for name in names:
        value, source = _env_lookup(name)
        if value:
            return value, source
    return str(default or "").strip(), "default" if default else ""


def _env_lookup(name: str) -> Tuple[str, str]:
    process_value = os.environ.get(name)
    if process_value is not None and str(process_value).strip():
        return str(process_value).strip(), f"process:{name}"

    env_map = _dotenv_values()
    file_value = env_map.get(name)
    if file_value is not None and str(file_value).strip():
        return str(file_value).strip(), f"{ENV_FILE}:{name}"

    return "", ""


def _dotenv_values() -> Dict[str, str]:
    if not ENV_FILE.exists():
        return {}

    result: Dict[str, str] = {}
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        if key:
            result[key] = value
    return result


def _post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    *,
    timeout: float,
    config: Optional[NanoBananaConfig] = None,
    stage: str = "http_post",
    task_id: str = "",
) -> Any:
    require_generation_access()
    _log_http_request("POST", url, payload, config=config, stage=stage, task_id=task_id)
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        _log_http_error("POST", url, exc, config=config, stage=stage, task_id=task_id)
        raise
    return _response_json(response, config=config, stage=stage, task_id=task_id, method="POST", url=url)


def _get_json(
    url: str,
    headers: Dict[str, str],
    *,
    timeout: float,
    config: Optional[NanoBananaConfig] = None,
    stage: str = "http_get",
    task_id: str = "",
) -> Any:
    _log_http_request("GET", url, {}, config=config, stage=stage, task_id=task_id)
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        _log_http_error("GET", url, exc, config=config, stage=stage, task_id=task_id)
        raise
    return _response_json(response, config=config, stage=stage, task_id=task_id, method="GET", url=url)


def _response_json(
    response: requests.Response,
    *,
    config: Optional[NanoBananaConfig],
    stage: str,
    task_id: str,
    method: str,
    url: str,
) -> Any:
    body_text = response.text
    parsed: Any = None
    parse_error = ""
    try:
        parsed = response.json()
    except ValueError as exc:
        parse_error = str(exc)

    image_url = ""
    image_url_path = ""
    if isinstance(parsed, dict) and config:
        image_url, image_url_path = _extract_image_url_with_path(parsed, config)
    _log_image_event(
        f"{stage}_response",
        provider=config.provider if config else "",
        model=config.model if config else "",
        generator=config.generator if config else "",
        task_id=task_id or (_extract_task_id(parsed) if isinstance(parsed, dict) else ""),
        url=url,
        method=method,
        http_status=response.status_code,
        status=_extract_status(parsed) if isinstance(parsed, dict) else "",
        code=str(_first_value(parsed, ("code", "status_code", "statusCode", "errcode")) or "").strip() if isinstance(parsed, dict) else "",
        message=_extract_error_message(parsed) if isinstance(parsed, dict) else "",
        image_url_present=bool(image_url),
        image_url_path=image_url_path,
        error=parse_error,
        raw_result_keys=_raw_result_keys(parsed) if isinstance(parsed, dict) else [],
    )
    if not response.ok:
        provider_message = _extract_failure_message(parsed) if isinstance(parsed, dict) else ""
        message = provider_message or _truncate_text(_redact_secret(body_text, config)) or response.reason
        raise RuntimeError(f"{method} {url} returned HTTP {response.status_code}: {message}")
    if parsed is None:
        raise RuntimeError(f"{method} {url} returned non-json response: {_truncate_text(body_text)}")
    return parsed


def _failure_response(prompt: str, config: NanoBananaConfig, error: str, *, status: str = "failed") -> PoetryImageResponse:
    safe_error = _redact_secret(error, config)
    _log_image_event(
        "failed",
        provider=config.provider,
        model=config.model,
        generator=config.generator,
        task_id="",
        status=status or "failed",
        error=safe_error,
        raw_result_keys=[],
    )
    return PoetryImageResponse(
        image_url="",
        prompt=prompt,
        provider=config.provider,
        model=config.model,
        status="failed",
        error=safe_error,
    )


def _raise_logged_image_error(
    config: NanoBananaConfig,
    *,
    task_id: str,
    status: str,
    error: str,
    payload: Any,
) -> None:
    safe_error = _redact_secret(error, config)
    _log_image_event(
        "failed",
        provider=config.provider,
        model=config.model,
        generator=config.generator,
        task_id=task_id,
        status=status,
        error=safe_error,
        raw_result_keys=_raw_result_keys(payload),
    )
    raise RuntimeError(safe_error)


def _log_http_request(
    method: str,
    url: str,
    payload: Dict[str, Any],
    *,
    config: Optional[NanoBananaConfig],
    stage: str,
    task_id: str,
) -> None:
    _log_image_event(
        stage,
        provider=config.provider if config else "",
        model=config.model if config else "",
        generator=config.generator if config else "",
        task_id=task_id,
        url=url,
        method=method,
        image_size=payload.get("imageSize") or payload.get("size"),
        aspect_ratio=payload.get("aspectRatio") or payload.get("aspect_ratio"),
        payload_keys=sorted(str(key) for key in payload.keys()),
    )


def _log_http_error(
    method: str,
    url: str,
    error: Exception,
    *,
    config: Optional[NanoBananaConfig],
    stage: str,
    task_id: str,
) -> None:
    response = getattr(error, "response", None)
    response_body = ""
    http_status = ""
    if response is not None:
        http_status = getattr(response, "status_code", "")
        response_body = getattr(response, "text", "") or ""
    _log_image_event(
        f"{stage}_error",
        provider=config.provider if config else "",
        model=config.model if config else "",
        generator=config.generator if config else "",
        task_id=task_id,
        url=url,
        method=method,
        http_status=http_status,
        response_body_present=bool(response_body),
        response_body_length=len(response_body),
        error=_redact_secret(f"{error.__class__.__name__}: {error}", config),
    )


def _truncate_text(value: Any, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _join_url(host: str, path: str) -> str:
    return f"{str(host or '').rstrip('/')}/{str(path or '').lstrip('/')}"


def _bearer_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _extract_status(payload: Any) -> str:
    return _first_text(payload, ("task_status", "status", "state", "taskStatus"))


def _extract_task_id(payload: Any) -> str:
    return _first_text(payload, ("task_id", "taskId", "id", "job_id", "jobId"))


def _extract_status_url(payload: Any) -> str:
    return _first_text(payload, ("status_url", "statusUrl", "task_url", "taskUrl", "poll_url", "pollUrl"))


def _extract_image_url(payload: Any, config: NanoBananaConfig) -> str:
    return _extract_image_url_with_path(payload, config)[0]


def _extract_image_url_with_path(payload: Any, config: NanoBananaConfig) -> Tuple[str, str]:
    for url, path in _collect_image_url_candidates(payload):
        if _looks_like_image_url(url):
            return url, path
    return "", ""


def _collect_image_url_candidates(payload: Any) -> List[Tuple[str, str]]:
    image_keys = (
        "url",
        "image_url",
        "imageUrl",
        "image",
        "images",
        "image_urls",
        "imageUrls",
        "output_url",
        "outputUrl",
        "result_url",
        "resultUrl",
        "download_url",
        "downloadUrl",
        "file_url",
        "fileUrl",
        "media_url",
        "mediaUrl",
        "result",
    )
    normalized_keys = {key.lower() for key in image_keys}
    candidates: List[Tuple[str, str]] = []

    def append_string(value: Any, path: str) -> None:
        text = str(value or "").strip()
        if text:
            candidates.append((text, path))

    def visit(node: Any, path: str = "$") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                key_lower = str(key).lower()
                if key_lower in normalized_keys:
                    collect_value(value, child_path)
                if isinstance(value, (dict, list)):
                    visit(value, child_path)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                child_path = f"{path}[{index}]"
                if isinstance(item, str):
                    append_string(item, child_path)
                elif isinstance(item, (dict, list)):
                    visit(item, child_path)

    def collect_value(value: Any, path: str) -> None:
        if isinstance(value, str):
            append_string(value, path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect_value(item, f"{path}[{index}]")
        elif isinstance(value, dict):
            visit(value, path)

    visit(payload)
    return candidates


def _looks_like_image_url(value: str) -> bool:
    text = str(value or "").strip()
    if not text.startswith(("http://", "https://")):
        return False
    path = urlparse(text).path.rstrip("/").lower()
    if path.endswith(GRSAI_NANO_BANANA_SUBMIT_PATH.lower()) or path.endswith(GRSAI_NANO_BANANA_RESULT_PATH.lower()):
        return False
    return True


def _first_text(payload: Any, keys: Iterable[str]) -> str:
    value = _first_value(payload, keys)
    if value is None:
        return ""
    return str(value).strip()


def _collect_texts(payload: Any, keys: Iterable[str]) -> List[str]:
    normalized_keys = {str(key).lower() for key in keys}
    results: List[str] = []

    def append_value(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                results.append(text)
            return
        if isinstance(value, list):
            for item in value:
                append_value(item)
            return
        if isinstance(value, dict):
            visit(value)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in normalized_keys:
                    append_value(value)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)
    return results


def _first_value(payload: Any, keys: Iterable[str]) -> Optional[Any]:
    normalized_keys = {str(key).lower() for key in keys}

    def visit(node: Any) -> Optional[Any]:
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in normalized_keys and not isinstance(value, (dict, list)):
                    return value
            for value in node.values():
                found = visit(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = visit(item)
                if found is not None:
                    return found
        return None

    return visit(payload)


def _normalize_status(value: str) -> str:
    return str(value or "").strip().lower()


def _normalize_provider(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"grsai", "grsai_nano_banana", "nano_banana_pro", "nano_banana"}:
        return "nano_banana"
    if text in {"qwen", "qwen_image", "dashscope", "dashscope_image"}:
        return "dashscope" if text == "dashscope_image" else text
    return text or "nano_banana"


def _is_dashscope_provider(provider: str) -> bool:
    return _normalize_provider(provider) in {"dashscope", "qwen", "qwen_image"}


def _dashscope_image_size(config: NanoBananaConfig) -> str:
    raw = str(_env_value("DASHSCOPE_IMAGE_SIZE", "") or config.image_size or "1024*1024").strip()
    normalized = raw.lower()
    if normalized in {"1k", "auto"}:
        return "1024*1024"
    if "*" in normalized:
        return normalized
    if normalized in {"720p", "768"}:
        return "768*1344"
    return raw


def _resolve_api_root(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return "https://dashscope.aliyuncs.com"
    return text.rstrip("/")


def _extract_error_message(payload: Any) -> str:
    return _first_text(payload, ("message", "msg", "error", "error_message", "errorMessage", "reason", "detail"))


def _extract_failure_message(payload: Any) -> str:
    message = _extract_error_message(payload)
    return "" if _is_success_text(message) else message


def _provider_error_message(payload: Any) -> str:
    code = _first_value(payload, ("code", "error_code", "errorCode", "errcode"))
    if code is not None:
        normalized_code = str(code).strip().lower()
        if normalized_code not in {"", *SUCCESS_CODES}:
            return _extract_failure_message(payload) or f"provider returned code={code}"

    status = _normalize_status(_extract_status(payload))
    if status in FAILURE_STATUSES:
        return _extract_failure_message(payload) or f"provider returned status={status}"
    return ""


def _is_success_payload(payload: Any) -> bool:
    status = _normalize_status(_extract_status(payload))
    if status in SUCCESS_STATUSES:
        return True
    code = _first_value(payload, ("code", "status_code", "statusCode", "errcode"))
    if code is not None and str(code).strip().lower() in SUCCESS_CODES:
        return True
    return _is_success_text(_extract_error_message(payload))


def _is_success_text(value: Any) -> bool:
    return str(value or "").strip().lower() in SUCCESS_CODES


def _missing_success_image_url_message(payload: Any, *, task_id: str = "") -> str:
    paths = sorted(dict.fromkeys(path for _url, path in _collect_image_url_candidates(payload)))
    path_text = ", ".join(paths[:12]) if paths else "none"
    task_text = f"; task_id={task_id}" if task_id else ""
    return f"生图成功但未找到图片 URL{task_text}; checked_image_url_paths={path_text}"


def _log_provider_response_summary(
    stage: str,
    payload: Any,
    *,
    config: NanoBananaConfig,
    task_id: str = "",
    image_url: str = "",
    image_url_path: str = "",
) -> None:
    result_node = payload.get("result") if isinstance(payload, dict) else None
    data_node = payload.get("data") if isinstance(payload, dict) else None
    output_node = payload.get("output") if isinstance(payload, dict) else None
    summary = {
        "provider": config.provider,
        "model": config.model,
        "generator": config.generator,
        "task_id": task_id or _extract_task_id(payload),
        "status": _extract_status(payload),
        "code": str(_first_value(payload, ("code", "status_code", "statusCode", "errcode")) or "").strip(),
        "message": _extract_error_message(payload),
        "image_url_present": bool(image_url),
        "image_url_path": image_url_path,
        "result_type": type(result_node).__name__ if result_node is not None else "",
        "data_keys": sorted(data_node.keys()) if isinstance(data_node, dict) else [],
        "output_keys": sorted(output_node.keys()) if isinstance(output_node, dict) else [],
        "raw_result_keys": _raw_result_keys(payload),
    }
    _log_image_event(f"{stage}_provider_summary", **summary)


def _raw_result_keys(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return [type(payload).__name__]
    keys = [str(key) for key in payload.keys()]
    for parent_key in ("output", "data", "result", "results"):
        child = payload.get(parent_key)
        if isinstance(child, dict):
            keys.extend(f"{parent_key}.{key}" for key in child.keys())
        elif isinstance(child, list):
            keys.append(f"{parent_key}[]")
            for item in child[:1]:
                if isinstance(item, dict):
                    keys.extend(f"{parent_key}[].{key}" for key in item.keys())
    return sorted(dict.fromkeys(keys))


def _log_image_event(stage: str, **payload: Any) -> None:
    summary = {"stage": stage, **payload}
    print(f"[poetry_image] {json.dumps(summary, ensure_ascii=False, default=str)}", flush=True)


def _redact_secret(value: Any, config: Optional[NanoBananaConfig]) -> str:
    text = str(value or "")
    if config and config.api_key:
        text = text.replace(config.api_key, "<redacted-api-key>")
    return text


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = _env_value(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = _env_value(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, value)
