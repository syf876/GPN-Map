from backend.admin_auth import require_generation_access
import hashlib
import ipaddress
import json
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import requests

from backend.types.schema import VeoGenerationRequest, VeoGenerationResponse

from .env import WAN_API_KEY_ENV_NAMES, env_value, first_env_value
from .logging_utils import safe_url_preview
from .prompt_builder import build_wan_negative_prompt, build_wan_prompt


SUCCESS_STATUSES = {"SUCCEEDED"}
FAILURE_STATUSES = {"FAILED", "CANCELED", "CANCELLED"}
WAITING_STATUSES = {"PENDING", "RUNNING"}
ALLOWED_IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
BEIJING_DASHSCOPE_ROOT = "https://dashscope.aliyuncs.com"
DEFAULT_WAN_I2V_MODEL = "wan2.6-i2v-flash"
DEFAULT_WAN_I2V_FALLBACK_MODELS = (
    "wan2.6-i2v-flash",
    "wan2.6-i2v",
    "wan2.5-i2v-preview",
    "wan2.2-i2v-plus",
    "wan2.2-i2v-flash",
    "wanx2.1-i2v-turbo",
    "wanx2.1-i2v-plus",
)
REPO_ROOT = Path(__file__).resolve().parents[3]
WAN_UPLOAD_DIR = REPO_ROOT / "artifacts" / "wan_upload_cache"
WAN_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CONTENT_SAFETY_ERROR_CODE = "content_safety"


@dataclass(frozen=True)
class WanConfig:
    api_root: str
    api_key: str
    api_key_source: str
    model: str
    resolution: str
    duration: int
    prompt_extend: bool
    shot_type: str
    audio_enabled: bool
    poll_max_attempts: int
    poll_interval_seconds: float
    poll_timeout_seconds: float
    poll_max_interval_seconds: float


class WanRequestError(RuntimeError):
    def __init__(self, detail: Dict[str, Any]):
        self.detail = detail
        super().__init__(_error_message_from_detail(detail))


def _error_detail(
    *,
    stage: str,
    code: str = "",
    message: str = "",
    request_id: str = "",
    task_id: str = "",
    http_status: str | int = "",
    raw_status: str = "",
) -> Dict[str, Any]:
    detail = {
        "stage": str(stage or "").strip(),
        "code": str(code or "").strip(),
        "message": str(message or "").strip(),
        "request_id": str(request_id or "").strip(),
        "task_id": str(task_id or "").strip(),
        "http_status": str(http_status or "").strip(),
        "raw_status": str(raw_status or "").strip(),
    }
    return _normalize_wan_error_detail(detail)


def _error_message_from_detail(detail: Dict[str, Any]) -> str:
    message = str(detail.get("message") or "").strip()
    code = str(detail.get("code") or "").strip()
    stage = str(detail.get("stage") or "").strip()
    if code and message:
        return f"{stage}: {code}: {message}" if stage else f"{code}: {message}"
    if message:
        return f"{stage}: {message}" if stage else message
    if code:
        return f"{stage}: {code}" if stage else code
    return stage or "wan request failed"


def _failed_response_from_detail(
    *,
    prompt: str,
    detail: Dict[str, Any],
    progress: float = 0.0,
    provider: str = "wan",
    video_url: str = "",
) -> VeoGenerationResponse:
    task_id = str(detail.get("task_id") or "").strip()
    raw_status = str(detail.get("raw_status") or detail.get("code") or "request_exception").strip()
    return VeoGenerationResponse(
        status="failed",
        provider=provider,
        id=task_id,
        video_url=video_url,
        progress=progress,
        prompt=prompt,
        error=_error_message_from_detail(detail),
        error_detail=detail,
        raw_status=raw_status,
    )


def generate_wan_video(req: VeoGenerationRequest) -> VeoGenerationResponse:
    require_generation_access()
    prompt = build_wan_prompt(req)
    negative_prompt = build_wan_negative_prompt(req)
    config = _load_config(req)
    preflight_error = _validate_i2v_preconditions(req, config, prompt)
    if preflight_error is not None:
        return preflight_error

    failures: List[Tuple[str, VeoGenerationResponse]] = []
    for attempt_index, attempt_config in enumerate(_i2v_attempt_configs(config)):
        previous_error = failures[-1][1].error if failures else ""
        _log_i2v_attempt(
            "direct-video",
            model=attempt_config.model,
            fallback=attempt_index > 0,
            previous_error=previous_error,
        )
        response = _generate_wan_video_once(
            req=req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            config=attempt_config,
        )
        if _response_has_playable_video(response):
            return response
        if str(response.status or "").lower() != "failed":
            return response
        if _is_image_asset_validation_failure(response.error):
            return response
        if _is_wan_content_safety_failure(response.error_detail):
            return response
        if _is_non_retryable_wan_failure(response.error_detail):
            return response
        failures.append((attempt_config.model, response))

    return _failure_response_from_attempts(prompt, failures)


def _generate_wan_video_once(
    *,
    req: VeoGenerationRequest,
    prompt: str,
    negative_prompt: str,
    config: WanConfig,
) -> VeoGenerationResponse:
    missing_dependencies = _missing_runtime_dependencies(config.model)
    if missing_dependencies:
        return _failed_response_from_detail(
            prompt=prompt,
            detail=_error_detail(
                stage="runtime-dependencies",
                code="missing_runtime_dependencies",
                message=f"missing runtime dependencies: {', '.join(missing_dependencies)}",
            ),
        )
    if not config.api_key:
        return _failed_response_from_detail(
            prompt=prompt,
            detail=_error_detail(
                stage="config",
                code="missing_api_key",
                message="missing Wanxiang API key environment variable; checked DASHSCOPE_API_KEY, QWEN_VL_API_KEY, WAN_API_KEY, WAN_I2V_API_KEY and compatible aliases",
            ),
        )

    image_url = str(req.image_url or "").strip()
    audio_url = str(req.audio_url or "").strip()
    if not image_url:
        return _failed_response_from_detail(
            prompt=prompt,
            detail=_error_detail(
                stage="request-validation",
                code="missing_image_url",
                message=f"{config.model} requires image_url",
            ),
        )
    try:
        if _resolve_local_asset_path(image_url) is None:
            _log_remote_asset_preflight(
                "image_url",
                image_url,
                expected_content_prefixes=("image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"),
            )
        if audio_url:
            if _resolve_local_asset_path(audio_url) is None:
                _log_remote_asset_preflight(
                    "audio_url",
                    audio_url,
                    expected_content_prefixes=("audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"),
                )

        image_local_path = None
        image_content_type = ""
        should_upload_image_for_wan = True
        if should_upload_image_for_wan:
            _log_wan_safe(
                model=config.model,
                http_status="",
                task_id="",
                error="",
                stage="image-url-to-dashscope-oss",
                source_image_url_preview=safe_url_preview(image_url),
                reason="ensure_wan_can_access_first_frame",
                local_or_non_public=_resolve_local_asset_path(image_url) is not None or _is_non_public_http_url(image_url),
            )
            image_local_path, image_content_type = _download_asset_to_local_path(
                source_url=image_url,
                asset_kind="image",
                expected_content_prefixes=("image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"),
                default_extension=".png",
            )
        elif not _is_http_url(image_url):
            return _failed_response_from_detail(
                prompt=prompt,
                detail=_error_detail(
                    stage="request-validation",
                    code="invalid_image_url",
                    message="poetry video image_url must be an http(s) URL from the successful poetry image generation step",
                ),
            )
        audio_local_path = None
        audio_content_type = ""
        if audio_url:
            audio_local_path, audio_content_type = _download_asset_to_local_path(
                source_url=audio_url,
                asset_kind="audio",
                expected_content_prefixes=("audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"),
                default_extension=".mp3",
            )

        create_payload = _build_create_payload(
            config=config,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_url=image_url,
            image_local_path=image_local_path,
            image_content_type=image_content_type,
            audio_local_path=audio_local_path,
            audio_content_type=audio_content_type,
        )
        _log_wan_request("video-synthesis", create_payload, source_image_url=image_url)
        create_response = _post_json(
            _video_synthesis_endpoint(config),
            create_payload,
            headers=_json_headers(config.api_key, resolve_oss=_contains_oss_url(create_payload), async_mode=True),
            stage="video-synthesis",
        )
        task_id = _extract_task_id(create_response)
        raw_status = _extract_task_status(create_response) or "PENDING"
        if not task_id:
            return _failed_response_from_detail(
                prompt=prompt,
                detail=_error_detail_from_payload(
                    stage="video-synthesis",
                    payload=create_response,
                    http_status="",
                    fallback_code="missing_task_id",
                    fallback_message="wan create response missing task id",
                    raw_status=raw_status or "missing_task_id",
                ),
            )

        progress = 5.0
        poll_started_at = time.monotonic()
        for attempt in range(config.poll_max_attempts):
            elapsed_seconds = time.monotonic() - poll_started_at
            if elapsed_seconds >= config.poll_timeout_seconds:
                break

            poll_error = ""
            task_response = None
            try:
                task_response = _get_json(
                    f"{config.api_root}/api/v1/tasks/{task_id}",
                    headers=_auth_headers(config.api_key),
                    stage="task-query",
                )
            except WanRequestError as exc:
                detail = dict(exc.detail)
                detail["task_id"] = detail.get("task_id") or task_id
                detail["raw_status"] = detail.get("raw_status") or raw_status
                poll_error = _error_message_from_detail(detail)
                _log_poll_summary(
                    attempt=attempt + 1,
                    elapsed_seconds=elapsed_seconds,
                    next_sleep_seconds=0.0,
                    task_id=task_id,
                    task_status=raw_status,
                    progress=progress,
                    video_url="",
                    error=poll_error,
                )
                if _is_non_retryable_wan_failure(detail) or attempt >= config.poll_max_attempts - 1:
                    return _failed_response_from_detail(prompt=prompt, detail=detail, progress=progress)
                sleep_seconds = _next_poll_interval_seconds(config, attempt)
                if elapsed_seconds + sleep_seconds > config.poll_timeout_seconds:
                    break
                time.sleep(sleep_seconds)
                continue
            except Exception as exc:
                poll_error = str(exc)
                _log_poll_summary(
                    attempt=attempt + 1,
                    elapsed_seconds=elapsed_seconds,
                    next_sleep_seconds=0.0,
                    task_id=task_id,
                    task_status=raw_status,
                    progress=progress,
                    video_url="",
                    error=poll_error,
                )
                if attempt >= config.poll_max_attempts - 1 or elapsed_seconds >= config.poll_timeout_seconds:
                    break
                sleep_seconds = _next_poll_interval_seconds(config, attempt)
                if elapsed_seconds + sleep_seconds > config.poll_timeout_seconds:
                    break
                time.sleep(sleep_seconds)
                continue

            task_status = _extract_task_status(task_response) or raw_status
            video_url = _extract_video_url(task_response)
            progress = _status_progress(task_status, progress)
            raw_status = task_status or raw_status
            sleep_seconds = _next_poll_interval_seconds(config, attempt)
            _log_poll_summary(
                attempt=attempt + 1,
                elapsed_seconds=elapsed_seconds,
                next_sleep_seconds=sleep_seconds,
                task_id=task_id,
                task_status=task_status,
                progress=progress,
                video_url=video_url,
                error="",
            )

            if video_url and task_status in SUCCESS_STATUSES:
                return VeoGenerationResponse(
                    status="ready",
                    provider="wan",
                    id=task_id,
                    video_url=video_url,
                    progress=100.0,
                    prompt=prompt,
                    error="",
                    raw_status=raw_status,
                )

            explicit_error = _extract_message(task_response) or _extract_reason(task_response)
            if task_status in FAILURE_STATUSES or (task_status == "UNKNOWN" and explicit_error):
                detail = _error_detail_from_payload(
                    stage="task-query-final-failure",
                    payload=task_response,
                    http_status="",
                    fallback_code=_extract_code(task_response) or task_status or "wan_task_failed",
                    fallback_message=explicit_error or "wan generation failed",
                    task_id=task_id,
                    raw_status=raw_status,
                )
                _log_wan_safe(
                    model=config.model,
                    http_status="",
                    task_id=task_id,
                    error=explicit_error or "wan generation failed",
                    stage="task-query-final-failure",
                    task_status=task_status,
                    raw_status=raw_status,
                    error_code=_extract_code(task_response),
                    message=_extract_message(task_response),
                    reason=_extract_reason(task_response),
                )
                return _failed_response_from_detail(prompt=prompt, detail=detail, progress=progress)

            if task_status not in WAITING_STATUSES and video_url:
                return VeoGenerationResponse(
                    status="ready",
                    provider="wan",
                    id=task_id,
                    video_url=video_url,
                    progress=100.0,
                    prompt=prompt,
                    error="",
                    raw_status=raw_status,
                )

            if attempt < config.poll_max_attempts - 1 and elapsed_seconds + sleep_seconds < config.poll_timeout_seconds:
                time.sleep(sleep_seconds)

        _log_wan_safe(
            model=config.model,
            http_status="",
            task_id=task_id,
            error="wan polling ended before terminal success",
            stage="task-query-final-processing",
            task_status=raw_status,
            progress=progress,
        )
        return VeoGenerationResponse(
            status="processing",
            provider="wan",
            id=task_id,
            video_url="",
            progress=progress,
            prompt=prompt,
            error="",
            raw_status=raw_status or "RUNNING",
        )
    except WanRequestError as exc:
        return _failed_response_from_detail(prompt=prompt, detail=exc.detail)
    except Exception as exc:
        return _failed_response_from_detail(
            prompt=prompt,
            detail=_error_detail_from_exception(exc),
        )
    finally:
        for path in ("image_local_path", "audio_local_path"):
            value = locals().get(path)
            if isinstance(value, Path):
                try:
                    value.unlink(missing_ok=True)
                except Exception:
                    pass


def _build_create_payload(
    *,
    config: WanConfig,
    prompt: str,
    negative_prompt: str,
    image_url: str,
    image_local_path: Path | None,
    image_content_type: str,
    audio_local_path: Path | None,
    audio_content_type: str,
) -> Dict[str, Any]:
    if _uses_legacy_s2v(config.model):
        raise RuntimeError("legacy wan s2v flow is disabled for poetry image-to-video generation")
    return _build_i2v_payload(
        config=config,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_url=image_url,
        image_local_path=image_local_path,
        image_content_type=image_content_type,
        audio_local_path=audio_local_path,
        audio_content_type=audio_content_type,
    )


def _build_i2v_payload(
    *,
    config: WanConfig,
    prompt: str,
    negative_prompt: str,
    image_url: str,
    image_local_path: Path | None,
    image_content_type: str,
    audio_local_path: Path | None,
    audio_content_type: str,
) -> Dict[str, Any]:
    create_image_url = str(image_url or "").strip()
    if image_local_path is not None:
        create_image_url = _upload_local_file_to_dashscope_oss(
            api_key=config.api_key,
            api_root=config.api_root,
            model_name=config.model,
            local_path=image_local_path,
            asset_kind="image",
            content_type=image_content_type,
        )
    create_prompt = _sanitize_wan_prompt(prompt, max_chars=1500)
    create_negative_prompt = _sanitize_wan_prompt(negative_prompt, max_chars=500)
    input_payload: Dict[str, Any] = {
        "prompt": create_prompt,
        "negative_prompt": create_negative_prompt,
        "img_url": create_image_url,
    }
    if audio_local_path is not None:
        create_audio_url = _upload_local_file_to_dashscope_oss(
            api_key=config.api_key,
            api_root=config.api_root,
            model_name=config.model,
            local_path=audio_local_path,
            asset_kind="audio",
            content_type=audio_content_type,
        )
        input_payload["audio_url"] = create_audio_url

    parameters: Dict[str, Any] = {
        "resolution": config.resolution,
        "prompt_extend": config.prompt_extend,
        "watermark": False,
    }
    if _supports_duration_parameter(config.model):
        parameters["duration"] = config.duration
    if _supports_multi_shot(config.model) and config.prompt_extend:
        parameters["shot_type"] = config.shot_type
    if _supports_audio_toggle(config.model):
        parameters["audio"] = bool(config.audio_enabled)
        if config.audio_enabled and audio_local_path is None:
            # Wan audio=true is prompt-constrained generated audio, not guaranteed strict TTS.
            # Future strict narration should provide an independent TTS audio_url; this function
            # already uploads it and passes it as audio_url / driving_audio where the model supports it.
            _log_wan_audio_prompt_notice(config, create_prompt)

    if _uses_new_wan_i2v_protocol(config.model):
        media = [
            {
                "type": "first_frame",
                "url": create_image_url,
            }
        ]
        if audio_local_path is not None:
            media.append(
                {
                    "type": "driving_audio",
                    "url": input_payload["audio_url"],
                }
            )
        return {
            "model": config.model,
            "input": {
                "prompt": create_prompt,
                "media": media,
            },
            "parameters": parameters,
        }

    return {
        "model": config.model,
        "input": input_payload,
        "parameters": parameters,
    }


def _sanitize_wan_prompt(text: str, *, max_chars: int) -> str:
    lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("source_image_url:") or "http://" in lowered or "https://" in lowered or "oss://" in lowered:
            continue
        lines.append(line)
    if max_chars >= 1000:
        compact = _compact_wan_i2v_prompt(lines)
        if compact:
            return compact[:max_chars].rstrip()
    else:
        compact_negative = _compact_wan_negative_prompt()
        if compact_negative:
            return compact_negative[:max_chars].rstrip()
    cleaned = "\n".join(lines).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()


def _compact_wan_i2v_prompt(lines: List[str]) -> str:
    content = _prompt_field(lines, "voiceover_script") or _prompt_field(lines, "content")
    title = _prompt_field(lines, "title")
    places = _prompt_field(lines, "places")
    imagery = _prompt_field(lines, "imagery")
    emotion = _prompt_field(lines, "emotion")
    parts = [
        "Use the provided image as the exact opening first frame and visual style reference; first 0-1s match the original composition, then continue into connected narrative motion over about 15 seconds.",
        "以输入图片作为视频第一帧和视觉风格参考，开场保持原图构图；随后生成连续叙事镜头，后续画面必须有新景别、新前景/中景/远景细节和可见空间推进。",
        "Show visual-only motion: moving sand and dust, shifting cloud shadows, flowing water or sandbar texture, swaying willow branches, birds crossing, a traveler walking, distant camel caravan, camera push, pan, and pull-back.",
        "Maintain Chinese guofeng ink-wash and pale-color Hexi Corridor frontier mood, same palette and world continuity. No static poster loop, subtle zoom, slight shake, subtitles, watermark, logo, modern UI, or unrelated scene.",
        "Audio must be a clear Mandarin Chinese voice reading the poem aloud.",
        "Use natural human narration only.",
        "Read the poem text exactly as provided.",
        "Do not add flute music, background music, singing, sound effects, wind noise, ambient sound, explanation, commentary, or extra words.",
        "The voice should be calm, poetic, and synchronized within the 15-second video.",
        "音频必须是自然普通话人声朗读当前完整诗词，只朗读诗词正文，不要解释、旁白扩写、音乐、笛声、环境音、风声、马蹄声或其他音效。",
    ]
    if title:
        parts.append(f"Poem title: {title}.")
    if content:
        parts.append(f"voiceover_script: {content}.")
        parts.append(f"Voiceover text / 朗读文本：{content}")
    if places:
        parts.append(f"Place mood: {places}.")
    if imagery:
        parts.append(f"Imagery: {imagery}.")
    if emotion:
        parts.append(f"Emotion: {emotion}.")
    return " ".join(part for part in parts if part).strip()


def _compact_wan_negative_prompt() -> str:
    return "text overlay, subtitles, watermark, logo, modern UI, poster layout, map, coordinates, violent content, explicit content, distorted body, unrelated scene, photorealistic live action"


def _prompt_field(lines: List[str], field_name: str) -> str:
    prefix = f"{field_name}:"
    for line in lines:
        if line.lower().startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _log_wan_audio_prompt_notice(config: WanConfig, prompt: str) -> None:
    print(
        "[wan_audio_prompt_notice] "
        + json.dumps(
            {
                "model": config.model,
                "audio": True,
                "strict_tts": False,
                "limitation": "audio=true may generate model audio and may not strictly perform poem TTS; prompt is strongly constrained to Mandarin narration",
                "future_tts_hook": "pass independent TTS audio_url so Wan payload can upload it as audio_url/driving_audio before video synthesis",
                "prompt_has_voiceover_script": "voiceover_script:" in str(prompt or ""),
                "prompt_has_mandarin_audio_rule": "Audio must be a clear Mandarin Chinese voice reading the poem aloud." in str(prompt or ""),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

def _load_config(req: VeoGenerationRequest) -> WanConfig:
    api_root = _resolve_api_root(
        env_value("WAN_API_BASE", "")
        or env_value("DASHSCOPE_API_BASE", "")
        or env_value("QWEN_VL_BASE_URL", "")
        or BEIJING_DASHSCOPE_ROOT
    )
    configured_model = env_value("WAN_I2V_MODEL", "") or env_value("WAN_MODEL", "")
    api_key, api_key_source = first_env_value(WAN_API_KEY_ENV_NAMES, "")
    config = WanConfig(
        api_root=api_root.rstrip("/"),
        api_key=api_key,
        api_key_source=api_key_source,
        model=_coerce_i2v_model(configured_model, DEFAULT_WAN_I2V_MODEL),
        resolution=_normalize_resolution(req.resolution or env_value("WAN_I2V_RESOLUTION", "") or env_value("WAN_S2V_RESOLUTION", "720P")),
        duration=_int_env("WAN_I2V_DURATION", 15, minimum=2),
        prompt_extend=_bool_env("WAN_I2V_PROMPT_EXTEND", False),
        shot_type=_normalize_shot_type(env_value("WAN_I2V_SHOT_TYPE", "multi")),
        audio_enabled=_bool_env("WAN_I2V_AUDIO", True),
        poll_max_attempts=_int_env("WAN_I2V_POLL_MAX_ATTEMPTS", _int_env("WAN_S2V_POLL_MAX_ATTEMPTS", 180, minimum=1), minimum=1),
        poll_interval_seconds=_float_env("WAN_I2V_POLL_INTERVAL_SECONDS", _float_env("WAN_S2V_POLL_INTERVAL_SECONDS", 5.0, minimum=0.1), minimum=0.1),
        poll_timeout_seconds=_float_env("WAN_I2V_POLL_TIMEOUT_SECONDS", _float_env("WAN_S2V_POLL_TIMEOUT_SECONDS", 900.0, minimum=30.0), minimum=30.0),
        poll_max_interval_seconds=_float_env("WAN_I2V_POLL_MAX_INTERVAL_SECONDS", _float_env("WAN_S2V_POLL_MAX_INTERVAL_SECONDS", 20.0, minimum=0.1), minimum=0.1),
    )
    _log_wan_config(config)
    return config


def _i2v_attempt_configs(config: WanConfig) -> List[WanConfig]:
    models = [config.model]
    fallback_model = _coerce_i2v_model(
        env_value("WAN_I2V_FALLBACK_MODEL", ""),
        "",
    )
    for model in (fallback_model, *_configured_extra_i2v_models(), *DEFAULT_WAN_I2V_FALLBACK_MODELS):
        model = _coerce_i2v_model(model, "")
        if model and model.lower() not in {existing.lower() for existing in models}:
            models.append(model)
    return [replace(config, model=model) for model in models]


def _configured_extra_i2v_models() -> List[str]:
    raw = env_value("WAN_I2V_MODEL_CANDIDATES", "")
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]


def _coerce_i2v_model(model: str, default: str) -> str:
    text = str(model or "").strip()
    if not text:
        return default
    if _uses_legacy_s2v(text):
        _log_i2v_model_coercion(text, default)
        return default
    return text


def _uses_legacy_s2v(model: str) -> bool:
    text = str(model or "").strip().lower()
    return text.endswith("s2v") or "-s2v" in text or text == "wan_s2v"


def _uses_new_wan_i2v_protocol(model: str) -> bool:
    return str(model or "").strip().lower().startswith("wan2.7")


def _supports_duration_parameter(model: str) -> bool:
    text = str(model or "").strip().lower()
    return text.startswith(("wan2.5", "wan2.6", "wan2.7"))


def _supports_multi_shot(model: str) -> bool:
    return str(model or "").strip().lower().startswith("wan2.6")


def _supports_audio_toggle(model: str) -> bool:
    return str(model or "").strip().lower() == "wan2.6-i2v-flash"


def _video_synthesis_endpoint(config: WanConfig) -> str:
    return f"{config.api_root}/api/v1/services/aigc/video-generation/video-synthesis"


def _resolve_api_root(raw_value: str) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return BEIJING_DASHSCOPE_ROOT
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return text.rstrip("/")


def _json_headers(api_key: str, *, resolve_oss: bool = False, async_mode: bool = False) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if async_mode:
        headers["X-DashScope-Async"] = "enable"
    if resolve_oss:
        headers["X-DashScope-OssResourceResolve"] = "enable"
    return headers


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], *, stage: str) -> Dict[str, Any]:
    require_generation_access()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers=headers, method="POST")
    return _execute_json_request(request, service_name="wan", stage=stage)


def _get_json(url: str, headers: Dict[str, str], *, stage: str) -> Dict[str, Any]:
    request = Request(url, headers=headers, method="GET")
    return _execute_json_request(request, service_name="wan", stage=stage)


def _execute_json_request(request: Request, *, service_name: str, stage: str) -> Dict[str, Any]:
    require_generation_access()
    try:
        with urlopen(request, timeout=120) as response:
            raw_body = response.read()
            parsed = _decode_json_body(raw_body)
            _log_wan_response(stage, response.status, parsed)
    except HTTPError as exc:
        error_body = exc.read()
        detail = _decode_json_body(error_body)
        _log_wan_response(stage, exc.code, detail)
        raise WanRequestError(
            _error_detail_from_payload(
                stage=stage,
                payload=detail,
                http_status=exc.code,
                fallback_code=f"http_{exc.code}",
                fallback_message=_extract_message(detail) or _extract_reason(detail) or error_body.decode("utf-8", errors="ignore") or f"{service_name} http {exc.code}",
            )
        ) from exc
    except URLError as exc:
        raise WanRequestError(
            _error_detail(
                stage=stage,
                code="request_failed",
                message=f"{service_name} request failed: {exc.reason}",
            )
        ) from exc

    if isinstance(parsed, dict):
        return parsed
    raise WanRequestError(
        _error_detail(
            stage=stage,
            code="invalid_json_response",
            message=f"{service_name} response is not a JSON object",
        )
    )


def _decode_json_body(raw_body: bytes) -> Any:
    text = raw_body.decode("utf-8", errors="ignore").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}


def _extract_task_id(payload: Any) -> str:
    output = payload.get("output") if isinstance(payload, dict) else None
    if isinstance(output, dict):
        return str(output.get("task_id") or "").strip()
    return ""


def _extract_task_status(payload: Any) -> str:
    output = payload.get("output") if isinstance(payload, dict) else None
    if isinstance(output, dict):
        return str(output.get("task_status") or "").strip().upper()
    return ""


def _extract_video_url(payload: Any) -> str:
    output = payload.get("output") if isinstance(payload, dict) else None
    if not isinstance(output, dict):
        return ""
    direct_url = str(output.get("video_url") or output.get("url") or "").strip()
    if direct_url:
        return direct_url
    results = output.get("results")
    if isinstance(results, dict):
        return str(results.get("video_url") or "").strip()
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                text = str(item.get("video_url") or item.get("url") or "").strip()
                if text:
                    return text
    return ""


def _extract_message(payload: Any) -> str:
    if isinstance(payload, dict):
        output = payload.get("output")
        if isinstance(output, dict):
            for key in ("message", "msg", "code_message"):
                text = str(output.get(key) or "").strip()
                if text:
                    return text
        for key in ("message", "msg", "code_message", "error_message"):
            text = str(payload.get(key) or "").strip()
            if text:
                return text
    return ""


def _extract_reason(payload: Any) -> str:
    if isinstance(payload, dict):
        output = payload.get("output")
        if isinstance(output, dict):
            for key in ("reason", "detail", "details"):
                text = str(output.get(key) or "").strip()
                if text:
                    return text
        for key in ("reason", "detail", "details"):
            text = str(payload.get(key) or "").strip()
            if text:
                return text
    return ""


def _extract_request_id(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("request_id") or "").strip()
    return ""


def _extract_code(payload: Any) -> str:
    if isinstance(payload, dict):
        output = payload.get("output")
        if isinstance(output, dict):
            text = str(output.get("code") or "").strip()
            if text:
                return text
        return str(payload.get("code") or "").strip()
    return ""


def _error_detail_from_payload(
    *,
    stage: str,
    payload: Any,
    http_status: str | int = "",
    fallback_code: str = "",
    fallback_message: str = "",
    task_id: str = "",
    raw_status: str = "",
) -> Dict[str, Any]:
    return _error_detail(
        stage=stage,
        code=_extract_code(payload) or fallback_code,
        message=_extract_message(payload) or _extract_reason(payload) or fallback_message,
        request_id=_extract_request_id(payload),
        task_id=task_id or _extract_task_id(payload),
        http_status=http_status,
        raw_status=raw_status or _extract_task_status(payload),
    )


def _error_detail_from_exception(exc: Exception) -> Dict[str, Any]:
    message = str(exc or "").strip()
    stage = "request-exception"
    code = exc.__class__.__name__
    if message.startswith("image_url returned non-image MIME"):
        stage = "image-download"
        code = "non_image_mime"
    elif "image_url download failed" in message:
        stage = "image-download"
        code = "image_download_failed"
    elif "audio_url download failed" in message:
        stage = "audio-download"
        code = "audio_download_failed"
    elif "dashscope upload" in message:
        stage = "dashscope-oss-upload"
        code = "oss_upload_failed"
    elif "legacy wan s2v" in message:
        stage = "request-validation"
        code = "legacy_s2v_disabled"
    return _error_detail(stage=stage, code=code, message=message)


def _normalize_wan_error_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    if _is_wan_content_safety_failure(detail):
        normalized = dict(detail)
        normalized["code"] = CONTENT_SAFETY_ERROR_CODE
        if not str(normalized.get("message") or "").strip():
            normalized["message"] = "Input data may contain inappropriate content"
        normalized["raw_status"] = str(normalized.get("raw_status") or CONTENT_SAFETY_ERROR_CODE).strip()
        return normalized
    return detail


def _is_wan_content_safety_failure(detail: Dict[str, Any] | None = None, text: str = "") -> bool:
    payload = detail or {}
    combined = " ".join(
        str(item or "")
        for item in (
            text,
            payload.get("stage") if isinstance(payload, dict) else "",
            payload.get("code") if isinstance(payload, dict) else "",
            payload.get("message") if isinstance(payload, dict) else "",
            payload.get("raw_status") if isinstance(payload, dict) else "",
        )
    ).strip().lower()
    if not combined:
        return False
    return any(
        token in combined
        for token in (
            "inappropriate content",
            "input data may contain inappropriate content",
            "data inspection",
            "datainspection",
            "data_inspection",
            "data inspection failed",
            "content safety",
            "content_safety",
            "safety inspection",
            "risk content",
            "inappropriate",
            "内容安全",
            "安全检查",
            "不适宜",
            "违规内容",
        )
    )


def _is_non_retryable_wan_failure(detail: Dict[str, Any]) -> bool:
    code = str((detail or {}).get("code") or "").strip().lower()
    message = str((detail or {}).get("message") or "").strip().lower()
    http_status = str((detail or {}).get("http_status") or "").strip()
    raw_status = str((detail or {}).get("raw_status") or "").strip().upper()
    if _is_wan_content_safety_failure(detail):
        return True
    if raw_status in {"FAILED", "CANCELED", "CANCELLED"}:
        return True
    if code in {
        "datainspectionfailed",
        "invalidapikey",
        "invalidparameter",
        "invalidinput",
        "forbidden",
        "permissiondenied",
        "quotaexceeded",
        "throttling",
        "non_image_mime",
        "missing_image_url",
        "invalid_image_url",
    }:
        return True
    if http_status in {"400", "401", "403"}:
        return True
    return any(token in message for token in ("inappropriate content", "api key", "permission", "quota", "balance", "unsupported", "invalid"))


def _status_progress(task_status: str, current: float) -> float:
    if task_status in SUCCESS_STATUSES:
        return 100.0
    if task_status == "RUNNING":
        return min(95.0, max(current, 55.0) + 3.0)
    if task_status == "PENDING":
        return max(current, 10.0)
    return current


def _next_poll_interval_seconds(config: WanConfig, attempt: int) -> float:
    base_interval = max(0.1, float(config.poll_interval_seconds))
    max_interval = max(base_interval, float(config.poll_max_interval_seconds))
    return min(max_interval, base_interval * (1.18 ** attempt))


def _normalize_resolution(value: str) -> str:
    text = str(value or "").strip().upper()
    return text if text in {"480P", "720P", "1080P"} else "720P"

def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _normalize_shot_type(value: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"single", "multi"} else "multi"


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = env_value(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = env_value(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _bool_env(name: str, default: bool) -> bool:
    raw = str(env_value(name, "true" if default else "false")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _validate_i2v_preconditions(
    req: VeoGenerationRequest,
    config: WanConfig,
    prompt: str,
) -> VeoGenerationResponse | None:
    missing_dependencies = _missing_runtime_dependencies(config.model)
    if missing_dependencies:
        return _failed_response_from_detail(
            prompt=prompt,
            detail=_error_detail(
                stage="runtime-dependencies",
                code="missing_runtime_dependencies",
                message=f"missing runtime dependencies: {', '.join(missing_dependencies)}",
            ),
        )
    if not config.api_key:
        return _failed_response_from_detail(
            prompt=prompt,
            detail=_error_detail(
                stage="config",
                code="missing_api_key",
                message="missing Wanxiang API key environment variable; checked DASHSCOPE_API_KEY, QWEN_VL_API_KEY, WAN_API_KEY, WAN_I2V_API_KEY and compatible aliases",
            ),
        )
    if not str(req.image_url or "").strip():
        return _failed_response_from_detail(
            prompt=prompt,
            detail=_error_detail(
                stage="request-validation",
                code="missing_image_url",
                message=f"{config.model} requires image_url",
            ),
        )
    return None


def _response_has_playable_video(response: VeoGenerationResponse) -> bool:
    return str(response.status or "").lower() == "ready" and bool(
        str(response.video_url or "").strip() or str(response.normalized_video_url or "").strip()
    )


def _is_image_asset_validation_failure(error: str) -> bool:
    text = str(error or "").strip()
    return text.startswith("image_url returned non-image MIME")


def _failure_response_from_attempts(prompt: str, failures: List[Tuple[str, VeoGenerationResponse]]) -> VeoGenerationResponse:
    if not failures:
        return VeoGenerationResponse(
            status="failed",
            provider="wan",
            prompt=prompt,
            error="Wan-I2V generation failed before any model attempt",
            raw_status="i2v_not_attempted",
        )
    last_model, last_response = failures[-1]
    details = []
    for model, response in failures:
        reason = response.error or response.raw_status or response.status or "unknown failure"
        details.append(f"{model}: {reason}")
    error_detail = dict(last_response.error_detail or {})
    if not error_detail:
        error_detail = _error_detail(
            stage="wan-i2v-attempts",
            code=last_response.raw_status or "wan_i2v_failed",
            message=last_response.error or "Wan-I2V generation failed",
            task_id=last_response.id,
            raw_status=last_response.raw_status,
        )
    return VeoGenerationResponse(
        status="failed",
        provider="wan",
        id=last_response.id,
        video_url="",
        progress=last_response.progress,
        prompt=prompt,
        error="Wan-I2V generation failed: " + " | ".join(details),
        error_detail=error_detail,
        raw_status=last_response.raw_status or f"{last_model}:failed",
    )


def _missing_runtime_dependencies(model: str) -> List[str]:
    return []


def _log_remote_asset_preflight(label: str, asset_url: str, *, expected_content_prefixes: Tuple[str, ...]) -> None:
    raw_url = str(asset_url or "").strip()
    summary: Dict[str, Any] = {
        "label": label,
        "url_preview": safe_url_preview(raw_url),
        "status_code": None,
        "content_type": "",
        "content_length": "",
        "final_url_preview": {"present": False},
        "ok": False,
        "matches_expected_content_type": False,
        "error": "",
    }
    if not raw_url:
        summary["error"] = "missing_url"
        _log_wan_safe(model="", http_status="", task_id="", error="missing_url", stage="remote-asset-preflight", label=label)
        return

    try:
        response = requests.get(
            raw_url,
            stream=True,
            allow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        try:
            content_type = str(response.headers.get("content-type") or "").strip().lower()
            summary["status_code"] = response.status_code
            summary["content_type"] = content_type
            summary["content_length"] = str(response.headers.get("content-length") or "").strip()
            summary["final_url_preview"] = safe_url_preview(str(response.url or ""))
            summary["ok"] = response.ok
            summary["matches_expected_content_type"] = any(content_type.startswith(prefix) for prefix in expected_content_prefixes)
        finally:
            response.close()
    except Exception as exc:
        summary["error"] = exc.__class__.__name__

    _log_wan_safe(
        model="",
        http_status=str(summary["status_code"] or ""),
        task_id="",
        error=str(summary["error"] or ""),
        stage="remote-asset-preflight",
        label=label,
        url_preview=summary["url_preview"],
        final_url_preview=summary["final_url_preview"],
        content_type=summary["content_type"],
        content_length=summary["content_length"],
        ok=summary["ok"],
        matches_expected_content_type=summary["matches_expected_content_type"],
    )


def _normalized_content_type(content_type: str) -> str:
    return str(content_type or "").split(";", 1)[0].strip().lower()


def _detect_image_magic_type(first_bytes: bytes) -> str:
    data = bytes(first_bytes or b"")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    return ""


def _validate_image_asset_or_raise(*, content_type: str, first_bytes: bytes, source_label: str) -> str:
    normalized = _normalized_content_type(content_type)
    magic_type = _detect_image_magic_type(first_bytes)
    if normalized and normalized not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise RuntimeError(f"{source_label} returned non-image MIME {normalized}")
    if not magic_type:
        mime_text = normalized or "missing"
        raise RuntimeError(f"{source_label} returned non-image MIME {mime_text}; image magic invalid")
    return magic_type


def _download_asset_to_local_path(
    *,
    source_url: str,
    asset_kind: str,
    expected_content_prefixes: Tuple[str, ...],
    default_extension: str,
) -> Tuple[Path, str]:
    local_source_path = _resolve_local_asset_path(source_url)
    if local_source_path is not None:
        extension = local_source_path.suffix or default_extension
        content_type = _content_type_from_extension(extension, default_extension)
        if asset_kind == "image":
            with local_source_path.open("rb") as file_obj:
                detected_content_type = _validate_image_asset_or_raise(
                    content_type=content_type,
                    first_bytes=file_obj.read(32),
                    source_label="image_url",
                )
            content_type = detected_content_type
            extension = _extension_for_content_type(content_type, default_extension)
        token = hashlib.sha1(f"{asset_kind}:{local_source_path}".encode("utf-8")).hexdigest()[:24]
        local_path = WAN_UPLOAD_DIR / f"{asset_kind}_{token}{extension}"
        shutil.copyfile(local_source_path, local_path)
        _log_wan_safe(
            model="",
            http_status="",
            task_id="",
            error="",
            stage="asset-download-local",
            asset_kind=asset_kind,
            source_url_preview=safe_url_preview(source_url),
            content_type=content_type,
        )
        return local_path, content_type

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        with requests.get(source_url, headers=headers, stream=True, timeout=120, allow_redirects=True) as response:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(f"{asset_kind}_url download failed with HTTP {response.status_code}") from exc
            content_type = str(response.headers.get("Content-Type") or "").strip().lower()
            chunk_iter = response.iter_content(chunk_size=65536)
            first_chunk = b""
            for chunk in chunk_iter:
                if chunk:
                    first_chunk = chunk
                    break
            if asset_kind == "image":
                detected_content_type = _validate_image_asset_or_raise(
                    content_type=content_type,
                    first_bytes=first_chunk[:32],
                    source_label="image_url",
                )
                content_type = detected_content_type
            extension = _extension_for_content_type(content_type, default_extension)
            token = hashlib.sha1(f"{asset_kind}:{source_url}".encode("utf-8")).hexdigest()[:24]
            local_path = WAN_UPLOAD_DIR / f"{asset_kind}_{token}{extension}"
            with local_path.open("wb") as file_obj:
                if first_chunk:
                    file_obj.write(first_chunk)
                for chunk in chunk_iter:
                    if chunk:
                        file_obj.write(chunk)
    except requests.RequestException as exc:
        raise RuntimeError(f"{asset_kind}_url download failed: {exc.__class__.__name__}") from exc

    _log_wan_safe(
        model="",
        http_status=str(response.status_code),
        task_id="",
        error="",
        stage="asset-download-remote",
        asset_kind=asset_kind,
        source_url_preview=safe_url_preview(source_url),
        content_type=content_type,
    )
    return local_path, content_type


def _resolve_local_asset_path(source_url: str) -> Path | None:
    text = str(source_url or "").strip()
    if not text:
        return None
    if text.startswith("/static/"):
        static_root = (REPO_ROOT / "frontend" / "static").resolve()
        candidate = (REPO_ROOT / "frontend" / text.lstrip("/")).resolve()
        if not candidate.is_relative_to(static_root):
            return None
    elif text.startswith("file://"):
        candidate = Path(text[7:])
    else:
        candidate = Path(text)
    return candidate if candidate.exists() and candidate.is_file() else None


def _is_http_url(source_url: str) -> bool:
    parsed = urlparse(str(source_url or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_non_public_http_url(source_url: str) -> bool:
    parsed = urlparse(str(source_url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = (parsed.hostname or "").strip().lower()
    if host in {"localhost", "0.0.0.0"} or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local or address.is_reserved


def _content_type_from_extension(extension: str, default_extension: str) -> str:
    normalized = str(extension or default_extension).lower()
    if normalized == ".png":
        return "image/png"
    if normalized in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if normalized == ".webp":
        return "image/webp"
    if normalized == ".bmp":
        return "image/bmp"
    if normalized == ".mp3":
        return "audio/mpeg"
    if normalized == ".wav":
        return "audio/wav"
    return "application/octet-stream"


def _upload_local_file_to_dashscope_oss(
    *,
    api_key: str,
    api_root: str,
    model_name: str,
    local_path: Path,
    asset_kind: str,
    content_type: str,
) -> str:
    policy_url = f"{_resolve_api_root(api_root)}/api/v1/uploads"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params = {
        "action": "getPolicy",
        "model": model_name,
    }
    response = requests.get(policy_url, headers=headers, params=params, timeout=120)
    payload = _safe_response_json(response)
    _log_upload_policy_response(
        asset_kind=asset_kind,
        model_name=model_name,
        url=policy_url,
        status_code=response.status_code,
        payload=payload,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = _extract_message(payload) or _extract_reason(payload) or str(exc)
        raise WanRequestError(
            _error_detail_from_payload(
                stage="dashscope-oss-policy",
                payload=payload,
                http_status=response.status_code,
                fallback_code=f"http_{response.status_code}",
                fallback_message=f"dashscope upload/getPolicy failed for model {model_name}: {message}",
            )
        ) from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise WanRequestError(
            _error_detail(
                stage="dashscope-oss-policy",
                code="missing_policy_data",
                message="dashscope upload policy response missing data",
            )
        )

    key = f"{data['upload_dir']}/{local_path.name}"
    with local_path.open("rb") as file_obj:
        files = {
            "OSSAccessKeyId": (None, data["oss_access_key_id"]),
            "Signature": (None, data["signature"]),
            "policy": (None, data["policy"]),
            "x-oss-object-acl": (None, data["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": (None, data["x_oss_forbid_overwrite"]),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (local_path.name, file_obj, content_type or "application/octet-stream"),
        }
        require_generation_access()
        upload_response = requests.post(data["upload_host"], files=files, timeout=120)
        _log_upload_file_response(
            asset_kind=asset_kind,
            model_name=model_name,
            status_code=upload_response.status_code,
            upload_host=str(data.get("upload_host", "")),
        )
        try:
            upload_response.raise_for_status()
        except requests.HTTPError as exc:
            raise WanRequestError(
                _error_detail(
                    stage="dashscope-oss-upload",
                    code=f"http_{upload_response.status_code}",
                    message=f"dashscope OSS upload failed with HTTP {upload_response.status_code}",
                    http_status=upload_response.status_code,
                )
            ) from exc

    oss_url = f"oss://{key}"
    _log_wan_safe(model=model_name, http_status=str(upload_response.status_code), task_id="", error="")
    return oss_url


def _safe_response_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"message": response.text[:500]}


def _log_upload_policy_response(*, asset_kind: str, model_name: str, url: str, status_code: int, payload: Any) -> None:
    error = _extract_message(payload) or _extract_reason(payload)
    _log_wan_safe(model=model_name, http_status=str(status_code), task_id="", error=error)


def _log_upload_file_response(*, asset_kind: str, model_name: str, status_code: int, upload_host: str) -> None:
    _log_wan_safe(model=model_name, http_status=str(status_code), task_id="", error="")


def _extension_for_content_type(content_type: str, default_extension: str) -> str:
    normalized = str(content_type or "").lower()
    if normalized.startswith("image/png"):
        return ".png"
    if normalized.startswith("image/jpeg") or normalized.startswith("image/jpg"):
        return ".jpg"
    if normalized.startswith("image/webp"):
        return ".webp"
    if normalized.startswith("image/bmp"):
        return ".bmp"
    if normalized.startswith("audio/mpeg") or normalized.startswith("audio/mp3"):
        return ".mp3"
    if normalized.startswith("audio/wav") or normalized.startswith("audio/x-wav") or normalized.startswith("audio/wave") or normalized.startswith("audio/vnd.wave"):
        return ".wav"
    return default_extension


def _contains_oss_url(payload: Any) -> bool:
    if isinstance(payload, dict):
        return any(_contains_oss_url(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_oss_url(item) for item in payload)
    return str(payload or "").strip().startswith("oss://")


def _log_wan_config(config: WanConfig) -> None:
    _log_wan_safe(
        model=config.model,
        api_key_present=bool(config.api_key),
        api_key_env_name=_env_name_from_source(config.api_key_source),
        http_status="",
        task_id="",
        error="",
        stage="config",
        resolution=config.resolution,
        duration=config.duration,
        audio=config.audio_enabled,
        prompt_extend=config.prompt_extend,
        poll_max_attempts=config.poll_max_attempts,
        poll_timeout_seconds=config.poll_timeout_seconds,
    )


def _payload_first_frame_url(payload: Dict[str, Any]) -> str:
    input_payload = payload.get("input") if isinstance(payload, dict) else None
    if not isinstance(input_payload, dict):
        return ""
    img_url = str(input_payload.get("img_url") or "").strip()
    if img_url:
        return img_url
    media = input_payload.get("media")
    if isinstance(media, list):
        for item in media:
            if isinstance(item, dict) and str(item.get("type") or "") == "first_frame":
                return str(item.get("url") or "").strip()
    return ""


def _log_wan_request(stage: str, payload: Dict[str, Any], *, source_image_url: str = "") -> None:
    first_frame_url = _payload_first_frame_url(payload)
    input_payload = payload.get("input") if isinstance(payload, dict) else {}
    parameters = payload.get("parameters") if isinstance(payload, dict) else {}
    media = input_payload.get("media") if isinstance(input_payload, dict) else None
    _log_wan_safe(
        model=str(payload.get("model") or ""),
        http_status="",
        task_id="",
        error="",
        stage=stage,
        request_path="i2v" if first_frame_url else "t2v",
        input_keys=sorted(input_payload.keys()) if isinstance(input_payload, dict) else [],
        media_types=[str(item.get("type") or "") for item in media if isinstance(item, dict)] if isinstance(media, list) else [],
        parameters={
            "resolution": parameters.get("resolution") if isinstance(parameters, dict) else "",
            "duration": parameters.get("duration") if isinstance(parameters, dict) else "",
            "audio": parameters.get("audio") if isinstance(parameters, dict) else "",
            "prompt_extend": parameters.get("prompt_extend") if isinstance(parameters, dict) else "",
            "shot_type": parameters.get("shot_type") if isinstance(parameters, dict) else "",
            "watermark": parameters.get("watermark") if isinstance(parameters, dict) else "",
        },
        source_image_url_present=bool(str(source_image_url or "").strip()),
        source_image_url_preview=safe_url_preview(source_image_url),
        payload_first_frame_present=bool(first_frame_url),
        payload_first_frame_preview=safe_url_preview(first_frame_url),
        prompt_present=bool(input_payload.get("prompt")) if isinstance(input_payload, dict) else False,
        prompt_chars=len(str(input_payload.get("prompt") or "")) if isinstance(input_payload, dict) else 0,
        negative_prompt_present=bool(input_payload.get("negative_prompt")) if isinstance(input_payload, dict) else False,
        negative_prompt_chars=len(str(input_payload.get("negative_prompt") or "")) if isinstance(input_payload, dict) else 0,
    )


def _log_i2v_attempt(stage: str, *, model: str, fallback: bool, previous_error: str = "") -> None:
    _log_wan_safe(model=model, http_status="", task_id="", error=str(previous_error or "").strip()[:500], stage=stage, fallback=fallback)


def _log_i2v_model_coercion(configured_model: str, fallback_model: str) -> None:
    _log_wan_safe(model=fallback_model, http_status="", task_id="", error="legacy_s2v_disabled_for_poetry_i2v")


def _log_wan_response(stage: str, status_code: int, payload: Any) -> None:
    _log_wan_safe(
        model="",
        http_status=str(status_code),
        task_id=_extract_task_id(payload),
        error=_extract_message(payload) or _extract_reason(payload),
        stage=stage,
        error_code=_extract_code(payload),
        message=_extract_message(payload),
        reason=_extract_reason(payload),
        task_status=_extract_task_status(payload),
        request_id=_extract_request_id(payload),
        video_url_present=bool(_extract_video_url(payload)),
        video_url_preview=safe_url_preview(_extract_video_url(payload)),
    )


def _log_poll_summary(
    *,
    attempt: int,
    elapsed_seconds: float,
    next_sleep_seconds: float,
    task_id: str,
    task_status: str,
    progress: float,
    video_url: str,
    error: str,
) -> None:
    _log_wan_safe(
        model="",
        http_status="",
        task_id=task_id,
        error=str(error or "").strip(),
        stage="task-query-poll",
        attempt=attempt,
        elapsed_seconds=round(float(elapsed_seconds), 2),
        next_sleep_seconds=round(float(next_sleep_seconds), 2),
        task_status=task_status,
        progress=progress,
        video_url_present=bool(str(video_url or "").strip()),
        video_url_preview=safe_url_preview(video_url),
    )


def _log_wan_safe(
    *,
    model: str,
    api_key_present: bool | None = None,
    api_key_env_name: str = "",
    http_status: str | int = "",
    task_id: str = "",
    error: str = "",
    **extra: Any,
) -> None:
    summary = {
        "provider": "wan",
        "model": str(model or "").strip(),
        "api_key_present": bool(api_key_present) if api_key_present is not None else bool(first_env_value(WAN_API_KEY_ENV_NAMES, "")[0]),
        "api_key_env_name": str(api_key_env_name or "").strip() or _env_name_from_source(first_env_value(WAN_API_KEY_ENV_NAMES, "")[1]),
        "http_status": str(http_status or "").strip(),
        "task_id": str(task_id or "").strip(),
        "error": str(error or "").strip(),
    }
    for key, value in extra.items():
        summary[key] = value
    print(f"[wan] {json.dumps(summary, ensure_ascii=False)}", flush=True)


def _env_name_from_source(source: str) -> str:
    text = str(source or "").strip()
    if not text:
        return ""
    return text.rsplit(":", 1)[-1].strip()
