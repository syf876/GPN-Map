from backend.admin_auth import require_generation_access
import json

from backend.types.schema import PoetryImageRequest, VeoGenerationRequest, VeoGenerationResponse

from .env import WAN_API_KEY_ENV_NAMES, env_value, first_env_value
from .logging_utils import safe_url_preview


I2V_PROVIDER_NAMES = {
    "wan",
    "wan_i2v",
    "wan-i2v",
    "wan2.7-i2v",
    "wan2.7-i2v-2026-04-25",
    "wan2.6-i2v",
    "wan2.6-i2v-flash",
    "wan2.5-i2v-preview",
    "wan2.2-i2v-plus",
    "wan2.2-i2v-flash",
    "wanx2.1-i2v-turbo",
    "wanx2.1-i2v-plus",
}


def generate_poetry_video(req: VeoGenerationRequest) -> VeoGenerationResponse:
    require_generation_access()
    provider = "wan"
    api_key, api_key_source = first_env_value(WAN_API_KEY_ENV_NAMES, "")
    _log_video_event(
        provider=provider,
        model=_response_model(provider),
        api_key_present=bool(api_key),
        api_key_env_name=_env_name_from_source(api_key_source),
        http_status="",
        task_id="",
        error="",
        **_request_log_fields(req),
    )
    preflight_error = _validate_poetry_video_request(req, provider)
    if preflight_error is not None:
        _log_video_event(
            provider=provider,
            model=_response_model(provider),
            api_key_present=bool(api_key),
            api_key_env_name=_env_name_from_source(api_key_source),
            http_status="",
            task_id="",
            error=preflight_error.error,
        )
        return preflight_error

    _log_video_event(
        provider=provider,
        model=_response_model(provider),
        api_key_present=bool(api_key),
        api_key_env_name=_env_name_from_source(api_key_source),
        http_status="",
        task_id="",
        error="",
        **_request_log_fields(req),
    )
    initial_response = _generate_wan_video(req)
    if _is_wan_content_safety_response(initial_response) and not _is_safe_video_request(req):
        _log_video_event(
            provider=provider,
            model=_response_model(provider),
            api_key_present=bool(api_key),
            api_key_env_name=_env_name_from_source(api_key_source),
            http_status=str((initial_response.error_detail or {}).get("http_status") or ""),
            task_id=initial_response.id,
            error="content safety fallback: regenerating safe first frame and safe video prompt",
            safe_fallback=True,
            **_error_detail_log_fields(initial_response.error_detail),
        )
        fallback_response = _generate_safe_first_frame_and_video(req, initial_response)
        return _finalize_response(_normalize_generation_response(fallback_response), requested_provider=provider)
    return _finalize_response(_normalize_generation_response(initial_response), requested_provider=provider)


def _normalize_generation_response(response: VeoGenerationResponse) -> VeoGenerationResponse:
    from .normalize import normalize_generation_response

    return normalize_generation_response(response)


def _build_veo_prompt(req: VeoGenerationRequest) -> str:
    from .prompt_builder import build_veo_prompt

    return build_veo_prompt(req)


def _generate_wan_video(req: VeoGenerationRequest) -> VeoGenerationResponse:
    from .wan_client import generate_wan_video

    return generate_wan_video(req)


def _generate_safe_first_frame_and_video(
    req: VeoGenerationRequest,
    initial_response: VeoGenerationResponse,
) -> VeoGenerationResponse:
    require_generation_access()
    from backend.service.image_generation.nano_banana import generate_nano_banana_poetry_image

    prediction = req.prediction if isinstance(req.prediction, dict) else {}
    safe_image_request = PoetryImageRequest(
        poem=str(req.narration or req.text or prediction.get("text") or "").strip() or "河西走廊古道云影",
        place=str(req.location or _prediction_label(prediction.get("place")) or "").strip(),
        dynasty=str(req.dynasty or _prediction_label(prediction.get("dynasty")) or "").strip(),
        emotion=str(req.emotion or _prediction_label(prediction.get("emotion")) or "").strip(),
        imagery=list(req.imagery or []),
        mode="safe_i2v_first_frame",
    )
    safe_image = generate_nano_banana_poetry_image(safe_image_request)
    safe_image_url = str(safe_image.image_url or "").strip()
    if not safe_image_url or str(safe_image.status or "").strip().lower() not in {"succeeded", "success", "completed", "complete", "done", "finished", "ready"}:
        detail = {
            "stage": "safe-first-frame",
            "code": "safe_first_frame_failed",
            "message": safe_image.error or "safe first frame generation failed",
            "task_id": str(initial_response.id or ""),
            "raw_status": str(safe_image.status or "failed"),
        }
        return VeoGenerationResponse(
            status="failed",
            provider="wan",
            prompt=initial_response.prompt,
            error="内容安全检查未通过，安全版首帧生成失败，请更换诗句或重新生成更简洁的风景首帧",
            error_detail=detail,
            raw_status="safe_first_frame_failed",
        )

    safe_prediction = dict(prediction)
    safe_prediction["safe_i2v_first_frame"] = True
    safe_prediction["safe_video_prompt"] = True
    safe_prediction["unsafe_wan_task_id"] = initial_response.id
    safe_req = req.copy(
        update={
            "image_url": safe_image_url,
            "prediction": safe_prediction,
        }
    )
    _log_video_event(
        provider="wan",
        model=_response_model("wan"),
        http_status="",
        task_id="",
        error="safe fallback submitting regenerated first frame",
        safe_fallback=True,
        safe_image_url_present=True,
        safe_image_url_preview=safe_url_preview(safe_image_url),
    )
    safe_response = _generate_wan_video(safe_req)
    if _is_wan_content_safety_response(safe_response):
        detail = dict(safe_response.error_detail or {})
        detail["stage"] = detail.get("stage") or "safe-video-content-safety"
        detail["code"] = detail.get("code") or "content_safety"
        detail["safe_image_url_preview"] = safe_url_preview(safe_image_url)
        return safe_response.copy(
            update={
                "status": "failed",
                "video_url": "",
                "normalized_video_url": "",
                "error": "内容安全检查未通过，已尝试安全版首帧，请更换诗句或重新生成更简洁的风景首帧",
                "error_detail": detail,
                "raw_status": "content_safety_after_safe_fallback",
            }
        )
    return safe_response


def _validate_poetry_video_request(req: VeoGenerationRequest, provider: str) -> VeoGenerationResponse | None:
    image_url = str(req.image_url or "").strip()
    if not image_url:
        return VeoGenerationResponse(
            status="failed",
            provider=provider,
            prompt=_build_veo_prompt(req),
            error="poetry video generation requires a real image_url from a successful poetry image generation step",
            raw_status="missing_image_url",
        )
    if _looks_like_mock_image_url(image_url):
        return VeoGenerationResponse(
            status="failed",
            provider=provider,
            prompt=_build_veo_prompt(req),
            error="poetry video generation requires a real generated image_url; mock or local generated image URLs are not accepted",
            raw_status="mock_image_url_rejected",
        )
    credential_error = _validate_provider_credentials(provider, req)
    if credential_error is not None:
        return credential_error
    return None


def _looks_like_mock_image_url(image_url: str) -> bool:
    text = str(image_url or "").strip().lower()
    return (
        text.startswith("mock:")
        or "mock_poetry" in text
        or "/api/poetry-image/generated/" in text
    )


def _validate_provider_credentials(provider: str, req: VeoGenerationRequest) -> VeoGenerationResponse | None:
    prompt = _build_veo_prompt(req)
    if provider in I2V_PROVIDER_NAMES:
        api_key, api_key_source = first_env_value(WAN_API_KEY_ENV_NAMES, "")
        _log_video_event(
            provider="wan",
            model=_response_model("wan"),
            api_key_present=bool(api_key),
            api_key_env_name=_env_name_from_source(api_key_source),
            http_status="",
            task_id="",
            error="",
        )
        if not api_key:
            return VeoGenerationResponse(
                status="failed",
                provider="wan",
                prompt=prompt,
                error="missing Wanxiang API key environment variable; checked DASHSCOPE_API_KEY, QWEN_VL_API_KEY, WAN_API_KEY, WAN_I2V_API_KEY and compatible aliases",
                raw_status="missing_api_key",
            )
    if provider == "jimeng" and not (env_value("JIMENG_ACCESS_KEY_ID", "") and env_value("JIMENG_SECRET_ACCESS_KEY", "")):
        return VeoGenerationResponse(
            status="failed",
            provider="jimeng",
            prompt=prompt,
            error="missing JIMENG_ACCESS_KEY_ID or JIMENG_SECRET_ACCESS_KEY environment variable",
            raw_status="missing_credentials",
        )
    return None


def _finalize_response(response: VeoGenerationResponse, *, requested_provider: str) -> VeoGenerationResponse:
    if _looks_like_mock_video_response(response):
        response = response.copy(
            update={
                "status": "failed",
                "provider": response.provider or requested_provider,
                "video_url": "",
                "normalized_video_url": "",
                "error": "mock video responses are disabled for poetry video generation",
                "raw_status": "mock_video_rejected",
            }
        )
    api_key, api_key_source = first_env_value(WAN_API_KEY_ENV_NAMES, "")
    _log_video_event(
        provider=response.provider or requested_provider,
        model=_response_model(response.provider or requested_provider),
        api_key_present=bool(api_key) if (response.provider or requested_provider) == "wan" else False,
        api_key_env_name=_env_name_from_source(api_key_source) if (response.provider or requested_provider) == "wan" else "",
        http_status="",
        task_id=response.id,
        error=response.error,
        **_error_detail_log_fields(response.error_detail),
    )
    return response


def _looks_like_mock_video_response(response: VeoGenerationResponse) -> bool:
    provider = str(response.provider or "").strip().lower()
    task_id = str(response.id or "").strip().lower()
    video_url = str(response.video_url or "").strip().lower()
    normalized_url = str(response.normalized_video_url or "").strip().lower()
    return (
        provider == "mock"
        or task_id.startswith("mock-")
        or video_url.startswith("mock:")
        or normalized_url.startswith("mock:")
    )


def _is_safe_video_request(req: VeoGenerationRequest) -> bool:
    prediction = req.prediction if isinstance(req.prediction, dict) else {}
    return bool(prediction.get("safe_video_prompt") or prediction.get("safe_i2v_first_frame"))


def _is_wan_content_safety_response(response: VeoGenerationResponse) -> bool:
    detail = response.error_detail if isinstance(response.error_detail, dict) else {}
    text = " ".join(
        str(item or "")
        for item in (
            response.error,
            response.raw_status,
            detail.get("stage"),
            detail.get("code"),
            detail.get("message"),
            detail.get("raw_status"),
        )
    ).lower()
    return any(
        token in text
        for token in (
            "content_safety",
            "content safety",
            "inappropriate content",
            "input data may contain inappropriate content",
            "data inspection",
            "datainspection",
            "安全检查",
            "内容安全",
            "违规内容",
            "不适宜",
        )
    )


def _response_model(provider: str) -> str:
    text = str(provider or "").strip().lower()
    if text in I2V_PROVIDER_NAMES or text == "wan":
        return env_value("WAN_I2V_MODEL", "") or env_value("WAN_MODEL", "") or "wan2.6-i2v-flash"
    if text == "veo":
        return env_value("VEO_MODEL", "veo3.1-fast")
    if text == "jimeng":
        return env_value("JIMENG_REQ_KEY", "jimeng_t2v_v30_1080p")
    return ""


def _log_video_event(**payload) -> None:
    summary = {
        "provider": payload.get("provider", "wan"),
        "model": payload.get("model", ""),
        "api_key_present": bool(payload.get("api_key_present", False)),
        "api_key_env_name": payload.get("api_key_env_name", ""),
        "http_status": payload.get("http_status", ""),
        "task_id": payload.get("task_id", ""),
        "error": str(payload.get("error", "") or "").strip(),
    }
    for key in (
        "poem_key",
        "title",
        "content_preview",
        "image_url_present",
        "image_url_preview",
        "audio_url_present",
        "audio_url_preview",
        "audio",
        "resolution",
        "duration",
        "requested_model",
        "imagery",
        "location",
        "dynasty",
        "emotion",
        "error_stage",
        "error_code",
        "error_message",
        "request_id",
        "raw_status",
        "safe_fallback",
        "safe_image_url_present",
        "safe_image_url_preview",
    ):
        if key in payload:
            summary[key] = payload.get(key)
    print(f"[poetry_video] {json.dumps(summary, ensure_ascii=False, default=str)}", flush=True)


def _env_name_from_source(source: str) -> str:
    text = str(source or "").strip()
    if not text:
        return ""
    return text.rsplit(":", 1)[-1].strip()


def _prediction_label(value) -> str:
    if isinstance(value, dict):
        return str(value.get("label") or value.get("name") or value.get("id") or "").strip()
    return str(value or "").strip()


def _poem_key(req: VeoGenerationRequest) -> str:
    text = str(req.text or req.narration or "").strip()
    if not text:
        return ""
    return text[:80]


def _request_log_fields(req: VeoGenerationRequest) -> dict:
    prediction = req.prediction if isinstance(req.prediction, dict) else {}
    title = str(prediction.get("title") or prediction.get("poem_title") or "").strip()
    content = str(req.text or req.narration or "").strip()
    image_url = str(req.image_url or "").strip()
    audio_url = str(req.audio_url or "").strip()
    return {
        "poem_key": _poem_key(req),
        "title": title,
        "content_preview": content[:120],
        "image_url_present": bool(image_url),
        "image_url_preview": safe_url_preview(image_url),
        "audio_url_present": bool(audio_url),
        "audio_url_preview": safe_url_preview(audio_url),
        "audio": bool(audio_url) or bool(prediction.get("audio", False)),
        "resolution": str(req.resolution or "").strip(),
        "duration": prediction.get("duration", ""),
        "requested_model": str(prediction.get("model") or prediction.get("video_model") or "").strip(),
        "imagery": req.imagery,
        "location": str(req.location or "").strip(),
        "dynasty": str(req.dynasty or "").strip(),
        "emotion": str(req.emotion or "").strip(),
    }


def _error_detail_log_fields(detail: dict | None) -> dict:
    if not isinstance(detail, dict) or not detail:
        return {}
    return {
        "error_stage": str(detail.get("stage") or "").strip(),
        "error_code": str(detail.get("code") or "").strip(),
        "error_message": str(detail.get("message") or "").strip()[:500],
        "request_id": str(detail.get("request_id") or "").strip(),
        "raw_status": str(detail.get("raw_status") or "").strip(),
    }
