from backend.admin_auth import require_generation_access
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.types.schema import VeoGenerationRequest, VeoGenerationResponse

from .env import VEO_API_KEY_ENV_NAMES, env_lookup, env_presence, env_value, first_env_value
from .prompt_builder import build_veo_prompt


SUCCESS_STATUSES = {"succeeded", "success", "completed", "ready", "done", "finished"}
FAILURE_STATUSES = {"failed", "error", "cancelled", "canceled", "timeout"}
VEO_API_BASE_ENV_NAMES = (
    "VEO_API_BASE",
)


@dataclass(frozen=True)
class VeoConfig:
    api_base: str
    api_base_source: str
    api_key: str
    api_key_source: str
    model: str
    submit_path: str
    result_path: str
    result_method: str
    aspect_ratio: str
    image_field: str
    duration: int
    resolution: str
    poll_max_attempts: int
    poll_interval_seconds: float


def generate_veo_video(req: VeoGenerationRequest) -> VeoGenerationResponse:
    require_generation_access()
    prompt = build_veo_prompt(req)
    config = _load_config()
    if not config.api_key:
        return VeoGenerationResponse(
            status="failed",
            provider="veo",
            prompt=prompt,
            error="missing VEO API key environment variable; checked VEO_API_KEY",
            raw_status="missing_api_key",
        )
    image_url = str(req.image_url or "").strip()
    if not image_url:
        return VeoGenerationResponse(
            status="failed",
            provider="veo",
            prompt=prompt,
            error="veo image-to-video generation requires image_url from the successful poetry image step",
            raw_status="missing_image_url",
        )

    create_url = f"{config.api_base}{config.submit_path}"
    payload = {
        "model": config.model,
        "prompt": prompt,
        config.image_field: image_url,
        "aspectRatio": config.aspect_ratio,
        "duration": config.duration,
        "resolution": config.resolution,
        "shutProgress": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    try:
        create_response = _post_json(create_url, payload, headers, stage="submit")
        create_status = _extract_status(create_response)
        create_raw_status = str(create_status or "")
        create_id = _extract_id(create_response)
        create_video_url = _extract_video_url(create_response)
        create_progress = _extract_progress(create_response)
        create_error = _provider_error(create_response)
        if create_error:
            return VeoGenerationResponse(
                status="failed",
                provider="veo",
                id=create_id,
                video_url="",
                progress=create_progress,
                prompt=prompt,
                error=create_error,
                raw_status=create_raw_status or "submit_failed",
            )

        if create_video_url:
            return VeoGenerationResponse(
                status="ready",
                provider="veo",
                id=create_id,
                video_url=create_video_url,
                progress=create_progress,
                prompt=prompt,
                error="",
                raw_status=create_raw_status or "ready",
            )

        if create_status in FAILURE_STATUSES:
            return VeoGenerationResponse(
                status="failed",
                provider="veo",
                id=create_id,
                video_url="",
                progress=create_progress,
                prompt=prompt,
                error=_extract_error(create_response) or "veo generation failed",
                raw_status=create_raw_status,
            )

        if not create_id:
            return VeoGenerationResponse(
                status="failed",
                provider="veo",
                id="",
                video_url="",
                progress=create_progress,
                prompt=prompt,
                error="veo create response missing both video url and task id",
                raw_status=create_raw_status or "missing_id",
            )

        last_progress = create_progress
        last_raw_status = create_raw_status or "submitted"
        for attempt in range(config.poll_max_attempts):
            result_response = _query_result(config, create_id, headers)
            result_status = _extract_status(result_response)
            result_raw_status = str(result_status or last_raw_status or "")
            result_video_url = _extract_video_url(result_response)
            result_progress = _extract_progress(result_response)
            result_error = _provider_error(result_response)
            if result_progress > 0:
                last_progress = result_progress
            last_raw_status = result_raw_status or last_raw_status

            if result_error:
                return VeoGenerationResponse(
                    status="failed",
                    provider="veo",
                    id=create_id,
                    video_url="",
                    progress=last_progress,
                    prompt=prompt,
                    error=result_error,
                    raw_status=last_raw_status or "poll_failed",
                )

            if result_video_url and (result_status in SUCCESS_STATUSES or not result_status):
                return VeoGenerationResponse(
                    status="ready",
                    provider="veo",
                    id=create_id,
                    video_url=result_video_url,
                    progress=last_progress,
                    prompt=prompt,
                    error="",
                    raw_status=last_raw_status or "ready",
                )

            if result_status in FAILURE_STATUSES:
                return VeoGenerationResponse(
                    status="failed",
                    provider="veo",
                    id=create_id,
                    video_url="",
                    progress=last_progress,
                    prompt=prompt,
                    error=_extract_error(result_response) or "veo generation failed during polling",
                    raw_status=last_raw_status,
                )

            if attempt < config.poll_max_attempts - 1:
                time.sleep(config.poll_interval_seconds)

        return VeoGenerationResponse(
            status="failed",
            provider="veo",
            id=create_id,
            video_url="",
            progress=last_progress,
            prompt=prompt,
            error="veo polling timed out before reaching a terminal status",
            raw_status=last_raw_status or "timeout",
        )
    except Exception as exc:
        return VeoGenerationResponse(
            status="failed",
            provider="veo",
            prompt=prompt,
            error=str(exc),
            raw_status="request_exception",
        )


def _load_config() -> VeoConfig:
    api_key, api_key_source = first_env_value(VEO_API_KEY_ENV_NAMES, "")
    api_base, api_base_source = _first_env_lookup(VEO_API_BASE_ENV_NAMES, "")
    submit_path = _normalize_path(env_value("VEO_SUBMIT_PATH", "/v1/videos") or "/v1/videos")
    result_path = _normalize_path(env_value("VEO_RESULT_PATH", "/v1/videos/{task_id}") or "/v1/videos/{task_id}")
    config = VeoConfig(
        api_base=api_base.rstrip("/"),
        api_base_source=api_base_source,
        api_key=api_key,
        api_key_source=api_key_source,
        model=env_value("VEO_MODEL", "veo3.1-fast") or "veo3.1-fast",
        submit_path=submit_path,
        result_path=result_path,
        result_method=str(env_value("VEO_RESULT_METHOD", "GET") or "GET").strip().upper(),
        aspect_ratio=env_value("VEO_ASPECT_RATIO", "16:9") or "16:9",
        image_field=env_value("VEO_IMAGE_FIELD", "image") or "image",
        duration=_int_env("VEO_DURATION", 8, minimum=1),
        resolution=env_value("VEO_RESOLUTION", "720p") or "720p",
        poll_max_attempts=_int_env("VEO_POLL_MAX_ATTEMPTS", 60, minimum=1),
        poll_interval_seconds=_float_env("VEO_POLL_INTERVAL_SECONDS", 3.0, minimum=0.1),
    )
    _log_veo_config(config)
    return config


def _log_veo_config(config: VeoConfig) -> None:
    summary = {
        "api_base": config.api_base,
        "api_base_source": config.api_base_source,
        "model": config.model,
        "submit_path": config.submit_path,
        "result_path": config.result_path,
        "result_method": config.result_method,
        "image_field": config.image_field,
        "duration": config.duration,
        "resolution": config.resolution,
        "api_key_present": bool(config.api_key),
        "api_key_source": config.api_key_source,
        "checked_variables": env_presence(VEO_API_KEY_ENV_NAMES),
    }
    print(f"[veo_config] {json.dumps(summary, ensure_ascii=False)}", flush=True)


def _query_result(config: VeoConfig, task_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
    path = config.result_path.replace("{task_id}", task_id)
    url = f"{config.api_base}{path}"
    if config.result_method == "GET":
        return _get_json(url, headers, stage="poll", task_id=task_id)
    body = {"taskId": task_id} if "check-result" in path else {"id": task_id}
    return _post_json(url, body, headers, stage="poll", task_id=task_id)


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], *, stage: str, task_id: str = "") -> Dict[str, Any]:
    require_generation_access()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=120) as response:
            raw_body = response.read()
            parsed = _decode_json_body(raw_body)
            _log_veo_http(
                stage,
                url=url,
                http_status=response.status,
                payload=parsed,
                task_id=task_id or _extract_id(parsed),
                request_payload=payload,
            )
    except HTTPError as exc:
        error_body = exc.read()
        detail = _decode_json_body(error_body)
        _log_veo_http(
            stage,
            url=url,
            http_status=exc.code,
            payload=detail,
            task_id=task_id or _extract_id(detail),
            request_payload=payload,
        )
        message = _extract_error(detail) or error_body.decode("utf-8", errors="ignore") or str(exc)
        raise RuntimeError(f"veo http {exc.code}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"veo request failed: {exc.reason}") from exc

    if isinstance(parsed, dict):
        return parsed
    raise RuntimeError("veo response is not a JSON object")


def _get_json(url: str, headers: Dict[str, str], *, stage: str, task_id: str = "") -> Dict[str, Any]:
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=120) as response:
            raw_body = response.read()
            parsed = _decode_json_body(raw_body)
            _log_veo_http(
                stage,
                url=url,
                http_status=response.status,
                payload=parsed,
                task_id=task_id or _extract_id(parsed),
                request_payload={"model": "", "image": ""},
            )
    except HTTPError as exc:
        error_body = exc.read()
        detail = _decode_json_body(error_body)
        _log_veo_http(
            stage,
            url=url,
            http_status=exc.code,
            payload=detail,
            task_id=task_id or _extract_id(detail),
            request_payload={"model": "", "image": ""},
        )
        message = _extract_error(detail) or error_body.decode("utf-8", errors="ignore") or str(exc)
        raise RuntimeError(f"veo http {exc.code}: {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"veo request failed: {exc.reason}") from exc

    if isinstance(parsed, dict):
        return parsed
    raise RuntimeError("veo response is not a JSON object")


def _first_env_lookup(names: Iterable[str], default: str) -> tuple[str, str]:
    for name in names:
        value, source = env_lookup(name)
        if value:
            return value, source
    return default, "default"


def _decode_json_body(raw_body: bytes) -> Any:
    text = raw_body.decode("utf-8", errors="ignore").strip()
    if not text:
        return {}
    if text.startswith("data:"):
        events = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data_text = line[len("data:") :].strip()
            if not data_text:
                continue
            try:
                events.append(json.loads(data_text))
            except json.JSONDecodeError:
                events.append({"message": data_text})
        if events:
            return events[-1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}


def _normalize_path(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "/"
    return text if text.startswith("/") else f"/{text}"


def _extract_id(payload: Any) -> str:
    return _first_text(payload, ("id", "task_id", "taskId", "job_id", "jobId"))


def _extract_status(payload: Any) -> str:
    return _normalize_status(_first_text(payload, ("status", "state")))


def _extract_video_url(payload: Any) -> str:
    for url in _video_url_candidates(payload):
        if url.startswith("http://") or url.startswith("https://"):
            return url
    return ""


def _video_url_candidates(payload: Any) -> list[str]:
    keys = {
        "video_url",
        "videourl",
        "file_url",
        "fileurl",
        "result_url",
        "resulturl",
        "download_url",
        "downloadurl",
        "url",
    }
    candidates: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                normalized = str(key).replace("-", "_").lower()
                if normalized in keys:
                    if isinstance(value, str) and value.strip():
                        candidates.append(value.strip())
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and item.strip():
                                candidates.append(item.strip())
                if isinstance(value, (dict, list)):
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)
    return candidates


def _extract_progress(payload: Any) -> float:
    value = _first_value(payload, ("progress", "percent", "percentage"))
    if value is None:
        return 0.0
    try:
        progress = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, progress))


def _extract_error(payload: Any) -> str:
    return _first_text(payload, ("error", "message", "msg", "detail", "reason"))


def _provider_error(payload: Any) -> str:
    code = _first_value(payload, ("code", "error_code", "errorCode", "errcode"))
    if code in (None, "", 0, "0"):
        return ""
    message = _extract_error(payload)
    return f"{code}: {message}" if message else str(code)


def _log_veo_http(
    stage: str,
    *,
    url: str,
    http_status: int,
    payload: Any,
    task_id: str,
    request_payload: Dict[str, Any],
) -> None:
    summary = {
        "stage": stage,
        "url": url,
        "http_status": http_status,
        "model": str(request_payload.get("model") or ""),
        "image_url_present": bool(
            request_payload.get("imageUrl")
            or request_payload.get("image_url")
            or request_payload.get("image")
        ),
        "task_id": task_id,
        "status": _extract_status(payload),
        "progress": _extract_progress(payload),
        "has_video_url": bool(_extract_video_url(payload)),
        "error": _provider_error(payload),
    }
    print(f"[veo_http] {json.dumps(summary, ensure_ascii=False)}", flush=True)


def _normalize_status(value: str) -> str:
    return str(value or "").strip().lower()


def _first_text(payload: Any, keys: Iterable[str]) -> str:
    value = _first_value(payload, keys)
    if value is None:
        return ""
    return str(value).strip()


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


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    raw = env_value(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    raw = env_value(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, value)
