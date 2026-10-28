from backend.admin_auth import require_generation_access
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from backend.types.schema import VeoGenerationRequest, VeoGenerationResponse

from .env import env_value
from .prompt_builder import build_veo_prompt


SUCCESS_STATUSES = {"done", "success", "succeeded", "completed", "ready", "finished"}
FAILURE_STATUSES = {"failed", "error", "canceled", "cancelled", "timeout", "rejected"}
WAITING_STATUSES = {"in_queue", "queueing", "queued", "pending", "submitted", "processing", "running", "generating", "created"}
MISSING_STATUSES = {"not_found", "expired"}
DONE_WITHOUT_URL_RETRIES = 3


@dataclass(frozen=True)
class JimengConfig:
    access_key_id: str
    secret_access_key: str
    api_base: str
    req_key: str
    region: str
    service: str
    version: str
    frames: int
    aspect_ratio: str
    poll_max_attempts: int
    poll_interval_seconds: float


def generate_jimeng_video(req: VeoGenerationRequest) -> VeoGenerationResponse:
    require_generation_access()
    prompt = build_veo_prompt(req)
    config = _load_config()
    if not config.access_key_id or not config.secret_access_key:
        return VeoGenerationResponse(
            status="failed",
            provider="jimeng",
            prompt=prompt,
            error="missing JIMENG_ACCESS_KEY_ID or JIMENG_SECRET_ACCESS_KEY environment variable",
            raw_status="missing_credentials",
        )

    submit_payload = {
        "req_key": config.req_key,
        "prompt": prompt,
        "seed": -1,
        "frames": config.frames,
        "aspect_ratio": config.aspect_ratio,
    }
    try:
        submit_response = _signed_post(config, "CVSync2AsyncSubmitTask", submit_payload)
        submit_id = _extract_id(submit_response)
        submit_status = _extract_status(submit_response)
        submit_video_url = _extract_video_url(submit_response)
        submit_progress = _extract_progress(submit_response)

        if submit_video_url and (submit_status in SUCCESS_STATUSES or not submit_status):
            return VeoGenerationResponse(
                status="ready",
                provider="jimeng",
                id=submit_id,
                video_url=submit_video_url,
                progress=submit_progress or 100.0,
                prompt=prompt,
                error="",
                raw_status=submit_status or "done",
            )

        if submit_status in FAILURE_STATUSES:
            return VeoGenerationResponse(
                status="failed",
                provider="jimeng",
                id=submit_id,
                video_url="",
                progress=submit_progress,
                prompt=prompt,
                error=_extract_error(submit_response) or "jimeng task submission failed",
                raw_status=submit_status,
            )

        if not submit_id:
            return VeoGenerationResponse(
                status="failed",
                provider="jimeng",
                id="",
                video_url="",
                progress=submit_progress,
                prompt=prompt,
                error="jimeng submit response missing task id",
                raw_status=submit_status or "missing_task_id",
            )

        last_progress = submit_progress
        last_raw_status = submit_status or "submitted"
        poll_payload = {"req_key": config.req_key, "task_id": submit_id}
        done_without_url_count = 0
        for attempt in range(config.poll_max_attempts):
            result_response = _signed_post(config, "CVSync2AsyncGetResult", poll_payload)
            result_status = _extract_task_status(result_response) or last_raw_status
            result_video_url = _extract_video_url(result_response)
            result_progress = _extract_progress(result_response)
            result_code = _extract_code(result_response)
            result_message = _extract_error(result_response)
            result_request_id = _extract_request_id(result_response)
            if result_progress > 0:
                last_progress = result_progress
            last_raw_status = result_status or last_raw_status
            _log_poll_summary(
                attempt=attempt + 1,
                code=result_code,
                message=result_message,
                status=result_status,
                has_video_url=bool(result_video_url),
                request_id=result_request_id,
            )

            if result_video_url and result_status in SUCCESS_STATUSES:
                return VeoGenerationResponse(
                    status="ready",
                    provider="jimeng",
                    id=submit_id,
                    video_url=result_video_url,
                    progress=last_progress or 100.0,
                    prompt=prompt,
                    error="",
                    raw_status=last_raw_status,
                )

            if result_status in MISSING_STATUSES:
                return VeoGenerationResponse(
                    status="failed",
                    provider="jimeng",
                    id=submit_id,
                    video_url="",
                    progress=last_progress,
                    prompt=prompt,
                    error=result_message or "jimeng task result not found or expired",
                    raw_status=last_raw_status,
                )

            if result_status in FAILURE_STATUSES:
                return VeoGenerationResponse(
                    status="failed",
                    provider="jimeng",
                    id=submit_id,
                    video_url="",
                    progress=last_progress,
                    prompt=prompt,
                    error=_extract_error(result_response) or "jimeng generation failed during polling",
                    raw_status=last_raw_status,
                )

            if result_status in SUCCESS_STATUSES and not result_video_url:
                done_without_url_count += 1
                if done_without_url_count > DONE_WITHOUT_URL_RETRIES:
                    return VeoGenerationResponse(
                        status="failed",
                        provider="jimeng",
                        id=submit_id,
                        video_url="",
                        progress=last_progress,
                        prompt=prompt,
                        error="jimeng task reached done status but video_url is still missing",
                        raw_status=last_raw_status,
                    )
            elif result_status in WAITING_STATUSES or not result_status:
                done_without_url_count = 0
            else:
                done_without_url_count = 0

            if attempt < config.poll_max_attempts - 1:
                time.sleep(config.poll_interval_seconds)

        return VeoGenerationResponse(
            status="failed",
            provider="jimeng",
            id=submit_id,
            video_url="",
            progress=last_progress,
            prompt=prompt,
            error="jimeng polling timed out before reaching a terminal status",
            raw_status=last_raw_status or "timeout",
        )
    except Exception as exc:
        return VeoGenerationResponse(
            status="failed",
            provider="jimeng",
            prompt=prompt,
            error=str(exc),
            raw_status="request_exception",
        )


def _load_config() -> JimengConfig:
    return JimengConfig(
        access_key_id=env_value("JIMENG_ACCESS_KEY_ID", ""),
        secret_access_key=env_value("JIMENG_SECRET_ACCESS_KEY", ""),
        api_base=env_value("JIMENG_API_BASE", "https://visual.volcengineapi.com").rstrip("/"),
        req_key=env_value("JIMENG_REQ_KEY", "jimeng_t2v_v30_1080p") or "jimeng_t2v_v30_1080p",
        region=env_value("JIMENG_REGION", "cn-north-1") or "cn-north-1",
        service=env_value("JIMENG_SERVICE", "cv") or "cv",
        version=env_value("JIMENG_VERSION", "2022-08-31") or "2022-08-31",
        frames=_int_env("JIMENG_FRAMES", 121, minimum=1),
        aspect_ratio=env_value("JIMENG_ASPECT_RATIO", "16:9") or "16:9",
        poll_max_attempts=_int_env("JIMENG_POLL_MAX_ATTEMPTS", 120, minimum=1),
        poll_interval_seconds=_float_env("JIMENG_POLL_INTERVAL_SECONDS", 5.0, minimum=0.1),
    )


def _signed_post(config: JimengConfig, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    parsed = urlparse(config.api_base)
    host = parsed.netloc
    canonical_uri = parsed.path or "/"
    query_pairs = [("Action", action), ("Version", config.version)]
    canonical_query = _canonical_query(query_pairs)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    body_hash = hashlib.sha256(body).hexdigest()

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    signed_headers = "content-type;host;region;servicename;x-content-sha256;x-date"
    canonical_headers = (
        f"content-type:application/json\n"
        f"host:{host}\n"
        f"region:{config.region}\n"
        f"servicename:{config.service}\n"
        f"x-content-sha256:{body_hash}\n"
        f"x-date:{timestamp}\n"
    )
    canonical_request = "\n".join(
        [
            "POST",
            canonical_uri,
            canonical_query,
            canonical_headers,
            signed_headers,
            body_hash,
        ]
    )
    credential_scope = f"{datestamp}/{config.region}/{config.service}/request"
    string_to_sign = "\n".join(
        [
            "HMAC-SHA256",
            timestamp,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signature = _signature(config.secret_access_key, datestamp, config.region, config.service, string_to_sign)
    authorization = (
        "HMAC-SHA256 "
        f"Credential={config.access_key_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )
    headers = {
        "Content-Type": "application/json",
        "Host": host,
        "Region": config.region,
        "ServiceName": config.service,
        "X-Date": timestamp,
        "X-Content-Sha256": body_hash,
        "Authorization": authorization,
    }
    request_url = f"{config.api_base}?{canonical_query}"
    request = Request(request_url, data=body, headers=headers, method="POST")
    try:
        require_generation_access()
        with urlopen(request, timeout=120) as response:
            raw_body = response.read()
    except HTTPError as exc:
        error_body = exc.read()
        detail = _decode_json_body(error_body)
        message = _extract_error(detail) or error_body.decode("utf-8", errors="ignore") or str(exc)
        code = _extract_code(detail)
        request_id = _extract_request_id(detail)
        suffix = []
        if code:
            suffix.append(f"code={code}")
        if request_id:
            suffix.append(f"request_id={request_id}")
        suffix_text = f" ({', '.join(suffix)})" if suffix else ""
        raise RuntimeError(f"jimeng http {exc.code}: {message}{suffix_text}") from exc
    except URLError as exc:
        raise RuntimeError(f"jimeng request failed: {exc.reason}") from exc

    parsed_body = _decode_json_body(raw_body)
    if isinstance(parsed_body, dict):
        return parsed_body
    raise RuntimeError("jimeng response is not a JSON object")


def _signature(secret_access_key: str, datestamp: str, region: str, service: str, string_to_sign: str) -> str:
    k_date = _hmac(secret_access_key.encode("utf-8"), datestamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, service)
    k_signing = _hmac(k_service, "request")
    return hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


def _hmac(key: bytes, value: str) -> bytes:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()


def _canonical_query(pairs: Iterable[tuple[str, str]]) -> str:
    encoded = []
    for key, value in sorted((str(key), str(value)) for key, value in pairs):
        encoded.append(f"{quote(key, safe='-_.~')}={quote(value, safe='-_.~')}")
    return "&".join(encoded)


def _decode_json_body(raw_body: bytes) -> Any:
    text = raw_body.decode("utf-8", errors="ignore").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}


def _extract_id(payload: Any) -> str:
    return _first_text(payload, ("task_id", "taskId", "id", "job_id", "jobId"))


def _extract_status(payload: Any) -> str:
    return _normalize_status(_first_text(payload, ("status", "state")))


def _extract_task_status(payload: Any) -> str:
    data_node = _first_dict(payload, ("data", "result"))
    if data_node:
        value = _first_text(data_node, ("status", "state"))
        if value:
            return _normalize_status(value)
    return _extract_status(payload)


def _extract_video_url(payload: Any) -> str:
    url = _first_text(
        payload,
        (
            "video_url",
            "videoUrl",
            "url",
            "file_url",
            "fileUrl",
            "download_url",
            "downloadUrl",
        ),
    )
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return ""


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
    return _first_text(
        payload,
        (
            "error",
            "message",
            "msg",
            "detail",
            "reason",
            "code_message",
        ),
    )


def _extract_code(payload: Any) -> str:
    return _first_text(payload, ("code", "Code", "error_code", "ErrorCode"))


def _extract_request_id(payload: Any) -> str:
    return _first_text(payload, ("request_id", "requestId", "RequestId", "x_tt_logid"))


def _log_poll_summary(
    *,
    attempt: int,
    code: str,
    message: str,
    status: str,
    has_video_url: bool,
    request_id: str,
) -> None:
    summary = {
        "attempt": attempt,
        "code": code,
        "message": message,
        "data_status": status,
        "has_video_url": has_video_url,
        "request_id": request_id,
    }
    print(f"[jimeng_poll] {json.dumps(summary, ensure_ascii=False)}", flush=True)


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


def _first_dict(payload: Any, keys: Iterable[str]) -> Optional[Dict[str, Any]]:
    normalized_keys = {str(key).lower() for key in keys}
    if not isinstance(payload, dict):
        return None
    for key, value in payload.items():
        if str(key).lower() in normalized_keys and isinstance(value, dict):
            return value
    return None


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    raw = env_value(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    raw = env_value(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, value)
