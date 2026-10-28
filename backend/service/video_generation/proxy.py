from typing import Dict
from urllib.parse import urlparse

import requests
from fastapi import HTTPException
from starlette.background import BackgroundTask
from starlette.responses import Response, StreamingResponse


ALLOWED_VIDEO_HOST_SUFFIXES = (
    ".aigc-cloud.com",
)
ALLOWED_VIDEO_HOSTS = {
    "aigc-cloud.com",
}
FORWARDED_RESPONSE_HEADERS = (
    "Content-Type",
    "Content-Length",
    "Accept-Ranges",
    "Content-Range",
    "Cache-Control",
    "ETag",
    "Last-Modified",
)


def proxy_video_request(url: str, method: str, range_header: str = ""):
    normalized_url = _validate_video_url(url)
    headers: Dict[str, str] = {
        "User-Agent": "Mozilla/5.0",
    }
    if range_header:
        headers["Range"] = range_header

    try:
        if method.upper() == "HEAD":
            upstream = requests.head(normalized_url, headers=headers, allow_redirects=True, timeout=60)
        else:
            upstream = requests.get(normalized_url, headers=headers, allow_redirects=True, stream=True, timeout=60)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"video proxy upstream request failed: {exc}") from exc

    response_headers = _filter_response_headers(upstream.headers)
    response_headers["Access-Control-Allow-Origin"] = "*"

    if method.upper() == "HEAD":
        upstream.close()
        return Response(status_code=upstream.status_code, headers=response_headers)

    return StreamingResponse(
        upstream.iter_content(chunk_size=65536),
        status_code=upstream.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream.close),
    )


def _validate_video_url(url: str) -> str:
    text = str(url or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in {"https"}:
        raise HTTPException(status_code=400, detail="video proxy only allows https urls")

    hostname = str(parsed.hostname or "").lower()
    if not hostname:
        raise HTTPException(status_code=400, detail="video proxy requires a valid host")

    if hostname not in ALLOWED_VIDEO_HOSTS and not any(hostname.endswith(suffix) for suffix in ALLOWED_VIDEO_HOST_SUFFIXES):
        raise HTTPException(status_code=400, detail="video proxy host is not allowed")

    return text


def _filter_response_headers(headers: requests.structures.CaseInsensitiveDict) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key in FORWARDED_RESPONSE_HEADERS:
        value = headers.get(key)
        if value:
            result[key] = value
    return result
