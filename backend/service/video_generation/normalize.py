import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import requests
from fastapi import HTTPException
from starlette.responses import FileResponse

from backend.types.schema import VeoGenerationResponse


REPO_ROOT = Path(__file__).resolve().parents[3]
NORMALIZED_VIDEO_DIR = REPO_ROOT / "artifacts" / "normalized_ai_video"
NORMALIZED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def normalize_generation_response(response: VeoGenerationResponse) -> VeoGenerationResponse:
    if str(response.status or "").lower() != "ready":
        return response
    source_url = str(response.video_url or "").strip()
    if not source_url.startswith(("http://", "https://")):
        return response

    try:
        normalized_url = normalize_remote_video(source_url)
    except Exception as exc:
        summary = {
            "source_host": safe_video_host(source_url),
            "error_type": exc.__class__.__name__,
            "message": str(exc),
        }
        print(f"[video_normalize_error] {json.dumps(summary, ensure_ascii=False)}", flush=True)
        return response

    update = {"normalized_video_url": normalized_url}
    if hasattr(response, "model_copy"):
        return response.model_copy(update=update)
    return response.copy(update=update)


def normalize_remote_video(source_url: str) -> str:
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    token = hashlib.sha1(source_url.encode("utf-8")).hexdigest()[:20]
    work_dir = NORMALIZED_VIDEO_DIR / token
    work_dir.mkdir(parents=True, exist_ok=True)

    source_path = work_dir / "source.mp4"
    output_path = work_dir / "normalized.mp4"

    source_summary = _download_source_video(source_url, source_path)
    input_probe = _probe_video(ffmpeg_exe, source_path)
    print(
        f"[video_normalize_probe] {json.dumps({'source_host': safe_video_host(source_url), **source_summary, **input_probe}, ensure_ascii=False)}",
        flush=True,
    )

    if not output_path.exists() or output_path.stat().st_size <= 0:
        _transcode_video(ffmpeg_exe, source_path, output_path, has_audio=input_probe["has_audio"])

    output_probe = _probe_video(ffmpeg_exe, output_path)
    print(
        f"[video_normalize_output] {json.dumps({'source_host': safe_video_host(source_url), **output_probe}, ensure_ascii=False)}",
        flush=True,
    )

    return f"/api/video-generation/normalized/{output_path.parent.name}/{output_path.name}"


def serve_normalized_video(folder_name: str, file_name: str):
    base = (NORMALIZED_VIDEO_DIR / folder_name).resolve()
    target = (base / file_name).resolve()
    if base.parent != NORMALIZED_VIDEO_DIR.resolve() and base != NORMALIZED_VIDEO_DIR.resolve():
        raise HTTPException(status_code=404, detail="normalized video not found")
    try:
        target.relative_to(NORMALIZED_VIDEO_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="normalized video not found") from exc

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="normalized video not found")
    return FileResponse(target, media_type="video/mp4", filename=target.name)


def safe_video_host(source_url: str) -> str:
    return str(urlparse(source_url).hostname or "")


def _download_source_video(source_url: str, source_path: Path) -> Dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(source_url, headers=headers, stream=True, timeout=120, allow_redirects=True) as response:
        response.raise_for_status()
        with source_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    file_obj.write(chunk)
        return {
            "download_status": str(response.status_code),
            "download_content_type": response.headers.get("Content-Type", ""),
            "download_content_length": response.headers.get("Content-Length", ""),
        }


def _probe_video(ffmpeg_exe: str, file_path: Path) -> Dict[str, object]:
    command = [ffmpeg_exe, "-hide_banner", "-i", str(file_path)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    stderr_lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
    interesting: List[str] = []
    for line in stderr_lines:
        if line.startswith("Input #") or line.startswith("Duration:") or "Video:" in line or "Audio:" in line:
            interesting.append(line)
    has_audio = any("Audio:" in line for line in stderr_lines)
    return {
        "probe_lines": interesting[:6],
        "has_audio": has_audio,
        "file_size": str(file_path.stat().st_size if file_path.exists() else 0),
    }


def _transcode_video(ffmpeg_exe: str, source_path: Path, output_path: Path, *, has_audio: bool) -> None:
    temp_output = output_path.with_suffix(".tmp.mp4")
    if temp_output.exists():
        temp_output.unlink()
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if has_audio:
        command.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        command.append("-an")
    command.append(str(temp_output))

    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if completed.returncode != 0 or not temp_output.exists() or temp_output.stat().st_size <= 0:
        raise RuntimeError(_truncate_error(completed.stderr) or "ffmpeg normalization failed")
    shutil.move(str(temp_output), str(output_path))


def _truncate_error(text: str, limit: int = 800) -> str:
    return str(text or "").strip()[:limit]
