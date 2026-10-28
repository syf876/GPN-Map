from backend.admin_auth import require_generation_access
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

try:
    import imageio_ffmpeg
except ModuleNotFoundError:  # pragma: no cover
    imageio_ffmpeg = None

try:
    from PIL import Image, ImageFilter
except ModuleNotFoundError:  # pragma: no cover
    Image = None
    ImageFilter = None

from backend.types.schema import VeoGenerationResponse

from . import wan_client as wan_runtime
from .normalize import NORMALIZED_VIDEO_DIR
from .poetry_audio import PreparedPoetryVideoRequest, synthesize_poetry_audio
from .prompt_builder import StoryboardShot, build_storyboard, build_wan_negative_prompt, build_wan_prompt


def generate_wan_story_video(prepared: PreparedPoetryVideoRequest) -> VeoGenerationResponse:
    require_generation_access()
    req = prepared.request
    overview_prompt = _with_poem_meta(build_wan_prompt(req), prepared)
    config = wan_runtime._load_config(req)
    attempt_configs = wan_runtime._i2v_attempt_configs(config)
    missing_dependencies = wan_runtime._missing_runtime_dependencies(config.model)
    if imageio_ffmpeg is None:
        missing_dependencies = [*missing_dependencies, "imageio-ffmpeg"]
    if missing_dependencies:
        return VeoGenerationResponse(
            status="failed",
            provider="wan",
            prompt=overview_prompt,
            error=f"missing runtime dependencies: {', '.join(sorted(set(missing_dependencies)))}",
            raw_status="missing_runtime_dependencies",
        )
    if not config.api_key:
        return VeoGenerationResponse(
            status="failed",
            provider="wan",
            prompt=overview_prompt,
            error="missing Wanxiang API key environment variable; checked DASHSCOPE_API_KEY, QWEN_VL_API_KEY, WAN_API_KEY, WAN_I2V_API_KEY and compatible aliases",
            raw_status="missing_api_key",
        )

    image_url = str(req.image_url or "").strip()
    if not image_url:
        return VeoGenerationResponse(
            status="failed",
            provider="wan",
            prompt=overview_prompt,
            error=f"{config.model} requires image_url",
            raw_status="missing_image_url",
        )

    story_id = _story_id(prepared)
    _log_story_event(
        "start",
        story_id=story_id,
        matched=prepared.poem_match.matched,
        title=prepared.poem_match.title,
        author=prepared.poem_match.author,
        dynasty=prepared.poem_match.dynasty,
        image_url=bool(image_url),
        narration_len=len(str(req.narration or "")),
        models=[attempt.model for attempt in attempt_configs],
    )
    work_dir = NORMALIZED_VIDEO_DIR / story_id
    work_dir.mkdir(parents=True, exist_ok=True)
    story_file = work_dir / "story.mp4"
    metadata_path = work_dir / "storyboard.json"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    image_local_path = None
    full_audio_local_path = None
    segment_video_paths: List[Path] = []
    segment_task_ids: List[str] = []
    segment_debug: List[Dict[str, object]] = []
    shot_audio_paths: List[Path] = []
    shot_variant_paths: List[Path] = []

    try:
        image_local_path, image_content_type = wan_runtime._download_asset_to_local_path(
            source_url=image_url,
            asset_kind="image",
            expected_content_prefixes=("image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"),
            default_extension=".png",
        )
        shots = build_storyboard(req)
        _log_story_event(
            "planned",
            story_id=story_id,
            shot_count=len(shots),
            planned_shots=[shot.shot_id for shot in shots],
        )
        for shot in shots:
            _log_story_event(
                f"shot_{shot.index}_plan",
                story_id=story_id,
                shot_id=shot.shot_id,
                title=shot.title,
                image_variant=shot.image_variant,
            )
        shot_variant_map = _build_storyboard_reference_images(image_local_path, work_dir, shots)
        shot_variant_paths = [shot_variant_map[shot.image_variant] for shot in shots]

        full_audio_url = str(req.audio_url or "").strip()
        if full_audio_url:
            full_audio_local_path, _ = wan_runtime._download_asset_to_local_path(
                source_url=full_audio_url,
                asset_kind="story_audio",
                expected_content_prefixes=("audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"),
                default_extension=".mp3",
            )

        for shot in shots:
            _log_story_event(
                f"shot_{shot.index}_prepare",
                story_id=story_id,
                shot_id=shot.shot_id,
                title=shot.title,
                image_variant=shot.image_variant,
                duration_seconds=shot.duration_seconds,
            )
            shot_audio_source = synthesize_poetry_audio(shot.narration_text)
            shot_audio_local_path, shot_audio_content_type = wan_runtime._download_asset_to_local_path(
                source_url=shot_audio_source.audio_url,
                asset_kind=f"shot_audio_{shot.index}",
                expected_content_prefixes=("audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"),
                default_extension=".mp3",
            )
            shot_audio_paths.append(shot_audio_local_path)

            shot_image_local_path = shot_variant_map[shot.image_variant]
            shot_response = _generate_single_shot_segment_with_fallback(
                configs=attempt_configs,
                shot=shot,
                prompt=_with_poem_meta(shot.prompt, prepared),
                negative_prompt=build_wan_negative_prompt(req),
                image_local_path=shot_image_local_path,
                image_content_type=image_content_type,
                audio_local_path=shot_audio_local_path,
                audio_content_type=shot_audio_content_type,
            )
            segment_debug.append(
                {
                    "index": shot.index,
                    "shot_id": shot.shot_id,
                    "title": shot.title,
                    "task_id": shot_response.id,
                    "status": shot_response.status,
                    "raw_status": shot_response.raw_status,
                    "video_url": shot_response.video_url,
                    "narration_text": shot.narration_text,
                }
            )
            if str(shot_response.status or "").lower() != "ready" or not str(shot_response.video_url or "").strip():
                _log_story_event(
                    f"shot_{shot.index}_failed",
                    story_id=story_id,
                    shot_id=shot.shot_id,
                    task_id=shot_response.id,
                    status=shot_response.status,
                    raw_status=shot_response.raw_status,
                    error=shot_response.error,
                )
                error_message = shot_response.error or f"story segment {shot.index} did not produce a playable video"
                return VeoGenerationResponse(
                    status="failed",
                    provider="wan",
                    id=shot_response.id or story_id,
                    prompt=overview_prompt,
                    error=f"shot {shot.index} {shot.title}: {error_message}",
                    raw_status=shot_response.raw_status or "segment_failed",
                )

            segment_task_ids.append(shot_response.id)
            _log_story_event(
                f"shot_{shot.index}_ready",
                story_id=story_id,
                shot_id=shot.shot_id,
                task_id=shot_response.id,
                video_url=shot_response.video_url,
            )
            segment_source_path = work_dir / f"segment_{shot.index}_source.mp4"
            segment_path = work_dir / f"segment_{shot.index}.mp4"
            _download_video_asset(shot_response.video_url, segment_source_path)
            _transcode_video(ffmpeg_exe, segment_source_path, segment_path, has_audio=False)
            segment_video_paths.append(segment_path)

        _log_story_event(
            "concat_start",
            story_id=story_id,
            segment_count=len(segment_video_paths),
            segments=[path.name for path in segment_video_paths],
        )
        _concat_videos(ffmpeg_exe, segment_video_paths, story_file)
        _log_story_event(
            "concat_done",
            story_id=story_id,
            output=str(story_file),
            file_size=story_file.stat().st_size if story_file.exists() else 0,
        )
        if full_audio_local_path is not None:
            _log_story_event(
                "mux_audio_start",
                story_id=story_id,
                audio_path=str(full_audio_local_path),
            )
            _mux_audio(ffmpeg_exe, story_file, full_audio_local_path, story_file)
            _log_story_event(
                "mux_audio_done",
                story_id=story_id,
                output=str(story_file),
                file_size=story_file.stat().st_size if story_file.exists() else 0,
            )

        metadata = {
            "story_id": story_id,
            "provider": "wan",
            "model": config.model,
            "fallback_model": attempt_configs[-1].model if len(attempt_configs) > 1 else "",
            "poem_match": {
                "matched": prepared.poem_match.matched,
                "match_type": prepared.poem_match.match_type,
                "title": prepared.poem_match.title,
                "author": prepared.poem_match.author,
                "dynasty": prepared.poem_match.dynasty,
                "source_file": prepared.poem_match.source_file,
            },
            "narration_source": prepared.audio_source,
            "full_narration_text": req.narration,
            "segment_task_ids": segment_task_ids,
            "shots": [
                {
                    "index": shot.index,
                    "shot_id": shot.shot_id,
                    "title": shot.title,
                    "duration_seconds": shot.duration_seconds,
                    "image_variant": shot.image_variant,
                    "scene": shot.scene,
                    "action": shot.action,
                    "camera": shot.camera,
                    "emotion": shot.emotion,
                    "narration_text": shot.narration_text,
                }
                for shot in shots
            ],
            "segment_debug": segment_debug,
            "output_url": f"/api/video-generation/normalized/{story_id}/{story_file.name}",
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        _log_story_event(
            "done",
            story_id=story_id,
            output_url=metadata["output_url"],
            segment_task_ids=segment_task_ids,
        )

        return VeoGenerationResponse(
            status="ready",
            provider="wan",
            id=story_id,
            video_url=f"/api/video-generation/normalized/{story_id}/{story_file.name}",
            normalized_video_url=f"/api/video-generation/normalized/{story_id}/{story_file.name}",
            progress=100.0,
            prompt=overview_prompt,
            error="",
            raw_status="SUCCEEDED",
        )
    except Exception as exc:
        error_text = str(exc)
        _log_story_event("error", story_id=story_id, error=error_text)
        if metadata_path.parent.exists():
            fail_payload = {
                "story_id": story_id,
                "error": error_text,
                "segment_debug": segment_debug,
            }
            metadata_path.write_text(json.dumps(fail_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return VeoGenerationResponse(
            status="failed",
            provider="wan",
            id=story_id,
            prompt=overview_prompt,
            error=error_text,
            raw_status="story_pipeline_exception",
        )
    finally:
        for path in [image_local_path, full_audio_local_path, *shot_audio_paths]:
            if isinstance(path, Path):
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
        for path in shot_variant_paths:
            if isinstance(path, Path) and path.exists() and path.parent != work_dir:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass


def _generate_single_shot_segment(
    *,
    config,
    shot: StoryboardShot,
    prompt: str,
    negative_prompt: str,
    image_local_path: Path,
    image_content_type: str,
    audio_local_path: Path,
    audio_content_type: str,
) -> VeoGenerationResponse:
    create_payload = wan_runtime._build_create_payload(
        config=config,
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_url="",
        image_local_path=image_local_path,
        image_content_type=image_content_type,
        audio_local_path=audio_local_path,
        audio_content_type=audio_content_type,
    )
    wan_runtime._log_wan_request("video-synthesis", create_payload)
    create_response = wan_runtime._post_json(
        wan_runtime._video_synthesis_endpoint(config),
        create_payload,
        headers=wan_runtime._json_headers(config.api_key, resolve_oss=wan_runtime._contains_oss_url(create_payload), async_mode=True),
        stage="video-synthesis",
    )
    task_id = wan_runtime._extract_task_id(create_response)
    raw_status = wan_runtime._extract_task_status(create_response) or "PENDING"
    if not task_id:
        return VeoGenerationResponse(
            status="failed",
            provider="wan",
            prompt=prompt,
            error=wan_runtime._extract_message(create_response) or "wan create response missing task id",
            raw_status=raw_status or "missing_task_id",
        )

    progress = 5.0
    elapsed_seconds = 0.0
    for attempt in range(config.poll_max_attempts):
        if elapsed_seconds >= config.poll_timeout_seconds:
            break
        try:
            task_response = wan_runtime._get_json(
                f"{config.api_root}/api/v1/tasks/{task_id}",
                headers=wan_runtime._auth_headers(config.api_key),
                stage="task-query",
            )
        except Exception as exc:
            return VeoGenerationResponse(
                status="failed",
                provider="wan",
                id=task_id,
                prompt=prompt,
                error=str(exc),
                raw_status=raw_status or "task_query_failed",
            )

        task_status = wan_runtime._extract_task_status(task_response) or raw_status
        video_url = wan_runtime._extract_video_url(task_response)
        progress = wan_runtime._status_progress(task_status, progress)
        raw_status = task_status or raw_status
        sleep_seconds = wan_runtime._next_poll_interval_seconds(config, attempt)
        wan_runtime._log_poll_summary(
            attempt=attempt + 1,
            elapsed_seconds=elapsed_seconds,
            next_sleep_seconds=sleep_seconds,
            task_id=task_id,
            task_status=task_status,
            progress=progress,
            video_url=video_url,
            error="",
        )

        if video_url and task_status in wan_runtime.SUCCESS_STATUSES:
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

        explicit_error = wan_runtime._extract_message(task_response) or wan_runtime._extract_reason(task_response)
        if task_status in wan_runtime.FAILURE_STATUSES or (task_status == "UNKNOWN" and explicit_error):
            return VeoGenerationResponse(
                status="failed",
                provider="wan",
                id=task_id,
                prompt=prompt,
                error=explicit_error or f"wan story segment {shot.index} failed",
                raw_status=raw_status,
            )

        elapsed_seconds += sleep_seconds
        if attempt < config.poll_max_attempts - 1 and elapsed_seconds < config.poll_timeout_seconds:
            import time

            time.sleep(sleep_seconds)

    return VeoGenerationResponse(
        status="failed",
        provider="wan",
        id=task_id,
        prompt=prompt,
        error=f"wan story segment {shot.index} timed out before completion",
        raw_status=raw_status or "segment_timeout",
    )


def _generate_single_shot_segment_with_fallback(
    *,
    configs: List[object],
    shot: StoryboardShot,
    prompt: str,
    negative_prompt: str,
    image_local_path: Path,
    image_content_type: str,
    audio_local_path: Path,
    audio_content_type: str,
) -> VeoGenerationResponse:
    failures: List[Tuple[str, VeoGenerationResponse]] = []
    for attempt_index, config in enumerate(configs):
        previous_error = failures[-1][1].error if failures else ""
        wan_runtime._log_i2v_attempt(
            f"story-shot-{shot.index}",
            model=config.model,
            fallback=attempt_index > 0,
            previous_error=previous_error,
        )
        try:
            response = _generate_single_shot_segment(
                config=config,
                shot=shot,
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_local_path=image_local_path,
                image_content_type=image_content_type,
                audio_local_path=audio_local_path,
                audio_content_type=audio_content_type,
            )
        except Exception as exc:
            response = VeoGenerationResponse(
                status="failed",
                provider="wan",
                prompt=prompt,
                error=str(exc),
                raw_status="segment_exception",
            )

        if wan_runtime._response_has_playable_video(response):
            return response
        if str(response.status or "").lower() != "failed":
            return response
        failures.append((config.model, response))

    return wan_runtime._failure_response_from_attempts(prompt, failures)


def _build_storyboard_reference_images(image_local_path: Path, work_dir: Path, shots: List[StoryboardShot]) -> Dict[str, Path]:
    if Image is None or ImageFilter is None:
        return {shot.image_variant: image_local_path for shot in shots}

    variant_names = {shot.image_variant for shot in shots}
    variant_map: Dict[str, Path] = {}
    with Image.open(image_local_path) as image:
        base = image.convert("RGB")
        if "opening_wide" in variant_names:
            variant_map["opening_wide"] = _save_blurred_canvas_variant(base, work_dir / "opening_wide.png", scale=1.9)
        if "travel_push" in variant_names:
            variant_map["travel_push"] = _save_center_crop_variant(base, work_dir / "travel_push.png", crop_ratio=0.74)
        if "pause_reflect" in variant_names:
            variant_map["pause_reflect"] = _save_side_focus_variant(base, work_dir / "pause_reflect.png", crop_ratio=0.52)
        if "closing_empty" in variant_names:
            variant_map["closing_empty"] = _save_offcenter_canvas_variant(base, work_dir / "closing_empty.png", scale=2.2)
    for shot in shots:
        variant_map.setdefault(shot.image_variant, image_local_path)
    return variant_map


def _save_blurred_canvas_variant(base, output_path: Path, *, scale: float) -> Path:
    width = max(base.width, int(round(base.width * scale)))
    height = max(base.height, int(round(base.height * scale)))
    background = base.resize((width, height), Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(radius=max(10, int(min(width, height) * 0.01)))
    )
    offset_x = (width - base.width) // 2
    offset_y = (height - base.height) // 2
    background.paste(base, (offset_x, offset_y))
    background.save(output_path, format="PNG")
    return output_path


def _save_center_crop_variant(base, output_path: Path, *, crop_ratio: float) -> Path:
    crop_width = max(64, int(round(base.width * crop_ratio)))
    crop_height = base.height
    left = max(0, (base.width - crop_width) // 2)
    cropped = base.crop((left, 0, left + crop_width, crop_height))
    cropped.save(output_path, format="PNG")
    return output_path


def _save_side_focus_variant(base, output_path: Path, *, crop_ratio: float) -> Path:
    crop_width = max(64, int(round(base.width * crop_ratio)))
    left = max(0, base.width - crop_width)
    focused = base.crop((left, 0, base.width, base.height))
    frame_width = int(round(focused.width * 1.25))
    frame_height = int(round(focused.height * 1.12))
    background = focused.resize((frame_width, frame_height), Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(radius=max(8, int(min(frame_width, frame_height) * 0.01)))
    )
    offset_x = (frame_width - focused.width) // 2
    offset_y = (frame_height - focused.height) // 2
    background.paste(focused, (offset_x, offset_y))
    background.save(output_path, format="PNG")
    return output_path


def _save_offcenter_canvas_variant(base, output_path: Path, *, scale: float) -> Path:
    width = max(base.width, int(round(base.width * scale)))
    height = max(base.height, int(round(base.height * scale)))
    background = base.resize((width, height), Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(radius=max(10, int(min(width, height) * 0.012)))
    )
    offset_x = int(round((width - base.width) * 0.2))
    offset_y = (height - base.height) // 2
    background.paste(base, (offset_x, offset_y))
    background.save(output_path, format="PNG")
    return output_path


def _download_video_asset(source_url: str, target_path: Path) -> None:
    local_source = wan_runtime._resolve_local_asset_path(source_url)
    if local_source is not None:
        shutil.copyfile(local_source, target_path)
        return

    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(source_url, headers=headers, stream=True, timeout=240, allow_redirects=True) as response:
        response.raise_for_status()
        with target_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    file_obj.write(chunk)


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
    _run_ffmpeg(command, "segment transcode failed")
    shutil.move(str(temp_output), str(output_path))


def _concat_videos(ffmpeg_exe: str, segment_paths: List[Path], output_path: Path) -> None:
    concat_file = output_path.with_suffix(".concat.txt")
    temp_output = output_path.with_suffix(".concat.mp4")
    concat_lines = []
    for path in segment_paths:
        resolved = str(path.resolve()).replace("\\", "/")
        concat_lines.append(f"file '{resolved}'")
    concat_file.write_text(
        "\n".join(concat_lines),
        encoding="utf-8",
    )
    if temp_output.exists():
        temp_output.unlink()
    command = [
        ffmpeg_exe,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(temp_output),
    ]
    try:
        _run_ffmpeg(command, "video concat failed")
        shutil.move(str(temp_output), str(output_path))
    finally:
        concat_file.unlink(missing_ok=True)


def _mux_audio(ffmpeg_exe: str, video_path: Path, audio_path: Path, output_path: Path) -> None:
    temp_output = output_path.with_suffix(".mux.mp4")
    if temp_output.exists():
        temp_output.unlink()
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(temp_output),
    ]
    _run_ffmpeg(command, "audio mux failed")
    shutil.move(str(temp_output), str(output_path))


def _run_ffmpeg(command: List[str], default_message: str) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or default_message).strip()
        raise RuntimeError(message[:1200] or default_message)


def _story_id(prepared: PreparedPoetryVideoRequest) -> str:
    payload = {
        "text": prepared.request.text,
        "narration": prepared.request.narration,
        "image_url": prepared.request.image_url,
        "poem_title": prepared.poem_match.title,
        "poem_author": prepared.poem_match.author,
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:18]
    nonce = format(int(time.time() * 1000), "x")[-8:]
    return f"story_{digest}_{nonce}"


def _with_poem_meta(prompt: str, prepared: PreparedPoetryVideoRequest) -> str:
    if not prepared.poem_match.matched:
        return prompt
    title = str(prepared.poem_match.title or "").strip()
    author = str(prepared.poem_match.author or "").strip()
    dynasty = str(prepared.poem_match.dynasty or "").strip()
    meta_bits = [item for item in (title, author, dynasty) if item]
    if not meta_bits:
        return prompt
    return f"Source poem reference: {' / '.join(meta_bits)}. {prompt}"


def _log_story_event(stage: str, **payload) -> None:
    summary = {"stage": stage, **payload}
    print(f"[story_pipeline] {json.dumps(summary, ensure_ascii=False)}", flush=True)
