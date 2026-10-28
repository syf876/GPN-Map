import json
import mimetypes
import shutil
import subprocess
from pathlib import Path
from typing import Any, List
from urllib.parse import urlparse

from backend.service.agent.schemas import AgentState, PlanStep, StepValidation


REPO_ROOT = Path(__file__).resolve().parents[3]
HEXI_LNG_RANGE = (88.0, 105.5)
HEXI_LAT_RANGE = (35.0, 43.5)


class Critic:
    def evaluate_step(self, state: AgentState, step: PlanStep) -> StepValidation:
        missing = [
            condition
            for condition in step.success_conditions
            if not self._condition_satisfied(state, condition)
        ]
        semantic_missing = self._semantic_missing(state, step.step_id)
        missing.extend(item for item in semantic_missing if item not in missing)
        return StepValidation(
            ok=not missing,
            step_id=step.step_id,
            success_conditions=list(step.success_conditions),
            missing_conditions=missing,
            message="success conditions satisfied" if not missing else "missing success conditions",
            recommendation=self._recommendation(state, step, missing),
        )

    def evaluate_final(self, state: AgentState) -> StepValidation:
        target_conditions = {
            "map": ["geojson"],
            "image": ["geojson", "narrative_chain", "image"],
            "video": ["geojson", "narrative_chain", "image", "video"],
        }
        conditions = target_conditions[state.target]
        missing = [condition for condition in conditions if not self._condition_satisfied(state, condition)]
        for step_id in ("rag_retrieval", "semantic_analysis", "place_alignment", "spatial_mapping"):
            missing.extend(item for item in self._semantic_missing(state, step_id) if item not in missing)
        if state.target in {"image", "video"}:
            missing.extend(item for item in self._semantic_missing(state, "poetry_image") if item not in missing)
        if state.target == "video":
            missing.extend(item for item in self._semantic_missing(state, "poetry_video") if item not in missing)
        return StepValidation(
            ok=not missing,
            step_id="final",
            success_conditions=conditions,
            missing_conditions=missing,
            message="target output is complete" if not missing else "target output is incomplete",
            recommendation=self._recommendation(state, None, missing),
        )

    def evaluate_failure(self, state: AgentState, step: PlanStep, error: str) -> StepValidation:
        message = str(error or "").strip()
        missing = ["skill_error"]
        recommendation = self._failure_recommendation(step, message)
        return StepValidation(
            ok=False,
            step_id=step.step_id,
            success_conditions=list(step.success_conditions),
            missing_conditions=missing,
            message=message or "skill execution failed",
            recommendation=recommendation,
        )

    def _semantic_missing(self, state: AgentState, step_id: str) -> List[str]:
        checks = {
            "rag_retrieval": self._check_rag,
            "semantic_analysis": self._check_semantic,
            "place_alignment": self._check_place_alignment,
            "spatial_mapping": self._check_geojson,
            "poetry_image": self._check_image,
            "poetry_video": self._check_video,
        }
        checker = checks.get(step_id)
        return checker(state) if checker else []

    def _check_rag(self, state: AgentState) -> List[str]:
        if not state.rag_context:
            return ["rag_context"]
        context = state.rag_context[0]
        missing = []
        if "vector_retrieval" not in context:
            missing.append("rag_context.vector_retrieval")
        if not isinstance(context.get("primary_poem"), dict):
            missing.append("rag_context.primary_poem")
        related = context.get("related_knowledge")
        if context.get("vector_retrieval") and not isinstance(related, list):
            missing.append("rag_context.related_knowledge")
        if context.get("vector_retrieval") is False and not str(context.get("fallback_reason") or "").strip():
            missing.append("rag_context.fallback_reason")
        return missing

    def _check_semantic(self, state: AgentState) -> List[str]:
        result = state.semantic_result if isinstance(state.semantic_result, dict) else {}
        missing = []
        if not isinstance(result.get("place"), dict) or not result.get("place"):
            missing.append("semantic_result.place")
        imagery = result.get("imagery")
        if not (isinstance(imagery, dict) and imagery.get("labels")) and not isinstance(imagery, list):
            missing.append("semantic_result.imagery")
        if not isinstance(result.get("emotion"), dict) or not result.get("emotion"):
            missing.append("semantic_result.emotion")
        if not isinstance(result.get("temporal_spatial_context"), dict):
            missing.append("semantic_result.temporal_spatial_context")
        return missing

    def _check_place_alignment(self, state: AgentState) -> List[str]:
        item = state.place_alignment if isinstance(state.place_alignment, dict) else {}
        required = [
            "historical_place",
            "modern_geographic_entity",
            "entity_type",
            "coordinates",
            "match_basis",
            "confidence",
        ]
        missing = [f"place_alignment.{key}" for key in required if key not in item]
        coords = item.get("coordinates") if isinstance(item.get("coordinates"), dict) else {}
        if coords and not _valid_hexi_coordinate(coords.get("lon"), coords.get("lat")):
            missing.append("place_alignment.coordinates_range")
        return missing

    def _check_geojson(self, state: AgentState) -> List[str]:
        geojson = state.geojson if isinstance(state.geojson, dict) else {}
        if geojson.get("type") != "FeatureCollection" or not isinstance(geojson.get("features"), list):
            return ["geojson.valid_feature_collection"]
        missing = [] if geojson.get("features") else ["geojson.point_required"]
        for index, feature in enumerate(geojson.get("features") or []):
            geometry = (feature.get("geometry") or {}) if isinstance(feature, dict) else {}
            coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
            if isinstance(geometry, dict) and geometry.get("type") == "Point" and isinstance(coordinates, list) and len(coordinates) >= 2:
                if not _valid_hexi_coordinate(coordinates[0], coordinates[1]):
                    missing.append(f"geojson.features.{index}.coordinates_range")
            else:
                missing.append(f"geojson.features.{index}.point_required")
        return missing

    def _check_image(self, state: AgentState) -> List[str]:
        image = state.image if isinstance(state.image, dict) else {}
        url = str(image.get("normalized_image_url") or image.get("image_url") or "").strip()
        if not url:
            return ["image.url"]
        if not _resource_exists(url):
            return ["image.accessible"]
        mime = mimetypes.guess_type(urlparse(url).path)[0] or ""
        if mime and not mime.startswith("image/"):
            return ["image.mime"]
        local_path = _local_path_for_url(url)
        if local_path and local_path.exists():
            ratio_ok = _image_ratio_valid(local_path)
            if ratio_ok is False:
                return ["image.landscape_ratio"]
        return []

    def _check_video(self, state: AgentState) -> List[str]:
        video = state.video if isinstance(state.video, dict) else {}
        if video.get("fault_type") == "video_low_motion_once":
            return ["video.frame_difference"]
        url = str(video.get("normalized_video_url") or video.get("video_url") or "").strip()
        if not url:
            return ["video.url"]
        if not _resource_exists(url):
            return ["video.accessible"]
        local_path = _local_path_for_url(url)
        if local_path and local_path.exists():
            ffprobe = _ffprobe_video(local_path)
            if ffprobe is False:
                return ["video.ffprobe"]
            frame_diff = _video_has_meaningful_frame_diff(local_path)
            if frame_diff is False:
                return ["video.frame_difference"]
        return []

    def _condition_satisfied(self, state: AgentState, condition: str) -> bool:
        value: Any = getattr(state, condition, None)
        if condition == "real_video_search":
            return (isinstance(value, dict) and value.get("ok") is True
                    and bool(value.get("best")) and bool(value.get("candidates")))
        if isinstance(value, dict):
            return bool(value)
        if isinstance(value, list):
            return len(value) > 0
        if isinstance(value, str):
            return bool(value.strip())
        return value is not None

    def _recommendation(self, state: AgentState, step: PlanStep | None, missing: List[str]) -> dict:
        step_id = step.step_id if step else "final"
        missing_set = set(missing)
        if any(item.endswith("coordinates_range") or item == "place_alignment.coordinates_range" for item in missing_set):
            return {
                "action": "rerun_dependency",
                "step_id": step_id,
                "dependencies": ["place_alignment", "spatial_mapping"],
                "reason": "geojson_or_place_coordinates_out_of_range",
                "failure_type": "geojson_coordinate_range",
            }
        if "image.landscape_ratio" in missing_set:
            return {
                "action": "retry_with_params",
                "step_id": "poetry_image",
                "params": {"mode": "landscape_retry", "aspect_ratio": "16:9"},
                "reason": "image_ratio_not_landscape",
                "failure_type": "image_ratio",
            }
        if missing_set.intersection({"image.url", "image.accessible", "image.mime"}):
            cached = state.image if isinstance(state.image, dict) and state.image.get("source_type") == "curated_featured_case" else {}
            if cached:
                return {
                    "action": "reuse_cache",
                    "step_id": "poetry_image",
                    "target": "poetry_image",
                    "source": "curated_image_cache",
                    "payload": {"image": cached},
                    "reason": "reuse_valid_curated_image_cache",
                    "failure_type": "image_resource",
                }
            return {
                "action": "retry_with_params",
                "step_id": "poetry_image",
                "params": {"mode": "safe_i2v_first_frame"},
                "reason": "image_resource_invalid",
                "failure_type": "image_resource",
            }
        if "video.frame_difference" in missing_set:
            return {
                "action": "retry_with_params",
                "step_id": "poetry_video",
                "params": {
                    "motion_prompt_boost": (
                        "strengthen visible subject motion, camera push-in, route movement, "
                        "foreground-background parallax, and scene transition"
                    )
                },
                "reason": "video_motion_too_static",
                "failure_type": "video_frame_difference",
            }
        if missing_set.intersection({"video.url", "video.accessible", "video.ffprobe"}):
            cached = state.video if isinstance(state.video, dict) and state.video.get("source_type") == "curated_featured_case" else {}
            if cached:
                return {
                    "action": "reuse_cache",
                    "step_id": "poetry_video",
                    "target": "poetry_video",
                    "source": "curated_video_cache",
                    "payload": {"video": cached},
                    "reason": "reuse_valid_curated_video_cache",
                    "failure_type": "video_resource",
                }
            return {
                "action": "retry_with_params",
                "step_id": "poetry_video",
                "params": {"retry_reason": "video_resource_invalid"},
                "reason": "video_resource_invalid",
                "failure_type": "video_resource",
            }
        return {
            "action": "skip_step" if not missing else "retry_with_params",
            "step_id": step_id,
            "params": {},
            "reason": "success" if not missing else "generic_retryable_validation_failure",
            "failure_type": "" if not missing else "validation_missing_conditions",
        }

    def _failure_recommendation(self, step: PlanStep, error: str) -> dict:
        text = str(error or "")
        if "restricted" in text or "administrator-approved" in text or "generation is restricted" in text:
            return {
                "action": "stop_non_retryable",
                "step_id": step.step_id,
                "reason": "generation_permission_denied",
                "failure_type": "permission_denied",
            }
        return {
            "action": "retry_with_params",
            "step_id": step.step_id,
            "params": {},
            "reason": "retryable_skill_error",
            "failure_type": "skill_error",
        }


def _valid_hexi_coordinate(lon: Any, lat: Any) -> bool:
    try:
        lon_value = float(lon)
        lat_value = float(lat)
    except (TypeError, ValueError):
        return False
    return HEXI_LNG_RANGE[0] <= lon_value <= HEXI_LNG_RANGE[1] and HEXI_LAT_RANGE[0] <= lat_value <= HEXI_LAT_RANGE[1]


def _resource_exists(url: str) -> bool:
    parsed = urlparse(str(url or ""))
    if parsed.scheme in {"http", "https"}:
        return True
    local_path = _local_path_for_url(url)
    return bool(local_path and local_path.exists() and local_path.stat().st_size > 0)


def _local_path_for_url(url: str) -> Path | None:
    text = str(url or "").strip()
    if not text:
        return None
    if text.startswith("/static/"):
        return REPO_ROOT / "frontend" / text.lstrip("/")
    if text.startswith("/api/video-generation/normalized/"):
        return REPO_ROOT / "artifacts" / "normalized_ai_video" / text.removeprefix("/api/video-generation/normalized/")
    if text.startswith("/"):
        return REPO_ROOT / "frontend" / text.lstrip("/")
    path = Path(text)
    return path if path.is_absolute() else None


def _ffprobe_video(path: Path) -> bool | None:
    ffprobe_exe = shutil.which("ffprobe")
    if not ffprobe_exe:
        return _ffmpeg_probe_video(path)
    try:
        result = subprocess.run(
            [
                ffprobe_exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,duration",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return False
    streams = payload.get("streams") if isinstance(payload, dict) else None
    return bool(streams)


def _ffmpeg_probe_video(path: Path) -> bool | None:
    ffmpeg_exe = _ffmpeg_executable()
    if not ffmpeg_exe:
        return None
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-i", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    if "Video:" not in output:
        return False
    return "Duration:" in output


def _image_ratio_valid(path: Path) -> bool | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError:
        return False
    if not width or not height:
        return False
    return (width / height) >= 1.2


def _video_has_meaningful_frame_diff(path: Path) -> bool | None:
    ffmpeg_exe = _ffmpeg_executable()
    if not ffmpeg_exe:
        return None
    try:
        result = subprocess.run(
            [
                ffmpeg_exe,
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                "fps=1/3,scale=64:36,format=gray",
                "-frames:v",
                "3",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            check=False,
            capture_output=True,
            timeout=25,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    frame_size = 64 * 36
    raw = result.stdout or b""
    frames = [raw[index : index + frame_size] for index in range(0, len(raw), frame_size)]
    frames = [frame for frame in frames if len(frame) == frame_size]
    if len(frames) < 2:
        return False
    diffs = []
    for left, right in zip(frames, frames[1:]):
        total = sum(abs(a - b) for a, b in zip(left, right))
        diffs.append(total / frame_size)
    return max(diffs or [0.0]) >= 0.4


def _ffmpeg_executable() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe() or "")
    except Exception:
        return ""
