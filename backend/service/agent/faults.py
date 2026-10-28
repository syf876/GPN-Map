from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from backend.service.agent.schemas import AgentState, PlanStep, SkillResult


SUPPORTED_FAULT_TYPES = {
    "invalid_geojson_once",
    "image_fail_once",
    "video_low_motion_once",
    "resource_unavailable_once",
    "permission_denied",
}


def normalize_fault_profile(profile: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(profile, dict) or not profile:
        return {}
    if isinstance(profile.get("faults"), list):
        faults = [item for item in profile.get("faults") or [] if isinstance(item, dict)]
    else:
        faults = [profile]
    normalized: List[Dict[str, Any]] = []
    for index, fault in enumerate(faults):
        fault_type = str(fault.get("fault_type") or "").strip()
        if fault_type not in SUPPORTED_FAULT_TYPES:
            continue
        normalized.append(
            {
                "fault_id": str(fault.get("fault_id") or f"{fault_type}:{index}"),
                "fault_type": fault_type,
                "step": str(fault.get("step") or _default_step(fault_type)).strip(),
                "attempt": max(1, int(fault.get("attempt") or 1)),
                "seed": int(fault.get("seed") or profile.get("seed") or 0),
            }
        )
    return {"enabled": bool(normalized), "faults": normalized, "seed": int(profile.get("seed") or 0)}


def maybe_inject_before_call(state: AgentState, step: PlanStep, *, skill_name: str, attempt: int) -> SkillResult | None:
    fault = _matching_fault(state, step.step_id, attempt)
    if not fault:
        return None
    fault_type = fault["fault_type"]
    if fault_type == "permission_denied":
        _record_fault(state, fault, step.step_id, skill_name, "before_call")
        return SkillResult(ok=False, error="generation is restricted by deterministic fault profile")
    if fault_type == "image_fail_once" and step.step_id == "poetry_image":
        _record_fault(state, fault, step.step_id, skill_name, "before_call")
        return SkillResult(ok=False, output={"image": {}}, error="deterministic image_fail_once fault")
    if fault_type == "resource_unavailable_once" and step.step_id in {"poetry_image", "poetry_video"}:
        _record_fault(state, fault, step.step_id, skill_name, "before_call")
        key = "image" if step.step_id == "poetry_image" else "video"
        return SkillResult(ok=True, output={key: _unavailable_payload(key)}, message="deterministic resource_unavailable_once fault")
    return None


def maybe_inject_after_call(state: AgentState, step: PlanStep, output: Dict[str, Any], *, skill_name: str, attempt: int) -> Dict[str, Any]:
    fault = _matching_fault(state, step.step_id, attempt)
    if not fault:
        return output
    fault_type = fault["fault_type"]
    patched = deepcopy(output)
    if fault_type == "invalid_geojson_once" and step.step_id == "spatial_mapping":
        patched["geojson"] = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [120.0, 10.0]},
                    "properties": {"fault_type": fault_type, "seed": fault.get("seed", 0)},
                }
            ],
        }
        _record_fault(state, fault, step.step_id, skill_name, "after_call")
    elif fault_type == "video_low_motion_once" and step.step_id == "poetry_video":
        video = dict(patched.get("video") or {})
        video.setdefault("video_url", "https://example.com/fault-low-motion.mp4")
        video.setdefault("normalized_video_url", video.get("video_url"))
        video["fault_type"] = fault_type
        video["seed"] = fault.get("seed", 0)
        patched["video"] = video
        _record_fault(state, fault, step.step_id, skill_name, "after_call")
    return patched


def _matching_fault(state: AgentState, step_id: str, attempt: int) -> Dict[str, Any]:
    profile = normalize_fault_profile(state.fault_profile)
    if not profile.get("enabled"):
        return {}
    applied_ids = {str(item.get("fault_id") or "") for item in state.fault_events}
    for fault in profile.get("faults", []):
        if str(fault.get("fault_id") or "") in applied_ids:
            continue
        if fault.get("step") != step_id:
            continue
        if int(fault.get("attempt") or 1) != int(attempt):
            continue
        return fault
    return {}


def _record_fault(state: AgentState, fault: Dict[str, Any], step_id: str, skill_name: str, stage: str) -> None:
    event = {
        "fault_id": fault.get("fault_id", ""),
        "fault_type": fault.get("fault_type", ""),
        "step": step_id,
        "skill_name": skill_name,
        "attempt": fault.get("attempt", 1),
        "seed": fault.get("seed", 0),
        "stage": stage,
    }
    state.fault_events.append(event)
    state.add_trace("fault_injected", step_id=step_id, skill_name=skill_name, detail=event)


def _default_step(fault_type: str) -> str:
    return {
        "invalid_geojson_once": "spatial_mapping",
        "image_fail_once": "poetry_image",
        "video_low_motion_once": "poetry_video",
        "resource_unavailable_once": "poetry_image",
        "permission_denied": "poetry_image",
    }.get(fault_type, "")


def _unavailable_payload(kind: str) -> Dict[str, Any]:
    if kind == "image":
        return {
            "status": "succeeded",
            "image_url": "/static/__missing_fault_image__.jpg",
            "normalized_image_url": "/static/__missing_fault_image__.jpg",
            "fault_type": "resource_unavailable_once",
        }
    return {
        "status": "succeeded",
        "video_url": "/static/__missing_fault_video__.mp4",
        "normalized_video_url": "/static/__missing_fault_video__.mp4",
        "fault_type": "resource_unavailable_once",
    }
