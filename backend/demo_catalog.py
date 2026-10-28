"""Read-only, fixed-ID demo catalog. No provider imports and no generation fallback."""
import json
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frontend/data/demo_video_cases.json"
router = APIRouter(prefix="/api/demo", tags=["saved demos"])


def _normalize(value):
    return re.sub(r"[\W_]+", "", str(value or "").lower())


def _local_resource(url):
    if not str(url).startswith("/static/demo/"):
        return None
    path = (ROOT / "frontend" / url.lstrip("/")).resolve()
    try:
        path.relative_to((ROOT / "frontend/static/demo").resolve())
    except ValueError:
        return None
    return path if path.is_file() and path.stat().st_size else None


def catalog():
    try:
        cases = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    for case in cases:
        case["availability"] = {k: bool(_local_resource(case.get(field, ""))) for k,field in {
            "result": "saved_result_url", "image": "cached_image_url", "video": "cached_video_url"}.items()}
        case["missing_resources"] = [key for key, available in case["availability"].items() if not available]
    return cases


def find_case(*values):
    values = {_normalize(x) for x in values if _normalize(x)}
    for case in catalog():
        matches = [case["id"], case.get("library_poem_id"), case.get("title"), case.get("content"), *(case.get("match_inputs") or [])]
        if values.intersection(_normalize(x) for x in matches if x):
            return case
    return None


def saved_result(case):
    path = _local_resource(case.get("saved_result_url", ""))
    if not path:
        raise HTTPException(status_code=503, detail="演示案例预存解析和定位暂不可用，不会自动生成")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(status_code=503, detail="演示案例预存结果暂不可用，不会自动生成")
    if not result.get("prediction") or not (result.get("map") or {}).get("features"):
        raise HTTPException(status_code=503, detail="演示案例预存解析或定位不完整，不会自动生成")
    result["video"] = {"ok": True, "query": "", "best": None, "candidates": [], **(result.get("video") or {})}
    if not _local_resource(case.get("cached_image_url", "")): result["image"] = {}
    if not _local_resource(case.get("cached_video_url", "")): result["ai_video"] = {}
    result["demo_case_id"] = case["id"]
    result["missing_resources"] = case["missing_resources"]
    return result


@router.get("/cases")
def list_cases():
    return catalog()


@router.get("/cases/{case_id}")
def get_case(case_id: str):
    case = next((x for x in catalog() if x["id"] == case_id), None)
    if not case:
        raise HTTPException(status_code=404, detail="演示案例不存在")
    return {"case": case, "result": saved_result(case)}
