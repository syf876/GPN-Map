import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_CASES_FILE = REPO_ROOT / "frontend" / "data" / "demo_video_cases.json"


@lru_cache(maxsize=1)
def load_curated_cases() -> List[Dict[str, Any]]:
    try:
        payload = json.loads(DEMO_CASES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = []
    cases = [dict(item) for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
    return [_normalize_case(case) for case in cases]


def find_curated_case(*values: Any) -> Optional[Dict[str, Any]]:
    normalized_values = [_normalize_match_text(value) for value in values if _normalize_match_text(value)]
    if not normalized_values:
        return None
    for case in load_curated_cases():
        candidates = [
            case.get("id"),
            case.get("title"),
            case.get("library_poem_id"),
            case.get("content"),
            f"{case.get('title', '')}{case.get('author', '')}",
            *(case.get("match_inputs") or []),
        ]
        normalized_candidates = [_normalize_match_text(value) for value in candidates if _normalize_match_text(value)]
        for value in normalized_values:
            for candidate in normalized_candidates:
                if value == candidate:
                    return case
    return None


def _normalize_case(case: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(case)
    if result.get("id") == "liangzhouci-yumenguan":
        content = "黄河远上白云间，一片孤城万仞山。\n羌笛何须怨杨柳，春风不度玉门关。"
        aliases = [
            "凉州词二首·其一",
            "凉州词",
            "出塞",
            "王之涣凉州词",
            "王之涣 凉州词",
            "凉州词王之涣",
            "羌笛何须怨杨柳，春风不度玉门关",
            "羌笛何须怨杨柳春风不度玉门关",
            "黄河远上白云间，一片孤城万仞山。羌笛何须怨杨柳，春风不度玉门关。",
        ]
        result.update(
            {
                "title": "凉州词二首·其一",
                "author": "王之涣",
                "dynasty": "唐",
                "content": content,
                "place": "玉门关",
                "places": ["玉门关", "敦煌", "河西走廊西段"],
                "imagery": ["边塞", "关隘", "风沙", "古道", "旅人", "驼队远景"],
                "emotion": "苍凉壮阔",
                "source_type": "curated_featured_case",
                "included_in_experiment_dataset": False,
            }
        )
        result["match_inputs"] = _unique([*(result.get("match_inputs") or []), *aliases])
    elif result.get("id") == "jiuquanzi-huayingliutiao":
        aliases = [
            "酒泉子·花映柳条",
            "酒泉子 花映柳条",
            "花映柳条",
            "酒泉子·花映柳条 温庭筠",
            "花映柳条，闲向绿萍池上。凭栏干，窥细浪，雨萧萧。",
        ]
        result.update(
            {
                "title": "酒泉子·花映柳条",
                "author": "温庭筠",
                "dynasty": "唐",
                "content": "花映柳条，闲向绿萍池上。凭栏干，窥细浪，雨萧萧。",
                "place": "酒泉",
                "places": ["酒泉", "河西绿洲", "春日水岸"],
                "imagery": ["柳条", "春风", "绿洲", "水岸", "酒泉地域意象"],
                "emotion": "清柔含蓄",
                "source_type": "curated_featured_case",
                "included_in_experiment_dataset": False,
            }
        )
        result["match_inputs"] = _unique([*(result.get("match_inputs") or []), *aliases])
    return result


def _unique(values: List[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalize_match_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    punctuation = set(",，.。!！?？;；:：'\"`·《》（）()[]【】、-_\\/")
    return "".join(ch for ch in text if not ch.isspace() and ch not in punctuation)
