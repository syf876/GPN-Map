"""Free, geographic reference retrieval. Never calls a planner, LLM or paid proxy."""
import logging
import re

import requests

from backend.service.hexi_poetry import PLACE_GAZETTEER
from backend.service.video_search.adapters.bilibili_adapter import BilibiliAdapter
from backend.service.video_search.bilibili_rules import normalize_bvid
from backend.types.schema import VideoSearchRequest

logger = logging.getLogger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36",
           "Referer": "https://www.bilibili.com/"}


def resolve_place(location):
    label = str(location or "").strip()
    return next((p for p in PLACE_GAZETTEER
                 if label in [p["name"], *p["aliases"]]), None)


def search_reality(req: VideoSearchRequest):
    place = resolve_place(req.location)
    location = place["name"] if place else req.location.strip()
    result = {"ok": False, "status": "empty", "location": location,
              "query": f"{location} 实景 航拍", "best": None, "candidates": [],
              "candidate_count": 0, "photos": [], "message": "", "warnings": [],
              "billing": {"paid_api": False, "llm_tokens": 0, "estimated_cost_cny": 0}}
    if not location or location.startswith("PLC_") or location in {"大漠", "沙漠", "瀚海", "流沙"}:
        result.update(status="invalid_location", message="尚未确定具体真实地名，请先完成地图定位。")
        return result
    candidates = []
    seen = set()
    video_error = ""
    last_error = None
    try:
        # Fixed public endpoint: no configurable third-party paid search provider.
        # A plain place query recovers real travel videos when the narrow query is empty.
        for query in [result["query"], f"{location} 旅游"]:
            try:
                response = requests.get("https://api.bilibili.com/x/web-interface/search/type",
                                        params={"search_type": "video", "keyword": query, "page": 1},
                                        headers=HEADERS, timeout=(3, 8))
                response.raise_for_status()
                response.encoding = "utf-8"
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("code") != 0:
                    raise ValueError("B站搜索受限或返回异常")
            except (requests.RequestException, ValueError, TypeError) as exc:
                last_error = exc
                continue
            adapter = BilibiliAdapter()
            terms = [location] + (list(place["aliases"]) if place else [])
            for item in adapter._extract_result_items(payload):
                if not isinstance(item, dict):
                    continue
                title = adapter._normalize_text(item.get("title"))
                # Match actual metadata, never tags injected from the query itself.
                metadata = title + " " + str(item.get("description") or "")
                if not any(term in metadata for term in terms):
                    continue
                if re.search(r"AI|生成|游戏|动画|绘画|混剪|朗诵|背诵|3D|效果图|复原|重建|建模", metadata, re.I):
                    continue
                video = adapter._map_result_item(item, req, query, "real_http", "public_bilibili_search", "geo")
                if video and normalize_bvid(video.bvid) and video.bvid not in seen:
                    seen.add(video.bvid)
                    candidate = vars(video).copy()
                    candidate["url"] = candidate["external_url"] = f"https://www.bilibili.com/video/{video.bvid}"
                    candidates.append(candidate)
            candidates = candidates[:6]
            if candidates:
                result["query"] = query
                break
        if not candidates and last_error is not None:
            raise last_error
    except (requests.RequestException, ValueError, TypeError) as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        video_error = f"HTTP {code}" if code else type(exc).__name__
        result["video_error"] = video_error
        logger.info("reality_bilibili_unavailable type=%s", video_error)
        result["warnings"].append("B站视频检索失败或受访问限制；请稍后重试。")
    result.update(photos=[], candidates=candidates, candidate_count=len(candidates), best=candidates[0] if candidates else None,
                  ok=bool(candidates), status="succeeded" if candidates else ("failed" if video_error else "empty"))
    result["message"] = (f"B站视频检索成功：{location}，{len(candidates)}条。" if candidates else
                         f"未获取到{location}的有效B站视频，" + (f"B站接口异常或受限（{video_error}），请稍后重试。" if video_error else "请重试或更换具体地名。"))
    return result
