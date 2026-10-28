from __future__ import annotations

import re
from typing import Any, Iterable, List

from backend.service.agent.schemas import AgentTarget, GoalConstraints


VIDEO_TERMS = ("视频", "短片", "影像", "镜头", "动态", "运动", "转场", "时长", "秒", "配乐", "旁白", "朗诵")
IMAGE_TERMS = ("意境图", "图片", "生图", "画面", "海报", "插画", "视觉")
MAP_TERMS = ("地图", "定位", "地名", "空间", "路线", "坐标", "地理")
NO_REUSE_TERMS = ("重新生成", "重做", "不要复用", "不复用", "不用缓存", "忽略缓存")
REUSE_TERMS = ("复用", "已有", "缓存", "直接用", "不重复生成")

STYLE_KEYWORDS = {
    "水墨": ("水墨", "国画", "墨色"),
    "写实": ("写实", "真实", "纪录片"),
    "电影感": ("电影", "电影感", "宽银幕"),
    "唐风": ("唐风", "盛唐", "唐代"),
    "边塞苍凉": ("边塞", "苍凉", "荒寒", "风沙"),
    "春日清柔": ("春日", "清柔", "柳", "细雨", "绿萍"),
    "国风": ("国风", "古风", "中式"),
}

ACTION_KEYWORDS = {
    "驼队远行": ("驼队", "商旅", "行旅"),
    "柳条拂动": ("柳条", "杨柳", "柳枝"),
    "风沙掠过": ("风沙", "黄沙", "边风"),
    "细雨落水": ("细雨", "雨", "雨萧萧", "水面"),
    "羌笛吹奏": ("羌笛", "笛"),
    "旅人望关": ("旅人", "远望", "登临", "凭栏"),
}

CAMERA_KEYWORDS = {
    "推镜": ("推镜", "推进", "push"),
    "拉远": ("拉远", "远景", "后撤"),
    "俯拍": ("俯拍", "航拍", "鸟瞰"),
    "跟拍": ("跟拍", "跟随"),
    "平移": ("平移", "横移", "pan"),
    "慢镜头": ("慢镜头", "慢动作"),
}

AUDIO_KEYWORDS = {
    "诗词朗诵": ("朗诵", "旁白", "念白"),
    "环境声": ("环境声", "自然声", "氛围声"),
    "配乐": ("配乐", "音乐", "背景乐"),
    "羌笛氛围": ("羌笛", "笛声"),
    "古琴配乐": ("古琴", "琴声"),
    "风声环境": ("风声", "风沙声"),
    "雨声环境": ("雨声", "雨萧萧"),
}


def interpret_goal(
    *,
    requirement: str = "",
    requested_target: AgentTarget = "map",
    poem_text: str = "",
) -> GoalConstraints:
    text = str(requirement or "").strip()
    target = _infer_target(text, requested_target)
    duration = _infer_duration_seconds(text)
    visual_style = _matches(text, STYLE_KEYWORDS)
    narrative_actions = _matches(" ".join([text, poem_text]), ACTION_KEYWORDS)
    camera_requirements = _matches(text, CAMERA_KEYWORDS)
    audio_requirements = _matches(text, AUDIO_KEYWORDS)
    reuse_existing = _infer_reuse(text)

    return GoalConstraints(
        raw_requirement=text,
        target=target,
        needs_map=True,
        needs_image=target in {"image", "video"},
        needs_video=target == "video",
        duration_seconds=duration,
        visual_style=visual_style,
        narrative_actions=narrative_actions,
        camera_requirements=camera_requirements,
        audio_requirements=audio_requirements,
        reuse_existing=reuse_existing,
    )


def _infer_target(text: str, requested_target: AgentTarget) -> AgentTarget:
    if not text:
        return requested_target
    if _contains_any(text, VIDEO_TERMS):
        return "video"
    if _contains_any(text, IMAGE_TERMS):
        return "image"
    if _contains_any(text, MAP_TERMS):
        return "map"
    return requested_target


def _infer_duration_seconds(text: str) -> int:
    for pattern in (r"(\d{1,3})\s*(?:秒|s|S)", r"时长\s*(\d{1,3})"):
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            return max(1, min(120, value))
    return 0


def _infer_reuse(text: str) -> bool:
    if _contains_any(text, NO_REUSE_TERMS):
        return False
    if _contains_any(text, REUSE_TERMS):
        return True
    return True


def _matches(text: str, mapping: dict[str, Iterable[str]]) -> List[str]:
    result: List[str] = []
    for label, terms in mapping.items():
        if _contains_any(text, terms):
            result.append(label)
    return result


def _contains_any(text: str, terms: Iterable[Any]) -> bool:
    haystack = str(text or "").lower()
    return any(str(term or "").lower() in haystack for term in terms if str(term or "").strip())
