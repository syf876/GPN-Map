from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from backend.types.schema import VeoGenerationRequest


@dataclass(frozen=True)
class StoryboardShot:
    index: int
    shot_id: str
    title: str
    duration_seconds: float
    image_variant: str
    scene: str
    action: str
    camera: str
    emotion: str
    narration_text: str
    prompt: str


def build_veo_prompt(req: VeoGenerationRequest) -> str:
    shots = build_storyboard(req)
    overview = build_wan_prompt(req)
    shot_block = " ".join(
        f"Shot {shot.index}: {shot.scene}; {shot.action}; {shot.camera}; {shot.emotion}."
        for shot in shots
    )
    return (
        f"{overview} Treat the supplied image as the persistent visual anchor for the entire film, not only as the opening keyframe. "
        f"Every shot must preserve its subject identity, scene geography, composition logic, color palette, ink-wash illustration style, and frontier mood. "
        f"Then advance through connected narrative shots with natural continuity. {shot_block}"
    )


def build_wan_prompt(req: VeoGenerationRequest) -> str:
    context = _resolve_context(req)
    if _is_safe_video_prompt_request(req):
        return build_safe_wan_prompt(req)
    shots = build_storyboard(req)
    base_prompt = _build_wan_base_prompt(context)
    anchor_requirements = _build_visual_anchor_requirements(context)
    director_requirements = _build_wan_director_requirements(context, shots)
    return f"{base_prompt}\n\n{anchor_requirements}\n\n{director_requirements}"


def build_safe_wan_prompt(req: VeoGenerationRequest) -> str:
    context = _resolve_context(req)
    places = _sanitize_safe_prompt_text(context.get("places") or context.get("location") or "Hexi Corridor")
    recitation = _voiceover_script_text(_sanitize_safe_prompt_text(
        context.get("recitation") or context.get("narration") or context.get("content") or context.get("text") or ""
    ))
    image_url = str(context.get("image_url") or "").strip()
    return "\n".join(
        [
            "Use the provided image as the exact opening first frame. After the first frame, continue into a cinematic narrative sequence with clear scene progression, not just subtle motion or slight camera shake. Do not remain a near-static animated illustration.",
            "Continue from this image into a 15-second narrative sequence. Do not create a new unrelated opening scene. Do not merely animate the still image.",
            "The first 0-1 seconds should match the original composition, then the film must move beyond a poster-like animation into new connected foreground, midground, and background details.",
            "Safe natural-scene version for Wan image-to-video. Keep the same guofeng ink-wash and pale-color illustration style, same horizon, road direction, distant gate-town silhouette, willow, sandbar, cloud shadow, mountains, and ancient road from the opening image.",
            "必须以当前意境图作为开场首帧，后续画面要继续生成叙事变化，而不是只让原图轻微晃动、呼吸、慢推或局部风动。",
            "以输入图片作为视频第一帧和视觉风格参考，开场保持原图构图；随后在 15 秒内生成连续叙事镜头，后续画面必须有新景别、新前景/中景/远景细节和可见空间推进。",
            "0-3 seconds: preserve the opening composition and first-frame geography; wind rises, clouds move, water or sandbar texture begins to flow, willow branches and grass start changing.",
            "3-10 seconds: a distant traveler or camel caravan moves along the riverbed or ancient road; the camera slowly pushes or tracks forward; the far gate town becomes closer and clearer; birds pass across the sky; moonlight, cloud shadow, water surface, sand, and vegetation visibly change.",
            "10-15 seconds: keep pushing toward the pass, mountains, or sandbar; environmental light and shadow intensify; the journey resolves into a homesick frontier travel ending in the same visual world.",
            "Required visible actions: moving cloud layers, flowing wind-sand, flowing water surface or sandbar texture, swaying willow branches, birds crossing, traveler walking, distant camel caravan passing, camera push, lateral pan, pull-back, and a far gate town gradually appearing.",
            "No text, no subtitles, no watermark, no logo, no map, no modern city, no photorealistic footage, no abrupt scene jump.",
            f"title: {_sanitize_safe_prompt_text(context.get('title') or '')}",
            f"author: {_sanitize_safe_prompt_text(context.get('author') or '')}",
            f"dynasty: {_sanitize_safe_prompt_text(context.get('dynasty') or '')}",
            f"content: {_sanitize_safe_prompt_text(context.get('content') or context.get('text') or '')}",
            f"places: {places}",
            f"voiceover_script: {recitation}",
            *_build_audio_instruction_lines(recitation),
            f"source_image_url: {image_url}",
        ]
    )


def _is_safe_video_prompt_request(req: VeoGenerationRequest) -> bool:
    prediction = req.prediction if isinstance(req.prediction, dict) else {}
    return bool(prediction.get("safe_video_prompt") or prediction.get("safe_i2v_first_frame"))


def _sanitize_safe_prompt_text(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "war": "remote hardship",
        "soldier": "traveler",
        "army": "caravan",
        "battle": "distant past",
        "military": "frontier",
        "weapon": "object",
        "armor": "robe",
        "blood": "red sunset",
        "attack": "advance",
        "战争": "苍茫往事",
        "士兵": "旅人",
        "军旅": "边地行旅",
        "军队": "驼队",
        "武器": "器物",
        "盔甲": "衣袍",
        "厮杀": "风沙",
        "攻城": "远城晨昏",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
        text = text.replace(source.capitalize(), replacement)
        text = text.replace(source.upper(), replacement)
    return text.strip()


def _build_visual_anchor_requirements(context: Dict[str, Any]) -> str:
    source_image_url = str(context.get("image_url") or "").strip()
    story_arc = _story_arc_text(context)
    return "\n".join(
        [
            "VISUAL ANCHOR: the supplied image_url is the first frame and master reference for the whole video, not merely a loose inspiration.",
            "Use the provided image as the opening first frame and continue the narrative beyond the first frame, not a subtle looping animation.",
            "Use the provided image as the exact opening first frame. After the first frame, continue into a cinematic narrative sequence with clear scene progression, not just subtle motion or slight camera shake. Do not remain a near-static animated illustration.",
            "Continue from this image into a 15-second narrative sequence. Do not create a new unrelated opening scene. Do not merely animate the still image.",
            "The first 0-1 seconds should match the original composition, then the film must move beyond a poster-like animation into new connected foreground, midground, and background details.",
            "After the first frame, generate visible narrative progression: clouds travel across the sky, wind and sand flow across the road or sandbar, water or riverbank texture moves, willow branches sway, birds cross the frame, a distant traveler walks, a camel caravan passes far away, and the camera slowly pushes or tracks from the opening scene toward the gate, mountains, sandbars, and old road. Maintain the same ink-wash Chinese style, same location, same atmosphere, and natural continuity from the first frame.",
            "The clip must open exactly from the first-frame composition, then naturally extend outward into new connected story beats and later visual changes.",
            "Do not only add minor shaking, breathing, slow zoom, or a cinemagraph effect to the original image; create visible narrative development after the opening frame.",
            "必须以当前意境图作为开场首帧，后续画面要继续生成叙事变化，而不是只让原图轻微晃动。",
            "Preserve the same main subject, architecture or landscape layout, horizon, road/wall/water direction, color palette, brush texture, lighting, and guofeng ink-wash pale-color illustration style from beginning to end.",
            "CONTINUITY: create one connected poetic story inside the same world. The camera may push, track, pan, orbit, or pull back, but every new beat must grow naturally from the previous frame.",
            "Do not jump to an unrelated location, new costume style, new season, new weather system, or new visual universe.",
            "STYLE LOCK: keep a Chinese ink-wash + pale color + illustrated cinematic look for the entire clip.",
            "Do not become photorealistic, live-action, real photo look, documentary footage, modern footage, 3D render, or stock-video style.",
            f"CONTINUOUS STORY ARC: {story_arc}",
            f"source_image_url: {source_image_url}",
        ]
    )


def _build_wan_base_prompt(context: Dict[str, Any]) -> str:
    imagery_text = _imagery_text(context["imagery"])
    places = context.get("places") or context.get("location") or ""
    recitation = _voiceover_script_text(
        context.get("recitation") or context.get("narration") or context.get("content") or context.get("text") or ""
    )
    story_arc = _story_arc_text(context)
    lines = [
        "请根据以下诗词信息生成一段有声中国古典诗词意境短视频。",
        "Wan I2V 参数语义：模型保持 wan2.6-i2v-flash，resolution=720P，audio=true，目标时长约 15 秒。",
        "English hard rule: Use the provided image as the exact opening first frame. After the first frame, continue into a cinematic narrative sequence with clear scene progression, not just subtle motion or slight camera shake. Do not remain a near-static animated illustration.",
        "Continue from this image into a 15-second narrative sequence. Do not create a new unrelated opening scene. Do not merely animate the still image.",
        "The first 0-1 seconds should match the original composition, then the film must move beyond a poster-like animation into new connected foreground, midground, and background details.",
        "核心视频要求：必须以当前意境图作为开场首帧，后续画面要继续生成叙事变化，而不是只让原图轻微晃动、呼吸、慢推、局部风动或做静态图微动。",
        "以输入图片作为视频第一帧和视觉风格参考，开场保持原图构图；随后在 15 秒内生成连续叙事镜头，后续画面必须有新景别、新前景/中景/远景细节和可见空间推进。",
        "15 秒连续镜头叙事：0-3 秒保留首帧构图，风起云动，水流/沙洲/草木开始变化；3-10 秒旅人沿河道或古道前行，镜头缓慢推进或平移，远处关城逐渐靠近，飞鸟掠过，月色、云影、水面、风沙和柳枝持续变化；10-15 秒镜头继续推进到关隘、远山或沙洲，环境光影变化增强，形成思乡、边塞、旅途的故事感结尾。",
        "连续变化的具体视觉动作必须明显出现，可组合使用：云层移动、风沙流动、水面流动、柳枝摇曳、飞鸟掠过、旅人行走、驼队远景经过、镜头推进、镜头平移、镜头拉远、远处关城逐渐显现。",
        "镜头要保持首帧画风、场景、主体、空间方位和整体构图逻辑连续，不要突然换成完全不同的画面，也不要只做局部轻微动效；不要现代 UI、字幕、海报排版、水印或地图元素。",
        "画面风格：国风、水墨与淡彩结合、边塞诗意、电影感光影、层次分明、气氛连贯。",
        "连续叙事设计：" + story_arc,
        "音频要求：开启 audio=true，并严格按以下音频指令生成。",
        *_build_audio_instruction_lines(recitation),
        f"title: {context.get('title') or ''}",
        f"author: {context.get('author') or ''}",
        f"dynasty: {context.get('dynasty') or ''}",
        f"content: {context.get('content') or context.get('text') or ''}",
        f"places: {places}",
        f"imagery: {imagery_text}",
        f"emotion: {context.get('emotion') or ''}",
        f"analysis_text: {context.get('analysis_text') or ''}",
        f"voiceover_script: {recitation}",
        "请将河西走廊相关地貌、地点、意象和情绪融合进镜头运动与光影变化，形成一段完整、克制、连贯的诗意视频。",
    ]
    return "\n".join(lines)


def _build_wan_director_requirements(context: Dict[str, Any], shots: List[StoryboardShot]) -> str:
    shot_summary = " ".join(
        f"Shot {shot.index} {shot.title}: {shot.scene}; {shot.action}; {shot.camera}; {shot.emotion}."
        for shot in shots
    )
    continuity_summary = (
        "Continuity lock: each shot must reuse the same reference-image world, color family, illustrated brushwork, "
        "lighting direction, and spatial relationship between foreground, subject, and background. "
        "Shot transitions must be motivated by camera movement, wind, dust, water, cloth, willow branches, cloud layers, or light changes, not by replacing the scene. "
        "Do not introduce photorealistic texture, real-camera footage, photographic skin, modern clothing, modern buildings, or a completely new scene."
    )
    places = context.get("places") or context.get("location") or "河西走廊"
    imagery_text = _imagery_text(context["imagery"])
    recitation = _voiceover_script_text(
        context.get("recitation") or context.get("narration") or context.get("content") or context.get("text") or ""
    )
    story_arc = _story_arc_text(context)
    audio_requirements = " ".join(_build_audio_instruction_lines(recitation))
    return (
        "补充视频导演要求："
        "English hard rule: Use the provided image as the exact opening first frame. After the first frame, continue into a cinematic narrative sequence with clear scene progression, not just subtle motion or slight camera shake. Do not remain a near-static animated illustration. "
        "Continue from this image into a 15-second narrative sequence. Do not create a new unrelated opening scene. Do not merely animate the still image. "
        "The first 0-1 seconds should match the original composition, then the film must move beyond a poster-like animation into new connected foreground, midground, and background details. "
        "镜头分段必须保留：0-3 秒严格从输入首帧开场，第一秒保持首帧构图可辨，然后风起云动，水流、沙洲、草木、柳枝或水墨纹理开始变化；"
        "3-10 秒旅人沿河道或古道前行，镜头缓慢推进或平移，远处关城逐渐靠近，飞鸟掠过，月色、云影、水面、风沙和柳枝持续变化；"
        "10-15 秒镜头继续推进到关隘、远山或沙洲，环境光影变化增强，形成思乡、边塞、旅途的故事感结尾。"
        f"自动叙事线：{story_arc}"
        f"关键词动态叙事元素：地点 {places}；意象 {imagery_text}；情绪 {context.get('emotion') or '克制、苍茫'}。"
        "这些关键词必须转化为云层移动、风沙流动、水面流动、柳枝摇曳、飞鸟掠过、旅人行走、驼队远景经过、远山、关隘、道路、月色或光影推进等可见动态元素。"
        f"配音要求：audio=true；{audio_requirements}"
        "负面要求：不要静态图轻微抖动，不要只做原图慢推，不要只有局部风动，不要近静态插画动效，不要突然切镜，不要字幕，不要文字排版，不要水印，不要 logo，不要地图，不要坐标轴，不要现代广告海报，不要采访感旁白。"
        f"Storyboard: {continuity_summary} {shot_summary}"
    )


def _build_audio_instruction_lines(recitation: str) -> List[str]:
    script = _voiceover_script_text(recitation)
    return [
        "Audio must be a clear Mandarin Chinese voice reading the poem aloud.",
        "Use natural human narration only.",
        "Read the poem text exactly as provided.",
        "Do not add flute music, background music, singing, sound effects, wind noise, ambient sound, explanation, commentary, or extra words.",
        "The voice should be calm, poetic, and synchronized within the 15-second video.",
        "音频必须是自然普通话人声朗读当前完整诗词，只朗读诗词正文，不要解释、旁白扩写、音乐、笛声、环境音、风声、马蹄声或其他音效。",
        "Voiceover text / 朗读文本：",
        script,
    ]


def _voiceover_script_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", "\n").splitlines()).strip()


def build_wan_negative_prompt(req: VeoGenerationRequest) -> str:
    context = _resolve_context(req)
    emotion = str(context.get("emotion") or "").lower()
    extra = (
        "emotionally empty posing, frozen expression, no story turn"
        if any(token in emotion for token in ("homesick", "sorrow", "grief"))
        else "stiff body, tiny movement, no spatial change"
    )
    parts = [
        "static image animation",
        "cinemagraph",
        "only subtle motion",
        "only minor shake",
        "only image breathing",
        "single scene loop",
        "repeating first frame",
        "only slight wind",
        "only slow camera drift",
        "only slow zoom",
        "only breathing motion",
        "near-static animated illustration",
        "still-image screensaver",
        "only lip movement",
        "no shot progression",
        "no scene transition",
        "no visible story development",
        "no later-frame content change",
        "no character action",
        "no narrative change",
        "modern city",
        "photorealistic",
        "real photo look",
        "photo-real",
        "live action",
        "documentary footage",
        "modern footage",
        "stock video",
        "realistic photography",
        "camera phone footage",
        "3D render",
        "scene replacement",
        "unrelated new location",
        "style shift",
        "different art style",
        "new subject identity",
        "subtitle",
        "watermark",
        "cartoon",
        "low resolution",
        "deformed anatomy",
        extra,
    ]
    return ", ".join(parts)


def build_storyboard_preview(req: VeoGenerationRequest) -> Dict[str, Any]:
    context = _resolve_context(req)
    shots = build_storyboard(req)
    return {
        "narration_text": str(context.get("narration") or context.get("text") or "").strip(),
        "source": "prompt_builder.build_storyboard",
        "shots": [
            {
                "order": shot.index,
                "scene": shot.scene,
                "visual_prompt": shot.prompt,
                "duration": shot.duration_seconds,
                "place": str(context.get("location") or "").strip(),
                "imagery": list(context.get("imagery") or []),
                "emotion": shot.emotion or str(context.get("emotion") or "").strip(),
            }
            for shot in shots
        ],
    }


def build_storyboard(req: VeoGenerationRequest) -> List[StoryboardShot]:
    context = _resolve_context(req)
    text = context["text"]
    if not text:
        raise ValueError("poem text is required for storyboard generation")

    poem_lines = context.get("poem_lines") or _poem_lines(req)
    shot_count = 4 if len(poem_lines) >= 4 else 3
    narration_groups = _group_lines_for_shots(poem_lines, shot_count)
    shot_specs = _shot_specs(context, shot_count)

    shots: List[StoryboardShot] = []
    for index, spec in enumerate(shot_specs, start=1):
        narration_text = narration_groups[index - 1] or context["text"]
        shots.append(
            StoryboardShot(
                index=index,
                shot_id=spec["shot_id"],
                title=spec["title"],
                duration_seconds=float(spec["duration_seconds"]),
                image_variant=str(spec["image_variant"]),
                scene=str(spec["scene"]),
                action=str(spec["action"]),
                camera=str(spec["camera"]),
                emotion=str(spec["emotion"]),
                narration_text=narration_text,
                prompt=_build_shot_prompt(context, spec, narration_text, index, shot_count),
            )
        )
    return shots


def _build_shot_prompt(
    context: Dict[str, Any],
    spec: Dict[str, Any],
    narration_text: str,
    index: int,
    shot_count: int,
) -> str:
    poem_meta = _poem_meta_line(context)
    imagery_text = _imagery_text(context["imagery"])
    subject_line = _subject_line(context)
    sequence_line = (
        f"This is shot {index} of {shot_count} in a chronological classical-poetry narrative short film. "
        "It must clearly continue the previous story beat rather than looping one still frame."
    )
    return (
        "Generate one short narrative video segment in a guofeng ink-wash pale-color illustrated cinematic style. "
        f"{sequence_line} The reference image is the visual anchor for this segment and the entire film; begin from it and then extend into visible story action, not a static-image micro-motion loop. "
        "Preserve its subject identity, architecture/landscape layout, horizon, composition, color palette, brush texture, lighting direction, and poetic frontier mood. "
        f"Poem line focus: {context['text']}. {poem_meta} "
        f"Location: {context['location'] or 'Hexi Corridor'}. Dynasty context: {context['dynasty'] or 'classical China'}. "
        f"Emotion arc: {context['emotion'] or 'solemn frontier feeling'}. Imagery: {imagery_text}. "
        f"Scene continuity requirement: {subject_line}. "
        f"Scene: {spec['scene']}. Action: {spec['action']}. Camera: {spec['camera']}. Emotional beat: {spec['emotion']}. "
        f"Narration for this segment: {narration_text}. "
        "The video must show visible movement, blocking changes, spatial depth, and story progression while staying in the same reference-image world. "
        "Do not make a cinemagraph, do not keep a single scene loop, do not rely on only subtle motion, slight shake, breathing, or slow zoom. "
        "Do not become photorealistic, real photo look, live action, documentary footage, modern footage, 3D render, or stock-video style."
    )


def _resolve_context(req: VeoGenerationRequest) -> Dict[str, Any]:
    prediction = req.prediction if isinstance(req.prediction, dict) else {}
    text = _request_text(req, "text") or _prediction_text(prediction)
    narration = _request_text(req, "narration")
    poem_match = _resolve_poem_match(narration or text)
    title = _request_text(req, "title") or _prediction_field(prediction, ("title", "poem_title")) or poem_match.get("title", "")
    author = _request_text(req, "author") or _prediction_field(prediction, ("author", "poem_author")) or poem_match.get("author", "")
    dynasty = (
        _request_text(req, "dynasty")
        or _prediction_label(prediction.get("dynasty"))
        or poem_match.get("dynasty", "")
    )
    content = (
        _request_text(req, "content")
        or poem_match.get("poem_text", "")
        or narration
        or text
    )
    location = _request_text(req, "location") or _prediction_label(prediction.get("place"))
    places = _request_text(req, "places") or location
    emotion = _request_text(req, "emotion") or _prediction_label(prediction.get("emotion"))
    imagery = req.imagery or _prediction_imagery(prediction)
    analysis_text = _request_text(req, "analysis_text") or _prediction_field(
        prediction,
        ("analysis_text", "analysisText", "interpretation", "summary"),
    )
    recitation = _request_text(req, "recitation") or poem_match.get("poem_text", "") or narration or content or text
    poem_lines = _split_lines(recitation) or _split_lines(content) or _split_lines(text)
    return {
        "text": text,
        "title": title,
        "author": author,
        "content": content,
        "location": location,
        "places": places,
        "dynasty": dynasty,
        "emotion": emotion,
        "imagery": imagery,
        "narration": narration,
        "analysis_text": analysis_text,
        "recitation": recitation,
        "poem_lines": poem_lines,
        "image_url": _request_text(req, "image_url"),
    }


def _poem_lines(req: VeoGenerationRequest) -> List[str]:
    text = _request_text(req, "recitation") or _request_text(req, "narration") or _request_text(req, "text")
    lines = _split_lines(text)
    if lines:
        return lines
    fallback = _request_text(req, "text")
    return [fallback] if fallback else []


def _request_text(req: VeoGenerationRequest, field_name: str, default: str = "") -> str:
    value = getattr(req, field_name, default)
    if value is None:
        value = default
    if isinstance(value, (list, tuple, set)):
        return "、".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _resolve_poem_match(text: str) -> Dict[str, str]:
    if not str(text or "").strip():
        return {}
    try:
        from .poetry_audio import resolve_full_poem

        match = resolve_full_poem(str(text or "").strip())
    except Exception:
        return {}
    return {
        "title": str(getattr(match, "title", "") or "").strip(),
        "author": str(getattr(match, "author", "") or "").strip(),
        "dynasty": str(getattr(match, "dynasty", "") or "").strip(),
        "poem_text": str(getattr(match, "poem_text", "") or "").strip(),
    }


def _split_lines(text: str) -> List[str]:
    normalized = str(text or "").replace("\r", "\n")
    chunks = [
        item.strip()
        for item in normalized.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n")
    ]
    lines = [item for item in chunks if item]
    if len(lines) == 1 and "，" in lines[0]:
        phrases = [item.strip() for item in lines[0].replace("，", "\n").split("\n") if item.strip()]
        if len(phrases) > 1:
            return phrases
    return lines


def _group_lines_for_shots(lines: Sequence[str], shot_count: int) -> List[str]:
    clean_lines = [str(item or "").strip() for item in lines if str(item or "").strip()]
    if not clean_lines:
        return [""] * shot_count
    if len(clean_lines) >= shot_count:
        groups: List[List[str]] = [[] for _ in range(shot_count)]
        for index, line in enumerate(clean_lines):
            bucket = min(shot_count - 1, int(index * shot_count / len(clean_lines)))
            groups[bucket].append(line)
        return ["，".join(group) if group else clean_lines[min(index, len(clean_lines) - 1)] for index, group in enumerate(groups)]
    if len(clean_lines) == 1:
        return [clean_lines[0] for _ in range(shot_count)]
    groups = list(clean_lines)
    while len(groups) < shot_count:
        groups.append(clean_lines[-1])
    return groups[:shot_count]


def _shot_specs(context: Dict[str, Any], shot_count: int) -> List[Dict[str, Any]]:
    closing_scene = _closing_scene(context)
    middle_reflection = _reflection_action(context)
    opening_scene = _opening_scene(context)
    travel_scene = _travel_scene(context)
    if shot_count == 4:
        return [
            {
                "shot_id": "opening_wide",
                "title": "Opening Frontier Establishing",
                "duration_seconds": 3.0,
                "image_variant": "opening_wide",
                "scene": opening_scene,
                "action": "Open from the exact supplied first-frame composition, keep the same road, water, dunes, mountains, trees, gate-town, sandbar, or architectural silhouettes; then wind, sand, water texture, willow branches, cloud layers, birds, and ink diffusion begin to move inside that same world.",
                "camera": "Hold the first frame long enough to recognize it, then make a restrained push inward without replacing the scene.",
                "emotion": "Austere anticipation and frontier vastness.",
            },
            {
                "shot_id": "travel_push",
                "title": "Traveler Advances",
                "duration_seconds": 4.0,
                "image_variant": "travel_push",
                "scene": travel_scene,
                "action": "A distant traveler, camel caravan, willow branch, dust stream, water surface, flying bird, or cloud shadow advances within the same image-world, creating a clear story beat rather than a tiny loop.",
                "camera": "Track through the same space from wide to medium and reveal depth layers toward the pass, wall, desert road, distant mountains, or architecture details without changing art style.",
                "emotion": "Narrative momentum and determination.",
            },
            {
                "shot_id": "pause_reflect",
                "title": "Pause and Look Back",
                "duration_seconds": 4.0,
                "image_variant": "pause_reflect",
                "scene": "A relay road bend, ridge, watchtower edge, or borderland stop where the traveler can pause and react.",
                "action": f"{middle_reflection} Let willow leaves, sand, water ripples, cloud shadows, or ink washes respond around the figure; keep the same distant traveler silhouette, landscape geometry, and color palette as the reference image.",
                "camera": "Use a side-follow or slight arc in the same location to show the figure pausing, turning, raising the head, or slowly walking onward before the next beat.",
                "emotion": "Reflection, homesickness, or restrained frontier lyricism.",
            },
            {
                "shot_id": "closing_empty",
                "title": "Closing Empty Shot",
                "duration_seconds": 4.0,
                "image_variant": "closing_empty",
                "scene": closing_scene,
                "action": "Let the traveler, camel caravan, or focal landscape element move deeper toward the pass, road, desert, sandbar, gate town, or far mountains, then become a silhouette, exit, or diminish inside the same landscape so the ending lands on environment and feeling.",
                "camera": "Pull back or glide toward a detail of the gate, desert, far ridge, or city wall, then let the frame breathe into a calm concluding composition that still matches the supplied image.",
                "emotion": "Aftertaste, distance, and emotional resolution.",
            },
        ]
    return [
        {
            "shot_id": "opening_wide",
            "title": "Opening Frontier Establishing",
            "duration_seconds": 4.0,
            "image_variant": "opening_wide",
            "scene": opening_scene,
            "action": "Open from the exact supplied first-frame composition, then animate wind, sand, water texture, willow branches, cloud layers, flying birds, and ink diffusion inside the same road, water, dunes, mountains, trees, gate town, sandbar, or architectural silhouettes.",
            "camera": "Hold the first frame long enough to recognize it, then make a restrained push inward without replacing the scene.",
            "emotion": "Austere anticipation and frontier vastness.",
        },
        {
            "shot_id": "travel_push",
            "title": "Traveler Advances",
            "duration_seconds": 6.0,
            "image_variant": "travel_push",
            "scene": travel_scene,
            "action": "A distant traveler, camel caravan, willow branch, dust stream, water surface, bird, or cloud shadow advances through the same image-world, giving a readable story beat and a clear spatial change.",
            "camera": "Track from wide to medium through the same space toward the pass, wall, desert road, distant mountains, or architecture details while preserving the reference-image palette and illustrated brushwork.",
            "emotion": "Narrative propulsion with visible human action.",
        },
        {
            "shot_id": "closing_empty",
            "title": "Closing Resolution",
            "duration_seconds": 5.0,
            "image_variant": "closing_empty",
            "scene": closing_scene,
            "action": "Resolve on the original landscape relationship: the traveler, caravan, or focal landscape element pauses, recedes, crosses deeper into the pass road, or becomes a silhouette while wind, light, ink texture, and frontier atmosphere continue from the same world.",
            "camera": "Pull back or glide toward a detail of the gate, desert, far ridge, or city wall, then settle into a final composed frame that still matches the supplied image's style and geography.",
            "emotion": "Lingering emotion and final poetic echo.",
        },
    ]


def _opening_scene(context: Dict[str, Any]) -> str:
    keywords = " ".join(context.get("imagery") or []) + " " + str(context.get("location") or "")
    if any(token in keywords for token in ("玉门关", "阳关", "关", "塞", "边")):
        return "A distant gate town, ancient road, willow line, sandbar, mountains, and open desert horizon in the Hexi Corridor."
    return "A broad borderland vista with dunes, cold wind, relay road, distant mountains, and a lone figure held inside the vast environment."


def _travel_scene(context: Dict[str, Any]) -> str:
    imagery = " ".join(context.get("imagery") or [])
    if any(token in imagery for token in ("马", "驿", "关", "城")):
        return "A relay road or pass-side route where the traveler can move from long shot into a clearer human view."
    return "A frontier road or open ground where the traveler can visibly advance through space and reveal another layer of the world."


def _reflection_action(context: Dict[str, Any]) -> str:
    emotion = str(context.get("emotion") or "")
    if any(token in emotion for token in ("思乡", "惆怅", "悲", "愁")):
        return "The traveler pauses, turns back, lifts the head, or looks toward the far gate town, letting emotion surface before moving on."
    return "The traveler pauses at a ridge or pass, looks back once, then gathers resolve with a subtle but readable physical beat."


def _closing_scene(context: Dict[str, Any]) -> str:
    location = str(context.get("location") or "")
    if any(token in location for token in ("玉门关", "阳关", "嘉峪关")):
        return "A closing empty or near-empty frame around the pass, wall, gate, or road, with sky, dust, and distance taking over."
    return "A closing landscape frame with road, wind, far mountains, and fading human presence so the story resolves into space."


def _story_arc_text(context: Dict[str, Any]) -> str:
    places = context.get("places") or context.get("location") or "河西走廊边塞"
    imagery_text = _imagery_text(context.get("imagery") or [])
    emotion = context.get("emotion") or "苍茫、克制、边塞诗意"
    opening = _opening_scene(context)
    travel = _travel_scene(context)
    closing = _closing_scene(context)
    return (
        f"以首帧为开场，保留首帧中的主体、构图、色调和水墨淡彩笔触；"
        f"0-3 秒在 {opening} 中让风沙流动、云影移动、水面或沙洲纹理变化、杨柳摇曳，建立 {places} 的边塞氛围；"
        f"3-6 秒镜头自然推进到关城、城墙、古道、沙漠、远山或首帧已有景物的细节，空间层次明显变化；"
        f"6-10 秒让旅人远景、驼队远景、飞鸟、云影或诗中主体转身、伫立、回望或缓步前行，在 {travel} 中形成故事动作；"
        f"10-13 秒跟随旅人/驼队/风沙继续前进，画面向关隘、沙漠、远山、城墙或道路深处递进；"
        f"13-15 秒在 {closing} 中完成情绪收束。"
        f"全程围绕诗词意象 {imagery_text} 和情绪 {emotion} 展开，不跳到无关新场景。"
    )


def _poem_meta_line(context: Dict[str, Any]) -> str:
    parts = []
    if context["dynasty"]:
        parts.append(f"Dynasty: {context['dynasty']}.")
    return " ".join(parts)


def _subject_line(context: Dict[str, Any]) -> str:
    emotion = str(context.get("emotion") or "")
    imagery = " ".join(context.get("imagery") or [])
    location = str(context.get("location") or "")
    text = f"{emotion} {imagery} {location}"
    if any(token in text for token in ("边塞", "关", "塞", "大漠", "沙", "驿路", "古道")):
        return (
            "Treat the poetry landscape image as the narrative anchor, not as a single-person portrait. "
            "Animate frontier wind, sand, clouds, banners, roads, distant architecture, light, and camera movement; if a person is present, keep the motion natural and secondary."
        )
    if any(token in emotion for token in ("思乡", "惆怅", "悲", "愁")):
        return "Use the landscape, light, wind, water, roads, and distant silhouettes to express the emotion; do not require a detectable single human subject."
    return "Use the supplied mood image or landscape as the persistent visual world for the whole film, with visible environmental motion, spatial depth, and restrained cinematic camera movement."


def _prediction_text(prediction: Dict[str, Any]) -> str:
    return str(prediction.get("text", "")).strip()


def _prediction_field(prediction: Dict[str, Any], keys: Sequence[str]) -> str:
    if not isinstance(prediction, dict):
        return ""
    for key in keys:
        value = prediction.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            continue
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _prediction_label(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    return str(node.get("label") or node.get("id") or "").strip()


def _prediction_imagery(prediction: Dict[str, Any]) -> List[str]:
    node = prediction.get("imagery")
    if not isinstance(node, dict):
        return []
    labels = node.get("labels", [])
    return _clean_texts(
        str(item.get("label") or item.get("id") or "").strip()
        for item in labels
        if isinstance(item, dict)
    )


def _clean_texts(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _imagery_text(imagery: List[str]) -> str:
    if imagery:
        return ", ".join(_audio_safe_visual_token(item) for item in imagery[:6])
    return "frontier vastness, pass road, dunes, distant mountains, relay route"


def _audio_safe_visual_token(value: Any) -> str:
    text = str(value or "").strip()
    replacements = {
        "羌笛": "边关怀乡意象",
        "笛声": "边关怀乡意象",
        "风声": "边地苍茫气氛",
        "flute": "frontier homesickness",
        "wind noise": "frontier vastness",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def _narration_text(req: VeoGenerationRequest) -> str:
    text = _request_text(req, "recitation") or _request_text(req, "narration") or _request_text(req, "text")
    if not text:
        return ""
    return text if len(text) <= 280 else f"{text[:277]}..."
