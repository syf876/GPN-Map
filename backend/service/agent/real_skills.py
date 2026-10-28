from dataclasses import asdict, is_dataclass
from backend.admin_auth import require_generation_access
import os
from typing import Any, Dict, List

from backend.service.agent.curated_cases import find_curated_case
from backend.service.agent.schemas import AgentState, SkillResult
from backend.service.agent.skills import SkillDefinition, SkillRegistry
from backend.service.agent.vector_rag import VectorIndexUnavailable, search_poetry_vectors
from backend.types.schema import (
    MainStageComposeRequest,
    PoetryImageRequest,
    VeoGenerationRequest,
)


SUCCESS_STATUSES = {"succeeded", "success", "completed", "complete", "done", "finished", "ready"}


def build_real_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    for definition in real_skill_definitions():
        registry.register_definition(definition)
    return registry


def real_skill_definitions() -> List[SkillDefinition]:
    return [
        SkillDefinition(
            name="poetry_lookup",
            input_keys=["user_input"],
            output_keys=["poem_info"],
            description="Resolve full poem metadata and local poetry-library candidates.",
            handler=poetry_lookup,
        ),
        SkillDefinition(
            name="rag_retrieval",
            input_keys=["user_input", "poem_info"],
            output_keys=["rag_context"],
            description="Build local BGE-M3 vector retrieval context with keyword fallback.",
            handler=rag_retrieval,
        ),
        SkillDefinition(
            name="semantic_analysis",
            input_keys=["user_input", "poem_info", "rag_context"],
            output_keys=["semantic_result"],
            description="Run the existing round-2 semantic predictor and demo overrides.",
            handler=semantic_analysis,
        ),
        SkillDefinition(
            name="place_alignment",
            input_keys=["semantic_result"],
            output_keys=["place_alignment"],
            description="Align historical place labels to modern geographic entities before spatial mapping.",
            handler=place_alignment,
        ),
        SkillDefinition(
            name="spatial_mapping",
            input_keys=["semantic_result", "place_alignment"],
            output_keys=["geojson", "place_alignment"],
            description="Build existing GeoJSON map output from semantic prediction.",
            handler=spatial_mapping,
        ),
        SkillDefinition(
            name="narrative_generation",
            input_keys=["poem_info", "semantic_result", "geojson", "query_plan", "real_video_search"],
            output_keys=["story_context", "narrative_chain"],
            description="Build existing story context and narrative chain.",
            handler=narrative_generation,
        ),
        SkillDefinition(
            name="poetry_image",
            input_keys=["story_context", "semantic_result"],
            output_keys=["image"],
            description="Call the existing Nano Banana poetry image generator.",
            handler=poetry_image,
        ),
        SkillDefinition(
            name="poetry_image_retry",
            input_keys=["story_context", "semantic_result"],
            output_keys=["image"],
            description="Retry the existing Nano Banana image generator with the safe first-frame prompt mode.",
            handler=poetry_image_retry,
        ),
        SkillDefinition(
            name="poetry_video",
            input_keys=["story_context", "semantic_result", "image"],
            output_keys=["video"],
            description="Call the existing Wan poetry video generation dispatcher.",
            handler=poetry_video,
        ),
        SkillDefinition(
            name="real_video_search",
            input_keys=["semantic_result"],
            output_keys=["query_plan", "real_video_search"],
            description="Run the existing real-world video search aggregation.",
            handler=real_video_search,
        ),
    ]


def poetry_lookup(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    try:
        from backend.service.hexi_poetry import list_hexi_poems
        from backend.service.video_generation.poetry_audio import resolve_full_poem

        text = state.user_input
        poem_match = resolve_full_poem(text)
        candidates = list_hexi_poems(q=text, limit=int(params.get("limit", 5) or 5), offset=0)
        curated = find_curated_case(text, getattr(poem_match, "title", ""), getattr(poem_match, "poem_text", ""))
        poem_info = {
            "input_text": text,
            "resolved": _to_plain_data(poem_match),
            "title": curated.get("title") if curated else poem_match.title,
            "author": curated.get("author") if curated else poem_match.author,
            "dynasty": curated.get("dynasty") if curated else poem_match.dynasty,
            "full_poem": curated.get("content") if curated else (poem_match.poem_text or text),
            "matched": True if curated else poem_match.matched,
            "match_type": "curated_featured_case" if curated else poem_match.match_type,
            "source_type": curated.get("source_type", "curated_featured_case") if curated else getattr(poem_match, "source_type", ""),
            "included_in_experiment_dataset": False if curated else getattr(poem_match, "included_in_experiment_dataset", True),
            "candidates": candidates.get("items", []),
            "candidate_total": candidates.get("total", 0),
            "source": "resolve_full_poem+list_hexi_poems",
        }
        return SkillResult(output={"poem_info": poem_info}, message="poetry lookup completed")
    except Exception as exc:
        return _failure(exc, "poetry_lookup")


def rag_retrieval(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    top_k = int(params.get("top_k") or params.get("limit") or 5)
    recall_k = max(int(params.get("recall_k") or 20), top_k)
    query = _build_rag_query(state)
    try:
        recalled_results = search_poetry_vectors(query, top_k=recall_k)
        primary_poem = _primary_poem_context(state)
        related_knowledge = _rerank_related_knowledge(recalled_results, state, top_k=top_k)
        rag_context = [
            {
                "source": "backend.service.agent.vector_rag.BgeM3PoetryVectorIndex",
                "retrieval_type": "bge_m3_vector",
                "vector_retrieval": True,
                "embedding_model": os.getenv("BGE_M3_MODEL_PATH") or "BAAI/bge-m3",
                "fallback_reason": "",
                "query": query,
                "top_k": top_k,
                "recall_k": recall_k,
                "primary_poem": primary_poem,
                "primary_poem_context": primary_poem,
                "related_knowledge": related_knowledge,
                "results": related_knowledge,
                "enhanced_context": _format_vector_rag_context(related_knowledge, primary_poem=primary_poem),
            }
        ]
        return SkillResult(output={"rag_context": rag_context}, message="BGE-M3 vector RAG context retrieved")
    except VectorIndexUnavailable as vector_exc:
        fallback_reason = f"{vector_exc.__class__.__name__}: {vector_exc}"

    try:
        import poem_db

        place = _prediction_label(state.semantic_result.get("place")) if state.semantic_result else ""
        keyword_query = state.poem_info.get("title") or state.user_input
        context_text = poem_db.build_context_by_place(place or keyword_query, limit=top_k)
        prompt = poem_db.build_prompt_with_local_poems(state.user_input, place=place or None, limit=top_k)
        rag_context = [
            {
                "source": "poem_db.build_context_by_place",
                "primary_poem": _primary_poem_context(state),
                "retrieval_type": "keyword_local_temporary_fallback",
                "vector_retrieval": False,
                "fallback_reason": fallback_reason,
                "query": keyword_query,
                "place": place,
                "content": context_text,
            },
            {
                "source": "poem_db.build_prompt_with_local_poems",
                "retrieval_type": "keyword_local_temporary_fallback",
                "vector_retrieval": False,
                "fallback_reason": fallback_reason,
                "query": state.user_input,
                "content": prompt,
            },
        ]
        return SkillResult(output={"rag_context": rag_context}, message="RAG context retrieved")
    except Exception as exc:
        return _failure(exc, "rag_retrieval")


def semantic_analysis(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    try:
        curated = _cached_curated_case(state)
        if curated:
            place = str(curated.get("place") or (curated.get("places") or [""])[0] or "").strip()
            imagery = [
                {"label": str(item).strip(), "confidence": 0.9}
                for item in (curated.get("imagery") or [])
                if str(item).strip()
            ]
            prediction = {
                "text": state.user_input,
                "confidence": 0.95,
                "place": {"id": "PLC_YUMENGUAN" if place == "玉门关" else "PLC_JIUQUAN", "label": place, "confidence": 0.95},
                "imagery": {
                    "labels": [
                        {
                            "id": "IMG_FRONTIER" if index == 0 else "IMG_NATURE",
                            "label": item["label"],
                            "confidence": item["confidence"],
                        }
                        for index, item in enumerate(imagery)
                    ]
                },
                "emotion": {"id": "EMO_HOMESICK", "label": str(curated.get("emotion") or "").strip(), "confidence": 0.9},
                "dynasty": {"id": str(state.poem_info.get("dynasty") or curated.get("dynasty") or ""), "label": state.poem_info.get("dynasty") or curated.get("dynasty") or "", "confidence": 1.0},
                "temporal_spatial_context": {
                    "dynasty": state.poem_info.get("dynasty") or curated.get("dynasty") or "",
                    "place": place,
                    "source_type": "curated_featured_case",
                },
                "poem_info": {
                    "title": state.poem_info.get("title", ""),
                    "author": state.poem_info.get("author", ""),
                    "dynasty": state.poem_info.get("dynasty", ""),
                    "full_poem": state.poem_info.get("full_poem", state.user_input),
                },
                "rag_context_count": len(state.rag_context),
                "source_type": "curated_featured_case",
            }
            state.add_trace(
                "cached_result_used",
                step_id="semantic_analysis",
                skill_name="semantic_analysis",
                detail={"source": "curated_featured_case"},
            )
            state.add_trace(
                "cache_hit",
                step_id="semantic_analysis",
                skill_name="semantic_analysis",
                detail={"source": "curated_featured_case"},
            )
            return SkillResult(output={"semantic_result": prediction}, message="curated semantic analysis loaded")
        from backend.service.orchestration.main_stage_pipeline import _apply_demo_semantic_override, predict_poem

        threshold = float(params.get("imagery_threshold", 0.5))
        # The selected title may contain no place at all. Infer from the
        # resolved poem, rather than from its title alone.
        analysis_text = state.poem_info.get("full_poem") or state.user_input
        prediction = predict_poem(analysis_text, imagery_threshold=threshold)
        prediction = _apply_demo_semantic_override(state.user_input, prediction)
        prediction["poem_info"] = {
            "title": state.poem_info.get("title", ""),
            "author": state.poem_info.get("author", ""),
            "dynasty": state.poem_info.get("dynasty", ""),
            "full_poem": state.poem_info.get("full_poem", state.user_input),
        }
        prediction["rag_context_count"] = len(state.rag_context)
        prediction["temporal_spatial_context"] = {
            "dynasty": _prediction_label(prediction.get("dynasty")) or state.poem_info.get("dynasty", ""),
            "place": _prediction_label(prediction.get("place")),
        }
        return SkillResult(output={"semantic_result": prediction}, message="semantic analysis completed")
    except Exception as exc:
        return _failure(exc, "semantic_analysis")


def place_alignment(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    try:
        from backend.service.hexi_poetry import PLACE_GAZETTEER

        prediction = state.semantic_result
        place = prediction.get("place", {}) if isinstance(prediction, dict) else {}
        place_label = _prediction_label(place)
        gazetteer_match = _match_gazetteer_place(place_label, PLACE_GAZETTEER)
        modern_entity = gazetteer_match.get("name") or place_label
        aliases = list(gazetteer_match.get("aliases") or [])
        match_basis = "gazetteer_alias_exact" if gazetteer_match else "semantic_label_only"
        aligned = {
            "historical_place": place_label,
            "modern_geographic_entity": modern_entity,
            "entity_type": _infer_entity_type(modern_entity),
            "id": str(place.get("id") or "").strip() if isinstance(place, dict) else "",
            "label": modern_entity,
            "aliases": aliases,
            "confidence": float(place.get("confidence") or 0.0) if isinstance(place, dict) else 0.0,
            "coordinates": {
                "lon": float(gazetteer_match.get("lng") or 0.0) if gazetteer_match else 0.0,
                "lat": float(gazetteer_match.get("lat") or 0.0) if gazetteer_match else 0.0,
            },
            "lng": float(gazetteer_match.get("lng") or 0.0) if gazetteer_match else 0.0,
            "lat": float(gazetteer_match.get("lat") or 0.0) if gazetteer_match else 0.0,
            "match_basis": match_basis,
            "source": "backend.service.hexi_poetry.PLACE_GAZETTEER",
            "matched": bool(gazetteer_match),
        }
        return SkillResult(output={"place_alignment": aligned}, message="place alignment completed")
    except Exception as exc:
        return _failure(exc, "place_alignment")


def spatial_mapping(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    try:
        from backend.service.orchestration.main_stage_pipeline import build_map_prediction

        geojson = build_map_prediction(state.semantic_result)
        aligned = state.place_alignment
        coordinates = aligned.get("coordinates") or {}
        if aligned.get("matched") and coordinates:
            for feature in geojson.get("features", []):
                feature["geometry"] = {
                    "type": "Point",
                    "coordinates": [coordinates["lon"], coordinates["lat"]],
                }
                feature["properties"]["place"]["coordinates"] = dict(coordinates)
                feature["properties"]["place"]["modern_name"] = aligned.get("modern_geographic_entity")
            geojson.setdefault("metadata", {})["coordinate_source"] = aligned.get("source")
        output: Dict[str, Any] = {"geojson": geojson}
        features = geojson.get("features", []) if isinstance(geojson, dict) else []
        if features:
            place_payload = features[0].get("properties", {}).get("place", {})
            coordinates = place_payload.get("coordinates", {})
            if coordinates:
                output["place_alignment"] = {
                    **state.place_alignment,
                    "coordinates": coordinates,
                    "lng": float(coordinates.get("lon") or state.place_alignment.get("lng") or 0.0),
                    "lat": float(coordinates.get("lat") or state.place_alignment.get("lat") or 0.0),
                    "matched": True,
                }
        return SkillResult(output=output, message="spatial mapping completed")
    except Exception as exc:
        return _failure(exc, "spatial_mapping")


def real_video_search(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    try:
        from backend.service.video_search.reality import search_reality
        from backend.types.schema import VideoSearchRequest

        location = (state.place_alignment.get("modern_geographic_entity")
                    or _prediction_label(state.semantic_result.get("place")))
        response = search_reality(VideoSearchRequest(location=location, poem=state.user_input))
        return SkillResult(
            ok=response["ok"],
            output={"query_plan": {"provider": "local_rules", "query": response["query"]},
                    "real_video_search": response},
            message=response["message"],
            error="" if response["ok"] else response["message"],
        )
    except Exception as exc:
        return _failure(exc, "real_video_search")


def narrative_generation(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    try:
        curated = _cached_curated_case(state)
        if curated:
            place = state.place_alignment or {}
            story_payload = {
                "input_text": state.user_input,
                "full_poem": state.poem_info.get("full_poem") or curated.get("content") or state.user_input,
                "title": state.poem_info.get("title") or curated.get("title") or "",
                "author": state.poem_info.get("author") or curated.get("author") or "",
                "dynasty": state.poem_info.get("dynasty") or curated.get("dynasty") or "",
                "emotion": str(curated.get("emotion") or _prediction_label(state.semantic_result.get("emotion"))),
                "imagery": list(curated.get("imagery") or _prediction_imagery(state.semantic_result)),
                "primary_place": {
                    "id": str(state.semantic_result.get("place", {}).get("id") or ""),
                    "label": place.get("label") or place.get("modern_geographic_entity") or curated.get("place") or "",
                    "lng": float(place.get("lng") or (place.get("coordinates") or {}).get("lon") or 0.0),
                    "lat": float(place.get("lat") or (place.get("coordinates") or {}).get("lat") or 0.0),
                    "confidence": float(place.get("confidence") or 0.0),
                },
                "narrative_steps": [
                    {
                        "id": "poem_identification",
                        "title": "诗词识别",
                        "summary": state.poem_info.get("full_poem") or curated.get("content") or "",
                        "status": "ready",
                    },
                    {
                        "id": "spatial_locating",
                        "title": "空间定位",
                        "summary": str(curated.get("narrative_note") or ""),
                        "status": "ready",
                    },
                    {
                        "id": "multimodal_context",
                        "title": "多模态上下文",
                        "summary": str(curated.get("narrative_note") or ""),
                        "status": "ready",
                    },
                ],
                "agent_goal_constraints": _goal_constraints_payload(state),
                "source_type": "curated_featured_case",
            }
            query_plan_payload = {
                "provider": "curated-cache",
                "mode": "cached",
                "query": str(curated.get("narrative_note") or curated.get("place") or ""),
                "keywords": list(curated.get("places") or []),
                "scene_tags": list(curated.get("imagery") or []),
                "negative_terms": [],
                "platform_hints": [],
                "query_variants": [],
                "reasoning": "curated featured case narrative context",
            }
            state.add_trace(
                "cached_result_used",
                step_id="narrative_generation",
                skill_name="narrative_generation",
                detail={"source": "curated_featured_case"},
            )
            state.add_trace(
                "cache_hit",
                step_id="narrative_generation",
                skill_name="narrative_generation",
                detail={"source": "curated_featured_case"},
            )
            return SkillResult(
                output={
                    "query_plan": query_plan_payload,
                    "story_context": story_payload,
                    "narrative_chain": story_payload["narrative_steps"],
                },
                message="curated narrative context loaded",
            )
        from backend.service.orchestration.main_stage_pipeline import _build_story_context
        from backend.service.query_planner import build_query_plan
        from backend.types.schema import QueryPlan

        query_plan_payload = state.query_plan or _to_plain_data(build_query_plan(state.semantic_result))
        query_plan = QueryPlan(**query_plan_payload)
        reference_query = str(state.real_video_search.get("query") or query_plan.query or "").strip()
        req = MainStageComposeRequest(text=state.user_input, imagery_threshold=float(params.get("imagery_threshold", 0.5)))
        story_context = _build_story_context(
            req,
            state.semantic_result,
            state.geojson,
            query_plan,
            reference_query,
        )
        story_payload = _to_plain_data(story_context)
        story_payload["agent_goal_constraints"] = _goal_constraints_payload(state)
        story_payload["image_prompt_context"] = _structured_prompt_context(state, kind="image")
        story_payload["video_prompt_context"] = _structured_prompt_context(state, kind="video")
        return SkillResult(
            output={
                "query_plan": query_plan_payload,
                "story_context": story_payload,
                "narrative_chain": story_payload.get("narrative_steps", []),
            },
            message="narrative generation completed",
        )
    except Exception as exc:
        return _failure(exc, "narrative_generation")


def poetry_image(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    try:
        cached = _cached_curated_image(state)
        if cached:
            state.add_trace("cached_result_used", step_id="poetry_image", skill_name="poetry_image", detail={"source": "curated_featured_case"})
            state.add_trace("cache_hit", step_id="poetry_image", skill_name="poetry_image", detail={"source": "curated_featured_case"})
            return SkillResult(output={"image": cached}, message="cached poetry image loaded")
        if not state.generation_allowed:
            return SkillResult(ok=False, output={"image": {}}, error="image generation is restricted to cached featured cases or administrator-approved realtime generation")
        from backend.service.image_generation import generate_nano_banana_poetry_image

        require_generation_access()
        state.paid_generation_calls += 1
        req = _build_image_request(state, mode=str(params.get("mode") or ""))
        response = generate_nano_banana_poetry_image(req)
        payload = _to_plain_data(response)
        if not payload.get("image_url") or str(payload.get("status") or "").strip().lower() not in SUCCESS_STATUSES:
            return SkillResult(ok=False, output={"image": payload}, error=payload.get("error") or "image generation failed")
        return SkillResult(output={"image": payload}, message="poetry image generated")
    except Exception as exc:
        return _failure(exc, "poetry_image")


def poetry_image_retry(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    try:
        cached = _cached_curated_image(state)
        if cached:
            state.add_trace("cached_result_used", step_id="poetry_image", skill_name="poetry_image_retry", detail={"source": "curated_featured_case"})
            state.add_trace("cache_hit", step_id="poetry_image", skill_name="poetry_image_retry", detail={"source": "curated_featured_case"})
            return SkillResult(output={"image": cached}, message="cached poetry image loaded")
        if not state.generation_allowed:
            return SkillResult(ok=False, output={"image": {}}, error="image retry is restricted to cached featured cases or administrator-approved realtime generation")
        from backend.service.image_generation import generate_nano_banana_poetry_image

        require_generation_access()
        state.paid_generation_calls += 1
        req = _build_image_request(state, mode="safe_i2v_first_frame")
        response = generate_nano_banana_poetry_image(req)
        payload = _to_plain_data(response)
        if not payload.get("image_url") or str(payload.get("status") or "").strip().lower() not in SUCCESS_STATUSES:
            return SkillResult(ok=False, output={"image": payload}, error=payload.get("error") or "image retry failed")
        return SkillResult(output={"image": payload}, message="poetry image retry generated")
    except Exception as exc:
        return _failure(exc, "poetry_image_retry")


def poetry_video(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    try:
        cached = _cached_curated_video(state)
        if cached:
            state.add_trace("cached_result_used", step_id="poetry_video", skill_name="poetry_video", detail={"source": "curated_featured_case"})
            state.add_trace("cache_hit", step_id="poetry_video", skill_name="poetry_video", detail={"source": "curated_featured_case"})
            return SkillResult(output={"video": cached}, message="cached poetry video loaded")
        if not state.generation_allowed:
            return SkillResult(ok=False, output={"video": {}}, error="video generation is restricted to cached featured cases or administrator-approved realtime generation")
        from backend.service.video_generation import generate_poetry_video

        require_generation_access()
        state.paid_generation_calls += 1
        story = state.story_context or {}
        image_url = str(state.image.get("normalized_image_url") or state.image.get("image_url") or params.get("image_url") or "").strip()
        video_retry_count = int(state.retry_counts.get("poetry_video", 0) or 0)
        prediction_payload = dict(state.semantic_result or {})
        goal_constraints = _goal_constraints_payload(state)
        if goal_constraints:
            prediction_payload["agent_goal_constraints"] = goal_constraints
        for key in ("duration_seconds", "camera_requirements", "narrative_actions", "audio_requirements"):
            if params.get(key):
                prediction_payload[key] = params.get(key)
        motion_prompt_boost = str(params.get("motion_prompt_boost") or "").strip()
        if motion_prompt_boost:
            prediction_payload["motion_prompt_boost"] = motion_prompt_boost
            state.add_trace(
                "video_prompt_adjusted",
                step_id="poetry_video",
                skill_name="poetry_video",
                detail={"reason": "critic_repair", "motion_prompt_boost": motion_prompt_boost[:160]},
            )
        if video_retry_count > 0:
            prediction_payload["agent_video_retry"] = video_retry_count
            prediction_payload["motion_prompt_boost"] = (
                "strengthen visible subject motion, camera push-in, route movement, "
                "foreground-background parallax, and scene transition; avoid a static image with slight shake"
            )
            state.add_trace(
                "video_prompt_adjusted",
                step_id="poetry_video",
                skill_name="poetry_video",
                detail={"retry_count": video_retry_count, "reason": "video validation requested stronger motion"},
            )
        req = VeoGenerationRequest(
            text=str(story.get("full_poem") or state.poem_info.get("full_poem") or state.user_input),
            location=str((story.get("primary_place") or {}).get("label") or state.place_alignment.get("label") or ""),
            dynasty=str(story.get("dynasty") or state.poem_info.get("dynasty") or ""),
            emotion=str(story.get("emotion") or _prediction_label(state.semantic_result.get("emotion"))),
            imagery=list(story.get("imagery") or _prediction_imagery(state.semantic_result)),
            image_url=image_url,
            narration=str(story.get("full_poem") or state.user_input),
            resolution=str(params.get("resolution") or ""),
            prediction=prediction_payload,
        )
        response = generate_poetry_video(req)
        payload = _to_plain_data(response)
        if not (payload.get("normalized_video_url") or payload.get("video_url")) or str(payload.get("status") or "").strip().lower() not in SUCCESS_STATUSES:
            return SkillResult(ok=False, output={"video": payload}, error=payload.get("error") or "video generation failed")
        return SkillResult(output={"video": payload}, message="poetry video generated")
    except Exception as exc:
        return _failure(exc, "poetry_video")


def _build_image_request(state: AgentState, *, mode: str = "") -> PoetryImageRequest:
    story = state.story_context or {}
    primary_place = story.get("primary_place") or {}
    constraints = _goal_constraints_payload(state)
    structured_terms = _structured_visual_terms(constraints)
    base_imagery = list(story.get("imagery") or _prediction_imagery(state.semantic_result))
    imagery = _unique_texts([*base_imagery, *structured_terms])
    mode_parts = [mode] if mode else []
    if constraints.get("visual_style"):
        mode_parts.append("style:" + "|".join(str(item) for item in constraints.get("visual_style") or []))
    if constraints.get("camera_requirements"):
        mode_parts.append("camera:" + "|".join(str(item) for item in constraints.get("camera_requirements") or []))
    return PoetryImageRequest(
        title=str(story.get("title") or state.poem_info.get("title") or ""),
        author=str(story.get("author") or state.poem_info.get("author") or ""),
        poem=str(story.get("full_poem") or state.poem_info.get("full_poem") or state.user_input),
        place=str(primary_place.get("label") or state.place_alignment.get("label") or _prediction_label(state.semantic_result.get("place"))),
        dynasty=str(story.get("dynasty") or state.poem_info.get("dynasty") or _prediction_label(state.semantic_result.get("dynasty"))),
        emotion=str(story.get("emotion") or _prediction_label(state.semantic_result.get("emotion"))),
        imagery=imagery,
        mode=";".join(part for part in mode_parts if part),
    )


def _cached_curated_case(state: AgentState) -> Dict[str, Any]:
    poem = state.poem_info or {}
    return find_curated_case(
        poem.get("title"),
        poem.get("author"),
        poem.get("full_poem"),
        state.user_input,
        state.poetry_id,
    ) or {}


def _cached_curated_image(state: AgentState) -> Dict[str, Any]:
    if state.generation_allowed and not state.goal_constraints.reuse_existing:
        return {}
    case = _cached_curated_case(state)
    image_url = str(case.get("cached_image_url") or "").strip()
    if not image_url:
        return {}
    return {
        "image_url": image_url,
        "normalized_image_url": image_url,
        "prompt": str(case.get("narrative_note") or ""),
        "agent_goal_constraints": _goal_constraints_payload(state),
        "provider": "curated-cache",
        "model": "featured-case-cache",
        "status": "succeeded",
        "source_type": "curated_featured_case",
        "included_in_experiment_dataset": False,
        "case_id": str(case.get("id") or ""),
    }


def _cached_curated_video(state: AgentState) -> Dict[str, Any]:
    if state.generation_allowed and not state.goal_constraints.reuse_existing:
        return {}
    case = _cached_curated_case(state)
    video_url = str(case.get("cached_video_url") or "").strip()
    if not video_url:
        return {}
    image_url = str(case.get("cached_image_url") or "").strip()
    if image_url and not state.image:
        state.image = _cached_curated_image(state)
    return {
        "status": "ready",
        "provider": "curated-cache",
        "id": f"curated-{case.get('id') or 'featured-case'}",
        "video_url": video_url,
        "normalized_video_url": video_url,
        "progress": 100.0,
        "prompt": str(case.get("narrative_note") or ""),
        "agent_goal_constraints": _goal_constraints_payload(state),
        "error": "",
        "source_type": "curated_featured_case",
        "included_in_experiment_dataset": False,
        "case_id": str(case.get("id") or ""),
    }


def _build_rag_query(state: AgentState) -> str:
    poem = state.poem_info or {}
    parts = [
        state.user_input,
        poem.get("title"),
        poem.get("author"),
        poem.get("dynasty"),
        poem.get("full_poem"),
    ]
    return "\n".join(str(item).strip() for item in parts if str(item or "").strip())


def _primary_poem_context(state: AgentState) -> Dict[str, Any]:
    poem = state.poem_info or {}
    content = str(poem.get("full_poem") or "").strip()
    matched = bool(poem.get("matched"))
    has_metadata = any(str(poem.get(key) or "").strip() for key in ("title", "author", "dynasty"))
    if not content:
        return {}
    return {
        "matched": matched,
        "match_type": str(poem.get("match_type") or "").strip(),
        "title": str(poem.get("title") or "").strip(),
        "author": str(poem.get("author") or "").strip(),
        "dynasty": str(poem.get("dynasty") or "").strip(),
        "content": content,
        "source": str(poem.get("source") or "").strip(),
        "source_type": str(poem.get("source_type") or "").strip(),
        "included_in_experiment_dataset": bool(poem.get("included_in_experiment_dataset", True)),
        "is_complete_poem": bool(matched and has_metadata),
        "missing_reason": "" if matched and has_metadata else "poetry_lookup did not resolve a complete poem from the 1230-record corpus",
    }


def _rerank_related_knowledge(results: List[Dict[str, Any]], state: AgentState, *, top_k: int) -> List[Dict[str, Any]]:
    signals = _query_rerank_signals(state)
    reranked = []
    for item in results:
        payload = dict(item)
        vector_similarity = float(payload.get("similarity_score") or payload.get("vector_similarity") or 0.0)
        bonus, reasons = _rerank_bonus_and_reasons(payload, signals)
        payload["vector_similarity"] = vector_similarity
        payload["similarity_score"] = vector_similarity
        payload["final_score"] = round(vector_similarity + bonus, 6)
        payload["match_reasons"] = reasons or ["vector_similarity"]
        reranked.append(payload)
    reranked.sort(key=lambda value: (float(value.get("final_score") or 0.0), float(value.get("vector_similarity") or 0.0)), reverse=True)
    return reranked[: max(1, int(top_k))]


def _query_rerank_signals(state: AgentState) -> Dict[str, Any]:
    from backend.service.hexi_poetry import PLACE_GAZETTEER, THEME_RULES

    poem = state.poem_info or {}
    query_text = " ".join(
        str(value or "").strip()
        for value in [state.user_input, poem.get("title"), poem.get("author"), poem.get("dynasty"), poem.get("full_poem")]
        if str(value or "").strip()
    )
    normalized_query = _normalize_match_text(query_text)
    historical_places: List[str] = []
    modern_places: List[str] = []
    for entry in PLACE_GAZETTEER:
        name = str(entry.get("name") or "").strip()
        aliases = [str(alias or "").strip() for alias in entry.get("aliases", ()) if str(alias or "").strip()]
        if any(alias and alias in query_text for alias in aliases + [name]):
            for alias in aliases:
                _append_unique(historical_places, alias)
            _append_unique(modern_places, name)
    imagery = _infer_query_imagery(query_text)
    theme = _match_rule_label(query_text, THEME_RULES)
    phrases = [
        phrase
        for phrase in _split_query_phrases(query_text)
        if len(_normalize_match_text(phrase)) >= 6
    ]
    terms = [term for term in ("玉门关", "玉门", "羌笛", "杨柳", "春风", "凉州", "边塞", "塞", "关") if term in query_text]
    return {
        "query_text": query_text,
        "normalized_query": normalized_query,
        "phrases": phrases,
        "terms": terms,
        "historical_places": historical_places,
        "modern_places": modern_places,
        "imagery": imagery,
        "theme": theme,
    }


def _rerank_bonus_and_reasons(item: Dict[str, Any], signals: Dict[str, Any]) -> tuple[float, List[str]]:
    bonus = 0.0
    reasons: List[str] = []
    content = str(item.get("content") or "")
    title = str(item.get("title") or "")
    record_text = " ".join(
        [
            title,
            content,
            " ".join(str(value) for value in item.get("historical_places", []) or []),
            " ".join(str(value) for value in item.get("modern_places", []) or []),
            " ".join(str(value) for value in item.get("imagery", []) or []),
            str(item.get("theme") or ""),
            str(item.get("cultural_context") or ""),
        ]
    )
    normalized_record = _normalize_match_text(record_text)
    normalized_query = str(signals.get("normalized_query") or "")
    if normalized_query and normalized_query in normalized_record:
        bonus += 0.45
        reasons.append("完整输入文本匹配")
    else:
        phrase_hits = [
            phrase
            for phrase in signals.get("phrases", [])
            if _normalize_match_text(phrase) and _normalize_match_text(phrase) in normalized_record
        ]
        if phrase_hits:
            bonus += 0.32
            reasons.append("连续诗句片段匹配：" + "、".join(phrase_hits[:2]))

    historical_overlap = _overlap(item.get("historical_places", []), signals.get("historical_places", []))
    if historical_overlap:
        bonus += 0.28
        reasons.append("历史地名一致：" + "、".join(historical_overlap))

    modern_overlap = _overlap(item.get("modern_places", []), signals.get("modern_places", []))
    if modern_overlap:
        bonus += 0.16
        reasons.append("现代地理实体一致：" + "、".join(modern_overlap))

    imagery_overlap = _overlap(item.get("imagery", []), signals.get("imagery", []))
    if imagery_overlap:
        bonus += min(0.18, 0.06 * len(imagery_overlap))
        reasons.append("意象一致：" + "、".join(imagery_overlap))

    theme = str(item.get("theme") or "").strip()
    if theme and theme == str(signals.get("theme") or "").strip():
        bonus += 0.08
        reasons.append("主题一致：" + theme)

    term_hits = [term for term in signals.get("terms", []) if term and term in record_text]
    if term_hits:
        bonus += min(0.12, 0.04 * len(term_hits))
        reasons.append("关键语义词命中：" + "、".join(term_hits[:4]))
    return bonus, reasons


def _infer_query_imagery(text: str) -> List[str]:
    result: List[str] = []
    if any(token in text for token in ("边塞", "塞", "关", "羌笛", "戍", "征", "战", "沙场", "胡")):
        result.append("边塞军旅")
    if any(token in text for token in ("玉门", "阳关", "关", "杨柳", "折柳", "离", "别")):
        result.append("关塞离别")
    if any(token in text for token in ("乡", "怨", "愁", "望", "归", "客")):
        result.append("思乡怀人")
    if any(token in text for token in ("沙", "漠", "大漠", "黄河", "黄尘")):
        result.append("大漠风光")
    if any(token in text for token in ("春", "风", "柳", "杨柳", "云", "山", "水", "月", "雪")):
        result.append("自然意象")
    return result


def _split_query_phrases(text: str) -> List[str]:
    normalized = str(text or "").replace("\n", " ")
    for mark in "，。！？；、,.!?;":
        normalized = normalized.replace(mark, "|")
    return [item.strip() for item in normalized.split("|") if item.strip()]


def _normalize_match_text(value: Any) -> str:
    return "".join(str(value or "").split())


def _overlap(left: Any, right: Any) -> List[str]:
    left_values = [str(item or "").strip() for item in (left or []) if str(item or "").strip()]
    right_values = [str(item or "").strip() for item in (right or []) if str(item or "").strip()]
    result: List[str] = []
    for item in left_values:
        if any(item == other or item in other or other in item for other in right_values):
            _append_unique(result, item)
    return result


def _append_unique(items: List[str], value: str) -> None:
    text = str(value or "").strip()
    if text and text not in items:
        items.append(text)


def _match_rule_label(text: str, rules: Any) -> str:
    for label, keywords in rules:
        if any(keyword and keyword in text for keyword in keywords):
            return label
    return ""


def _format_vector_rag_context(results: List[Dict[str, Any]], *, primary_poem: Dict[str, Any] | None = None) -> str:
    lines: List[str] = []
    if primary_poem:
        status = "matched" if primary_poem.get("is_complete_poem") else "unresolved"
        lines.append(
            "\n".join(
                part
                for part in [
                    f"[primary_poem:{status}] {primary_poem.get('title', '')} / {primary_poem.get('author', '')} / {primary_poem.get('dynasty', '')}",
                    f"正文：{primary_poem.get('content', '')}",
                    f"说明：{primary_poem.get('missing_reason', '')}" if primary_poem.get("missing_reason") else "",
                ]
                if part
            )
        )
    for index, item in enumerate(results, start=1):
        places = "、".join(str(value) for value in item.get("historical_places", []) if str(value).strip())
        imagery = "、".join(str(value) for value in item.get("imagery", []) if str(value).strip())
        vector_score = item.get("vector_similarity", item.get("similarity_score", 0.0))
        final_score = item.get("final_score", vector_score)
        reasons = "、".join(str(value) for value in item.get("match_reasons", []) if str(value).strip())
        lines.append(
            "\n".join(
                part
                for part in [
                    f"[related_knowledge:{index}] {item.get('title', '')} / {item.get('author', '')} / {item.get('dynasty', '')} / final={float(final_score):.4f} / vector={float(vector_score):.4f}",
                    f"正文：{item.get('content', '')}",
                    f"历史地名：{places}" if places else "",
                    f"意象：{imagery}" if imagery else "",
                    f"情感：{item.get('emotion', '')}" if item.get("emotion") else "",
                    f"文化背景：{item.get('cultural_context', '')}" if item.get("cultural_context") else "",
                    f"匹配原因：{reasons}" if reasons else "",
                ]
                if part
            )
        )
    return "\n\n".join(lines)


def _prediction_label(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("label") or node.get("id") or "").strip()
    return str(node or "").strip()


def _prediction_imagery(prediction: Dict[str, Any]) -> List[str]:
    node = prediction.get("imagery") if isinstance(prediction, dict) else {}
    labels = node.get("labels", []) if isinstance(node, dict) else []
    result = []
    for item in labels:
        label = _prediction_label(item)
        if label:
            result.append(label)
    return result


def _goal_constraints_payload(state: AgentState) -> Dict[str, Any]:
    constraints = getattr(state, "goal_constraints", None)
    if constraints is None:
        return {}
    if hasattr(constraints, "model_dump"):
        return constraints.model_dump()
    if hasattr(constraints, "dict"):
        return constraints.dict()
    return dict(constraints or {})


def _structured_visual_terms(constraints: Dict[str, Any]) -> List[str]:
    return _unique_texts(
        [
            *(constraints.get("visual_style") or []),
            *(constraints.get("narrative_actions") or []),
            *(constraints.get("camera_requirements") or []),
        ]
    )


def _structured_prompt_context(state: AgentState, *, kind: str) -> str:
    constraints = _goal_constraints_payload(state)
    parts: List[str] = []
    if constraints.get("visual_style"):
        parts.append("style=" + " / ".join(str(item) for item in constraints.get("visual_style") or []))
    if constraints.get("narrative_actions"):
        parts.append("actions=" + " / ".join(str(item) for item in constraints.get("narrative_actions") or []))
    if kind == "video" and constraints.get("camera_requirements"):
        parts.append("camera=" + " / ".join(str(item) for item in constraints.get("camera_requirements") or []))
    if kind == "video" and constraints.get("audio_requirements"):
        parts.append("audio=" + " / ".join(str(item) for item in constraints.get("audio_requirements") or []))
    if kind == "video" and int(constraints.get("duration_seconds") or 0):
        parts.append(f"duration={int(constraints.get('duration_seconds') or 0)}s")
    return "; ".join(parts)


def _unique_texts(values: List[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _match_gazetteer_place(place_label: str, gazetteer: List[Dict[str, Any]]) -> Dict[str, Any]:
    label = str(place_label or "").strip()
    if not label:
        return {}
    for item in gazetteer:
        aliases = [str(alias or "").strip() for alias in item.get("aliases", ()) if str(alias or "").strip()]
        name = str(item.get("name") or "").strip()
        if label == name or label in aliases:
            return item
    for item in gazetteer:
        aliases = [str(alias or "").strip() for alias in item.get("aliases", ()) if str(alias or "").strip()]
        name = str(item.get("name") or "").strip()
        if label and (label in name or any(label in alias or alias in label for alias in aliases)):
            return item
    return {}


def _infer_entity_type(name: str) -> str:
    text = str(name or "").strip()
    if any(token in text for token in ("关", "关城")):
        return "关隘/遗址"
    if any(token in text for token in ("山", "漠", "海", "沙", "泉")):
        return "自然地理实体"
    if any(token in text for token in ("走廊", "河西")):
        return "区域"
    return "现代城镇/地理实体"


def _to_plain_data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    return value


def _failure(exc: Exception, module_name: str) -> SkillResult:
    return SkillResult(ok=False, error=f"{module_name}: {exc.__class__.__name__}: {exc}")
