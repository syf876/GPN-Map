from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from backend.service.agent.schemas import AgentState, SkillResult


SkillCallable = Callable[[AgentState, Dict[str, Any]], SkillResult]


class SkillDefinition(BaseModel):
    name: str
    input_keys: List[str] = Field(default_factory=list)
    output_keys: List[str] = Field(default_factory=list)
    description: str = ""
    handler: SkillCallable

    class Config:
        arbitrary_types_allowed = True


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, SkillDefinition] = {}

    def register(
        self,
        name: str,
        handler: SkillCallable,
        *,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        normalized = str(name or "").strip()
        if not normalized:
            raise ValueError("skill name is required")
        if not callable(handler):
            raise TypeError(f"skill handler is not callable: {normalized}")
        self._skills[normalized] = SkillDefinition(
            name=normalized,
            input_keys=list(input_keys or []),
            output_keys=list(output_keys or []),
            description=str(description or "").strip(),
            handler=handler,
        )

    def register_definition(self, definition: SkillDefinition) -> None:
        self.register(
            definition.name,
            definition.handler,
            input_keys=definition.input_keys,
            output_keys=definition.output_keys,
            description=definition.description,
        )

    def has(self, name: str) -> bool:
        return str(name or "").strip() in self._skills

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._skills))

    def definition(self, name: str) -> SkillDefinition:
        normalized = str(name or "").strip()
        definition = self._skills.get(normalized)
        if definition is None:
            raise PermissionError(f"unregistered skill is not allowed: {normalized}")
        return definition

    def call(self, name: str, state: AgentState, params: Dict[str, Any] | None = None) -> SkillResult:
        definition = self.definition(name)
        missing_inputs = [key for key in definition.input_keys if not hasattr(state, key)]
        if missing_inputs:
            return SkillResult(
                ok=False,
                error=f"skill input contract failed for {definition.name}: missing state fields {missing_inputs}",
            )
        result = definition.handler(state, dict(params or {}))
        if isinstance(result, SkillResult):
            normalized_result = result
        elif isinstance(result, dict):
            normalized_result = SkillResult(**result)
        else:
            raise TypeError(f"skill returned unsupported result: {definition.name}")
        if normalized_result.ok:
            missing_outputs = [key for key in definition.output_keys if key not in normalized_result.output]
            if missing_outputs:
                return SkillResult(
                    ok=False,
                    output=normalized_result.output,
                    message=normalized_result.message,
                    error=f"skill output contract failed for {definition.name}: missing output keys {missing_outputs}",
                )
        return normalized_result


def build_mock_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register("poetry_lookup", mock_rag_context)
    registry.register("rag_retrieval", mock_rag_context)
    registry.register("semantic_analysis", _contract_mock_semantic_analysis)
    registry.register("place_alignment", _contract_mock_place_alignment)
    registry.register("spatial_mapping", mock_geojson_builder)
    registry.register("narrative_generation", mock_narrative_chain)
    registry.register("poetry_image", _contract_mock_image_module)
    registry.register("poetry_image_retry", _contract_mock_image_module)
    registry.register("poetry_video", _contract_mock_video_module)
    registry.register("real_video_search", mock_real_video_search)
    registry.register("mock_rag_context", mock_rag_context)
    registry.register("mock_semantic_analysis", _contract_mock_semantic_analysis)
    registry.register("mock_semantic_analysis_fallback", _contract_mock_semantic_analysis_fallback)
    registry.register("mock_place_alignment", _contract_mock_place_alignment)
    registry.register("mock_geojson_builder", mock_geojson_builder)
    registry.register("mock_geojson_builder_fallback", mock_geojson_builder_fallback)
    registry.register("mock_narrative_chain", mock_narrative_chain)
    registry.register("mock_image_module", _contract_mock_image_module)
    registry.register("mock_video_module", _contract_mock_video_module)
    return registry


def _contract_mock_rag_context(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    text = state.user_input or params.get("text") or ""
    title = "未命名诗句"
    if "凉州词" in text:
        title = "凉州词"
    elif "酒泉子" in text:
        title = "酒泉子"
    return SkillResult(
        output={
            "poem_info": {
                "title": title,
                "raw_text": text,
                "source": "mock_local_poetry_index",
            },
            "rag_context": [
                {
                    "id": "mock_context_1",
                    "summary": "河西走廊诗词地理叙事的本地占位上下文。",
                    "source": "mock_memory",
                }
            ],
        },
        message="mock RAG context prepared",
    )


# Contract-complete mock overrides used by the agent tests. They are kept at the
# end of the module so build_mock_skill_registry resolves these definitions.
def mock_rag_context(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    text = state.user_input or params.get("text") or "mock poem"
    title = "Mock Poem"
    return SkillResult(
        output={
            "poem_info": {
                "title": title,
                "author": "Mock Author",
                "dynasty": "Mock Dynasty",
                "full_poem": text,
                "matched": True,
                "source": "mock_local_poetry_index",
            },
            "rag_context": [
                {
                    "id": "mock_context_1",
                    "source": "mock_memory",
                    "retrieval_type": "mock_vector",
                    "vector_retrieval": True,
                    "fallback_reason": "",
                    "primary_poem": {
                        "matched": True,
                        "title": title,
                        "author": "Mock Author",
                        "dynasty": "Mock Dynasty",
                        "content": text,
                    },
                    "related_knowledge": [
                        {
                            "record_id": "mock_context_1",
                            "title": title,
                            "author": "Mock Author",
                            "dynasty": "Mock Dynasty",
                            "content": text,
                            "historical_places": ["Yumen Pass"],
                            "modern_places": ["Dunhuang"],
                            "imagery": ["frontier"],
                            "emotion": "solitude",
                            "cultural_context": "mock Hexi Corridor context",
                            "similarity_score": 0.9,
                            "vector_similarity": 0.9,
                            "final_score": 0.9,
                            "match_reasons": ["mock_vector_similarity"],
                        }
                    ],
                    "results": [],
                }
            ],
        },
        message="mock RAG context prepared",
    )


def _contract_mock_semantic_analysis(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    place = "Yumen Pass" if "玉门" in state.user_input or "Yumen" in state.user_input else "Hexi Corridor"
    return SkillResult(
        output={
            "semantic_result": {
                "place": {"label": place, "confidence": 0.91},
                "imagery": {"labels": [{"label": "frontier", "confidence": 0.86}]},
                "emotion": {"label": "solitude", "confidence": 0.78},
                "temporal_spatial_context": {"place": place, "dynasty": "Tang"},
                "mode": "mock",
            }
        },
        message="mock semantic result prepared",
    )


def _contract_mock_semantic_analysis_fallback(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    return SkillResult(
        output={
            "semantic_result": {
                "place": {"label": "Hexi Corridor", "confidence": 0.5},
                "imagery": {"labels": [{"label": "frontier", "confidence": 0.5}]},
                "emotion": {"label": "unknown", "confidence": 0.5},
                "temporal_spatial_context": {"place": "Hexi Corridor", "dynasty": "unknown"},
                "mode": "mock_fallback",
            }
        },
        message="fallback semantic result prepared",
    )


def _contract_mock_place_alignment(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    place = state.semantic_result.get("place", {}).get("label") or "Hexi Corridor"
    coord = {"lng": 93.8631, "lat": 40.3603} if place == "Yumen Pass" else {"lng": 98.6, "lat": 39.6}
    return SkillResult(
        output={
            "place_alignment": {
                "historical_place": place,
                "modern_geographic_entity": place,
                "entity_type": "mock_geographic_entity",
                "coordinates": {"lon": coord["lng"], "lat": coord["lat"]},
                "label": place,
                "lng": coord["lng"],
                "lat": coord["lat"],
                "match_basis": "mock_semantic_alignment",
                "confidence": state.semantic_result.get("place", {}).get("confidence", 0.5),
                "source": "mock_place_index",
            }
        },
        message="mock place alignment prepared",
    )


def _contract_mock_image_module(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    place = state.place_alignment.get("label") or "Hexi Corridor"
    return SkillResult(
        output={
            "image": {
                "status": "mock_ready",
                "image_url": "/static/demo/liangzhouci/image.jpg",
                "normalized_image_url": "/static/demo/liangzhouci/image.jpg",
                "prompt": f"{place} frontier poetry landscape",
                "provider": "mock",
            }
        },
        message="mock image module prepared",
    )


def _contract_mock_video_module(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    place = state.place_alignment.get("label") or "Hexi Corridor"
    return SkillResult(
        output={
            "video": {
                "status": "mock_ready",
                "video_url": "/static/demo/liangzhouci/video.mp4",
                "normalized_video_url": "/static/demo/liangzhouci/video.mp4",
                "prompt": f"{place} moving poetry shot",
                "provider": "mock",
                "depends_on_image": bool(state.image),
            }
        },
        message="mock video module prepared",
    )


def mock_semantic_analysis(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    text = state.user_input
    place = "河西走廊"
    if "玉门" in text:
        place = "玉门关"
    elif "酒泉" in text:
        place = "酒泉"
    elif "凉州" in text:
        place = "武威"
    return SkillResult(
        output={
            "semantic_result": {
                "place": {"label": place, "confidence": 0.91},
                "imagery": [{"label": "边塞", "confidence": 0.86}],
                "emotion": {"label": "苍凉", "confidence": 0.78},
                "mode": "mock",
            }
        },
        message="mock semantic result prepared",
    )


def mock_semantic_analysis_fallback(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    return SkillResult(
        output={
            "semantic_result": {
                "place": {"label": "河西走廊", "confidence": 0.5},
                "imagery": [{"label": "边塞", "confidence": 0.5}],
                "emotion": {"label": "未定", "confidence": 0.5},
                "mode": "mock_fallback",
            }
        },
        message="fallback semantic result prepared",
    )


def mock_place_alignment(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    place = state.semantic_result.get("place", {}).get("label") or "河西走廊"
    coordinates = {
        "玉门关": {"lng": 93.8631, "lat": 40.3603},
        "酒泉": {"lng": 98.5174, "lat": 39.7432},
        "武威": {"lng": 102.6347, "lat": 37.9282},
        "河西走廊": {"lng": 98.6, "lat": 39.6},
    }
    coord = coordinates.get(place, coordinates["河西走廊"])
    return SkillResult(
        output={
            "place_alignment": {
                "label": place,
                "lng": coord["lng"],
                "lat": coord["lat"],
                "source": "mock_place_index",
                "confidence": state.semantic_result.get("place", {}).get("confidence", 0.5),
            }
        },
        message="mock place alignment prepared",
    )


def mock_geojson_builder(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    place = state.place_alignment or {"label": "河西走廊", "lng": 98.6, "lat": 39.6}
    return SkillResult(
        output={
            "geojson": {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "name": place.get("label", "河西走廊"),
                            "source": "mock_agent_geojson",
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [place.get("lng", 98.6), place.get("lat", 39.6)],
                        },
                    }
                ],
            }
        },
        message="mock GeoJSON prepared",
    )


def mock_geojson_builder_fallback(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    return SkillResult(
        output={
            "geojson": {
                "type": "FeatureCollection",
                "features": [],
                "metadata": {"fallback": True, "reason": "mock empty map fallback"},
            }
        },
        message="fallback GeoJSON prepared",
    )


def mock_narrative_chain(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    place = state.place_alignment.get("label") or "河西走廊"
    return SkillResult(
        output={
            "story_context": {
                "full_poem": state.poem_info.get("full_poem") or state.user_input,
                "dynasty": state.poem_info.get("dynasty", ""),
                "primary_place": {
                    "label": place,
                    "lng": state.place_alignment.get("lng", 0.0),
                    "lat": state.place_alignment.get("lat", 0.0),
                },
                "imagery": ["frontier"],
                "emotion": "mock",
                "narrative_steps": [
                    {
                        "id": "spatial_locating",
                        "title": "spatial locating",
                        "summary": f"Mock narrative space for {place}.",
                        "status": "mock_ready",
                    }
                ],
            },
            "narrative_chain": [
                {
                    "id": "spatial_locating",
                    "title": "空间定位",
                    "summary": f"将诗句定位到{place}相关叙事空间。",
                    "status": "mock_ready",
                },
                {
                    "id": "imagery_reasoning",
                    "title": "意象推理",
                    "summary": "根据边塞意象组织地图和多模态生成上下文。",
                    "status": "mock_ready",
                },
            ]
        },
        message="mock narrative chain prepared",
    )


def mock_real_video_search(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    return SkillResult(
        output={
            "query_plan": {
                "provider": "mock",
                "mode": "mock",
                "query": f"{state.place_alignment.get('label') or 'Hexi Corridor'} video reference",
                "keywords": ["Hexi Corridor"],
                "scene_tags": ["landscape"],
                "negative_terms": [],
                "platform_hints": ["bilibili"],
                "query_variants": [],
                "reasoning": "mock query planning",
            },
            "real_video_search": {
                "ok": True,
                "query": "mock video query",
                "best": None,
                "candidates": [],
                "best_reason": "mock",
                "priority_hit": "",
                "candidate_source_distribution": {},
                "candidate_count": 0,
                "selected_index": 0,
            },
        },
        message="mock real video search prepared",
    )


def mock_image_module(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    place = state.place_alignment.get("label") or "河西走廊"
    return SkillResult(
        output={
            "image": {
                "status": "mock_ready",
                "prompt": f"{place} 边塞诗意宽屏画面",
                "provider": "mock",
            }
        },
        message="mock image module prepared",
    )


def mock_video_module(state: AgentState, params: Dict[str, Any]) -> SkillResult:
    place = state.place_alignment.get("label") or "河西走廊"
    return SkillResult(
        output={
            "video": {
                "status": "mock_ready",
                "prompt": f"{place} 古道云影叙事镜头",
                "provider": "mock",
                "depends_on_image": bool(state.image),
            }
        },
        message="mock video module prepared",
    )
