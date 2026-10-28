from typing import Any, Dict, List, Optional

from backend.service.agent.curated_cases import find_curated_case
from backend.service.agent.schemas import AgentPlan, AgentState, AgentTarget, ExperimentMode, PlanStep, StepValidation


class Planner:
    def build_plan(
        self,
        state_or_target: AgentState | AgentTarget,
        experiment_mode: ExperimentMode = "static_agent",
        critic_feedback: Optional[StepValidation | Dict[str, Any]] = None,
    ) -> AgentPlan:
        if isinstance(state_or_target, str):
            return self._static_plan(state_or_target, experiment_mode="static_agent")

        state = state_or_target
        mode = experiment_mode or state.experiment_mode or "static_agent"
        if mode in {"fixed", "static_agent"}:
            return self._static_plan(state.target, experiment_mode=mode)
        return self._dynamic_plan(state, critic_feedback=critic_feedback)

    def _static_plan(self, target: AgentTarget, *, experiment_mode: ExperimentMode) -> AgentPlan:
        builders = {
            "map": self._map_steps,
            "image": self._image_steps,
            "video": self._video_steps,
        }
        return AgentPlan(target=target, steps=builders[target](), experiment_mode=experiment_mode, reason="static_target_plan")

    def _dynamic_plan(
        self,
        state: AgentState,
        *,
        critic_feedback: Optional[StepValidation | Dict[str, Any]] = None,
    ) -> AgentPlan:
        feedback = _feedback_dict(critic_feedback)
        recommendation = feedback.get("recommendation") if isinstance(feedback.get("recommendation"), dict) else {}
        reuse_existing = _reuse_existing_allowed(state)
        if recommendation.get("action") == "stop_non_retryable":
            return AgentPlan(
                target=state.target,
                experiment_mode="dynamic_agent",
                reason="critic_stop_non_retryable",
                steps=[
                    PlanStep(
                        step_id="agent_stop",
                        skill_name="agent_stop",
                        action="stop",
                        description="Stop because Critic marked the failure as non-retryable.",
                        max_retries=0,
                        params={
                            "reason": recommendation.get("reason") or feedback.get("message") or "non_retryable",
                            "failure_type": recommendation.get("failure_type") or "non_retryable",
                        },
                    )
                ],
            )

        repair_steps = self._repair_steps_from_feedback(state, feedback, recommendation)
        if repair_steps:
            return AgentPlan(
                target=state.target,
                experiment_mode="dynamic_agent",
                reason=f"critic_repair:{recommendation.get('action') or 'feedback'}",
                steps=repair_steps,
            )

        cached_case = _curated_case(state)
        if state.target in {"image", "video"} and not state.generation_allowed:
            cached_payload = _cached_payload(cached_case, state.target)
            if not cached_payload:
                return AgentPlan(
                    target=state.target,
                    experiment_mode="dynamic_agent",
                    reason="restricted_generation_without_cache",
                    steps=[
                        PlanStep(
                            step_id="agent_stop",
                            skill_name="agent_stop",
                            action="stop",
                            description="Realtime generation is unavailable and no curated cache can satisfy this target.",
                            max_retries=0,
                            params={"reason": "restricted_generation", "failure_type": "permission_denied"},
                        )
                    ],
                )

        steps: List[PlanStep] = []
        completed_ids: set[str] = set()

        def add(step: PlanStep) -> None:
            steps.append(step)
            if step.action in {"reuse", "skip"}:
                completed_ids.add(step.step_id)

        if reuse_existing and _present(state.poem_info):
            add(_reuse_step("poetry_lookup", "poetry_lookup", "poem_info", {"poem_info": state.poem_info}))
        else:
            add(_base_step("poetry_lookup", []))

        if reuse_existing and _present(state.semantic_result):
            if _present(state.rag_context):
                add(_reuse_step("rag_retrieval", "rag_retrieval", "rag_context", {"rag_context": state.rag_context}))
            else:
                add(_base_step("rag_retrieval", ["poetry_lookup"]))
            add(_reuse_step("semantic_analysis", "semantic_analysis", "semantic_result", {"semantic_result": state.semantic_result}))
        else:
            if reuse_existing and _present(state.rag_context):
                add(_reuse_step("rag_retrieval", "rag_retrieval", "rag_context", {"rag_context": state.rag_context}))
            else:
                add(_base_step("rag_retrieval", ["poetry_lookup"]))
            add(_base_step("semantic_analysis", ["rag_retrieval"]))

        if reuse_existing and _valid_geojson(state.geojson):
            if reuse_existing and _present(state.place_alignment):
                add(_reuse_step("place_alignment", "place_alignment", "place_alignment", {"place_alignment": state.place_alignment}))
            else:
                add(_base_step("place_alignment", ["semantic_analysis"] if _present(state.semantic_result) else []))
            add(_reuse_step("spatial_mapping", "spatial_mapping", "geojson", {"geojson": state.geojson}))
        else:
            if reuse_existing and _present(state.place_alignment):
                add(_reuse_step("place_alignment", "place_alignment", "place_alignment", {"place_alignment": state.place_alignment}))
            else:
                add(_base_step("place_alignment", ["semantic_analysis"]))
            add(_base_step("spatial_mapping", ["place_alignment"]))

        if state.target in {"image", "video"}:
            if reuse_existing and _present(state.story_context) and _present(state.narrative_chain):
                add(
                    _reuse_step(
                        "narrative_generation",
                        "narrative_generation",
                        "story_context",
                        {"story_context": state.story_context, "narrative_chain": state.narrative_chain, "query_plan": state.query_plan},
                    )
                )
            else:
                add(
                    PlanStep(
                        step_id="narrative_generation",
                        skill_name="narrative_generation",
                        description="Build the poetry-geography narrative chain and story context.",
                        depends_on=["semantic_analysis", "spatial_mapping"],
                        success_conditions=["narrative_chain", "story_context"],
                        max_retries=1,
                        params=_goal_step_params(state, "narrative_generation"),
                    )
                )

        if state.target == "image":
            cached_image = _cached_payload(cached_case, "image")
            if reuse_existing and _valid_image(state.image):
                add(_reuse_step("poetry_image", "poetry_image", "image", {"image": state.image}))
            elif cached_image:
                add(_reuse_step("poetry_image", "poetry_image", "curated_image_cache", {"image": cached_image}, cache_key=cached_image.get("case_id", "")))
            else:
                add(_image_step(depends_on=["narrative_generation"], params=_goal_step_params(state, "poetry_image")))

        if state.target == "video":
            cached_video = _cached_payload(cached_case, "video")
            if reuse_existing and _valid_video(state.video):
                add(_reuse_step("poetry_video", "poetry_video", "video", {"video": state.video}))
            elif cached_video:
                cached_image = _cached_payload(cached_case, "image")
                if cached_image and not (reuse_existing and _valid_image(state.image)):
                    add(_reuse_step("poetry_image", "poetry_image", "curated_image_cache", {"image": cached_image}, cache_key=cached_image.get("case_id", "")))
                elif reuse_existing and _valid_image(state.image):
                    add(_reuse_step("poetry_image", "poetry_image", "image", {"image": state.image}))
                else:
                    add(_skip_step("poetry_image", "poetry_image", "curated video cache does not require image generation"))
                add(_reuse_step("poetry_video", "poetry_video", "curated_video_cache", {"video": cached_video}, cache_key=cached_video.get("case_id", "")))
            else:
                if reuse_existing and _valid_image(state.image):
                    add(_reuse_step("poetry_image", "poetry_image", "image", {"image": state.image}))
                else:
                    add(_image_step(depends_on=["narrative_generation"], params=_goal_step_params(state, "poetry_image")))
                add(
                    PlanStep(
                        step_id="poetry_video",
                        skill_name="poetry_video",
                        description="Generate Wan image-to-video output from the poetry image.",
                        depends_on=["poetry_image"],
                        success_conditions=["video"],
                        max_retries=0,
                        params=_goal_step_params(state, "poetry_video"),
                    )
                )

        return AgentPlan(target=state.target, steps=steps, experiment_mode="dynamic_agent", reason="state_aware_goal_plan")

    def _repair_steps_from_feedback(
        self,
        state: AgentState,
        feedback: Dict[str, Any],
        recommendation: Dict[str, Any],
    ) -> List[PlanStep]:
        action = recommendation.get("action")
        if action == "rerun_dependency":
            dependencies = recommendation.get("dependencies") or []
            steps: List[PlanStep] = []
            if "place_alignment" in dependencies:
                steps.append(_repair_step(_base_step("place_alignment", [] if _present(state.semantic_result) else ["semantic_analysis"]), recommendation))
            if "spatial_mapping" in dependencies:
                depends = ["place_alignment"] if any(step.step_id == "place_alignment" for step in steps) else []
                steps.append(_repair_step(_base_step("spatial_mapping", depends), recommendation))
            return steps
        if action == "retry_with_params":
            step_id = str(recommendation.get("step_id") or feedback.get("step_id") or "").strip()
            params = dict(recommendation.get("params") or {})
            if step_id == "poetry_image":
                step = _image_step(depends_on=[] if _present(state.story_context) else ["narrative_generation"], params=_goal_step_params(state, "poetry_image"))
                step.params.update(params)
                return [_repair_step(step, recommendation)]
            if step_id == "poetry_video":
                step = PlanStep(
                    step_id="poetry_video",
                    skill_name="poetry_video",
                    description="Repair video generation using Critic feedback.",
                    depends_on=[] if _valid_image(state.image) else ["poetry_image"],
                    success_conditions=["video"],
                    max_retries=0,
                    params={**_goal_step_params(state, "poetry_video"), **params},
                )
                return [_repair_step(step, recommendation)]
        if action == "reuse_cache":
            payload = recommendation.get("payload") if isinstance(recommendation.get("payload"), dict) else {}
            target = str(recommendation.get("target") or "").strip()
            if target and payload:
                return [_reuse_step(target, target, recommendation.get("source") or "critic_cache", payload)]
        if action == "skip_step":
            step_id = str(recommendation.get("step_id") or feedback.get("step_id") or "").strip()
            if step_id:
                return [_skip_step(step_id, step_id, recommendation.get("reason") or "critic recommended skip")]
        return []

    def _base_steps(self) -> List[PlanStep]:
        return [
            PlanStep(
                step_id="poetry_lookup",
                skill_name="poetry_lookup",
                description="Resolve the input poem and retrieve local poetry-library candidates.",
                depends_on=[],
                success_conditions=["poem_info"],
                max_retries=1,
            ),
            PlanStep(
                step_id="rag_retrieval",
                skill_name="rag_retrieval",
                description="Retrieve local poetry context for downstream reasoning.",
                depends_on=["poetry_lookup"],
                success_conditions=["rag_context"],
                max_retries=1,
            ),
            PlanStep(
                step_id="semantic_analysis",
                skill_name="semantic_analysis",
                description="Extract structured semantic labels for place, imagery, emotion, and temporal-spatial context.",
                depends_on=["rag_retrieval"],
                success_conditions=["semantic_result"],
                max_retries=1,
            ),
            PlanStep(
                step_id="place_alignment",
                skill_name="place_alignment",
                description="Align historical place labels to modern geographic entities.",
                depends_on=["semantic_analysis"],
                success_conditions=["place_alignment"],
                max_retries=1,
            ),
            PlanStep(
                step_id="spatial_mapping",
                skill_name="spatial_mapping",
                description="Build coordinate mapping and GeoJSON output.",
                depends_on=["place_alignment"],
                success_conditions=["geojson"],
                max_retries=1,
            ),
        ]

    def _map_steps(self) -> List[PlanStep]:
        return self._base_steps()

    def _image_steps(self) -> List[PlanStep]:
        return self._base_steps() + [
            PlanStep(
                step_id="narrative_generation",
                skill_name="narrative_generation",
                description="Build the poetry-geography narrative chain and story context.",
                depends_on=["semantic_analysis", "spatial_mapping"],
                success_conditions=["narrative_chain", "story_context"],
                max_retries=1,
            ),
            PlanStep(
                step_id="poetry_image",
                skill_name="poetry_image",
                description="Generate poetry landscape image from story context.",
                depends_on=["narrative_generation"],
                success_conditions=["image"],
                max_retries=2,
                fallback_skill="poetry_image_retry",
            )
        ]

    def _video_steps(self) -> List[PlanStep]:
        return self._base_steps() + [
            PlanStep(
                step_id="narrative_generation",
                skill_name="narrative_generation",
                description="Build the poetry-geography narrative chain and story context.",
                depends_on=["semantic_analysis", "spatial_mapping"],
                success_conditions=["narrative_chain", "story_context"],
                max_retries=1,
            ),
            PlanStep(
                step_id="poetry_image",
                skill_name="poetry_image",
                description="Generate poetry landscape image when video input image is missing.",
                depends_on=["narrative_generation"],
                success_conditions=["image"],
                max_retries=2,
                fallback_skill="poetry_image_retry",
            ),
            PlanStep(
                step_id="poetry_video",
                skill_name="poetry_video",
                description="Generate Wan image-to-video output from the generated poetry image.",
                depends_on=["poetry_image"],
                success_conditions=["video"],
                max_retries=2,
            )
        ]


def plan_skill_names(plan: AgentPlan) -> Dict[str, str]:
    return {step.step_id: step.skill_name for step in plan.steps}


def _base_step(step_id: str, depends_on: List[str]) -> PlanStep:
    definitions = {
        "poetry_lookup": ("poetry_lookup", "Resolve the input poem and retrieve local poetry-library candidates.", ["poem_info"], 1),
        "rag_retrieval": ("rag_retrieval", "Retrieve local poetry context for downstream reasoning.", ["rag_context"], 1),
        "semantic_analysis": ("semantic_analysis", "Extract structured semantic labels.", ["semantic_result"], 1),
        "place_alignment": ("place_alignment", "Align historical place labels to modern geographic entities.", ["place_alignment"], 1),
        "spatial_mapping": ("spatial_mapping", "Build coordinate mapping and GeoJSON output.", ["geojson"], 1),
    }
    skill_name, description, conditions, retries = definitions[step_id]
    return PlanStep(
        step_id=step_id,
        skill_name=skill_name,
        description=description,
        depends_on=depends_on,
        success_conditions=conditions,
        max_retries=retries,
    )


def _image_step(*, depends_on: List[str], params: Dict[str, Any] | None = None) -> PlanStep:
    return PlanStep(
        step_id="poetry_image",
        skill_name="poetry_image",
        description="Generate poetry landscape image from story context.",
        depends_on=depends_on,
        success_conditions=["image"],
        max_retries=0,
        fallback_skill="poetry_image_retry",
        params=dict(params or {}),
    )


def _reuse_step(
    step_id: str,
    skill_name: str,
    source: str,
    reuse_output: Dict[str, Any],
    *,
    cache_key: str = "",
) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        skill_name=skill_name,
        action="reuse",
        description=f"Reuse existing {source}.",
        depends_on=[],
        success_conditions=[],
        max_retries=0,
        params={"reuse_source": source, "reuse_output": reuse_output, "cache_key": cache_key},
    )


def _skip_step(step_id: str, skill_name: str, reason: str) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        skill_name=skill_name,
        action="skip",
        description=f"Skip {step_id}.",
        depends_on=[],
        success_conditions=[],
        max_retries=0,
        params={"skip_reason": reason},
    )


def _repair_step(step: PlanStep, recommendation: Dict[str, Any]) -> PlanStep:
    step.action = "repair"
    step.max_retries = 0
    step.params = {**step.params, "repair_recommendation": recommendation}
    return step


def _feedback_dict(feedback: Optional[StepValidation | Dict[str, Any]]) -> Dict[str, Any]:
    if feedback is None:
        return {}
    if hasattr(feedback, "model_dump"):
        return feedback.model_dump()
    if hasattr(feedback, "dict"):
        return feedback.dict()
    return dict(feedback)


def _present(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def _valid_geojson(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") == "FeatureCollection" and isinstance(value.get("features"), list) and bool(value.get("features"))


def _valid_image(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return bool(str(value.get("normalized_image_url") or value.get("image_url") or "").strip())


def _valid_video(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return bool(str(value.get("normalized_video_url") or value.get("video_url") or "").strip())


def _reuse_existing_allowed(state: AgentState) -> bool:
    constraints = getattr(state, "goal_constraints", None)
    if constraints is None:
        return True
    return bool(getattr(constraints, "reuse_existing", True))


def _goal_step_params(state: AgentState, step_id: str) -> Dict[str, Any]:
    constraints = getattr(state, "goal_constraints", None)
    if constraints is None:
        return {}
    payload = constraints.model_dump() if hasattr(constraints, "model_dump") else constraints.dict()
    params: Dict[str, Any] = {"goal_constraints": payload}
    if step_id == "poetry_image":
        style = payload.get("visual_style") or []
        if style:
            params["mode"] = "agent_style:" + "|".join(str(item) for item in style)
    if step_id == "poetry_video":
        duration = int(payload.get("duration_seconds") or 0)
        if duration:
            params["duration_seconds"] = duration
        if payload.get("camera_requirements"):
            params["camera_requirements"] = payload.get("camera_requirements")
        if payload.get("narrative_actions"):
            params["narrative_actions"] = payload.get("narrative_actions")
        if payload.get("audio_requirements"):
            params["audio_requirements"] = payload.get("audio_requirements")
    return params


def _curated_case(state: AgentState) -> Dict[str, Any]:
    poem = state.poem_info or {}
    return find_curated_case(
        poem.get("title"),
        poem.get("author"),
        poem.get("full_poem"),
        state.user_input,
        state.poetry_id,
    ) or {}


def _cached_payload(case: Dict[str, Any], target: str) -> Dict[str, Any]:
    if not case:
        return {}
    case_id = str(case.get("id") or "")
    if target == "image":
        image_url = str(case.get("cached_image_url") or "").strip()
        if not image_url:
            return {}
        return {
            "image_url": image_url,
            "normalized_image_url": image_url,
            "prompt": str(case.get("narrative_note") or ""),
            "provider": "curated-cache",
            "model": "featured-case-cache",
            "status": "succeeded",
            "source_type": "curated_featured_case",
            "included_in_experiment_dataset": False,
            "case_id": case_id,
        }
    if target == "video":
        video_url = str(case.get("cached_video_url") or "").strip()
        if not video_url:
            return {}
        return {
            "status": "ready",
            "provider": "curated-cache",
            "id": f"curated-{case_id or 'featured-case'}",
            "video_url": video_url,
            "normalized_video_url": video_url,
            "progress": 100.0,
            "prompt": str(case.get("narrative_note") or ""),
            "error": "",
            "source_type": "curated_featured_case",
            "included_in_experiment_dataset": False,
            "case_id": case_id,
        }
    return {}
