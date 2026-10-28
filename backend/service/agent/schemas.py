from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator


AgentTarget = Literal["map", "image", "video"]
AgentStatus = Literal["created", "planned", "running", "succeeded", "failed", "stopped"]
ExperimentMode = Literal["fixed", "static_agent", "dynamic_agent"]
StepAction = Literal["execute", "skip", "reuse", "repair", "stop"]
StepStatus = Literal["pending", "running", "succeeded", "failed", "skipped", "reused"]


def _task_id() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.utcnow()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


class AgentError(BaseModel):
    step_id: str = ""
    skill_name: str = ""
    message: str
    recoverable: bool = True
    timestamp: datetime = Field(default_factory=_now)


class TraceEvent(BaseModel):
    event: str
    step_id: str = ""
    skill_name: str = ""
    detail: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_now)


class SkillResult(BaseModel):
    ok: bool = True
    output: Dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    error: str = ""


class StepValidation(BaseModel):
    ok: bool
    step_id: str
    success_conditions: List[str] = Field(default_factory=list)
    missing_conditions: List[str] = Field(default_factory=list)
    message: str = ""
    recommendation: Dict[str, Any] = Field(default_factory=dict)


class GoalConstraints(BaseModel):
    raw_requirement: str = ""
    target: AgentTarget = "map"
    needs_map: bool = True
    needs_image: bool = False
    needs_video: bool = False
    duration_seconds: int = 0
    visual_style: List[str] = Field(default_factory=list)
    narrative_actions: List[str] = Field(default_factory=list)
    camera_requirements: List[str] = Field(default_factory=list)
    audio_requirements: List[str] = Field(default_factory=list)
    reuse_existing: bool = True
    source: str = "rule_based_goal_interpreter"

    @validator("raw_requirement", "source", pre=True)
    def normalize_goal_text(cls, value: Any) -> str:
        return _clean_text(value)

    @validator("visual_style", "narrative_actions", "camera_requirements", "audio_requirements", pre=True)
    def normalize_goal_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        result: List[str] = []
        for raw in value:
            item = _clean_text(raw)
            if item and item not in result:
                result.append(item)
        return result


class ExecutionStepRecord(BaseModel):
    step_id: str
    skill_name: str
    action: StepAction = "execute"
    status: StepStatus = "pending"
    attempt: int = 0
    fallback_used: bool = False
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    output: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    validation: Optional[StepValidation] = None


class PlanStep(BaseModel):
    step_id: str
    skill_name: str
    action: StepAction = "execute"
    description: str = ""
    depends_on: List[str] = Field(default_factory=list)
    success_conditions: List[str] = Field(default_factory=list)
    max_retries: int = Field(1, ge=0, le=5)
    fallback_skill: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    @validator("step_id", "skill_name", "description", "fallback_skill", pre=True, always=True)
    def normalize_text(cls, value: Any) -> str:
        return _clean_text(value)

    @validator("depends_on", "success_conditions", pre=True)
    def normalize_text_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [item for item in (_clean_text(raw) for raw in value) if item]


class AgentPlan(BaseModel):
    target: AgentTarget
    steps: List[PlanStep]
    experiment_mode: ExperimentMode = "static_agent"
    reason: str = ""
    created_at: datetime = Field(default_factory=_now)

    @validator("steps")
    def validate_unique_steps(cls, value: List[PlanStep]) -> List[PlanStep]:
        seen = set()
        for step in value:
            if step.step_id in seen:
                raise ValueError(f"duplicate plan step_id: {step.step_id}")
            seen.add(step.step_id)
        missing_deps = {
            dep
            for step in value
            for dep in step.depends_on
            if dep not in seen and dep not in {item.step_id for item in value}
        }
        if missing_deps:
            raise ValueError(f"unknown plan dependencies: {sorted(missing_deps)}")
        return value


class AgentState(BaseModel):
    task_id: str = Field(default_factory=_task_id)
    target: AgentTarget
    user_input: str = ""
    input_type: str = "text"
    poetry_id: str = ""
    generation_allowed: bool = False
    experiment_mode: ExperimentMode = "static_agent"
    task_requirement: str = ""
    goal_constraints: GoalConstraints = Field(default_factory=GoalConstraints)
    fault_profile: Dict[str, Any] = Field(default_factory=dict)
    poem_info: Dict[str, Any] = Field(default_factory=dict)
    rag_context: List[Dict[str, Any]] = Field(default_factory=list)
    query_plan: Dict[str, Any] = Field(default_factory=dict)
    semantic_result: Dict[str, Any] = Field(default_factory=dict)
    place_alignment: Dict[str, Any] = Field(default_factory=dict)
    geojson: Dict[str, Any] = Field(default_factory=dict)
    story_context: Dict[str, Any] = Field(default_factory=dict)
    narrative_chain: List[Dict[str, Any]] = Field(default_factory=list)
    image: Dict[str, Any] = Field(default_factory=dict)
    video: Dict[str, Any] = Field(default_factory=dict)
    real_video_search: Dict[str, Any] = Field(default_factory=dict)
    execution_steps: List[ExecutionStepRecord] = Field(default_factory=list)
    errors: List[AgentError] = Field(default_factory=list)
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    final_status: AgentStatus = "created"
    agent_trace: List[TraceEvent] = Field(default_factory=list)
    current_plan: Dict[str, Any] = Field(default_factory=dict)
    plan_history: List[Dict[str, Any]] = Field(default_factory=list)
    planned: List[Dict[str, Any]] = Field(default_factory=list)
    executed: List[Dict[str, Any]] = Field(default_factory=list)
    skipped: List[Dict[str, Any]] = Field(default_factory=list)
    reused: List[Dict[str, Any]] = Field(default_factory=list)
    validated: List[Dict[str, Any]] = Field(default_factory=list)
    repair_actions: List[Dict[str, Any]] = Field(default_factory=list)
    cache_events: List[Dict[str, Any]] = Field(default_factory=list)
    failure_taxonomy: List[Dict[str, Any]] = Field(default_factory=list)
    termination: Dict[str, Any] = Field(default_factory=dict)
    fault_events: List[Dict[str, Any]] = Field(default_factory=list)
    experiment_metrics: Dict[str, Any] = Field(default_factory=dict)
    paid_generation_calls: int = 0
    loop_count: int = 0
    stop_reason: str = ""
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @validator("user_input", pre=True)
    def normalize_user_input(cls, value: Any) -> str:
        return _clean_text(value)

    def add_trace(
        self,
        event: str,
        *,
        step_id: str = "",
        skill_name: str = "",
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_trace.append(
            TraceEvent(
                event=event,
                step_id=step_id,
                skill_name=skill_name,
                detail=detail or {},
            )
        )
        self.updated_at = _now()

    def add_error(
        self,
        message: str,
        *,
        step_id: str = "",
        skill_name: str = "",
        recoverable: bool = True,
    ) -> None:
        self.errors.append(
            AgentError(
                step_id=step_id,
                skill_name=skill_name,
                message=message,
                recoverable=recoverable,
            )
        )
        self.updated_at = _now()
