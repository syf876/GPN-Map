from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, validator

from backend.deployment_security import generation_access_allowed
from backend.admin_auth import SESSION, generation_session, require_generation_access
from backend.service.agent.core import PoetryGeoNarrativeAgent
from backend.service.agent.faults import normalize_fault_profile
from backend.service.agent.goal_interpreter import interpret_goal
from backend.service.agent.memory import JsonMemory
from backend.service.agent.planner import Planner
from backend.service.agent.schemas import AgentState, AgentTarget, ExperimentMode


TaskApiStatus = Literal["pending", "planning", "running", "retrying", "succeeded", "failed"]
InputType = Literal["text", "poetry_id"]


STEP_MESSAGES = {
    "poetry_lookup": "正在检索诗词知识",
    "rag_retrieval": "正在检索诗词知识",
    "semantic_analysis": "正在解析诗词语义",
    "place_alignment": "正在进行地理实体对齐",
    "spatial_mapping": "正在构建空间映射",
    "narrative_generation": "正在生成诗词地理叙事",
    "poetry_image": "正在生成 AI 意境图",
    "poetry_image_retry": "正在调整并重新生成",
    "poetry_video": "正在生成 AI 意境视频",
    "final": "正在检查生成结果",
}


class AgentTaskCreateRequest(BaseModel):
    input_type: InputType = "text"
    poetry_text: str = ""
    poetry_id: str = ""
    target: AgentTarget
    experiment_mode: ExperimentMode = "dynamic_agent"
    task_requirement: str = ""
    initial_state: Dict[str, Any] = Field(default_factory=dict)
    fault_profile: Dict[str, Any] = Field(default_factory=dict)

    @validator("poetry_text", "poetry_id", "task_requirement", pre=True)
    def normalize_text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()


class AgentTaskCreateResponse(BaseModel):
    task_id: str
    status: TaskApiStatus
    target: AgentTarget
    created_at: datetime
    updated_at: datetime


class AgentTaskResponse(BaseModel):
    task_id: str
    status: TaskApiStatus
    target: AgentTarget
    current_step: str = ""
    progress: int = Field(0, ge=0, le=100)
    message: str = ""
    result: Dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    created_at: datetime
    updated_at: datetime


class AgentTraceResponse(BaseModel):
    task_id: str
    status: TaskApiStatus
    target: AgentTarget
    agent_trace: list[Dict[str, Any]] = Field(default_factory=list)
    execution_steps: list[Dict[str, Any]] = Field(default_factory=list)
    errors: list[Dict[str, Any]] = Field(default_factory=list)
    stop_reason: str = ""
    current_plan: Dict[str, Any] = Field(default_factory=dict)
    plan_history: list[Dict[str, Any]] = Field(default_factory=list)
    planned: list[Dict[str, Any]] = Field(default_factory=list)
    executed: list[Dict[str, Any]] = Field(default_factory=list)
    skipped: list[Dict[str, Any]] = Field(default_factory=list)
    reused: list[Dict[str, Any]] = Field(default_factory=list)
    validated: list[Dict[str, Any]] = Field(default_factory=list)
    repair_actions: list[Dict[str, Any]] = Field(default_factory=list)
    cache_events: list[Dict[str, Any]] = Field(default_factory=list)
    failure_taxonomy: list[Dict[str, Any]] = Field(default_factory=list)
    experiment_mode: ExperimentMode = "dynamic_agent"
    task_requirement: str = ""
    goal_constraints: Dict[str, Any] = Field(default_factory=dict)
    termination: Dict[str, Any] = Field(default_factory=dict)
    fault_profile: Dict[str, Any] = Field(default_factory=dict)
    fault_events: list[Dict[str, Any]] = Field(default_factory=list)
    experiment_metrics: Dict[str, Any] = Field(default_factory=dict)
    paid_generation_calls: int = 0


class AgentTaskService:
    def __init__(
        self,
        *,
        memory: Optional[JsonMemory] = None,
        planner: Optional[Planner] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        self.memory = memory or JsonMemory()
        self.planner = planner or Planner()
        self.executor = executor or ThreadPoolExecutor(max_workers=3, thread_name_prefix="poetry-agent")
        self._started_at = datetime.utcnow()
        self._task_sessions: Dict[str, str] = {}

    def create_task(self, request: AgentTaskCreateRequest) -> AgentTaskResponse:
        user_input = request.poetry_text or request.poetry_id
        if not user_input:
            raise ValueError("poetry_text or poetry_id is required")

        goal = interpret_goal(
            requirement=request.task_requirement,
            requested_target=request.target,
            poem_text=request.poetry_text or request.poetry_id,
        )
        if goal.target in {"image", "video"}:
            require_generation_access(goal.target)
        state = AgentState(
            target=goal.target,
            user_input=user_input,
            input_type=request.input_type,
            poetry_id=request.poetry_id,
            generation_allowed=self._generation_allowed(goal.target, ""),
            experiment_mode=request.experiment_mode,
            task_requirement=request.task_requirement,
            goal_constraints=goal,
            fault_profile=normalize_fault_profile(request.fault_profile),
        )
        # Guest featured cases read the complete saved snapshot, never a live skill.
        from backend.demo_catalog import find_case, saved_result
        case = find_case(request.poetry_id, request.poetry_text)
        if case and not generation_access_allowed("image") and goal.target == "map":
            cached = saved_result(case)
            for attr, key in {"semantic_result":"prediction", "geojson":"map", "story_context":"storyContext",
                              "query_plan":"planner", "narrative_chain":"narrative_chain",
                              "image":"image", "video":"ai_video"}.items():
                setattr(state, attr, cached.get(key, [] if attr == "narrative_chain" else {}))
            state.final_status = "succeeded"
            state.add_trace("saved_demo_read", detail={"case_id": case["id"]})
            self.memory.save(state)
            return self.to_response(state)
        # Guests cannot smuggle media or plans through initial_state.
        if goal.target == "map" and not generation_access_allowed("image"):
            request.initial_state = {}
        _apply_initial_state(state, request.initial_state)
        state.add_trace(
            "task_created",
            detail={
                "requested_target": request.target,
                "target": state.target,
                "input_type": request.input_type,
                "experiment_mode": request.experiment_mode,
                "poetry_id_present": bool(request.poetry_id),
                "task_requirement_present": bool(request.task_requirement),
                "goal_constraints": _model_dump(goal),
                "initial_state": _initial_state_summary(request.initial_state),
                "generation_allowed": state.generation_allowed,
                "fault_profile_enabled": bool(state.fault_profile.get("enabled")),
            },
        )
        self.memory.save(state)
        self._task_sessions[state.task_id] = SESSION.get()
        self.executor.submit(self._run_task, state.task_id)
        return self.to_response(state)

    def get_task(self, task_id: str) -> AgentTaskResponse | None:
        state = self.memory.load(task_id)
        if not state:
            return None
        # A map worker from a previous process cannot resume after restart.
        # Report its interruption instead of leaving the client polling forever.
        if state.target == "map" and state.updated_at < self._started_at and state.final_status not in {"succeeded", "failed", "stopped"}:
            state.final_status = "failed"
            state.stop_reason = "map_task_interrupted_by_restart"
            state.add_error("地图分析任务因服务重启中断，请重新选择诗词。", recoverable=False)
            self.memory.save(state)
        return self.to_response(state)

    def get_trace(self, task_id: str) -> AgentTraceResponse | None:
        state = self.memory.load(task_id)
        if not state:
            return None
        return AgentTraceResponse(
            task_id=state.task_id,
            status=self.api_status(state),
            target=state.target,
            agent_trace=[_model_dump(event) for event in state.agent_trace],
            execution_steps=[_model_dump(step) for step in state.execution_steps],
            errors=[_model_dump(error) for error in state.errors],
            stop_reason=state.stop_reason,
            current_plan=state.current_plan,
            plan_history=state.plan_history,
            planned=state.planned,
            executed=state.executed,
            skipped=state.skipped,
            reused=state.reused,
            validated=state.validated,
            repair_actions=state.repair_actions,
            cache_events=state.cache_events,
            failure_taxonomy=state.failure_taxonomy,
            experiment_mode=state.experiment_mode,
            task_requirement=state.task_requirement,
            goal_constraints=_model_dump(state.goal_constraints),
            termination=state.termination,
            fault_profile=state.fault_profile,
            fault_events=state.fault_events,
            experiment_metrics=state.experiment_metrics,
            paid_generation_calls=state.paid_generation_calls,
        )

    def to_response(self, state: AgentState) -> AgentTaskResponse:
        current_step = self.current_step(state)
        status = self.api_status(state)
        return AgentTaskResponse(
            task_id=state.task_id,
            status=status,
            target=state.target,
            current_step=current_step,
            progress=self.progress(state, status),
            message=self.message(state, status, current_step),
            result=self.result_payload(state) if status == "succeeded" else self.partial_result_payload(state),
            error=self.error_message(state) if status == "failed" else "",
            created_at=state.created_at,
            updated_at=state.updated_at,
        )

    def api_status(self, state: AgentState) -> TaskApiStatus:
        if state.final_status == "succeeded":
            return "succeeded"
        if state.final_status in {"failed", "stopped"}:
            return "failed"
        if self._latest_trace_event(state) == "step_retry_scheduled":
            return "retrying"
        if any(step.status == "running" for step in state.execution_steps):
            return "running"
        if state.final_status == "planned":
            return "planning"
        if state.final_status == "running":
            return "running"
        return "pending"

    def current_step(self, state: AgentState) -> str:
        running = [step for step in state.execution_steps if step.status == "running"]
        if running:
            return running[-1].step_id
        if state.execution_steps:
            return state.execution_steps[-1].step_id
        if state.final_status == "planned":
            return "planning"
        return ""

    def progress(self, state: AgentState, status: TaskApiStatus) -> int:
        if status == "succeeded":
            return 100
        if status == "failed":
            return min(99, max(1, self._completed_progress(state)))
        if status == "pending":
            return 0
        return max(5, min(95, self._completed_progress(state)))

    def message(self, state: AgentState, status: TaskApiStatus, current_step: str) -> str:
        if status == "succeeded":
            return "任务已完成"
        if status == "failed":
            return self.error_message(state) or "任务执行失败"
        if status == "retrying":
            return "正在调整并重新生成"
        if current_step:
            return STEP_MESSAGES.get(current_step, "正在执行任务步骤")
        if status == "planning":
            return "正在规划任务步骤"
        return "任务已创建，等待执行"

    def result_payload(self, state: AgentState) -> Dict[str, Any]:
        payload = self.partial_result_payload(state)
        payload["ok"] = True
        return payload

    def partial_result_payload(self, state: AgentState) -> Dict[str, Any]:
        rag = state.rag_context[0] if state.rag_context else {}
        return {
            "ok": state.final_status == "succeeded",
            "target": state.target,
            "prediction": state.semantic_result or {},
            "map": state.geojson or {},
            "geojson": state.geojson or {},
            "planner": state.query_plan or {},
            "video": state.real_video_search or {},
            "storyContext": state.story_context or {},
            "narrative_chain": state.narrative_chain or [],
            "image": state.image or {},
            "poetry_image": state.image or {},
            "ai_video": state.video or {},
            "primary_poem": rag.get("primary_poem") or rag.get("primary_poem_context") or {},
            "related_knowledge": rag.get("related_knowledge") or rag.get("results") or [],
            "rag": rag,
            "agent": {
                "task_id": state.task_id,
                "status": self.api_status(state),
                "current_step": self.current_step(state),
                "stop_reason": state.stop_reason,
                "experiment_mode": state.experiment_mode,
                "task_requirement": state.task_requirement,
                "goal_constraints": _model_dump(state.goal_constraints),
            },
            "experiment": self.experiment_payload(state),
        }

    def experiment_payload(self, state: AgentState) -> Dict[str, Any]:
        metrics = state.experiment_metrics or {}
        return {
            "run_id": state.task_id,
            "sample_id": state.poetry_id or state.task_id,
            "mode": state.experiment_mode,
            "target": state.target,
            "initial_state": {
                "has_semantic_result": bool(state.semantic_result),
                "has_geojson": bool(state.geojson),
                "has_image": bool(state.image),
                "has_video": bool(state.video),
                "generation_allowed": state.generation_allowed,
                "fault_profile": state.fault_profile,
                "goal_constraints": _model_dump(state.goal_constraints),
            },
            "planned_steps": state.planned,
            "executed_steps": state.executed,
            "skipped_steps": state.skipped,
            "reused_steps": state.reused,
            "repair_actions": state.repair_actions,
            "critic_results": state.validated,
            "cache_hits": state.cache_events,
            "metrics": metrics,
            "tool_calls": metrics.get("tool_calls", 0),
            "retries": metrics.get("retries", 0),
            "duration_ms": metrics.get("duration_ms", 0),
            "paid_generation_calls": metrics.get("paid_generation_calls", state.paid_generation_calls),
            "final_status": state.final_status,
            "stop_reason": state.stop_reason,
            "permission_compliance": metrics.get("permission_compliance", True),
            "trace_complete": metrics.get("trace_complete", False),
        }

    def error_message(self, state: AgentState) -> str:
        if state.errors:
            return state.errors[-1].message
        return state.stop_reason

    def _run_task(self, task_id: str) -> None:
        token = self._task_sessions.pop(task_id, "")
        with generation_session(token):
            self._execute_task(task_id)

    def _execute_task(self, task_id: str) -> None:
        state = self.memory.load(task_id)
        if not state:
            return
        try:
            if state.target in {"image", "video"}:
                require_generation_access(state.target)
            try:
                agent = PoetryGeoNarrativeAgent(memory=self.memory, max_loop_count=15, experiment_mode=state.experiment_mode)
            except TypeError:
                agent = PoetryGeoNarrativeAgent(memory=self.memory, max_loop_count=15)
            plan = agent.plan(state)
            agent.execute(state, plan)
            agent.finish(state)
        except Exception as exc:
            state.final_status = "failed"
            state.stop_reason = "agent_exception"
            state.add_error(str(exc), recoverable=False)
            state.add_trace("agent_exception", detail={"error": str(exc)})
            self.memory.save(state)

    def _completed_progress(self, state: AgentState) -> int:
        try:
            if state.current_plan and isinstance(state.current_plan.get("steps"), list):
                total_steps = len(state.current_plan.get("steps") or [])
            else:
                total_steps = len(self.planner.build_plan(state.target).steps)
        except Exception:
            total_steps = max(1, len(state.execution_steps))
        completed = len({step.step_id for step in state.execution_steps if step.status == "succeeded"})
        return int((completed / max(1, total_steps)) * 100)

    def _generation_allowed(self, target: AgentTarget, supplied_token: str) -> bool:
        if target == "map":
            return False
        kind = "video" if target == "video" else "image"
        return generation_access_allowed(kind, supplied_token)

    def _latest_trace_event(self, state: AgentState) -> str:
        if not state.agent_trace:
            return ""
        return state.agent_trace[-1].event


def _model_dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return {}


def _apply_initial_state(state: AgentState, payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict) or not payload:
        return
    allowed_fields = {
        "poem_info",
        "rag_context",
        "query_plan",
        "semantic_result",
        "place_alignment",
        "geojson",
        "story_context",
        "narrative_chain",
        "image",
        "video",
        "real_video_search",
    }
    for key in allowed_fields:
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        current = getattr(state, key, None)
        if isinstance(current, dict) and isinstance(value, dict):
            setattr(state, key, value)
        elif isinstance(current, list) and isinstance(value, list):
            setattr(state, key, value)
    if payload:
        state.add_trace("initial_state_loaded", detail=_initial_state_summary(payload))


def _initial_state_summary(payload: Dict[str, Any]) -> Dict[str, bool]:
    data = payload if isinstance(payload, dict) else {}
    return {
        "has_poem_info": bool(data.get("poem_info")),
        "has_rag_context": bool(data.get("rag_context")),
        "has_semantic_result": bool(data.get("semantic_result")),
        "has_place_alignment": bool(data.get("place_alignment")),
        "has_geojson": bool(data.get("geojson")),
        "has_story_context": bool(data.get("story_context")),
        "has_narrative_chain": bool(data.get("narrative_chain")),
        "has_image": bool(data.get("image")),
        "has_video": bool(data.get("video")),
    }


agent_task_service = AgentTaskService()
