from datetime import datetime
from typing import Any, Dict, Optional, Set

from backend.service.agent.critic import Critic
from backend.service.agent.executor import Executor
from backend.service.agent.memory import JsonMemory
from backend.service.agent.planner import Planner
from backend.service.agent.real_skills import build_real_skill_registry
from backend.service.agent.schemas import AgentPlan, AgentState, AgentTarget, ExperimentMode, ExecutionStepRecord, PlanStep, StepValidation
from backend.service.agent.skills import SkillRegistry


class PoetryGeoNarrativeAgent:
    def __init__(
        self,
        *,
        planner: Optional[Planner] = None,
        registry: Optional[SkillRegistry] = None,
        critic: Optional[Critic] = None,
        memory: Optional[JsonMemory] = None,
        max_loop_count: int = 20,
        experiment_mode: ExperimentMode = "static_agent",
    ) -> None:
        self.planner = planner or Planner()
        self.registry = registry or build_real_skill_registry()
        self.critic = critic or Critic()
        self.memory = memory or JsonMemory()
        self.executor = Executor(self.registry, on_update=self.memory.save)
        self.max_loop_count = max(1, int(max_loop_count))
        self.experiment_mode = experiment_mode

    def run(self, user_input: str, target: AgentTarget = "map") -> AgentState:
        state = AgentState(target=target, user_input=user_input, experiment_mode=self.experiment_mode)
        plan = self.plan(state)
        self.execute(state, plan)
        self.finish(state)
        return state

    def plan(self, state: AgentState) -> AgentPlan:
        mode = state.experiment_mode or self.experiment_mode
        plan = self.planner.build_plan(state, experiment_mode=mode)
        self._validate_plan_skills(plan)
        state.final_status = "planned"
        plan_payload = _model_dump(plan)
        state.current_plan = plan_payload
        state.plan_history.append(plan_payload)
        for step in plan.steps:
            state.planned.append(_planned_payload(step, plan.reason, plan.experiment_mode))
        state.add_trace(
            "plan_created",
            detail={
                "target": plan.target,
                "experiment_mode": plan.experiment_mode,
                "reason": plan.reason,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "skill_name": step.skill_name,
                        "action": step.action,
                        "depends_on": step.depends_on,
                        "success_conditions": step.success_conditions,
                        "max_retries": step.max_retries,
                        "fallback_skill": step.fallback_skill or "",
                    }
                    for step in plan.steps
                ],
            },
        )
        self.memory.save(state)
        return plan

    def execute(self, state: AgentState, plan: AgentPlan) -> AgentState:
        state.final_status = "running"
        completed: Set[str] = set()
        for step in self.executor.ordered_steps(plan):
            if state.loop_count >= self.max_loop_count:
                state.final_status = "stopped"
                state.stop_reason = "max_loop_count_reached"
                state.add_trace("agent_stopped", step_id=step.step_id, detail={"reason": state.stop_reason})
                self.memory.save(state)
                return state
            missing_deps = [dep for dep in step.depends_on if dep not in completed]
            if missing_deps:
                state.final_status = "failed"
                state.stop_reason = "dependency_failed"
                state.add_error(
                    f"step dependencies are not complete: {missing_deps}",
                    step_id=step.step_id,
                    skill_name=step.skill_name,
                    recoverable=False,
                )
                state.add_trace("step_skipped", step_id=step.step_id, detail={"missing_deps": missing_deps})
                self.memory.save(state)
                return state
            if not self._run_planned_step(state, step):
                if state.final_status != "stopped":
                    state.final_status = "failed"
                if not state.stop_reason:
                    state.stop_reason = f"step_failed:{step.step_id}"
                self.memory.save(state)
                return state
            completed.add(step.step_id)
        return state

    def finish(self, state: AgentState) -> AgentState:
        if state.final_status in {"failed", "stopped"}:
            state.add_trace("agent_finished", detail={"status": state.final_status, "stop_reason": state.stop_reason})
            self._update_experiment_metrics(state)
            self.memory.save(state)
            return state

        final_validation = self.critic.evaluate_final(state)
        state.add_trace(
            "final_evaluation",
            detail={
                "ok": final_validation.ok,
                "missing_conditions": final_validation.missing_conditions,
                "success_conditions": final_validation.success_conditions,
            },
        )
        state.final_status = "succeeded" if final_validation.ok else "failed"
        state.stop_reason = "completed" if final_validation.ok else "final_validation_failed"
        if state.stop_reason == "completed":
            state.termination = {"reason": "completed", "mode": state.experiment_mode}
        self._update_experiment_metrics(state)
        self.memory.save(state)
        return state

    def _run_step_with_retry(self, state: AgentState, step: PlanStep) -> bool:
        max_attempts = step.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            if state.loop_count >= self.max_loop_count:
                return False
            state.loop_count += 1
            fallback_used = bool(step.fallback_skill and attempt > 1)
            active_skill = step.fallback_skill if fallback_used else step.skill_name
            record = self.executor.execute_step(
                state,
                step,
                skill_name=active_skill,
                attempt=attempt,
                fallback_used=fallback_used,
            )
            validation = self.critic.evaluate_step(state, step) if record.status == "succeeded" else self.critic.evaluate_failure(state, step, record.error)
            record.validation = validation
            self._record_validation(state, step, validation)
            state.add_trace(
                "step_evaluated",
                step_id=step.step_id,
                skill_name=active_skill or step.skill_name,
                detail={
                    "attempt": attempt,
                    "ok": bool(validation and validation.ok),
                    "missing_conditions": validation.missing_conditions if validation else step.success_conditions,
                    "recommendation": validation.recommendation if validation else {},
                },
            )
            self.memory.save(state)
            if validation and validation.ok:
                return True
            if validation and validation.recommendation.get("action") == "stop_non_retryable":
                state.final_status = "stopped"
                state.stop_reason = validation.recommendation.get("reason") or "non_retryable"
                state.termination = {
                    "reason": state.stop_reason,
                    "step_id": step.step_id,
                    "failure_type": validation.recommendation.get("failure_type", ""),
                }
                state.add_trace("agent_stopped", step_id=step.step_id, skill_name=active_skill or step.skill_name, detail=state.termination)
                return False
            if attempt < max_attempts:
                retry_count = state.retry_counts.get(step.step_id, 0) + 1
                state.retry_counts[step.step_id] = retry_count
                state.add_trace(
                    "step_retry_scheduled",
                    step_id=step.step_id,
                    skill_name=active_skill or step.skill_name,
                    detail={
                        "retry_count": retry_count,
                        "next_skill": step.fallback_skill if step.fallback_skill else step.skill_name,
                    },
                )
        state.add_trace(
            "step_retry_exhausted",
            step_id=step.step_id,
            skill_name=step.skill_name,
            detail={"max_retries": step.max_retries},
        )
        return False

    def _run_planned_step(self, state: AgentState, step: PlanStep) -> bool:
        if state.experiment_mode == "fixed":
            return self._run_fixed_step_once(state, step)
        if state.experiment_mode != "dynamic_agent":
            return self._run_step_with_retry(state, step)
        return self._run_dynamic_step(state, step)

    def _run_fixed_step_once(self, state: AgentState, step: PlanStep) -> bool:
        if state.loop_count >= self.max_loop_count:
            return False
        state.loop_count += 1
        record = self.executor.execute_step(state, step, attempt=1, fallback_used=False)
        validation = self.critic.evaluate_step(state, step) if record.status == "succeeded" else self.critic.evaluate_failure(state, step, record.error)
        record.validation = validation
        self._record_validation(state, step, validation)
        self.memory.save(state)
        if validation.ok:
            return True
        if validation.recommendation.get("action") == "stop_non_retryable":
            state.final_status = "stopped"
            state.stop_reason = validation.recommendation.get("reason") or "non_retryable"
            state.termination = {
                "reason": state.stop_reason,
                "step_id": step.step_id,
                "failure_type": validation.recommendation.get("failure_type", ""),
            }
            state.add_trace("agent_stopped", step_id=step.step_id, skill_name=step.skill_name, detail=state.termination)
            return False
        self._record_failure_taxonomy(state, step, validation)
        return False

    def _run_dynamic_step(self, state: AgentState, step: PlanStep) -> bool:
        if step.action == "skip":
            self._record_skipped(state, step)
            return True
        if step.action == "reuse":
            self._record_reused(state, step)
            return True
        if step.action == "stop":
            self._record_stop(state, step)
            return False
        if step.action == "repair":
            self._record_repair_action(state, step)
        return self._execute_dynamic_once(state, step, allow_repair=True)

    def _execute_dynamic_once(self, state: AgentState, step: PlanStep, *, allow_repair: bool) -> bool:
        if state.loop_count >= self.max_loop_count:
            state.final_status = "stopped"
            state.stop_reason = "max_loop_count_reached"
            state.termination = {"reason": state.stop_reason, "step_id": step.step_id}
            state.add_trace("agent_stopped", step_id=step.step_id, detail=state.termination)
            self.memory.save(state)
            return False
        state.loop_count += 1
        record = self.executor.execute_step(state, step, attempt=1, fallback_used=False)
        if record.status == "succeeded":
            validation = self.critic.evaluate_step(state, step)
        else:
            validation = self.critic.evaluate_failure(state, step, record.error)
        record.validation = validation
        self._record_validation(state, step, validation)
        self.memory.save(state)
        if validation.ok:
            return True
        self._record_failure_taxonomy(state, step, validation)
        recommendation = validation.recommendation or {}
        if recommendation.get("action") == "stop_non_retryable":
            state.final_status = "stopped"
            state.stop_reason = recommendation.get("reason") or "non_retryable"
            state.termination = {
                "reason": state.stop_reason,
                "step_id": step.step_id,
                "failure_type": recommendation.get("failure_type", ""),
            }
            state.add_trace("agent_stopped", step_id=step.step_id, skill_name=step.skill_name, detail=state.termination)
            self.memory.save(state)
            return False
        if allow_repair:
            repair_plan = self.planner.build_plan(state, experiment_mode="dynamic_agent", critic_feedback=validation)
            self._validate_plan_skills(repair_plan)
            repair_payload = _model_dump(repair_plan)
            state.current_plan = repair_payload
            state.plan_history.append(repair_payload)
            state.add_trace(
                "plan_repaired",
                step_id=step.step_id,
                skill_name=step.skill_name,
                detail={"reason": repair_plan.reason, "recommendation": recommendation, "steps": [_model_dump(item) for item in repair_plan.steps]},
            )
            self.memory.save(state)
            return self._execute_repair_plan(state, repair_plan)
        return False

    def _execute_repair_plan(self, state: AgentState, repair_plan: AgentPlan) -> bool:
        if not repair_plan.steps:
            return False
        completed: Set[str] = set()
        ok = True
        for repair_step in self.executor.ordered_steps(repair_plan):
            missing_deps = [dep for dep in repair_step.depends_on if dep not in completed]
            if missing_deps:
                state.add_trace("repair_step_skipped", step_id=repair_step.step_id, detail={"missing_deps": missing_deps})
                return False
            state.planned.append(_planned_payload(repair_step, repair_plan.reason, repair_plan.experiment_mode))
            if repair_step.action == "skip":
                self._record_skipped(state, repair_step)
            elif repair_step.action == "reuse":
                self._record_reused(state, repair_step)
            elif repair_step.action == "stop":
                self._record_stop(state, repair_step)
                ok = False
            else:
                if repair_step.action == "repair":
                    self._record_repair_action(state, repair_step)
                ok = self._execute_dynamic_once(state, repair_step, allow_repair=False)
            if not ok:
                return False
            completed.add(repair_step.step_id)
        return True

    def _record_skipped(self, state: AgentState, step: PlanStep) -> None:
        payload = {"step_id": step.step_id, "skill_name": step.skill_name, "reason": step.params.get("skip_reason", "")}
        state.skipped.append(payload)
        state.execution_steps.append(
            ExecutionStepRecord(
                step_id=step.step_id,
                skill_name=step.skill_name,
                action=step.action,
                status="skipped",
                started_at=datetime.utcnow(),
                finished_at=datetime.utcnow(),
                output={},
            )
        )
        state.add_trace("step_skipped", step_id=step.step_id, skill_name=step.skill_name, detail=payload)
        self.memory.save(state)

    def _record_reused(self, state: AgentState, step: PlanStep) -> None:
        output = dict(step.params.get("reuse_output") or {})
        self.executor.apply_output(state, output)
        payload = {
            "step_id": step.step_id,
            "skill_name": step.skill_name,
            "source": step.params.get("reuse_source", ""),
            "cache_key": step.params.get("cache_key", ""),
            "output_keys": sorted(output),
        }
        state.reused.append(payload)
        if payload["source"] or payload["cache_key"]:
            cache_event = {"event": "cache_hit", **payload}
            state.cache_events.append(cache_event)
            state.add_trace("cache_hit", step_id=step.step_id, skill_name=step.skill_name, detail=cache_event)
        state.execution_steps.append(
            ExecutionStepRecord(
                step_id=step.step_id,
                skill_name=step.skill_name,
                action=step.action,
                status="reused",
                started_at=datetime.utcnow(),
                finished_at=datetime.utcnow(),
                output=output,
            )
        )
        validation = self.critic.evaluate_step(state, step)
        self._record_validation(state, step, validation)
        state.add_trace("step_reused", step_id=step.step_id, skill_name=step.skill_name, detail=payload)
        self.memory.save(state)

    def _record_stop(self, state: AgentState, step: PlanStep) -> None:
        reason = step.params.get("reason") or "stopped_by_plan"
        state.final_status = "stopped"
        state.stop_reason = reason
        state.termination = {"reason": reason, "step_id": step.step_id, "failure_type": step.params.get("failure_type", "")}
        state.add_trace("agent_stopped", step_id=step.step_id, skill_name=step.skill_name, detail=state.termination)
        self.memory.save(state)

    def _record_repair_action(self, state: AgentState, step: PlanStep) -> None:
        payload = {
            "step_id": step.step_id,
            "skill_name": step.skill_name,
            "params": dict(step.params or {}),
            "recommendation": step.params.get("repair_recommendation", {}),
        }
        state.repair_actions.append(payload)
        state.add_trace("step_repaired", step_id=step.step_id, skill_name=step.skill_name, detail=payload)

    def _record_validation(self, state: AgentState, step: PlanStep, validation: StepValidation) -> None:
        payload = {
            "step_id": step.step_id,
            "skill_name": step.skill_name,
            "ok": validation.ok,
            "missing_conditions": validation.missing_conditions,
            "recommendation": validation.recommendation,
        }
        state.validated.append(payload)
        state.add_trace("step_validated", step_id=step.step_id, skill_name=step.skill_name, detail=payload)
        state.add_trace("step_evaluated", step_id=step.step_id, skill_name=step.skill_name, detail=payload)

    def _record_failure_taxonomy(self, state: AgentState, step: PlanStep, validation: StepValidation) -> None:
        recommendation = validation.recommendation or {}
        payload = {
            "step_id": step.step_id,
            "skill_name": step.skill_name,
            "failure_type": recommendation.get("failure_type") or "validation_failure",
            "missing_conditions": validation.missing_conditions,
            "message": validation.message,
        }
        state.failure_taxonomy.append(payload)

    def _update_experiment_metrics(self, state: AgentState) -> None:
        duration_ms = max(0, int((state.updated_at - state.created_at).total_seconds() * 1000))
        tool_calls = len([step for step in state.execution_steps if step.action in {"execute", "repair"}])
        retries = sum(int(value or 0) for value in state.retry_counts.values()) + len(state.repair_actions)
        existing_cache_keys = {
            (
                str(item.get("step_id", "")),
                str(item.get("skill_name", "")),
                str(item.get("source", "")),
                str(item.get("cache_key", "")),
            )
            for item in state.cache_events
        }
        for event in state.agent_trace:
            if event.event != "cache_hit":
                continue
            detail = dict(event.detail or {})
            cache_event = {
                "event": "cache_hit",
                "step_id": event.step_id,
                "skill_name": event.skill_name,
                "source": detail.get("source", ""),
                "cache_key": detail.get("cache_key", ""),
                "detail": detail,
            }
            key = (
                cache_event["step_id"],
                cache_event["skill_name"],
                str(cache_event["source"]),
                str(cache_event["cache_key"]),
            )
            if key not in existing_cache_keys:
                state.cache_events.append(cache_event)
                existing_cache_keys.add(key)
        permission_compliance = True
        if not state.generation_allowed and state.target in {"image", "video"}:
            permission_compliance = state.paid_generation_calls == 0
        state.experiment_metrics = {
            "run_id": state.task_id,
            "mode": state.experiment_mode,
            "target": state.target,
            "planned_steps": len(state.planned),
            "executed_steps": len(state.executed),
            "skipped_steps": len(state.skipped),
            "reused_steps": len(state.reused),
            "repair_actions": len(state.repair_actions),
            "critic_results": len(state.validated),
            "cache_hits": len(state.cache_events),
            "tool_calls": tool_calls,
            "retries": retries,
            "duration_ms": duration_ms,
            "paid_generation_calls": state.paid_generation_calls,
            "final_status": state.final_status,
            "stop_reason": state.stop_reason,
            "permission_compliance": permission_compliance,
            "trace_complete": bool(state.agent_trace and state.plan_history and state.final_status in {"succeeded", "failed", "stopped"}),
        }

    def _validate_plan_skills(self, plan: AgentPlan) -> None:
        for step in plan.steps:
            if step.action in {"skip", "reuse", "stop"}:
                continue
            if not self.registry.has(step.skill_name):
                raise PermissionError(f"plan references unregistered skill: {step.skill_name}")
            if step.fallback_skill and not self.registry.has(step.fallback_skill):
                raise PermissionError(f"plan references unregistered fallback skill: {step.fallback_skill}")


def _model_dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return {}


def _planned_payload(step: PlanStep, reason: str, experiment_mode: str) -> Dict[str, Any]:
    return {
        "step_id": step.step_id,
        "skill_name": step.skill_name,
        "action": step.action,
        "depends_on": list(step.depends_on),
        "reason": reason,
        "experiment_mode": experiment_mode,
        "params": dict(step.params or {}),
    }
