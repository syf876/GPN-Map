from datetime import datetime
from typing import Callable, Dict, List, Set

from backend.service.agent.faults import maybe_inject_after_call, maybe_inject_before_call
from backend.service.agent.schemas import AgentPlan, AgentState, ExecutionStepRecord, PlanStep
from backend.service.agent.skills import SkillRegistry


class Executor:
    def __init__(self, registry: SkillRegistry, on_update: Callable[[AgentState], None] | None = None) -> None:
        self.registry = registry
        self.on_update = on_update

    def ordered_steps(self, plan: AgentPlan) -> List[PlanStep]:
        by_id: Dict[str, PlanStep] = {step.step_id: step for step in plan.steps}
        visited: Set[str] = set()
        visiting: Set[str] = set()
        ordered: List[PlanStep] = []

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            if step_id in visiting:
                raise ValueError(f"cyclic plan dependency at step: {step_id}")
            step = by_id.get(step_id)
            if step is None:
                raise ValueError(f"unknown plan step dependency: {step_id}")
            visiting.add(step_id)
            for dep in step.depends_on:
                visit(dep)
            visiting.remove(step_id)
            visited.add(step_id)
            ordered.append(step)

        for step in plan.steps:
            visit(step.step_id)
        return ordered

    def execute_step(
        self,
        state: AgentState,
        step: PlanStep,
        *,
        skill_name: str | None = None,
        attempt: int = 1,
        fallback_used: bool = False,
    ) -> ExecutionStepRecord:
        active_skill = skill_name or step.skill_name
        record = ExecutionStepRecord(
            step_id=step.step_id,
            skill_name=active_skill,
            action=step.action,
            status="running",
            attempt=attempt,
            fallback_used=fallback_used,
            started_at=datetime.utcnow(),
        )
        state.execution_steps.append(record)
        if self.on_update:
            self.on_update(state)
        state.add_trace(
            "skill_call_started",
            step_id=step.step_id,
            skill_name=active_skill,
            detail={"attempt": attempt, "fallback_used": fallback_used},
        )
        try:
            result = maybe_inject_before_call(state, step, skill_name=active_skill, attempt=attempt)
            if result is None:
                result = self.registry.call(active_skill, state, step.params)
            if not result.ok:
                raise RuntimeError(result.error or result.message or "skill returned ok=false")
            result.output = maybe_inject_after_call(state, step, result.output, skill_name=active_skill, attempt=attempt)
            self._apply_output(state, result.output)
            state.executed.append(
                {
                    "step_id": step.step_id,
                    "skill_name": active_skill,
                    "attempt": attempt,
                    "fallback_used": fallback_used,
                    "output_keys": sorted(result.output),
                }
            )
            record.status = "succeeded"
            record.output = result.output
            state.add_trace(
                "skill_call_finished",
                step_id=step.step_id,
                skill_name=active_skill,
                detail={"attempt": attempt, "message": result.message, "output_keys": sorted(result.output)},
            )
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            state.add_error(str(exc), step_id=step.step_id, skill_name=active_skill)
            state.add_trace(
                "skill_call_failed",
                step_id=step.step_id,
                skill_name=active_skill,
                detail={"attempt": attempt, "error": str(exc)},
            )
        finally:
            record.finished_at = datetime.utcnow()
            state.updated_at = record.finished_at
        return record

    def _apply_output(self, state: AgentState, output: Dict[str, object]) -> None:
        for key, value in output.items():
            if hasattr(state, key):
                setattr(state, key, value)
            else:
                state.add_trace("ignored_skill_output", detail={"key": key})

    def apply_output(self, state: AgentState, output: Dict[str, object]) -> None:
        self._apply_output(state, output)
