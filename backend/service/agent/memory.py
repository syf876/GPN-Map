import json
from threading import RLock
from pathlib import Path
from typing import Any, Dict, Optional

from backend.service.agent.schemas import AgentState


class JsonMemory:
    _lock = RLock()

    def __init__(self, path: str | Path = "artifacts/agent_memory/agent_tasks.json") -> None:
        self.path = Path(path)

    def save(self, state: AgentState) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._read_store()
            payload.setdefault("tasks", {})
            payload["tasks"][state.task_id] = self._model_dump(state)
            with self.path.open("w", encoding="utf-8") as file_handle:
                json.dump(payload, file_handle, ensure_ascii=False, indent=2, default=str)

    def load(self, task_id: str) -> Optional[AgentState]:
        with self._lock:
            task = self._read_store().get("tasks", {}).get(task_id)
        if not task:
            return None
        if hasattr(AgentState, "model_validate"):
            return AgentState.model_validate(task)
        return AgentState.parse_obj(task)

    def _read_store(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"tasks": {}}
        try:
            with self.path.open("r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
        except (json.JSONDecodeError, OSError):
            return {"tasks": {}}
        if not isinstance(data, dict):
            return {"tasks": {}}
        if not isinstance(data.get("tasks"), dict):
            data["tasks"] = {}
        return data

    def _model_dump(self, state: AgentState) -> Dict[str, Any]:
        if hasattr(state, "model_dump"):
            return state.model_dump()
        return state.dict()
