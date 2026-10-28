from importlib import import_module


_EXPORTS = {
    "QwenQueryPlanner": (
        "backend.service.query_planner.qwen_planner",
        "QwenQueryPlanner",
    ),
    "build_query_plan": (
        "backend.service.query_planner.qwen_planner",
        "build_query_plan",
    ),
    "get_query_planner": (
        "backend.service.query_planner.qwen_planner",
        "get_query_planner",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value
