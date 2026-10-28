from importlib import import_module


_EXPORTS = {
    "build_map_prediction": (
        "backend.service.orchestration.main_stage_pipeline",
        "build_map_prediction",
    ),
    "compose_main_stage": (
        "backend.service.orchestration.main_stage_pipeline",
        "compose_main_stage",
    ),
    "get_place_coordinates": (
        "backend.service.orchestration.main_stage_pipeline",
        "get_place_coordinates",
    ),
    "get_predictor": (
        "backend.service.orchestration.main_stage_pipeline",
        "get_predictor",
    ),
    "predict_poem": (
        "backend.service.orchestration.main_stage_pipeline",
        "predict_poem",
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
