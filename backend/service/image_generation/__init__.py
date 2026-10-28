from importlib import import_module


_EXPORTS = {
    "generate_nano_banana_poetry_image": (
        "backend.service.image_generation.nano_banana",
        "generate_nano_banana_poetry_image",
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
