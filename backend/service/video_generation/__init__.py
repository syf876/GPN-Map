from importlib import import_module


_EXPORTS = {
    "build_veo_prompt": ("backend.service.video_generation.prompt_builder", "build_veo_prompt"),
    "generate_jimeng_video": ("backend.service.video_generation.jimeng_client", "generate_jimeng_video"),
    "generate_poetry_video": ("backend.service.video_generation.dispatcher", "generate_poetry_video"),
    "generate_veo_video": ("backend.service.video_generation.veo_client", "generate_veo_video"),
    "generate_wan_video": ("backend.service.video_generation.wan_client", "generate_wan_video"),
    "normalize_generation_response": ("backend.service.video_generation.normalize", "normalize_generation_response"),
    "proxy_video_request": ("backend.service.video_generation.proxy", "proxy_video_request"),
    "serve_normalized_video": ("backend.service.video_generation.normalize", "serve_normalized_video"),
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
