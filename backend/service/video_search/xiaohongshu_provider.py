from typing import List, Optional

from backend.types.schema import VideoCandidate, VideoSearchRequest

from .adapters.xiaohongshu_adapter import adapter
from .provider_format import format_adapter_results
from .provider_runtime import provider_runtime_mode


def search(
    req: VideoSearchRequest,
    query: str,
    mode: Optional[str] = None,
) -> List[VideoCandidate]:
    with provider_runtime_mode("XIAOHONGSHU_FIXTURE_MODE", mode) as adapter_mode:
        return format_adapter_results(adapter.search(req, query, mode=adapter_mode))
