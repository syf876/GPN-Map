import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Protocol

from backend.types.schema import VideoSearchRequest


SearchMode = Literal["mock", "real"]


@dataclass(frozen=True)
class AdapterVideo:
    platform: str
    title: str
    url: str
    cover: str
    author: str
    score: float
    bvid: str = ""
    video_url: str = ""
    embed_url: str = ""
    external_url: str = ""
    playback_type: str = ""
    reason: str = ""
    tags: List[str] = field(default_factory=list)
    source_mode: str = ""
    source_note: str = ""
    media_playability_score: float = 0.0
    duplicate_key: str = ""


def get_video_search_mode(value: Optional[str] = None) -> SearchMode:
    raw = (value or os.getenv("VIDEO_SEARCH_MODE", "mock")).strip().lower()
    return "real" if raw == "real" else "mock"


class PlatformAdapter(Protocol):
    platform: str

    def search(
        self,
        req: VideoSearchRequest,
        query: str,
        mode: Optional[str] = None,
    ) -> List[AdapterVideo]:
        ...


class BasePlatformAdapter:
    platform = ""

    def search(
        self,
        req: VideoSearchRequest,
        query: str,
        mode: Optional[str] = None,
    ) -> List[AdapterVideo]:
        if get_video_search_mode(mode) == "real":
            return self.search_real(req, query)
        return self.search_mock(req, query)

    def search_mock(self, req: VideoSearchRequest, query: str) -> List[AdapterVideo]:
        raise NotImplementedError

    def search_real(self, req: VideoSearchRequest, query: str) -> List[AdapterVideo]:
        return []
