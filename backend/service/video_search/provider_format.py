from typing import Iterable, List

from backend.service.video_search.adapters import AdapterVideo
from backend.types.schema import VideoCandidate


def format_adapter_results(items: Iterable[AdapterVideo]) -> List[VideoCandidate]:
    return [
        VideoCandidate(
            platform=item.platform,
            title=item.title,
            url=item.url,
            cover=item.cover,
            author=item.author,
            score=item.score,
            bvid=item.bvid,
            video_url=item.video_url,
            embed_url=item.embed_url,
            external_url=item.external_url or item.url,
            playback_type=item.playback_type,
            reason=item.reason,
            tags=list(item.tags),
            source_mode=item.source_mode,
            source_note=item.source_note,
            media_playability_score=item.media_playability_score,
            duplicate_key=item.duplicate_key,
        )
        for item in items
    ]
