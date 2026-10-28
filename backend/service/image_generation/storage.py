"""Keep immutable copies of provider images; never replace the curated assets."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

ROOT = Path(__file__).resolve().parents[3] / "frontend" / "static" / "generated"


def persist_image(source_url, request, response):
    with requests.get(source_url, timeout=120) as remote:
        remote.raise_for_status()
        body = remote.content
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        extension = ".png"
    elif body.startswith(b"\xff\xd8\xff"):
        extension = ".jpg"
    elif body.startswith(b"RIFF") and body[8:12] == b"WEBP":
        extension = ".webp"
    else:
        raise ValueError("provider returned an invalid image while saving result")
    folder = ROOT / uuid4().hex
    folder.mkdir(parents=True)
    filename = "image" + extension
    (folder / filename).write_bytes(body)
    url = f"/static/generated/{folder.name}/{filename}"
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": request.dict(),
        "provider": response.provider,
        "model": response.model,
        "image_url": url,
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return url
