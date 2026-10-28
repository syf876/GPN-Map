import hashlib
from typing import Any, Dict
from urllib.parse import urlparse


def safe_url_preview(url: Any) -> Dict[str, Any]:
    text = str(url or "").strip()
    if not text:
        return {"present": False}
    try:
        parsed = urlparse(text)
        path_hash = hashlib.sha1(str(parsed.path or "").encode("utf-8")).hexdigest()[:12]
        return {
            "present": True,
            "scheme": parsed.scheme,
            "host": parsed.hostname or "",
            "path_hash": path_hash,
            "query_present": bool(parsed.query),
            "fragment_present": bool(parsed.fragment),
        }
    except Exception:
        return {
            "present": True,
            "scheme": "",
            "host": "",
            "path_hash": "",
            "query_present": False,
            "fragment_present": False,
        }
