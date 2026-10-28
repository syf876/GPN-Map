"""Lightweight readiness metadata; never loads models or contacts providers."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(service):
    paths = ["backend/service/video_search/reality.py", "backend/reality_runtime.py",
             "backend/types/schema.py"]
    paths += (["frontend/server.py"] if service == "frontend" else
              ["api_round2.py", "backend/route/api.py"])
    return hashlib.sha256(str(ROOT).encode("utf-8") + b"".join((ROOT / p).read_bytes() for p in paths)).hexdigest()


def metadata(service):
    return {"service": service, "project": "hexi-poetry-map", "fingerprint": fingerprint(service)}


if __name__ == "__main__":
    import sys
    print(fingerprint(sys.argv[1]))
