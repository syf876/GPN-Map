import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.deployment_security import (
    deployment_security_settings,
    generation_access_allowed,
    normalized_video_path_allowed,
)

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"
MAP_CONFIG = ROOT / "map-config.js"
AMAP_ADAPTER = ROOT / "amap-adapter.js"
VISUAL_RULES = ROOT / "visual-rules.js"
STATIC_ROOT = ROOT / "static"
ENV_FILE = ROOT.parent / ".env"
API_BASE = os.environ.get("HEXI_API_BASE", "http://127.0.0.1:8000").rstrip("/")
from backend.reality_runtime import metadata as reality_runtime_metadata
REALITY_RUNTIME = {**reality_runtime_metadata("frontend"), "api_base": API_BASE}
AMAP_KEY_ENV_NAMES = (
    "VITE_AMAP_KEY",
    "AMAP_KEY",
    "AMAP_API_KEY",
    "GAODE_MAP_KEY",
    "AMAP_WEB_API_KEY",
    "GAODE_API_KEY",
    "VITE_GAODE_MAP_KEY",
)


class DemoHandler(BaseHTTPRequestHandler):
    API_TIMEOUT_SECONDS = 120
    IMAGE_GENERATION_TIMEOUT_SECONDS = 900
    VIDEO_GENERATION_TIMEOUT_SECONDS = 900

    def do_HEAD(self):
        request_path = unquote(urlparse(self.path).path)
        if request_path.startswith("/static/"):
            self._serve_static_file(request_path, head_only=True)
            return
        if request_path == "/api/video-generation/proxy":
            self._proxy_passthrough(self.path, method="HEAD")
            return
        if request_path.startswith("/api/video-generation/normalized/"):
            self._proxy_passthrough(self.path, method="HEAD")
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        request_path = unquote(urlparse(self.path).path)
        if request_path == "/api/auth/session" or request_path.startswith("/api/demo/"):
            self._proxy_passthrough(self.path, method="GET")
            return
        if request_path == "/api/reality/runtime":
            self._send_json(200, REALITY_RUNTIME)
            return
        if request_path in ("/", "/index.html"):
            body = INDEX.read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
            return
        if request_path == "/map-config.js":
            self._send(200, self._map_config_body(), "application/javascript; charset=utf-8")
            return
        if request_path == "/amap-js.js":
            self._send_json(410, {"detail": "retired map service"})
            return
        if request_path == "/amap-adapter.js" and AMAP_ADAPTER.exists():
            self._send(200, AMAP_ADAPTER.read_bytes(), "application/javascript; charset=utf-8")
            return
        if request_path == "/visual-rules.js" and VISUAL_RULES.exists():
            self._send(200, VISUAL_RULES.read_bytes(), "application/javascript; charset=utf-8")
            return
        if request_path == "/demo-config.js":
            self._send(200, self._demo_config_body(), "application/javascript; charset=utf-8")
            return
        if request_path.startswith("/static/"):
            self._serve_static_file(request_path)
            return
        if request_path == "/api/video-generation/proxy":
            self._proxy_passthrough(self.path, method="GET")
            return
        if request_path.startswith("/api/video-generation/normalized/"):
            self._proxy_passthrough(self.path, method="GET")
            return
        if request_path in {"/api/hexi-poetry/filters", "/api/hexi-poetry/poems", "/api/hexi-poetry/place-heatmap"}:
            self._proxy_passthrough(self.path, method="GET")
            return
        if request_path.startswith("/api/agent/tasks"):
            self._proxy_passthrough(self.path, method="GET")
            return
        if "hexi_corridor.geojson" in request_path:
            target = ROOT / "data" / "hexi_corridor.geojson"
            if target.exists():
                self._send(200, target.read_bytes(), "application/geo+json; charset=utf-8")
                return
        if request_path.startswith("/data/"):
            target = (ROOT / request_path.lstrip("/")).resolve()
            try:
                target.relative_to((ROOT / "data").resolve())
            except ValueError:
                target = None
            if target and target.exists() and target.is_file():
                content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self._send(200, target.read_bytes(), content_type)
                return
        self._send_json(404, {"error": "not found", "path": request_path})

    def do_POST(self):
        if self.path in {"/api/auth/login", "/api/auth/logout"}:
            self._proxy(self.path)
            return
        if self.path == "/api/reality/search":
            self._proxy("/api/reality/search")
            return
        if self.path == "/api/agent/tasks":
            self._proxy("/api/agent/tasks")
            return
        if self.path == "/api/main-stage/compose":
            self._proxy("/api/main-stage/compose")
            return
        if self.path == "/api/poetry-image/nano-banana":
            self._proxy("/api/poetry-image/nano-banana")
            return
        if self.path == "/api/video-generation/poetry-video":
            self._proxy("/api/video-generation/poetry-video")
            return
        if self.path == "/api/video-generation/veo":
            self._proxy("/api/video-generation/veo")
            return
        self._send_json(404, {"error": "frontend only exposes /api/reality/search, /api/reality/runtime, /api/agent/tasks, /api/main-stage/compose, /api/poetry-image/nano-banana, /api/video-generation/poetry-video, /api/video-generation/proxy, /api/video-generation/normalized/* and /api/video-generation/veo"})

    def _proxy(self, upstream_path: str):
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        headers = {"Content-Type": "application/json"}
        for key in ("Cookie", "X-CSRF-Token", "Origin"):
            if self.headers.get(key): headers[key] = self.headers[key]
        headers["X-Forwarded-For"] = self.client_address[0]
        req = Request(
            API_BASE + upstream_path,
            data=payload,
            headers=headers,
            method="POST",
        )
        timeout_seconds = (
            self.VIDEO_GENERATION_TIMEOUT_SECONDS
            if upstream_path in {"/api/video-generation/poetry-video", "/api/video-generation/veo"}
            else self.IMAGE_GENERATION_TIMEOUT_SECONDS
            if upstream_path == "/api/poetry-image/nano-banana"
            else self.API_TIMEOUT_SECONDS
        )
        try:
            with urlopen(req, timeout=timeout_seconds) as res:
                self._send(res.status, res.read(), "application/json; charset=utf-8", dict(res.headers))
        except HTTPError as exc:
            body = exc.read()
            if body:
                self._send(exc.code, body, "application/json; charset=utf-8", dict(exc.headers))
            else:
                self._send_json(exc.code, {"error": f"backend returned {exc.code} with empty body", "upstream": API_BASE + upstream_path})
        except URLError as exc:
            self._send_json(502, {"error": f"api unavailable: {exc.reason}"})
        except Exception as exc:
            self._send_json(502, {"error": f"api proxy failed: {exc}", "upstream": API_BASE + upstream_path})

    def _proxy_passthrough(self, upstream_path: str, method: str = "GET"):
        headers = {key: self.headers[key] for key in ("Cookie", "Origin") if self.headers.get(key)}
        range_header = self.headers.get("Range")
        if range_header:
            headers["Range"] = range_header
        req = Request(API_BASE + upstream_path, headers=headers, method=method)
        try:
            with urlopen(req, timeout=120) as res:
                self.send_response(res.status)
                for key, value in res.headers.items():
                    if key.lower() in {"content-type", "content-length", "accept-ranges", "content-range", "cache-control", "etag", "last-modified", "access-control-allow-origin", "set-cookie", "retry-after"}:
                        self.send_header(key, value)
                self.end_headers()
                if method != "HEAD":
                    while True:
                        chunk = res.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
        except HTTPError as exc:
            self.send_response(exc.code)
            for key, value in exc.headers.items():
                if key.lower() in {"content-type", "content-length", "accept-ranges", "content-range", "cache-control", "etag", "last-modified", "access-control-allow-origin", "set-cookie", "retry-after"}:
                    self.send_header(key, value)
            self.end_headers()
            if method != "HEAD":
                while True:
                    chunk = exc.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except URLError as exc:
            self._send_json(502, {"error": f"api unavailable: {exc.reason}"})

    def _send_json(self, status: int, payload: dict):
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _map_config_body(self) -> bytes:
        source, _key = _resolve_amap_key()
        payload = {
            "amapKeySource": source,
            "amapMaxZoom": 20,
        }
        body = "window.HEXI_MAP_CONFIG = " + json.dumps(payload, ensure_ascii=False) + ";\n"
        return body.encode("utf-8")

    def _demo_config_body(self) -> bytes:
        settings = deployment_security_settings()
        payload = {
            "publicDemoMode": settings.demo_mode,
            "isLocalDevelopment": False,
            "casesUrl": "/api/demo/cases",
        }
        body = "window.HEXI_DEMO_CONFIG = " + json.dumps(payload, ensure_ascii=False) + ";\n"
        return body.encode("utf-8")

    def _serve_static_file(self, request_path: str, head_only: bool = False):
        target = (ROOT / request_path.lstrip("/")).resolve()
        try:
            target.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self._send_json(403, {"error": "invalid static path"})
            return
        if target.is_relative_to((STATIC_ROOT / "generated").resolve()):
            canonical = "/" + target.relative_to(ROOT.resolve()).as_posix()
            self._proxy_passthrough(canonical, method="HEAD" if head_only else "GET")
            return
        if not target.exists() or not target.is_file():
            self._send_json(404, {"error": "static file not found", "path": request_path})
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        size = target.stat().st_size
        range_header = str(self.headers.get("Range") or "").strip()
        if range_header.startswith("bytes="):
            start_text, _, end_text = range_header[6:].partition("-")
            try:
                if not start_text:
                    suffix = int(end_text)
                    if suffix <= 0:
                        raise ValueError("invalid suffix")
                    start, end = max(0, size - suffix), size - 1
                else:
                    start = int(start_text)
                    end = min(size - 1, int(end_text) if end_text else size - 1)
                if start < 0 or start > end:
                    raise ValueError("invalid range")
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if not head_only:
                with target.open("rb") as file_handle:
                    file_handle.seek(start)
                    self.wfile.write(file_handle.read(length))
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if not head_only:
            self.wfile.write(target.read_bytes())

    def _amap_js_body(self) -> bytes:
        _source, key = _resolve_amap_key()
        if not key:
            return b'window.HEXI_AMAP_LOAD_ERROR = "missing_key";\n'
        amap_url = (
            "https://webapi.amap.com/maps?v=2.0"
            f"&key={key}"
            "&plugin=AMap.ToolBar,AMap.Scale"
        )
        try:
            with urlopen(amap_url, timeout=30) as res:
                return res.read()
        except Exception:
            return b'window.HEXI_AMAP_LOAD_ERROR = "load_failed";\n'

    def _send(self, status: int, body: bytes, content_type: str, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            if key.lower() in {"set-cookie", "cache-control", "retry-after"}:
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


def _dotenv_values(path: Path) -> dict:
    if not path.exists():
        return {}
    result = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        name, value = line.split("=", 1)
        name = name.strip().lstrip("\ufeff")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        if name:
            result[name] = value
    return result


def _resolve_amap_key() -> tuple[str, str]:
    env_values = _dotenv_values(ENV_FILE)
    for name in AMAP_KEY_ENV_NAMES:
        value = str(os.environ.get(name) or env_values.get(name) or "").strip()
        if value:
            return name, value
    return "", ""


def main():
    port = int(os.environ.get("HEXI_FRONTEND_PORT", "8080"))
    server = ThreadingHTTPServer(("127.0.0.1", port), DemoHandler)
    print(f"Frontend demo: http://127.0.0.1:{port}")
    print(f"Proxy API base: {API_BASE}")
    server.serve_forever()


if __name__ == "__main__":
    main()
