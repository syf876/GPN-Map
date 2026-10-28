import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
DEMO_CASES_FILE = REPO_ROOT / "frontend" / "data" / "demo_video_cases.json"
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class DeploymentSecuritySettings:
    demo_mode: bool
    app_env: str
    enable_public_video_generation: bool
    enable_public_image_generation: bool
    allow_local_realtime_generation: bool

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_public_demo(self) -> bool:
        return self.demo_mode and not self.is_development


def deployment_security_settings() -> DeploymentSecuritySettings:
    return DeploymentSecuritySettings(
        demo_mode=_setting_bool("DEMO_MODE", True),
        app_env=_setting_value("APP_ENV", "production").lower(),
        enable_public_video_generation=_setting_bool("ENABLE_PUBLIC_VIDEO_GENERATION", False),
        enable_public_image_generation=_setting_bool("ENABLE_PUBLIC_IMAGE_GENERATION", False),
        allow_local_realtime_generation=_setting_bool("ALLOW_LOCAL_REALTIME_GENERATION", False),
    )


def generation_access_allowed(kind: str, supplied_token: str = "") -> bool:
    # Legacy flags, request-body tokens and development mode never grant access.
    from backend.admin_auth import SESSION, session_valid
    return session_valid(SESSION.get())


def normalized_video_path_allowed(request_path: str) -> bool:
    settings = deployment_security_settings()
    if generation_access_allowed("video"):
        return True
    normalized_path = _normalize_url_path(request_path)
    return normalized_path in demo_cached_video_paths()


def demo_cached_video_paths() -> set[str]:
    try:
        cases = json.loads(DEMO_CASES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(cases, list):
        return set()
    return {
        path
        for path in (_normalize_url_path(item.get("cached_video_url", "")) for item in cases if isinstance(item, dict))
        if path
    }


def _setting_bool(name: str, default: bool) -> bool:
    raw = _setting_value(name, "")
    if not raw:
        return default
    value = raw.lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return default


def _setting_value(name: str, default: str) -> str:
    process_value = os.environ.get(name)
    if process_value is not None:
        return str(process_value).strip()
    return _dotenv_values(ENV_FILE).get(name, default).strip()


def _dotenv_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
            value = value[1:-1]
        if key:
            result[key] = value
    return result


def _normalize_url_path(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return "/" + text.split("?", 1)[0].split("#", 1)[0].lstrip("/")
