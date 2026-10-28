import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
WAN_API_KEY_ENV_NAMES = (
    "DASHSCOPE_API_KEY",
    "QWEN_VL_API_KEY",
    "WAN_API_KEY",
    "WAN_I2V_API_KEY",
    "WANX_API_KEY",
    "WANXIANG_API_KEY",
    "TONGYI_WANXIANG_API_KEY",
    "QWEN_API_KEY",
    "ALIYUN_DASHSCOPE_API_KEY",
    "ALIBABA_DASHSCOPE_API_KEY",
)
VEO_API_KEY_ENV_NAMES = ("VEO_API_KEY",)


def env_value(name: str, default: str = "") -> str:
    value, _source = env_lookup(name)
    return value or str(default or "").strip()


def env_lookup(name: str) -> Tuple[str, str]:
    process_value = os.environ.get(name)
    if process_value is not None and str(process_value).strip():
        return str(process_value).strip(), f"process:{name}"

    env_map = dotenv_values()
    file_value = env_map.get(name)
    if file_value is not None and str(file_value).strip():
        return str(file_value).strip(), f"{ENV_FILE}:{name}"

    return "", ""


def first_env_value(names: Iterable[str], default: str = "") -> Tuple[str, str]:
    for name in names:
        value, source = env_lookup(name)
        if value:
            return value, source
    return str(default or "").strip(), "default" if default else ""


def env_presence(names: Iterable[str]) -> List[Dict[str, str | bool]]:
    results: List[Dict[str, str | bool]] = []
    for name in names:
        value, source = env_lookup(name)
        results.append({"name": name, "present": bool(value), "source": source})
    return results


def dotenv_values() -> Dict[str, str]:
    if not ENV_FILE.exists():
        return {}

    result: Dict[str, str] = {}
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
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
