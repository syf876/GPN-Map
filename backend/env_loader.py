import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
SAFE_ENV_NAMES_TO_REPORT = (
    "IMAGE_GENERATION_PROVIDER",
    "GRSAI_NANO_BANANA_BASE_URL",
    "GRSAI_NANO_BANANA_HOST",
    "GRSAI_NANO_BANANA_API_KEY",
    "GRSAI_API_KEY",
    "GRSAI_NANO_BANANA_MODEL",
    "VIDEO_GENERATION_PROVIDER",
    "DASHSCOPE_API_KEY",
    "QWEN_VL_API_KEY",
    "WAN_API_KEY",
    "WAN_I2V_API_KEY",
    "WANXIANG_API_KEY",
    "WANX_API_KEY",
    "TONGYI_WANXIANG_API_KEY",
    "QWEN_API_KEY",
    "GRSAI_VEO_API_KEY",
    "VEO_API_KEY",
    "WAN_API_BASE",
    "DASHSCOPE_API_BASE",
    "QWEN_VL_BASE_URL",
    "ALIYUN_DASHSCOPE_API_KEY",
    "ALIBABA_DASHSCOPE_API_KEY",
    "WAN_I2V_MODEL",
    "WAN_MODEL",
)


def load_root_env(env_file: Path | None = None) -> Dict[str, object]:
    path = env_file or ENV_FILE
    values = _dotenv_values(path)
    loaded_names: List[str] = []
    preserved_names: List[str] = []

    for name, value in values.items():
        clean_value = str(value or "").strip()
        if not name or not clean_value:
            continue
        process_value = os.environ.get(name)
        if process_value is None or not str(process_value).strip():
            os.environ[name] = clean_value
            loaded_names.append(name)
        else:
            preserved_names.append(name)

    checked_variables = _checked_variables(SAFE_ENV_NAMES_TO_REPORT, values, loaded_names, preserved_names)
    matched_secret_variables = [
        item["name"]
        for item in checked_variables
        if item["present"] and str(item["name"]).endswith(("API_KEY", "SECRET_ACCESS_KEY"))
    ]
    summary: Dict[str, object] = {
        "env_file": path.name,
        "env_file_exists": path.exists(),
        "loaded_count": len(loaded_names),
        "matched_secret_variables": matched_secret_variables,
    }
    print(f"[env_loader] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return summary


def _checked_variables(
    names: Iterable[str],
    file_values: Dict[str, str],
    loaded_names: List[str],
    preserved_names: List[str],
) -> List[Dict[str, object]]:
    loaded = set(loaded_names)
    preserved = set(preserved_names)
    results: List[Dict[str, object]] = []
    for name in names:
        process_present = bool(str(os.environ.get(name, "")).strip())
        file_present = bool(str(file_values.get(name, "")).strip())
        if name in loaded:
            source = ".env_loaded"
        elif name in preserved or process_present:
            source = "process"
        elif file_present:
            source = ".env"
        else:
            source = ""
        results.append(
            {
                "name": name,
                "present": process_present or file_present,
                "source": source,
            }
        )
    return results


def _dotenv_values(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    result: Dict[str, str] = {}
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
