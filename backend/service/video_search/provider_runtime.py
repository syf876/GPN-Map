import os
from contextlib import contextmanager
from typing import Iterator, Literal, Optional


ProviderRunMode = Literal["mock", "real", "fixture"]


def normalize_provider_run_mode(mode: Optional[str]) -> Optional[ProviderRunMode]:
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if normalized in {"mock", "real", "fixture"}:
        return normalized
    raise ValueError(f"unsupported provider mode: {mode}")


@contextmanager
def provider_runtime_mode(
    fixture_env_name: str,
    mode: Optional[str],
) -> Iterator[Optional[str]]:
    normalized = normalize_provider_run_mode(mode)
    if normalized is None:
        yield None
        return

    if normalized == "mock":
        yield "mock"
        return

    previous = os.environ.get(fixture_env_name)
    os.environ[fixture_env_name] = "1" if normalized == "fixture" else "0"
    try:
        yield "real"
    finally:
        if previous is None:
            os.environ.pop(fixture_env_name, None)
        else:
            os.environ[fixture_env_name] = previous
