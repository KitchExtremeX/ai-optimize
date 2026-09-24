"""Load a local .env file without third-party dependencies."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path


def load_env_file(
    path: Path,
    environ: MutableMapping[str, str] | None = None,
) -> bool:
    """Set missing keys from a ``.env`` file.

    Existing variables win. Blank lines and ``#`` comments are ignored.
    A leading ``export`` is accepted. Lines without ``=`` are skipped.
    Returns True when the file existed.
    """
    target = os.environ if environ is None else environ
    if not path.is_file():
        return False
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in target:
            continue
        target[key] = _unquote(value.strip())
    return True


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
