import os
from pathlib import Path
from typing import Dict, Optional


def find_env_file(start: Path) -> Optional[Path]:
    start = start.resolve()
    for p in [start, *start.parents]:
        candidate = p / ".env"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_dotenv(path: Path, *, override: bool = False) -> Dict[str, str]:
    """
    Minimal .env loader (no dependency on python-dotenv).
    Returns parsed key-values; also sets os.environ.
    """
    out: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip("'").strip('"')
        out[key] = val
        if override or key not in os.environ:
            os.environ[key] = val
    return out

