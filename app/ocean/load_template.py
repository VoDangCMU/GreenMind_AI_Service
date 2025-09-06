import json
from pathlib import Path
from typing import Dict, Any

_TPL_CACHE: Dict[str, Any] | None = None

def load_templates() -> Dict[str, Any]:
    global _TPL_CACHE
    if _TPL_CACHE is not None:
        return _TPL_CACHE

    path = Path(__file__).with_name("templates.json")

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy templates.json ở {path}")

    _TPL_CACHE = json.loads(path.read_text(encoding="utf-8"))
    return _TPL_CACHE
