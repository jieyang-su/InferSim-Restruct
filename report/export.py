"""
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable objects."""
    if is_dataclass(obj):
        return asdict(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    # Fallback: stringify unknown objects (e.g., numpy types)
    return str(obj)


def dump_result_json(path: str, payload: Dict[str, Any]) -> None:
    """Write payload to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
