from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type

from adapter.base import BaseAdapter
from adapter.generic_hf import GenericHFAdapter
from adapter.deepseek_v3 import DeepSeekV3Adapter
from adapter.deepseek_v32 import DeepSeekV32Adapter


_ADAPTERS: List[BaseAdapter] = [
    DeepSeekV32Adapter(),
    DeepSeekV3Adapter(),
    GenericHFAdapter(),
]


def select_adapter(hf_config: Dict[str, Any]) -> BaseAdapter:
    architectures = hf_config.get("architectures") or []
    arch0 = architectures[0] if architectures else ""
    model_prefix = arch0[:10] if arch0 else ""
    for ad in _ADAPTERS:
        if ad.match(architectures, model_prefix):
            return ad
    # fallback
    return GenericHFAdapter()
