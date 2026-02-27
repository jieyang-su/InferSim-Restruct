from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from kvcache.kvcache import get_kvcache_size


@dataclass
class KVCacheCalculator:
    """KV Cache module.

    - KV bytes-per-token depends on dtype (fp8 vs non-fp8).
    - Optional indexer KV is included when model config indicates indexer is enabled.
    """

    @staticmethod
    def bytes_per_token(config, use_fp8_kv: bool, is_indexer: bool = False) -> int:
        
        total = get_kvcache_size(config, use_fp8_kv, is_indexer)
        return int(total)

    @staticmethod
    def summary(config, use_fp8_kv: bool) -> Dict[str, int]:
        base = KVCacheCalculator.bytes_per_token(config, use_fp8_kv, False)
        indexer = 0
        if bool(getattr(config, "enable_indexer", False)) or bool(getattr(config, "is_indexer", False)):
            indexer = KVCacheCalculator.bytes_per_token(config, True, True)
        return {"kv_bytes_per_token": base, "indexer_kv_bytes_per_token": indexer, "total_kv_bytes_per_token": base + indexer}
