from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from flops.flops import get_attn_gflops, get_moe_gflops


@dataclass
class FLOPsCalculator:

    @staticmethod
    def attention_per_layer_per_token(config) -> float:
        return float(get_attn_gflops(config))

    @staticmethod
    def moe_per_layer_per_token(config, world_size: int) -> Dict[str, float]:
        moe, shared = get_moe_gflops(config, world_size)
        return {"moe_gflops": float(moe), "shared_gflops": float(shared)}

    @staticmethod
    def summarize(config, world_size: int) -> Dict[str, float]:
        attn = FLOPsCalculator.attention_per_layer_per_token(config)
        moe = FLOPsCalculator.moe_per_layer_per_token(config, world_size)
        total = attn + moe["moe_gflops"] + moe["shared_gflops"]
        return {"attn_gflops": attn, **moe, "total_gflops": total}
