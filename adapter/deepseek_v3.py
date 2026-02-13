from __future__ import annotations

from typing import Any, Dict, Sequence

from adapter.generic_hf import GenericHFAdapter
from spec.model_spec import ModelSpec, MoESpec


class DeepSeekV3Adapter(GenericHFAdapter):
    def match(self, architectures: Sequence[str], model_prefix: str) -> bool:
        return model_prefix == "DeepseekV3"

    def parse(self, hf_config: Dict[str, Any]) -> ModelSpec:
        spec = super().parse(hf_config)
        # DeepSeek V3: keep legacy split info used by current MoE simulator
        if spec.moe.enabled:
            # Create a new MoESpec with extra fields while keeping existing values.
            moe = MoESpec(
                enabled=True,
                num_routed_experts=spec.moe.num_routed_experts,
                num_experts_per_tok=spec.moe.num_experts_per_tok,
                intermediate_size=spec.moe.intermediate_size,
                num_shared_experts=spec.moe.num_shared_experts,
                intermediate_size_ffn=int(hf_config.get("intermediate_size")) if hf_config.get("intermediate_size") is not None else None,
                ffn_layers=3,
                moe_layers=58,
            )
            return ModelSpec(
                **{**spec.__dict__, "moe": moe}  # type: ignore[arg-type]
            )
        return spec
