from __future__ import annotations

from typing import Any, Dict, Sequence

from adapter.generic_hf import GenericHFAdapter
from spec.model_spec import IndexerSpec, ModelSpec


class DeepSeekV32Adapter(GenericHFAdapter):
    def match(self, architectures: Sequence[str], model_prefix: str) -> bool:
        return (architectures and architectures[0] == "DeepseekV32ForCausalLM")

    def parse(self, hf_config: Dict[str, Any]) -> ModelSpec:
        spec = super().parse(hf_config)
        # Enable Indexer spec (isolated here; must not leak to engine logic).
        indexer = IndexerSpec(
            enabled=True,
            index_head_dim=int(hf_config["index_head_dim"]),
            index_n_heads=int(hf_config["index_n_heads"]),
            index_topk=int(hf_config["index_topk"]),
            logits_dim=4096,
        )
        return ModelSpec(
            **{**spec.__dict__, "indexer": indexer}  # type: ignore[arg-type]
        )
