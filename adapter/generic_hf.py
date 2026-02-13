from __future__ import annotations

from typing import Any, Dict, Sequence

from adapter.base import BaseAdapter
from spec.model_spec import (
    AttentionSpec,
    HybridSpec,
    IndexerSpec,
    MoESpec,
    ModelSpec,
)


def _get_moe_num_routed_experts(d: Dict[str, Any]) -> int:
    for k in ("num_routed_experts", "num_experts", "n_routed_experts"):
        if k in d:
            return int(d[k])
    return 1


def _get_num_shared_experts(d: Dict[str, Any]) -> int:
    if "num_shared_experts" in d:
        return int(d["num_shared_experts"])
    if "n_shared_experts" in d:
        return int(d["n_shared_experts"])
    return 0


class GenericHFAdapter(BaseAdapter):
    def match(self, architectures: Sequence[str], model_prefix: str) -> bool:
        # Always matches as last resort.
        return True

    def parse(self, hf_config: Dict[str, Any]) -> ModelSpec:
        architectures = hf_config.get("architectures") or []
        model_name = architectures[0] if architectures else "UnknownModel"
        model_prefix = model_name[:10]

        hidden_size = int(hf_config["hidden_size"])
        num_hidden_layers = int(hf_config["num_hidden_layers"])

        # Hybrid linear attention (optional)
        is_hybrid_linear = hf_config.get("full_attention_interval") is not None
        if is_hybrid_linear:
            fai = int(hf_config["full_attention_interval"])
            num_full = num_hidden_layers // fai
            num_linear = num_hidden_layers - num_full
            hybrid = HybridSpec(
                enabled=True,
                full_attention_interval=fai,
                num_full_attn_layers=num_full,
                num_linear_attn_layers=num_linear,
                linear_conv_kernel_dim=int(hf_config["linear_conv_kernel_dim"]),
                linear_key_head_dim=int(hf_config["linear_key_head_dim"]),
                linear_num_key_heads=int(hf_config["linear_num_key_heads"]),
                linear_value_head_dim=int(hf_config["linear_value_head_dim"]),
                linear_num_value_heads=int(hf_config["linear_num_value_heads"]),
            )
        else:
            hybrid = HybridSpec(enabled=False)

        # Attention type
        attn_type = "MLA" if "kv_lora_rank" in hf_config else "MHA/GQA"
        if attn_type == "MHA/GQA":
            num_attention_heads = int(hf_config["num_attention_heads"])
            num_key_value_heads = int(hf_config["num_key_value_heads"])
            head_dim = int(hf_config.get("head_dim") or (hidden_size // num_attention_heads))
            attn = AttentionSpec(
                type="MHA/GQA",
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
            )
        else:
            num_attention_heads = int(hf_config["num_attention_heads"])
            # MLA has no explicit num_kv_heads in some configs; keep it equal for safety.
            num_key_value_heads = int(hf_config.get("num_key_value_heads") or num_attention_heads)
            attn = AttentionSpec(
                type="MLA",
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=int(hf_config.get("qk_nope_head_dim", 0)) + int(hf_config.get("qk_rope_head_dim", 0)),
                q_lora_rank=int(hf_config.get("q_lora_rank")) if hf_config.get("q_lora_rank") is not None else None,
                qk_nope_head_dim=int(hf_config.get("qk_nope_head_dim")) if hf_config.get("qk_nope_head_dim") is not None else None,
                qk_rope_head_dim=int(hf_config.get("qk_rope_head_dim")) if hf_config.get("qk_rope_head_dim") is not None else None,
                kv_lora_rank=int(hf_config.get("kv_lora_rank")) if hf_config.get("kv_lora_rank") is not None else None,
                v_head_dim=int(hf_config.get("v_head_dim")) if hf_config.get("v_head_dim") is not None else None,
            )

        # MoE / FFN
        num_routed_experts = _get_moe_num_routed_experts(hf_config)
        is_moe = num_routed_experts > 1
        if is_moe:
            moe = MoESpec(
                enabled=True,
                num_routed_experts=num_routed_experts,
                num_experts_per_tok=int(hf_config["num_experts_per_tok"]),
                intermediate_size=int(hf_config["moe_intermediate_size"]),
                num_shared_experts=_get_num_shared_experts(hf_config),
            )
        else:
            moe = MoESpec(
                enabled=False,
                num_routed_experts=1,
                num_experts_per_tok=1,
                intermediate_size=int(hf_config["intermediate_size"]),
                num_shared_experts=0,
            )

        # Indexer (default: off; model-specific adapters may enable)
        indexer = IndexerSpec(enabled=False)

        return ModelSpec(
            model_prefix=model_prefix,
            model_name=model_name,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            attn=attn,
            moe=moe,
            hybrid=hybrid,
            indexer=indexer,
            raw=dict(hf_config),
        )
