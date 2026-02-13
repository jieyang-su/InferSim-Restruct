from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AttentionSpec:
    type: str  # "MHA/GQA" | "MLA"
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    # MLA specific (optional; kept for downstream use)
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    v_head_dim: Optional[int] = None


@dataclass(frozen=True)
class MoESpec:
    enabled: bool
    num_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    num_shared_experts: int = 0

    # DeepSeek V3 special split (optional)
    intermediate_size_ffn: Optional[int] = None
    ffn_layers: Optional[int] = None
    moe_layers: Optional[int] = None


@dataclass(frozen=True)
class HybridSpec:
    enabled: bool
    full_attention_interval: Optional[int] = None
    num_full_attn_layers: Optional[int] = None
    num_linear_attn_layers: Optional[int] = None

    linear_conv_kernel_dim: Optional[int] = None
    linear_key_head_dim: Optional[int] = None
    linear_num_key_heads: Optional[int] = None
    linear_value_head_dim: Optional[int] = None
    linear_num_value_heads: Optional[int] = None


@dataclass(frozen=True)
class IndexerSpec:
    enabled: bool
    index_head_dim: Optional[int] = None
    index_n_heads: Optional[int] = None
    index_topk: Optional[int] = None
    logits_dim: Optional[int] = None


@dataclass(frozen=True)
class ModelSpec:
    """A normalized, engine-friendly model spec.

    Keep this structure stable; adapters map from external configs (HF etc.)
    into this representation. Engine/layer modules should depend on ModelSpec,
    not raw HF fields.
    """
    model_prefix: str
    model_name: str

    hidden_size: int
    num_hidden_layers: int

    attn: AttentionSpec
    moe: MoESpec
    hybrid: HybridSpec
    indexer: IndexerSpec

    raw: Dict[str, Any]  # original config dict for debugging only
