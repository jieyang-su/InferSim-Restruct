import json
from typing import Any, Dict, Optional

from adapter.registry import select_adapter
from spec.model_spec import ModelSpec


class ModelConfig:
    """Backward-compatible wrapper around the new ModelSpec.

    NOTE: New code should depend on ModelSpec directly (via adapter layer).
    This class exists to keep the current simulator implementation working
    while we incrementally move modules over to ModelSpec.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            d: Dict[str, Any] = json.load(f)

        adapter = select_adapter(d)
        self.spec: ModelSpec = adapter.parse(d)

        # ---- Legacy attribute surface (used by existing modules) ----
        self.modelPrefix = self.spec.model_prefix
        self.modelName = self.spec.model_name

        self.hidden_size = self.spec.hidden_size
        self.num_hidden_layers = self.spec.num_hidden_layers

        # Hybrid linear attn
        self.is_hybrid_linear = self.spec.hybrid.enabled
        if self.is_hybrid_linear:
            self.num_full_attn_layers = self.spec.hybrid.num_full_attn_layers
            self.num_linear_attn_layers = self.spec.hybrid.num_linear_attn_layers
            self.linear_conv_kernel_dim = self.spec.hybrid.linear_conv_kernel_dim
            self.linear_key_head_dim = self.spec.hybrid.linear_key_head_dim
            self.linear_num_key_heads = self.spec.hybrid.linear_num_key_heads
            self.linear_value_head_dim = self.spec.hybrid.linear_value_head_dim
            self.linear_num_value_heads = self.spec.hybrid.linear_num_value_heads

        # Indexer (DeepSeek V3.2)
        if self.spec.indexer.enabled:
            self.index_head_dim = self.spec.indexer.index_head_dim
            self.index_n_heads = self.spec.indexer.index_n_heads
            self.index_topk = self.spec.indexer.index_topk
            self.logits_dim = self.spec.indexer.logits_dim

        # Attention
        self.attn_type = self.spec.attn.type
        self.num_attention_heads = self.spec.attn.num_attention_heads
        self.num_key_value_heads = self.spec.attn.num_key_value_heads
        # for MLA, this "head_dim" is only used in a few places; keep a best-effort value
        self.head_dim = self.spec.attn.head_dim

        if self.attn_type == "MLA":
            self.q_lora_rank = self.spec.attn.q_lora_rank
            self.qk_nope_head_dim = self.spec.attn.qk_nope_head_dim
            self.qk_rope_head_dim = self.spec.attn.qk_rope_head_dim
            self.kv_lora_rank = self.spec.attn.kv_lora_rank
            self.v_head_dim = self.spec.attn.v_head_dim
            self.index_topk = getattr(self, "index_topk", None)
            if self.qk_nope_head_dim is not None and self.qk_rope_head_dim is not None:
                self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # FFN / MoE
        self.is_moe = self.spec.moe.enabled
        self.num_routed_experts = self.spec.moe.num_routed_experts
        self.num_experts_per_tok = self.spec.moe.num_experts_per_tok
        self.intermediate_size = self.spec.moe.intermediate_size
        self.num_shared_experts = self.spec.moe.num_shared_experts

        # legacy DeepSeek V3 fields
        if self.spec.moe.intermediate_size_ffn is not None:
            self.intermediate_size_ffn = self.spec.moe.intermediate_size_ffn
        if self.spec.moe.ffn_layers is not None:
            self.ffn_layers = self.spec.moe.ffn_layers
        if self.spec.moe.moe_layers is not None:
            self.moe_layers = self.spec.moe.moe_layers
