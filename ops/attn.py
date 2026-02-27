from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from flops.flops import gemm_flops, gemm_flops_fp8
from hardware.gpu import gpu_map
from mfu.mfu import (
    get_attn_decode_mfu,
    get_attn_prefill_mfu,
    get_indexer_decode_mfu,
    get_indexer_prefill_mfu,
)


from .gemm import gemm_latency_s


def _dtype_str(use_fp8_gemm: bool) -> str:
    # Bench data uses bf16 as dtype label for core attention paths.
    return "bf16" if not use_fp8_gemm else "bf16"


def _kv_dtype_str(use_fp8_kv: bool) -> str:
    return "fp8" if use_fp8_kv else "bf16"


@dataclass
class MHAAttentionSim:
    config: Any
    use_fp8_gemm: bool
    use_fp8_kv: bool
    is_indexer: bool = False

    # ---------- FLOPs ----------
    def _attn_core_gflops(self, bs: int, kv_len: int) -> float:
        core = gemm_flops(bs, self.config.num_attention_heads * self.config.head_dim, kv_len) * 2
        return core / 1e9

    # ---------- Core ----------
    def prefill_attn_core(self, seq_len: int, kvcache_bytes: int, indexer_kvcache_bytes: int, device_type: str) -> float:
        attn_core_time = None

        gpu = gpu_map[device_type]
        attn_core_gflops = self._attn_core_gflops(1, seq_len)
        attn_core_mfu = get_attn_prefill_mfu(self.config, seq_len, device_type)
        attn_core_time = seq_len * attn_core_gflops / 1.8 / (gpu.fp16_tflops * 1024 * attn_core_mfu)

        # KV load time stays formula-based.
        gpu = gpu_map[device_type]
        kv_load_time = (
            kvcache_bytes
            * seq_len
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )
        return attn_core_time, kv_load_time, 0., 0.

    def decode_attn_core(self, bs: int, kv_len: int, q_len: int, kvcache_bytes: int, indexer_kvcache_bytes: int, device_type: str) -> float:
        avg_context_len = int((kv_len + 2 * q_len) / 2)

        attn_core_time = None

        gpu = gpu_map[device_type]
        attn_core_gflops = self._attn_core_gflops(1, avg_context_len)
        attn_core_mfu = get_attn_decode_mfu(self.config, bs, avg_context_len, device_type, self.use_fp8_kv)
        attn_core_time = bs * attn_core_gflops / (gpu.fp16_tflops * 1024 * attn_core_mfu)

        gpu = gpu_map[device_type]
        kv_load_time = (
            kvcache_bytes
            * avg_context_len
            * bs
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )
        return attn_core_time, kv_load_time, 0., 0.

    # ---------- Others ----------
    def decode_attn_others(self, bs: int, device_type: str) -> float:
        # QKV projection + O projection (same as legacy MHA)
        qkv_proj = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=(self.config.num_attention_heads + self.config.num_key_value_heads * 2) * self.config.head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        o_proj = gemm_latency_s(
            m=bs,
            k=self.config.num_attention_heads * self.config.head_dim,
            n=self.config.hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        return float(qkv_proj + o_proj)

    def prefill_attn_others(self, seq_len: int, device_type: str) -> float:
        # For MHA, legacy reuses decode_attn_others with bs=seq_len
        return float(self.decode_attn_others(seq_len, device_type))

    # Indexer hooks (MHA has none)
    def prefill_indexer_others(self, max_prefill_tokens: int, bs: int, device_type: str) -> float:
        return 0.0

    def decode_indexer_others(self, bs: int, device_type: str) -> float:
        return 0.0


@dataclass
class MLAAttentionSim(MHAAttentionSim):

    def _attn_core_gflops_absorb(self, bs: int, kv_len: int) -> float:
        core = gemm_flops(
            bs,
            self.config.num_attention_heads * (self.config.kv_lora_rank + self.config.qk_rope_head_dim),
            kv_len,
        ) + gemm_flops(bs, kv_len, self.config.num_attention_heads * self.config.kv_lora_rank)
        return core / 1e9

    def _attn_core_gflops_noabsorb(self, bs: int, kv_len: int) -> float:
        core = gemm_flops(
            bs,
            self.config.num_attention_heads * (self.config.qk_nope_head_dim + self.config.qk_rope_head_dim),
            kv_len,
        ) + gemm_flops(bs, kv_len, self.config.num_attention_heads * self.config.v_head_dim)
        return core / 1e9

    def decode_attn_core(self, bs: int, kv_len: int, q_len: int, kvcache_bytes: int, indexer_kvcache_bytes: int, device_type: str) -> float:
        avg_context_len = int((kv_len + 2 * q_len) / 2)
        gpu = gpu_map[device_type]
        attn_core_gflops = self._attn_core_gflops_absorb(1, avg_context_len)
        attn_core_mfu = get_attn_decode_mfu(self.config, bs, avg_context_len, device_type, self.use_fp8_kv)
        attn_core_time = bs * attn_core_gflops / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        kv_load_time = (
            kvcache_bytes
            * avg_context_len
            * bs
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )
        return attn_core_time, kv_load_time, 0., 0.

    def prefill_attn_core(self, seq_len: int, kvcache_bytes: int, indexer_kvcache_bytes: int, device_type: str) -> float:
        gpu = gpu_map[device_type]
        attn_core_gflops = self._attn_core_gflops_noabsorb(1, seq_len)
        attn_core_mfu = get_attn_prefill_mfu(self.config, seq_len, device_type)
        attn_core_time = seq_len * attn_core_gflops / 1.8 / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        kv_load_time = (
            kvcache_bytes
            * seq_len
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )
        return attn_core_time, kv_load_time, 0., 0.

    def decode_attn_others(self, bs: int, device_type: str) -> float:
        q_down = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=self.config.q_lora_rank,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        q_up = gemm_latency_s(
            m=bs,
            k=self.config.q_lora_rank,
            n=self.config.num_attention_heads * self.config.qk_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        kv_down = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=self.config.kv_lora_rank + self.config.qk_rope_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        bmm_q_wk = gemm_latency_s(
            m=bs,
            k=self.config.num_attention_heads * self.config.qk_nope_head_dim,
            n=self.config.kv_lora_rank,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        bmm_o_wv = gemm_latency_s(
            m=bs,
            k=self.config.num_attention_heads * self.config.kv_lora_rank,
            n=self.config.v_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        o_proj = gemm_latency_s(
            m=bs,
            k=self.config.num_attention_heads * self.config.v_head_dim,
            n=self.config.hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        return float(q_down + q_up + kv_down + bmm_q_wk + bmm_o_wv + o_proj)

    def prefill_attn_others(self, bs: int, device_type: str) -> float:
        q_down = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=self.config.q_lora_rank,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        q_up = gemm_latency_s(
            m=bs,
            k=self.config.q_lora_rank,
            n=self.config.num_attention_heads * self.config.qk_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        kv_down = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=self.config.kv_lora_rank + self.config.qk_rope_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        kv_up = gemm_latency_s(
            m=bs,
            k=self.config.kv_lora_rank,
            n=self.config.num_attention_heads * (self.config.v_head_dim + self.config.qk_nope_head_dim),
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        o_proj = gemm_latency_s(
            m=bs,
            k=self.config.num_attention_heads * self.config.v_head_dim,
            n=self.config.hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        return float(q_down + q_up + kv_down + kv_up + o_proj)


@dataclass
class MLADsaAttentionSim(MLAAttentionSim):

    is_indexer: bool = True

    def _dsa_logits_gflops(self, bs: int, seq_len: int) -> float:
        logits = gemm_flops_fp8(bs * seq_len, self.config.index_head_dim, self.config.index_n_heads)
        return logits / 1e9

    def decode_indexer_core(self, bs: int, kv_len: int, indexer_kvcache_bytes: int, device_type: str) -> float:
        gpu = gpu_map[device_type]
        gflops = self._dsa_logits_gflops(1, kv_len)
        mfu = get_indexer_decode_mfu(self.config, bs, kv_len, device_type, 1)
        core_time = bs * gflops / (gpu.fp8_tflops * 1024 * mfu)
        kv_load = (
            indexer_kvcache_bytes
            * kv_len
            * bs
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )
        return core_time, kv_load

    def decode_attn_core(self, bs: int, kv_len: int, q_len: int, kvcache_bytes: int, indexer_kvcache_bytes: int, device_type: str) -> float:
        avg_context_len = int((kv_len + 2 * q_len) / 2)
        gpu = gpu_map[device_type]
        attn_core_gflops = self._attn_core_gflops_absorb(1, min(avg_context_len, self.config.index_topk))
        attn_core_mfu = get_attn_decode_mfu(self.config, bs, min(avg_context_len, self.config.index_topk), device_type, self.use_fp8_kv)
        attn_core_time = bs * attn_core_gflops / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        kv_load_time = (
            kvcache_bytes
            * avg_context_len
            * bs
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )
        indexer_core, indexer_core_kv_load_time = self.decode_indexer_core(bs, avg_context_len, indexer_kvcache_bytes, device_type)
        return attn_core_time, kv_load_time, indexer_core, indexer_core_kv_load_time


    def prefill_indexer_core(self, seq_len, indexer_kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        indexer_core_gflops = self._dsa_logits_gflops(1, seq_len)
        indexer_core_mfu = get_indexer_prefill_mfu(self.config, seq_len, device_type)
        indexer_core_time = (
            seq_len * indexer_core_gflops / (gpu.fp8_tflops * 1024 * indexer_core_mfu)
        )
        indexer_kv_load_time = (
            indexer_kvcache_bytes
            * seq_len
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )
        return indexer_core_time, indexer_kv_load_time

    def prefill_attn_core(self, seq_len, kvcache_bytes, indexer_kvcache_bytes, device_type):
        gpu = gpu_map[device_type]
        # dsa prefill use mqa-mode mla
        attn_core_gflops = self._attn_core_gflops_absorb(1, min(seq_len, self.config.index_topk))
        attn_core_mfu = get_attn_prefill_mfu(self.config, min(seq_len, self.config.index_topk), device_type)
        attn_core_time = (
            min(seq_len, self.config.index_topk) * attn_core_gflops / 1.8 / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        )
        indexer_core_time, index_kv_load_time = self.prefill_indexer_core(seq_len, indexer_kvcache_bytes, device_type)
        kv_load_time = (
            kvcache_bytes
            * seq_len
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )
        return attn_core_time, kv_load_time, indexer_core_time, index_kv_load_time


    def prefill_indexer_others(self, bs: int, seq_len: int, device_type: str) -> float:
        Q_index_linear = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=seq_len * self.config.index_n_heads,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )

        KV_index_linear = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=seq_len * self.config.index_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )

        W_index_linear = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=seq_len * self.config.index_n_heads,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )

        return Q_index_linear + KV_index_linear + W_index_linear

    def decode_indexer_others(self, bs: int, seq_len: int, device_type: str) -> float:
        Q_index_linear = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=seq_len * self.config.index_n_heads,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )

        KV_index_linear = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=seq_len * self.config.index_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )

        W_index_linear = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=seq_len * self.config.index_n_heads,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )

        return Q_index_linear + KV_index_linear + W_index_linear 


@dataclass
class LinearAttentionSim:
    """GDN linear attention (ported from legacy layers/linear_attn.py, no printing)."""

    config: Any
    use_fp8_gemm: bool

    def prefill(self, isl: int, max_prefill_tokens: int, states_bytes: int, device_type: str) -> Dict[str, float]:
        from mfu.mfu import get_linear_attn_prefill_latency

        gpu = gpu_map[device_type]
        t_attn_core_us = get_linear_attn_prefill_latency(self.config, isl, device_type)
        t_states_load = states_bytes / self.config.num_linear_attn_layers / 1024 / 1024 / 1024 / gpu.mem_bw
        core_s = max(t_attn_core_us / 1e6, t_states_load)
        other_s = self._others(bs=isl, device_type=device_type)
        return {"core_s": float(core_s), "other_s": float(other_s)}

    def decode(self, bs: int, states_bytes: int, device_type: str) -> Dict[str, float]:
        from mfu.mfu import get_linear_attn_decode_latency

        gpu = gpu_map[device_type]
        t_attn_core_us = get_linear_attn_decode_latency(self.config, bs, device_type)
        t_states_load = states_bytes * bs / self.config.num_linear_attn_layers / 1024 / 1024 / 1024 / gpu.mem_bw
        core_s = max(t_attn_core_us / 1e6, t_states_load)
        other_s = self._others(bs=bs, device_type=device_type)
        return {"core_s": float(core_s), "other_s": float(other_s)}

    def _others(self, bs: int, device_type: str) -> float:
        key_dim = self.config.linear_num_key_heads * self.config.linear_key_head_dim
        value_dim = self.config.linear_num_value_heads * self.config.linear_value_head_dim
        projection_size_qkvz = key_dim * 2 + value_dim * 2
        projection_size_ba = self.config.linear_num_value_heads * 2

        qkvz_proj = gemm_latency_s(
            m=bs,
            k=self.config.hidden_size,
            n=projection_size_qkvz,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        qkvzba_proj = qkvz_proj * (projection_size_qkvz + projection_size_ba) / projection_size_qkvz

        o_proj = gemm_latency_s(
            m=bs,
            k=self.config.linear_num_value_heads * self.config.linear_value_head_dim,
            n=self.config.hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )
        return float(qkvzba_proj + o_proj)


class AttentionFactory:
    """Factory: choose MHA/GQA/MLA implementations based on config (no legacy runtime deps)."""

    @staticmethod
    def create(config: Any, use_fp8_gemm: bool, use_fp8_kv: bool):
        if getattr(config, "attn_type", "MHA/GQA") == "MHA/GQA":
            return MHAAttentionSim(config, use_fp8_gemm, use_fp8_kv)
        if getattr(config, "attn_type", "") == "MLA":
            if getattr(config, "modelName", "") == "DeepseekV32ForCausalLM":
                return MLADsaAttentionSim(config, use_fp8_gemm, use_fp8_kv)
            return MLAAttentionSim(config, use_fp8_gemm, use_fp8_kv)
        # Safe default
        return MHAAttentionSim(config, use_fp8_gemm, use_fp8_kv)

    @staticmethod
    def create_linear(config: Any, use_fp8_gemm: bool):
        return LinearAttentionSim(config, use_fp8_gemm)
