from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from comm.comm import Comm
from flops.flops import get_attn_gflops, get_moe_gflops
from hardware.gpu import gpu_map
from ops.kv import KVCacheCalculator
from ops.attn import AttentionFactory
from ops.moe import MoESimulator
from params.params import get_attn_params_size, get_expert_params_size
from params.params import get_linear_attn_params_size
from kvcache.kvcache import get_states_size


@dataclass(frozen=True)
class SimConfig:

    device_type: str
    world_size: int
    num_nodes: int
    max_prefill_tokens: int
    target_isl: int
    target_osl: int
    target_tgs: float
    target_tpot_ms: float
    decode_bs: Optional[int]
    use_fp8_gemm: bool
    use_fp8_kv: bool
    enable_deepep: bool
    enable_tbo: bool
    sm_ratio: float
    prefill_only: bool
    decode_only: bool

    @staticmethod
    def from_args(args) -> "SimConfig":
        return SimConfig(
            device_type=args.device_type,
            world_size=int(args.world_size),
            num_nodes=int(args.num_nodes),
            max_prefill_tokens=int(args.max_prefill_tokens),
            target_isl=int(args.target_isl),
            target_osl=int(args.target_osl),
            target_tgs=float(args.target_tgs),
            target_tpot_ms=float(args.target_tpot),
            decode_bs=args.decode_bs if args.decode_bs is None else int(args.decode_bs),
            use_fp8_gemm=bool(args.use_fp8_gemm),
            use_fp8_kv=bool(args.use_fp8_kv),
            enable_deepep=bool(args.enable_deepep),
            enable_tbo=bool(args.enable_tbo),
            sm_ratio=float(args.sm_ratio),
            prefill_only=bool(args.prefill_only),
            decode_only=bool(args.decode_only),
        )


class SimulationEngine:

    def __init__(self, sim: SimConfig, model_config):
        self.sim = sim
        self.config = model_config
        self.gpu = gpu_map[sim.device_type]

        # Populated during compute_* steps.
        self._kvcache_budget_gb: Optional[float] = None
        self._kvcache_bytes_per_token: Optional[int] = None
        self._indexer_kvcache_bytes_per_token: int = 0
        self._target_decode_bs: Optional[int] = None
        self._avg_context_len: Optional[int] = None

    # ---------------------------- Meta helpers ----------------------------
    def build_meta(self, config_path: str) -> Dict[str, Any]:
        return {
            "config_path": config_path,
            "device_type": self.sim.device_type,
            "world_size": self.sim.world_size,
            "num_nodes": self.sim.num_nodes,
            "max_prefill_tokens": self.sim.max_prefill_tokens,
            "decode_bs": self.sim.decode_bs,
            "target_tgs": self.sim.target_tgs,
            "target_tpot_ms": self.sim.target_tpot_ms,
            "target_isl": self.sim.target_isl,
            "target_osl": self.sim.target_osl,
            "use_fp8_gemm": bool(self.sim.use_fp8_gemm),
            "use_fp8_kv": bool(self.sim.use_fp8_kv),
            "enable_deepep": bool(self.sim.enable_deepep),
            "enable_tbo": bool(self.sim.enable_tbo),
            "sm_ratio": self.sim.sm_ratio,
            "prefill_only": bool(self.sim.prefill_only),
            "decode_only": bool(self.sim.decode_only),
        }

    # ---------------------------- Core outputs ----------------------------
    def compute_weights(self) -> Dict[str, Any]:
        # HybridLinear models have separate param accounting.
        if bool(getattr(self.config, "is_hybrid_linear", False)):
            full_attn_params_bytes = get_attn_params_size(self.config, self.sim.use_fp8_gemm)
            linear_attn_params_bytes = get_linear_attn_params_size(
                self.config, self.sim.use_fp8_gemm
            )
            expert_params_bytes = get_expert_params_size(self.config, self.sim.use_fp8_gemm)

            params_per_gpu_bytes = expert_params_bytes * (
                self.config.num_shared_experts
                + self.config.num_routed_experts / max(1, self.sim.world_size)
            )
            params_per_gpu_bytes *= self.config.num_hidden_layers
            params_per_gpu_bytes += self.config.num_full_attn_layers * full_attn_params_bytes
            params_per_gpu_bytes += self.config.num_linear_attn_layers * full_attn_params_bytes

            params_per_gpu_gb = params_per_gpu_bytes / 1024 / 1024 / 1024

            kvcache_budget_gb = self.gpu.mem - params_per_gpu_gb - 15 - 5
            self._kvcache_budget_gb = kvcache_budget_gb

            return {
                "full_attn_params_mb": full_attn_params_bytes / 1024 / 1024,
                "linear_attn_params_mb": linear_attn_params_bytes / 1024 / 1024,
                "expert_params_mb": expert_params_bytes / 1024 / 1024,
                "params_per_gpu_gb": params_per_gpu_gb,
                "kvcache_budget_gb": kvcache_budget_gb,
            }

        attn_params_bytes = get_attn_params_size(self.config, self.sim.use_fp8_gemm)
        expert_params_bytes = get_expert_params_size(self.config, self.sim.use_fp8_gemm)

        params_per_gpu_bytes = attn_params_bytes + expert_params_bytes * (
            self.config.num_shared_experts
            + self.config.num_routed_experts / max(1, self.sim.world_size)
        )
        params_per_gpu_gb = (
            params_per_gpu_bytes / 1024 / 1024 / 1024
        ) * self.config.num_hidden_layers

        # 15GB runtime + 5GB encoder are empirical constants from legacy code.
        kvcache_budget_gb = self.gpu.mem - params_per_gpu_gb - 15 - 5
        self._kvcache_budget_gb = kvcache_budget_gb

        return {
            "attn_params_mb": attn_params_bytes / 1024 / 1024,
            "expert_params_mb": expert_params_bytes / 1024 / 1024,
            "params_per_gpu_gb": params_per_gpu_gb,
            "kvcache_budget_gb": kvcache_budget_gb,
        }

    def compute_kvcache(self) -> Dict[str, Any]:
        if self._kvcache_budget_gb is None:
            raise RuntimeError("compute_weights() must be called before compute_kvcache().")

        context_len = self.sim.target_isl + self.sim.target_osl
        if self.sim.decode_bs is None:
            target_bs = math.ceil(self.sim.target_tgs * self.sim.target_tpot_ms / 1000)
        else:
            target_bs = int(self.sim.decode_bs)

        # HybridLinear models account cache per-request (KV for full-attn layers + states).
        if bool(getattr(self.config, "is_hybrid_linear", False)):
            target_per_req_cache_bytes = (
                self._kvcache_budget_gb * 1024 * 1024 * 1024 / target_bs
            )
            # get_kvcache_size returns total kv bytes per layer; legacy divides by num_hidden_layers.
            full_attn_kv_bytes = (
                KVCacheCalculator.bytes_per_token(self.config, self.sim.use_fp8_kv, False) / self.config.num_hidden_layers
            )
            full_attn_kv_bytes *= self.config.num_full_attn_layers * context_len
            states_bytes = get_states_size(self.config)
            current_total_cache_bytes = full_attn_kv_bytes + states_bytes

            # store per-token kv (for core kernels that still take per-token kv bytes)
            self._kvcache_bytes_per_token = int(full_attn_kv_bytes / max(1, context_len))
            self._indexer_kvcache_bytes_per_token = 0
            self._target_decode_bs = int(target_bs)

            return {
                "kvcache_budget_gb": self._kvcache_budget_gb,
                "input_seq_len": self.sim.target_isl,
                "output_seq_len": self.sim.target_osl,
                "context_len": context_len,
                "target_decode_bs": target_bs,
                "target_per_req_cache_mb": target_per_req_cache_bytes / 1024 / 1024,
                "current_full_attn_kv_mb": full_attn_kv_bytes / 1024 / 1024,
                "current_states_mb": states_bytes / 1024 / 1024,
                "current_total_cache_mb": current_total_cache_bytes / 1024 / 1024,
                "kvcache_bytes_per_token": self._kvcache_bytes_per_token,
                "states_bytes_per_req": int(states_bytes),
                "budget_ok": bool(current_total_cache_bytes <= target_per_req_cache_bytes),
            }

        target_per_token_kv_bytes = (
            self._kvcache_budget_gb * 1024 * 1024 * 1024 / target_bs / context_len
        )

        # KV bytes-per-token (dtype-aware) + optional indexer KV.
        kvcache_bytes = KVCacheCalculator.bytes_per_token(self.config, self.sim.use_fp8_kv, False)
        indexer_kvcache_bytes = 0
        if bool(getattr(self.config, "enable_indexer", False)) or getattr(self.config, "modelName", None) == "DeepseekV32ForCausalLM":
            indexer_kvcache_bytes = KVCacheCalculator.bytes_per_token(self.config, self.sim.use_fp8_kv, True)

        self._kvcache_bytes_per_token = int(kvcache_bytes)
        self._indexer_kvcache_bytes_per_token = int(indexer_kvcache_bytes)
        self._target_decode_bs = int(target_bs)

        current_per_token_kv_bytes = kvcache_bytes + indexer_kvcache_bytes
        return {
            "kvcache_budget_gb": self._kvcache_budget_gb,
            "input_seq_len": self.sim.target_isl,
            "output_seq_len": self.sim.target_osl,
            "context_len": context_len,
            "target_decode_bs": target_bs,
            "target_per_token_kv_kb": target_per_token_kv_bytes / 1024,
            "current_per_token_kv_kb": current_per_token_kv_bytes / 1024,
            "kvcache_bytes_per_token": int(kvcache_bytes),
            "indexer_kvcache_bytes_per_token": int(indexer_kvcache_bytes),
            "budget_ok": bool(current_per_token_kv_bytes <= target_per_token_kv_bytes),
        }

    def compute_flops(self) -> Dict[str, Any]:
        if bool(getattr(self.config, "is_hybrid_linear", False)):
            self._avg_context_len = int(self.sim.target_isl + self.sim.target_osl / 2)
            attn_core_gflops, other_gflops = get_attn_gflops(
                self.config, self._avg_context_len, absorb=True
            )
            moe_gflops = get_moe_gflops(self.config)
            return {
                "num_hidden_layers": self.config.num_hidden_layers,
                "avg_context_len": self._avg_context_len,
                "per_token_per_layer_gflops": {
                    "full_attn_core": attn_core_gflops,
                    "moe_ffn": moe_gflops,
                    "others": other_gflops,
                },
                "per_token_gflops": {
                    "full_attn_core": attn_core_gflops * self.config.num_full_attn_layers,
                    "moe_ffn": moe_gflops * self.config.num_hidden_layers,
                    "others": other_gflops * self.config.num_hidden_layers,
                    "total": (attn_core_gflops + moe_gflops + other_gflops)
                    * self.config.num_hidden_layers,
                },
            }

        self._avg_context_len = int(2 * self.sim.target_isl + self.sim.target_osl / 2)
        attn_core_gflops, other_gflops = get_attn_gflops(
            self.config, self._avg_context_len, absorb=True
        )
        moe_gflops = get_moe_gflops(self.config)

        return {
            "num_hidden_layers": self.config.num_hidden_layers,
            "avg_context_len": self._avg_context_len,
            "per_token_per_layer_gflops": {
                "attn_core": attn_core_gflops,
                "moe_ffn": moe_gflops,
                "others": other_gflops,
            },
            "per_token_gflops": {
                "attn_core": attn_core_gflops * self.config.num_hidden_layers,
                "moe_ffn": moe_gflops * self.config.num_hidden_layers,
                "others": other_gflops * self.config.num_hidden_layers,
                "total": (attn_core_gflops + moe_gflops + other_gflops)
                * self.config.num_hidden_layers,
            },
        }

    # ---------------------------- Stage simulation ----------------------------
    def _build_comm(self) -> Comm:
        return Comm(
            self.config,
            self.gpu,
            self.sim.world_size,
            self.sim.num_nodes,
            self.sim.enable_deepep,
        )

    def simulate_prefill(self) -> Dict[str, Any]:
        if self._kvcache_bytes_per_token is None:
            raise RuntimeError("compute_kvcache() must be called before simulate_prefill().")

        # HybridLinear path (no DeepEP/TBO today; keep identical to legacy formulas).
        if bool(getattr(self.config, "is_hybrid_linear", False)):
            context_len = self.sim.target_isl + self.sim.target_osl
            # compute_kvcache stores per-token kv bytes for full-attn layers; compute states per-req.
            states_bytes = get_states_size(self.config)

            full_attn = AttentionFactory.create(self.config, self.sim.use_fp8_gemm, self.sim.use_fp8_kv)
            t_full_core, t_full_kv_load_time, _, _ = full_attn.prefill_attn_core(
                self.sim.target_isl,
                self._kvcache_bytes_per_token,
                self._indexer_kvcache_bytes_per_token,
                self.sim.device_type,
            )
            t_full_other = full_attn.prefill_attn_others(self.sim.max_prefill_tokens, self.sim.device_type)
            t_full_core = max(t_full_core, t_full_kv_load_time)
            t_full_core *= math.ceil(self.sim.max_prefill_tokens / self.sim.target_isl)

            linear_attn = AttentionFactory.create_linear(self.config, self.sim.use_fp8_gemm)
            lin_pref = linear_attn.prefill(self.sim.target_isl, self.sim.max_prefill_tokens, states_bytes, self.sim.device_type)
            t_linear_core = lin_pref["core_s"]
            t_linear_core *= math.ceil(self.sim.max_prefill_tokens / self.sim.target_isl)
            t_linear_other = lin_pref["other_s"]

            moe = MoESimulator(self.config, self.sim.use_fp8_gemm)
            moe_pref = moe.prefill(self.sim.max_prefill_tokens, self.sim.device_type, self.sim.world_size)
            t_moe, t_shared = moe_pref["routed_s"], moe_pref["shared_s"]

            comm = self._build_comm()
            comm_before_s, comm_after_s = comm.stage_comm(self.sim.max_prefill_tokens, stage="prefill")

            ttft_s = (t_full_core + t_full_other) * self.config.num_full_attn_layers
            ttft_s += (t_linear_core + t_linear_other) * self.config.num_linear_attn_layers
            ttft_s += (comm_before_s + comm_after_s) * self.config.num_hidden_layers + t_moe + t_shared
            ttft_ms = ttft_s * 1000 + 30
            tgs = self.sim.max_prefill_tokens / (ttft_ms / 1000)

            return {
                "max_prefill_tokens": self.sim.max_prefill_tokens,
                "num_tokens": self.sim.max_prefill_tokens,
                "breakdown": {
                    "full_attn_ms": (t_full_core + t_full_other) * self.config.num_full_attn_layers * 1000,
                    "linear_attn_ms": (t_linear_core + t_linear_other) * self.config.num_linear_attn_layers * 1000,
                    "indexer_core_ms": None,
                    "moe_ms": t_moe * 1000,
                    "shared_expert_ms": t_shared * 1000,
                    "comm_before_us": comm_before_s * 1e6,
                    "comm_after_us": comm_after_s * 1e6,
                    "kv_load_ms": t_full_kv_load_time * self.config.num_full_attn_layers * 1000,
                    "indexer_kv_load_ms": None,
                },
                "enable_tbo": False,
                "sm_ratio": None,
                "ttft_ms": ttft_ms,
                "tgs_tok_per_gpu_s": tgs,
                "note": "hybrid_linear",
            }

        attn = AttentionFactory.create(self.config, self.sim.use_fp8_gemm, self.sim.use_fp8_kv)
        attn_core_s, attn_core_kv_load_time, indexer_core_s, indexer_core_kv_load_time = attn.prefill_attn_core(
            self.sim.target_isl,
            self._kvcache_bytes_per_token,
            self._indexer_kvcache_bytes_per_token,
            self.sim.device_type,
        )
        attn_other_s = attn.prefill_attn_others(self.sim.max_prefill_tokens, self.sim.device_type)
        if getattr(attn, "is_indexer", False):
            indexer_other_s = attn.prefill_indexer_others(
                self.sim.max_prefill_tokens, 1, self.sim.device_type
            )
        else:
            indexer_other_s = 0.0

        # prefill core time scales with number of chunks.
        origin_attn_core_s = attn_core_s * math.ceil(self.sim.max_prefill_tokens / self.sim.target_isl)
        core_s = max(attn_core_s, attn_core_kv_load_time) + max(indexer_core_s, indexer_core_kv_load_time)
        core_s *= math.ceil(self.sim.max_prefill_tokens / self.sim.target_isl)

        moe = MoESimulator(self.config, self.sim.use_fp8_gemm)
        moe_s, shared_expert_s = moe.prefill_moe(
            self.sim.max_prefill_tokens, self.sim.device_type, self.sim.world_size
        )

        comm = self._build_comm()
        comm_before_s, comm_after_s = comm.stage_comm(self.sim.max_prefill_tokens, stage="prefill")

        num_tokens = self.sim.max_prefill_tokens
        if self.sim.enable_tbo:
            # Legacy rule: overlap two batches.
            num_tokens *= 2
            ttft_s = max(
                (core_s + attn_other_s + indexer_other_s) / self.sim.sm_ratio,
                comm_before_s,
            )
            ttft_s += max(
                (core_s + attn_other_s + indexer_other_s) / self.sim.sm_ratio,
                comm_after_s,
            )
            ttft_s *= self.config.num_hidden_layers
            ttft_s += max(
                (moe_s + shared_expert_s) / self.sim.sm_ratio,
                comm_before_s * self.config.num_hidden_layers,
            )
            ttft_s += max(
                (moe_s + shared_expert_s) / self.sim.sm_ratio,
                comm_after_s * self.config.num_hidden_layers,
            )
        else:
            ttft_s = core_s
            ttft_s += attn_other_s + indexer_other_s
            ttft_s += comm_before_s + comm_after_s
            ttft_s *= self.config.num_hidden_layers
            ttft_s += moe_s + shared_expert_s

        ttft_ms = ttft_s * 1000 + 30  # scheduler constant
        tgs = num_tokens / (ttft_ms / 1000)

        return {
            "max_prefill_tokens": self.sim.max_prefill_tokens,
            "num_tokens": num_tokens,
            "breakdown": {
                "attn_core_ms": origin_attn_core_s * self.config.num_hidden_layers * 1000,
                "indexer_core_ms": indexer_core_s * self.config.num_hidden_layers * 1000,
                "attn_other_ms": (attn_other_s + indexer_other_s)
                * self.config.num_hidden_layers
                * 1000,
                "moe_ms": moe_s * 1000,
                "shared_expert_ms": shared_expert_s * 1000,
                "comm_before_us": comm_before_s * 1e6,
                "comm_after_us": comm_after_s * 1e6,
                "kv_load_ms": attn_core_kv_load_time * self.config.num_hidden_layers * 1000,
                "indexer_kv_load_ms": indexer_core_kv_load_time * self.config.num_hidden_layers * 1000,
            },
            "enable_tbo": bool(self.sim.enable_tbo),
            "sm_ratio": self.sim.sm_ratio if self.sim.enable_tbo else None,
            "ttft_ms": ttft_ms,
            "tgs_tok_per_gpu_s": tgs,
        }

    def simulate_decode(self) -> Dict[str, Any]:
        if self._kvcache_bytes_per_token is None or self._target_decode_bs is None:
            raise RuntimeError("compute_kvcache() must be called before simulate_decode().")

        # HybridLinear path (keep identical to legacy formulas).
        if bool(getattr(self.config, "is_hybrid_linear", False)):
            bs = int(self._target_decode_bs)
            context_len = int(self.sim.target_isl + self.sim.target_osl / 2)
            states_bytes = get_states_size(self.config)

            full_attn = AttentionFactory.create(self.config, self.sim.use_fp8_gemm, self.sim.use_fp8_kv)
            t_full_core,t_full_core_kv_load_time, _, _ = full_attn.decode_attn_core(
                bs, self.sim.target_osl, self.sim.target_isl, self._kvcache_bytes_per_token, self._indexer_kvcache_bytes_per_token, self.sim.device_type
            )
            t_full_core = max(t_full_core, t_full_core_kv_load_time)
            t_full_other = full_attn.decode_attn_others(bs, self.sim.device_type)

            linear_attn = AttentionFactory.create_linear(self.config, self.sim.use_fp8_gemm)
            t_linear_dict = linear_attn.decode(bs, states_bytes, self.sim.device_type)
            t_linear_core = t_linear_dict["core_s"]
            t_linear_other = t_linear_dict["other_s"]

            moe = MoESimulator(self.config, self.sim.use_fp8_gemm)
            t_moe, t_shared = moe.decode_moe(bs, self.sim.device_type, self.sim.world_size)

            comm = self._build_comm()
            comm_before_s, comm_after_s = comm.stage_comm(bs, stage="decode")

            tpot_s = (t_full_core + t_full_other) * self.config.num_full_attn_layers
            tpot_s += (t_linear_core + t_linear_other) * self.config.num_linear_attn_layers
            tpot_s += (comm_before_s + comm_after_s) * self.config.num_hidden_layers + t_moe + t_shared
            tpot_ms = tpot_s * 1000 + 5
            tgs = bs / tpot_ms * 1000

            return {
                "decode_bs": bs,
                "num_tokens": bs,
                "breakdown": {
                    "full_attn_ms": (t_full_core + t_full_other) * self.config.num_full_attn_layers * 1000,
                    "linear_attn_ms": (t_linear_core + t_linear_other) * self.config.num_linear_attn_layers * 1000,
                    "indexer_core_ms": None,
                    "moe_ms": t_moe * 1000,
                    "shared_expert_ms": t_shared * 1000,
                    "comm_before_us": comm_before_s * 1e6,
                    "comm_after_us": comm_after_s * 1e6,
                    "kv_load_ms": t_full_core_kv_load_time * self.config.num_full_attn_layers * 1000,
                    "indexer_kv_load_ms": None,
                },
                "enable_tbo": False,
                "tpot_ms": tpot_ms,
                "tgs_tok_per_gpu_s": tgs,
                "target_tpot_ms": self.sim.target_tpot_ms,
                "slo_violation": bool(tpot_ms > self.sim.target_tpot_ms),
                "note": "hybrid_linear",
            }

        bs = int(self._target_decode_bs)
        attn = AttentionFactory.create(self.config, self.sim.use_fp8_gemm, self.sim.use_fp8_kv)
        attn_core_s, attn_core_kv_load_time, indexer_core_s, indexer_core_kv_load_time = attn.decode_attn_core(
            bs,
            self.sim.target_osl,
            self.sim.target_isl,
            self._kvcache_bytes_per_token,
            self._indexer_kvcache_bytes_per_token,
            self.sim.device_type,
        )
        core_s = max(attn_core_s, attn_core_kv_load_time) + max(indexer_core_s, indexer_core_kv_load_time)
        attn_other_s = attn.decode_attn_others(bs, self.sim.device_type)
        if getattr(attn, "is_indexer", False):
            indexer_other_s = attn.decode_indexer_others(bs, 1, self.sim.device_type)
        else:
            indexer_other_s = 0.0

        moe = MoESimulator(self.config, self.sim.use_fp8_gemm)
        moe_s, shared_expert_s = moe.decode_moe(bs, self.sim.device_type, self.sim.world_size)

        comm = self._build_comm()
        comm_before_s, comm_after_s = comm.stage_comm(bs, stage="decode")

        num_tokens = bs
        if self.sim.enable_tbo:
            num_tokens *= 2
            temp_attn_core_s = core_s * self.config.num_hidden_layers
            temp_attn_other_s = (attn_other_s + indexer_other_s) * self.config.num_hidden_layers
            temp_comm_before_s = comm_before_s * self.config.num_hidden_layers
            temp_comm_after_s = comm_after_s * self.config.num_hidden_layers
            tpot_s = max(temp_attn_other_s + shared_expert_s, temp_comm_before_s) + max(
                temp_comm_after_s, temp_attn_core_s + moe_s
            )
            tpot_s *= 2
        else:
            tpot_s = core_s
            tpot_s += attn_other_s + indexer_other_s
            tpot_s += comm_before_s + comm_after_s
            tpot_s *= self.config.num_hidden_layers
            tpot_s += moe_s + shared_expert_s

        tpot_ms = tpot_s * 1000 + 5
        tgs = num_tokens / tpot_ms * 1000

        return {
            "decode_bs": bs,
            "num_tokens": num_tokens,
            "breakdown": {
                "attn_core_ms": attn_core_s * self.config.num_hidden_layers * 1000,
                "indexer_core_ms": indexer_core_s * self.config.num_hidden_layers * 1000,
                "attn_other_ms": (attn_other_s + indexer_other_s)
                * self.config.num_hidden_layers
                * 1000,
                "moe_ms": moe_s * 1000,
                "shared_expert_ms": shared_expert_s * 1000,
                "comm_before_us": comm_before_s * 1e6,
                "comm_after_us": comm_after_s * 1e6,
                "kv_load_ms": attn_core_kv_load_time * self.config.num_hidden_layers * 1000,
                "indexer_kv_load_ms": indexer_core_kv_load_time * self.config.num_hidden_layers * 1000,
            },
            "enable_tbo": bool(self.sim.enable_tbo),
            "tpot_ms": tpot_ms,
            "tgs_tok_per_gpu_s": tgs,
            "target_tpot_ms": self.sim.target_tpot_ms,
            "slo_violation": bool(tpot_ms > self.sim.target_tpot_ms),
        }

    # ---------------------------- End-to-end ----------------------------
    def run(self, config_path: str) -> Dict[str, Any]:
        weights = self.compute_weights()
        kvcache = self.compute_kvcache()
        flops = self.compute_flops()

        prefill = None
        if not self.sim.decode_only:
            prefill = self.simulate_prefill()

        decode = None
        if not self.sim.prefill_only:
            decode = self.simulate_decode()

        return {
            "meta": self.build_meta(config_path),
            "model": {
                "attn_type": getattr(self.config, "attn_type", None),
                "model_name": getattr(self.config, "modelName", None),
                "is_hybrid_linear": bool(getattr(self.config, "is_hybrid_linear", False)),
            },
            "weights": weights,
            "kvcache": kvcache,
            "flops": flops,
            "prefill": prefill,
            "decode": decode,
        }
