from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from flops.flops import gemm_flops
from hardware.gpu import gpu_map
from mfu.mfu import get_gemm_mfu, get_groupedgemm_decode_mfu, get_groupedgemm_prefill_mfu
from params.params import load_moe_weights_time

from .gemm import gemm_latency_s


@dataclass
class MoESimulator:
    """MoE simulator (pure, structured, no legacy runtime dependency).

    Output is intentionally aligned with the legacy engine expectations.
    """

    config: any
    use_fp8_gemm: bool

    def prefill(self, tokens: int, device_type: str, world_size: int) -> Dict[str, float]:
        t_moe_s, t_shared_s = self.prefill_moe(tokens, device_type, world_size)
        return {"routed_s": float(t_moe_s), "shared_s": float(t_shared_s)}

    def decode(self, bs: int, device_type: str, world_size: int) -> Dict[str, float]:
        t_moe_s, t_shared_s = self.decode_moe(bs, device_type, world_size)
        return {"routed_s": float(t_moe_s), "shared_s": float(t_shared_s)}

    # Legacy-compatible tuple-returning API
    def prefill_moe(self, tokens: int, device_type: str, world_size: int) -> Tuple[float, float]:
        cfg = self.config
        gpu = gpu_map[device_type]

        routed_experts_gflops = gemm_flops(1, cfg.hidden_size, cfg.intermediate_size)
        routed_experts_gflops *= tokens * cfg.num_experts_per_tok * 3.0 / 1e9

        moe_load_time = load_moe_weights_time(cfg, self.use_fp8_gemm, gpu, world_size) * cfg.num_hidden_layers

        # MFU for routed experts
        if cfg.is_moe:
            if getattr(cfg, "modelPrefix", "") == "DeepseekV3":
                routed_experts_mfu = max(
                    get_groupedgemm_prefill_mfu(cfg, tokens, device_type, world_size, self.use_fp8_gemm)
                )
                routed_ffn_mfu = get_gemm_mfu(
                    device_type,
                    tokens,
                    cfg.hidden_size,
                    cfg.intermediate_size * 2 // world_size,
                )
            else:
                routed_experts_mfu = max(
                    get_groupedgemm_prefill_mfu(cfg, tokens, device_type, world_size, self.use_fp8_gemm)
                )
                routed_ffn_mfu = 0.0
        else:
            routed_experts_mfu = get_gemm_mfu(
                device_type,
                tokens,
                cfg.hidden_size,
                cfg.intermediate_size * 2 // world_size,
            )
            routed_ffn_mfu = 0.0

        tflops = gpu.fp8_tflops if self.use_fp8_gemm else gpu.fp16_tflops

        if cfg.is_moe and getattr(cfg, "modelPrefix", "") == "DeepseekV3":
            routed_experts_latency = (
                routed_experts_gflops / (tflops * 1024 * routed_experts_mfu) * cfg.moe_layers
            ) + (
                routed_experts_gflops / (tflops * 1024 * routed_ffn_mfu) * cfg.ffn_layers
            )
        else:
            routed_experts_latency = routed_experts_gflops / (tflops * 1024 * routed_experts_mfu) * cfg.num_hidden_layers

        t = max(routed_experts_latency, moe_load_time)

        # Shared experts: modeled via 2 GEMMs (up/down)
        s_t = 0.0
        if getattr(cfg, "num_shared_experts", 0) > 0:
            up = gemm_latency_s(
                m=tokens,
                k=cfg.hidden_size,
                n=cfg.intermediate_size * 2 * cfg.num_shared_experts,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )
            down = gemm_latency_s(
                m=tokens,
                k=cfg.intermediate_size * cfg.num_shared_experts,
                n=cfg.hidden_size,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )
            s_t = (up + down) * cfg.num_hidden_layers

        return float(t), float(s_t)

    def decode_moe(self, bs: int, device_type: str, world_size: int) -> Tuple[float, float]:
        cfg = self.config
        gpu = gpu_map[device_type]

        routed_experts_gflops = gemm_flops(1, cfg.hidden_size, cfg.intermediate_size)
        routed_experts_gflops *= bs * cfg.num_experts_per_tok * 3.0 / 1e9

        moe_load_time = load_moe_weights_time(cfg, self.use_fp8_gemm, gpu, world_size) * cfg.num_hidden_layers

        if cfg.is_moe:
            if getattr(cfg, "modelPrefix", "") == "DeepseekV3":
                routed_experts_mfu = max(
                    get_groupedgemm_decode_mfu(cfg, bs, device_type, world_size, self.use_fp8_gemm)
                ) * cfg.moe_layers + get_gemm_mfu(
                    device_type,
                    bs,
                    cfg.hidden_size,
                    cfg.intermediate_size * 2 // world_size,
                ) * cfg.ffn_layers
            else:
                routed_experts_mfu = max(
                    get_groupedgemm_decode_mfu(cfg, bs, device_type, world_size, self.use_fp8_gemm)
                ) * cfg.num_hidden_layers
        else:
            routed_experts_mfu = get_gemm_mfu(
                device_type,
                bs,
                cfg.hidden_size,
                cfg.intermediate_size * 2 // world_size,
            ) * cfg.num_hidden_layers

        tflops = gpu.fp8_tflops if self.use_fp8_gemm else gpu.fp16_tflops
        routed_experts_latency = routed_experts_gflops / (tflops * 1024 * routed_experts_mfu)

        t = max(routed_experts_latency, moe_load_time)

        s_t = 0.0
        if getattr(cfg, "num_shared_experts", 0) > 0:
            up = gemm_latency_s(
                m=bs,
                k=cfg.hidden_size,
                n=cfg.intermediate_size * 2 * cfg.num_shared_experts,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )
            down = gemm_latency_s(
                m=bs,
                k=cfg.intermediate_size * cfg.num_shared_experts,
                n=cfg.hidden_size,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )
            s_t = (up + down) * cfg.num_hidden_layers

        return float(t), float(s_t)
