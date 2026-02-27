from __future__ import annotations

from flops.flops import gemm_flops
from hardware.gpu import gpu_map
from mfu.mfu import get_gemm_mfu


def gemm_latency_s(*, m: int, k: int, n: int, device_type: str, use_fp8_gemm: bool) -> float:

    gpu = gpu_map[device_type]
    gflops = gemm_flops(m, k, n) / 1e9
    mfu = get_gemm_mfu(device_type, m, k, n)
    tflops = gpu.fp8_tflops if use_fp8_gemm else gpu.fp16_tflops
    return gflops / (tflops * 1024 * mfu)
