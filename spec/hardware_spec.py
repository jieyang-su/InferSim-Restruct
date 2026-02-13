from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwareSpec:
    device_type: str
    dtype_gemm: str = "fp16"   # "fp16" | "fp8"
    dtype_kv: str = "fp16"     # "fp16" | "fp8"
    enable_tbo: bool = False
    sm_ratio: Optional[float] = None
