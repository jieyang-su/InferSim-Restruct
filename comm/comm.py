from __future__ import annotations

from typing import Sequence, Tuple

from config.model_config import ModelConfig
from hardware.gpu import GPU

from comm.estimator import CommContext, CommEstimator


class Comm:
    """Backward-compatible COMM wrapper.

    New engine code should directly depend on CommEstimator (comm/estimator.py).
    """

    def __init__(
        self,
        config: ModelConfig,
        gpu: GPU,
        world_size: int,
        num_nodes: int = 1,
        enable_deepep: bool = False,
    ):
        self.config = config
        self.gpu = gpu
        self.world_size = world_size
        self.num_nodes = num_nodes
        self.enable_deepep = enable_deepep

        ctx = CommContext(
            world_size=world_size,
            num_nodes=num_nodes,
            gpu=gpu,
            hidden_size=config.hidden_size,
            num_experts_per_tok=getattr(config, "num_experts_per_tok", 1),
            enable_deepep=enable_deepep,
        )
        self._est = CommEstimator(ctx)

    # ----------- New standardized API (SR_F_DS_COMM_0008) -----------

    def estimate(
        self,
        op: str | None,
        tensor: Sequence[int],
        stage: str | None = None,
        mode: str | None = None,
    ) -> float:
        return self._est.estimate(op=op, tensor=tensor, stage=stage, mode=mode)

    def register_op(self, op: str, fn):
        """SR_F_DS_COMM_0009: add new comm op without touching engine."""
        self._est.register_op(op, fn)

    # ----------- Legacy API (kept for compatibility) -----------

    def size_bw_model(self, tensor_shape, use_fp8=False, inter_node=False):
        bytes_per_elem = 1 if use_fp8 else 2
        return self._est._size_bw_model(
            tensor_shape=tensor_shape,
            bytes_per_elem=bytes_per_elem,
            inter_node=inter_node,
        )

    def all_reduce(self, num_tokens):
        tensor_shape = [num_tokens * self.world_size, self.config.hidden_size]
        return self.estimate("allreduce", tensor_shape)

    def dispatch(self, num_tokens, mode="normal"):
        tensor_shape = [num_tokens, self.config.hidden_size]
        return self.estimate("dispatch", tensor_shape, mode=mode)

    def combine(self, num_tokens, mode="normal"):
        tensor_shape = [num_tokens, self.config.hidden_size]
        return self.estimate("combine", tensor_shape, mode=mode)

    def prefill_comm(self, num_tokens: int):
        if self.enable_deepep:
            return self.dispatch(num_tokens, "normal"), self.combine(num_tokens, "normal")
        return self.all_reduce(num_tokens), self.all_reduce(num_tokens)

    def decode_comm(self, num_tokens: int):
        if self.enable_deepep:
            return (
                self.dispatch(num_tokens, "low_latency"),
                self.combine(num_tokens, "low_latency"),
            )
        return self.all_reduce(num_tokens), self.all_reduce(num_tokens)
