from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

from hardware.gpu import GPU


TensorShape = Sequence[int]
Stage = Literal["prefill", "decode"]
Mode = Literal["normal", "low_latency"]
Op = Literal["allreduce", "dispatch", "combine", "allgather", "reducescatter"]


@dataclass(frozen=True)
class CommContext:
    world_size: int
    num_nodes: int
    gpu: GPU
    hidden_size: int
    num_experts_per_tok: int
    enable_deepep: bool = False


class CommEstimator:
    """Communication estimator with a stable external interface.

    Engine code should call:
        estimate(op, tensor_shape, stage, mode)

    and never call internal helpers directly.
    """

    def __init__(self, ctx: CommContext):
        self.ctx = ctx
        self._op_impls: Dict[str, Callable[..., float]] = {
            "allreduce": self._allreduce,
            "dispatch": self._dispatch,
            "combine": self._combine,
        }

    def register_op(self, op: str, fn: Callable[..., float]) -> None:
        """Register a new op implementation without modifying engine code."""
        self._op_impls[op] = fn

    def estimate(
        self,
        op: Optional[str],
        tensor: TensorShape,
        stage: Optional[Stage] = None,
        mode: Optional[Mode] = None,
    ) -> float:
        """Unified entrypoint.

        - If ctx.world_size==1, always returns 0.0 (explicit branch).
        - If op is None, dispatch by stage + ctx.enable_deepep:
            prefill: allreduce OR (dispatch+combine normal)
            decode : allreduce OR (dispatch+combine low_latency)
        - tensor is a shape in elements; dtype sizing is handled per-op.
        """
        if self.ctx.world_size <= 1:
            return 0.0

        if op is None:
            if stage is None:
                raise ValueError("stage must be provided when op is None")
            if self.ctx.enable_deepep:
                m: Mode = "normal" if stage == "prefill" else "low_latency"
                # For staged comm, return the sum of dispatch+combine
                return self._dispatch(tensor, mode=m) + self._combine(tensor, mode=m)
            # default allreduce
            return self._allreduce(tensor)

        fn = self._op_impls.get(op.lower())
        if fn is None:
            raise KeyError(f"Unknown comm op: {op}")
        return fn(tensor, mode=mode)  # type: ignore[arg-type]

    # ------------------ core helpers ------------------

    def _size_bw_model(
        self, tensor_shape: TensorShape, bytes_per_elem: int, inter_node: bool
    ) -> float:
        """Estimate time (seconds) based purely on BW model."""
        if self.ctx.world_size <= 1:
            return 0.0
        n_elem = 1
        for v in tensor_shape:
            n_elem *= int(v)
        size_bytes = n_elem * int(bytes_per_elem)
        bw = self.ctx.gpu.rdma_bw if inter_node else self.ctx.gpu.nvlink_bw
        return (size_bytes / (1024**3)) / bw

    def _allreduce(self, tensor_shape: TensorShape, mode: Optional[str] = None) -> float:
        # legacy assumption: allreduce payload is fp16 (2 bytes)
        inter_node = self.ctx.num_nodes > 1
        return self._size_bw_model(tensor_shape, bytes_per_elem=2, inter_node=inter_node)

    def _dispatch(self, tensor_shape: TensorShape, mode: Optional[Mode] = "normal") -> float:
        # legacy: dispatch payload is fp8 (1 byte)
        if mode == "normal":
            # intra+inter split (matches previous logic)
            temp_num_nodes = min(self.ctx.num_nodes, 4)
            # tensor_shape is [num_tokens, hidden]; treat first dim as tokens
            num_tokens = int(tensor_shape[0])
            send_tokens = num_tokens * (temp_num_nodes - 1)
            t1 = self._size_bw_model(
                [send_tokens, self.ctx.hidden_size], bytes_per_elem=1, inter_node=True
            )
            t2 = self._size_bw_model(
                [num_tokens, self.ctx.hidden_size], bytes_per_elem=1, inter_node=False
            )
            return t1 + t2

        # low_latency: send tokens * topk (experts_per_tok)
        num_tokens = int(tensor_shape[0])
        send_tokens = num_tokens * int(self.ctx.num_experts_per_tok)
        return self._size_bw_model(
            [send_tokens, self.ctx.hidden_size],
            bytes_per_elem=1,
            inter_node=(self.ctx.num_nodes > 1),
        )

    def _combine(self, tensor_shape: TensorShape, mode: Optional[Mode] = "normal") -> float:
        # legacy: combine payload is fp16 (2 bytes)
        if mode == "normal":
            temp_num_nodes = min(self.ctx.num_nodes, 4)
            num_tokens = int(tensor_shape[0])
            rcv_tokens = num_tokens * (temp_num_nodes - 1)
            t1 = self._size_bw_model(
                [rcv_tokens, self.ctx.hidden_size], bytes_per_elem=2, inter_node=True
            )
            t2 = self._size_bw_model(
                [num_tokens, self.ctx.hidden_size], bytes_per_elem=2, inter_node=False
            )
            return t1 + t2

        num_tokens = int(tensor_shape[0])
        rcv_tokens = num_tokens * int(self.ctx.num_experts_per_tok)
        return self._size_bw_model(
            [rcv_tokens, self.ctx.hidden_size],
            bytes_per_elem=2,
            inter_node=(self.ctx.num_nodes > 1),
        )
