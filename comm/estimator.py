from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Sequence

from hardware.gpu import GPU


TensorShape = Sequence[int]
Stage = Literal["prefill", "decode"]
Mode = Literal["normal", "low_latency"]
Op = Literal[
    "allreduce",
    "dispatch",
    "combine",
    "allgather",
    "reducescatter",
    "stage_before",
    "stage_after",
]


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
            "stage_before": self._stage_before,
            "stage_after": self._stage_after,
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
        - If op is None, return the *total* stage comm:
            stage_before + stage_after
        - tensor is a shape in elements; dtype sizing is handled per-op.
        """
        if self.ctx.world_size <= 1:
            return 0.0

        if op is None:
            if stage is None:
                raise ValueError("stage must be provided when op is None")
            return self._stage_before(tensor, stage=stage) + self._stage_after(
                tensor, stage=stage
            )

        fn = self._op_impls.get(op.lower())
        if fn is None:
            raise KeyError(f"Unknown comm op: {op}")
        if op.lower().startswith("stage_"):
            if stage is None:
                raise ValueError("stage must be provided for stage_* ops")
            return fn(tensor, stage=stage)  # type: ignore[misc]
        return fn(tensor, mode=mode)  # type: ignore[arg-type]

    # ------------------ core helpers ------------------

    def _size_bw_model(
        self, tensor_shape: TensorShape, bytes_per_elem: int, inter_node: bool
    ) -> float:
        """
        Estimate time (seconds) based on a BW model.
        time scales with tensor size and uses different links for intra-node vs inter-node communication.
        """
        if self.ctx.world_size <= 1:
            return 0.0
        n_elem = 1
        for v in tensor_shape:
            n_elem *= int(v)
        size_bytes = n_elem * int(bytes_per_elem)
        bw = self.ctx.gpu.rdma_bw if inter_node else self.ctx.gpu.nvlink_bw
        return (size_bytes / (1024**3)) / bw

    def _ring_factor(self, p: int) -> float:
        """单机 Ring Algorithmic factor for ring collectives (bandwidth-dominated)."""
        if p <= 1:
            return 0.0
        # 2*(p-1)/p is the well-known volume factor for ring allreduce.
        return 2.0 * (p - 1) / p

    def _allreduce(self, tensor_shape: TensorShape, mode: Optional[str] = None) -> float:
        """AllReduce estimate.

        - 单机: ring allreduce over NVLink.
        - 多机: simple hierarchical model:
            intra-node reduce-scatter + inter-node allreduce(shard) + intra-node allgather

        Todo:still a bandwidth-dominated model (no explicit latency term).
        """

        bytes_per_elem = 2
        n_elem = 1
        for v in tensor_shape:
            n_elem *= int(v)
        size_bytes = n_elem * int(bytes_per_elem)

        if self.ctx.num_nodes <= 1:
            # ring allreduce over nvlink
            bw = self.ctx.gpu.nvlink_bw
            return self._ring_factor(self.ctx.world_size) * (size_bytes / (1024**3)) / bw

        # Hierarchical: assume even split of ranks across nodes.
        p_total = self.ctx.world_size
        p_nodes = self.ctx.num_nodes
        p_local = max(1, p_total // p_nodes)

        # Intra-node reduce-scatter + allgather each move (p_local-1)/p_local of full tensor.
        bw_intra = self.ctx.gpu.nvlink_bw
        intra_factor = (p_local - 1) / p_local
        t_intra = 2.0 * intra_factor * (size_bytes / (1024**3)) / bw_intra

        # Inter-node allreduce happens on the shard each rank owns after reduce-scatter.
        shard_bytes = size_bytes / p_local
        bw_inter = self.ctx.gpu.rdma_bw
        t_inter = self._ring_factor(p_nodes) * (shard_bytes / (1024**3)) / bw_inter

        return t_intra + t_inter

    def _stage_before(self, tensor_shape: TensorShape, stage: Stage) -> float:
        """Stage-oriented comm before compute."""
        if self.ctx.enable_deepep:
            m: Mode = "normal" if stage == "prefill" else "low_latency"
            return self._dispatch(tensor_shape, mode=m)
        return self._allreduce(tensor_shape)

    def _stage_after(self, tensor_shape: TensorShape, stage: Stage) -> float:
        """Stage-oriented comm after compute."""
        if self.ctx.enable_deepep:
            m: Mode = "normal" if stage == "prefill" else "low_latency"
            return self._combine(tensor_shape, mode=m)
        return self._allreduce(tensor_shape)

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