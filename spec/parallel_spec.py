from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelSpec:
    world_size: int
    num_nodes: int = 1

    # place-holders for future extensions (EP/DP/etc.)
    ep: int = 1
    dp: int = 1
