from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from spec.model_spec import ModelSpec


class BaseAdapter(ABC):
    """Adapter maps an external model config into ModelSpec."""

    @abstractmethod
    def match(self, architectures: Sequence[str], model_prefix: str) -> bool:
        ...

    @abstractmethod
    def parse(self, hf_config: Dict[str, Any]) -> ModelSpec:
        ...
