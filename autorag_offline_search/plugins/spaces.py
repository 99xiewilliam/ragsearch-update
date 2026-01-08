from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from ..search_space import BinarySpace, SearchSpace


@dataclass
class DefaultSpace(SearchSpace):
    """
    Example space plugin: default space (same as SearchSpace).
    Use:
      --space_plugin autorag_offline_search.plugins.spaces:DefaultSpace
    """


@dataclass
class Bits14Space:
    """
    Example space plugin: fixed 2^14 binary space.
    Use:
      --space_plugin autorag_offline_search.plugins.spaces:Bits14Space
    """

    bits: int = 14

    def all_configs(self):
        return BinarySpace(bits=self.bits).all_configs()

    def space_dict(self) -> Dict[str, Sequence]:
        return BinarySpace(bits=self.bits).space_dict()


