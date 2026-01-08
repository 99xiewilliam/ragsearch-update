from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class QAExample:
    qid: str
    query: str
    generation_gt: List[str]  # allow multiple refs


@dataclass(frozen=True)
class Doc:
    doc_id: str
    contents: str
    metadata: Optional[JsonDict] = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class EvalResult:
    per_metric: JsonDict
    weighted_reward: float


Config = JsonDict


