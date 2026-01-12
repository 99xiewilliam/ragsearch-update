from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .llm import LLMConfig, OpenAICompatLLM


_INT_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class PrunerConfig:
    enabled: bool
    model: str  # preset id or explicit model path/name
    prompt: str
    max_tokens: int


def prune_chunks(
    *,
    query: str,
    chunks: List[str],
    cfg: PrunerConfig,
    llm_base_url: str,
    model_resolver,
) -> List[str]:
    if not cfg.enabled or not chunks:
        return chunks
    prompt = (cfg.prompt or "").format(
        query=query,
        chunks="\n".join([f"[{i}] {t}" for i, t in enumerate(chunks)]),
    )
    if not prompt.strip():
        return chunks
    llm_model = model_resolver(cfg.model)
    llm = OpenAICompatLLM(LLMConfig(base_url=llm_base_url, model=llm_model, temperature=0.0))
    out = llm.generate(prompt=prompt, max_tokens=int(cfg.max_tokens))
    idx = [int(m.group(0)) for m in _INT_RE.finditer(out or "")]
    keep = []
    seen = set()
    for i in idx:
        if 0 <= i < len(chunks) and i not in seen:
            seen.add(i)
            keep.append(chunks[i])
    return keep or chunks

