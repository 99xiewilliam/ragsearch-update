from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .llm import LLMConfig, OpenAICompatLLM


_INT_RE = re.compile(r"\d+")
_COMPRESS_RE = re.compile(r"\[(\d+)\]:\s*(.+?)(?=\n\[\d+\]:|\Z)", re.DOTALL)


@dataclass(frozen=True)
class PrunerConfig:
    enabled: bool
    model: str  # preset id or explicit model path/name
    prompt: str
    max_tokens: int
    mode: str = "select"  # "select" (choose which chunks) or "compress" (compress each chunk)


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
    
    if cfg.mode == "compress":
        # Parse compressed chunks: [0]: content0\n[1]: content1\n...
        compressed_map: dict[int, str] = {}
        for match in _COMPRESS_RE.finditer(out or ""):
            idx = int(match.group(1))
            content = match.group(2).strip()
            if 0 <= idx < len(chunks) and content:
                compressed_map[idx] = content
        
        # Return compressed chunks in original order, fallback to original if not compressed
        result = []
        for i, orig in enumerate(chunks):
            if i in compressed_map:
                result.append(compressed_map[i])
            else:
                # If this chunk wasn't compressed, keep original (or could skip it)
                result.append(orig)
        return result if compressed_map else chunks
    else:
        # Original "select" mode: parse indices and return selected chunks
        idx = [int(m.group(0)) for m in _INT_RE.finditer(out or "")]
        keep = []
        seen = set()
        for i in idx:
            if 0 <= i < len(chunks) and i not in seen:
                seen.add(i)
                keep.append(chunks[i])
        return keep or chunks

