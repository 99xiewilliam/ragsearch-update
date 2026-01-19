from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMConfig, OpenAICompatAsyncLLM, OpenAICompatLLM


@dataclass(frozen=True)
class RewriterConfig:
    enabled: bool
    model: str  # preset id or explicit model path/name
    prompt: str
    max_tokens: int


def rewrite_query(*, query: str, cfg: RewriterConfig, llm_base_url: str, model_resolver) -> str:
    if not cfg.enabled:
        return query
    prompt = (cfg.prompt or "").format(query=query)
    if not prompt.strip():
        return query
    llm_model = model_resolver(cfg.model)
    llm = OpenAICompatLLM(LLMConfig(base_url=llm_base_url, model=llm_model, temperature=0.0))
    out = llm.generate(prompt=prompt, max_tokens=int(cfg.max_tokens))
    return out.strip() or query


async def rewrite_query_async(*, query: str, cfg: RewriterConfig, llm_base_url: str, model_resolver, semaphore=None) -> str:
    """
    Async rewrite with optional concurrency limiting via `asyncio.Semaphore`.
    """
    if not cfg.enabled:
        return query
    prompt = (cfg.prompt or "").format(query=query)
    if not prompt.strip():
        return query
    llm_model = model_resolver(cfg.model)
    llm = OpenAICompatAsyncLLM(LLMConfig(base_url=llm_base_url, model=llm_model, temperature=0.0))
    if semaphore is not None:
        async with semaphore:
            out = await llm.generate(prompt=prompt, max_tokens=int(cfg.max_tokens))
    else:
        out = await llm.generate(prompt=prompt, max_tokens=int(cfg.max_tokens))
    return out.strip() or query

