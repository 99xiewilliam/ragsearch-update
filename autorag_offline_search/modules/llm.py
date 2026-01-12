from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from .device import env_openai_api_key


@dataclass(frozen=True)
class LLMConfig:
    """
    Minimal LLM config for local OpenAI-compatible endpoints (e.g. vLLM).
    We intentionally drop other backends to keep the project focused.
    """

    base_url: str
    model: str
    temperature: float = 0.0


@lru_cache(maxsize=8)
def _client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


class OpenAICompatLLM:
    def __init__(self, cfg: LLMConfig, *, api_key: Optional[str] = None):
        if not cfg.base_url:
            raise ValueError("llm_base_url is required for OpenAI-compatible LLM calls (e.g. http://localhost:9000/v1)")
        self.cfg = cfg
        key = api_key or env_openai_api_key() or "EMPTY"
        self._c = _client(cfg.base_url, key)

    def generate(self, *, prompt: str, max_tokens: int) -> str:
        resp = self._c.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(self.cfg.temperature),
            max_tokens=int(max_tokens),
        )
        return (resp.choices[0].message.content or "").strip()

