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
        """
        max_tokens is the *output* token budget for OpenAI-compatible endpoints.

        Convention in this repo:
        - max_tokens >= 32000 means "do not pass max_tokens" (let the server decide),
          which avoids vLLM errors when users confuse context length (e.g. 32768) with output tokens.
        """
        kwargs = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.cfg.temperature),
        }
        if int(max_tokens) < 32000:
            kwargs["max_tokens"] = int(max_tokens)
        resp = self._c.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    def generate_with_images(self, *, prompt: str, image_paths: list[str], max_tokens: int) -> str:
        """
        Vision-capable call for OpenAI-compatible endpoints (e.g. vLLM with Qwen-VL).
        Sends images as data URLs (base64).
        """
        import base64
        import mimetypes

        content = [{"type": "text", "text": prompt}]
        for p in list(image_paths or [])[:4]:
            p = str(p or "").strip()
            if not p:
                continue
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            mime = mimetypes.guess_type(p)[0] or "image/jpeg"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        kwargs = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": float(self.cfg.temperature),
        }
        if int(max_tokens) < 32000:
            kwargs["max_tokens"] = int(max_tokens)
        resp = self._c.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

