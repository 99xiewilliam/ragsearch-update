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


_ASYNC_CLIENTS: dict[tuple[int, str, str], object] = {}


def _loop_key() -> int:
    """
    Return current running loop id (or -1 if no running loop).
    """
    try:
        import asyncio

        return id(asyncio.get_running_loop())
    except Exception:
        return -1


def _async_client(base_url: str, api_key: str):
    """
    IMPORTANT:
    - Do NOT reuse AsyncOpenAI across different event loops.
    - BUT within the same loop, reusing a single client is *good* (connection pooling, fewer objects),
      and avoids a large number of unclosed httpx AsyncClient instances (which can emit noisy
      "Event loop is closed" during interpreter shutdown).
    """
    from openai import AsyncOpenAI

    lk = _loop_key()
    k = (lk, str(base_url), str(api_key))
    c = _ASYNC_CLIENTS.get(k)
    if c is None:
        c = AsyncOpenAI(api_key=api_key, base_url=base_url)
        _ASYNC_CLIENTS[k] = c
    return c


async def close_async_clients_for_current_loop() -> None:
    """
    Close all cached AsyncOpenAI clients bound to the current running loop.
    Call this at the end of asyncio.run() to avoid "Event loop is closed" noise.
    """
    lk = _loop_key()
    if lk < 0:
        return
    to_close = [k for k in list(_ASYNC_CLIENTS.keys()) if k[0] == lk]
    for k in to_close:
        c = _ASYNC_CLIENTS.pop(k, None)
        if c is None:
            continue
        try:
            close = getattr(c, "close", None)
            if close is not None:
                # openai.AsyncOpenAI.close is async in this env
                await close()
        except Exception:
            pass


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
        """
        kwargs = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.cfg.temperature),
        }
        if int(max_tokens) < 32000:
            kwargs["max_tokens"] = int(max_tokens)

        import time

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._c.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                # backoff: 0.5s, 1.0s, 2.0s
                time.sleep(0.5 * (2**attempt))
        raise last_err  # type: ignore[misc]

    def generate_with_images(self, *, prompt: str, image_paths: list[str], max_tokens: int) -> str:
        """
        Vision-capable call for OpenAI-compatible endpoints.
        """
        import base64
        import mimetypes
        import time

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

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._c.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (2**attempt))
        raise last_err  # type: ignore[misc]


class OpenAICompatAsyncLLM:
    def __init__(self, cfg: LLMConfig, *, api_key: Optional[str] = None):
        if not cfg.base_url:
            raise ValueError("llm_base_url is required for OpenAI-compatible LLM calls (e.g. http://localhost:9000/v1)")
        self.cfg = cfg
        key = api_key or env_openai_api_key() or "EMPTY"
        self._c = _async_client(cfg.base_url, key)

    async def generate(self, *, prompt: str, max_tokens: int) -> str:
        """
        Async version of generate() for OpenAI-compatible endpoints (e.g. vLLM).
        See OpenAICompatLLM.generate() for the sentinel max_tokens convention.
        """
        kwargs = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(self.cfg.temperature),
        }
        if int(max_tokens) < 32000:
            kwargs["max_tokens"] = int(max_tokens)
        # Lightweight retries for flaky local endpoints under high concurrency.
        import asyncio

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._c.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                # backoff: 0.5s, 1.0s, 2.0s
                await asyncio.sleep(0.5 * (2**attempt))
        raise last_err  # type: ignore[misc]

    async def generate_with_images(self, *, prompt: str, image_paths: list[str], max_tokens: int) -> str:
        """
        Async vision-capable call: images are sent as data URLs (base64).
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
        import asyncio

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._c.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.5 * (2**attempt))
        raise last_err  # type: ignore[misc]

