from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMConfig, OpenAICompatAsyncLLM, OpenAICompatLLM
from .prompts import GENERATOR_PROMPT


@dataclass(frozen=True)
class GeneratorConfig:
    """
    Generator prompt is intentionally fixed (not a hyper-parameter).
    """

    model: str
    max_tokens: int = 128


def generate_answer(*, query: str, context: str, cfg: GeneratorConfig, llm_base_url: str) -> str:
    prompt = GENERATOR_PROMPT.format(context=context, question=query)
    llm = OpenAICompatLLM(LLMConfig(base_url=llm_base_url, model=cfg.model, temperature=0.0))
    return llm.generate(prompt=prompt, max_tokens=int(cfg.max_tokens)).strip()


def generate_answer_with_images(*, query: str, context: str, image_paths: list[str], cfg: GeneratorConfig, llm_base_url: str) -> str:
    """
    Same fixed prompt, but attach images to the user message for vision-capable models.
    """
    prompt = GENERATOR_PROMPT.format(context=context, question=query)
    llm = OpenAICompatLLM(LLMConfig(base_url=llm_base_url, model=cfg.model, temperature=0.0))
    return llm.generate_with_images(prompt=prompt, image_paths=list(image_paths or []), max_tokens=int(cfg.max_tokens)).strip()


async def generate_answer_async(*, query: str, context: str, cfg: GeneratorConfig, llm_base_url: str, semaphore=None) -> str:
    prompt = GENERATOR_PROMPT.format(context=context, question=query)
    llm = OpenAICompatAsyncLLM(LLMConfig(base_url=llm_base_url, model=cfg.model, temperature=0.0))
    if semaphore is not None:
        async with semaphore:
            out = await llm.generate(prompt=prompt, max_tokens=int(cfg.max_tokens))
    else:
        out = await llm.generate(prompt=prompt, max_tokens=int(cfg.max_tokens))
    return out.strip()


async def generate_answer_with_images_async(
    *, query: str, context: str, image_paths: list[str], cfg: GeneratorConfig, llm_base_url: str, semaphore=None
) -> str:
    prompt = GENERATOR_PROMPT.format(context=context, question=query)
    llm = OpenAICompatAsyncLLM(LLMConfig(base_url=llm_base_url, model=cfg.model, temperature=0.0))
    if semaphore is not None:
        async with semaphore:
            out = await llm.generate_with_images(prompt=prompt, image_paths=list(image_paths or []), max_tokens=int(cfg.max_tokens))
    else:
        out = await llm.generate_with_images(prompt=prompt, image_paths=list(image_paths or []), max_tokens=int(cfg.max_tokens))
    return out.strip()

