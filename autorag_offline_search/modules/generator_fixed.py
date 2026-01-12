from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMConfig, OpenAICompatLLM
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

