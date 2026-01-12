from __future__ import annotations

REWRITER_PROMPTS: dict[str, str] = {
    "rewrite_v1": (
        "你是查询改写器。把用户问题改写成更适合检索的版本：\n"
        "- 保留原意，不要编造事实\n"
        "- 输出 1 行，纯文本，不要解释\n\n"
        "原问题：{query}\n"
        "改写："
    ),
    "rewrite_v2_keywords": (
        "你是查询改写器。请把问题改写为“关键词 + 约束”的检索式：\n"
        "- 输出 1 行\n"
        "- 多用名词短语，尽量去掉虚词\n\n"
        "问题：{query}\n"
        "检索式："
    ),
}


PRUNER_PROMPTS: dict[str, str] = {
    "prune_v1": (
        "你是证据剪枝器。给定问题和若干候选证据段落，请选择最有用的段落子集。\n"
        "规则：\n"
        "- 只输出需要保留的段落编号，逗号分隔，例如：0,2,3\n"
        "- 如果都无关，输出空行\n\n"
        "问题：{query}\n\n"
        "候选段落：\n{chunks}\n\n"
        "保留编号："
    )
}


GENERATOR_PROMPT: str = (
    "你是问答系统。只能使用 CONTEXT 回答。\n"
    "要求：只输出最终答案，尽量短（适合 factoid），不要解释。\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "FINAL ANSWER:"
)

