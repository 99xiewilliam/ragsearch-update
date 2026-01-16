from __future__ import annotations

REWRITER_PROMPTS: dict[str, str] = {
    "rewrite_v1": (
        "You are a query rewriter. Rewrite the user's question to be more suitable for retrieval:\n"
        "- Preserve the original meaning, do not fabricate facts\n"
        "- Output 1 line, plain text only, no explanation\n\n"
        "Original question: {query}\n"
        "Rewritten:"
    ),
    "rewrite_v2_keywords": (
        "You are a query rewriter. Rewrite the question as a 'keywords + constraints' search query:\n"
        "- Output 1 line\n"
        "- Use more noun phrases, remove function words when possible\n\n"
        "Question: {query}\n"
        "Search query:"
    ),
}


PRUNER_PROMPTS: dict[str, str] = {
    "prune_v1": (
        "You are an evidence pruner. Given a question and several candidate evidence passages, select the most useful subset.\n"
        "Rules:\n"
        "- Output only the indices of passages to keep, comma-separated, e.g., 0,2,3\n"
        "- If all are irrelevant, output an empty line\n\n"
        "Question: {query}\n\n"
        "Candidate passages:\n{chunks}\n\n"
        "Keep indices:"
    ),
    "compress_v1": (
        "You are a content compressor. Given a question and several evidence passages, compress each passage to keep only the parts relevant to answering the question.\n"
        "Rules:\n"
        "- For each passage, output a compressed version that retains only relevant information\n"
        "- Preserve key facts, names, dates, and relationships mentioned in the question\n"
        "- Remove irrelevant details, background information, and tangential content\n"
        "- Output format: For each passage [i], output [i]: <compressed_content>\n"
        "- Keep the compressed content concise but complete enough to answer the question\n\n"
        "Question: {query}\n\n"
        "Passages to compress:\n{chunks}\n\n"
        "Compressed passages:"
    ),
}


GENERATOR_PROMPT: str = (
    "You are a question-answering system. Answer using only the CONTEXT provided.\n"
    "Requirements: Output only the final answer, keep it short (suitable for factoid questions), no explanation.\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "FINAL ANSWER:"
)

