#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


# Ensure repo root is importable when running as a script (sys.path[0] is scripts/).
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _b64_data_url(path: str) -> str:
    import mimetypes

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _auto_qa_for_image(*, llm_base_url: str, model: str, image_path: str, qid: str) -> Dict[str, Any]:
    """
    Ask the VL model to produce one QA with short deterministic answer (string).
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), base_url=llm_base_url)
    prompt = (
        "请基于这张图片生成一个“可以有唯一短答案”的问题，并给出标准答案。\n"
        "要求：\n"
        "- 只输出 JSON，一行\n"
        "- 字段：question (string), answer (string)\n"
        "- answer 尽量短（1-5 个词/数字），便于 exact match\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _b64_data_url(image_path)}},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=256,
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        obj = json.loads(txt)
        q = str(obj.get("question", "") or "").strip()
        a = str(obj.get("answer", "") or "").strip()
    except Exception:
        q = ""
        a = ""
    if not q or not a:
        # fallback: ask a deterministic caption question
        q = "请简要描述这张图片里最显著的对象是什么？只回答一个短词。"
        a = "unknown"
    return {"qid": qid, "query": q, "generation_gt": [a]}


def main() -> None:
    p = argparse.ArgumentParser(description="Create a tiny multimodal dataset (parquet) from local images.")
    p.add_argument("--out_dir", type=str, default="/home/xwh/ragsearch-update/tmp/mm_images_dataset")
    p.add_argument("--img_dir", type=str, default="/home/xwh/ragsearch-update/test")
    p.add_argument("--images", type=str, default="15-1.jpg,15-5.png,15-8.jpg", help="Comma-separated image filenames under img_dir.")
    p.add_argument("--llm_base_url", type=str, default="", help="If set, auto-generate extra QA per image using VL model.")
    p.add_argument("--vl_model", type=str, default="/home/xwh/models/Qwen3-VL-4B-Instruct")
    args = p.parse_args()

    import pandas as pd

    out = Path(args.out_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "validation").mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.img_dir)
    img_names = [x.strip() for x in str(args.images).split(",") if x.strip()]
    img_paths = [str((img_dir / x).resolve()) for x in img_names]
    for pth in img_paths:
        if not Path(pth).exists():
            raise SystemExit(f"Image not found: {pth}")

    # corpus: store metadata as JSON string to ensure parquet round-trip portability
    corpus_rows = []
    for i, pth in enumerate(img_paths):
        doc_id = f"img_{i+1}"
        meta = {"image_path": pth, "source": "local_test_images"}
        corpus_rows.append(
            {
                "doc_id": doc_id,
                "contents": "这是一张图片，请结合图片回答问题。",
                "metadata": json.dumps(meta, ensure_ascii=False),
            }
        )
    corpus = pd.DataFrame(corpus_rows)

    # qa: always include one deterministic question for 15-1.jpg if present (route number = 82)
    qa_rows: List[Dict[str, Any]] = []
    if any(Path(p).name == "15-1.jpg" for p in img_paths):
        # Store as JSON string for robust parquet round-trip.
        qa_rows.append(
            {"qid": "q_route_82", "query": "窗外绿色双层电车的线路号是多少？只回答数字。", "generation_gt": json.dumps(["82"])}
        )

    # optional: auto QA for each image
    if str(args.llm_base_url or "").strip():
        for i, pth in enumerate(img_paths):
            row = _auto_qa_for_image(llm_base_url=args.llm_base_url, model=args.vl_model, image_path=pth, qid=f"q_auto_{i+1}")
            # store as JSON string
            row["generation_gt"] = json.dumps(row.get("generation_gt", [""]))
            qa_rows.append(row)

    if not qa_rows:
        # minimal fallback
        qa_rows.append({"qid": "q1", "query": "这张图片里最显著的对象是什么？", "generation_gt": json.dumps(["unknown"])})

    qa = pd.DataFrame(qa_rows)

    for split in ("train", "validation"):
        corpus.to_parquet(out / split / "corpus.parquet", index=False)
        qa.to_parquet(out / split / "qa.parquet", index=False)

    print(f"[OK] wrote dataset to: {out}")
    print(f"[OK] images={len(img_paths)} qas={len(qa_rows)}")


if __name__ == "__main__":
    main()

