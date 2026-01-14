from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .device import auto_device
from .multimodal_utils import is_probably_clip_model


def _looks_like_qwen_vl_embedding(model_name: str) -> bool:
    s = str(model_name or "")
    s_low = s.lower()
    return ("qwen3-vl-embedding" in s_low) or ("qwen3_vl_embedding" in s_low)


@lru_cache(maxsize=2)
def _qwen_vl_embedder(model_name_or_path: str):
    """
    Load Qwen3VLEmbedder using the model-provided script if available.
    This keeps our repo lightweight while still supporting local model directories.
    """
    import importlib.util
    from pathlib import Path

    p = Path(str(model_name_or_path))
    # Prefer local model dir scripts if present (your setup uses /home/xwh/models/Qwen3-VL-Embedding-2B)
    script = p / "scripts" / "qwen3_vl_embedding.py"
    if script.exists():
        spec = importlib.util.spec_from_file_location("_qwen3_vl_embedding", str(script))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import: {script}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        Qwen3VLEmbedder = getattr(mod, "Qwen3VLEmbedder")
    else:
        raise ModuleNotFoundError(
            "Qwen3-VL embedding helper script not found. "
            "Set embedder_model to a local model directory that contains scripts/qwen3_vl_embedding.py "
            "(e.g. /home/xwh/models/Qwen3-VL-Embedding-2B), or use a CLIP embedder "
            "(e.g. openai/clip-vit-base-patch32)."
        )

    # Prefer bf16 on CUDA; CPU falls back internally.
    kwargs = {}
    try:
        import torch

        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.bfloat16
    except Exception:
        pass

    return Qwen3VLEmbedder(model_name_or_path=str(model_name_or_path), **kwargs)


@lru_cache(maxsize=4)
def _clip(model_name: str):
    """
    Load CLIP model/processor via transformers.
    Note: this is only used when the embedder model name looks like a CLIP model.
    """
    from transformers import CLIPModel, CLIPProcessor

    dev = auto_device()
    m = CLIPModel.from_pretrained(model_name)
    p = CLIPProcessor.from_pretrained(model_name)
    try:
        m.to(dev)
    except Exception:
        m.to("cpu")
    m.eval()
    return m, p


@lru_cache(maxsize=8)
def _st(model_name: str):
    from sentence_transformers import SentenceTransformer

    dev = auto_device()
    if dev == "cuda":
        try:
            return SentenceTransformer(model_name, device="cuda")
        except Exception:
            return SentenceTransformer(model_name, device="cpu")
    return SentenceTransformer(model_name, device="cpu")


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def embed_mm_texts(model_name: str, texts: Sequence[str]) -> np.ndarray:
    """
    Text embedding for multimodal pipeline:
    - If CLIP-like: uses CLIP text features.
    - Else: uses SentenceTransformer (same as common pipeline).
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    if _looks_like_qwen_vl_embedding(model_name):
        try:
            emb = _qwen_vl_embedder(model_name).process([{"text": t} for t in list(texts)], normalize=True)
            return np.asarray(emb.detach().float().cpu().numpy(), dtype=np.float32)
        except Exception:
            # Fallback to a robust text embedder so the demo can keep running.
            # (Prefer a model that is likely already cached in this repo's workflows.)
            return embed_mm_texts("BAAI/bge-m3", texts)

    if is_probably_clip_model(model_name):
        m, proc = _clip(model_name)
        dev = "cuda" if auto_device() == "cuda" else "cpu"
        try:
            import torch

            inputs = proc(text=list(texts), return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                feats = m.get_text_features(**inputs)
            out = feats.detach().float().cpu().numpy()
            return _l2norm(out.astype(np.float32))
        except Exception:
            # conservative fallback
            return embed_mm_texts("sentence-transformers/all-MiniLM-L6-v2", texts)

    m = _st(model_name)
    emb = m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def embed_mm_query(model_name: str, query: str) -> np.ndarray:
    return embed_mm_texts(model_name, [query])[0]


def embed_mm_chunks(
    *,
    model_name: str,
    chunk_texts: Sequence[str],
    chunk_images: Sequence[List[str]],
) -> np.ndarray:
    """
    Return per-chunk embeddings.

    Current behavior (minimal but practical):
    - Always embed text.
    - If the model is CLIP-like and a chunk has at least one image path, also embed the first image,
      then average (text + image) in embedding space (both are normalized).
    - For non-CLIP models (e.g. Qwen/Qwen3-VL-Embedding-*), we keep text-only (many VL embedding models
      still support text inputs; image support varies and is backend-dependent).
    """
    # Qwen3-VL embedding supports multimodal inputs directly (text/image/mix) into the same space.
    if _looks_like_qwen_vl_embedding(model_name):
        try:
            inputs = []
            for t, imgs in zip(list(chunk_texts), list(chunk_images)):
                p0: Optional[str] = None
                if isinstance(imgs, list) and imgs:
                    p0 = str(imgs[0] or "").strip()
                item = {}
                if t and str(t).strip():
                    item["text"] = str(t)
                if p0:
                    item["image"] = p0
                if not item:
                    item["text"] = "NULL"
                inputs.append(item)
            emb = _qwen_vl_embedder(model_name).process(inputs, normalize=True)
            return np.asarray(emb.detach().float().cpu().numpy(), dtype=np.float32)
        except Exception:
            # Fallback: text-only embedding (keeps pipeline running even without Qwen helper script).
            return embed_mm_texts("BAAI/bge-m3", chunk_texts).astype(np.float32)

    txt = embed_mm_texts(model_name, chunk_texts)
    if not is_probably_clip_model(model_name):
        return txt.astype(np.float32)

    # CLIP image features (best-effort)
    try:
        from PIL import Image
        import torch

        m, proc = _clip(model_name)
        dev = "cuda" if auto_device() == "cuda" else "cpu"
        img_vecs: List[np.ndarray] = []
        for imgs in chunk_images:
            p0: Optional[str] = None
            if isinstance(imgs, list) and imgs:
                p0 = str(imgs[0] or "").strip()
            if not p0:
                img_vecs.append(None)  # type: ignore[arg-type]
                continue
            try:
                im = Image.open(p0).convert("RGB")
            except Exception:
                img_vecs.append(None)  # type: ignore[arg-type]
                continue

            inputs = proc(images=im, return_tensors="pt")
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            with torch.no_grad():
                feats = m.get_image_features(**inputs)
            v = feats.detach().float().cpu().numpy()
            v = _l2norm(v.astype(np.float32))[0]
            img_vecs.append(v)

        out = []
        for i in range(len(chunk_texts)):
            t = txt[i]
            iv = img_vecs[i]
            if iv is None:
                out.append(t)
            else:
                out.append(_l2norm(np.asarray([(t + iv) / 2.0], dtype=np.float32))[0])
        return np.asarray(out, dtype=np.float32)
    except Exception:
        return txt.astype(np.float32)

