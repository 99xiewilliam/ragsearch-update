from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score
import os
import zipfile


@dataclass(frozen=True)
class MetricsConfig:
    # weights for the final reward
    weights: Dict[str, float]
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"  # strong default, but heavy


def _safe_mean(xs: List[float]) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / max(1, len(xs))


def compute_rouge(pred: str, refs: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    # take best over multiple refs (common in QA)
    r1, r2, rl = 0.0, 0.0, 0.0
    for ref in refs:
        s = scorer.score(ref, pred)
        r1 = max(r1, float(s["rouge1"].fmeasure))
        r2 = max(r2, float(s["rouge2"].fmeasure))
        rl = max(rl, float(s["rougeL"].fmeasure))
    return {"rouge1": r1, "rouge2": r2, "rougeL": rl}


def compute_bleu(pred: str, refs: List[str]) -> float:
    """
    Sentence BLEU (best over references) using sacrebleu if available.
    """
    if not refs:
        return 0.0
    try:
        import sacrebleu
    except Exception:
        return 0.0
    best = 0.0
    for r in refs:
        # sacrebleu expects refs as list-of-refs for one sentence: [[ref1, ref2, ...]]
        s = sacrebleu.sentence_bleu(pred, [r])
        best = max(best, float(s.score) / 100.0)
    return best


def compute_chrf(pred: str, refs: List[str]) -> float:
    """
    Character n-gram F-score (chrF). Very robust and common in modern NLP papers.
    """
    if not refs or not pred:
        return 0.0
    try:
        import sacrebleu
        # sacrebleu sentence_chrf expects refs as a list of strings
        res = sacrebleu.sentence_chrf(pred, refs)
        return float(res.score) / 100.0
    except Exception:
        return 0.0


def compute_exact_match(pred: str, refs: List[str]) -> float:
    if not refs:
        return 0.0
    p = pred.strip().lower()
    for r in refs:
        if p == r.strip().lower():
            return 1.0
    return 0.0


def compute_qa_f1(pred: str, refs: List[str]) -> float:
    """
    Token-level F1 score, common in SQuAD/QA tasks.
    """
    if not refs or not pred:
        return 0.0

    def get_tokens(s):
        return s.lower().split()

    p_tokens = get_tokens(pred)
    if not p_tokens:
        return 0.0

    best_f1 = 0.0
    from collections import Counter

    for ref in refs:
        r_tokens = get_tokens(ref)
        if not r_tokens:
            continue
        common = Counter(p_tokens) & Counter(r_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(p_tokens)
            recall = 1.0 * num_same / len(r_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def compute_meteor(pred: str, refs: List[str]) -> float:
    if not refs or not pred:
        return 0.0

    # Ensure required NLTK data is available.
    # NOTE: in some environments downloads can be interrupted and leave corrupted files
    # (e.g. a non-zip file where NLTK expects a zip). We treat both LookupError and
    # BadZipFile as "not available" and try to (re)download into a project-local cache.
    nltk_dir = os.environ.get("NLTK_DATA") or os.path.join(os.path.expanduser("~"), ".cache", "autorag_nltk")
    os.makedirs(nltk_dir, exist_ok=True)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    for res in ["wordnet", "punkt", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{res}" if res != "punkt" else "tokenizers/punkt")
        except (LookupError, zipfile.BadZipFile, OSError):
            try:
                nltk.download(res, download_dir=nltk_dir, quiet=True)
            except Exception:
                # Keep evaluation robust even if NLTK cannot download (offline env).
                pass

    # NLTK meteor_score expects references as a list of lists of tokens,
    # and hypothesis as a list of tokens.
    try:
        from nltk.tokenize import word_tokenize

        # Attempt to use word_tokenize for better results than simple split()
        try:
            hyp_tok = word_tokenize(pred)
            refs_tok = [word_tokenize(r) for r in refs]
        except Exception:
            # Fallback to simple split if word_tokenize fails (e.g. punkt issues)
            hyp_tok = pred.split()
            refs_tok = [r.split() for r in refs]

        if not hyp_tok or not any(refs_tok):
            return 0.0

        return float(meteor_score(refs_tok, hyp_tok))
    except Exception:
        # Final fallback to stay robust during bulk evaluation
        return 0.0


def compute_bertscore_f1_batch(preds: List[str], refs: List[str], *, model: str, batch_size: int = 16) -> List[float]:
    """
    Compute BERTScore F1 for aligned lists (preds[i], refs[i]).
    This will download the model on first use (Transformers cache).
    """
    from bert_score import score as bert_score

    if not preds:
        return []
    P, R, F1 = bert_score(
        preds,
        refs,
        model_type=model,
        lang="en",
        batch_size=batch_size,
        verbose=False,
    )
    return [float(x.item()) for x in F1]


def aggregate_reward(per_metric: Dict[str, float], weights: Dict[str, float]) -> float:
    total_w = 0.0
    s = 0.0
    for k, w in weights.items():
        if k not in per_metric:
            continue
        s += float(w) * float(per_metric[k])
        total_w += float(w)
    return s / total_w if total_w > 0 else 0.0


def parse_weights(spec: str) -> Dict[str, float]:
    """
    Example: "rougeL:0.34,bertscore_f1:0.33,meteor:0.33"
    """
    out: Dict[str, float] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out


