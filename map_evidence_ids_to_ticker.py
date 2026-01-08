from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


_WS_RE = re.compile(r"\s+")
_TICKER_RE = re.compile(r"\bTicker:\s*([A-Z0-9]{1,10})\b")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip()).lower()


def _extract_candidate_tickers_from_evidence(text: str) -> List[str]:
    return list(dict.fromkeys(_TICKER_RE.findall(text or "")))


def _fragments(text: str) -> List[str]:
    """
    Create a small set of normalized fragments to check containment.
    We prefer short-ish stable pieces to reduce false positives.
    """
    t = (text or "").strip()
    if not t:
        return []
    # Drop very common separators that may pollute matching.
    t = t.replace("================================================================================", " ")
    t = _WS_RE.sub(" ", t).strip()
    # Take a few windows from the beginning (often contains the key facts).
    outs = []
    for n in (120, 180, 240):
        if len(t) >= n:
            outs.append(_norm(t[:n]))
    # Also take a couple of sentences.
    # (Naive split; good enough for matching.)
    sents = re.split(r"(?<=[.!?])\s+", t)
    for s in sents[:3]:
        s = s.strip()
        if 60 <= len(s) <= 240:
            outs.append(_norm(s))
    # Dedup and filter tiny
    outs = [x for x in dict.fromkeys(outs) if len(x) >= 40]
    return outs[:8]


def _ticker_local_window(text: str, ticker: str, *, window: int = 800) -> str:
    """
    Extract a local window around the first occurrence of "Ticker: {ticker}".
    This helps when the evidence text contains multiple records; the relevant part
    for the matched ticker is usually near its own record block, not at the beginning.
    """
    if not text or not ticker:
        return ""
    m = re.search(rf"\bTicker:\s*{re.escape(ticker)}\b", text)
    if not m:
        return ""
    # Prefer starting at the ticker marker so fragments include the ticker's own record content.
    start = max(0, m.start() - 80)  # small pre-context (e.g., Record #..)
    # Try to end at the next record delimiter if present.
    delim = "================================================================================"
    nxt = text.find(delim, m.end())
    if nxt != -1:
        end = min(len(text), nxt)
    else:
        end = min(len(text), m.end() + window)
    return text[start:end]


def _record_content_hint(local: str) -> str:
    """
    Try to focus matching on the descriptive content (which is what should exist in combined_text),
    not on metadata like 'Record #..' / 'Ticker:' lines.
    """
    if not local:
        return ""
    # Prefer content after 'Description:' if present.
    m = re.search(r"\bDescription:\s*", local)
    if m:
        return local[m.end() : m.end() + 600]
    # Otherwise, try after the first line break following 'Ticker:'.
    m2 = re.search(r"\bTicker:\s*[A-Z0-9]{1,10}\b", local)
    if m2:
        return local[m2.end() : m2.end() + 600]
    return local[:600]


@dataclass
class MatchResult:
    tickers: List[str]
    best_score: int
    considered: int


def match_tickers(
    evidence_text: str,
    *,
    ticker_to_combined_norm: Dict[str, str],
    restrict_to_mentions: bool = True,
    min_score: int = 1,
    mode: str = "all",  # all | best
) -> MatchResult:
    mentioned = _extract_candidate_tickers_from_evidence(evidence_text)
    candidates: Iterable[str]
    if restrict_to_mentions and mentioned:
        candidates = [t for t in mentioned if t in ticker_to_combined_norm]
    else:
        candidates = ticker_to_combined_norm.keys()

    best_score = 0
    best: List[str] = []
    all_good: List[str] = []
    considered = 0

    for tkr in candidates:
        combined = ticker_to_combined_norm.get(tkr, "")
        if not combined:
            continue
        considered += 1
        # Prefer ticker-local fragments when the ticker is explicitly mentioned.
        local = _ticker_local_window(evidence_text, tkr)
        focus = _record_content_hint(local) if local else evidence_text
        frags = _fragments(focus)
        if not frags:
            continue
        score = sum(1 for frag in frags if frag and frag in combined)
        if score >= min_score:
            all_good.append(tkr)
        if score > best_score:
            best_score = score
            best = [tkr]
        elif score == best_score and score > 0:
            best.append(tkr)

    if mode == "best":
        if best_score < min_score:
            return MatchResult(tickers=[], best_score=best_score, considered=considered)
        best = sorted(set(best))
        return MatchResult(tickers=best, best_score=best_score, considered=considered)

    # mode == all
    all_good = sorted(set(all_good))
    return MatchResult(tickers=all_good, best_score=best_score, considered=considered)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str, required=True)
    ap.add_argument("--json_in", type=str, required=True)
    ap.add_argument("--json_out", type=str, required=True)
    ap.add_argument("--report_csv", type=str, required=True)
    ap.add_argument("--restrict_to_mentions", action="store_true", help="Only match against tickers mentioned in evidence text (Ticker: XXX).")
    ap.add_argument("--min_score", type=int, default=1)
    ap.add_argument("--mode", type=str, default="all", help="Matching mode: all (default) or best")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx)
    if "Ticker" not in df.columns or "combined_text" not in df.columns:
        raise ValueError(f"Excel must contain columns Ticker and combined_text. Got: {list(df.columns)}")

    ticker_to_combined_norm: Dict[str, str] = {}
    for _, r in df.iterrows():
        t = str(r["Ticker"]).strip()
        c = str(r["combined_text"] or "")
        ticker_to_combined_norm[t] = _norm(c)

    with open(args.json_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    report_rows: List[Dict[str, str]] = []

    for qi, q in enumerate(data):
        meta = q.get("metadata") or {}
        qid = str(meta.get("question_id") or f"q{qi}")
        contexts = q.get("contexts") or []
        for ci, ctx in enumerate(contexts):
            t = str(ctx.get("type") or "")
            if t not in {"partial_supportive", "fully_supportive"}:
                continue
            evid = str(ctx.get("evidence_ids") or "")
            text = str(ctx.get("text") or "")

            mr = match_tickers(
                text,
                ticker_to_combined_norm=ticker_to_combined_norm,
                restrict_to_mentions=bool(args.restrict_to_mentions),
                min_score=int(args.min_score),
                mode=str(args.mode),
            )

            mapped = ",".join(mr.tickers) if mr.tickers else ""
            # preserve original for audit
            ctx.setdefault("evidence_ids_orig", evid)
            if mapped:
                ctx["evidence_ids"] = mapped

            report_rows.append(
                {
                    "question_id": qid,
                    "q_index": str(qi),
                    "context_index": str(ci),
                    "type": t,
                    "evidence_ids_orig": evid,
                    "evidence_ids_new": mapped or evid,
                    "matched_tickers": mapped,
                    "best_score": str(mr.best_score),
                    "candidates_considered": str(mr.considered),
                }
            )

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open(args.report_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "q_index",
                "context_index",
                "type",
                "evidence_ids_orig",
                "evidence_ids_new",
                "matched_tickers",
                "best_score",
                "candidates_considered",
            ],
        )
        w.writeheader()
        w.writerows(report_rows)


if __name__ == "__main__":
    main()


