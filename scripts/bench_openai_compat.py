from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark an OpenAI-compatible chat endpoint (e.g., SGLang/vLLM) for TTFT/latency/throughput."
    )
    p.add_argument("--base_url", type=str, required=True, help="OpenAI-compatible base URL, e.g. http://localhost:9000/v1")
    p.add_argument("--model", type=str, required=True, help="Model id as returned by /v1/models")
    p.add_argument("--concurrency", type=str, default="1,2,4,8,16", help="Comma-separated concurrencies to test.")
    p.add_argument("--requests_per_concurrency", type=int, default=20, help="Number of requests per concurrency level.")
    p.add_argument("--max_tokens", type=int, default=256, help="Max output tokens for each request.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompt", type=str, default="", help="User prompt. If empty, uses a built-in prompt.")
    p.add_argument("--prompt_chars", type=int, default=0, help="If >0, override prompt with a synthetic prompt of this many chars.")
    p.add_argument("--stream", action="store_true", help="If set, use streaming to measure TTFT.")
    p.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout seconds.")
    return p.parse_args()


def _normalize_base_url(s: str) -> str:
    s = str(s or "").strip().rstrip("/")
    # Expect OpenAI-compat base_url to include /v1 in this repo.
    # If user passes host:port, we append /v1 for convenience.
    if not s.endswith("/v1"):
        s = s + "/v1"
    return s


def _make_prompt(args: argparse.Namespace) -> str:
    if int(args.prompt_chars) > 0:
        n = int(args.prompt_chars)
        core = "HotPotQA RAG benchmark. "
        return (core * ((n // len(core)) + 1))[:n]
    if str(args.prompt or "").strip():
        return str(args.prompt)
    # Default prompt: stable, English, moderately long.
    return (
        "You are a helpful assistant.\n\n"
        "Answer the question concisely.\n\n"
        "Question: In two sentences, explain what retrieval-augmented generation (RAG) is, "
        "and give one practical benefit.\n"
    )


@dataclass
class OneResult:
    ok: bool
    ttft_s: Optional[float]
    latency_s: float
    completion_tokens: Optional[int]
    error: Optional[str] = None


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(len(xs) - 1, f + 1)
    if f == c:
        return float(xs[f])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


async def _one_call(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
    timeout_s: float,
) -> OneResult:
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    completion_tokens: Optional[int] = None
    try:
        if stream:
            # Streaming TTFT: measure first chunk with content.
            stream_obj = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    stream=True,
                ),
                timeout=timeout_s,
            )
            async for ev in stream_obj:
                # OpenAI style: ev.choices[0].delta.content
                try:
                    delta = ev.choices[0].delta
                    piece = getattr(delta, "content", None)
                except Exception:
                    piece = None
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
                    # keep consuming to finish request (for fair load)
            t1 = time.perf_counter()
            return OneResult(ok=True, ttft_s=ttft, latency_s=t1 - t0, completion_tokens=None, error=None)

        # Non-streaming: measure latency and throughput tokens/s using usage.completion_tokens when available.
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                stream=False,
            ),
            timeout=timeout_s,
        )
        t1 = time.perf_counter()
        try:
            usage = getattr(resp, "usage", None)
            completion_tokens = int(getattr(usage, "completion_tokens", None)) if usage is not None else None
        except Exception:
            completion_tokens = None
        return OneResult(ok=True, ttft_s=None, latency_s=t1 - t0, completion_tokens=completion_tokens, error=None)

    except Exception as e:
        t1 = time.perf_counter()
        return OneResult(ok=False, ttft_s=ttft, latency_s=t1 - t0, completion_tokens=completion_tokens, error=repr(e))


async def _run_level(
    *,
    concurrency: int,
    n_reqs: int,
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
    timeout_s: float,
) -> List[OneResult]:
    sem = asyncio.Semaphore(int(concurrency))

    async def _wrapped() -> OneResult:
        async with sem:
            return await _one_call(
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
                timeout_s=timeout_s,
            )

    tasks = [asyncio.create_task(_wrapped()) for _ in range(int(n_reqs))]
    return list(await asyncio.gather(*tasks))


def _summarize(results: List[OneResult]) -> Dict[str, Any]:
    oks = [r for r in results if r.ok]
    errs = [r for r in results if not r.ok]
    lat = [r.latency_s for r in oks]
    ttft = [r.ttft_s for r in oks if r.ttft_s is not None]
    toks = [r.completion_tokens for r in oks if r.completion_tokens is not None]

    out: Dict[str, Any] = {
        "n": len(results),
        "ok": len(oks),
        "err": len(errs),
        "latency_s": {
            "p50": _percentile(lat, 50) if lat else None,
            "p95": _percentile(lat, 95) if lat else None,
            "mean": (sum(lat) / len(lat)) if lat else None,
        },
        "ttft_s": {
            "p50": _percentile(ttft, 50) if ttft else None,
            "p95": _percentile(ttft, 95) if ttft else None,
            "mean": (sum(ttft) / len(ttft)) if ttft else None,
        },
        "completion_tokens": {
            "sum": int(sum(toks)) if toks else None,
            "mean": (sum(toks) / len(toks)) if toks else None,
        },
    }

    # Overall throughput: total completion tokens / total wall time across OK requests
    # For streaming runs we don't reliably have token usage -> set None.
    if toks and lat:
        total_toks = float(sum(toks))
        total_time = float(sum(lat))
        out["throughput_tokens_per_s"] = (total_toks / total_time) if total_time > 0 else None
    else:
        out["throughput_tokens_per_s"] = None

    # Include a small sample error to aid debugging
    if errs:
        out["sample_error"] = errs[0].error
    return out


async def main_async() -> None:
    args = _parse_args()
    base_url = _normalize_base_url(args.base_url)
    model = str(args.model)
    concs = [int(x) for x in str(args.concurrency).split(",") if str(x).strip()]
    prompt = _make_prompt(args)

    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")

    report: Dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "stream": bool(args.stream),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "prompt_chars": len(prompt),
        "levels": [],
    }

    for c in concs:
        t0 = time.perf_counter()
        results = await _run_level(
            concurrency=int(c),
            n_reqs=int(args.requests_per_concurrency),
            client=client,
            model=model,
            prompt=prompt,
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            stream=bool(args.stream),
            timeout_s=float(args.timeout),
        )
        t1 = time.perf_counter()
        summ = _summarize(results)
        summ["concurrency"] = int(c)
        summ["wall_s"] = float(t1 - t0)
        report["levels"].append(summ)

        # human-friendly line
        lat_p50 = summ["latency_s"]["p50"]
        lat_p95 = summ["latency_s"]["p95"]
        ttft_p50 = summ["ttft_s"]["p50"]
        ttft_p95 = summ["ttft_s"]["p95"]
        tps = summ.get("throughput_tokens_per_s", None)
        print(
            f"[c={c}] ok={summ['ok']}/{summ['n']} "
            f"lat_p50={lat_p50:.3f}s lat_p95={lat_p95:.3f}s "
            + (f"ttft_p50={ttft_p50:.3f}s ttft_p95={ttft_p95:.3f}s " if ttft_p50 is not None else "")
            + (f"toks/s={tps:.1f}" if tps is not None else "toks/s=N/A")
        )

    print("\n--- JSON report ---")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

