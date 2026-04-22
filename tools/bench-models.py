#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai>=1.60",
#     "httpx>=0.28",
#     "rich>=13.0",
# ]
# ///
"""Benchmark coding performance across all enabled models on the LiteLLM proxy.

Usage:
    uv run python tools/bench-models.py
    uv run python tools/bench-models.py --url http://euclid.local:4010
    uv run python tools/bench-models.py --model coder
    uv run python tools/bench-models.py --timeout 120 --max-tokens 1024
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import httpx
from openai import OpenAI
from rich.console import Console
from rich.table import Table

# Models whose capabilities make them unsuitable for chat completions.
# We skip these even if they appear in /v1/models.
NON_CHAT_TAGS = {"embedding", "reranker", "stt", "tts", "image_gen", "music_gen", "gui_agent", "image_edit"}

# Model ID substrings that strongly suggest non-chat models.
NON_CHAT_SUBSTRINGS = (
    "embed",
    "rerank",
    "whisper",
    "kokoro",
    "orpheus",
    "flux",
    "tts",
    "stt",
    "acestep",
    "music",
    "ui-tars",
    "image-edit",
    "qwen-image-edit",
    "clip",
    "codet5p",
)

# Auto-router models wrap other models -- skip to avoid benchmarking
# the router overhead rather than a real model.
AUTO_ROUTER_PREFIXES = ("auto", "auto-", "coder-resilient")

DEFAULT_PROMPT = (
    "Write a Python async web scraper class with rate limiting, retry logic, "
    "and proper error handling. Use aiohttp and include type hints."
)


@dataclass
class BenchResult:
    """Result for a single model benchmark run."""

    model: str
    ttft_s: float | None = None
    total_s: float | None = None
    tokens: int = 0
    tok_per_sec: float | None = None
    error: str | None = None


@dataclass
class ModelInfo:
    """Minimal model info from /v1/models."""

    id: str
    owned_by: str = ""
    extra: dict[str, object] = field(default_factory=dict)


def fetch_models(base_url: str, timeout: float) -> list[ModelInfo]:
    """Fetch the list of models from the LiteLLM proxy."""
    url = base_url.rstrip("/") + "/v1/models"
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    models: list[ModelInfo] = []
    for entry in data.get("data", []):
        models.append(
            ModelInfo(
                id=entry["id"],
                owned_by=entry.get("owned_by", ""),
                extra=entry,
            )
        )
    return models


def should_skip(model: ModelInfo) -> str | None:
    """Return a reason string if this model should be skipped, else None."""
    mid_lower = model.id.lower()

    # Skip auto-routers
    for prefix in AUTO_ROUTER_PREFIXES:
        if mid_lower == prefix or mid_lower.startswith(prefix + "-"):
            # Exact match "auto" or starts with "auto-" / "coder-resilient-"
            if mid_lower in ("auto", "auto-free", "auto-full", "coder-resilient"):
                return "auto-router"

    # Skip by known non-chat substrings
    for substr in NON_CHAT_SUBSTRINGS:
        if substr in mid_lower:
            return f"non-chat ({substr})"

    return None


def bench_model(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    console: Console,
) -> BenchResult:
    """Benchmark a single model using streaming to measure TTFT."""
    result = BenchResult(model=model)

    try:
        t_start = time.perf_counter()
        t_first: float | None = None
        token_count = 0

        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
            timeout=timeout,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if t_first is None:
                    t_first = time.perf_counter()
                # Approximate token count from chunk content.
                # The OpenAI SDK streaming doesn't always give usage,
                # so we count non-empty content chunks as a rough proxy.
                # Each chunk is typically one token for most backends.
                token_count += 1

        t_end = time.perf_counter()

        result.total_s = t_end - t_start
        result.tokens = token_count

        if t_first is not None:
            result.ttft_s = t_first - t_start
            gen_time = t_end - t_first
            if gen_time > 0 and token_count > 1:
                # Exclude the first token (that's TTFT, not generation)
                result.tok_per_sec = (token_count - 1) / gen_time
            elif token_count == 1:
                result.tok_per_sec = 0.0
        else:
            result.error = "no content received"

    except KeyboardInterrupt:
        raise
    except Exception as e:
        err_msg = str(e)
        # Truncate very long error messages
        if len(err_msg) > 200:
            err_msg = err_msg[:200] + "..."
        result.error = err_msg

    return result


def fmt_float(val: float | None, fmt: str = ".1f") -> str:
    """Format a float or return '-' if None."""
    if val is None:
        return "-"
    return f"{val:{fmt}}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark coding performance across LiteLLM proxy models",
    )
    parser.add_argument(
        "--url",
        default="http://euclid.local:4010",
        help="LiteLLM proxy base URL (default: http://euclid.local:4010)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model name to test (default: all enabled models)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Timeout per model in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override the default coding prompt",
    )
    args = parser.parse_args()

    console = Console()
    prompt = args.prompt or DEFAULT_PROMPT
    base_url = args.url.rstrip("/")

    # Fetch models from proxy
    console.print(f"\nFetching models from [bold]{base_url}[/bold]...")
    try:
        models = fetch_models(base_url, timeout=30)
    except Exception as e:
        console.print(f"[red]Failed to fetch models: {e}[/red]")
        sys.exit(1)

    if not models:
        console.print("[red]No models found at the proxy.[/red]")
        sys.exit(1)

    console.print(f"Found {len(models)} model(s) on the proxy.\n")

    # Filter to requested model or apply skip logic
    if args.model:
        matched = [m for m in models if m.id == args.model]
        if not matched:
            # Try substring match
            matched = [m for m in models if args.model in m.id]
        if not matched:
            console.print(f"[red]Model '{args.model}' not found. Available models:[/red]")
            for m in sorted(models, key=lambda x: x.id):
                console.print(f"  {m.id}")
            sys.exit(1)
        to_test = matched
        skipped: list[tuple[str, str]] = []
    else:
        to_test = []
        skipped = []
        for m in models:
            reason = should_skip(m)
            if reason:
                skipped.append((m.id, reason))
            else:
                to_test.append(m)

    # Sort alphabetically for consistent ordering
    to_test.sort(key=lambda m: m.id)

    if skipped:
        console.print(f"Skipping {len(skipped)} non-chat model(s):")
        for mid, reason in sorted(skipped):
            console.print(f"  [dim]{mid}[/dim] ({reason})")
        console.print()

    console.print(f"Benchmarking {len(to_test)} model(s):")
    for m in to_test:
        console.print(f"  {m.id}")
    console.print()
    console.print(f"Prompt: [dim]{prompt[:80]}{'...' if len(prompt) > 80 else ''}[/dim]")
    console.print(f"Max tokens: {args.max_tokens}, Timeout: {args.timeout}s\n")

    # Create OpenAI client pointed at the proxy
    client = OpenAI(
        base_url=base_url + "/v1",
        api_key="not-needed",  # LiteLLM proxy doesn't require a key by default
    )

    # Run benchmarks sequentially
    results: list[BenchResult] = []
    for i, m in enumerate(to_test, 1):
        console.print(f"[{i}/{len(to_test)}] [bold]{m.id}[/bold] ... ", end="")
        result = bench_model(
            client=client,
            model=m.id,
            prompt=prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            console=console,
        )
        if result.error:
            console.print(f"[red]FAILED[/red] ({result.error})")
        else:
            ttft_ms = result.ttft_s * 1000 if result.ttft_s is not None else 0
            console.print(
                f"[green]OK[/green]  "
                f"TTFT={ttft_ms:.0f}ms  "
                f"tokens={result.tokens}  "
                f"tok/s={fmt_float(result.tok_per_sec)}  "
                f"total={fmt_float(result.total_s)}s"
            )
        results.append(result)

    # Build summary table sorted by tok/s (successful first, then failed)
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    # Sort successful by tok/s descending
    successful.sort(key=lambda r: r.tok_per_sec or 0, reverse=True)

    console.print()
    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Model", style="bold", min_width=20)
    table.add_column("TTFT", justify="right", min_width=8)
    table.add_column("Total", justify="right", min_width=8)
    table.add_column("Tokens", justify="right", min_width=8)
    table.add_column("Tok/s", justify="right", min_width=8, style="cyan")
    table.add_column("Status", justify="center", min_width=8)

    for r in successful:
        ttft_str = f"{r.ttft_s * 1000:.0f}ms" if r.ttft_s is not None else "-"
        total_str = f"{r.total_s:.1f}s" if r.total_s is not None else "-"
        tps_str = f"{r.tok_per_sec:.1f}" if r.tok_per_sec is not None else "-"
        table.add_row(
            r.model,
            ttft_str,
            total_str,
            str(r.tokens),
            tps_str,
            "[green]OK[/green]",
        )

    for r in failed:
        err_short = (r.error or "unknown")[:60]
        table.add_row(
            r.model,
            "-",
            "-",
            "-",
            "-",
            f"[red]{err_short}[/red]",
        )

    console.print(table)

    # Summary line
    console.print(
        f"\n{len(successful)} succeeded, {len(failed)} failed "
        f"out of {len(results)} tested.\n"
    )


if __name__ == "__main__":
    main()
