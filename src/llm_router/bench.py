"""Benchmark tool for comparing LLM inference across targets."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import click
import httpx
from pydantic import BaseModel, Field

from llm_router.config import load_registry

DEFAULT_PROMPT = "Explain the concept of recursion in programming. Give a simple example."


# ── Pydantic models for results ──────────────────────────────────────────


class BenchResult(BaseModel):
    """Single benchmark iteration result."""

    iteration: int
    ttft_ms: float | None = None
    tokens_per_sec: float | None = None
    prompt_tok_per_sec: float | None = None
    total_time_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class TargetStats(BaseModel):
    """Aggregated statistics for one target."""

    target: str
    model_name: str
    runs: int = 0
    successful: int = 0
    failed: int = 0
    ttft_ms: dict[str, float] = Field(default_factory=dict)
    tokens_per_sec: dict[str, float] = Field(default_factory=dict)
    prompt_tok_per_sec: dict[str, float] = Field(default_factory=dict)
    total_time_ms: dict[str, float] = Field(default_factory=dict)
    results: list[BenchResult] = Field(default_factory=list)


class BenchReport(BaseModel):
    """Full benchmark report."""

    prompt: str
    iterations: int
    warmup: int
    max_tokens: int
    temperature: float
    targets: list[TargetStats]


# ── Target resolution ────────────────────────────────────────────────────


@dataclass
class BenchTarget:
    """Resolved benchmark target with URL and model name."""

    label: str
    api_base: str
    model_name: str

    @classmethod
    def from_spec(
        cls,
        spec: str,
        registry_path: Path | None,
        model_name_override: str | None,
    ) -> BenchTarget:
        if spec.startswith("http://") or spec.startswith("https://"):
            api_base = spec.rstrip("/")
            model_name = model_name_override or _discover_model(api_base)
            return cls(label=spec, api_base=api_base, model_name=model_name)

        # Registry model ID
        registry = load_registry(registry_path)
        if spec not in registry.models:
            raise click.ClickException(f"Model {spec!r} not found in registry")
        model = registry.models[spec]
        api_base = registry.get_api_base(spec)
        model_name = model_name_override or model.hf_repo
        return cls(label=spec, api_base=api_base, model_name=model_name)


def _discover_model(api_base: str) -> str:
    """GET /v1/models to auto-discover the served model name."""
    url = api_base.rstrip("/")
    if not url.endswith("/v1"):
        url = url.rstrip("/") + "/v1"
    try:
        resp = httpx.get(f"{url}/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if models:
            return models[0]["id"]
    except Exception as e:
        raise click.ClickException(f"Failed to discover model at {api_base}: {e}") from e
    raise click.ClickException(f"No models found at {api_base}/models")


# ── Benchmark execution ──────────────────────────────────────────────────


def run_single_benchmark(
    api_base: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    iteration: int,
) -> BenchResult:
    """Run a single streaming completion and measure timing."""
    url = api_base.rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    url = f"{url}/chat/completions"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    prompt_tokens = 0
    completion_tokens = 0
    t_first_token: float | None = None

    try:
        t_start = time.perf_counter()

        with httpx.Client(timeout=120) as client, client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()

            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                chunk = json.loads(data_str)

                # Record TTFT on first content chunk
                if t_first_token is None:
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        t_first_token = time.perf_counter()

                # Capture usage from final chunk
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else None

        if t_first_token and completion_tokens > 0:
            gen_time = t_end - t_first_token
            tps = completion_tokens / gen_time if gen_time > 0 else 0.0
        else:
            tps = None

        # Approximate prompt processing speed: prompt_tokens / TTFT
        prompt_tps = prompt_tokens / (ttft_ms / 1000) if ttft_ms and prompt_tokens > 0 else None

        return BenchResult(
            iteration=iteration,
            ttft_ms=ttft_ms,
            tokens_per_sec=tps,
            prompt_tok_per_sec=prompt_tps,
            total_time_ms=total_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        t_end = time.perf_counter()
        return BenchResult(
            iteration=iteration,
            total_time_ms=(t_end - t_start) * 1000,
            error=str(e),
        )


def bench_target(
    target: BenchTarget,
    prompt: str,
    iterations: int,
    warmup: int,
    max_tokens: int,
    temperature: float,
) -> list[BenchResult]:
    """Run warmup + benchmark iterations for a single target."""
    # Warmup
    for i in range(warmup):
        click.echo(f"  warmup {i + 1}/{warmup}...", nl=False)
        run_single_benchmark(target.api_base, target.model_name, prompt, max_tokens, temperature, i)
        click.echo(" done")

    # Benchmark
    results: list[BenchResult] = []
    for i in range(iterations):
        click.echo(f"  run {i + 1}/{iterations}...", nl=False)
        result = run_single_benchmark(
            target.api_base, target.model_name, prompt, max_tokens, temperature, i + 1
        )
        if result.error:
            click.echo(f" ERROR: {result.error}")
        else:
            pp = f" pp={result.prompt_tok_per_sec:.0f}t/s" if result.prompt_tok_per_sec else ""
            click.echo(
                f" ttft={result.ttft_ms:.0f}ms{pp}"
                f" gen={result.tokens_per_sec:.1f}t/s"
                f" total={result.total_time_ms:.0f}ms"
                f" tokens={result.completion_tokens}"
            )
        results.append(result)

    return results


# ── Statistics ───────────────────────────────────────────────────────────


def _stat_dict(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of values."""
    if not values:
        return {}
    result = {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    if len(values) >= 2:
        result["stdev"] = statistics.stdev(values)
    return result


def compute_stats(target: BenchTarget, results: list[BenchResult]) -> TargetStats:
    """Aggregate per-iteration results into statistics."""
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    ttft_values = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    tps_values = [r.tokens_per_sec for r in successful if r.tokens_per_sec is not None]
    pp_values = [r.prompt_tok_per_sec for r in successful if r.prompt_tok_per_sec is not None]
    total_values = [r.total_time_ms for r in successful]

    return TargetStats(
        target=target.label,
        model_name=target.model_name,
        runs=len(results),
        successful=len(successful),
        failed=len(failed),
        ttft_ms=_stat_dict(ttft_values),
        tokens_per_sec=_stat_dict(tps_values),
        prompt_tok_per_sec=_stat_dict(pp_values),
        total_time_ms=_stat_dict(total_values),
        results=results,
    )


# ── Output ───────────────────────────────────────────────────────────────


def print_results_table(all_stats: list[TargetStats]) -> None:
    """Print a comparison table to the terminal."""
    click.echo()
    click.echo("=" * 78)
    click.echo("BENCHMARK RESULTS")
    click.echo("=" * 78)

    # Comparison table (medians)
    click.echo()
    header = (
        f"{'Target':<28} {'TTFT(ms)':>9} {'PP(t/s)':>9}"
        f" {'Gen(t/s)':>9} {'Total(ms)':>10} {'OK/Fail':>8}"
    )
    click.echo(header)
    click.echo("-" * 78)

    for s in all_stats:
        ttft = f"{s.ttft_ms.get('median', 0):.0f}" if s.ttft_ms else "n/a"
        pp = f"{s.prompt_tok_per_sec.get('median', 0):.0f}" if s.prompt_tok_per_sec else "n/a"
        tps = f"{s.tokens_per_sec.get('median', 0):.1f}" if s.tokens_per_sec else "n/a"
        total = f"{s.total_time_ms.get('median', 0):.0f}" if s.total_time_ms else "n/a"
        ok_fail = f"{s.successful}/{s.failed}"
        click.echo(f"{s.target:<28} {ttft:>9} {pp:>9} {tps:>9} {total:>10} {ok_fail:>8}")

    # Per-target detail
    for s in all_stats:
        click.echo()
        click.echo(f"── {s.target} ({s.model_name}) ──")
        for label, stats in [
            ("TTFT (ms)", s.ttft_ms),
            ("Prompt (tok/s)", s.prompt_tok_per_sec),
            ("Gen (tok/s)", s.tokens_per_sec),
            ("Total (ms)", s.total_time_ms),
        ]:
            if not stats:
                click.echo(f"  {label}: no data")
                continue
            parts = [f"{k}={v:.1f}" for k, v in stats.items()]
            click.echo(f"  {label}: {', '.join(parts)}")

        if s.failed > 0:
            errors = [r for r in s.results if r.error]
            for r in errors:
                click.echo(f"  ERROR (iter {r.iteration}): {r.error}")

    click.echo()


def write_json_report(report: BenchReport, path: str) -> None:
    """Write the full report as JSON."""
    with open(path, "w") as f:
        f.write(report.model_dump_json(indent=2))
    click.echo(f"Report written to {path}")


# ── CLI ──────────────────────────────────────────────────────────────────


@click.command("llm-router-bench")
@click.option(
    "-t",
    "--target",
    "targets",
    multiple=True,
    required=True,
    help="Model ID from registry or raw URL (repeatable)",
)
@click.option("-p", "--prompt", "prompt", default=DEFAULT_PROMPT, help="Prompt to benchmark")
@click.option("-n", "--iterations", default=5, help="Runs per target")
@click.option("-w", "--warmup", default=1, help="Warmup runs (discarded)")
@click.option("--max-tokens", default=256, help="Max completion tokens")
@click.option("--temperature", default=0.0, help="Sampling temperature")
@click.option("-o", "--output", "output_path", default=None, help="JSON output file")
@click.option("-r", "--registry", "registry_path", default=None, help="Path to models.yaml")
@click.option("--model-name", default=None, help="Override model name for raw URL targets")
def cli(
    targets: tuple[str, ...],
    prompt: str,
    iterations: int,
    warmup: int,
    max_tokens: int,
    temperature: float,
    output_path: str | None,
    registry_path: str | None,
    model_name: str | None,
) -> None:
    """Benchmark LLM inference across multiple targets."""
    reg_path = Path(registry_path) if registry_path else None

    # Resolve targets
    resolved: list[BenchTarget] = []
    for spec in targets:
        t = BenchTarget.from_spec(spec, reg_path, model_name)
        click.echo(f"Target: {t.label} -> {t.api_base} (model: {t.model_name})")
        resolved.append(t)

    click.echo(
        f"\nConfig: {iterations} iterations, {warmup} warmup,"
        f" {max_tokens} max_tokens, temp={temperature}"
    )
    click.echo()

    # Run benchmarks
    all_stats: list[TargetStats] = []
    for target in resolved:
        click.echo(f"Benchmarking: {target.label}")
        results = bench_target(target, prompt, iterations, warmup, max_tokens, temperature)
        stats = compute_stats(target, results)
        all_stats.append(stats)

    # Output
    print_results_table(all_stats)

    if output_path:
        report = BenchReport(
            prompt=prompt,
            iterations=iterations,
            warmup=warmup,
            max_tokens=max_tokens,
            temperature=temperature,
            targets=all_stats,
        )
        write_json_report(report, output_path)


if __name__ == "__main__":
    cli()
