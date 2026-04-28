# Coding Model Comparison — Frontier Models, April 2026

**Analysis date:** 2026-04-30
**Models compared:** GLM 5.1, Kimi K2.6, MiniMax M2.7, Qwen 3.6 Plus, DeepSeek V4 Pro, DeepSeek V4 Flash, Claude Opus 4.7
**Access:** All available either via OpenCode Zen or direct API.

---

## SWE-bench Verified (500 human-validated GitHub issues — the standard agentic coding bench)

| Model | SWE-bench Verified | SWE-bench Pro | Notes |
|---|---|---|---|
| **Claude Opus 4.7** | **87.6%** | **64.3%** | Clear leader on both, but priciest |
| DeepSeek V4 Pro (1.6T MoE / 49B active) | 80.6% | — | Codeforces 3,206 (highest comp-prog score at release) |
| Kimi K2.6 (1T MoE) | 80.2% | 58.6% | Open-weight; designed for long agentic runs (300 sub-agents × 4k steps) |
| DeepSeek V4 Flash (284B / 13B active) | 79.0% | — | The cheap-and-fast V4 sibling |
| Qwen3.6 Plus (closed/API tier) | 78.8% | — | 1M context, ~2-3× faster tok/s than Opus 4.6, "zero flaky tool calls" |
| MiniMax M2.7 (~10B active) | ~78% | 56.2% (SWE-Pro)\* | Open weights; 50–60× cheaper than Opus; Terminal-Bench 2: 57.0% |
| GLM 5.1 (744B MoE) | ~77.8% (base GLM-5)† | 58.4% | First open-source model to *lead* SWE-bench Pro briefly; surpassed by Opus 4.7 |

\* MiniMax cites "SWE-Pro" — likely the same benchmark, but their internal eval; treat with mild caveat.
† GLM-5.1's specific Verified score wasn't published directly — the 77.8% is the base GLM-5 model. Z.ai promoted SWE-bench *Pro* (58.4%) as the headline.

---

## Practical positioning

- **Claude Opus 4.7** — best raw quality; use it when correctness matters more than latency/cost (production migrations, gnarly bugfixes, multi-file refactors). The ~7-pt jump from 4.6 is mostly fault localization + multi-file edit coherence.
- **DeepSeek V4 Pro** — the only open-weight model that's truly within shouting distance of Opus 4.7 on Verified. Good when you want frontier coding without API lock-in.
- **DeepSeek V4 Flash** — punches well above its weight (1.6 pts behind V4 Pro). Best Verified-per-dollar in the open-weight tier.
- **Kimi K2.6** — *long-horizon* coding work (50+ step tasks not losing the plot). Pricing is the killer feature: $0.60 input / $2.50 output per 1M tokens.
- **Qwen3.6 Plus** — the speed champion. ~78.8% Verified at 2-3× Opus's tokens/sec, plus 1M context. Use it for fast iteration loops or large repo grokking. Tool-call stability is a real edge.
- **MiniMax M2.7** — extreme cost/perf. Hits ~78% Verified with only 10B active params. Use it when you need quality cheap and at scale (test generation, batch refactors, agentic swarms).
- **GLM 5.1** — middle of this pack on coding specifically. Strong all-rounder but not a clear coding-best-in-class for any niche; pick it more for breadth than coding peak.

---

## Recommendations by use case

| Task | First pick | Second pick | Why |
|---|---|---|---|
| Hardest production bugfix or one-shot multi-file edit | Opus 4.7 | DeepSeek V4 Pro | Quality gap is real (7+ pts on Verified) |
| Long agentic session (50+ steps) | Kimi K2.6 | Opus 4.7 | K2.6 explicitly tuned for it; cheap enough to retry |
| Tight feedback loop / IDE autocomplete-adjacent | Qwen3.6 Plus | DeepSeek V4 Flash | Speed + 1M context |
| Bulk code generation (tests, refactors at scale) | MiniMax M2.7 | DeepSeek V4 Flash | Cheapest quality tier |
| Coding from local Sparks (no API) | DeepSeek V4 Pro | Kimi K2.6 / GLM 5.1 | Open weights, frontier-class |

---

## Bottom line

**Opus 4.7 is still the quality king for hard problems**, but the open-weight gap has narrowed enough that for >70% of coding tasks, DeepSeek V4 Pro or Kimi K2.6 will produce results indistinguishable from Opus while costing 5-30× less. Qwen3.6 Plus and MiniMax M2.7 are the pragmatic daily drivers.

---

## Sources

- [Claude Opus 4.7 Benchmarks Explained — Vellum](https://www.vellum.ai/blog/claude-opus-4-7-benchmarks-explained)
- [Lightspeed on X: Claude Opus 4.7 +11% SWE-bench Pro, 87.6% Verified](https://x.com/lightspeedvp/status/2044800343041573313)
- [DeepSeek V4-Pro Review — buildfastwithai](https://www.buildfastwithai.com/blogs/deepseek-v4-pro-review-2026)
- [DeepSeek V4 Flash Review — buildfastwithai](https://www.buildfastwithai.com/blogs/deepseek-v4-flash-review-2026)
- [Kimi K2.6 Tech Blog](https://www.kimi.com/blog/kimi-k2-6)
- [Kimi K2.6 Review — CodeRouter](https://www.coderouter.io/blog/kimi-k2-6-review-coding-benchmarks-2026)
- [MiniMax M2.7 — Self-Evolving Agent Model](https://www.minimax.io/news/minimax-m27-en)
- [MiniMax M2.7 — MarkTechPost](https://www.marktechpost.com/2026/04/12/minimax-just-open-sourced-minimax-m2-7-a-self-evolving-agent-model-that-scores-56-22-on-swe-pro-and-57-0-on-terminal-bench-2/)
- [Qwen3.6 Plus Benchmarks — BenchLM](https://benchlm.ai/models/qwen3-6-plus)
- [Qwen3.6 Plus Review — MindStudio](https://www.mindstudio.ai/blog/qwen-3-6-plus-review-agentic-coding-model)
- [GLM-5.1 — Effloow](https://effloow.com/articles/glm-5-1-744b-moe-open-source-swebench-guide-2026)
- [GLM-5.1 Benchmarks Breakdown — Lushbinary](https://lushbinary.com/blog/glm-5-1-benchmarks-breakdown-swe-bench-pro-nl2repo-cybergym/)
