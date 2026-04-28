# Proposed Upstream PR: Fix default padding/truncation in `/v1/embeddings`

**Status:** Draft (not yet filed upstream)
**Authored:** 2026-05-01
**Target repo:** [SearchSavior/OpenArc](https://github.com/SearchSavior/OpenArc)
**Target branch:** `main` (currently at `1842fa6`)
**Source branch:** `fix/embeddings-default-padding-truncation`
**Local clone:** `~/code/OpenArc` (delphi)

---

## Title

Fix: default `padding` and `truncation` in `/v1/embeddings` so OpenAI-spec batches don't fail

## Summary

The `/v1/embeddings` handler in `src/server/main.py` constructs a `PreTrainedTokenizerConfig` without overriding the Pydantic defaults (`padding=False`, `truncation=False`). Any OpenAI-spec multi-input batch then fails inside `tokenizer(...)` when it tries to stack variable-length token lists into a single tensor:

```
ValueError: Unable to create tensor, you should probably activate truncation
and/or padding with 'padding=True' 'truncation=True' to have batched tensors
with the same length. Perhaps your features (`input_ids` in this case) have
excessive nesting (inputs type `list` where type `int` is expected).
```

The maintainer's own integration test at `src/tests/test_optimum_emb_integration.py` constructs the config with `padding="longest", truncation=True, max_length=32, return_tensors="pt"`, so the correct shape is already established — it just isn't applied to the OpenAI-spec entry point.

This change brings the `/v1/embeddings` default-construction path in line with that test pattern.

---

## Problem reproduction

Steady-state OpenArc with any embedding model loaded (e.g. `Qwen3-Embedding-4B INT8`):

```bash
# Single input — works
curl http://host:5404/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-4b", "input": "hello"}'
# → 200 OK, one embedding returned

# Single-element list — works
curl http://host:5404/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-4b", "input": ["hello"]}'
# → 200 OK, one embedding returned

# Multi-element list (the actual OpenAI batch shape) — fails
curl http://host:5404/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-4b", "input": ["hello", "world"]}'
# → 500, ValueError as above

# With explicit config — works (proves the underlying engine is fine)
curl http://host:5404/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding-4b",
    "input": ["hello", "world"],
    "config": {
      "text": ["hello", "world"],
      "padding": "longest",
      "truncation": true,
      "return_tensors": "pt"
    }
  }'
# → 200 OK, two embeddings returned
```

The bug is purely in the default-construction path of the OpenAI-spec endpoint, not in the embedding engine.

---

## Root cause

`src/server/main.py` (current `main`):

```python
tok_config = PreTrainedTokenizerConfig(
    text=request.input
)
```

`PreTrainedTokenizerConfig` in `src/server/models/optimum.py` defines the relevant fields with these defaults:

| Field | Default |
|---|---|
| `padding` | `False` |
| `truncation` | `False` |
| `return_tensors` | `"pt"` |
| `max_length` | `None` |

So a multi-input batch reaches the underlying tokenizer with `padding=False, truncation=False`, which is incompatible with stacking variable-length sequences into a single tensor.

---

## Proposed change

```diff
diff --git a/src/server/main.py b/src/server/main.py
index ee5d920..d812538 100644
--- a/src/server/main.py
+++ b/src/server/main.py
@@ -785,7 +785,9 @@ async def embeddings(request: EmbeddingsRequest):
         logger.info(f'"{request.model}" request received')

         tok_config = PreTrainedTokenizerConfig(
-            text=request.input
+            text=request.input,
+            padding="longest",
+            truncation=True,
         )

         if request.config:
```

3 lines added, 1 removed. No new imports, no new dependencies.

---

## Design notes / why this approach

- **`padding="longest"`** matches the maintainer's integration test and pads each batch to the length of its longest member. This is the correct OpenAI-spec batching behavior and a no-op for single-input requests.
- **`truncation=True`** lets the tokenizer respect the model's `model_max_length`, avoiding hard failures on inputs that exceed the model's context.
- **`return_tensors="pt"`** is already the Pydantic default in `PreTrainedTokenizerConfig`, so no change is needed.
- **User overrides via `request.config` continue to work unchanged** — the existing branch right after default construction still replaces `tok_config` wholesale when `request.config` is provided.
- **No behavior change for existing single-input callers** — `padding="longest"` with one input is a no-op, and `truncation=True` only kicks in when an input exceeds `model_max_length`, which is desirable.

---

## Test plan

1. Existing unit + integration tests under `src/tests/` continue to pass (no test changes needed; the integration test already passes the same fields explicitly).
2. Manual smoke against a running OpenArc with a Qwen3-Embedding-4B INT8 model:
   - Single string input — 200 OK.
   - Single-element list — 200 OK.
   - **Multi-element batch (e.g. `["hello", "world"]`) — 200 OK with two embeddings (regression target).**
   - Inputs with mixed lengths — 200 OK; longest input determines pad length.
   - Inputs longer than `model_max_length` — 200 OK; truncated, not error.
3. `request.config` override path:
   - Pass an explicit `config` with `padding=False` (intentionally) — server should honor the override (i.e. the existing pre-fix failure mode must still be reachable when explicitly requested, since users may want strict no-pad behavior for offline preprocessing).

### Validation results (2026-05-01)

Fix validated end-to-end against `openarc:battlemage-fix-padding-20260501` on euclid (Arc B50 Pro), Qwen3-Embedding-4B INT8:

| Test | Input | Result |
|---|---|---|
| 1 | `"hello world"` (string) | 200 OK, one 2560-dim embedding |
| 2 | `["only one"]` (single-element list) | 200 OK, one 2560-dim embedding |
| 3 | `["hello","world","this is a longer third input"]` (mixed lengths) | 200 OK, three 2560-dim embeddings |
| 4 | `["a","b","c","d","e"]` (5-input batch) | 200 OK, five distinct 2560-dim embeddings, indices `[0,1,2,3,4]` |

Pre-patch: tests 3 and 4 failed with `ValueError: Unable to create tensor...` and the worker queue stuck (every subsequent request hung). Post-patch: all four tests succeed; no worker hang observed.

---

## Out of scope (separate follow-up)

A second, larger bug exists in OpenArc: when a tokenizer error occurs inside `WorkerOps.infer_emb`, subsequent requests to that worker hang indefinitely (15s+ timeouts) until container restart. The exception is caught and stored in `packet.response`, but a slot or semaphore in the dispatch/queue layer is not released. **This PR does not attempt that fix** — it fixes the most common trigger of the hang (the padding/truncation error), but the underlying queue-state issue still exists for any future tokenizer-level error. Tracked separately.

---

## References

- Maintainer's own embedding-config pattern: [`src/tests/test_optimum_emb_integration.py`](https://github.com/SearchSavior/OpenArc/blob/main/src/tests/test_optimum_emb_integration.py) — uses `padding="longest", truncation=True, max_length=32, return_tensors="pt"`.
- `PreTrainedTokenizerConfig` definition: `src/server/models/optimum.py`.
- HuggingFace tokenizer `padding`/`truncation` parameters: [transformers tokenizer call signature](https://huggingface.co/docs/transformers/main_classes/tokenizer).

---

## Local commit

Branch `fix/embeddings-default-padding-truncation` in `~/code/OpenArc` on delphi has the 3-line patch as a single commit. Same branch exists on euclid:`/opt/OpenArc` for the local Battlemage image rebuild.
