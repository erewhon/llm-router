"""Tool-calling proxy — sits between clients and vLLM.

Refactored from agent-service.py. Adds tool execution (web search, calculator, etc.)
on top of vLLM's inference. Clients send standard OpenAI chat completion requests;
this service injects tool definitions, executes proxy-owned tools, and returns
the final response in OpenAI-compatible format.

Features:
- Streaming SSE support (tool loop runs non-streaming, final answer streams)
- Thinking token passthrough via reasoning_content
- Multiple search backends (DuckDuckGo + Tavily)
- Calculator (safe math via simpleeval)
- URL fetching with text extraction
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import click
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI

from llm_router.tool_proxy.extraction import (
    extract_thinking,
    extract_tool_calls,
    strip_tool_call_tags,
)
from llm_router.tool_proxy.streaming import (
    build_response,
    build_sse_chunk,
    build_tool_calls_response,
)
from llm_router.tool_proxy.thinking import ThinkingStreamParser
from llm_router.tool_proxy.tools.registry import ToolRegistry

logger = logging.getLogger("tool-proxy")

app = FastAPI(title="LLM Tool Proxy")

# Configured at startup
_vllm_client: AsyncOpenAI | None = None
_vllm_url: str = "http://localhost:5391"
_max_output_tokens: int = 32768
_max_tool_rounds: int = 5
_registry: ToolRegistry = ToolRegistry()
_backend_clients: dict[str, AsyncOpenAI] = {}  # model_name -> client
_model_registry = None  # ModelRegistry for backend resolution


def _get_client(model: str = "auto") -> AsyncOpenAI:
    """Get the OpenAI client for a model, resolving backend URL from registry."""
    if _model_registry is not None and model != "auto":
        # Check if this model, alias, or hf_repo maps to a different backend
        # LiteLLM may send "openai/Repo/Name" — strip the prefix
        model_clean = model.removeprefix("openai/")
        for model_id, mdef in _model_registry.models.items():
            hf_base = mdef.hf_repo.split("#")[0]  # Strip #nothink etc.
            if model_id == model_clean or model_clean in mdef.aliases or hf_base == model_clean or mdef.hf_repo == model_clean:
                # Resolve the actual backend URL (not tool proxy port)
                if mdef.multi_node:
                    head = mdef.multi_node.head_node or mdef.multi_node.nodes[0]
                    host = _model_registry.nodes[head].host
                else:
                    host = _model_registry.nodes[mdef.node].host
                port = mdef.api_port or 5391
                backend_url = f"http://{host}:{port}"
                if backend_url not in _backend_clients:
                    _backend_clients[backend_url] = AsyncOpenAI(
                        base_url=f"{backend_url}/v1", api_key="dummy"
                    )
                    logger.info(f"Created backend client for {model} -> {backend_url}")
                return _backend_clients[backend_url]
    assert _vllm_client is not None, "vLLM client not configured"
    return _vllm_client


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    messages = list(body.get("messages", []))
    raw_model = body.get("model", "auto")
    # Strip #nothink suffix — used for routing, not sent to backend
    model = raw_model.split("#")[0] if "#" in raw_model else raw_model
    raw_max_tokens = body.get("max_tokens", 4096)
    max_tokens = min(raw_max_tokens, _max_output_tokens)
    if raw_max_tokens != max_tokens:
        logger.info(f"Clamped max_tokens: {raw_max_tokens} -> {max_tokens}")
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)

    extra_kwargs: dict[str, Any] = {}
    for key in ("top_p", "frequency_penalty", "presence_penalty", "stop", "seed"):
        if key in body:
            extra_kwargs[key] = body[key]

    # Pass through extra_body for backend-specific params (e.g. chat_template_kwargs)
    extra_body: dict[str, Any] = {}
    if "chat_template_kwargs" in body:
        extra_body["chat_template_kwargs"] = body["chat_template_kwargs"]
    # Check for nothink tag — disables thinking and skips proxy tool injection
    is_nothink = False
    if _model_registry is not None:
        raw_clean = raw_model.removeprefix("openai/")
        for mid, mdef in _model_registry.models.items():
            hf_base = mdef.hf_repo.split("#")[0]
            if mid == raw_clean or raw_clean in mdef.aliases or hf_base == raw_clean or mdef.hf_repo == raw_clean:
                if "nothink" in mdef.tags:
                    is_nothink = True
                    extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
                    logger.info(f"Disabled thinking for {raw_model} (nothink tag)")
                break
    if extra_body:
        extra_kwargs["extra_body"] = extra_body

    # Merge client tools with proxy tools (proxy names take precedence)
    # Skip proxy tool injection for nothink models — tools cause tool-loop overhead
    client_tools = body.get("tools") or []
    tool_choice = body.get("tool_choice", "auto")
    if is_nothink:
        all_tools = list(client_tools)
        logger.info("Skipping proxy tool injection (nothink model)")
    else:
        proxy_names = _registry.names
        all_tools = list(_registry.definitions)
        for ct in client_tools:
            ct_name = ct.get("function", {}).get("name")
            if ct_name and ct_name not in proxy_names:
                all_tools.append(ct)

    if stream:
        return StreamingResponse(
            _stream_chat_completion(
                messages,
                model,
                max_tokens,
                temperature,
                extra_kwargs,
                all_tools,
                tool_choice,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return await _non_streaming_chat_completion(
        messages, model, max_tokens, temperature, extra_kwargs, all_tools, tool_choice
    )


async def _non_streaming_chat_completion(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    extra_kwargs: dict[str, Any],
    all_tools: list[dict[str, Any]],
    tool_choice: str | dict,
) -> JSONResponse:
    """Non-streaming tool loop — executes proxy tools, returns client tool calls."""
    client = _get_client(model)

    for round_num in range(_max_tool_rounds):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=all_tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                **extra_kwargs,
            )
        except Exception as e:
            logger.error(f"vLLM request failed: {e}")
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"vLLM backend error: {e}"}},
            )

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        # vLLM with --reasoning-parser puts thinking in msg.reasoning
        backend_reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)

        tool_calls = extract_tool_calls(msg, content)
        content = strip_tool_call_tags(content)

        if not tool_calls:
            reasoning, clean_content = extract_thinking(content)
            # Prefer backend-provided reasoning over tag-extracted
            reasoning = backend_reasoning or reasoning
            return build_response(model, clean_content, response.usage, reasoning or None)

        # Split into proxy-owned vs client-owned
        proxy_calls = [tc for tc in tool_calls if _registry.has_tool(tc["function"]["name"])]
        client_calls = [tc for tc in tool_calls if not _registry.has_tool(tc["function"]["name"])]

        if client_calls:
            reasoning, clean_content = extract_thinking(content)
            logger.info(
                f"Returning {len(client_calls)} client tool call(s) to frontend"
                + (f" (also had {len(proxy_calls)} proxy call(s))" if proxy_calls else "")
            )
            return build_response(
                model,
                clean_content,
                response.usage,
                reasoning or None,
                tool_calls=tool_calls,
                finish_reason="tool_calls",
            )

        # All proxy-owned — execute them
        logger.info(f"Round {round_num + 1}: {len(proxy_calls)} tool call(s)")
        _, clean_assistant = extract_thinking(content)
        messages.append(
            {
                "role": "assistant",
                "content": clean_assistant,
                "tool_calls": tool_calls,
            }
        )

        for tc in proxy_calls:
            result = _registry.execute(tc["function"]["name"], tc["function"]["arguments"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )

    logger.warning(f"Max tool rounds ({_max_tool_rounds}) reached")
    reasoning, final = extract_thinking(msg.content or "(max tool rounds reached)")
    return build_response(model, final, None, reasoning or None)


async def _stream_chat_completion(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    extra_kwargs: dict[str, Any],
    all_tools: list[dict[str, Any]],
    tool_choice: str | dict,
):
    """Streaming generator: tool loop non-streaming, then stream the final answer."""
    client = _get_client(model)
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # --- Tool loop (non-streaming) ---
    for round_num in range(_max_tool_rounds):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=all_tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                **extra_kwargs,
            )
        except Exception as e:
            logger.error(f"vLLM request failed during tool loop: {e}")
            yield build_sse_chunk(chunk_id, model, role="assistant")
            yield build_sse_chunk(
                chunk_id,
                model,
                content=f"Error: vLLM backend error: {e}",
                finish_reason="stop",
            )
            yield "data: [DONE]\n\n"
            return

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""

        tool_calls = extract_tool_calls(msg, content)
        content = strip_tool_call_tags(content)

        if not tool_calls:
            break

        proxy_calls = [tc for tc in tool_calls if _registry.has_tool(tc["function"]["name"])]
        client_calls = [tc for tc in tool_calls if not _registry.has_tool(tc["function"]["name"])]

        if client_calls:
            logger.info(
                f"Returning {len(client_calls)} client tool call(s) to frontend (streaming)"
            )
            reasoning, clean_content = extract_thinking(content)
            resp = build_tool_calls_response(
                chunk_id,
                model,
                clean_content,
                tool_calls,
                reasoning_content=reasoning or None,
            )
            yield f"data: {json.dumps(resp)}\n\n"
            yield "data: [DONE]\n\n"
            return

        logger.info(f"Stream tool round {round_num + 1}: {len(proxy_calls)} tool call(s)")
        _, clean_assistant = extract_thinking(content)
        messages.append(
            {
                "role": "assistant",
                "content": clean_assistant,
                "tool_calls": tool_calls,
            }
        )

        for tc in proxy_calls:
            result = _registry.execute(tc["function"]["name"], tc["function"]["arguments"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )
    else:
        # Max rounds exhausted
        logger.warning(f"Max tool rounds ({_max_tool_rounds}) reached in streaming mode")
        yield build_sse_chunk(chunk_id, model, role="assistant")
        reasoning, final = extract_thinking(msg.content or "(max tool rounds reached)")
        if reasoning:
            yield build_sse_chunk(chunk_id, model, reasoning_content=reasoning)
        yield build_sse_chunk(chunk_id, model, content=final, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # --- Stream the final generation (no tools) ---
    try:
        stream_response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **extra_kwargs,
        )
    except Exception as e:
        logger.error(f"vLLM streaming request failed: {e}")
        yield build_sse_chunk(chunk_id, model, role="assistant")
        yield build_sse_chunk(
            chunk_id,
            model,
            content=f"Error: vLLM backend error: {e}",
            finish_reason="stop",
        )
        yield "data: [DONE]\n\n"
        return

    yield build_sse_chunk(chunk_id, model, role="assistant")

    parser = ThinkingStreamParser()

    async for chunk in stream_response:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason

        # vLLM with --reasoning-parser sends thinking as delta.reasoning
        delta_reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if delta_reasoning:
            yield build_sse_chunk(chunk_id, model, reasoning_content=delta_reasoning)

        delta_content = delta.content or ""

        if delta_content:
            reasoning_text, content_text = parser.feed(delta_content)
            if reasoning_text:
                yield build_sse_chunk(chunk_id, model, reasoning_content=reasoning_text)
            if content_text:
                yield build_sse_chunk(chunk_id, model, content=content_text)

        if finish_reason:
            # Flush any remaining buffer
            if parser._buffer:
                remaining_r, remaining_c = parser.feed("")
                if not remaining_r and not remaining_c:
                    if parser._state == "tool_call":
                        pass  # Discard incomplete tool_call
                    elif parser.in_think:
                        remaining_r = parser._buffer
                    else:
                        remaining_c = parser._buffer
                    parser._buffer = ""
                if remaining_r:
                    yield build_sse_chunk(chunk_id, model, reasoning_content=remaining_r)
                if remaining_c:
                    yield build_sse_chunk(chunk_id, model, content=remaining_c)

            yield build_sse_chunk(chunk_id, model, finish_reason=finish_reason)
            break

    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    """Pass through to vLLM."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{_vllm_url}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"vLLM backend error: {e}"}},
        )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "max_output_tokens": _max_output_tokens,
        "tools": list(_registry.names),
        "streaming": True,
    }


def _fetch_max_model_len(vllm_url: str) -> int | None:
    """Query vLLM for the served model's max_model_len."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{vllm_url}/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = data.get("data", [])
        if not models:
            return None
        return models[0].get("max_model_len")
    except Exception:
        return None


def _setup_ssl_certs() -> None:
    """Set SSL cert path if not already configured."""
    if os.environ.get("SSL_CERT_FILE"):
        return
    for cert_path in (
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
    ):
        if os.path.exists(cert_path):
            os.environ["SSL_CERT_FILE"] = cert_path
            os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
            break


def create_app(
    vllm_url: str = "http://localhost:5391",
    tavily_key: str | None = None,
    proxy: str | None = None,
) -> FastAPI:
    """Configure and return the FastAPI app."""
    global _vllm_client, _vllm_url, _max_output_tokens, _registry, _model_registry

    _setup_ssl_certs()
    _vllm_url = vllm_url
    _vllm_client = AsyncOpenAI(base_url=f"{vllm_url}/v1", api_key="dummy")

    # Load model registry for multi-backend routing
    try:
        from llm_router.config import load_registry
        _model_registry = load_registry()
        logger.info(f"Loaded model registry ({len(_model_registry.models)} models)")
    except Exception as e:
        logger.warning(f"Could not load model registry: {e} — using default backend only")

    # Try to get actual max from vLLM
    max_len = _fetch_max_model_len(vllm_url)
    if max_len:
        _max_output_tokens = max_len
        logger.info(f"Max output tokens from vLLM: {_max_output_tokens}")

    if proxy:
        logger.info(f"Outbound proxy: {proxy}")

    # Register tools
    _registry = ToolRegistry()
    from llm_router.tool_proxy.tools import calculator, fetch_url, tavily, web_search

    web_search.register(_registry, proxy=proxy)
    fetch_url.register(_registry, proxy=proxy)
    calculator.register(_registry)
    tavily.register(_registry, api_key=tavily_key or os.environ.get("TAVILY_API_KEY"), proxy=proxy)

    logger.info(f"Tools: {', '.join(_registry.names)}")
    return app


@click.command()
@click.option("--port", type=int, default=5392, help="Bind port")
@click.option("--vllm-url", default="http://localhost:5391", help="vLLM backend URL")
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--tavily-key", default=None, help="Tavily API key")
@click.option("--proxy", default=None, help="SOCKS5/HTTP proxy for outbound tool requests (e.g. socks5://host:1080)")
def cli(port: int, vllm_url: str, host: str, tavily_key: str | None, proxy: str | None) -> None:
    """Start the tool-calling proxy."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    create_app(vllm_url=vllm_url, tavily_key=tavily_key, proxy=proxy)
    logger.info(f"Starting tool proxy on {host}:{port}")
    logger.info(f"vLLM backend: {vllm_url}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    cli()
