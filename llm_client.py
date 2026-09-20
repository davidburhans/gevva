#!/usr/bin/env python3
"""llm_client.py - OpenAI-compatible client for llama-server (llama-swap) endpoints.

Moved out of generate_sdk_synthetic_data.py so the validation committee and the
data compiler can share one transport implementation without circular imports.

Usage example:
    from llm_client import LLMEndpointClient
    client = LLMEndpointClient(model="gemma-4-31b-q4")
    reply = client.query_chat("You are a judge.", "Classify: ...", response_format=schema)
"""

import json
from typing import Any, Dict, Optional


class LLMEndpointClient:
    """Communicates with any OpenAI-compatible API endpoint (llama-server, vLLM, Ollama, etc.).

    Supports:
    - User-configured model aliases with optimized presets in llama-server.
    - GBNF token restriction via structured JSON Schema (compiled to GBNF by llama-server).
    - Raw GBNF grammars (`grammar=`).
    - Automatic model unloading via POST /models/unload to manage single-model memory slots
      (llama-swap runs with --models-max 1, so only one model is resident at a time).

    After each `query_chat` call the instance exposes `last_latency_ms` and `last_usage`
    (OpenAI usage dict or None) so callers can record throughput metrics.
    """

    def __init__(self, base_url: str = "http://localhost:8080/v1", model: str = "gemma-4-31b-q4", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        # WHY: llama-swap management endpoints (/models/unload) live at the server
        # root, not under /v1.
        self.server_root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.model = model
        self.timeout = timeout
        self.last_latency_ms: Optional[float] = None
        self.last_usage: Optional[Dict[str, Any]] = None

    def unload_model(self) -> bool:
        """Sends POST /models/unload to free GPU VRAM in llama-server.

        Example: client.unload_model() -> True when the slot was released.
        """
        import urllib.request
        try:
            req = urllib.request.Request(
                f"{self.server_root}/models/unload",
                data=json.dumps({"model": self.model}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("success", False)
        except Exception:
            return False

    def query_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        response_format: Optional[Dict[str, Any]] = None,
        grammar: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Single chat completion; returns content (or reasoning_content fallback), None on failure.

        Example: client.query_chat(sys, usr, temperature=0.0, response_format=json_schema)
        """
        import time
        import urllib.request

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if grammar is not None:
            body["grammar"] = grammar
        if chat_template_kwargs is not None:
            body["chat_template_kwargs"] = chat_template_kwargs

        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            self.last_latency_ms = (time.monotonic() - started) * 1000.0
            self.last_usage = None
            print(f"  [LLMEndpointClient Error ({self.model})] {e}")
            return None

        self.last_latency_ms = (time.monotonic() - started) * 1000.0
        self.last_usage = data.get("usage")
        msg = data["choices"][0]["message"]
        content = msg.get("content", "")
        # WHY: DeepSeek-style reasoning models emit analysis in `reasoning_content`
        # while the GBNF-constrained answer lands in `content`; only fall back when
        # content is empty so constrained JSON is never replaced by prose.
        if not content and "reasoning_content" in msg:
            content = msg["reasoning_content"]
        return content.strip() if content else None
