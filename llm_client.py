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
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Sequence, Union


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
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme '{parsed.scheme}': only http and https are allowed")
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
        try:
            req = urllib.request.Request(
                f"{self.server_root}/models/unload",
                data=json.dumps({"model": self.model}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read(10 * 1024 * 1024).decode("utf-8"))
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
        images: Optional[Sequence[Any]] = None,
    ) -> Optional[str]:
        """Single chat completion; returns content (or reasoning_content fallback), None on failure.

        Supports multimodal vision inputs: `images` can be file paths or PIL Image objects.
        """
        import base64
        import io
        import time
        import urllib.request
        from pathlib import Path
        from PIL import Image

        if images:
            user_content = [{"type": "text", "text": user_prompt}]
            for img in images:
                if isinstance(img, (str, Path)):
                    img_path = Path(img)
                    if img_path.exists():
                        b64_str = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_str}"}})
                elif isinstance(img, Image.Image):
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}})
        else:
            user_content = user_prompt

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
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
                raw = resp.read(10 * 1024 * 1024).decode("utf-8")
        except Exception as e:
            self.last_latency_ms = (time.monotonic() - started) * 1000.0
            self.last_usage = None
            print(f"  [LLMEndpointClient Error ({self.model})] {e}")
            return None

        self.last_latency_ms = (time.monotonic() - started) * 1000.0
        try:
            # WHY: a 200 response with a non-OpenAI body (proxy error page, HTML) must
            # degrade to None, never crash the committee (audit HIGH: parse outside try).
            data = json.loads(raw)
            self.last_usage = data.get("usage")
            msg = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print(f"  [LLMEndpointClient Error ({self.model})] malformed response body: {e}")
            return None
        content = msg.get("content", "")
        # WHY: DeepSeek-style reasoning models emit analysis in `reasoning_content`
        # while the GBNF-constrained answer lands in `content`; only fall back when
        # content is empty so constrained JSON is never replaced by prose.
        if not content and "reasoning_content" in msg:
            content = msg["reasoning_content"]
        return content.strip() if content else None
