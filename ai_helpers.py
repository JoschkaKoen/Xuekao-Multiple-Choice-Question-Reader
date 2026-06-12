"""Vision call (image → JSON) + JPEG encoding — trimmed from eXercise.

``ai_image_call`` is non-streaming (thinking must be 0) and returns the raw
content string. Accepts one image or several. Determinism (temperature/seed)
is injected from ALL_AI_TEMPERATURE / ALL_AI_SEED when set.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

from PIL import Image

import usage
from ai_client import build_thinking_kwargs
from api_retry import retry_api_call

log = logging.getLogger("course_reader")


def to_jpeg_bytes(image: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def page_to_jpeg_b64(image: Image.Image, quality: int = 90) -> str:
    return base64.b64encode(to_jpeg_bytes(image, quality=quality)).decode("utf-8")


def _inject_determinism(kwargs: dict, provider: str) -> None:
    t = os.environ.get("ALL_AI_TEMPERATURE", "").strip()
    if t and "temperature" not in kwargs:
        try:
            kwargs["temperature"] = float(t)
        except ValueError:
            pass
    s = os.environ.get("ALL_AI_SEED", "").strip()
    if s and provider != "gemini" and "seed" not in kwargs:  # Gemini OpenAI-compat rejects seed
        try:
            kwargs["seed"] = int(s)
        except ValueError:
            pass


def _collect_stream(stream):
    """Consume a streaming chat completion; return (answer_text, usage_obj).
    Reasoning chunks (delta.reasoning_content) are discarded; usage rides on the
    final no-choices chunk (requires stream_options.include_usage)."""
    parts: list[str] = []
    last_usage = None
    for chunk in stream:
        u = getattr(chunk, "usage", None)
        if u is not None:
            last_usage = u
        if chunk.choices:
            c = getattr(chunk.choices[0].delta, "content", None)
            if c:
                parts.append(c)
    return "".join(parts).strip(), last_usage


def ai_image_call(
    client: Any,
    image_b64s: "str | list[str]",
    prompt: str,
    *,
    model_id: str,
    provider: str,
    max_tokens: int = 512,
    thinking_tokens: int | None = 0,
    response_json: bool = True,
    request_timeout: "float | None" = 180.0,
) -> str:
    """Multimodal call → raw content string ("" on failure).

    A single user message holds the text prompt followed by the image(s); no
    system message (keeps Qwen's json_object mode happy). When *thinking_tokens*
    > 0 the call streams (Qwen requires it for thinking) and we collect the
    answer text, skipping json_object mode (the prompt + strip_json_fences
    recover the JSON).
    """
    if isinstance(image_b64s, str):
        image_b64s = [image_b64s]
    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in image_b64s:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    kwargs: dict[str, Any] = dict(
        model=model_id,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )
    use_stream, think_kw = build_thinking_kwargs(provider, thinking_tokens)
    kwargs.update(think_kw)
    if request_timeout is not None:
        kwargs["timeout"] = request_timeout
    _inject_determinism(kwargs, provider)

    if use_stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        def _do() -> str:
            text, u = _collect_stream(client.chat.completions.create(**kwargs))
            usage.record_response_usage(model_id, u)
            return text
    else:
        if response_json:
            kwargs["response_format"] = {"type": "json_object"}

        def _do() -> str:
            resp = client.chat.completions.create(**kwargs)
            usage.record_response_usage(model_id, getattr(resp, "usage", None))
            return resp.choices[0].message.content or ""

    try:
        return retry_api_call(_do, label=f"AI image ({model_id})")
    except Exception as exc:  # noqa: BLE001 — empty string = "no usable response"
        log.warning("[%s] vision call failed after retries: %s: %s", model_id, type(exc).__name__, exc)
        return ""


def ai_text_call(
    client: Any,
    prompt: str,
    *,
    model_id: str,
    provider: str,
    max_tokens: int = 2048,
    thinking_tokens: int | None = 0,
    response_json: bool = True,
    request_timeout: "float | None" = 120.0,
) -> str:
    """Non-streaming text-only call → raw content string ("" on failure)."""
    kwargs: dict[str, Any] = dict(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    use_stream, think_kw = build_thinking_kwargs(provider, thinking_tokens)
    if use_stream:
        raise RuntimeError(f"ai_text_call is non-streaming; set thinking_tokens=0 (provider={provider})")
    kwargs.update(think_kw)
    if response_json:
        kwargs["response_format"] = {"type": "json_object"}
    if request_timeout is not None:
        kwargs["timeout"] = request_timeout
    _inject_determinism(kwargs, provider)

    def _do() -> str:
        resp = client.chat.completions.create(**kwargs)
        usage.record_response_usage(model_id, getattr(resp, "usage", None))
        return resp.choices[0].message.content or ""

    try:
        return retry_api_call(_do, label=f"AI text ({model_id})")
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] text call failed after retries: %s: %s", model_id, type(exc).__name__, exc)
        return ""
