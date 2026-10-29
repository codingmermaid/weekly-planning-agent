"""Switchable LLM clients (Gemini, Ollama) with structured output and a disk cache.

Every client takes a Pydantic schema and returns a validated instance of it,
and records latency and token usage per call for cost evaluation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

# 3.5 Flash is the fast default; 3.8 is often congested on the free tier.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_FALLBACKS = "gemini-3.8-flash,gemini-3.1-flash-lite"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


@dataclass
class CallRecord:
    provider: str
    model: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    cached: bool = False


class LLMClient(Protocol):
    provider: str
    model: str
    calls: list[CallRecord]

    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T: ...


class GeminiClient:
    provider = "gemini"
    RETRYABLE_STATUS = {429, 500, 503, 504}  # rate limited, overloaded or timed out: common on the free tier

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        fallback_models: list[str] | None = None,
        attempts_per_model: int = 2,
        timeout_s: float = 30.0,
    ):
        from google import genai

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY in .env (free key: https://aistudio.google.com/apikey)")
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        if fallback_models is None:
            raw = os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACKS)
            fallback_models = [m.strip() for m in raw.split(",") if m.strip()]
        self.fallback_models = [m for m in fallback_models if m != self.model]
        self.attempts_per_model = attempts_per_model
        self.calls: list[CallRecord] = []
        # Disable the SDK's own retries so this class alone decides when to back off or fall back.
        from google.genai import types

        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        # Extraction needs little reasoning; low thinking is faster and cheaper (Gemini 3 models).
        self.thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "low")

    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T:
        import httpx
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level.upper())
            if self.thinking_level else None,
        )
        models =[self.model, *self.fallback_models]
        last_error: Exception | None = None
        for model in models:
            for attempt in range(self.attempts_per_model):
                started = time.perf_counter()
                try:
                    response = self._client.models.generate_content(model=model, contents=prompt, config=config)
                except (errors.APIError, httpx.TimeoutException) as exc:
                    if isinstance(exc, errors.APIError) and exc.code not in self.RETRYABLE_STATUS:
                        raise
                    last_error = exc
                    log.warning("Gemini %s attempt %d failed: %s", model, attempt + 1, exc)
                    if attempt < self.attempts_per_model - 1:
                        time.sleep(2 * 2**attempt)  # 2s, 4s, ...
                    continue
                usage = response.usage_metadata
                # Thinking tokens are billed as output on models that use them.
                output_tokens = (getattr(usage, "candidates_token_count", 0) or 0) + (
                    getattr(usage, "thoughts_token_count", 0) or 0
                )
                self.calls.append(CallRecord(
                    self.provider, model, time.perf_counter() - started,
                    getattr(usage, "prompt_token_count", None), output_tokens,
                ))
                return schema.model_validate_json(response.text)
        raise RuntimeError(f"Gemini unavailable on {', '.join(models)}; try again later. Last error: {last_error}")


class OllamaClient:
    provider = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        self.base_url = (base_url or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.calls: list[CallRecord] = []

    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "format": schema.model_json_schema(),
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as resp:
            data = json.load(resp)
        self.calls.append(CallRecord(
            self.provider, self.model, time.perf_counter() - started,
            data.get("prompt_eval_count"), data.get("eval_count"),
        ))
        return schema.model_validate_json(data["message"]["content"])


class ScriptedLLM:
    """Returns canned responses keyed by schema. For tests and offline demos.

    A list of responses is returned in order, repeating the last one.
    """
    provider = "scripted"
    model = "scripted"

    def __init__(self, responses: dict[type[BaseModel], BaseModel | list[BaseModel]]):
        self.responses = {k: list(v) if isinstance(v, list) else [v] for k, v in responses.items()}
        self.calls: list[CallRecord] = []
        self.prompts: list[str] = []

    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        self.calls.append(CallRecord(self.provider, self.model, 0.0, 0, 0))
        queue = self.responses[schema]
        return queue.pop(0) if len(queue) > 1 else queue[0]


class CachedLLM:
    """Caches responses on disk so re-running evaluations doesn't spend free-tier quota."""

    def __init__(self, inner: LLMClient, cache_dir: str | Path = ".cache/llm"):
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.cache_dir = Path(cache_dir)
        self.calls: list[CallRecord] = []

    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T:
        key_material = [self.provider, self.model, system, prompt, schema.model_json_schema()]
        key = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            if "_cache_v" not in entry:  # older entries stored the bare result
                entry = {"model": self.model, "result": entry}
            self.calls.append(CallRecord(self.provider, entry["model"], 0.0, 0, 0, cached=True))
            return schema.model_validate(entry["result"])
        result = self.inner.generate(system=system, prompt=prompt, schema=schema)
        record = self.inner.calls[-1]
        self.calls.append(record)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Store the model that actually answered, which may be a fallback.
        entry = {"_cache_v": 1, "model": record.model, "result": result.model_dump(mode="json")}
        path.write_text(json.dumps(entry), encoding="utf-8")
        return result


def make_llm(provider: str | None = None, *, cache: bool = True) -> LLMClient:
    from dotenv import load_dotenv

    load_dotenv()
    provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
    if provider == "gemini":
        client: LLMClient = GeminiClient()
    elif provider == "ollama":
        client = OllamaClient()
    else:
        raise ValueError(f"unknown LLM provider {provider!r} (use gemini or ollama)")
    return CachedLLM(client) if cache else client
