from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from planner import llm as llm_module
from planner.intake import ParsedRequest
from planner.llm import GeminiClient

ANSWER = ParsedRequest(tasks=[], constraints=[], questions=[])


class FakeModels:
    """Plays back one outcome per call: an exception to raise, or None for success."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.models_called = []

    def generate_content(self, *, model, contents, config):
        self.models_called.append(model)
        outcome = self.outcomes.pop(0)
        if outcome is not None:
            raise outcome
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=2)
        return SimpleNamespace(text=ANSWER.model_dump_json(), usage_metadata=usage)


def client_with(outcomes, monkeypatch) -> tuple[GeminiClient, FakeModels]:
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    client = GeminiClient(model="main", api_key="test", fallback_models=["backup"])
    fake = FakeModels(outcomes)
    client._client = SimpleNamespace(models=fake)
    return client, fake


def busy():
    return errors.ServerError(503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}})


def test_falls_back_after_retries_and_records_the_answering_model(monkeypatch):
    client, fake = client_with([busy(), httpx.ReadTimeout("slow"), None], monkeypatch)
    assert client.generate(system="s", prompt="p", schema=ParsedRequest) == ANSWER
    assert fake.models_called == ["main", "main", "backup"]
    assert client.calls[-1].model == "backup"
    assert client.calls[-1].output_tokens == 7  # thinking tokens count as output


def test_non_retryable_errors_are_raised_immediately(monkeypatch):
    bad_request = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
    client, fake = client_with([bad_request], monkeypatch)
    with pytest.raises(errors.ClientError):
        client.generate(system="s", prompt="p", schema=ParsedRequest)
    assert fake.models_called == ["main"]


def test_gives_up_with_a_clear_error_when_every_model_fails(monkeypatch):
    client, _ = client_with([busy()] * 4, monkeypatch)
    with pytest.raises(RuntimeError, match="Gemini unavailable on main, backup"):
        client.generate(system="s", prompt="p", schema=ParsedRequest)
