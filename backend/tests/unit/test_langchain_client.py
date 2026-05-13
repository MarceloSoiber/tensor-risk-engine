from __future__ import annotations

import json
from io import BytesIO

from app.llm.langchain_client import LangChainLocalLlmClient


class FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return BytesIO(json.dumps(self._payload).encode("utf-8")).read()


def test_lm_studio_chat_endpoint_returns_output(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHttpResponse({"output": '{"answer":"ok","insights":[],"recommended_actions":[]}'})

    monkeypatch.setattr("app.llm.langchain_client.urlopen", fake_urlopen)
    client = LangChainLocalLlmClient(
        base_url="http://localhost:1234/api/v1/chat",
        model="openai/gpt-oss-20b",
        api_key="lm-studio",
    )

    response = client.generate("Analyze this")

    assert response == '{"answer":"ok","insights":[],"recommended_actions":[]}'
    assert captured["url"] == "http://localhost:1234/api/v1/chat"
    assert captured["timeout"] == 120
    assert captured["payload"] == {
        "model": "openai/gpt-oss-20b",
        "system_prompt": (
            "You are an AI assistant supporting fraud analysts. "
            "Return only the strict JSON requested by the user prompt."
        ),
        "input": "Analyze this",
    }


def test_lm_studio_chat_endpoint_extracts_message_output(monkeypatch) -> None:
    def fake_urlopen(request, timeout):  # noqa: ANN001, ANN202, ARG001
        return FakeHttpResponse(
            {
                "output": [
                    {"type": "reasoning", "content": "thinking"},
                    {
                        "type": "message",
                        "content": '<|channel|>final <|constrain|>JSON<|message|>{"answer":"blue","insights":[],"recommended_actions":[]}',
                    },
                ]
            }
        )

    monkeypatch.setattr("app.llm.langchain_client.urlopen", fake_urlopen)
    client = LangChainLocalLlmClient(
        base_url="http://localhost:1234/api/v1/chat",
        model="openai/gpt-oss-20b",
        api_key="lm-studio",
    )

    response = client.generate("Analyze this")

    assert response == '<|channel|>final <|constrain|>JSON<|message|>{"answer":"blue","insights":[],"recommended_actions":[]}'
