from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Protocol


class AiAnalysisProviderError(RuntimeError):
    pass


class AiAnalysisLlmClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class LangChainLocalLlmClient:
    def __init__(self, *, base_url: str, model: str, api_key: str) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty.")
        if not model.strip():
            raise ValueError("model must not be empty.")
        if not api_key.strip():
            raise ValueError("api_key must not be empty.")
        self._base_url = base_url
        self._model = model
        self._api_key = api_key

    def generate(self, prompt: str) -> str:
        if self._base_url.rstrip("/").endswith("/api/v1/chat"):
            return self._generate_with_lm_studio_chat(prompt)
        return self._generate_with_openai_compatible_chat(prompt)

    def _generate_with_openai_compatible_chat(self, prompt: str) -> str:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise AiAnalysisProviderError("LangChain OpenAI integration is not installed.") from exc

        try:
            llm = ChatOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                model=self._model,
                temperature=0.1,
            )
            response = llm.invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            raise AiAnalysisProviderError("Local LLM analysis is unavailable.") from exc

        content = getattr(response, "content", "")
        if isinstance(content, list):
            return "\n".join(str(item) for item in content)
        return str(content)

    def _generate_with_lm_studio_chat(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "system_prompt": (
                "You are an AI assistant supporting fraud analysts. "
                "Return only the strict JSON requested by the user prompt."
            ),
            "input": prompt,
        }
        request = Request(
            self._base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:  # noqa: S310
                raw_body = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise AiAnalysisProviderError("Local LLM analysis is unavailable.") from exc

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            return raw_body

        output = body.get("output")
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            message_parts: list[str] = []
            for item in output:
                if isinstance(item, dict) and item.get("type") == "message" and isinstance(item.get("content"), str):
                    message_parts.append(item["content"])
            if message_parts:
                return "\n".join(message_parts)

        for key in ("answer", "response", "content", "text"):
            value = body.get(key)
            if isinstance(value, str):
                return value

        message = body.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                choice_message = first_choice.get("message")
                if isinstance(choice_message, dict) and isinstance(choice_message.get("content"), str):
                    return choice_message["content"]
                if isinstance(first_choice.get("text"), str):
                    return first_choice["text"]

        return raw_body
