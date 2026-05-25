from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIConnectionError, APIError, OpenAI

class LLMError(Exception):
    pass


class LLMClientProtocol(Protocol):

    def complete(self, messages: Iterable['Message']) -> str:
        ...

    def stream(self, messages: Iterable['Message']) -> Iterator[str]:
        ...


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {'role': self.role, 'content': self.content}


class ChatHistory:
    def __init__(
        self,
        limit_messages: int | None = None,
        limit_chars: int | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._limit_messages = limit_messages
        self._limit_chars = limit_chars
        self._system = Message('system', system_prompt) if system_prompt else None
        self._messages: list[Message] = []

    def reset(self) -> None:
        self._messages.clear()

    def add(self, role: str, content: str) -> None:
        if role == 'system':
            self._system = Message('system', content)
            return
        self._messages.append(Message(role, content))
        self._enforce_limits()

    def pop_last(self) -> None:
        if self._messages:
            self._messages.pop()

    def messages(self) -> list[Message]:
        result: list[Message] = []
        if self._system is not None:
            result.append(self._system)
        result.extend(self._messages)
        return result

    def __len__(self) -> int:
        return len(self._messages)

    def _total_chars(self) -> int:
        total = sum(len(m.content) for m in self._messages)
        if self._system is not None:
            total += len(self._system.content)
        return total

    def _enforce_limits(self) -> None:
        if self._limit_messages is not None:
            while len(self._messages) > self._limit_messages:
                self._messages.pop(0)

        if self._limit_chars is None:
            return

        while self._total_chars() > self._limit_chars and len(self._messages) > 1:
            self._messages.pop(0)

        if self._total_chars() <= self._limit_chars or not self._messages:
            return

        last = self._messages[-1]
        system_len = len(self._system.content) if self._system is not None else 0
        available = self._limit_chars - system_len
        if available <= 0:
            self._messages.clear()
            return
        self._messages[-1] = Message(last.role, last.content[-available:])


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
        self._model = model
        self._temperature = temperature

    def _payload(self, messages: Iterable[Message]) -> list[Any]:
        return [m.to_dict() for m in messages]

    def complete(self, messages: Iterable[Message]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=self._payload(messages),
            )
        except APIConnectionError as exc:
            raise LLMError('llm connection error') from exc
        except APIError as exc:
            raise LLMError('llm api error') from exc

        choices = response.choices
        if not choices:
            raise LLMError('llm returned blank response')
        return choices[0].message.content or ''

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        try:
            stream: Any = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=self._payload(messages),
                stream=True,
            )
        except APIConnectionError as exc:
            raise LLMError('llm connection error') from exc
        except APIError as exc:
            raise LLMError('llm api error') from exc

        try:
            for chunk in stream:
                choices = chunk.choices
                if not choices:
                    continue
                piece = choices[0].delta.content
                if piece:
                    yield piece
        except APIError as exc:
            raise LLMError('llm streaming error') from exc
