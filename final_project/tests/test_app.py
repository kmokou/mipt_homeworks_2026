from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
import httpx

import pytest
from openai import APIConnectionError, APIError

import main as app
from chat import ChatHistory, LLMClient, LLMError, Message
from config import AppConfig, ConfigError, load_config
from files import (
    FileAttachError,
    expand_attachments,
    iter_chunks,
    parse_chunk_args,
    read_text_file,
)


def write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / 'config.yaml'
    path.write_text(content, encoding='utf-8')
    return path


def test_config_from_yaml(tmp_path: Path) -> None:
    path = write_yaml(
        tmp_path,
        'api_key: secret\napi_host: http://x/v1/\nlimit_messages: 5\ntemperature: 0.3\n',
    )
    cfg = load_config(yaml_path=path, env={})
    assert cfg.api_key == 'secret'
    assert cfg.limit_messages == 5
    assert cfg.temperature == 0.3


def test_config_env_overrides_yaml(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, 'api_key: yaml_key\napi_host: http://y/v1/\n')
    cfg = load_config(yaml_path=path, env={'API_KEY': 'env_key'})
    assert cfg.api_key == 'env_key'
    assert cfg.api_host == 'http://y/v1/'


def test_config_env_only(tmp_path: Path) -> None:
    cfg = load_config(
        yaml_path=tmp_path / 'no.yaml',
        env={'API_KEY': 'k', 'API_HOST': 'http://h/v1/', 'LIMIT_CHARS': '1000'},
    )
    assert cfg.limit_chars == 1000
    assert cfg.system_prompt is None


def test_config_no_sources(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(yaml_path=tmp_path / 'no.yaml', env={})


def test_config_bad_temperature(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, 'api_key: k\napi_host: h\ntemperature: 5\n')
    with pytest.raises(ConfigError):
        load_config(yaml_path=path, env={})


def test_history_limit_messages() -> None:
    h = ChatHistory(limit_messages=2)
    h.add('user', 'a')
    h.add('assistant', 'b')
    h.add('user', 'c')
    assert [m.content for m in h.messages()] == ['b', 'c']


def test_history_limit_chars() -> None:
    h = ChatHistory(limit_chars=10)
    h.add('user', 'aaaaaa')
    h.add('assistant', 'bbb')
    h.add('user', 'cc')
    assert [m.content for m in h.messages()] == ['bbb', 'cc']


def test_read_text_file_ok(tmp_path: Path) -> None:
    p = tmp_path / 'a.txt'
    p.write_text('hello', encoding='utf-8')
    assert read_text_file(p) == 'hello'


def test_read_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileAttachError):
        read_text_file(tmp_path / 'no.txt')


def test_expand_single_attachment(tmp_path: Path) -> None:
    p = tmp_path / 'n.txt'
    p.write_text('CONTENT', encoding='utf-8')
    assert 'CONTENT' in expand_attachments(f'see @::{p}::')


def test_chunk_parse() -> None:
    spec = parse_chunk_args('')
    assert spec.paragraphs == 1
    assert spec.chars is None
    assert spec.auto is False

    spec = parse_chunk_args('paragraph=3 -y')
    assert spec.paragraphs == 3
    assert spec.auto is True

    assert parse_chunk_args('len=150').chars == 150


def test_iter_chunks_paragraphs() -> None:
    text = 'a\n\nb\n\nc\n\nd'
    spec = parse_chunk_args('paragraph=2')
    assert list(iter_chunks(text, spec)) == ['a\n\nb', 'c\n\nd']


class FakeOpenAI:
    def __init__(self, **_: Any) -> None:
        self.chat = FakeChat(self)
        self.behavior = 'ok'

    def create(self, **kwargs: Any) -> Any:
        req = httpx.Request('POST', 'http://x/v1/chat/completions')
        if self.behavior == 'connection':
            raise APIConnectionError(request=req)
        if self.behavior == 'api':
            raise APIError('boom', request=req, body=None)
        if self.behavior == 'empty':
            return FakeResp([])
        if kwargs.get('stream'):
            chunks = [
                FakeResp([FakeChoice('hel')]),
                FakeResp([FakeChoice('lo')]),
            ]
            return iter(chunks)
        return FakeResp([FakeChoice('pong')])


class FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMsg(content)
        self.delta = FakeMsg(content)


class FakeResp:
    def __init__(self, choices: list[FakeChoice]) -> None:
        self.choices = choices


class FakeChat:
    def __init__(self, client: FakeOpenAI) -> None:
        self.completions = client


def make_client(monkeypatch: Any, behavior: str = 'ok') -> LLMClient:
    fake = FakeOpenAI()
    fake.behavior = behavior
    monkeypatch.setattr('chat.OpenAI', lambda **_: fake)
    return LLMClient(api_key='k', base_url='http://x/v1/', model='m', temperature=0.5)


def test_llm_complete_ok(monkeypatch: Any) -> None:
    client = make_client(monkeypatch, 'ok')
    assert client.complete([Message('user', 'hi')]) == 'pong'


def test_llm_complete_connection(monkeypatch: Any) -> None:
    client = make_client(monkeypatch, 'connection')
    with pytest.raises(LLMError):
        client.complete([Message('user', 'hi')])


class FakeClient:
    def __init__(self, reply: str = 'pong') -> None:
        self.reply = reply
        self.calls: list[list[Message]] = []
        self.fail_with: Exception | None = None
        self.raise_kb = False

    def complete(self, messages: Iterable[Message]) -> str:
        self.calls.append(list(messages))
        if self.fail_with:
            raise self.fail_with
        if self.raise_kb:
            raise KeyboardInterrupt
        return self.reply

    def stream(self, messages: Iterable[Message]) -> Iterator[str]:
        self.calls.append(list(messages))
        if self.fail_with:
            raise self.fail_with
        if self.raise_kb:

            def fail() -> Iterator[str]:
                raise KeyboardInterrupt
                yield ''

            return fail()

        def ok() -> Iterator[str]:
            yield from self.reply

        return ok()


def feed(monkeypatch: Any, inputs: list[str]) -> None:
    queue = list(inputs)

    def read(_prompt: str = '') -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr('main.read_input', read)


def make_history(system: str | None = 'SP') -> ChatHistory:
    return ChatHistory(limit_messages=10, system_prompt=system)


def test_loop_simple_message(
    monkeypatch: Any,
    capsys: Any,
) -> None:
    feed(monkeypatch, ['hello', r'\q'])
    client = FakeClient(reply='hi there')
    history = make_history()
    code = app.chat_loop(client, history)
    assert code == 0
    assert 'hi there' in capsys.readouterr().out


def test_loop_attachment(monkeypatch: Any, tmp_path: Path) -> None:
    note = tmp_path / 'n.txt'
    note.write_text('SECRET', encoding='utf-8')
    feed(monkeypatch, [f'@::{note}::', r'\q'])
    client = FakeClient(reply='ok')
    history = make_history()
    app.chat_loop(client, history)
    user = next(m for m in history.messages() if m.role == 'user')
    assert 'SECRET' in user.content


def test_main_missing_config(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('os.environ', {})
    assert app.main([str(tmp_path / 'absent.yaml')]) == 2


def test_appconfig_basic_fields() -> None:
    cfg = AppConfig(
        api_key='k',
        api_host='h',
        model='m',
        temperature=0.5,
        limit_messages=None,
        limit_chars=None,
        system_prompt=None,
    )
    assert cfg.model == 'm'
