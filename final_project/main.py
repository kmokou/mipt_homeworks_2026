import os
import sys
from collections.abc import Iterator
from pathlib import Path

from chat import ChatHistory, LLMClient, LLMClientProtocol, LLMError, Message
from config import AppConfig, ConfigError, load_config
from files import (
    ChunkSpec,
    ChunkSpecError,
    FileAttachError,
    expand_attachments,
    iter_chunks,
    parse_chunk_args,
    read_text_file,
)

PROMPT = '>>> '


def print_info(message: str) -> None:
    print(f'[info] {message}', flush=True)


def print_error(message: str) -> None:
    print(f'[error] {message}', file=sys.stderr, flush=True)


def print_assistant(text: str) -> None:
    print(f'assistant> {text}', flush=True)


def print_stream(stream: Iterator[str]) -> str:
    print('assistant> ', end='', flush=True)
    collected: list[str] = []
    for piece in stream:
        collected.append(piece)
        print(piece, end='', flush=True)
    print('', flush=True)
    return ''.join(collected)


def clear_screen() -> None:
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')
    else:
        sys.stdout.write('\n' * 50)
        sys.stdout.flush()


def read_input(prompt: str = PROMPT) -> str:
    return input(prompt)


def ask_streaming(client: LLMClientProtocol, messages: list[Message]) -> str:
    try:
        return print_stream(client.stream(messages))
    except KeyboardInterrupt:
        sys.stdout.write('\n')
        print_info('(прервано пользователем)')
        raise


def run_file_chunk(client: LLMClientProtocol, args: str) -> None:
    try:
        spec = parse_chunk_args(args)
    except ChunkSpecError as exc:
        print_error(str(exc))
        return

    print_info('Введите путь до файла (или \\q для отмены):')
    try:
        path = read_input().strip()
    except (EOFError, KeyboardInterrupt):
        print_info('Отмена.')
        return
    if path == r'\q' or not path:
        print_info('Отмена.')
        return

    try:
        text = read_text_file(path)
    except FileAttachError as exc:
        print_error(str(exc))
        return

    print_info('Принято. Что нужно сделать для каждого фрагмента (User Prompt)?')
    try:
        prompt = read_input().strip()
    except (EOFError, KeyboardInterrupt):
        print_info('Отмена.')
        return
    if prompt == r'\q' or not prompt:
        print_info('Отмена.')
        return

    print_info('Принято. Начинаю обработку:')
    process_chunks(client, text, prompt, spec)
    print_info('Обработка файла завершена.')


def process_chunks(client: LLMClientProtocol, text: str, prompt: str, spec: ChunkSpec) -> None:
    for chunk in iter_chunks(text, spec):
        messages = [Message('user', f'{prompt}\n\n{chunk}')]
        try:
            ask_streaming(client, messages)
        except KeyboardInterrupt:
            print_info('Обработка прервана.')
            return
        except LLMError as exc:
            print_error(str(exc))
            return

        if spec.auto:
            continue
        try:
            line = read_input()
        except (EOFError, KeyboardInterrupt):
            return
        if line.strip() == r'\q':
            return


def handle_message(client: LLMClientProtocol, history: ChatHistory, raw: str) -> None:
    text = raw.strip()
    if not text:
        return
    try:
        expanded = expand_attachments(text)
    except FileAttachError as exc:
        print_error(str(exc))
        return

    history.add('user', expanded)
    try:
        reply = ask_streaming(client, history.messages())
    except KeyboardInterrupt:
        history.pop_last()
        print_info('Запрос прерван. Можно дополнить или задать новый вопрос.')
        return
    except LLMError as exc:
        history.pop_last()
        print_error(str(exc))
        return

    if reply:
        history.add('assistant', reply)


def chat_loop(client: LLMClientProtocol, history: ChatHistory) -> int:
    print_info(
        'GigaVibeMiptCode готов к работе. Команды: \\q - выход, '
        '/reset - сброс истории, /file_chunk - пакетная обработка файла.',
    )
    while True:
        try:
            raw = read_input()
        except EOFError:
            print_info('Завершение работы.')
            return 0
        except KeyboardInterrupt:
            print_info('Для выхода введите \\q')
            continue

        stripped = raw.strip()
        if stripped == r'\q':
            print_info('До встречи!')
            return 0
        if stripped == '/reset':
            history.reset()
            clear_screen()
            print_info('История очищена.')
            continue
        if stripped == '/file_chunk' or stripped.startswith('/file_chunk '):
            args = stripped.removeprefix('/file_chunk').strip()
            run_file_chunk(client, args)
            continue
        handle_message(client, history, raw)


def build_app(config: AppConfig) -> tuple[LLMClient, ChatHistory]:
    client = LLMClient(
        api_key=config.api_key,
        base_url=config.api_host,
        model=config.model,
        temperature=config.temperature,
    )
    history = ChatHistory(
        limit_messages=config.limit_messages,
        limit_chars=config.limit_chars,
        system_prompt=config.system_prompt,
    )
    return client, history


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    yaml_path = Path(args[0]) if args else Path('config.yaml')

    try:
        config = load_config(yaml_path=yaml_path)
    except ConfigError as exc:
        print_error(str(exc))
        return 2

    client, history = build_app(config)
    return chat_loop(client, history)


if __name__ == '__main__':
    raise SystemExit(main())
