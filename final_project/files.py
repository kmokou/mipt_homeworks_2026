import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_BYTES = 5 * 1024 * 1024

_ATTACH_PATTERN = re.compile(r'@::([^:]+)::')


class FileAttachError(Exception):
    pass


class ChunkSpecError(Exception):
    pass


@dataclass
class ChunkSpec:
    paragraphs: int = 1
    chars: int | None = None
    auto: bool = False


def read_text_file(path: str | Path) -> str:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileAttachError('file not found')
    if not file_path.is_file():
        raise FileAttachError('not a file')

    try:
        size = file_path.stat().st_size
    except OSError:
        raise FileAttachError('stat failed') from None
    if size > MAX_FILE_BYTES:
        raise FileAttachError('file too large')

    try:
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        raise FileAttachError('not a text file') from None
    except OSError:
        raise FileAttachError('read failed') from None


def expand_attachments(text: str) -> str:
    parts: list[str] = []
    last_end = 0
    for match in _ATTACH_PATTERN.finditer(text):
        parts.append(text[last_end : match.start()].rstrip())
        parts.append('\n' + read_text_file(match.group(1).strip()))
        last_end = match.end()
    parts.append(text[last_end:])
    return ''.join(parts)


def parse_chunk_args(raw: str) -> ChunkSpec:
    paragraphs = 1
    chars: int | None = None
    auto = False

    for token in raw.split():
        if token == '-y':
            auto = True
            continue
        if '=' not in token:
            raise ChunkSpecError('bad argument')
        name, _, value = token.partition('=')
        if name == 'paragraph':
            try:
                paragraphs = int(value)
            except ValueError:
                raise ChunkSpecError('bad paragraph') from None
            if paragraphs <= 0:
                raise ChunkSpecError('bad paragraph')
        elif name == 'len':
            try:
                chars = int(value)
            except ValueError:
                raise ChunkSpecError('bad len') from None
            if chars <= 0:
                raise ChunkSpecError('bad len')
        else:
            raise ChunkSpecError('bad argument')

    return ChunkSpec(paragraphs=paragraphs, chars=chars, auto=auto)


def iter_chunks(text: str, spec: ChunkSpec) -> Iterator[str]:
    if spec.chars is not None:
        cleaned = text.strip()
        for i in range(0, len(cleaned), spec.chars):
            chunk = cleaned[i : i + spec.chars]
            if chunk:
                yield chunk
        return

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    for i in range(0, len(paragraphs), spec.paragraphs):
        chunk = '\n\n'.join(paragraphs[i : i + spec.paragraphs])
        if chunk:
            yield chunk
