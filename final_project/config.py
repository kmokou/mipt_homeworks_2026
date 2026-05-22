import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    pass


@dataclass
class AppConfig:
    api_key: str
    api_host: str
    model: str
    temperature: float
    limit_messages: int | None
    limit_chars: int | None
    system_prompt: str | None


DEFAULT_MODEL = 'gemma3:270m'
DEFAULT_TEMPERATURE = 0.7

ENV_KEYS = ('API_KEY', 'API_HOST', 'LIMIT_MESSAGES', 'LIMIT_CHARS', 'TEMPERATURE', 'MODEL')


def _to_int(value: Any, name: str) -> int | None:
    if value is None or value == '':
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ConfigError('bad integer') from None
    if result <= 0:
        raise ConfigError('limit must be positive')
    return result


def _to_float(value: Any, name: str) -> float | None:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError('bad float') from None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except (OSError, yaml.YAMLError):
        raise ConfigError('bad config file') from None
    if not isinstance(data, dict):
        raise ConfigError('config must be dict')
    return data


def load_config(yaml_path: Path | None = None, env: dict[str, str] | None = None) -> AppConfig:
    env_map = dict(os.environ) if env is None else env
    yaml_file = Path('config.yaml') if yaml_path is None else yaml_path
    yaml_data = _load_yaml(yaml_file)

    if not yaml_data and not any(k in env_map for k in ENV_KEYS):
        raise ConfigError('no config values')

    def pick(env_key: str, yaml_key: str) -> Any:
        if env_map.get(env_key):
            return env_map[env_key]
        return yaml_data.get(yaml_key)

    api_key = pick('API_KEY', 'api_key')
    api_host = pick('API_HOST', 'api_host')
    model = pick('MODEL', 'model') or DEFAULT_MODEL
    temperature = _to_float(pick('TEMPERATURE', 'temperature'), 'temperature')
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    if not 0.0 <= temperature <= 1.0:
        raise ConfigError('bad temperature')

    limit_messages = _to_int(pick('LIMIT_MESSAGES', 'limit_messages'), 'limit_messages')
    limit_chars = _to_int(pick('LIMIT_CHARS', 'limit_chars'), 'limit_chars')

    system_prompt = yaml_data.get('system_prompt')
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ConfigError('bad system_prompt')

    if not api_key:
        raise ConfigError('missing api_key')
    if not api_host:
        raise ConfigError('missing api_host')

    return AppConfig(
        api_key=str(api_key),
        api_host=str(api_host),
        model=str(model),
        temperature=float(temperature),
        limit_messages=limit_messages,
        limit_chars=limit_chars,
        system_prompt=system_prompt,
    )
