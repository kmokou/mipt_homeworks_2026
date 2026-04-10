from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from part4_oop.interfaces import Cache, HasCache, Policy, Storage

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class DictStorage(Storage[K, V]):
    _data: dict[K, V] = field(default_factory=dict, init=False)

    def set(self, key: K, value: V) -> None:
        self._data[key] = value

    def get(self, key: K) -> V | None:
        return self._data.get(key)

    def exists(self, key: K) -> bool:
        return key in self._data

    def remove(self, key: K) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()


@dataclass
class FIFOPolicy(Policy[K]):
    capacity: int = 5
    _order: list[K] = field(default_factory=list, init=False)

    def register_access(self, key: K) -> None:
        if key not in self._order:
            self._order.append(key)

    def get_key_to_evict(self) -> K | None:
        if len(self._order) > self.capacity:
            return self._order[0]

        return None

    def remove_key(self, key: K) -> None:
        if key in self._order:
            self._order.remove(key)

    def clear(self) -> None:
        self._order.clear()

    @property
    def has_keys(self) -> bool:
        return bool(self._order)


@dataclass
class LRUPolicy(Policy[K]):
    capacity: int = 5
    _order: list[K] = field(default_factory=list, init=False)

    def register_access(self, key: K) -> None:
        if key in self._order:
            self._order.remove(key)

        self._order.append(key)

    def get_key_to_evict(self) -> K | None:
        if len(self._order) > self.capacity:
            return self._order[0]

        return None

    def remove_key(self, key: K) -> None:
        if key in self._order:
            self._order.remove(key)

    def clear(self) -> None:
        self._order.clear()

    @property
    def has_keys(self) -> bool:
        return bool(self._order)


@dataclass
class LFUPolicy(Policy[K]):
    capacity: int = 5
    _key_counter: dict[K, int] = field(default_factory=dict, init=False)
    _last_accessed: K | None = field(default=None, init=False)

    def register_access(self, key: K) -> None:
        self._key_counter[key] = self._key_counter.get(key, 0) + 1
        self._last_accessed = key

    def get_key_to_evict(self) -> K | None:
        if len(self._key_counter) > self.capacity:
            items = list(self._key_counter.items())
            if self._last_accessed is not None:
                filtered_items = [item for item in items if item[0] != self._last_accessed]
                if filtered_items:
                    items = filtered_items

            key_to_evict, _ = min(items, key=lambda item: item[1])
            return key_to_evict

        return None

    def remove_key(self, key: K) -> None:
        self._key_counter.pop(key, None)
        if self._last_accessed == key:
            self._last_accessed = None

    def clear(self) -> None:
        self._key_counter.clear()
        self._last_accessed = None

    @property
    def has_keys(self) -> bool:
        return bool(self._key_counter)


class MIPTCache(Cache[K, V]):
    def __init__(self, storage: Storage[K, V], policy: Policy[K]) -> None:
        self.storage = storage
        self.policy = policy

    def set(self, key: K, value: V) -> None:
        self.storage.set(key, value)
        self.policy.register_access(key)

        key_to_evict = self.policy.get_key_to_evict()
        if key_to_evict is not None:
            self.storage.remove(key_to_evict)
            self.policy.remove_key(key_to_evict)

    def get(self, key: K) -> V | None:
        if not self.storage.exists(key):
            return None
        self.policy.register_access(key)
        return self.storage.get(key)

    def exists(self, key: K) -> bool:
        return self.storage.exists(key)

    def remove(self, key: K) -> None:
        self.storage.remove(key)
        self.policy.remove_key(key)

    def clear(self) -> None:
        self.storage.clear()
        self.policy.clear()


class CachedProperty[V]:
    def __init__(self, func: Callable[[Any], V]) -> None:
        self._func = func
        self._cache_key = func.__name__

    def __get__(self, instance: HasCache[str, V] | None, owner: type[Any]) -> V:
        if instance is None:
            return self  # type: ignore[return-value]

        cache = instance.cache

        cached_value = cache.get(self._cache_key)
        if cached_value is not None:
            return cached_value

        value = self._func(instance)
        cache.set(self._cache_key, value)
        return value
