import json
from datetime import UTC, datetime
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar
from urllib.request import urlopen

INVALID_CRITICAL_COUNT = "Breaker count must be positive integer!"
INVALID_RECOVERY_TIME = "Breaker recovery time must be positive integer!"
VALIDATIONS_FAILED = "Invalid decorator args."
TOO_MUCH = "Too much requests, just wait."


P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


class CallableWithMeta(Protocol[P, R_co]):
    __name__: str
    __module__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...


class BreakerError(Exception):
    def __init__(self, func_name: str, block_time: datetime):
        super().__init__(TOO_MUCH)
        self.func_name = func_name
        self.block_time = block_time


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _get_func_name[**FuncParams, FuncResult](func: CallableWithMeta[FuncParams, FuncResult]) -> str:
    return f"{func.__module__}.{func.__name__}"


class CircuitBreaker:
    def __init__(
        self,
        critical_count: int = 5,
        time_to_recover: int = 30,
        triggers_on: type[Exception] = Exception,
    ):
        errors: list[ValueError] = []
        if not _is_positive_integer(critical_count):
            errors.append(ValueError(INVALID_CRITICAL_COUNT))
        if not _is_positive_integer(time_to_recover):
            errors.append(ValueError(INVALID_RECOVERY_TIME))
        if errors:
            raise ExceptionGroup(VALIDATIONS_FAILED, errors)

        self.critical_count = critical_count
        self.time_to_recover = time_to_recover
        self.triggers_on = triggers_on
        self._errors_in_row = 0
        self._block_time: datetime | None = None

    def __call__(self, func: CallableWithMeta[P, R_co]) -> CallableWithMeta[P, R_co]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R_co:
            return self._call_with_breaker(func, *args, **kwargs)

        return wrapper

    def _reset_state(self) -> None:
        self._errors_in_row = 0
        self._block_time = None

    def _raise_if_blocked(self, func_name: str) -> None:
        if self._block_time is None:
            return

        elapsed = (datetime.now(UTC) - self._block_time).total_seconds()
        if elapsed < self.time_to_recover:
            raise BreakerError(func_name, self._block_time)

        self._reset_state()

    def _handle_triggered_exception(self, func_name: str, exc: Exception) -> None:
        self._errors_in_row += 1
        if self._errors_in_row >= self.critical_count:
            self._block_time = datetime.now(UTC)
            raise BreakerError(func_name, self._block_time) from exc
        raise exc

    def _call_with_breaker(
        self,
        func: CallableWithMeta[P, R_co],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_co:
        func_name = _get_func_name(func)
        self._raise_if_blocked(func_name)

        try:
            result = func(*args, **kwargs)
        except self.triggers_on as exc:
            self._handle_triggered_exception(func_name, exc)

        self._errors_in_row = 0
        return result


circuit_breaker = CircuitBreaker(5, 30, Exception)


# @circuit_breaker
def get_comments(post_id: int) -> Any:
    """
    Получает комментарии к посту

    Args:
        post_id (int): Идентификатор поста

    Returns:
        list[dict[int | str]]: Список комментариев
    """
    response = urlopen(f"https://jsonplaceholder.typicode.com/comments?postId={post_id}")
    return json.loads(response.read())


if __name__ == "__main__":
    comments = get_comments(1)
