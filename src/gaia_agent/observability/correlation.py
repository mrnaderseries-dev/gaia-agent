from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator
from uuid import UUID, uuid4

_correlation_id: ContextVar[UUID | None] = ContextVar(
    "gaia_correlation_id",
    default=None,
)


def create_correlation_id() -> UUID:
    return uuid4()


def set_correlation_id(
    correlation_id: UUID,
) -> Token[UUID | None]:
    if not isinstance(correlation_id, UUID):
        raise TypeError(
            "correlation_id must be a UUID"
        )

    return _correlation_id.set(correlation_id)


def get_correlation_id() -> UUID | None:
    return _correlation_id.get()


def clear_correlation_id() -> None:
    _correlation_id.set(None)


def reset_correlation_id(
    token: Token[UUID | None],
) -> None:
    _correlation_id.reset(token)


@contextmanager
def correlation_context(
    correlation_id: UUID | None = None,
) -> Iterator[UUID]:
    value = correlation_id or create_correlation_id()

    token = set_correlation_id(value)

    try:
        yield value
    finally:
        reset_correlation_id(token)