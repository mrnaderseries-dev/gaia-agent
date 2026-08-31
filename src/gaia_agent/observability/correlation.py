from __future__ import annotations

from contextvars import ContextVar
from uuid import UUID, uuid4
_correlation_id:ContextVar[UUID|None]=ContextVar("context_id", default=None)
def create_correlation_id()->UUID :
    return uuid4()
def set_correlation_id(correlation_id:UUID)->None:
    _correlation_id.set(correlation_id)
def get_correlation_id()->UUID|None:
    return _correlation_id.get()
def clear_correlation_id()->None:
    _correlation_id.set(None)
        
