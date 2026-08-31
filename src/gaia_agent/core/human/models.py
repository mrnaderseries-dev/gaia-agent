from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class HumanDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"


@dataclass(frozen=True, slots=True)
class HumanResponse:
    decision: HumanDecision
    modified_arguments: dict[str, Any] | None = None
    message: str | None = None