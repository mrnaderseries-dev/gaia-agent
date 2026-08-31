from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ContextBudget import ContextBudget


@dataclass(slots=True)
class ContextValidationResult:
    valid: bool
    errors: list[str]


class ContextValidator:

    def __init__(
        self,
        budget: ContextBudget,
    ) -> None:
        self.budget = budget

    def validate(
        self,
        context: list[Any],
    ) -> ContextValidationResult:

        errors: list[str] = []

        if not context:
            errors.append("Context is empty.")

        if any(item is None for item in context):
            errors.append("Context contains None items.")

        if context and not self.budget.fits(context):
            token_count = self.budget.count_tokens(context)

            errors.append(
                f"Context exceeds token budget: "
                f"{token_count} > {self.budget.max_tokens}."
            )

        return ContextValidationResult(
            valid=not errors,
            errors=errors,
        )