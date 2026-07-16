from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class BeeBudget:
    total: int
    spent: int = 0
    ledger: list[dict[str, Any]] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.spent)

    def consume(self, amount: int, *, bee_type: str, operation: str, task_id: str) -> int:
        amount = max(0, int(amount))
        if amount > self.remaining:
            raise BudgetExhausted(f"required={amount}, remaining={self.remaining}")
        self.spent += amount
        self.ledger.append(
            {
                "bee_type": bee_type,
                "operation": operation,
                "task_id": task_id,
                "cost": amount,
                "remaining": self.remaining,
            }
        )
        return self.remaining

    def try_consume(self, amount: int, *, bee_type: str, operation: str, task_id: str) -> bool:
        try:
            self.consume(amount, bee_type=bee_type, operation=operation, task_id=task_id)
            return True
        except BudgetExhausted:
            return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "spent": self.spent,
            "remaining": self.remaining,
            "ledger": list(self.ledger),
        }
