from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Order:
    id: str
    value: float = 1.0
    pickup_time: float | None = None
    deadline: float | None = None


@dataclass(frozen=True)
class Rider:
    id: str
    capacity: int = 1
    willingness: float = 1.0


@dataclass(frozen=True)
class AssignmentOption:
    order_id: str
    rider_id: str
    cost: float
    feasible: bool = True


@dataclass(frozen=True)
class Assignment:
    order_id: str
    rider_id: str
    cost: float


@dataclass(frozen=True)
class ProblemInstance:
    orders: tuple[Order, ...]
    riders: tuple[Rider, ...]
    options: tuple[AssignmentOption, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProblemInstance":
        orders = tuple(Order(**item) for item in payload.get("orders", []))
        riders = tuple(Rider(**item) for item in payload.get("riders", []))
        options = tuple(AssignmentOption(**item) for item in payload.get("options", []))
        return cls(orders=orders, riders=riders, options=options)


@dataclass
class Score:
    accepted_orders: int
    rejected_orders: int
    total_cost: float

    @property
    def objective(self) -> tuple[int, float]:
        return (self.accepted_orders, -self.total_cost)


@dataclass
class Solution:
    assignments: list[Assignment] = field(default_factory=list)
    strategy_name: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
