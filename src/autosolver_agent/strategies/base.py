from __future__ import annotations

from abc import ABC, abstractmethod

from autosolver_agent.models import ProblemInstance, Solution


class Strategy(ABC):
    name: str

    @abstractmethod
    def solve(self, problem: ProblemInstance) -> Solution:
        """Return a candidate solution for the given problem."""
