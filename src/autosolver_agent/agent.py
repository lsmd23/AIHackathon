from __future__ import annotations

from autosolver_agent.models import ProblemInstance, Solution
from autosolver_agent.solver import Solver
from autosolver_agent.strategies import GreedyStrategy, Strategy


class AutoSolverAgent:
    def __init__(self, strategies: list[Strategy] | None = None) -> None:
        self.strategies = strategies or [GreedyStrategy()]

    def run(self, problem: ProblemInstance) -> Solution:
        solver = Solver(self.strategies)
        return solver.solve(problem)
