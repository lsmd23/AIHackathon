from __future__ import annotations

from autosolver_agent.models import ProblemInstance, Score, Solution
from autosolver_agent.scoring import is_better, score_solution
from autosolver_agent.strategies.base import Strategy


class Solver:
    def __init__(self, strategies: list[Strategy]) -> None:
        self.strategies = strategies

    def solve(self, problem: ProblemInstance) -> Solution:
        best_solution: Solution | None = None
        best_score: Score | None = None

        for strategy in self.strategies:
            candidate = strategy.solve(problem)
            candidate_score = score_solution(problem, candidate)
            candidate.metadata["score"] = {
                "accepted_orders": candidate_score.accepted_orders,
                "rejected_orders": candidate_score.rejected_orders,
                "total_cost": candidate_score.total_cost,
            }
            if is_better(candidate_score, best_score):
                best_solution = candidate
                best_score = candidate_score

        if best_solution is None:
            return Solution(strategy_name="empty")
        return best_solution
