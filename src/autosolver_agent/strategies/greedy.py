from __future__ import annotations

from collections import defaultdict

from autosolver_agent.models import Assignment, ProblemInstance, Solution
from autosolver_agent.strategies.base import Strategy


class GreedyStrategy(Strategy):
    name = "greedy_min_cost"

    def solve(self, problem: ProblemInstance) -> Solution:
        remaining_capacity = {rider.id: rider.capacity for rider in problem.riders}
        options_by_order = defaultdict(list)
        for option in problem.options:
            if option.feasible:
                options_by_order[option.order_id].append(option)

        assignments: list[Assignment] = []
        sorted_orders = sorted(problem.orders, key=lambda item: (-item.value, item.id))

        for order in sorted_orders:
            candidates = sorted(options_by_order.get(order.id, []), key=lambda item: item.cost)
            for option in candidates:
                if remaining_capacity.get(option.rider_id, 0) <= 0:
                    continue
                assignments.append(
                    Assignment(
                        order_id=option.order_id,
                        rider_id=option.rider_id,
                        cost=option.cost,
                    )
                )
                remaining_capacity[option.rider_id] -= 1
                break

        return Solution(assignments=assignments, strategy_name=self.name)
