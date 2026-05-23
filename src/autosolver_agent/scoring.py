from __future__ import annotations

from collections import Counter

from autosolver_agent.models import ProblemInstance, Score, Solution


def score_solution(problem: ProblemInstance, solution: Solution) -> Score:
    order_ids = {order.id for order in problem.orders}
    accepted = {item.order_id for item in solution.assignments if item.order_id in order_ids}
    total_cost = sum(item.cost for item in solution.assignments if item.order_id in order_ids)
    return Score(
        accepted_orders=len(accepted),
        rejected_orders=len(order_ids - accepted),
        total_cost=total_cost,
    )


def is_capacity_feasible(problem: ProblemInstance, solution: Solution) -> bool:
    capacity_by_rider = {rider.id: rider.capacity for rider in problem.riders}
    usage = Counter(item.rider_id for item in solution.assignments)
    return all(count <= capacity_by_rider.get(rider_id, 0) for rider_id, count in usage.items())


def is_better(candidate: Score, incumbent: Score | None) -> bool:
    if incumbent is None:
        return True
    return candidate.objective > incumbent.objective
