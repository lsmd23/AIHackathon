from autosolver_agent.agent import AutoSolverAgent
from autosolver_agent.models import Assignment, ProblemInstance, Rider, Order, Solution
from autosolver_agent.scoring import is_capacity_feasible, score_solution


def test_score_solution_counts_accepted_and_rejected_orders() -> None:
    problem = ProblemInstance(
        orders=(Order(id="o1"), Order(id="o2")),
        riders=(Rider(id="r1", capacity=2),),
        options=(),
    )
    solution = Solution(assignments=[Assignment(order_id="o1", rider_id="r1", cost=2.5)])

    score = score_solution(problem, solution)

    assert score.accepted_orders == 1
    assert score.rejected_orders == 1
    assert score.total_cost == 2.5


def test_capacity_feasibility_detects_overuse() -> None:
    problem = ProblemInstance(
        orders=(Order(id="o1"), Order(id="o2")),
        riders=(Rider(id="r1", capacity=1),),
        options=(),
    )
    solution = Solution(
        assignments=[
            Assignment(order_id="o1", rider_id="r1", cost=1.0),
            Assignment(order_id="o2", rider_id="r1", cost=1.0),
        ]
    )

    assert not is_capacity_feasible(problem, solution)


def test_agent_runs_greedy_baseline() -> None:
    problem = ProblemInstance.from_dict(
        {
            "orders": [{"id": "o1", "value": 2}, {"id": "o2", "value": 1}],
            "riders": [{"id": "r1", "capacity": 1}, {"id": "r2", "capacity": 1}],
            "options": [
                {"order_id": "o1", "rider_id": "r1", "cost": 5},
                {"order_id": "o1", "rider_id": "r2", "cost": 1},
                {"order_id": "o2", "rider_id": "r1", "cost": 2},
            ],
        }
    )

    solution = AutoSolverAgent().run(problem)
    score = score_solution(problem, solution)

    assert solution.strategy_name == "greedy_min_cost"
    assert score.accepted_orders == 2
    assert score.total_cost == 3
