from __future__ import annotations

import argparse
import json
from pathlib import Path

from autosolver_agent.agent import AutoSolverAgent
from autosolver_agent.models import ProblemInstance, Solution


def solution_to_dict(solution: Solution) -> dict[str, object]:
    return {
        "strategy": solution.strategy_name,
        "assignments": [
            {"order_id": item.order_id, "rider_id": item.rider_id, "cost": item.cost}
            for item in solution.assignments
        ],
        "metadata": solution.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoSolver on a problem JSON file.")
    parser.add_argument("input", type=Path, help="Path to problem JSON.")
    parser.add_argument("--output", type=Path, help="Optional path for solution JSON.")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    problem = ProblemInstance.from_dict(payload)
    solution = AutoSolverAgent().run(problem)
    result = solution_to_dict(solution)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
