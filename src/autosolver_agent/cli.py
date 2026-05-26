from __future__ import annotations

import argparse
import json
from pathlib import Path

from autosolver_agent.competition import parse_competition_input, solve_competition_text


def _metrics(input_text: str, result: list[tuple[str, list[str]]]) -> dict[str, object]:
    instance = parse_competition_input(input_text)
    score_by_pair = {
        (candidate.task_id_list_str, candidate.courier_id): candidate.total_score
        for candidate in instance.candidates
    }
    willingness_by_pair = {
        (candidate.task_id_list_str, candidate.courier_id): candidate.willingness
        for candidate in instance.candidates
    }
    covered_tasks = set()
    used_couriers = set()
    total_score = 0.0
    total_willingness = 0.0
    primary_score = 0.0
    expected_score = 0.0
    reject_penalty = max(100.0, max(score_by_pair.values(), default=100.0) * 3.0)

    for task_id_list_str, courier_ids in result:
        covered_tasks.update(task_id_list_str.split(","))
        used_couriers.update(courier_ids)
        fail_probability = 1.0
        for courier_id in courier_ids:
            score = score_by_pair.get((task_id_list_str, courier_id), 0.0)
            willingness = willingness_by_pair.get((task_id_list_str, courier_id), 0.0)
            total_score += score
            total_willingness += willingness
            expected_score += fail_probability * willingness * score
            fail_probability *= 1.0 - willingness
        expected_score += fail_probability * reject_penalty
        if courier_ids:
            primary_score += score_by_pair.get((task_id_list_str, courier_ids[0]), 0.0)

    return {
        "assignments": len(result),
        "covered_tasks": len(covered_tasks),
        "used_couriers": len(used_couriers),
        "primary_score": round(primary_score, 3),
        "assigned_total_score": round(total_score, 3),
        "expected_score": round(expected_score, 3),
        "total_willingness": round(total_willingness, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoSolver on official TSV input.")
    parser.add_argument("input", type=Path, help="Path to official TSV input.")
    parser.add_argument("--output", type=Path, help="Optional path for solution JSON.")
    args = parser.parse_args()

    input_text = args.input.read_text(encoding="utf-8")
    result = solve_competition_text(input_text)
    payload = {
        "result": result,
        "metrics": _metrics(input_text, result),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
