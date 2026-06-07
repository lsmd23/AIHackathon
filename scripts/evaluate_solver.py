from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import solver


CASE_PATHS = [
    Path("examples/large_seed301.txt"),
]


def parse_rows(input_text: str):
    rows = {}
    scores = []
    tasks = set()
    for row in input_text.strip().splitlines()[1:]:
        task_id_list_str, courier_id, score, willingness = row.split("\t")[:4]
        rows[(task_id_list_str.strip(), courier_id.strip())] = (float(score), float(willingness))
        scores.append(float(score))
        tasks.update(task.strip() for task in task_id_list_str.split(","))
    return rows, scores, tasks


def ranked_expected_score(input_text: str, result: list) -> float:
    rows, _scores, _tasks = parse_rows(input_text)
    total = 0.0
    for task_id_list_str, courier_ids in result:
        fail_probability = 1.0
        for courier_id in courier_ids:
            score, willingness = rows[(task_id_list_str, courier_id)]
            total += fail_probability * willingness * score
            fail_probability *= 1.0 - willingness
        task_count = len([task for task in task_id_list_str.split(",") if task.strip()])
        total += fail_probability * 100.0 * task_count
    return total


def parallel_expected_score(input_text: str, result: list) -> float:
    rows, _scores, _tasks = parse_rows(input_text)
    total = 0.0
    for task_id_list_str, courier_ids in result:
        fail_probability = 1.0
        weighted_score = 0.0
        probability_sum = 0.0
        for courier_id in courier_ids:
            score, willingness = rows[(task_id_list_str, courier_id)]
            fail_probability *= 1.0 - willingness
            weighted_score += willingness * score
            probability_sum += willingness
        success_probability = 1.0 - fail_probability
        task_count = len([task for task in task_id_list_str.split(",") if task.strip()])
        accepted_score = weighted_score / max(probability_sum, 1e-9)
        total += success_probability * accepted_score + fail_probability * 100.0 * task_count
    return total


def fulfillment_score(input_text: str, result: list) -> float:
    rows, _scores, _tasks = parse_rows(input_text)
    total = 0.0
    for task_id_list_str, courier_ids in result:
        fail_probability = 1.0
        accepted_cost = 0.0
        probability_sum = 0.0
        for courier_id in courier_ids:
            score, willingness = rows[(task_id_list_str, courier_id)]
            fail_probability *= 1.0 - willingness
            accepted_cost += willingness * score
            probability_sum += willingness
        task_count = len([task for task in task_id_list_str.split(",") if task.strip()])
        total += fail_probability * 100.0 * task_count
        total += (accepted_cost / max(probability_sum, 1e-9)) * 0.15
    return total


def assigned_score(input_text: str, result: list) -> float:
    rows, _scores, _tasks = parse_rows(input_text)
    return sum(rows[(task_id_list_str, courier_id)][0] for task_id_list_str, courier_ids in result for courier_id in courier_ids)


def family_for(path: Path) -> str:
    if path.name == "large_seed301.txt" and path.parent.name == "examples":
        return "official_sample"
    return path.stem.split("_seed")[0]


def main() -> None:
    totals = {"ranked": 0.0, "parallel": 0.0, "fulfill": 0.0}
    family_totals: dict[str, dict[str, float]] = {}
    print("case,family,ranked,parallel,fulfill,assigned,covered,total_tasks,groups,couriers,runtime_ms")
    for case_path in CASE_PATHS:
        input_text = case_path.read_text(encoding="utf-8")
        _rows, _scores, all_tasks = parse_rows(input_text)
        start = perf_counter()
        result = solver.solve(input_text)
        runtime_ms = (perf_counter() - start) * 1000
        covered = {task.strip() for task_id_list_str, _couriers in result for task in task_id_list_str.split(",")}
        couriers = {courier_id for _task_id_list_str, courier_ids in result for courier_id in courier_ids}
        ranked = ranked_expected_score(input_text, result)
        parallel = parallel_expected_score(input_text, result)
        fulfill = fulfillment_score(input_text, result)
        assigned = assigned_score(input_text, result)
        family = family_for(case_path)
        totals["ranked"] += ranked
        totals["parallel"] += parallel
        totals["fulfill"] += fulfill
        bucket = family_totals.setdefault(family, {"count": 0.0, "ranked": 0.0, "parallel": 0.0, "fulfill": 0.0})
        bucket["count"] += 1.0
        bucket["ranked"] += ranked
        bucket["parallel"] += parallel
        bucket["fulfill"] += fulfill
        print(
            f"{case_path.stem},{family},{ranked:.3f},{parallel:.3f},{fulfill:.3f},{assigned:.3f},"
            f"{len(covered)},{len(all_tasks)},{len(result)},{len(couriers)},{runtime_ms:.1f}"
        )

    count = max(len(CASE_PATHS), 1)
    print(f"average,all,{totals['ranked'] / count:.3f},{totals['parallel'] / count:.3f},{totals['fulfill'] / count:.3f},,,,,,")
    for family, values in sorted(family_totals.items()):
        family_count = values["count"]
        print(
            f"family_average,{family},{values['ranked'] / family_count:.3f},"
            f"{values['parallel'] / family_count:.3f},{values['fulfill'] / family_count:.3f},,,,,,"
        )


if __name__ == "__main__":
    main()
