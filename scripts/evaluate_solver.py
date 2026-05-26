from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import solver


CASE_PATHS = [
    Path("examples/large_seed301.txt"),
    *sorted(Path("examples/blackbox_like").glob("*.txt")),
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


def expected_score(input_text: str, result: list) -> float:
    rows, scores, _tasks = parse_rows(input_text)
    reject_penalty = max(100.0, max(scores, default=100.0) * 3.0)
    total = 0.0
    for task_id_list_str, courier_ids in result:
        fail_probability = 1.0
        for courier_id in courier_ids:
            score, willingness = rows[(task_id_list_str, courier_id)]
            total += fail_probability * willingness * score
            fail_probability *= 1.0 - willingness
        total += fail_probability * reject_penalty
    return total


def main() -> None:
    total_expected = 0.0
    print("case,expected,covered,total_tasks,groups,couriers,runtime_ms")
    for case_path in CASE_PATHS:
        input_text = case_path.read_text(encoding="utf-8")
        _rows, _scores, all_tasks = parse_rows(input_text)
        start = perf_counter()
        result = solver.solve(input_text)
        runtime_ms = (perf_counter() - start) * 1000
        covered = {task.strip() for task_id_list_str, _couriers in result for task in task_id_list_str.split(",")}
        couriers = {courier_id for _task_id_list_str, courier_ids in result for courier_id in courier_ids}
        score = expected_score(input_text, result)
        total_expected += score
        print(
            f"{case_path.stem},{score:.3f},{len(covered)},{len(all_tasks)},"
            f"{len(result)},{len(couriers)},{runtime_ms:.1f}"
        )
    print(f"average,{total_expected / len(CASE_PATHS):.3f},,,,,")


if __name__ == "__main__":
    main()
