from __future__ import annotations

from pathlib import Path

import solver
from tests.case_validation import assert_valid_submission


def _make_case(task_count: int, courier_count: int, include_bundles: bool = True) -> str:
    lines = ["task_id_list\tcourier_id\ttotal_score\twillingness"]
    for task in range(task_count):
        for courier in range(courier_count):
            score = 10 + ((task * 17 + courier * 13) % 90)
            willingness = 0.05 + (((task * 7 + courier * 11) % 90) / 100)
            lines.append(f"T{task:04d}\tC{courier:03d}\t{score:.3f}\t{willingness:.4f}")

    if include_bundles:
        for left in range(task_count):
            for right in range(left + 1, task_count):
                for courier in range(courier_count):
                    score = 18 + ((left * 19 + right * 23 + courier * 5) % 80)
                    willingness = 0.03 + (((left * 3 + right * 5 + courier * 7) % 90) / 100)
                    lines.append(
                        f"T{left:04d},T{right:04d}\tC{courier:03d}\t{score:.3f}\t{willingness:.4f}"
                    )
    return "\n".join(lines)


def _expected_score(input_text: str, result: list) -> float:
    rows = {}
    scores = []
    for row in input_text.strip().splitlines()[1:]:
        task_id_list_str, courier_id, score, willingness = row.split("\t")[:4]
        rows[(task_id_list_str, courier_id)] = (float(score), float(willingness))
        scores.append(float(score))

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


def test_solver_tiny_like_case_is_valid() -> None:
    input_text = _make_case(task_count=10, courier_count=20)
    result = solver.solve(input_text)

    assert_valid_submission(input_text, result)


def test_solver_small_like_case_is_valid() -> None:
    input_text = _make_case(task_count=20, courier_count=40)
    result = solver.solve(input_text)

    assert_valid_submission(input_text, result)


def test_solver_official_large_case_is_valid() -> None:
    input_text = Path("examples/large_seed301.txt").read_text(encoding="utf-8")
    result = solver.solve(input_text)

    assert_valid_submission(input_text, result)


def test_solver_uses_multiple_couriers_to_reduce_expected_score() -> None:
    input_text = "\n".join(
        [
            "task_id_list\tcourier_id\ttotal_score\twillingness",
            "T0001\tC001\t10.0\t0.20",
            "T0001\tC002\t11.0\t0.80",
            "T0001\tC003\t12.0\t0.70",
            "T0002\tC004\t10.0\t0.20",
            "T0002\tC005\t11.0\t0.80",
            "T0002\tC006\t12.0\t0.70",
        ]
    )

    result = solver.solve(input_text)
    single_courier_result = [(task_id_list_str, [courier_ids[0]]) for task_id_list_str, courier_ids in result]

    assert_valid_submission(input_text, result)
    assert any(len(courier_ids) > 1 for _task_id_list_str, courier_ids in result)
    assert _expected_score(input_text, result) < _expected_score(input_text, single_courier_result)
