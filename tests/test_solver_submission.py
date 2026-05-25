from __future__ import annotations

from pathlib import Path

import solver


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


def _assert_valid_submission(input_text: str, result: list) -> None:
    rows = input_text.strip().splitlines()[1:]
    valid_pairs = set()
    known_tasks = set()
    known_couriers = set()
    for row in rows:
        task_id_list_str, courier_id, *_rest = row.split("\t")
        valid_pairs.add((task_id_list_str, courier_id))
        known_couriers.add(courier_id)
        known_tasks.update(task_id_list_str.split(","))

    used_tasks = set()
    used_couriers = set()
    for task_id_list_str, courier_ids in result:
        assert isinstance(task_id_list_str, str)
        assert isinstance(courier_ids, list)
        assert len(courier_ids) == 1
        courier_id = courier_ids[0]
        assert (task_id_list_str, courier_id) in valid_pairs
        assert courier_id not in used_couriers
        assert courier_id in known_couriers
        used_couriers.add(courier_id)

        for task_id in task_id_list_str.split(","):
            assert task_id in known_tasks
            assert task_id not in used_tasks
            used_tasks.add(task_id)


def test_solver_tiny_like_case_is_valid() -> None:
    input_text = _make_case(task_count=10, courier_count=20)
    result = solver.solve(input_text)

    _assert_valid_submission(input_text, result)


def test_solver_small_like_case_is_valid() -> None:
    input_text = _make_case(task_count=20, courier_count=40)
    result = solver.solve(input_text)

    _assert_valid_submission(input_text, result)


def test_solver_official_large_case_is_valid() -> None:
    input_text = Path("examples/large_seed301.txt").read_text(encoding="utf-8")
    result = solver.solve(input_text)

    _assert_valid_submission(input_text, result)
