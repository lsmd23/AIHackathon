from __future__ import annotations

from pathlib import Path

import pytest

import solver
from tests.case_validation import assert_valid_submission


CASE_PATHS = [
    Path("examples/large_seed301.txt"),
    *sorted(Path("examples/blackbox_like").glob("*.txt")),
]


@pytest.mark.parametrize("case_path", CASE_PATHS, ids=lambda path: path.stem)
def test_solver_example_case_file(case_path: Path) -> None:
    input_text = case_path.read_text(encoding="utf-8")
    result = solver.solve(input_text)

    assert_valid_submission(input_text, result, require_complete=True)


def test_low_willingness_example_uses_backup_couriers() -> None:
    input_text = Path("examples/blackbox_like/low_willingness_seed501.txt").read_text(encoding="utf-8")
    result = solver.solve(input_text)

    assert_valid_submission(input_text, result, require_complete=True)
    assert any(len(courier_ids) > 1 for _task_id_list_str, courier_ids in result)
