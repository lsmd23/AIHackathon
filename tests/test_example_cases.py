from __future__ import annotations

from pathlib import Path

import pytest

import solver
from tests.case_validation import assert_valid_submission


CASE_PATH = Path("examples/large_seed301.txt")


def test_solver_large_seed301_is_legal_and_complete() -> None:
    input_text = CASE_PATH.read_text(encoding="utf-8")
    result = solver.solve(input_text)

    assert_valid_submission(input_text, result, require_complete=True)
    assert any(len(courier_ids) > 1 for _task_id_list_str, courier_ids in result), \
        "large case should attach backup couriers to some bundles"
