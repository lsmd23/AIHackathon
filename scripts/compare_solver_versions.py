from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from time import perf_counter
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import solver as current_solver
from scripts.evaluate_solver import CASE_PATHS, parse_rows, ranked_expected_score


def load_head_solver():
    source = subprocess.check_output(["git", "show", "HEAD:solver.py"], text=True)
    module = types.ModuleType("head_solver")
    exec(compile(source, "HEAD:solver.py", "exec"), module.__dict__)
    return module


def score(module, input_text: str):
    start = perf_counter()
    result = module.solve(input_text)
    runtime_ms = (perf_counter() - start) * 1000
    return ranked_expected_score(input_text, result), runtime_ms


def main() -> None:
    head_solver = load_head_solver()
    print("case,family,head_ranked,current_ranked,delta,head_ms,current_ms")
    total_head = 0.0
    total_current = 0.0
    for path in CASE_PATHS:
        text = path.read_text(encoding="utf-8")
        rows, _scores, _tasks = parse_rows(text)
        candidates = current_solver._parse_input(text)
        meta = current_solver._metadata(candidates)
        if path.parent.name == "probe_suite":
            family = "probe_" + path.stem.split("_")[1]
        elif path.name == "large_seed301.txt" and path.parent.name == "examples":
            family = "official_sample"
        else:
            family = path.stem.split("_seed")[0]
        head_value, head_ms = score(head_solver, text)
        current_value, current_ms = score(current_solver, text)
        total_head += head_value
        total_current += current_value
        print(
            f"{path.stem},{family},{head_value:.3f},{current_value:.3f},"
            f"{current_value - head_value:.3f},{head_ms:.1f},{current_ms:.1f}"
        )
    count = max(len(CASE_PATHS), 1)
    print(f"average,all,{total_head / count:.3f},{total_current / count:.3f},{(total_current - total_head) / count:.3f},,")


if __name__ == "__main__":
    main()
