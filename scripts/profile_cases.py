from __future__ import annotations

from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import solver


CASE_PATHS = [
    Path("examples/large_seed301.txt"),
    *sorted(Path("examples/blackbox_like").glob("*.txt")),
    *sorted(Path("examples/probe_suite").glob("*.txt")),
]


def quantile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    return sorted(values)[int((len(values) - 1) * ratio)]


def profile(path: Path) -> dict[str, float | int | str]:
    text = path.read_text(encoding="utf-8")
    candidates = solver._parse_input(text)
    metadata = solver._metadata(candidates)
    willingness = [item[4] for item in candidates]
    singles = [item[4] for item in candidates if len(item[0]) == 1]
    pairs = [item[4] for item in candidates if len(item[0]) == 2]
    pressure = solver._willingness_pressure(candidates)
    return {
        "case": path.stem,
        "tasks": metadata["task_count"],
        "couriers": metadata["courier_count"],
        "mean_w": statistics.mean(willingness) if willingness else 0.0,
        "q10_w": quantile(willingness, 0.1),
        "q50_w": quantile(willingness, 0.5),
        "q90_w": quantile(willingness, 0.9),
        "single_w": statistics.mean(singles) if singles else 0.0,
        "pair_w": statistics.mean(pairs) if pairs else 0.0,
        "top5_w": pressure["top5_mean"],
        "cheap5_w": pressure["cheap5_mean"],
        "low_flag": int(solver._is_low_willingness_case(candidates, metadata)),
    }


def main() -> None:
    fields = [
        "case",
        "tasks",
        "couriers",
        "mean_w",
        "q10_w",
        "q50_w",
        "q90_w",
        "single_w",
        "pair_w",
        "top5_w",
        "cheap5_w",
        "low_flag",
    ]
    print(",".join(fields))
    for path in CASE_PATHS:
        row = profile(path)
        print(",".join(str(round(row[field], 4)) if isinstance(row[field], float) else str(row[field]) for field in fields))


if __name__ == "__main__":
    main()
