# AIHackathon AutoSolver

AutoSolver is a competition solver for the official precomputed dispatch candidate format.

The current task is to choose non-conflicting task-bundle/courier candidates from a TSV input. The solver should cover as many tasks as possible, then minimize total score, with courier willingness as a secondary signal.

## Goals

- Match the official `solve(input_text: str) -> list` submission API.
- Keep `solver.py` self-contained for upload.
- Maintain a local package version for tests and strategy iteration.
- Optimize candidate selection under task and courier conflict constraints.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
pytest
```

## Competition Submission

The current online judge requires a `solver.py` file with:

```python
def solve(input_text: str) -> list:
    ...
```

The repository includes [solver.py](solver.py), which parses the official TSV candidate format and returns:

```python
[(task_id_list_str, [courier_id]), ...]
```

Local smoke test:

```powershell
python -c "from pathlib import Path; import solver; print(len(solver.solve(Path('examples/large_seed301.txt').read_text(encoding='utf-8'))))"
```

## Repository Structure

```text
.
├── docs/
│   ├── 0.md
│   ├── architecture.md
│   ├── baseline_evaluation.md
│   └── contest_plan.md
├── examples/
│   ├── example_solution.py
│   └── large_seed301.txt
├── solver.py
├── src/
│   └── autosolver_agent/
│       ├── competition.py
│       ├── cli.py
│       └── __init__.py
└── tests/
    ├── test_competition.py
    └── test_solver_submission.py
```

## Current Baseline

The baseline implements a lightweight deterministic Agent:

1. Parse candidate rows from TSV.
2. Compute metadata such as task count, courier count, score statistics, and bundle size.
3. Choose `heuristic_search` for bundle-heavy cases, otherwise `greedy`.
4. Select non-conflicting candidates and return official submission-shaped tuples.

This baseline has passed the current public evaluation set: `10/10` cases, all with `100%` task coverage. See [baseline_evaluation.md](docs/baseline_evaluation.md).
