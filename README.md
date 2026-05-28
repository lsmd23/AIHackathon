# AIHackathon AutoSolver

AutoSolver is a competition solver for the official precomputed dispatch candidate format.

The current task is to choose non-conflicting task-bundle/courier candidates from a TSV input. The solver should cover as many tasks as possible, minimize expected score, and use courier willingness as acceptance probability.

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
│   ├── blackbox_like/
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

The current baseline implements a deterministic multi-strategy Agent:

1. Parse candidate rows from TSV.
2. Compute metadata such as task count, courier count, score statistics, and bundle size.
3. Generate task partitions with greedy rules, low-willingness split rules, and component-level bitmask DP.
4. For each partition, compare seeded backup assignment with global marginal-gain courier assignment.
5. Evaluate expected score and assigned-cost proxy, then keep the best complete plan.

This baseline has passed the current public evaluation set: `10/10` cases, all with `100%` task coverage and an average penalty score of `771.89`. See [baseline_evaluation.md](docs/baseline_evaluation.md).

## Local Cases

Black-box-like TSV cases live in `examples/blackbox_like/`. Regenerate them with:

```powershell
python scripts\generate_blackbox_like_examples.py
```

Run the unified example-case interface with:

```powershell
python -m pytest tests\test_example_cases.py
```

Run the local score proxy:

```powershell
python scripts\evaluate_solver.py
```
