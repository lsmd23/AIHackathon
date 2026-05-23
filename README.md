# Hackathon AutoSolver

AutoSolver is an initial competition framework for an AI Agent that autonomously explores and solves delivery order dispatching problems.

The target problem is to assign delivery tasks to riders under time, distance, rider willingness, order rejection, and bundle delivery constraints. The system should maximize accepted orders and minimize the total score/cost within the competition time limit.

## Goals

- Provide a clear implementation plan for a two-person team.
- Keep a runnable baseline solver from day one.
- Leave extension points for greedy, ILP, heuristic search, and LLM-guided strategy selection.
- Standardize repository structure, data models, tests, and CLI entry points.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
python -m autosolver_agent examples/sample_case.json
pytest
```

## Repository Structure

```text
.
├── docs/
│   ├── architecture.md
│   └── contest_plan.md
├── examples/
│   └── sample_case.json
├── src/
│   └── autosolver_agent/
│       ├── agent.py
│       ├── cli.py
│       ├── models.py
│       ├── scoring.py
│       ├── solver.py
│       └── strategies/
│           ├── base.py
│           └── greedy.py
└── tests/
    └── test_scoring.py
```

## Current Baseline

The baseline implements a deterministic greedy strategy:

1. Sort orders by predicted benefit.
2. Try every rider for each order.
3. Accept the lowest incremental cost feasible assignment.
4. Skip an order when no rider can serve it.

This is intentionally simple. It gives the team a stable reference for scoring, testing, and later strategy comparison.
