from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class CandidateBundle:
    task_id_list: tuple[str, ...]
    task_id_list_str: str
    courier_id: str
    total_score: float
    willingness: float

    @property
    def task_count(self) -> int:
        return len(self.task_id_list)


@dataclass(frozen=True)
class CompetitionInstance:
    candidates: tuple[CandidateBundle, ...]

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(task_id for item in self.candidates for task_id in item.task_id_list)

    @property
    def courier_ids(self) -> frozenset[str]:
        return frozenset(item.courier_id for item in self.candidates)


@dataclass(frozen=True)
class ProblemMetadata:
    candidate_count: int
    task_count: int
    courier_count: int
    min_score: float
    mean_score: float
    max_score: float
    min_willingness: float
    mean_willingness: float
    max_willingness: float
    max_bundle_size: int


@dataclass(frozen=True)
class SolverResult:
    selected: tuple[CandidateBundle, ...]
    algorithm: str

    def to_submission(self) -> list[tuple[str, list[str]]]:
        return [(item.task_id_list_str, [item.courier_id]) for item in self.selected]


def parse_competition_input(input_text: str) -> CompetitionInstance:
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    candidates: list[CandidateBundle] = []

    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        task_id_list_str, courier_id, score_str, willingness_str = parts[:4]
        try:
            total_score = float(score_str)
            willingness = float(willingness_str)
        except ValueError:
            continue

        task_ids = tuple(task.strip() for task in task_id_list_str.split(",") if task.strip())
        if not task_ids or not courier_id.strip():
            continue

        candidates.append(
            CandidateBundle(
                task_id_list=task_ids,
                task_id_list_str=task_id_list_str.strip(),
                courier_id=courier_id.strip(),
                total_score=total_score,
                willingness=willingness,
            )
        )

    return CompetitionInstance(candidates=tuple(candidates))


def compute_metadata(instance: CompetitionInstance) -> ProblemMetadata:
    scores = [item.total_score for item in instance.candidates]
    willingness = [item.willingness for item in instance.candidates]
    if not scores:
        return ProblemMetadata(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    return ProblemMetadata(
        candidate_count=len(instance.candidates),
        task_count=len(instance.task_ids),
        courier_count=len(instance.courier_ids),
        min_score=min(scores),
        mean_score=sum(scores) / len(scores),
        max_score=max(scores),
        min_willingness=min(willingness),
        mean_willingness=sum(willingness) / len(willingness),
        max_willingness=max(willingness),
        max_bundle_size=max(item.task_count for item in instance.candidates),
    )


def result_key(selected: tuple[CandidateBundle, ...]) -> tuple[int, float, float]:
    covered = {task_id for item in selected for task_id in item.task_id_list}
    total_score = sum(item.total_score for item in selected)
    total_willingness = sum(item.willingness for item in selected)
    return (len(covered), -total_score, total_willingness)


def _is_better(candidate: tuple[CandidateBundle, ...], incumbent: tuple[CandidateBundle, ...]) -> bool:
    return result_key(candidate) > result_key(incumbent)


def _greedy_by_key(
    candidates: tuple[CandidateBundle, ...],
    key,
    algorithm: str,
) -> SolverResult:
    assigned_tasks: set[str] = set()
    assigned_couriers: set[str] = set()
    selected: list[CandidateBundle] = []

    for candidate in sorted(candidates, key=key):
        if candidate.courier_id in assigned_couriers:
            continue
        if any(task_id in assigned_tasks for task_id in candidate.task_id_list):
            continue
        assigned_couriers.add(candidate.courier_id)
        assigned_tasks.update(candidate.task_id_list)
        selected.append(candidate)

    return SolverResult(tuple(selected), algorithm)


def greedy_algorithm(instance: CompetitionInstance, metadata: ProblemMetadata) -> SolverResult:
    return _greedy_by_key(
        instance.candidates,
        lambda item: (
            item.total_score / item.task_count,
            item.total_score,
            -item.willingness,
            item.task_id_list_str,
            item.courier_id,
        ),
        "greedy",
    )


def heuristic_search_algorithm(instance: CompetitionInstance, metadata: ProblemMetadata) -> SolverResult:
    if not instance.candidates:
        return SolverResult((), "heuristic_search")

    min_score_by_task: dict[str, float] = {}
    for candidate in instance.candidates:
        for task_id in candidate.task_id_list:
            min_score_by_task[task_id] = min(
                min_score_by_task.get(task_id, candidate.total_score),
                candidate.total_score / candidate.task_count,
            )

    def rarity(candidate: CandidateBundle) -> float:
        return sum(min_score_by_task[task_id] for task_id in candidate.task_id_list) / candidate.task_count

    key_functions = [
        lambda item: (
            item.total_score / item.task_count,
            item.total_score,
            -item.willingness,
            item.task_id_list_str,
        ),
        lambda item: (
            item.total_score / item.task_count - metadata.mean_willingness * item.willingness,
            item.total_score,
            -item.willingness,
            item.task_id_list_str,
        ),
        lambda item: (
            (item.total_score / item.task_count) - rarity(item),
            -item.task_count,
            -item.willingness,
            item.total_score,
        ),
        lambda item: (
            -item.task_count,
            item.total_score / item.task_count,
            -item.willingness,
            item.total_score,
        ),
    ]

    best = SolverResult((), "heuristic_search")
    for key in key_functions:
        candidate_result = _greedy_by_key(instance.candidates, key, "heuristic_search")
        if _is_better(candidate_result.selected, best.selected):
            best = candidate_result

    return best


def branch_bound_algorithm(
    instance: CompetitionInstance,
    metadata: ProblemMetadata,
    time_limit_seconds: float = 1.5,
    max_candidates_per_task: int = 24,
) -> SolverResult:
    if not instance.candidates:
        return SolverResult((), "branch_bound")

    task_ids = sorted(instance.task_ids)
    task_to_bit = {task_id: 1 << index for index, task_id in enumerate(task_ids)}
    courier_to_bit = {courier_id: 1 << index for index, courier_id in enumerate(sorted(instance.courier_ids))}

    candidate_rows = []
    by_task: dict[str, list[tuple[int, int, CandidateBundle]]] = {task_id: [] for task_id in task_ids}
    for candidate in instance.candidates:
        task_mask = 0
        for task_id in candidate.task_id_list:
            task_mask |= task_to_bit[task_id]
        courier_mask = courier_to_bit[candidate.courier_id]
        row = (task_mask, courier_mask, candidate)
        candidate_rows.append(row)
        for task_id in candidate.task_id_list:
            by_task[task_id].append(row)

    for task_id, rows in by_task.items():
        rows.sort(
            key=lambda row: (
                row[2].total_score / row[2].task_count,
                row[2].total_score,
                -row[2].willingness,
            )
        )
        del rows[max_candidates_per_task:]

    greedy = heuristic_search_algorithm(instance, metadata).selected
    best: tuple[CandidateBundle, ...] = greedy
    start = perf_counter()
    full_mask = (1 << len(task_ids)) - 1

    def upper_bound_covered(task_mask: int) -> int:
        return task_mask.bit_count() + (full_mask ^ task_mask).bit_count()

    def search(task_mask: int, courier_mask: int, selected: list[CandidateBundle]) -> None:
        nonlocal best
        if perf_counter() - start > time_limit_seconds:
            return
        if upper_bound_covered(task_mask) < result_key(best)[0]:
            return
        if task_mask == full_mask:
            current = tuple(selected)
            if _is_better(current, best):
                best = current
            return

        uncovered = [task_id for task_id in task_ids if not (task_mask & task_to_bit[task_id])]
        pivot = min(uncovered, key=lambda task_id: len(by_task[task_id]))

        # Branch 1: select a candidate covering the hardest uncovered task.
        for candidate_task_mask, candidate_courier_mask, candidate in by_task[pivot]:
            if task_mask & candidate_task_mask:
                continue
            if courier_mask & candidate_courier_mask:
                continue
            selected.append(candidate)
            search(task_mask | candidate_task_mask, courier_mask | candidate_courier_mask, selected)
            selected.pop()

        # Branch 2: allow this task to remain uncovered.
        search(task_mask | task_to_bit[pivot], courier_mask, selected)

        current = tuple(selected)
        if _is_better(current, best):
            best = current

    search(0, 0, [])
    return SolverResult(best, "branch_bound")


def llm_direct_reasoning_algorithm(instance: CompetitionInstance, metadata: ProblemMetadata) -> SolverResult:
    # The judge environment cannot call an external LLM. This strategy keeps the
    # same library slot and uses a deterministic rule-based approximation.
    return heuristic_search_algorithm(instance, metadata)


class AutoSolverAgent:
    def decide_algorithm(self, metadata: ProblemMetadata) -> str:
        if metadata.candidate_count == 0:
            return "greedy"
        if metadata.max_bundle_size >= 2:
            return "heuristic_search"
        return "greedy"

    def run(self, instance: CompetitionInstance) -> SolverResult:
        metadata = compute_metadata(instance)
        algorithm = self.decide_algorithm(metadata)

        if algorithm == "branch_bound":
            return branch_bound_algorithm(instance, metadata)
        if algorithm == "heuristic_search":
            return heuristic_search_algorithm(instance, metadata)
        if algorithm == "llm_direct_reasoning":
            return llm_direct_reasoning_algorithm(instance, metadata)
        return greedy_algorithm(instance, metadata)


def greedy_solve_competition(instance: CompetitionInstance) -> list[tuple[str, list[str]]]:
    return greedy_algorithm(instance, compute_metadata(instance)).to_submission()


def solve_competition_text(input_text: str) -> list[tuple[str, list[str]]]:
    instance = parse_competition_input(input_text)
    return AutoSolverAgent().run(instance).to_submission()
