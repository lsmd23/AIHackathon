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
    top5_willingness: float = 1.0
    cheap5_willingness: float = 1.0


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

    pressure = willingness_pressure(instance.candidates)
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
        top5_willingness=pressure["top5_mean"],
        cheap5_willingness=pressure["cheap5_mean"],
    )


def willingness_pressure(candidates: tuple[CandidateBundle, ...]) -> dict[str, float]:
    by_bundle: dict[str, list[CandidateBundle]] = {}
    for candidate in candidates:
        by_bundle.setdefault(candidate.task_id_list_str, []).append(candidate)
    if not by_bundle:
        return {"top5_mean": 1.0, "cheap5_mean": 1.0}

    top_values: list[float] = []
    cheap_values: list[float] = []
    for rows in by_bundle.values():
        top = sorted((row.willingness for row in rows), reverse=True)[:5]
        cheap = sorted(rows, key=lambda row: (row.total_score, -row.willingness))[:5]
        top_values.append(sum(top) / len(top))
        cheap_values.append(sum(row.willingness for row in cheap) / len(cheap))
    return {
        "top5_mean": sum(top_values) / len(top_values),
        "cheap5_mean": sum(cheap_values) / len(cheap_values),
    }


def is_low_willingness_case(
    candidates: tuple[CandidateBundle, ...],
    metadata: ProblemMetadata,
) -> bool:
    return (
        metadata.mean_willingness < 0.18
        or metadata.top5_willingness < 0.45
        or metadata.cheap5_willingness < 0.22
    )


def robust_alpha(candidates: tuple[CandidateBundle, ...], metadata: ProblemMetadata) -> float:
    if is_low_willingness_case(candidates, metadata):
        return 0.65
    if metadata.courier_count / max(metadata.task_count, 1) < 1.25:
        return 1.2
    return 0.85


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

    bundle_profiles = _bundle_profiles(instance.candidates, metadata)
    low_case = is_low_willingness_case(instance.candidates, metadata)
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
            bundle_profiles[item.task_id_list_str]["expected"] / item.task_count,
            bundle_profiles[item.task_id_list_str]["fail_probability"],
            item.total_score,
            -item.willingness,
            item.task_id_list_str,
        ),
        lambda item: (
            bundle_profiles[item.task_id_list_str]["expected"] / item.task_count
            - metadata.mean_willingness * bundle_profiles[item.task_id_list_str]["success_probability"],
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

    def evaluated_result_key(selected: tuple[CandidateBundle, ...]) -> tuple[int, float, float]:
        covered = {task_id for item in selected for task_id in item.task_id_list}
        submission = assign_backup_couriers(instance.candidates, list(selected), metadata)
        expected = expected_submission_score(instance.candidates, submission, metadata)
        primary_score = sum(item.total_score for item in selected)
        is_scarce = metadata.courier_count / max(metadata.task_count, 1) < 1.25
        is_low_willingness = low_case
        if is_scarce or is_low_willingness:
            row_map = {(candidate.task_id_list_str, candidate.courier_id): candidate for candidate in instance.candidates}
            assigned_score = sum(
                row_map[(task_id_list_str, courier_id)].total_score
                for task_id_list_str, courier_ids in submission
                for courier_id in courier_ids
            )
            expected = max(expected, assigned_score * robust_alpha(instance.candidates, metadata))
        return (len(covered), -expected, -primary_score)

    best = SolverResult((), "heuristic_search")
    for key in key_functions:
        candidate_result = _greedy_by_key(instance.candidates, key, "heuristic_search")
        if evaluated_result_key(candidate_result.selected) > evaluated_result_key(best.selected):
            best = candidate_result
    for selected in pair_first_candidates(instance.candidates, bundle_profiles):
        selected_tuple = tuple(selected)
        if evaluated_result_key(selected_tuple) > evaluated_result_key(best.selected):
            best = SolverResult(selected_tuple, "heuristic_search")
    for selected in minimum_group_candidates(instance.candidates, bundle_profiles):
        selected_tuple = tuple(selected)
        if evaluated_result_key(selected_tuple) > evaluated_result_key(best.selected):
            best = SolverResult(selected_tuple, "heuristic_search")
    for selected in budgeted_pairing_candidates(instance.candidates, metadata):
        selected_tuple = tuple(selected)
        if evaluated_result_key(selected_tuple) > evaluated_result_key(best.selected):
            best = SolverResult(selected_tuple, "heuristic_search")
    if low_case:
        for selected in low_willingness_candidates(instance.candidates, bundle_profiles):
            selected_tuple = tuple(selected)
            if evaluated_result_key(selected_tuple) > evaluated_result_key(best.selected):
                best = SolverResult(selected_tuple, "heuristic_search")
        for selected in hybrid_split_candidates(instance.candidates, metadata):
            selected_tuple = tuple(selected)
            if evaluated_result_key(selected_tuple) > evaluated_result_key(best.selected):
                best = SolverResult(selected_tuple, "heuristic_search")

    local_budget = 0.25 if metadata.candidate_count > 30000 else 0.7
    improved = tuple(local_search_improve(instance.candidates, list(best.selected), time_limit_seconds=local_budget))
    if evaluated_result_key(improved) > evaluated_result_key(best.selected):
        return SolverResult(improved, "heuristic_search")
    return best


def pair_first_candidates(
    candidates: tuple[CandidateBundle, ...],
    bundle_profiles: dict[str, dict[str, float]],
) -> list[list[CandidateBundle]]:
    pair_rows = [item for item in candidates if item.task_count == 2]
    single_rows = [item for item in candidates if item.task_count == 1]
    single_best: dict[str, CandidateBundle] = {}
    for item in sorted(
        single_rows,
        key=lambda row: (row.total_score / max(row.willingness, 0.05), row.total_score, -row.willingness),
    ):
        single_best.setdefault(item.task_id_list[0], item)

    sorters = [
        lambda item: (
            bundle_profiles[item.task_id_list_str]["expected"] / 2,
            bundle_profiles[item.task_id_list_str]["fail_probability"],
            item.total_score,
            -item.willingness,
        ),
        lambda item: (
            item.total_score / 2,
            -bundle_profiles[item.task_id_list_str]["success_probability"],
            item.total_score,
            -item.willingness,
        ),
        lambda item: (
            -bundle_profiles[item.task_id_list_str]["success_probability"],
            bundle_profiles[item.task_id_list_str]["expected"] / 2,
            item.total_score,
        ),
    ]

    results: list[list[CandidateBundle]] = []
    for sorter in sorters:
        used_tasks: set[str] = set()
        used_couriers: set[str] = set()
        selected: list[CandidateBundle] = []
        for item in sorted(pair_rows, key=sorter):
            if any(task in used_tasks for task in item.task_id_list):
                continue
            if item.courier_id in used_couriers:
                continue
            selected.append(item)
            used_tasks.update(item.task_id_list)
            used_couriers.add(item.courier_id)
        for task_id, item in single_best.items():
            if task_id in used_tasks or item.courier_id in used_couriers:
                continue
            selected.append(item)
            used_tasks.add(task_id)
            used_couriers.add(item.courier_id)
        results.append(selected)
    return results


def low_willingness_candidates(
    candidates: tuple[CandidateBundle, ...],
    bundle_profiles: dict[str, dict[str, float]],
) -> list[list[CandidateBundle]]:
    pair_rows = [item for item in candidates if item.task_count == 2]
    single_rows = [item for item in candidates if item.task_count == 1]
    single_best: dict[str, CandidateBundle] = {}
    for item in sorted(
        single_rows,
        key=lambda row: (row.total_score / max(row.willingness, 0.03), row.total_score, -row.willingness),
    ):
        single_best.setdefault(item.task_id_list[0], item)

    def pair_value(item: CandidateBundle) -> tuple[float, float, float, float]:
        profile = bundle_profiles[item.task_id_list_str]
        return (
            profile["fail_probability"],
            profile["expected"] / 2,
            item.total_score / max(item.willingness, 0.03),
            item.total_score,
        )

    results: list[list[CandidateBundle]] = []
    for reverse_success in (False, True):
        used_tasks: set[str] = set()
        used_couriers: set[str] = set()
        selected: list[CandidateBundle] = []
        rows = sorted(
            pair_rows,
            key=(
                (
                    lambda item: (
                        -bundle_profiles[item.task_id_list_str]["success_probability"],
                        item.total_score / max(item.willingness, 0.03),
                        item.total_score,
                    )
                )
                if reverse_success
                else pair_value
            ),
        )
        for item in rows:
            if any(task in used_tasks for task in item.task_id_list):
                continue
            if item.courier_id in used_couriers:
                continue
            selected.append(item)
            used_tasks.update(item.task_id_list)
            used_couriers.add(item.courier_id)
        for task_id, item in single_best.items():
            if task_id in used_tasks or item.courier_id in used_couriers:
                continue
            selected.append(item)
            used_tasks.add(task_id)
            used_couriers.add(item.courier_id)
        results.append(selected)
    return results


def hybrid_split_candidates(
    candidates: tuple[CandidateBundle, ...],
    metadata: ProblemMetadata,
) -> list[list[CandidateBundle]]:
    single_rows = [item for item in candidates if item.task_count == 1]
    pair_rows = [item for item in candidates if item.task_count == 2]
    if not single_rows or not pair_rows:
        return []

    by_bundle: dict[str, list[CandidateBundle]] = {}
    for candidate in candidates:
        by_bundle.setdefault(candidate.task_id_list_str, []).append(candidate)
    for rows in by_bundle.values():
        rows.sort(
            key=lambda item: (
                item.total_score / max(item.willingness, 0.03),
                item.total_score,
                -item.willingness,
                item.courier_id,
            )
        )

    reject_penalty = _reject_penalty(metadata)

    def preview_cost(rows: list[CandidateBundle], limit: int = 4, alpha: float = 0.85) -> float:
        preview = rows[:limit]
        expected = _expected_bundle_score(preview, reject_penalty)
        assigned = sum(row.total_score for row in preview)
        return max(expected, assigned * alpha)

    single_info: dict[str, dict] = {}
    for item in single_rows:
        task_id = item.task_id_list[0]
        if task_id in single_info:
            continue
        rows = by_bundle[item.task_id_list_str]
        single_info[task_id] = {
            "rows": rows,
            "cost": preview_cost(rows),
        }

    pair_infos = []
    seen: set[frozenset[str]] = set()
    for item in pair_rows:
        tasks = frozenset(item.task_id_list)
        if tasks in seen:
            continue
        seen.add(tasks)
        left, right = item.task_id_list
        if left not in single_info or right not in single_info:
            continue
        rows = by_bundle[item.task_id_list_str]
        pair_cost = preview_cost(rows)
        split_cost = single_info[left]["cost"] + single_info[right]["cost"]
        pair_infos.append(
            {
                "tasks": tasks,
                "rows": rows,
                "saving": split_cost - pair_cost,
                "cost": pair_cost,
            }
        )

    def build(min_saving: float) -> list[CandidateBundle]:
        used_tasks: set[str] = set()
        used_couriers: set[str] = set()
        selected: list[CandidateBundle] = []
        for info in sorted(pair_infos, key=lambda row: (-row["saving"], row["cost"])):
            if info["saving"] < min_saving:
                continue
            if info["tasks"] & used_tasks:
                continue
            row = next((candidate for candidate in info["rows"][:80] if candidate.courier_id not in used_couriers), None)
            if row is None:
                continue
            selected.append(row)
            used_tasks.update(info["tasks"])
            used_couriers.add(row.courier_id)

        for task_id, info in sorted(single_info.items(), key=lambda item: item[1]["cost"]):
            if task_id in used_tasks:
                continue
            row = next((candidate for candidate in info["rows"][:80] if candidate.courier_id not in used_couriers), None)
            if row is None:
                continue
            selected.append(row)
            used_tasks.add(task_id)
            used_couriers.add(row.courier_id)
        return selected

    thresholds = (-50.0, 0.0, 25.0, 50.0, 100.0)
    return [build(threshold) for threshold in thresholds]


def low_beam_candidates(
    candidates: tuple[CandidateBundle, ...],
    metadata: ProblemMetadata,
) -> list[list[CandidateBundle]]:
    single_rows = [item for item in candidates if item.task_count == 1]
    pair_rows = [item for item in candidates if item.task_count == 2]
    if not single_rows or not pair_rows:
        return []

    by_bundle: dict[str, list[CandidateBundle]] = {}
    for candidate in candidates:
        by_bundle.setdefault(candidate.task_id_list_str, []).append(candidate)
    for rows in by_bundle.values():
        rows.sort(
            key=lambda item: (
                item.total_score / max(item.willingness, 0.03),
                item.total_score,
                -item.willingness,
                item.courier_id,
            )
        )

    reject_penalty = _reject_penalty(metadata)
    alpha = robust_alpha(candidates, metadata)
    all_tasks = sorted({task for item in candidates for task in item.task_id_list})
    full_task_count = len(all_tasks)
    choices_by_task: dict[str, list[tuple[frozenset[str], list[CandidateBundle], float]]] = {task: [] for task in all_tasks}
    seen_groups: set[frozenset[str]] = set()

    for item in single_rows + pair_rows:
        task_set = frozenset(item.task_id_list)
        if task_set in seen_groups:
            continue
        seen_groups.add(task_set)
        rows = by_bundle[item.task_id_list_str]
        preview = rows[:4]
        expected = _expected_bundle_score(preview, reject_penalty)
        assigned = sum(row.total_score for row in preview)
        cost = max(expected, assigned * alpha)
        choice = (task_set, rows, cost)
        for task in task_set:
            choices_by_task[task].append(choice)

    for choices in choices_by_task.values():
        choices.sort(key=lambda choice: (choice[2] / len(choice[0]), choice[2]))

    results: list[list[CandidateBundle]] = []
    for pair_bonus in (-100.0, 0.0, 25.0):
        states = [(0.0, frozenset(), frozenset(), [])]
        for _ in range(full_task_count):
            next_states = []
            for cost, covered, used_couriers, selected in states:
                if len(covered) == full_task_count:
                    next_states.append((cost, covered, used_couriers, selected))
                    continue
                pivot = next(task for task in all_tasks if task not in covered)
                for task_set, rows, group_cost in choices_by_task[pivot][:45]:
                    if task_set & covered:
                        continue
                    row = next((candidate for candidate in rows[:50] if candidate.courier_id not in used_couriers), None)
                    if row is None:
                        continue
                    bonus = pair_bonus if len(task_set) == 2 else 0.0
                    next_states.append(
                        (
                            cost + group_cost - bonus,
                            covered | task_set,
                            used_couriers | {row.courier_id},
                            selected + [row],
                        )
                    )
            states = sorted(next_states, key=lambda state: (state[0], len(state[3])))[:260]
            if states and all(len(state[1]) == full_task_count for state in states):
                break
        results.extend(state[3] for state in states[:2] if len(state[1]) == full_task_count)
    return results


def budgeted_pairing_candidates(
    candidates: tuple[CandidateBundle, ...],
    metadata: ProblemMetadata,
) -> list[list[CandidateBundle]]:
    pair_rows = [item for item in candidates if item.task_count == 2]
    single_rows = [item for item in candidates if item.task_count == 1]
    if not pair_rows:
        return []

    by_bundle: dict[str, list[CandidateBundle]] = {}
    for candidate in candidates:
        by_bundle.setdefault(candidate.task_id_list_str, []).append(candidate)
    for rows in by_bundle.values():
        rows.sort(
            key=lambda item: (
                item.total_score / max(item.willingness, 0.03),
                item.total_score,
                -item.willingness,
                item.courier_id,
            )
        )

    reject_penalty = _reject_penalty(metadata)
    low_case = is_low_willingness_case(candidates, metadata)
    budget = 4 if low_case else 2
    if metadata.courier_count / max(metadata.task_count, 1) < 1.25:
        budget = 2

    pair_infos = []
    seen_pairs: set[frozenset[str]] = set()
    for item in pair_rows:
        task_pair = frozenset(item.task_id_list)
        if task_pair in seen_pairs:
            continue
        seen_pairs.add(task_pair)
        rows = by_bundle[item.task_id_list_str]
        preview = rows[:budget]
        fail_probability = 1.0
        for row in preview:
            fail_probability *= 1.0 - max(0.0, min(1.0, row.willingness))
        pair_infos.append(
            {
                "tasks": task_pair,
                "bundle": item.task_id_list_str,
                "rows": rows,
                "primary": rows[0],
                "expected": _expected_bundle_score(preview, reject_penalty),
                "fail": fail_probability,
                "success": 1.0 - fail_probability,
            }
        )

    single_best: dict[str, CandidateBundle] = {}
    for item in sorted(
        single_rows,
        key=lambda row: (row.total_score / max(row.willingness, 0.03), row.total_score, -row.willingness, row.courier_id),
    ):
        single_best.setdefault(item.task_id_list[0], item)

    def build(sorter) -> list[CandidateBundle]:
        used_tasks: set[str] = set()
        used_couriers: set[str] = set()
        selected: list[CandidateBundle] = []
        for info in sorted(pair_infos, key=sorter):
            if info["tasks"] & used_tasks:
                continue
            chosen = None
            for row in info["rows"][:80]:
                if row.courier_id not in used_couriers:
                    chosen = row
                    break
            if chosen is None:
                continue
            selected.append(chosen)
            used_tasks.update(info["tasks"])
            used_couriers.add(chosen.courier_id)

        for task_id, item in single_best.items():
            if task_id in used_tasks:
                continue
            chosen = item
            if chosen.courier_id in used_couriers:
                for row in by_bundle.get(chosen.task_id_list_str, ())[:80]:
                    if row.courier_id not in used_couriers:
                        chosen = row
                        break
                else:
                    continue
            selected.append(chosen)
            used_tasks.add(task_id)
            used_couriers.add(chosen.courier_id)
        return selected

    sorters = [
        lambda info: (info["expected"] / 2, info["fail"], info["primary"].total_score, -info["primary"].willingness),
        lambda info: (
            info["fail"],
            info["expected"] / 2,
            info["primary"].total_score / max(info["primary"].willingness, 0.03),
        ),
        lambda info: (-info["success"], info["expected"] / 2, info["primary"].total_score),
        lambda info: (info["primary"].total_score / max(info["primary"].willingness, 0.03), info["expected"] / 2),
        lambda info: (info["primary"].total_score, -info["primary"].willingness, info["expected"] / 2),
    ]
    return [build(sorter) for sorter in sorters]


def minimum_group_candidates(
    candidates: tuple[CandidateBundle, ...],
    bundle_profiles: dict[str, dict[str, float]],
) -> list[list[CandidateBundle]]:
    pair_rows = [item for item in candidates if item.task_count == 2]
    single_rows = [item for item in candidates if item.task_count == 1]
    single_best: dict[str, CandidateBundle] = {}
    for item in sorted(single_rows, key=lambda row: (row.total_score, -row.willingness, row.courier_id)):
        single_best.setdefault(item.task_id_list[0], item)

    sorters = [
        lambda item: (-item.task_count, bundle_profiles[item.task_id_list_str]["expected"] / 2, item.total_score),
        lambda item: (-item.task_count, item.total_score / 2, -item.willingness, item.total_score),
        lambda item: (
            -item.task_count,
            -bundle_profiles[item.task_id_list_str]["success_probability"],
            bundle_profiles[item.task_id_list_str]["expected"],
        ),
    ]
    results: list[list[CandidateBundle]] = []
    for sorter in sorters:
        used_tasks: set[str] = set()
        used_couriers: set[str] = set()
        selected: list[CandidateBundle] = []
        for item in sorted(pair_rows, key=sorter):
            if any(task in used_tasks for task in item.task_id_list) or item.courier_id in used_couriers:
                continue
            selected.append(item)
            used_tasks.update(item.task_id_list)
            used_couriers.add(item.courier_id)
        for task_id, item in single_best.items():
            if task_id in used_tasks or item.courier_id in used_couriers:
                continue
            selected.append(item)
            used_tasks.add(task_id)
            used_couriers.add(item.courier_id)
        results.append(selected)
    return results


def _reject_penalty(metadata: ProblemMetadata) -> float:
    return 100.0


def _bundle_profiles(
    candidates: tuple[CandidateBundle, ...],
    metadata: ProblemMetadata,
    max_rows: int = 5,
) -> dict[str, dict[str, float]]:
    by_bundle: dict[str, list[CandidateBundle]] = {}
    for candidate in candidates:
        by_bundle.setdefault(candidate.task_id_list_str, []).append(candidate)

    reject_penalty = _reject_penalty(metadata)
    profiles: dict[str, dict[str, float]] = {}
    for bundle, rows in by_bundle.items():
        ranked = sorted(
            rows,
            key=lambda item: (
                item.total_score / max(item.willingness, 0.05),
                item.total_score,
                -item.willingness,
                item.courier_id,
            ),
        )
        chosen: list[CandidateBundle] = []
        used: set[str] = set()
        for row in ranked:
            if row.courier_id in used:
                continue
            chosen.append(row)
            used.add(row.courier_id)
            if len(chosen) >= max_rows:
                break
        fail_probability = 1.0
        for row in chosen:
            fail_probability *= 1.0 - max(0.0, min(1.0, row.willingness))
        profiles[bundle] = {
            "expected": _expected_bundle_score(chosen, reject_penalty),
            "success_probability": 1.0 - fail_probability,
            "fail_probability": fail_probability,
        }
    return profiles


def local_search_improve(
    candidates: tuple[CandidateBundle, ...],
    selected: list[CandidateBundle],
    time_limit_seconds: float = 1.0,
) -> list[CandidateBundle]:
    if not candidates or not selected:
        return selected

    start = perf_counter()
    by_task_set: dict[frozenset[str], list[CandidateBundle]] = {}
    for candidate in candidates:
        by_task_set.setdefault(frozenset(candidate.task_id_list), []).append(candidate)
    for rows in by_task_set.values():
        rows.sort(key=lambda item: (item.total_score, -item.willingness, item.task_id_list_str, item.courier_id))

    def selected_key(items: list[CandidateBundle]) -> tuple[int, float, float]:
        return (
            sum(item.task_count for item in items),
            -len(items),
            -sum(item.total_score for item in items),
            sum(item.willingness for item in items),
        )

    def best_cover(
        uncovered_tasks: set[str],
        blocked_couriers: set[str],
        max_items: int,
    ) -> list[CandidateBundle]:
        uncovered_frozen = frozenset(uncovered_tasks)
        pool: list[CandidateBundle] = []
        for task_set, rows in by_task_set.items():
            if not task_set or not task_set.issubset(uncovered_frozen):
                continue
            for candidate in rows[:80]:
                if candidate.courier_id not in blocked_couriers:
                    pool.append(candidate)

        pool_by_task: dict[str, list[CandidateBundle]] = {}
        for candidate in pool:
            for task_id in candidate.task_id_list:
                pool_by_task.setdefault(task_id, []).append(candidate)

        best: list[CandidateBundle] = []

        def dfs(remaining: set[str], used_couriers: set[str], chosen: list[CandidateBundle]) -> None:
            nonlocal best
            if perf_counter() - start > time_limit_seconds:
                return
            if not remaining:
                if not best or selected_key(chosen) > selected_key(best):
                    best = list(chosen)
                return
            if len(chosen) >= max_items:
                return

            pivot = min(remaining, key=lambda task_id: len(pool_by_task.get(task_id, ())))
            for candidate in pool_by_task.get(pivot, ()):
                task_set = set(candidate.task_id_list)
                if not task_set.issubset(remaining):
                    continue
                if candidate.courier_id in used_couriers:
                    continue
                chosen.append(candidate)
                dfs(remaining - task_set, used_couriers | {candidate.courier_id}, chosen)
                chosen.pop()

        dfs(set(uncovered_tasks), set(), [])
        return best

    improved = True
    while improved and perf_counter() - start <= time_limit_seconds:
        improved = False
        selected.sort(key=lambda item: (item.total_score / item.task_count, item.total_score), reverse=True)
        index_groups = [(index,) for index in range(len(selected))]
        index_groups.extend(
            (left, right)
            for left in range(len(selected))
            for right in range(left + 1, len(selected))
        )

        for indexes in index_groups:
            if perf_counter() - start > time_limit_seconds:
                break
            removed = [selected[index] for index in indexes]
            remaining_selected = [item for index, item in enumerate(selected) if index not in indexes]
            uncovered = {task_id for item in removed for task_id in item.task_id_list}
            blocked_couriers = {item.courier_id for item in remaining_selected}
            replacement = best_cover(uncovered, blocked_couriers, max_items=len(uncovered))
            if not replacement:
                continue
            if {task_id for item in replacement for task_id in item.task_id_list} != uncovered:
                continue
            if selected_key(replacement) > selected_key(removed):
                selected = remaining_selected + replacement
                improved = True
                break

    return selected


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
    result = AutoSolverAgent().run(instance)
    metadata = compute_metadata(instance)
    return assign_backup_couriers(instance.candidates, list(result.selected), metadata)


def _expected_bundle_score(rows: list[CandidateBundle], reject_penalty: float) -> float:
    if not rows:
        return 0.0
    ordered = sorted(rows, key=lambda item: (item.total_score, -item.willingness, item.courier_id))
    fail_probability = 1.0
    expected = 0.0
    for row in ordered:
        probability = max(0.0, min(1.0, row.willingness))
        expected += fail_probability * probability * row.total_score
        fail_probability *= 1.0 - probability
    expected += fail_probability * reject_penalty * ordered[0].task_count
    return expected


def expected_submission_score(
    candidates: tuple[CandidateBundle, ...],
    result: list[tuple[str, list[str]]],
    metadata: ProblemMetadata,
) -> float:
    rows = {(candidate.task_id_list_str, candidate.courier_id): candidate for candidate in candidates}
    reject_penalty = _reject_penalty(metadata)
    total = 0.0
    for task_id_list_str, courier_ids in result:
        bundle_rows = [rows[(task_id_list_str, courier_id)] for courier_id in courier_ids]
        total += _expected_bundle_score(bundle_rows, reject_penalty)
    return total


def assign_backup_couriers(
    candidates: tuple[CandidateBundle, ...],
    selected: list[CandidateBundle],
    metadata: ProblemMetadata,
) -> list[tuple[str, list[str]]]:
    if not selected:
        return []

    by_bundle: dict[str, list[CandidateBundle]] = {}
    for candidate in candidates:
        by_bundle.setdefault(candidate.task_id_list_str, []).append(candidate)
    for rows in by_bundle.values():
        rows.sort(key=lambda item: (item.total_score, -item.willingness, item.courier_id))

    reject_penalty = _reject_penalty(metadata)
    bundles = []
    used_couriers: set[str] = set()
    for item in selected:
        used_couriers.add(item.courier_id)
        bundles.append({"task_id_list_str": item.task_id_list_str, "rows": [item]})

    if metadata.courier_count / max(metadata.task_count, 1) < 1.25:
        max_couriers_per_bundle = 2
    else:
        max_couriers_per_bundle = 4
    use_robust_gain = (
        metadata.courier_count / max(metadata.task_count, 1) < 1.25
        or is_low_willingness_case(candidates, metadata)
    )
    while True:
        best_choice = None
        best_gain = 0.0

        for bundle_index, bundle in enumerate(bundles):
            rows = bundle["rows"]
            if len(rows) >= max_couriers_per_bundle:
                continue
            current_score = _expected_bundle_score(rows, reject_penalty)
            current_couriers = {row.courier_id for row in rows}

            for candidate in by_bundle.get(bundle["task_id_list_str"], ())[:160]:
                if candidate.courier_id in used_couriers or candidate.courier_id in current_couriers:
                    continue
                improved_score = _expected_bundle_score(rows + [candidate], reject_penalty)
                if use_robust_gain:
                    current_assigned = sum(row.total_score for row in rows)
                    improved_assigned = current_assigned + candidate.total_score
                    alpha = robust_alpha(candidates, metadata)
                    current_objective = max(current_score, current_assigned * alpha)
                    improved_objective = max(improved_score, improved_assigned * alpha)
                    gain = current_objective - improved_objective
                else:
                    gain = current_score - improved_score
                if gain > best_gain:
                    best_gain = gain
                    best_choice = (bundle_index, candidate)

        if best_choice is None or best_gain <= 1e-9:
            break

        bundle_index, candidate = best_choice
        bundles[bundle_index]["rows"].append(candidate)
        used_couriers.add(candidate.courier_id)

    return [
        (
            bundle["task_id_list_str"],
            [
                row.courier_id
                for row in sorted(
                    bundle["rows"],
                    key=lambda item: (item.total_score, -item.willingness, item.courier_id),
                )
            ],
        )
        for bundle in bundles
    ]
