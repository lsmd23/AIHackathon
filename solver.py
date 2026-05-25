from time import perf_counter


def _parse_input(input_text: str):
    lines = input_text.strip().splitlines()
    start = 1 if lines and lines[0].startswith("task_id_list") else 0
    candidates = []

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
        courier_id = courier_id.strip()
        if not task_ids or not courier_id:
            continue
        candidates.append((task_ids, task_id_list_str.strip(), courier_id, total_score, willingness))

    return candidates


def _metadata(candidates):
    task_ids = {task for item in candidates for task in item[0]}
    courier_ids = {item[2] for item in candidates}
    scores = [item[3] for item in candidates] or [0.0]
    willingness = [item[4] for item in candidates] or [0.0]
    max_bundle_size = max((len(item[0]) for item in candidates), default=0)
    return {
        "candidate_count": len(candidates),
        "task_count": len(task_ids),
        "courier_count": len(courier_ids),
        "mean_score": sum(scores) / len(scores),
        "mean_willingness": sum(willingness) / len(willingness),
        "max_bundle_size": max_bundle_size,
    }


def _result_key(selected):
    covered = {task for item in selected for task in item[0]}
    total_score = sum(item[3] for item in selected)
    total_willingness = sum(item[4] for item in selected)
    return (len(covered), -total_score, total_willingness)


def _greedy_by_key(candidates, key):
    assigned_tasks = set()
    assigned_couriers = set()
    selected = []

    for candidate in sorted(candidates, key=key):
        task_ids, _task_list_str, courier_id, _score, _willingness = candidate
        if courier_id in assigned_couriers:
            continue
        if any(task_id in assigned_tasks for task_id in task_ids):
            continue
        assigned_couriers.add(courier_id)
        assigned_tasks.update(task_ids)
        selected.append(candidate)

    return selected


def _greedy(candidates, meta):
    return _greedy_by_key(
        candidates,
        lambda item: (item[3] / len(item[0]), item[3], -item[4], item[1], item[2]),
    )


def _heuristic_search(candidates, meta):
    if not candidates:
        return []

    min_score_by_task = {}
    for item in candidates:
        task_ids, _task_list_str, _courier_id, score, _willingness = item
        average_score = score / len(task_ids)
        for task_id in task_ids:
            min_score_by_task[task_id] = min(min_score_by_task.get(task_id, average_score), average_score)

    def rarity(item):
        return sum(min_score_by_task[task_id] for task_id in item[0]) / len(item[0])

    key_functions = [
        lambda item: (item[3] / len(item[0]), item[3], -item[4], item[1], item[2]),
        lambda item: (
            item[3] / len(item[0]) - meta["mean_willingness"] * item[4],
            item[3],
            -item[4],
            item[1],
        ),
        lambda item: (
            item[3] / len(item[0]) - rarity(item),
            -len(item[0]),
            -item[4],
            item[3],
        ),
        lambda item: (-len(item[0]), item[3] / len(item[0]), -item[4], item[3]),
    ]

    best = []
    for key in key_functions:
        selected = _greedy_by_key(candidates, key)
        if _result_key(selected) > _result_key(best):
            best = selected
    return best


def _branch_bound(candidates, meta, time_limit_seconds=1.5, max_candidates_per_task=24):
    if not candidates:
        return []

    task_ids = sorted({task for item in candidates for task in item[0]})
    courier_ids = sorted({item[2] for item in candidates})
    task_to_bit = {task_id: 1 << index for index, task_id in enumerate(task_ids)}
    courier_to_bit = {courier_id: 1 << index for index, courier_id in enumerate(courier_ids)}
    by_task = {task_id: [] for task_id in task_ids}

    for candidate in candidates:
        task_mask = 0
        for task_id in candidate[0]:
            task_mask |= task_to_bit[task_id]
        courier_mask = courier_to_bit[candidate[2]]
        row = (task_mask, courier_mask, candidate)
        for task_id in candidate[0]:
            by_task[task_id].append(row)

    for task_id, rows in by_task.items():
        rows.sort(key=lambda row: (row[2][3] / len(row[2][0]), row[2][3], -row[2][4]))
        del rows[max_candidates_per_task:]

    best = _heuristic_search(candidates, meta)
    start = perf_counter()
    full_mask = (1 << len(task_ids)) - 1

    def search(task_mask, courier_mask, selected):
        nonlocal best
        if perf_counter() - start > time_limit_seconds:
            return
        if task_mask.bit_count() + (full_mask ^ task_mask).bit_count() < _result_key(best)[0]:
            return
        if task_mask == full_mask:
            if _result_key(selected) > _result_key(best):
                best = list(selected)
            return

        uncovered = [task_id for task_id in task_ids if not (task_mask & task_to_bit[task_id])]
        pivot = min(uncovered, key=lambda task_id: len(by_task[task_id]))

        for candidate_task_mask, candidate_courier_mask, candidate in by_task[pivot]:
            if task_mask & candidate_task_mask:
                continue
            if courier_mask & candidate_courier_mask:
                continue
            selected.append(candidate)
            search(task_mask | candidate_task_mask, courier_mask | candidate_courier_mask, selected)
            selected.pop()

        search(task_mask | task_to_bit[pivot], courier_mask, selected)
        if _result_key(selected) > _result_key(best):
            best = list(selected)

    search(0, 0, [])
    return best


def _llm_direct_reasoning(candidates, meta):
    # Online evaluation cannot call an external LLM, so this deterministic
    # rule-based approximation occupies the same algorithm-library slot.
    return _heuristic_search(candidates, meta)


def _agent_decide(meta):
    if meta["candidate_count"] == 0:
        return "greedy"
    if meta["max_bundle_size"] >= 2:
        return "heuristic_search"
    return "greedy"


def _agent_run(candidates):
    meta = _metadata(candidates)
    algorithm = _agent_decide(meta)
    if algorithm == "branch_bound":
        return _branch_bound(candidates, meta)
    if algorithm == "heuristic_search":
        return _heuristic_search(candidates, meta)
    if algorithm == "llm_direct_reasoning":
        return _llm_direct_reasoning(candidates, meta)
    return _greedy(candidates, meta)


def solve(input_text: str) -> list:
    candidates = _parse_input(input_text)
    selected = _agent_run(candidates)
    return [(item[1], [item[2]]) for item in selected]
