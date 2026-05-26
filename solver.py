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
        "max_score": max(scores),
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

    bundle_profiles = _bundle_profiles(candidates, meta)
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
            bundle_profiles[item[1]]["expected"] / len(item[0]),
            bundle_profiles[item[1]]["fail_probability"],
            item[3],
            -item[4],
            item[1],
        ),
        lambda item: (
            bundle_profiles[item[1]]["expected"] / len(item[0]) - meta["mean_willingness"] * bundle_profiles[item[1]]["success_probability"],
            item[3],
            -item[4],
            item[1],
        ),
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

    def evaluated_result_key(selected):
        covered = {task for item in selected for task in item[0]}
        expected = _expected_submission_score(candidates, _assign_backup_couriers(candidates, selected, meta), meta)
        primary_score = sum(item[3] for item in selected)
        return (len(covered), -expected, -primary_score)

    best = []
    for key in key_functions:
        selected = _greedy_by_key(candidates, key)
        if evaluated_result_key(selected) > evaluated_result_key(best):
            best = selected
    for selected in _pair_first_candidates(candidates, bundle_profiles):
        if evaluated_result_key(selected) > evaluated_result_key(best):
            best = selected
    for selected in _minimum_group_candidates(candidates, bundle_profiles):
        if evaluated_result_key(selected) > evaluated_result_key(best):
            best = selected
    if meta["mean_willingness"] < 0.18:
        for selected in _low_willingness_candidates(candidates, bundle_profiles):
            if evaluated_result_key(selected) > evaluated_result_key(best):
                best = selected
    return _local_search(candidates, best)


def _pair_first_candidates(candidates, bundle_profiles):
    pair_rows = [item for item in candidates if len(item[0]) == 2]
    single_rows = [item for item in candidates if len(item[0]) == 1]
    single_best = {}
    for item in sorted(single_rows, key=lambda row: (row[3] / max(row[4], 0.05), row[3], -row[4])):
        single_best.setdefault(item[0][0], item)

    sorters = [
        lambda item: (
            bundle_profiles[item[1]]["expected"] / 2,
            bundle_profiles[item[1]]["fail_probability"],
            item[3],
            -item[4],
        ),
        lambda item: (
            item[3] / 2,
            -bundle_profiles[item[1]]["success_probability"],
            item[3],
            -item[4],
        ),
        lambda item: (
            -bundle_profiles[item[1]]["success_probability"],
            bundle_profiles[item[1]]["expected"] / 2,
            item[3],
        ),
    ]

    results = []
    for sorter in sorters:
        used_tasks = set()
        used_couriers = set()
        selected = []
        for item in sorted(pair_rows, key=sorter):
            if any(task in used_tasks for task in item[0]):
                continue
            if item[2] in used_couriers:
                continue
            selected.append(item)
            used_tasks.update(item[0])
            used_couriers.add(item[2])
        for task_id, item in single_best.items():
            if task_id in used_tasks or item[2] in used_couriers:
                continue
            selected.append(item)
            used_tasks.add(task_id)
            used_couriers.add(item[2])
        results.append(selected)
    return results


def _low_willingness_candidates(candidates, bundle_profiles):
    pair_rows = [item for item in candidates if len(item[0]) == 2]
    single_rows = [item for item in candidates if len(item[0]) == 1]
    single_best = {}
    for item in sorted(single_rows, key=lambda row: (row[3] / max(row[4], 0.03), row[3], -row[4])):
        single_best.setdefault(item[0][0], item)

    def pair_value(item):
        profile = bundle_profiles[item[1]]
        return (
            profile["fail_probability"],
            profile["expected"] / 2,
            item[3] / max(item[4], 0.03),
            item[3],
        )

    results = []
    for reverse_success in (False, True):
        used_tasks = set()
        used_couriers = set()
        selected = []
        rows = sorted(
            pair_rows,
            key=(
                (lambda item: (-bundle_profiles[item[1]]["success_probability"], item[3] / max(item[4], 0.03), item[3]))
                if reverse_success
                else pair_value
            ),
        )
        for item in rows:
            if any(task in used_tasks for task in item[0]):
                continue
            if item[2] in used_couriers:
                continue
            selected.append(item)
            used_tasks.update(item[0])
            used_couriers.add(item[2])
        for task_id, item in single_best.items():
            if task_id in used_tasks or item[2] in used_couriers:
                continue
            selected.append(item)
            used_tasks.add(task_id)
            used_couriers.add(item[2])
        results.append(selected)
    return results


def _minimum_group_candidates(candidates, bundle_profiles):
    pair_rows = [item for item in candidates if len(item[0]) == 2]
    single_rows = [item for item in candidates if len(item[0]) == 1]
    single_best = {}
    for item in sorted(single_rows, key=lambda row: (row[3], -row[4], row[2])):
        single_best.setdefault(item[0][0], item)

    sorters = [
        lambda item: (-len(item[0]), bundle_profiles[item[1]]["expected"] / 2, item[3]),
        lambda item: (-len(item[0]), item[3] / 2, -item[4], item[3]),
        lambda item: (-len(item[0]), -bundle_profiles[item[1]]["success_probability"], bundle_profiles[item[1]]["expected"]),
    ]
    results = []
    for sorter in sorters:
        used_tasks = set()
        used_couriers = set()
        selected = []
        for item in sorted(pair_rows, key=sorter):
            if any(task in used_tasks for task in item[0]) or item[2] in used_couriers:
                continue
            selected.append(item)
            used_tasks.update(item[0])
            used_couriers.add(item[2])
        for task_id, item in single_best.items():
            if task_id in used_tasks or item[2] in used_couriers:
                continue
            selected.append(item)
            used_tasks.add(task_id)
            used_couriers.add(item[2])
        results.append(selected)
    return results


def _bundle_profiles(candidates, meta, max_rows=5):
    by_bundle = {}
    for candidate in candidates:
        by_bundle.setdefault(candidate[1], []).append(candidate)

    reject_penalty = _reject_penalty(meta)
    profiles = {}
    for bundle, rows in by_bundle.items():
        ranked = sorted(rows, key=lambda item: (item[3] / max(item[4], 0.05), item[3], -item[4], item[2]))
        chosen = []
        used = set()
        for row in ranked:
            if row[2] in used:
                continue
            chosen.append(row)
            used.add(row[2])
            if len(chosen) >= max_rows:
                break
        fail_probability = 1.0
        for row in chosen:
            fail_probability *= 1.0 - max(0.0, min(1.0, row[4]))
        profiles[bundle] = {
            "expected": _expected_bundle_score(chosen, reject_penalty),
            "success_probability": 1.0 - fail_probability,
            "fail_probability": fail_probability,
        }
    return profiles


def _local_search(candidates, selected, time_limit_seconds=1.0):
    if not candidates or not selected:
        return selected

    start = perf_counter()
    by_task_set = {}
    for candidate in candidates:
        task_set = frozenset(candidate[0])
        by_task_set.setdefault(task_set, []).append(candidate)
    for rows in by_task_set.values():
        rows.sort(key=lambda item: (item[3], -item[4], item[1], item[2]))

    selected = list(selected)

    def selected_key(items):
        return (sum(len(item[0]) for item in items), -sum(item[3] for item in items), sum(item[4] for item in items))

    def best_cover(uncovered_tasks, blocked_couriers, max_items):
        uncovered_tasks = frozenset(uncovered_tasks)
        pool = []
        for task_set, rows in by_task_set.items():
            if not task_set or not task_set.issubset(uncovered_tasks):
                continue
            for candidate in rows[:80]:
                if candidate[2] not in blocked_couriers:
                    pool.append(candidate)

        pool_by_task = {}
        for candidate in pool:
            for task_id in candidate[0]:
                pool_by_task.setdefault(task_id, []).append(candidate)

        best = []

        def dfs(remaining, used_couriers, chosen):
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
                task_set = set(candidate[0])
                if not task_set.issubset(remaining):
                    continue
                if candidate[2] in used_couriers:
                    continue
                chosen.append(candidate)
                dfs(remaining - task_set, used_couriers | {candidate[2]}, chosen)
                chosen.pop()

        dfs(set(uncovered_tasks), set(), [])
        return best

    improved = True
    while improved and perf_counter() - start <= time_limit_seconds:
        improved = False
        selected.sort(key=lambda item: (item[3] / len(item[0]), item[3]), reverse=True)
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
            uncovered = {task_id for item in removed for task_id in item[0]}
            blocked_couriers = {item[2] for item in remaining_selected}
            replacement = best_cover(uncovered, blocked_couriers, max_items=len(uncovered))
            if not replacement:
                continue
            if {task_id for item in replacement for task_id in item[0]} != uncovered:
                continue
            if selected_key(replacement) > selected_key(removed):
                selected = remaining_selected + replacement
                improved = True
                break

    return selected


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
        selected = _branch_bound(candidates, meta)
        return _assign_backup_couriers(candidates, selected, meta)
    if algorithm == "heuristic_search":
        selected = _heuristic_search(candidates, meta)
        return _assign_backup_couriers(candidates, selected, meta)
    if algorithm == "llm_direct_reasoning":
        selected = _llm_direct_reasoning(candidates, meta)
        return _assign_backup_couriers(candidates, selected, meta)
    selected = _greedy(candidates, meta)
    return _assign_backup_couriers(candidates, selected, meta)


def _expected_bundle_score(rows, reject_penalty):
    ordered = sorted(rows, key=lambda item: (item[3], -item[4], item[2]))
    fail_probability = 1.0
    expected = 0.0
    for row in ordered:
        probability = max(0.0, min(1.0, row[4]))
        expected += fail_probability * probability * row[3]
        fail_probability *= 1.0 - probability
    expected += fail_probability * reject_penalty
    return expected


def _expected_submission_score(candidates, result, meta):
    rows = {(candidate[1], candidate[2]): candidate for candidate in candidates}
    reject_penalty = _reject_penalty(meta)
    total = 0.0
    for task_id_list_str, courier_ids in result:
        bundle_rows = [rows[(task_id_list_str, courier_id)] for courier_id in courier_ids]
        total += _expected_bundle_score(bundle_rows, reject_penalty)
    return total


def _reject_penalty(meta):
    return max(100.0, meta.get("max_score", 100.0) * 3.0)


def _assign_backup_couriers(candidates, selected, meta):
    if not selected:
        return []

    by_bundle = {}
    for candidate in candidates:
        by_bundle.setdefault(candidate[1], []).append(candidate)
    for rows in by_bundle.values():
        rows.sort(key=lambda item: (item[3], -item[4], item[2]))

    reject_penalty = _reject_penalty(meta)
    bundles = []
    used_couriers = set()
    for item in selected:
        rows = [item]
        used_couriers.add(item[2])
        bundles.append({"task_id_list_str": item[1], "rows": rows})

    max_couriers_per_bundle = 5 if meta.get("mean_willingness", 1.0) < 0.18 else 4
    while True:
        best_choice = None
        best_gain = 0.0

        for bundle_index, bundle in enumerate(bundles):
            if len(bundle["rows"]) >= max_couriers_per_bundle:
                continue
            current_rows = bundle["rows"]
            current_score = _expected_bundle_score(current_rows, reject_penalty)
            current_couriers = {row[2] for row in current_rows}

            for candidate in by_bundle.get(bundle["task_id_list_str"], ())[:160]:
                if candidate[2] in used_couriers or candidate[2] in current_couriers:
                    continue
                improved_score = _expected_bundle_score(current_rows + [candidate], reject_penalty)
                gain = current_score - improved_score
                if gain > best_gain:
                    best_gain = gain
                    best_choice = (bundle_index, candidate)

        if best_choice is None or best_gain <= 1e-9:
            break

        bundle_index, candidate = best_choice
        bundles[bundle_index]["rows"].append(candidate)
        used_couriers.add(candidate[2])

    return [
        (bundle["task_id_list_str"], [row[2] for row in sorted(bundle["rows"], key=lambda item: (item[3], -item[4], item[2]))])
        for bundle in bundles
    ]


def solve(input_text: str) -> list:
    candidates = _parse_input(input_text)
    return _agent_run(candidates)
