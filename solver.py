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
    pressure = _willingness_pressure(candidates)
    return {
        "candidate_count": len(candidates),
        "task_count": len(task_ids),
        "courier_count": len(courier_ids),
        "max_score": max(scores),
        "mean_score": sum(scores) / len(scores),
        "mean_willingness": sum(willingness) / len(willingness),
        "max_bundle_size": max_bundle_size,
        "top5_willingness": pressure["top5_mean"],
        "cheap5_willingness": pressure["cheap5_mean"],
        "score_spread": _score_spread(scores),
    }


def _score_spread(scores):
    if not scores:
        return 0.0
    ordered = sorted(scores)
    q10 = ordered[int((len(ordered) - 1) * 0.1)]
    q90 = ordered[int((len(ordered) - 1) * 0.9)]
    mean = sum(scores) / len(scores)
    return (q90 - q10) / max(mean, 1e-9)


def _willingness_pressure(candidates):
    by_bundle = {}
    for item in candidates:
        by_bundle.setdefault(item[1], []).append(item)
    if not by_bundle:
        return {"top5_mean": 1.0, "cheap5_mean": 1.0}

    top_values = []
    cheap_values = []
    for rows in by_bundle.values():
        top = sorted((row[4] for row in rows), reverse=True)[:5]
        cheap = sorted(rows, key=lambda row: (row[3], -row[4]))[:5]
        top_values.append(sum(top) / len(top))
        cheap_values.append(sum(row[4] for row in cheap) / len(cheap))
    return {
        "top5_mean": sum(top_values) / len(top_values),
        "cheap5_mean": sum(cheap_values) / len(cheap_values),
    }


def _is_low_willingness_case(candidates, meta):
    return (
        meta.get("mean_willingness", 1.0) < 0.18
        or meta.get("top5_willingness", 1.0) < 0.45
        or meta.get("cheap5_willingness", 1.0) < 0.22
    )


def _robust_alpha(candidates, meta):
    if _is_low_willingness_case(candidates, meta):
        return 0.65
    if meta.get("courier_count", 0) / max(meta.get("task_count", 1), 1) < 1.25:
        return 1.2
    return 0.85


def _policy_alpha(candidates, meta, policy=None):
    if policy and "alpha" in policy:
        return policy["alpha"]
    return _robust_alpha(candidates, meta)


def _policy_max_couriers(meta, policy=None):
    if policy and "max_couriers" in policy:
        return policy["max_couriers"]
    if meta.get("courier_count", 0) / max(meta.get("task_count", 1), 1) < 1.25:
        return 2
    return 4


def _policy_score(candidates, result, meta, policy=None):
    covered = {task for task_id_list_str, _couriers in result for task in task_id_list_str.split(",") if task}
    expected = _expected_submission_score(candidates, result, meta)
    row_map = {(candidate[1], candidate[2]): candidate for candidate in candidates}
    assigned_score = sum(row_map[(bundle, courier_id)][3] for bundle, courier_ids in result for courier_id in courier_ids)
    primary_score = sum(row_map[(bundle, courier_ids[0])][3] for bundle, courier_ids in result if courier_ids)
    alpha = _policy_alpha(candidates, meta, policy)
    mode = policy.get("mode", "adaptive") if policy else "adaptive"
    if mode == "expected":
        objective = expected
    elif mode == "assigned":
        objective = max(expected, assigned_score * alpha)
    elif mode == "balanced":
        objective = expected + assigned_score * alpha * 0.15
    else:
        is_scarce = meta.get("courier_count", 0) / max(meta.get("task_count", 1), 1) < 1.25
        is_low = _is_low_willingness_case(candidates, meta)
        objective = max(expected, assigned_score * alpha) if (is_scarce or is_low) else expected
    return (len(covered), -objective, -primary_score)


def _raw_objective_from_score(expected, assigned_score, candidates, meta, policy=None):
    alpha = _policy_alpha(candidates, meta, policy)
    mode = policy.get("mode", "adaptive") if policy else "adaptive"
    if mode == "expected":
        return expected
    if mode == "assigned":
        return max(expected, assigned_score * alpha)
    if mode == "balanced":
        return expected + assigned_score * alpha * 0.15
    is_scarce = meta.get("courier_count", 0) / max(meta.get("task_count", 1), 1) < 1.25
    is_low = _is_low_willingness_case(candidates, meta)
    return max(expected, assigned_score * alpha) if (is_scarce or is_low) else expected


def _learning_policies(candidates, meta):
    is_scarce = meta.get("courier_count", 0) / max(meta.get("task_count", 1), 1) < 1.25
    is_low = _is_low_willingness_case(candidates, meta)
    if is_scarce:
        return [
            {"alpha": 0.85, "max_couriers": 2, "mode": "adaptive"},
            {"alpha": 1.0, "max_couriers": 2, "mode": "adaptive"},
            {"alpha": 1.2, "max_couriers": 2, "mode": "adaptive"},
            {"alpha": 1.5, "max_couriers": 2, "mode": "assigned"},
        ]
    if is_low:
        return [
            {"alpha": 0.65, "max_couriers": 4, "mode": "adaptive"},
            {"alpha": 0.55, "max_couriers": 4, "mode": "balanced"},
            {"alpha": 0.85, "max_couriers": 4, "mode": "adaptive"},
        ]
    return [
        {"alpha": 0.85, "max_couriers": 4, "mode": "expected"},
    ]


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
    low_case = _is_low_willingness_case(candidates, meta)
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

    if low_case:
        experiments = []
        for selected in _low_willingness_candidates(candidates, bundle_profiles):
            experiments.append(selected)
        for selected in _hybrid_split_candidates(candidates, meta):
            experiments.append(selected)
        for selected in _budgeted_pairing_candidates(candidates, meta):
            experiments.append(selected)
    else:
        experiments = []
        for key in key_functions:
            selected = _greedy_by_key(candidates, key)
            experiments.append(selected)
        for selected in _pair_first_candidates(candidates, bundle_profiles):
            experiments.append(selected)
        for selected in _minimum_group_candidates(candidates, bundle_profiles):
            experiments.append(selected)
        for selected in _budgeted_pairing_candidates(candidates, meta):
            experiments.append(selected)
        if meta.get("task_count", 0) <= 30 and meta.get("score_spread", 0.0) > 0.84:
            for selected in _component_dp_candidates(candidates, meta):
                experiments.append(selected)
        if meta.get("score_spread", 0.0) > 1.0:
            for selected in _low_beam_candidates(candidates, meta):
                experiments.append(selected)

    deduped = []
    seen_experiments = set()
    for selected in experiments:
        signature = tuple(sorted(item[1] for item in selected))
        if signature in seen_experiments:
            continue
        seen_experiments.add(signature)
        deduped.append(selected)
    experiments = deduped

    is_scarce_case = meta.get("courier_count", 0) / max(meta.get("task_count", 1), 1) < 1.25
    assigners = (_assign_backup_couriers,) if low_case else (_assign_backup_couriers, _assign_couriers_global)
    anchor_policy = {"alpha": _robust_alpha(candidates, meta), "max_couriers": _policy_max_couriers(meta), "mode": "adaptive"}
    anchor_best = []
    anchor_key = (-1, float("-inf"), float("-inf"))
    for selected in experiments:
        for assigner in assigners:
            submission = assigner(candidates, selected, meta, anchor_policy)
            key = _policy_score(candidates, submission, meta, anchor_policy)
            if key > anchor_key:
                anchor_key = key
                anchor_best = selected
                anchor_policy["_assigner"] = "global" if assigner is _assign_couriers_global else "seeded"

    if not low_case and not is_scarce_case:
        meta["_learned_policy"] = anchor_policy
        local_budget = 0.25 if meta["candidate_count"] > 30000 else 0.7
        improved = _local_search(candidates, anchor_best, time_limit_seconds=local_budget)
        improved_submission = _assign_backup_couriers(candidates, improved, meta, anchor_policy)
        if _policy_score(candidates, improved_submission, meta, anchor_policy) > anchor_key:
            return improved
        return anchor_best

    best = anchor_best
    best_policy = anchor_policy
    best_key = anchor_key
    history = []
    policies = _learning_policies(candidates, meta)
    scout_policy = policies[0]
    scout = []
    for selected in experiments:
        submission = _assign_backup_couriers(candidates, selected, meta, scout_policy)
        key = _policy_score(candidates, submission, meta, scout_policy)
        scout.append((key, selected))
        if key > best_key:
            best_key = key
            best = selected
            best_policy = scout_policy

    candidate_limit = 4 if low_case else 5
    for _scout_key, selected in sorted(scout, key=lambda item: item[0], reverse=True)[:candidate_limit]:
        for policy in policies:
            for assigner in assigners:
                trial_policy = dict(policy)
                trial_policy["_assigner"] = "global" if assigner is _assign_couriers_global else "seeded"
                submission = assigner(candidates, selected, meta, trial_policy)
                key = _policy_score(candidates, submission, meta, trial_policy)
                history.append((key, selected, trial_policy))
                if key > best_key:
                    best_key = key
                    best = selected
                    best_policy = trial_policy

    # Learning step: explore around the winning policy instead of keeping a fixed grid.
    if best_policy is not None and meta.get("courier_count", 0) / max(meta.get("task_count", 1), 1) < 1.25:
        neighbor_policies = []
        for delta in (-0.15, -0.075, 0.075, 0.15):
            neighbor = dict(best_policy)
            neighbor["alpha"] = max(0.25, min(2.0, best_policy.get("alpha", 0.85) + delta))
            neighbor_policies.append(neighbor)
        for max_couriers in (2, 3, 4):
            neighbor = dict(best_policy)
            neighbor["max_couriers"] = max_couriers
            neighbor_policies.append(neighbor)
        for _key, selected, _policy in sorted(history, key=lambda item: item[0], reverse=True)[:3]:
            for policy in neighbor_policies:
                assigner = _assign_couriers_global if policy.get("_assigner") == "global" else _assign_backup_couriers
                submission = assigner(candidates, selected, meta, policy)
                candidate_key = _policy_score(candidates, submission, meta, policy)
                if candidate_key > best_key:
                    best_key = candidate_key
                    best = selected
                    best_policy = policy

    # Keep the proven anchor unless learning finds a clearly better proxy.
    if best_key[0] == anchor_key[0] and best_key[1] < anchor_key[1] * 0.88:
        best = anchor_best
        best_policy = anchor_policy
        best_key = anchor_key

    meta["_learned_policy"] = best_policy
    local_budget = 0.25 if meta["candidate_count"] > 30000 else 0.7
    improved = _local_search(candidates, best, time_limit_seconds=local_budget)
    improved_submission = _assign_backup_couriers(candidates, improved, meta, best_policy)
    if _policy_score(candidates, improved_submission, meta, best_policy) > best_key:
        return improved
    return best


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


def _hybrid_split_candidates(candidates, meta):
    single_rows = [item for item in candidates if len(item[0]) == 1]
    pair_rows = [item for item in candidates if len(item[0]) == 2]
    if not single_rows or not pair_rows:
        return []

    by_bundle = {}
    for item in candidates:
        by_bundle.setdefault(item[1], []).append(item)
    for rows in by_bundle.values():
        rows.sort(key=lambda item: (item[3] / max(item[4], 0.03), item[3], -item[4], item[2]))

    reject_penalty = _reject_penalty(meta)

    def preview_cost(rows, limit=4, alpha=0.85):
        preview = rows[:limit]
        expected = _expected_bundle_score(preview, reject_penalty)
        assigned = sum(row[3] for row in preview)
        return max(expected, assigned * alpha)

    single_info = {}
    for item in single_rows:
        task_id = item[0][0]
        if task_id in single_info:
            continue
        rows = by_bundle[item[1]]
        single_info[task_id] = {
            "rows": rows,
            "cost": preview_cost(rows),
        }

    pair_infos = []
    seen = set()
    for item in pair_rows:
        tasks = frozenset(item[0])
        if tasks in seen:
            continue
        seen.add(tasks)
        left, right = item[0]
        if left not in single_info or right not in single_info:
            continue
        rows = by_bundle[item[1]]
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

    def build(min_saving):
        used_tasks = set()
        used_couriers = set()
        selected = []
        for info in sorted(pair_infos, key=lambda row: (-row["saving"], row["cost"])):
            if info["saving"] < min_saving:
                continue
            if info["tasks"] & used_tasks:
                continue
            row = next((candidate for candidate in info["rows"][:80] if candidate[2] not in used_couriers), None)
            if row is None:
                continue
            selected.append(row)
            used_tasks.update(info["tasks"])
            used_couriers.add(row[2])

        for task_id, info in sorted(single_info.items(), key=lambda item: item[1]["cost"]):
            if task_id in used_tasks:
                continue
            row = next((candidate for candidate in info["rows"][:80] if candidate[2] not in used_couriers), None)
            if row is None:
                continue
            selected.append(row)
            used_tasks.add(task_id)
            used_couriers.add(row[2])
        return selected

    thresholds = (-50.0, 0.0, 25.0, 50.0, 100.0)
    return [build(threshold) for threshold in thresholds]


def _low_beam_candidates(candidates, meta):
    single_rows = [item for item in candidates if len(item[0]) == 1]
    pair_rows = [item for item in candidates if len(item[0]) == 2]
    if not single_rows or not pair_rows:
        return []

    by_bundle = {}
    for item in candidates:
        by_bundle.setdefault(item[1], []).append(item)
    for rows in by_bundle.values():
        rows.sort(key=lambda item: (item[3] / max(item[4], 0.03), item[3], -item[4], item[2]))

    reject_penalty = _reject_penalty(meta)
    alpha = _robust_alpha(candidates, meta)
    all_tasks = sorted({task for item in candidates for task in item[0]})
    full_task_count = len(all_tasks)
    choices_by_task = {task: [] for task in all_tasks}
    seen_groups = set()

    for item in single_rows + pair_rows:
        task_set = frozenset(item[0])
        if task_set in seen_groups:
            continue
        seen_groups.add(task_set)
        rows = by_bundle[item[1]]
        preview = rows[:4]
        expected = _expected_bundle_score(preview, reject_penalty)
        assigned = sum(row[3] for row in preview)
        cost = max(expected, assigned * alpha)
        choice = (task_set, rows, cost)
        for task in task_set:
            choices_by_task[task].append(choice)

    for task, choices in choices_by_task.items():
        choices.sort(key=lambda choice: (choice[2] / len(choice[0]), choice[2]))

    results = []
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
                    row = next((candidate for candidate in rows[:50] if candidate[2] not in used_couriers), None)
                    if row is None:
                        continue
                    bonus = pair_bonus if len(task_set) == 2 else 0.0
                    next_states.append(
                        (
                            cost + group_cost - bonus,
                            covered | task_set,
                            used_couriers | {row[2]},
                            selected + [row],
                        )
                    )
            states = sorted(next_states, key=lambda state: (state[0], len(state[3])))[:260]
            if states and all(len(state[1]) == full_task_count for state in states):
                break
        results.extend(state[3] for state in states[:2] if len(state[1]) == full_task_count)
    return results


def _component_dp_candidates(candidates, meta):
    single_rows = [item for item in candidates if len(item[0]) == 1]
    pair_rows = [item for item in candidates if len(item[0]) == 2]
    if not single_rows or not pair_rows:
        return []

    by_bundle = {}
    for item in candidates:
        by_bundle.setdefault(item[1], []).append(item)
    for rows in by_bundle.values():
        rows.sort(key=lambda item: (item[3] / max(item[4], 0.03), item[3], -item[4], item[2]))

    reject_penalty = _reject_penalty(meta)
    alpha = _robust_alpha(candidates, meta)
    preview_limit = 2 if meta.get("courier_count", 0) / max(meta.get("task_count", 1), 1) < 1.25 else 4

    def group_cost(rows):
        preview = rows[:preview_limit]
        expected = _expected_bundle_score(preview, reject_penalty)
        assigned = sum(row[3] for row in preview)
        return _raw_objective_from_score(expected, assigned, candidates, meta, {"alpha": alpha, "mode": "adaptive"})

    group_info = {}
    for item in single_rows + pair_rows:
        task_set = frozenset(item[0])
        if task_set in group_info:
            continue
        rows = by_bundle[item[1]]
        group_info[task_set] = {"rows": rows, "cost": group_cost(rows)}

    tasks = sorted({task for item in candidates for task in item[0]})
    single_cost = {task: group_info[frozenset((task,))]["cost"] for task in tasks if frozenset((task,)) in group_info}
    edges = []
    for task_set, info in group_info.items():
        if len(task_set) != 2:
            continue
        left, right = tuple(task_set)
        if left in single_cost and right in single_cost:
            edges.append((single_cost[left] + single_cost[right] - info["cost"], left, right))

    results = []
    for min_saving in (-100.0, -25.0, 0.0, 50.0):
        adjacency = {task: set() for task in tasks}
        for saving, left, right in edges:
            if saving >= min_saving:
                adjacency[left].add(right)
                adjacency[right].add(left)

        components = []
        remaining = set(tasks)
        while remaining:
            start = remaining.pop()
            stack = [start]
            component = [start]
            while stack:
                task = stack.pop()
                for neighbor in adjacency.get(task, ()):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
                        component.append(neighbor)
            component.sort()
            for index in range(0, len(component), 14):
                components.append(component[index : index + 14])

        selected_sets = []
        for component in components:
            full_mask = (1 << len(component)) - 1
            dp = {0: (0.0, [])}
            for mask in range(full_mask + 1):
                if mask not in dp or mask == full_mask:
                    continue
                cost, groups = dp[mask]
                pivot = next(index for index in range(len(component)) if not (mask & (1 << index)))
                task = component[pivot]
                single_set = frozenset((task,))
                if single_set in group_info:
                    next_mask = mask | (1 << pivot)
                    next_cost = cost + group_info[single_set]["cost"]
                    if next_mask not in dp or next_cost < dp[next_mask][0]:
                        dp[next_mask] = (next_cost, groups + [single_set])
                for other in range(pivot + 1, len(component)):
                    if mask & (1 << other):
                        continue
                    pair_set = frozenset((task, component[other]))
                    if pair_set not in group_info:
                        continue
                    next_mask = mask | (1 << pivot) | (1 << other)
                    next_cost = cost + group_info[pair_set]["cost"]
                    if next_mask not in dp or next_cost < dp[next_mask][0]:
                        dp[next_mask] = (next_cost, groups + [pair_set])
            selected_sets.extend(dp.get(full_mask, (0.0, []))[1])

        used_couriers = set()
        selected = []
        for task_set in selected_sets:
            row = next((candidate for candidate in group_info[task_set]["rows"][:80] if candidate[2] not in used_couriers), None)
            if row is None:
                continue
            selected.append(row)
            used_couriers.add(row[2])
        if selected:
            results.append(selected)
    return results


def _budgeted_pairing_candidates(candidates, meta):
    pair_rows = [item for item in candidates if len(item[0]) == 2]
    single_rows = [item for item in candidates if len(item[0]) == 1]
    if not pair_rows:
        return []

    by_bundle = {}
    for item in candidates:
        by_bundle.setdefault(item[1], []).append(item)
    for rows in by_bundle.values():
        rows.sort(key=lambda item: (item[3] / max(item[4], 0.03), item[3], -item[4], item[2]))

    reject_penalty = _reject_penalty(meta)
    low_case = _is_low_willingness_case(candidates, meta)
    budget = 4 if low_case else 2
    if meta.get("courier_count", 0) / max(meta.get("task_count", 1), 1) < 1.25:
        budget = 2

    pair_infos = []
    seen_pairs = set()
    for item in pair_rows:
        task_pair = frozenset(item[0])
        if task_pair in seen_pairs:
            continue
        seen_pairs.add(task_pair)
        rows = by_bundle[item[1]]
        preview = rows[:budget]
        fail_probability = 1.0
        for row in preview:
            fail_probability *= 1.0 - max(0.0, min(1.0, row[4]))
        expected = _expected_bundle_score(preview, reject_penalty)
        primary = rows[0]
        pair_infos.append(
            {
                "tasks": task_pair,
                "bundle": item[1],
                "rows": rows,
                "primary": primary,
                "expected": expected,
                "fail": fail_probability,
                "success": 1.0 - fail_probability,
            }
        )

    single_best = {}
    for item in sorted(single_rows, key=lambda row: (row[3] / max(row[4], 0.03), row[3], -row[4], row[2])):
        single_best.setdefault(item[0][0], item)

    def build(sorter):
        used_tasks = set()
        used_couriers = set()
        selected = []
        for info in sorted(pair_infos, key=sorter):
            if info["tasks"] & used_tasks:
                continue
            chosen = None
            for row in info["rows"][:80]:
                if row[2] not in used_couriers:
                    chosen = row
                    break
            if chosen is None:
                continue
            selected.append(chosen)
            used_tasks.update(info["tasks"])
            used_couriers.add(chosen[2])

        for task_id, item in single_best.items():
            if task_id in used_tasks:
                continue
            chosen = item
            if chosen[2] in used_couriers:
                for row in by_bundle.get(chosen[1], ())[:80]:
                    if row[2] not in used_couriers:
                        chosen = row
                        break
                else:
                    continue
            selected.append(chosen)
            used_tasks.add(task_id)
            used_couriers.add(chosen[2])
        return selected

    sorters = [
        lambda info: (info["expected"] / 2, info["fail"], info["primary"][3], -info["primary"][4]),
        lambda info: (info["fail"], info["expected"] / 2, info["primary"][3] / max(info["primary"][4], 0.03)),
        lambda info: (-info["success"], info["expected"] / 2, info["primary"][3]),
        lambda info: (info["primary"][3] / max(info["primary"][4], 0.03), info["expected"] / 2),
        lambda info: (info["primary"][3], -info["primary"][4], info["expected"] / 2),
    ]
    return [build(sorter) for sorter in sorters]


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
        return (
            sum(len(item[0]) for item in items),
            -len(items),
            -sum(item[3] for item in items),
            sum(item[4] for item in items),
        )

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
        policy = meta.get("_learned_policy")
        return _finish_assignment(candidates, selected, meta, policy)
    if algorithm == "heuristic_search":
        selected = _heuristic_search(candidates, meta)
        policy = meta.get("_learned_policy")
        return _finish_assignment(candidates, selected, meta, policy)
    if algorithm == "llm_direct_reasoning":
        selected = _llm_direct_reasoning(candidates, meta)
        policy = meta.get("_learned_policy")
        return _finish_assignment(candidates, selected, meta, policy)
    selected = _greedy(candidates, meta)
    policy = meta.get("_learned_policy")
    return _finish_assignment(candidates, selected, meta, policy)


def _expected_bundle_score(rows, reject_penalty):
    if not rows:
        return 0.0
    ordered = sorted(rows, key=lambda item: (item[3], -item[4], item[2]))
    fail_probability = 1.0
    expected = 0.0
    for row in ordered:
        probability = max(0.0, min(1.0, row[4]))
        expected += fail_probability * probability * row[3]
        fail_probability *= 1.0 - probability
    expected += fail_probability * reject_penalty * len(ordered[0][0])
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
    return 100.0


def _assign_backup_couriers(candidates, selected, meta, policy=None):
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

    max_couriers_per_bundle = _policy_max_couriers(meta, policy)
    use_robust_gain = (
        meta.get("courier_count", 0) / max(meta.get("task_count", 1), 1) < 1.25
        or _is_low_willingness_case(candidates, meta)
    )
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
                if use_robust_gain:
                    current_assigned = sum(row[3] for row in current_rows)
                    improved_assigned = current_assigned + candidate[3]
                    alpha = _policy_alpha(candidates, meta, policy)
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
        used_couriers.add(candidate[2])

    return [
        (bundle["task_id_list_str"], [row[2] for row in sorted(bundle["rows"], key=lambda item: (item[3], -item[4], item[2]))])
        for bundle in bundles
    ]


def _finish_assignment(candidates, selected, meta, policy=None):
    if not selected:
        return []

    preferred_assigner = _assign_couriers_global if policy and policy.get("_assigner") == "global" else _assign_backup_couriers
    trial_assigners = [preferred_assigner]
    if _is_low_willingness_case(candidates, meta):
        other = _assign_backup_couriers if preferred_assigner is _assign_couriers_global else _assign_couriers_global
        trial_assigners.append(other)

    best_result = []
    best_key = (-1, float("-inf"), float("-inf"))
    for assigner in trial_assigners:
        trial_policy = dict(policy or {})
        trial_policy["_assigner"] = "global" if assigner is _assign_couriers_global else "seeded"
        result = assigner(candidates, selected, meta, trial_policy)
        result = _polish_assignment(candidates, result, meta, trial_policy)
        key = _policy_score(candidates, result, meta, trial_policy)
        if key > best_key:
            best_key = key
            best_result = result
    return best_result


def _assign_couriers_global(candidates, selected, meta, policy=None):
    if not selected:
        return []

    selected_bundles = []
    seen = set()
    for item in selected:
        if item[1] not in seen:
            selected_bundles.append(item[1])
            seen.add(item[1])

    by_bundle = {}
    for candidate in candidates:
        if candidate[1] in seen:
            by_bundle.setdefault(candidate[1], []).append(candidate)
    for rows in by_bundle.values():
        rows.sort(key=lambda item: (item[3], -item[4], item[2]))

    reject_penalty = _reject_penalty(meta)
    max_couriers_per_bundle = _policy_max_couriers(meta, policy)
    bundles = [{"task_id_list_str": bundle, "rows": []} for bundle in selected_bundles]
    used_couriers = set()

    def bundle_score(rows, task_count):
        if rows:
            expected = _expected_bundle_score(rows, reject_penalty)
            assigned = sum(row[3] for row in rows)
        else:
            expected = reject_penalty * task_count
            assigned = 0.0
        return _raw_objective_from_score(expected, assigned, candidates, meta, policy)

    while True:
        best_choice = None
        best_gain = 0.0
        for bundle_index, bundle in enumerate(bundles):
            current_rows = bundle["rows"]
            if len(current_rows) >= max_couriers_per_bundle:
                continue
            task_count = len(bundle["task_id_list_str"].split(","))
            current_objective = bundle_score(current_rows, task_count)
            current_couriers = {row[2] for row in current_rows}
            for candidate in by_bundle.get(bundle["task_id_list_str"], ())[:160]:
                if candidate[2] in used_couriers or candidate[2] in current_couriers:
                    continue
                improved_objective = bundle_score(current_rows + [candidate], task_count)
                gain = current_objective - improved_objective
                if gain > best_gain:
                    best_gain = gain
                    best_choice = (bundle_index, candidate)

        if best_choice is None or best_gain <= 1e-9:
            break

        bundle_index, candidate = best_choice
        bundles[bundle_index]["rows"].append(candidate)
        used_couriers.add(candidate[2])

    for bundle in bundles:
        if bundle["rows"]:
            continue
        for candidate in by_bundle.get(bundle["task_id_list_str"], ()):
            if candidate[2] not in used_couriers:
                bundle["rows"].append(candidate)
                used_couriers.add(candidate[2])
                break

    return [
        (bundle["task_id_list_str"], [row[2] for row in sorted(bundle["rows"], key=lambda item: (item[3], -item[4], item[2]))])
        for bundle in bundles
        if bundle["rows"]
    ]


def _polish_assignment(candidates, result, meta, policy=None):
    if not result:
        return result

    is_low = _is_low_willingness_case(candidates, meta)
    if not is_low:
        return result

    time_limit = 0.8
    start = perf_counter()
    rows_by_key = {(candidate[1], candidate[2]): candidate for candidate in candidates}
    reject_penalty = _reject_penalty(meta)
    max_couriers_per_bundle = _policy_max_couriers(meta, policy)

    bundles = []
    for task_id_list_str, courier_ids in result:
        rows = [rows_by_key[(task_id_list_str, courier_id)] for courier_id in courier_ids if (task_id_list_str, courier_id) in rows_by_key]
        if rows:
            bundles.append({"task_id_list_str": task_id_list_str, "rows": rows})
    if len(bundles) < 2:
        return result

    def objective(rows, task_id_list_str):
        if rows:
            expected = _expected_bundle_score(rows, reject_penalty)
            assigned = sum(row[3] for row in rows)
        else:
            expected = reject_penalty * len([task for task in task_id_list_str.split(",") if task])
            assigned = 0.0
        return _raw_objective_from_score(expected, assigned, candidates, meta, policy)

    bundle_objectives = [objective(bundle["rows"], bundle["task_id_list_str"]) for bundle in bundles]

    by_bundle = {}
    for candidate in candidates:
        by_bundle.setdefault(candidate[1], []).append(candidate)
    for rows in by_bundle.values():
        rows.sort(key=lambda item: (item[3] / max(item[4], 0.03), item[3], -item[4], item[2]))

    while perf_counter() - start <= time_limit:
        best = None
        best_gain = 1e-9
        used_couriers = {row[2] for bundle in bundles for row in bundle["rows"]}

        for bundle_index, bundle in enumerate(bundles):
            bundle_key = bundle["task_id_list_str"]
            current_score = bundle_objectives[bundle_index]
            for row_index, old_row in enumerate(bundle["rows"]):
                for candidate in by_bundle.get(bundle_key, ())[:120]:
                    if candidate[2] != old_row[2] and candidate[2] in used_couriers:
                        continue
                    if candidate[2] in {row[2] for idx, row in enumerate(bundle["rows"]) if idx != row_index}:
                        continue
                    if candidate == old_row:
                        continue
                    replacement_rows = list(bundle["rows"])
                    replacement_rows[row_index] = candidate
                    gain = current_score - objective(replacement_rows, bundle_key)
                    if gain > best_gain:
                        best_gain = gain
                        best = ("replace", bundle_index, row_index, candidate)

        for source_index, source in enumerate(bundles):
            if len(source["rows"]) <= 1:
                continue
            source_key = source["task_id_list_str"]
            source_score = bundle_objectives[source_index]
            for row_index, source_row in enumerate(source["rows"]):
                courier_id = source_row[2]
                source_after = source["rows"][:row_index] + source["rows"][row_index + 1 :]
                source_after_score = objective(source_after, source_key)

                for target_index, target in enumerate(bundles):
                    if target_index == source_index or len(target["rows"]) >= max_couriers_per_bundle:
                        continue
                    target_key = target["task_id_list_str"]
                    moved = rows_by_key.get((target_key, courier_id))
                    if moved is None:
                        continue
                    target_score = bundle_objectives[target_index]
                    target_after = target["rows"] + [moved]
                    gain = source_score + target_score - source_after_score - objective(target_after, target_key)
                    if gain > best_gain:
                        best_gain = gain
                        best = ("move", source_index, row_index, target_index, moved, source_after_score)

        assignment_refs = [
            (bundle_index, row_index, row)
            for bundle_index, bundle in enumerate(bundles)
            for row_index, row in enumerate(bundle["rows"])
        ]
        for left_pos, (left_bundle_index, left_row_index, left_row) in enumerate(assignment_refs):
            left_bundle = bundles[left_bundle_index]
            left_key = left_bundle["task_id_list_str"]
            left_score = bundle_objectives[left_bundle_index]
            for right_bundle_index, right_row_index, right_row in assignment_refs[left_pos + 1 :]:
                if right_bundle_index == left_bundle_index:
                    continue
                right_bundle = bundles[right_bundle_index]
                right_key = right_bundle["task_id_list_str"]
                left_replacement = rows_by_key.get((left_key, right_row[2]))
                right_replacement = rows_by_key.get((right_key, left_row[2]))
                if left_replacement is None or right_replacement is None:
                    continue
                if right_row[2] in {row[2] for idx, row in enumerate(left_bundle["rows"]) if idx != left_row_index}:
                    continue
                if left_row[2] in {row[2] for idx, row in enumerate(right_bundle["rows"]) if idx != right_row_index}:
                    continue
                left_after = list(left_bundle["rows"])
                right_after = list(right_bundle["rows"])
                left_after[left_row_index] = left_replacement
                right_after[right_row_index] = right_replacement
                gain = (
                    left_score
                    + bundle_objectives[right_bundle_index]
                    - objective(left_after, left_key)
                    - objective(right_after, right_key)
                )
                if gain > best_gain:
                    best_gain = gain
                    best = ("swap", left_bundle_index, left_row_index, left_replacement, right_bundle_index, right_row_index, right_replacement)

        if best is None:
            break

        if best[0] == "replace":
            _kind, bundle_index, row_index, candidate = best
            bundles[bundle_index]["rows"][row_index] = candidate
            bundle_objectives[bundle_index] = objective(bundles[bundle_index]["rows"], bundles[bundle_index]["task_id_list_str"])
        elif best[0] == "move":
            _kind, source_index, row_index, target_index, moved, source_after_score = best
            bundles[source_index]["rows"].pop(row_index)
            bundles[target_index]["rows"].append(moved)
            bundle_objectives[source_index] = source_after_score
            bundle_objectives[target_index] = objective(bundles[target_index]["rows"], bundles[target_index]["task_id_list_str"])
        else:
            (
                _kind,
                left_bundle_index,
                left_row_index,
                left_replacement,
                right_bundle_index,
                right_row_index,
                right_replacement,
            ) = best
            bundles[left_bundle_index]["rows"][left_row_index] = left_replacement
            bundles[right_bundle_index]["rows"][right_row_index] = right_replacement
            bundle_objectives[left_bundle_index] = objective(
                bundles[left_bundle_index]["rows"], bundles[left_bundle_index]["task_id_list_str"]
            )
            bundle_objectives[right_bundle_index] = objective(
                bundles[right_bundle_index]["rows"], bundles[right_bundle_index]["task_id_list_str"]
            )

    return [
        (bundle["task_id_list_str"], [row[2] for row in sorted(bundle["rows"], key=lambda item: (item[3], -item[4], item[2]))])
        for bundle in bundles
        if bundle["rows"]
    ]


def solve(input_text: str) -> list:
    candidates = _parse_input(input_text)
    return _agent_run(candidates)
