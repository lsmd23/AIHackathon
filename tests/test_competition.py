from autosolver_agent.competition import (
    AutoSolverAgent,
    branch_bound_algorithm,
    compute_metadata,
    parse_competition_input,
    solve_competition_text,
)


def test_parse_competition_input_reads_header_and_candidates() -> None:
    instance = parse_competition_input(
        "\n".join(
            [
                "task_id_list\tcourier_id\ttotal_score\twillingness",
                "T0001,T0002\tC001\t12.5\t0.8",
                "T0003\tC002\t8.0\t0.3",
            ]
        )
    )

    assert len(instance.candidates) == 2
    assert instance.candidates[0].task_id_list == ("T0001", "T0002")
    assert instance.candidates[0].courier_id == "C001"
    assert instance.task_ids == frozenset({"T0001", "T0002", "T0003"})
    assert instance.courier_ids == frozenset({"C001", "C002"})


def test_solve_competition_text_returns_submission_shape_without_conflicts() -> None:
    result = solve_competition_text(
        "\n".join(
            [
                "task_id_list\tcourier_id\ttotal_score\twillingness",
                "T0001,T0002\tC001\t10.0\t0.8",
                "T0001\tC002\t1.0\t0.9",
                "T0003\tC001\t1.0\t0.9",
            ]
        )
    )

    used_tasks = set()
    used_couriers = set()
    for task_id_list_str, courier_ids in result:
        assert isinstance(task_id_list_str, str)
        assert isinstance(courier_ids, list)
        for courier_id in courier_ids:
            assert courier_id not in used_couriers
            used_couriers.add(courier_id)
        for task_id in task_id_list_str.split(","):
            assert task_id not in used_tasks
            used_tasks.add(task_id)


def test_agent_selects_heuristic_for_small_bundle_cases() -> None:
    instance = parse_competition_input(
        "\n".join(
            [
                "task_id_list\tcourier_id\ttotal_score\twillingness",
                "T0001,T0002\tC001\t1.0\t0.5",
            ]
        )
    )

    metadata = compute_metadata(instance)

    assert AutoSolverAgent().decide_algorithm(metadata) == "heuristic_search"


def test_branch_bound_prefers_more_covered_tasks_then_lower_score() -> None:
    instance = parse_competition_input(
        "\n".join(
            [
                "task_id_list\tcourier_id\ttotal_score\twillingness",
                "T0001,T0002\tC001\t100.0\t0.1",
                "T0001\tC001\t1.0\t0.9",
                "T0002\tC002\t2.0\t0.8",
            ]
        )
    )
    metadata = compute_metadata(instance)

    result = branch_bound_algorithm(instance, metadata, time_limit_seconds=0.5)

    assert result.algorithm == "branch_bound"
    assert result.to_submission() == [("T0001", ["C001"]), ("T0002", ["C002"])]
