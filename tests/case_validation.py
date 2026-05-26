from __future__ import annotations


def input_index(input_text: str):
    pairs = set()
    tasks = set()
    for row in input_text.strip().splitlines()[1:]:
        task_id_list_str, courier_id, *_rest = row.split("\t")
        pairs.add((task_id_list_str, courier_id))
        tasks.update(task_id_list_str.split(","))
    return pairs, tasks


def assert_valid_submission(input_text: str, result: list, *, require_complete: bool = False) -> None:
    valid_pairs, known_tasks = input_index(input_text)
    used_tasks = set()
    used_couriers = set()

    for task_id_list_str, courier_ids in result:
        assert isinstance(task_id_list_str, str)
        assert isinstance(courier_ids, list)
        assert courier_ids

        for courier_id in courier_ids:
            assert (task_id_list_str, courier_id) in valid_pairs
            assert courier_id not in used_couriers
            used_couriers.add(courier_id)

        for task_id in task_id_list_str.split(","):
            assert task_id in known_tasks
            assert task_id not in used_tasks
            used_tasks.add(task_id)

    if require_complete:
        assert used_tasks == known_tasks
