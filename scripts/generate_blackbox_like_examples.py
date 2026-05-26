from __future__ import annotations

import random
from pathlib import Path


CASE_CONFIGS = {
    "tiny_seed42": dict(task_count=6, courier_count=12, seed=42, noise=0.5, willingness_scale=1.0),
    "small_seed100": dict(task_count=15, courier_count=30, seed=100, noise=0.8, willingness_scale=1.0),
    "medium_seed201": dict(task_count=30, courier_count=60, seed=201, noise=0.9, willingness_scale=1.0),
    "medium_seed202": dict(task_count=30, courier_count=60, seed=202, noise=1.0, willingness_scale=1.0),
    "medium_seed203": dict(task_count=30, courier_count=60, seed=203, noise=1.1, willingness_scale=1.0),
    "large_seed301": dict(task_count=40, courier_count=80, seed=301, noise=1.0, willingness_scale=1.0),
    "large_seed302": dict(task_count=40, courier_count=80, seed=302, noise=1.05, willingness_scale=1.0),
    "scarce_couriers_seed401": dict(
        task_count=40,
        courier_count=44,
        seed=401,
        noise=0.9,
        willingness_scale=1.0,
        bundle_discount=0.68,
    ),
    "low_willingness_seed501": dict(task_count=30, courier_count=70, seed=501, noise=0.8, willingness_scale=0.28),
    "high_noise_seed601": dict(task_count=30, courier_count=70, seed=601, noise=2.3, willingness_scale=1.0),
}


def make_case(
    *,
    task_count: int,
    courier_count: int,
    seed: int,
    noise: float = 1.0,
    willingness_scale: float = 1.0,
    bundle_discount: float = 0.72,
) -> str:
    rng = random.Random(seed)
    lines = ["task_id_list\tcourier_id\ttotal_score\twillingness"]

    def willingness(task_a: int, courier: int, task_b: int | None = None) -> float:
        base = 0.08 + ((task_a * 13 + courier * 17 + (task_b or 0) * 7) % 78) / 100.0
        jitter = rng.uniform(-0.04, 0.04)
        return max(0.01, min(0.95, (base + jitter) * willingness_scale))

    def score(task_a: int, courier: int, task_b: int | None = None) -> float:
        distance = abs((task_a * 11 + (task_b or task_a) * 5) % 97 - (courier * 9) % 97)
        base = 14.0 + distance * 0.82 + ((task_a * 19 + courier * 23 + (task_b or 0) * 29) % 31)
        jitter = rng.gauss(0.0, 10.0 * noise)
        if task_b is not None:
            base = (base + 14.0 + abs(task_a - task_b) * 1.7) * bundle_discount
        return max(3.0, min(120.0, base + jitter))

    for task in range(task_count):
        for courier in range(courier_count):
            lines.append(
                f"T{task:04d}\tC{courier:03d}\t{score(task, courier):.3f}\t"
                f"{willingness(task, courier):.4f}"
            )

    for left in range(task_count):
        for right in range(left + 1, task_count):
            for courier in range(courier_count):
                lines.append(
                    f"T{left:04d},T{right:04d}\tC{courier:03d}\t"
                    f"{score(left, courier, right):.3f}\t{willingness(left, courier, right):.4f}"
                )

    return "\n".join(lines) + "\n"


def main() -> None:
    output_dir = Path("examples/blackbox_like")
    output_dir.mkdir(parents=True, exist_ok=True)

    for case_name, config in CASE_CONFIGS.items():
        path = output_dir / f"{case_name}.txt"
        path.write_text(make_case(**config), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
