from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaseConfig:
    task_count: int
    courier_count: int
    seed: int
    family: str
    noise: float = 1.0
    willingness_scale: float = 1.0
    bundle_discount: float = 0.74
    pair_willingness_scale: float = 1.0
    single_willingness_scale: float = 1.0
    score_willingness_corr: float = -0.45
    pair_bias: float = 0.0
    spatial_cluster: float = 1.0
    courier_quality_spread: float = 1.0


PUBLIC_CASES = {
    "tiny_seed42": CaseConfig(6, 12, 42, "tiny", noise=0.55, bundle_discount=0.72),
    "small_seed100": CaseConfig(15, 30, 100, "small", noise=0.75, bundle_discount=0.73),
    "medium_seed201": CaseConfig(30, 60, 201, "medium", noise=0.85, bundle_discount=0.72),
    "medium_seed202": CaseConfig(30, 60, 202, "medium", noise=1.0, bundle_discount=0.75),
    "medium_seed203": CaseConfig(30, 60, 203, "medium", noise=1.15, bundle_discount=0.78),
    "large_seed301": CaseConfig(40, 80, 301, "large", noise=0.95, bundle_discount=0.73),
    "large_seed302": CaseConfig(40, 80, 302, "large", noise=1.08, bundle_discount=0.76),
    "high_noise_seed601": CaseConfig(
        30,
        70,
        601,
        "high_noise",
        noise=2.25,
        bundle_discount=0.76,
        score_willingness_corr=-0.15,
    ),
    "low_willingness_seed501": CaseConfig(
        30,
        70,
        501,
        "low_willingness",
        noise=0.9,
        willingness_scale=0.34,
        bundle_discount=0.84,
        pair_willingness_scale=0.58,
        single_willingness_scale=1.42,
        score_willingness_corr=-0.2,
    ),
    "scarce_couriers_seed401": CaseConfig(
        40,
        44,
        401,
        "scarce",
        noise=0.9,
        bundle_discount=0.69,
        score_willingness_corr=-0.35,
    ),
}


PROBE_CASES = {
    "probe_scarce_38_balanced_seed700": CaseConfig(40, 38, 700, "scarce", noise=0.85, bundle_discount=0.67),
    "probe_scarce_48_expensive_pair_seed701": CaseConfig(
        40,
        48,
        701,
        "scarce",
        noise=1.05,
        bundle_discount=0.83,
        pair_bias=6.0,
        pair_willingness_scale=0.85,
    ),
    "probe_low_flat_seed710": CaseConfig(
        30,
        70,
        710,
        "low_willingness",
        noise=0.8,
        willingness_scale=0.28,
        bundle_discount=0.8,
        pair_willingness_scale=0.7,
        single_willingness_scale=1.2,
    ),
    "probe_low_pair_trap_seed711": CaseConfig(
        30,
        70,
        711,
        "low_willingness",
        noise=0.95,
        willingness_scale=0.36,
        bundle_discount=0.66,
        pair_willingness_scale=0.36,
        single_willingness_scale=1.5,
    ),
    "probe_high_noise_wide_seed720": CaseConfig(
        30,
        70,
        720,
        "high_noise",
        noise=2.8,
        bundle_discount=0.78,
        score_willingness_corr=0.05,
    ),
    "probe_medium_pair_good_seed730": CaseConfig(30, 60, 730, "medium", noise=0.9, bundle_discount=0.62),
    "probe_medium_pair_weak_seed731": CaseConfig(30, 60, 731, "medium", noise=0.9, bundle_discount=0.88),
    "probe_medium_positive_corr_seed732": CaseConfig(
        30,
        60,
        732,
        "medium",
        noise=1.0,
        bundle_discount=0.76,
        score_willingness_corr=0.55,
    ),
    "probe_large_clustered_seed740": CaseConfig(
        40,
        80,
        740,
        "large",
        noise=1.0,
        bundle_discount=0.7,
        spatial_cluster=0.45,
    ),
    "probe_large_flat_scores_seed741": CaseConfig(
        40,
        80,
        741,
        "large",
        noise=0.35,
        bundle_discount=0.78,
        courier_quality_spread=0.25,
    ),
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def point(rng: random.Random, cluster: float) -> tuple[float, float]:
    if cluster < 0.75:
        center_x = rng.choice((25.0, 50.0, 75.0))
        center_y = rng.choice((25.0, 50.0, 75.0))
        return (
            clamp(rng.gauss(center_x, 18.0 * cluster), 0.0, 100.0),
            clamp(rng.gauss(center_y, 18.0 * cluster), 0.0, 100.0),
        )
    return (rng.uniform(0.0, 100.0), rng.uniform(0.0, 100.0))


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def make_case(config: CaseConfig) -> str:
    rng = random.Random(config.seed)
    lines = ["task_id_list\tcourier_id\ttotal_score\twillingness"]

    tasks = []
    for task in range(config.task_count):
        pickup = point(rng, config.spatial_cluster)
        dropoff = point(rng, config.spatial_cluster)
        urgency = rng.uniform(-0.35, 0.35)
        tasks.append({"pickup": pickup, "dropoff": dropoff, "urgency": urgency})

    couriers = []
    for courier in range(config.courier_count):
        loc = point(rng, 1.0)
        quality = rng.gauss(0.0, 0.32 * config.courier_quality_spread)
        cost_bias = rng.gauss(0.0, 6.0 * config.courier_quality_spread)
        couriers.append({"loc": loc, "quality": quality, "cost_bias": cost_bias})

    def single_score(task: int, courier: int) -> float:
        task_info = tasks[task]
        courier_info = couriers[courier]
        approach = distance(courier_info["loc"], task_info["pickup"])
        delivery = distance(task_info["pickup"], task_info["dropoff"])
        base = 8.0 + approach * 0.24 + delivery * 0.46 + courier_info["cost_bias"] + task_info["urgency"] * 8.0
        jitter = rng.gauss(0.0, 7.0 * config.noise)
        return clamp(base + jitter, 3.0, 140.0)

    def pair_score(left: int, right: int, courier: int) -> float:
        left_task = tasks[left]
        right_task = tasks[right]
        courier_info = couriers[courier]
        approach = min(
            distance(courier_info["loc"], left_task["pickup"]),
            distance(courier_info["loc"], right_task["pickup"]),
        )
        pickup_hop = distance(left_task["pickup"], right_task["pickup"])
        delivery_hop = distance(left_task["dropoff"], right_task["dropoff"])
        service = (
            distance(left_task["pickup"], left_task["dropoff"])
            + distance(right_task["pickup"], right_task["dropoff"])
        )
        raw = 12.0 + approach * 0.18 + pickup_hop * 0.2 + delivery_hop * 0.18 + service * 0.36
        raw += courier_info["cost_bias"] * 0.75 + (left_task["urgency"] + right_task["urgency"]) * 5.0
        raw = raw * config.bundle_discount + config.pair_bias
        jitter = rng.gauss(0.0, 8.5 * config.noise)
        return clamp(raw + jitter, 3.0, 180.0)

    single_cache: dict[tuple[int, int], float] = {}
    pair_cache: dict[tuple[int, int, int], float] = {}

    def willingness(score: float, courier: int, is_pair: bool) -> float:
        courier_quality = couriers[courier]["quality"]
        normalized_score = (score - 48.0) / 34.0
        # Negative correlation means cheaper/shorter candidates tend to be accepted more often.
        signal = 0.05 + courier_quality + config.score_willingness_corr * normalized_score
        signal += rng.gauss(0.0, 0.7)
        base = 0.08 + 0.78 * sigmoid(signal)
        scale = config.willingness_scale * (
            config.pair_willingness_scale if is_pair else config.single_willingness_scale
        )
        return clamp(base * scale, 0.01, 0.95)

    for task in range(config.task_count):
        for courier in range(config.courier_count):
            score = single_score(task, courier)
            single_cache[(task, courier)] = score
            lines.append(f"T{task:04d}\tC{courier:03d}\t{score:.3f}\t{willingness(score, courier, False):.4f}")

    for left in range(config.task_count):
        for right in range(left + 1, config.task_count):
            for courier in range(config.courier_count):
                score = pair_score(left, right, courier)
                pair_cache[(left, right, courier)] = score
                lines.append(
                    f"T{left:04d},T{right:04d}\tC{courier:03d}\t"
                    f"{score:.3f}\t{willingness(score, courier, True):.4f}"
                )

    return "\n".join(lines) + "\n"


def reset_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = output_dir.resolve()
    workspace = Path.cwd().resolve()
    if workspace not in resolved.parents and resolved != workspace:
        raise RuntimeError(f"Refusing to reset path outside workspace: {resolved}")
    for path in output_dir.glob("*.txt"):
        path.unlink()


def write_cases(output_dir: Path, cases: dict[str, CaseConfig]) -> None:
    reset_output_dir(output_dir)
    for case_name, config in cases.items():
        path = output_dir / f"{case_name}.txt"
        path.write_text(make_case(config), encoding="utf-8")
        print(path)


def main() -> None:
    write_cases(Path("examples/blackbox_like"), PUBLIC_CASES)
    write_cases(Path("examples/probe_suite"), PROBE_CASES)


if __name__ == "__main__":
    main()
