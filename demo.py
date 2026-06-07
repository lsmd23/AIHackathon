"""
demo.py — 可录屏演示(纯标准库伪前端)

ASCII 风格,终端宽度自适应。所有 UI 元素:
  - 顶/底大标题
  - 实时日志(时间戳 + 级别标签)
  - 旋转 spinner(超过 0.3s 的耗时步骤)
  - 进度小条
  - 结果表格(对齐列)
  - 合法性 ✓/✗ 大字 badge
  - 样本分组(箭头 ←)
  - 跑完汇总

运行:    python demo.py
录屏:    终端宽度 ≥ 90 列,字体等宽(monospace)
"""

from __future__ import annotations

import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import solver


HEADER = "task_id_list\tcourier_id\ttotal_score\twillingness"
WIDTH = 90


# ---------------------------------------------------------------------------
# UI 组件
# ---------------------------------------------------------------------------


class UI:
    """统一打印接口。所有输出经过这里,确保风格统一。"""

    def __init__(self, width: int = WIDTH) -> None:
        self.width = width

    def hr(self, char: str = "-") -> None:
        print(char * self.width)

    def hr_double(self) -> None:
        self.hr("=")

    def blank(self) -> None:
        print()

    def title(self, text: str) -> None:
        print()
        self.hr_double()
        print(text.center(self.width))
        self.hr_double()

    def section(self, text: str) -> None:
        self.blank()
        self.hr()
        print(text)
        self.hr()

    def case_header(self, idx: int, total: int, name: str) -> None:
        self.blank()
        self.hr_double()
        print(f"  CASE {idx}/{total}   {name}")
        self.hr_double()

    def _stamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def log(self, level: str, msg: str) -> None:
        # 用方括号框住级别,清楚分隔
        print(f"  [{self._stamp()}] [{level}]  {msg}")

    def info(self, msg: str) -> None:
        self.log("INFO", msg)

    def warn(self, msg: str) -> None:
        self.log("WARN", msg)

    def ok(self, msg: str) -> None:
        self.log(" OK ", msg)

    def fail(self, msg: str) -> None:
        self.log("FAIL", msg)

    def table(self, rows: list[tuple[str, str, str]], label_w: int = 14) -> None:
        for label, value, unit in rows:
            print(f"    {label:<{label_w}}  {value:>10}   {unit}")

    def pass_badge(self) -> None:
        self.blank()
        print("    " + "*" * 50)
        print("      " + "LEGALITY:  P A S S" + "  " + u"✓" * 5)
        print("    " + "*" * 50)
        self.blank()

    def fail_badge(self, reason: str) -> None:
        self.blank()
        print("    " + "*" * 50)
        print(f"      LEGALITY:  F A I L   ({reason})")
        print("    " + "*" * 50)
        self.blank()

    def final_summary(self, cases: list[dict]) -> None:
        self.blank()
        self.title("DONE  -  SUMMARY")
        self.blank()
        hdr = f"  {'CASE':<46} {'GROUPS':>7} {'COVERED':>9} {'COURIERS':>10} {'TIME_MS':>10} {'RANKED':>10}"
        print(hdr)
        self.hr()
        for c in cases:
            covered = f"{c['covered']}/{c['tasks']}"
            print(
                f"  {c['name']:<46} {c['groups']:>7} {covered:>9} {c['couriers']:>10} "
                f"{c['runtime_ms']:>10.0f} {c['ranked']:>10.2f}"
            )
        self.blank()
        all_pass = all(c["legal"] for c in cases)
        if all_pass:
            print(f"  All {len(cases)}/{len(cases)} cases PASS legality check.")
        else:
            failed = sum(1 for c in cases if not c["legal"])
            print(f"  WARNING: {failed} case(s) FAILED legality check.")

        self.blank()
        print("  Official online average penalty score: 727.85 (10/10 case, 100% coverage)")
        print("  This demo shows 5 local cases; online 10 are evaluated by the platform.")
        self.blank()
        self.hr_double()


class Spinner:
    """后台线程 spinner,用 \\r 覆盖同一行。"""

    FRAMES = "|/-\\"

    def __init__(self, message: str = "") -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread:
            self._thread.join()
        # 清掉 spinner 这一行
        sys.stdout.write("\r" + " " * (len(self.message) + 6) + "\r")
        sys.stdout.flush()

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r  [{frame}] {self.message}")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1


def timed_run(message: str, func, *args, min_spinner_seconds: float = 0.3):
    """执行 func(*args),若耗时 > min_spinner_seconds 显示 spinner。"""
    t0 = time.perf_counter()
    if min_spinner_seconds <= 0:
        result = func(*args)
        return result, time.perf_counter() - t0

    result_box: list = [None]

    def runner():
        result_box[0] = func(*args)

    th = threading.Thread(target=runner, daemon=True)
    th.start()
    spinner_started = False
    spinner = None
    while th.is_alive():
        elapsed = time.perf_counter() - t0
        if elapsed >= min_spinner_seconds and not spinner_started:
            spinner = Spinner(message)
            spinner.__enter__()
            spinner_started = True
        time.sleep(0.02)
    th.join()
    if spinner_started and spinner is not None:
        spinner.__exit__(None, None, None)
    return result_box[0], time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 数据:合成候选
# ---------------------------------------------------------------------------


def build_tsv(rows: Iterable[tuple[str, str, float, float]]) -> str:
    lines = [HEADER]
    for task_id_list_str, courier_id, total_score, willingness in rows:
        lines.append(
            f"{task_id_list_str}\t{courier_id}\t{total_score}\t{willingness}"
        )
    return "\n".join(lines) + "\n"


def synth_tiny() -> tuple[str, str]:
    rng = random.Random(42)
    rows = []
    for t in range(1, 7):
        for c in range(1, 13):
            score = 5 + rng.random() * 10
            will = 0.5 + rng.random() * 0.4
            rows.append((f"T000{t}", f"C{c:03d}", round(score, 2), round(will, 3)))
    return build_tsv(rows), "tiny (6 tasks / 12 couriers)"


def synth_small() -> tuple[str, str]:
    rng = random.Random(100)
    rows = []
    for t in range(1, 16):
        for c in range(1, 21):
            score = 8 + rng.random() * 12
            will = 0.4 + rng.random() * 0.5
            rows.append((f"T{t:04d}", f"C{c:03d}", round(score, 2), round(will, 3)))
    for t1 in range(1, 16, 3):
        t2 = t1 + 1
        if t2 > 15:
            break
        for c in range(1, 11):
            score = 18 + rng.random() * 20
            will = 0.3 + rng.random() * 0.4
            rows.append((f"T{t1:04d},T{t2:04d}", f"C{c:03d}", round(score, 2), round(will, 3)))
    return build_tsv(rows), "small (15 tasks / 30 couriers)"


def synth_low_willingness() -> tuple[str, str]:
    rng = random.Random(501)
    rows = []
    for t in range(1, 31):
        for c in range(1, 31):
            score = 10 + rng.random() * 20
            will = 0.2 + rng.random() * 0.4
            rows.append((f"T{t:04d}", f"C{c:03d}", round(score, 2), round(will, 3)))
    for t1 in range(1, 31, 4):
        t2 = t1 + 1
        if t2 > 30:
            break
        for c in range(1, 16):
            score = 30 + rng.random() * 40
            will = 0.15 + rng.random() * 0.3
            rows.append((f"T{t1:04d},T{t2:04d}", f"C{c:03d}", round(score, 2), round(will, 3)))
    return build_tsv(rows), "low_willingness (30 tasks / 30 couriers, mean will 0.4)"


def synth_scarce() -> tuple[str, str]:
    rng = random.Random(401)
    rows = []
    for t in range(1, 41):
        for c in range(1, 39):
            score = 12 + rng.random() * 18
            will = 0.4 + rng.random() * 0.5
            rows.append((f"T{t:04d}", f"C{c:03d}", round(score, 2), round(will, 3)))
    for t1 in range(1, 41, 4):
        t2 = t1 + 1
        if t2 > 40:
            break
        for c in range(1, 16):
            score = 25 + rng.random() * 30
            will = 0.3 + rng.random() * 0.4
            rows.append((f"T{t1:04d},T{t2:04d}", f"C{c:03d}", round(score, 2), round(will, 3)))
    return build_tsv(rows), "scarce (40 tasks / 38 couriers, tasks > couriers)"


def synth_large_official() -> tuple[str, str]:
    return (
        Path("examples/large_seed301.txt").read_text(encoding="utf-8"),
        "large_seed301 (official, 40 tasks / 80 couriers / 33780 candidates)",
    )


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def parse_rows(input_text: str):
    rows = {}
    for line in input_text.strip().splitlines()[1:]:
        if not line.strip():
            continue
        t, c, s, w = line.split("\t")[:4]
        rows[(t.strip(), c.strip())] = (float(s), float(w))
    return rows


def ranked_expected(input_text: str, result) -> float:
    rows = parse_rows(input_text)
    total = 0.0
    for t_str, couriers in result:
        fail = 1.0
        for c in couriers:
            s, w = rows[(t_str, c)]
            total += fail * w * s
            fail *= 1.0 - w
        n = len([x for x in t_str.split(",") if x.strip()])
        total += fail * 100.0 * n
    return total


def parallel_expected(input_text: str, result) -> float:
    rows = parse_rows(input_text)
    total = 0.0
    for t_str, couriers in result:
        fail = 1.0
        wsum = 0.0
        wscore = 0.0
        for c in couriers:
            s, w = rows[(t_str, c)]
            fail *= 1.0 - w
            wsum += w
            wscore += w * s
        n = len([x for x in t_str.split(",") if x.strip()])
        succ = 1.0 - fail
        avg = wscore / max(wsum, 1e-9)
        total += succ * avg + fail * 100.0 * n
    return total


def fulfill_score(input_text: str, result) -> float:
    rows = parse_rows(input_text)
    total = 0.0
    for t_str, couriers in result:
        fail = 1.0
        wsum = 0.0
        wscore = 0.0
        for c in couriers:
            s, w = rows[(t_str, c)]
            fail *= 1.0 - w
            wsum += w
            wscore += w * s
        n = len([x for x in t_str.split(",") if x.strip()])
        total += fail * 100.0 * n
        total += (wscore / max(wsum, 1e-9)) * 0.15
    return total


def assigned_total(input_text: str, result) -> float:
    rows = parse_rows(input_text)
    return sum(rows[(t, c)][0] for t, couriers in result for c in couriers)


def validate(input_text: str, result) -> tuple[bool, str]:
    rows = parse_rows(input_text)
    used_tasks = set()
    used_couriers = set()
    for t_str, couriers in result:
        for task in t_str.split(","):
            task = task.strip()
            if task in used_tasks:
                return False, f"task {task} appears twice"
            used_tasks.add(task)
        for c in couriers:
            if c in used_couriers:
                return False, f"courier {c} appears twice"
            used_couriers.add(c)
            if (t_str, c) not in rows:
                return False, f"({t_str},{c}) not in input"
    return True, "ok"


# ---------------------------------------------------------------------------
# 案例展示
# ---------------------------------------------------------------------------


def run_case(ui: UI, idx: int, total: int, name: str, input_text: str) -> dict:
    ui.case_header(idx, total, name)
    ui.blank()

    cand, t_parse = timed_run(
        "parsing input", solver._parse_input, input_text, min_spinner_seconds=0.3
    )
    tasks = {task for item in cand for task in item[0]}
    couriers = {item[2] for item in cand}
    ui.info(f"input size: {len(tasks)} tasks, {len(couriers)} couriers, {len(cand)} candidates (parse {t_parse * 1000:.0f}ms)")

    result, t_solve = timed_run(
        "solving (two-layer agent)", solver.solve, input_text, min_spinner_seconds=0.3
    )
    ui.info(f"solver done in {t_solve * 1000:.0f} ms")

    covered = {t for t_str, _ in result for t in t_str.split(",")}
    used_c = {c for _, cs in result for c in cs}
    ok, msg = validate(input_text, result)

    ui.info(
        f"output: {len(result)} bundles, {len(covered)}/{len(tasks)} covered"
        f" ({len(covered) / max(len(tasks), 1) * 100:.0f}%), {len(used_c)} couriers used"
    )
    if ok:
        ui.ok("legality: tasks unique / couriers unique / all from input")
    else:
        ui.fail(f"legality: {msg}")

    ui.blank()
    ui.info("proxy scores (ranked / parallel / fulfill / assigned):")
    ui.table(
        [
            ("ranked", f"{ranked_expected(input_text, result):.2f}", "expected score (sequential)"),
            ("parallel", f"{parallel_expected(input_text, result):.2f}", "expected score (parallel)"),
            ("fulfill", f"{fulfill_score(input_text, result):.2f}", "fulfillment proxy"),
            ("assigned", f"{assigned_total(input_text, result):.2f}", "ticket-face total cost"),
        ]
    )

    ui.blank()
    ui.info(f"sample bundles (first {min(3, len(result))}):")
    for t_str, cs in result[:3]:
        cs_str = ", ".join(cs) if cs else "(none)"
        print(f"    {t_str!r}  <-  [{cs_str}]")

    if ok:
        ui.pass_badge()
    else:
        ui.fail_badge(msg)

    return {
        "name": name,
        "groups": len(result),
        "covered": len(covered),
        "tasks": len(tasks),
        "couriers": len(used_c),
        "runtime_ms": t_solve * 1000,
        "ranked": ranked_expected(input_text, result),
        "legal": ok,
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    ui = UI()

    ui.title("Courier Dispatch Solver  -  Demo v1.0")
    ui.blank()
    ui.info(f"start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ui.info("runtime: Python 3.11+ / stdlib only / no third-party deps")
    ui.info("official score: 727.85 average penalty (10/10 case, 100% coverage)")
    ui.info("this demo runs 5 local cases end-to-end")

    if not Path("examples/large_seed301.txt").exists():
        ui.fail("examples/large_seed301.txt missing - run from repo root")
        sys.exit(1)

    _, t_prep = timed_run("preparing", lambda: None, min_spinner_seconds=0.3)
    ui.ok(f"ready (setup {t_prep * 1000:.0f}ms)")

    factories = [
        synth_tiny,
        synth_small,
        synth_low_willingness,
        synth_scarce,
        synth_large_official,
    ]

    summary: list[dict] = []
    total = len(factories)
    for idx, factory in enumerate(factories, 1):
        input_text, name = factory()
        record = run_case(ui, idx, total, name, input_text)
        summary.append(record)

    ui.final_summary(summary)


if __name__ == "__main__":
    main()
