"""
backend/api.py — FastAPI SSE 流式日志接口

启动:    cd frontend/backend && uvicorn api:app --reload --port 8000
接口:
  GET /api/cases           → case 列表
  GET /api/run             → SSE 流,每行 yield {ts, level, text}
  GET /api/ping            → 健康检查

事件类型 (SSE data 字段):
  {"type":"line", "ts":"...", "level":"INFO", "text":"..."}
  {"type":"case_start", "case":"tiny", "idx":1, "total":5}
  {"type":"case_end",   "case":"tiny", "ok":true}
  {"type":"done"}
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

# 把仓库根加进 path,这样能 import solver 和 demo
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# 复用 demo 里的合成器和指标计算(不动 demo.py 本体)
import demo as demo_mod  # noqa: E402

app = FastAPI(title="Courier Solver Demo API")

# Vite dev 默认 5173 端口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


CASES: list[dict] = [
    {"id": "tiny",            "name": "tiny (6 tasks / 12 couriers)",            "factory": "synth_tiny"},
    {"id": "small",           "name": "small (15 tasks / 30 couriers)",          "factory": "synth_small"},
    {"id": "low_willingness", "name": "low_willingness (30 tasks / 30 couriers, mean will 0.4)", "factory": "synth_low_willingness"},
    {"id": "scarce",          "name": "scarce (40 tasks / 38 couriers, tasks > couriers)",      "factory": "synth_scarce"},
    {"id": "large",           "name": "large_seed301 (official, 40 tasks / 80 couriers / 33780 candidates)", "factory": "synth_large_official"},
]


@app.get("/api/ping")
def ping() -> dict:
    return {"ok": True, "ts": datetime.now().isoformat()}


@app.get("/api/cases")
def list_cases() -> list[dict]:
    return CASES


# ---------------------------------------------------------------------------
# 流式 UI:把 demo.UI 替换为 emit 事件的版本
# ---------------------------------------------------------------------------


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


async def _emit(queue: asyncio.Queue, payload: dict) -> None:
    await queue.put(payload)


async def _emit_lines(queue: asyncio.Queue, lines: list[str], level: str = "INFO", delay: float = 0.02) -> None:
    """逐行 emit,每行间微小延迟,模拟实时日志。"""
    for line in lines:
        await _emit(queue, {"type": "line", "ts": _ts(), "level": level, "text": line})
        if delay > 0:
            await asyncio.sleep(delay)


class StreamUI:
    """demo.UI 的 SSE 版。"""

    def __init__(self, queue: asyncio.Queue) -> None:
        self.q = queue
        self.width = 90

    async def blank(self) -> None:
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "BLNK", "text": ""})

    async def hr(self) -> None:
        await self.blank()
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "HR", "text": "=" * self.width})
        await self.blank()

    async def case_header(self, idx: int, total: int, name: str) -> None:
        await _emit(self.q, {"type": "case_start", "case": name, "idx": idx, "total": total})
        await self.hr()
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "HEAD", "text": f"  CASE {idx}/{total}   {name}"})
        await self.hr()
        await self.blank()

    async def info(self, msg: str) -> None:
        await self._log("INFO", msg)

    async def ok(self, msg: str) -> None:
        await self._log("OK", msg)

    async def fail(self, msg: str) -> None:
        await self._log("FAIL", msg)

    async def _log(self, level: str, msg: str) -> None:
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": level, "text": msg})

    async def table(self, rows: list[tuple[str, str, str]], label_w: int = 14) -> None:
        for label, value, unit in rows:
            text = f"    {label:<{label_w}}  {value:>10}   {unit}"
            await _emit(self.q, {"type": "line", "ts": _ts(), "level": "TBL", "text": text})

    async def pass_badge(self) -> None:
        await self.blank()
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "BADP", "text": "    " + "*" * 50})
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "BADP", "text": "      LEGALITY:  P A S S  " + "✓" * 5})
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "BADP", "text": "    " + "*" * 50})
        await self.blank()

    async def fail_badge(self, reason: str) -> None:
        await self.blank()
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "BADF", "text": "    " + "*" * 50})
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "BADF", "text": f"      LEGALITY:  F A I L   ({reason})"})
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "BADF", "text": "    " + "*" * 50})
        await self.blank()

    async def final_summary(self, cases: list[dict]) -> None:
        await self.blank()
        await self.hr()
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "HEAD", "text": "DONE  -  SUMMARY".center(self.width)})
        await self.hr()
        await self.blank()
        hdr = f"  {'CASE':<46} {'GROUPS':>7} {'COVERED':>9} {'COURIERS':>10} {'TIME_MS':>10} {'RANKED':>10}"
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "TBL", "text": hdr})
        await _emit(self.q, {"type": "line", "ts": _ts(), "level": "HR",  "text": "-" * self.width})
        for c in cases:
            covered = f"{c['covered']}/{c['tasks']}"
            text = (
                f"  {c['name']:<46} {c['groups']:>7} {covered:>9} {c['couriers']:>10} "
                f"{c['runtime_ms']:>10.0f} {c['ranked']:>10.2f}"
            )
            await _emit(self.q, {"type": "line", "ts": _ts(), "level": "TBL", "text": text})
        await self.blank()
        all_pass = all(c["legal"] for c in cases)
        if all_pass:
            await self.info(f"All {len(cases)}/{len(cases)} cases PASS legality check.")
        else:
            failed = sum(1 for c in cases if not c["legal"])
            await self.fail(f"{failed} case(s) FAILED legality check.")
        await self.blank()
        await self.info("Official online average penalty score: 727.85 (10/10 case, 100% coverage)")
        await self.info("This demo shows 5 local cases; online 10 are evaluated by the platform.")
        await self.blank()
        await self.hr()


# ---------------------------------------------------------------------------
# 单 case 跑(供前端用 /api/run?case=xxx)
# ---------------------------------------------------------------------------


async def run_single_case(queue: asyncio.Queue, case_id: str) -> None:
    """跑单个 case,emit 日志到 queue。"""
    case = next((c for c in CASES if c["id"] == case_id), None)
    if case is None:
        await _emit(queue, {"type": "error", "msg": f"unknown case: {case_id}"})
        return

    factory = getattr(demo_mod, case["factory"])
    input_text, name = factory()

    ui = StreamUI(queue)
    await ui.case_header(1, 1, name)

    cand = demo_mod.solver._parse_input(input_text)
    tasks = {task for item in cand for task in item[0]}
    couriers = {item[2] for item in cand}
    await ui.info(f"input size: {len(tasks)} tasks, {len(couriers)} couriers, {len(cand)} candidates")

    t0 = time.perf_counter()
    result = demo_mod.solver.solve(input_text)
    elapsed = time.perf_counter() - t0
    await ui.info(f"solver done in {elapsed * 1000:.0f} ms")

    covered = {t for t_str, _ in result for t in t_str.split(",")}
    used_c = {c for _, cs in result for c in cs}
    ok, msg = demo_mod.validate(input_text, result)
    await ui.info(
        f"output: {len(result)} bundles, {len(covered)}/{len(tasks)} covered"
        f" ({len(covered) / max(len(tasks), 1) * 100:.0f}%), {len(used_c)} couriers used"
    )
    if ok:
        await ui.ok("legality: tasks unique / couriers unique / all from input")
    else:
        await ui.fail(f"legality: {msg}")

    await ui.blank()
    await ui.info("proxy scores (ranked / parallel / fulfill / assigned):")
    await ui.table(
        [
            ("ranked", f"{demo_mod.ranked_expected(input_text, result):.2f}", "expected score (sequential)"),
            ("parallel", f"{demo_mod.parallel_expected(input_text, result):.2f}", "expected score (parallel)"),
            ("fulfill", f"{demo_mod.fulfill_score(input_text, result):.2f}", "fulfillment proxy"),
            ("assigned", f"{demo_mod.assigned_total(input_text, result):.2f}", "ticket-face total cost"),
        ]
    )

    await ui.blank()
    await ui.info(f"sample bundles (first {min(3, len(result))}):")
    for t_str, cs in result[:3]:
        cs_str = ", ".join(cs) if cs else "(none)"
        await _emit(queue, {"type": "line", "ts": _ts(), "level": "SAMP", "text": f"    {t_str!r}  <-  [{cs_str}]"})

    if ok:
        await ui.pass_badge()
    else:
        await ui.fail_badge(msg)

    await _emit(queue, {"type": "case_end", "case": name, "ok": ok})


# ---------------------------------------------------------------------------
# 全跑(供前端用 /api/run)
# ---------------------------------------------------------------------------


async def run_all_cases(queue: asyncio.Queue) -> None:
    """跑全部 5 个 case,emit 日志。"""
    ui = StreamUI(queue)
    await ui.hr()
    await _emit(queue, {"type": "line", "ts": _ts(), "level": "HEAD", "text": "Courier Dispatch Solver  -  Demo v1.0".center(ui.width)})
    await ui.hr()
    await ui.blank()
    await ui.info(f"start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    await ui.info("runtime: Python 3.11+ / stdlib only / no third-party deps")
    await ui.info("official score: 727.85 average penalty (10/10 case, 100% coverage)")
    await ui.info("this demo runs 5 local cases end-to-end")
    await ui.blank()

    summary: list[dict] = []
    total = len(CASES)

    for idx, case in enumerate(CASES, 1):
        factory = getattr(demo_mod, case["factory"])
        input_text, name = factory()
        await _emit(queue, {"type": "case_start", "case": name, "idx": idx, "total": total})
        await ui.blank()
        await ui.hr()
        await _emit(queue, {"type": "line", "ts": _ts(), "level": "HEAD", "text": f"  CASE {idx}/{total}   {name}"})
        await ui.hr()
        await ui.blank()

        cand = demo_mod.solver._parse_input(input_text)
        tasks = {task for item in cand for task in item[0]}
        couriers = {item[2] for item in cand}
        await ui.info(f"input size: {len(tasks)} tasks, {len(couriers)} couriers, {len(cand)} candidates")

        t0 = time.perf_counter()
        # 跑大样例时给前端"心跳",让进度条动起来
        result_box: list = [None]

        def _solve():
            result_box[0] = demo_mod.solver.solve(input_text)

        th_task = asyncio.create_task(asyncio.to_thread(_solve))
        # 心跳:大样例每 200ms 推一次
        heartbeat = 0
        while not th_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(th_task), timeout=0.2)
            except asyncio.TimeoutError:
                heartbeat += 1
                if elapsed_so_far := (time.perf_counter() - t0):
                    await _emit(queue, {
                        "type": "heartbeat",
                        "elapsed_ms": int(elapsed_so_far * 1000),
                        "tick": heartbeat,
                    })
        await th_task
        result = result_box[0]
        elapsed = time.perf_counter() - t0
        await ui.info(f"solver done in {elapsed * 1000:.0f} ms")

        covered = {t for t_str, _ in result for t in t_str.split(",")}
        used_c = {c for _, cs in result for c in cs}
        ok, msg = demo_mod.validate(input_text, result)
        await ui.info(
            f"output: {len(result)} bundles, {len(covered)}/{len(tasks)} covered"
            f" ({len(covered) / max(len(tasks), 1) * 100:.0f}%), {len(used_c)} couriers used"
        )
        if ok:
            await ui.ok("legality: tasks unique / couriers unique / all from input")
        else:
            await ui.fail(f"legality: {msg}")

        await ui.blank()
        await ui.info("proxy scores (ranked / parallel / fulfill / assigned):")
        await ui.table(
            [
                ("ranked", f"{demo_mod.ranked_expected(input_text, result):.2f}", "expected score (sequential)"),
                ("parallel", f"{demo_mod.parallel_expected(input_text, result):.2f}", "expected score (parallel)"),
                ("fulfill", f"{demo_mod.fulfill_score(input_text, result):.2f}", "fulfillment proxy"),
                ("assigned", f"{demo_mod.assigned_total(input_text, result):.2f}", "ticket-face total cost"),
            ]
        )

        await ui.blank()
        await ui.info(f"sample bundles (first {min(3, len(result))}):")
        for t_str, cs in result[:3]:
            cs_str = ", ".join(cs) if cs else "(none)"
            await _emit(queue, {"type": "line", "ts": _ts(), "level": "SAMP", "text": f"    {t_str!r}  <-  [{cs_str}]"})

        if ok:
            await ui.pass_badge()
        else:
            await ui.fail_badge(msg)

        await _emit(queue, {"type": "case_end", "case": name, "ok": ok})

        summary.append({
            "name": name,
            "groups": len(result),
            "covered": len(covered),
            "tasks": len(tasks),
            "couriers": len(used_c),
            "runtime_ms": elapsed * 1000,
            "ranked": demo_mod.ranked_expected(input_text, result),
            "legal": ok,
        })

    await ui.final_summary(summary)
    await _emit(queue, {"type": "done"})


# ---------------------------------------------------------------------------
# SSE 端点
# ---------------------------------------------------------------------------


@app.get("/api/run")
async def api_run(case: str | None = None) -> StreamingResponse:
    """SSE 流。case=xxx 跑单 case,不传跑全部。"""
    queue: asyncio.Queue = asyncio.Queue()

    async def producer() -> None:
        try:
            if case:
                await run_single_case(queue, case)
            else:
                await run_all_cases(queue)
        except Exception as e:  # noqa: BLE001
            await _emit(queue, {"type": "error", "msg": str(e)})
        finally:
            await _emit(queue, {"type": "stream_end"})

    async def event_gen() -> AsyncIterator[bytes]:
        task = asyncio.create_task(producer())
        try:
            while True:
                payload = await queue.get()
                data = json.dumps(payload, ensure_ascii=False)
                yield f"data: {data}\n\n".encode("utf-8")
                if payload.get("type") in ("done", "stream_end", "error"):
                    break
        finally:
            if not task.done():
                task.cancel()


    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
