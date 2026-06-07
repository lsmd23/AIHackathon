# Frontend Demo (Vue 3 + FastAPI)

一个 Vue3 单页前端,实时流式展示 `solver.py` 求解 5 个 case 的日志。

## 架构

```
Vue 3 (Vite, 端口 5173)
  ↓ SSE (text/event-stream)
FastAPI (uvicorn, 端口 8000)
  ↓ 调用
solver.py + demo.py
```

Vite dev 代理 `/api/*` 到 FastAPI,所以前端代码只用相对路径。

## 启动

### 1. 启动后端(conda 环境)

```bash
conda run -n meituan-takeway-comp \
  uvicorn --app-dir frontend/backend api:app --host 127.0.0.1 --port 8000
```

> `--app-dir frontend/backend` 让 uvicorn 从 `frontend/backend/` 找 `api.py`。

### 2. 启动前端(pnpm)

```bash
cd frontend
pnpm install
pnpm dev
```

浏览器打开 http://127.0.0.1:5173

## 接口

| 路径 | 用途 |
|---|---|
| `GET /api/ping` | 健康检查 |
| `GET /api/cases` | 列出 5 个 case |
| `GET /api/run` | SSE 流,跑全部 5 个 case |
| `GET /api/run?case=tiny` | SSE 流,跑单个 case |

## SSE 事件类型

```json
{"type": "line",      "ts": "HH:MM:SS", "level": "INFO", "text": "..."}
{"type": "case_start","case": "tiny (6 tasks / 12 couriers)", "idx": 1, "total": 5}
{"type": "case_end",  "case": "...", "ok": true}
{"type": "heartbeat", "elapsed_ms": 1234, "tick": 6}
{"type": "done"}
```

`level` 取值:
- `INFO` 普通日志
- `OK` 成功
- `FAIL` 失败
- `HR` 水平分隔线
- `HEAD` 大标题
- `TBL` 表格行
- `SAMP` 样本分组
- `BADP` `BADF` 合法性 ✓/✗
- `BLNK` 空行

## 录屏建议

录屏脚本:`python demo.py` 这一版已经保留(老 demo 跑法,控制台版)。
新前端 demo:浏览器全屏,窗口宽度 ≥ 1280px,字体不小于 13px。

点 **Run All** 后,日志会从 1/5 一行行推过来,大约 12 秒跑完。
