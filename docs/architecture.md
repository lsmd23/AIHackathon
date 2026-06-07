# 架构与算法

## 1. 问题定义

官方输入是一张预计算候选表 TSV:

```text
task_id_list     courier_id   total_score   willingness
T0037,T0039      C028         52.016        0.582
T0012            C073         18.5          0.91
```

每行 = 一个候选派单方案(一个或多个任务组 + 一个骑手 + 票面成本 + 接单概率)。

提交函数:

```python
def solve(input_text: str) -> list[tuple[str, list[str]]]:
    ...
```

返回 `[("T0012", ["C073"]), ("T0037,T0039", ["C028", "C091"])]`。

**硬约束**(踩到就 0 分):
- 一个 `task_id` 在所有 `task_id_list` 里**最多出现一次**
- 一个 `courier_id` 在所有骑手列表里**最多出现一次**
- 每个 `(task_id_list_str, courier_id)` 必须**在输入里存在过**
- 单算例 < 10s(超时即 3000 分惩罚)

**评分**:同 `task_id_list` 可同时派多个骑手,**第一个接单的骑手拿到该任务组**;全失败则该组所有任务都按 `reject_penalty = 100 × 任务数` 罚。

期望分近似公式:

```
E = p1*score1 + (1-p1)*p2*score2 + ... + (1-p1)*...*(1-pn)*reject_penalty
```

`pi = willingness_i`,`scorei = total_score_i`。

**目标**(越低越好):
1. 最大化覆盖任务数
2. 覆盖相同时最小化 `total_score` 总和
3. tie-breaker:`willingness` 总和越大越好

## 2. 数据规模

以官方大样例 `examples/large_seed301.txt` 为例:

| 指标 | 数值 |
| --- | --- |
| 候选行数 | 33,780 |
| 任务数 | 40 |
| 骑手数 | 80 |
| 单任务候选 | 3,200 |
| 双任务合单候选 | 30,580 |
| 时间限制 | 10s |

10 个算例规模范围:6~40 任务、12~80 骑手、几千~33k 候选。

## 3. 总体架构:两层解耦 Agent

```text
parse_competition_input(input_text)
  → compute_metadata(任务/骑手/候选数/分数/意愿分布/最大合单规模/score_spread)
  → case 画像:
        is_scarce = courier_count / task_count < 1.25
        is_low    = _is_low_willingness_case(...)
  → [第 1 层] 任务分区 candidates = list of selected bundles
        ├── greedy_by_key(6 种排序键)
        ├── pair_first_candidates
        ├── minimum_group_candidates
        ├── budgeted_pairing_candidates
        ├── bundle_partition_beam_candidates (bitmask DP, 多 config)
        ├── component_dp_candidates (按合单边切组件, 组件内 bitmask DP)
        ├── scarce_pair_matching_candidates (强制 2-2 配对)
        ├── low_willingness_candidates
        ├── hybrid_split_candidates
        └── low_beam_candidates
  → [第 2 层] 骑手分配:对每个 partition 跑 1-3 个 assigner
        ├── _assign_backup_couriers (主+备)
        ├── _assign_couriers_global (从空按边际收益贪心)
        └── _assign_couriers_reserved_global (预留少量骑手)
  → 统一打分 _policy_score(覆盖数 → -objective → -primary_score)
  → 普通 case 跑 _local_search (1 替 1 / 1 替 2 / 2 替 2)
  → 输出最优
```

**为什么解耦**:
- 任务分区和骑手分配在数学上是两层不同难度的子问题(组合 vs 匹配)
- 一层失败不会污染另一层,容易定位回归
- 可以分别加新策略,互不干扰

## 4. 任务分区策略详解

### 4.1 贪心(6 种排序键)

按下列 key 排序后贪心选无冲突候选:

1. `(score/task_count, score, -willingness, ...)` — 平均成本优先
2. `(bundle_expected_score/task_count, fail_prob, score, ...)` — 期望分优先
3. `(expected_score/task_count - mean_will × success_prob, ...)` — 期望 vs 票面
4. `(score/task_count - mean_will × willingness, ...)` — 票面修正
5. `(score/task_count - rarity(task), -bundle_size, ...)` — 稀有任务优先
6. `(-bundle_size, score/task_count, ...)` — 大合单优先

### 4.2 pair_first_candidates

优先 2 任务合单,减少任务组数量。适合骑手稀缺的 case(组少则单组备选压力小)。

### 4.3 minimum_group_candidates

在骑手数受限 case 下尽量压低任务组数,让有限骑手集中给更少组合做备选。

### 4.4 budgeted_pairing_candidates

按预算配对,组数受 courier_count 直接约束。

### 4.5 bundle_partition_beam_candidates(bitmask DP)

按 task set 拆 DP,每步选 pivot 任务,尝试加入和它配对或自身的 task set,beam 宽度 `width ∈ {350, 450, 550, 650, 750, 850, 950}`,`limit ∈ {2, 3, 4, 6, 8}`(每组最多骑手数),`pair_bonus ∈ {0, 40, 45, 60, 80}`(对配对的减分),`metric ∈ {parallel, expected}`。

多种 config 同时跑,输出多个候选分区。

### 4.6 component_dp_candidates

按高价值合单边构造连通组件,把"任务邻接图"切成若干小组件,组件内跑 bitmask DP。再把所有组件的最优分区拼成全局。

### 4.7 scarce_pair_matching_candidates(稀缺 case 专用)

强制 2-2 配对,3 种 cost 模式:

| mode | 选什么 | cost |
|---|---|---|
| `primary` | 最便宜 1 个 | `total_score` |
| `expected` | 备选 2 个里取期望 | `_expected_bundle_score` |
| `accept` | 高 willingness 优先 2 个 | `_expected_bundle_score` |

每种 mode 跑 `width=700` beam,最多保留 4 个解。

只处理任务数为偶数 case,奇数直接返回。

### 4.8 low_willingness_candidates(低意愿专科)

低意愿 case 下偏向拆单 + 每个组多塞备选骑手。

### 4.9 hybrid_split_candidates

合单 vs 拆单成本对比,只保留**期望成本更小**的合单。

### 4.10 low_beam_candidates

精简 beam 搜索,适合低意愿 case 减小耗时。

## 5. 骑手分配策略详解

### 5.1 _assign_backup_couriers

每个 partition 组先放主骑手(partition 时挑出的),再贪心挑备选(边际收益最高)。

### 5.2 _assign_couriers_global(核心)

**从空方案开始**,对所有组的空集合,反复挑「加入候选骑手后该组目标值下降最多」的决策。贪心直到所有组都达到 `max_couriers` 或没有正收益。

**和 backup 的本质区别**:不固定主骑手,允许两组"互让"一个候选骑手(谁用谁收益大)。

### 5.3 _assign_couriers_reserved_global

为高价值组预留 1-2 个骑手名额,避免被通用 marginal 抢走。

## 6. 统一打分:_policy_score

```text
key = (covered_task_count, -objective, -primary_score)
```

其中 `objective` 由 `mode` 决定:
- `parallel`:同组期望分之和
- `expected`:更复杂的递归期望
- `assigned`:`total_score` 总和 × α
- `balanced`:`parallel + 0.15 × assigned × α`
- `adaptive`(默认):
  - 普通 case:`objective = parallel`
  - 稀缺/低意愿 case:`objective = max(parallel, assigned × α)`

α 和 `max_couriers` 由 `_learning_policies` 根据 case 画像决定。

## 7. case 画像与策略网格

```python
is_scarce = courier_count / task_count < 1.25
is_low    = _is_low_willingness_case(candidates, meta)
```

### 7.1 稀缺 case(courier/task < 1.25)

策略网格(`_learning_policies`):

```python
[
    {"alpha": 0.85, "max_couriers": 2, "mode": "parallel"},
    {"alpha": 1.0,  "max_couriers": 2, "mode": "parallel"},
    {"alpha": 0.85, "max_couriers": 2, "mode": "adaptive"},
    {"alpha": 1.0,  "max_couriers": 2, "mode": "adaptive"},
    {"alpha": 1.2,  "max_couriers": 2, "mode": "adaptive"},
    {"alpha": 1.5,  "max_couriers": 2, "mode": "assigned"},
]
```

分配器:`(backup, global, reserved_global)`(3 个全跑)。

**关闭 local search**(避免误改关键骑手)。

**目标函数**:`max(parallel, assigned × α)`(不签贵方案)。

### 7.2 低意愿 case

策略网格(10 组,`max_couriers ∈ {4, 6}`,`α ∈ {0.35, 0.5, 0.55, 0.65, 0.85}`),`mode ∈ {parallel, expected, adaptive, balanced}`。

分配器:`(backup, reserved_global)`。

**目标函数**:同样 `max(parallel, assigned × α)`。

### 7.3 普通 case

策略网格精简(`max_couriers=4`),`mode ∈ {parallel, expected}`。

分配器:`(backup, global)`。

**开启 local search**(`_local_search`, 1 替 1 / 1 替 2 / 2 替 2, time limit 0.25~0.7s)。

## 8. 局部搜索 _local_search

只对普通 case 启用,接受**覆盖任务数不下降且总分更低的替换**。

- 1 替 1:替换一条候选
- 1 替 2:一条 → 两条(覆盖相同任务,总成本更低)
- 2 替 2:两条 → 两条

**关闭条件**:低意愿和稀缺 case,因为替换可能误用关键骑手。

## 9. 复现步骤

```bash
pip install pytest

# 跑测试
python -m pytest tests/ -v

# 跑大样例
python -c "from pathlib import Path; import solver; print(len(solver.solve(Path('examples/large_seed301.txt').read_text(encoding='utf-8'))))"
# 期望: 40

# 跑本地代理
python scripts/evaluate_solver.py

# 跑录屏
python demo.py
```

## 10. 不做的事(避免)

- 不调用外部 LLM/网络/API
- 不引入第三方依赖
- 不使用随机种子(保持确定性)
- 不在稀疏/低意愿 case 跑 local search
- 不在普通 case 跑 scarce_pair_matching
