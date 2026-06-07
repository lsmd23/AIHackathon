# 双层多策略自评估派单求解器

> Courier Dispatch Solver — two-decoupled-layer multi-strategy self-evaluating agent.

## 1. 作品简介

本作品针对快递派单组合优化比赛,从官方预计算候选表中选出**任务无冲突、骑手无冲突**的派单方案,实现:

- 100% 任务覆盖(官方 10 个算例全部达成)
- 官方平均惩罚分 **727.85**(线上最佳成绩)
- 端到端运行时间 < 10s/算例

核心方法是一个**解耦的两层 Agent**:

| 层 | 关注 | 主要算子 |
|---|---|---|
| 任务分区(partition) | 把全部任务切成不重叠的 `task_id_list` 组 | 贪心(6 种排序键) / 配对优先 / 组件分治 DP / bitmask beam / 强制配对 / 低意愿拆单 / 混合拆合 |
| 骑手分配(assign) | 给每个组挑一组不重复的骑手 | 备选骑手(主+备) / 全局边际收益 / 预留骑手 |

每一组 (partition, assigner, policy) 都会被 `_policy_score` 统一打分(覆盖任务数 → 代理分 → 票面分),并保留最优。

## 2. 复现命令(3 步)

```bash
pip install pytest            # 仅一个 dev 依赖
python -m pytest tests/      # 合法性 + 完整性回归
python demo.py                # 跑 4 个 case,录屏用
```

烟测(等价于评测平台单算例调用):

```bash
python -c "from pathlib import Path; import solver; print(len(solver.solve(Path('examples/large_seed301.txt').read_text(encoding='utf-8'))))"
# 期望: 40
```

本地代理评测(只有大样例 `large_seed301`):

```bash
python scripts/evaluate_solver.py
```

## 3. 提交文件

- `solver.py`:**唯一提交入口**,自包含,无第三方依赖,2160 行
- `examples/large_seed301.txt`:官方大样例(33780 候选,40 任务,80 骑手)
- `examples/example_solution.py`:官方 baseline 参照
- `tests/`:合法性、完整性、提交入口回归
- `scripts/evaluate_solver.py`:本地代理评估(ranked / parallel / fulfill 三个分)
- `docs/architecture.md`:架构与算法
- `docs/baseline_evaluation.md`:线上分与对照
- `demo.py`:录屏演示脚本

## 4. 算法亮点

1. **两层解耦**:任务分区和骑手分配独立,失败的策略不影响另一层
2. **多策略竞速**:同一 case 同时跑 6-18 组 (partition, assigner, policy) trial,留最优
3. **case 画像 → 策略网格**:`_learning_policies` 根据 courier/task 比和意愿压力分配不同 (α, max_couriers, mode) 组合
4. **稀缺骑手专科**:courier_count / task_count < 1.25 时切换到强制配对 + 收紧备选 + 关闭局部搜索
5. **低意愿专科**:愿低于阈值时拆单主导,备选骑手数扩到 4-6
6. **组件分治 DP**:高价值合单边切成连通组件,组件内 bitmask DP,避免全局位图爆炸
7. **强制配对 DP**:偶数任务时强制 2-2 配对,3 种 cost 模式 × width=700 beam

## 5. 分工(2 人)

| 角色 | 姓名占位 | 职责 | 主要文件 |
|---|---|---|---|
| 队长 / 算法 | 张三 | 任务分区层(partition):6 种贪心排序键、pair_first、minimum_group、budgeted_pair、bitmask DP、组件分治、强制配对、低意愿拆单、混合拆合 | `solver.py` 中 `_greedy_by_key` / `_pair_first_candidates` / `_minimum_group_candidates` / `_budgeted_pairing_candidates` / `_bundle_partition_beam_candidates` / `_component_dp_candidates` / `_scarce_pair_matching_candidates` / `_low_willingness_candidates` / `_hybrid_split_candidates` / `_low_beam_candidates` |
| 工程 / 评测 | 李四 | 骑手分配层(assign):3 个分配器 + 局部搜索;TSV 解析、metadata、case 画像、policy 网格、合法性校验;评测脚本、回归测试、demo | `solver.py` 中 `_parse_input` / `_metadata` / `_is_low_willingness_case` / `_learning_policies` / `_policy_score` / `_assign_backup_couriers` / `_assign_couriers_global` / `_assign_couriers_reserved_global` / `_polish_assignment` / `_local_search`;`scripts/evaluate_solver.py`;`tests/*`;`demo.py` |

## 6. 关键约束与边界

- 单算例 10s 硬限;实测大样例 6-7s,小样例 < 1s
- `reject_penalty = 100 × 任务数`(代码内 `_reject_penalty`)
- 期望分公式(同组):`E = p1*score1 + (1-p1)*p2*score2 + ... + (1-p1)*...*(1-pn)*reject_penalty`
- 全程确定性,无随机种子、无外部 API
- `solver.py` 自包含,提交时不需要任何其他文件

## 7. 评测结果

| 算例 | 任务/骑手 | 线上分 | 状态 |
|---|---:|---:|---|
| `tiny_seed42` | 6/12 | 159.40 | 100% |
| `small_seed100` | 15/30 | 329.11 | 100% |
| `medium_seed201` | 30/60 | 493.14 | 100% |
| `medium_seed202` | 30/60 | 537.69 | 100% |
| `medium_seed203` | 30/60 | 516.82 | 100% |
| `high_noise_seed601` | 30/60 | 494.12 | 100% |
| `large_seed301` | 40/80 | 671.18 | 100% |
| `large_seed302` | 40/80 | 639.70 | 100% |
| `low_willingness_seed501` | 30/70 | 1806.02 | 100% |
| `scarce_couriers_seed401` | 40/38 | 1631.27 | 100% |
| **平均** | | **727.85** | **10/10** |

瓶颈仍在 `low_willingness_seed501`(1806)和 `scarce_couriers_seed401`(1631)两个 case,合计占总失分 ~47%。

## 8. 目录结构(产品交付)

```text
.
├── README.md                          ← 本文件
├── pyproject.toml                     ← 项目元数据
├── requirements.txt                   ← 仅 pytest
├── solver.py                          ← 唯一提交入口(2160 行)
├── demo.py                            ← 录屏演示
├── docs/
│   ├── architecture.md                ← 架构与算法
│   └── baseline_evaluation.md         ← 线上分与对照
├── examples/
│   ├── large_seed301.txt              ← 官方大样例
│   └── example_solution.py            ← 官方 baseline 参照
├── scripts/
│   └── evaluate_solver.py             ← 本地代理评估
└── tests/
    ├── case_validation.py             ← 合法性/完整性校验器
    ├── test_example_cases.py          ← 大样例回归
    └── test_solver_submission.py      ← 提交入口构造测例
```
