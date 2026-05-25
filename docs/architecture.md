# 技术栈与架构规范

## 基本技术栈

- 语言：Python 3.11+
- 提交入口：根目录 `solver.py`
- 依赖策略：提交文件不依赖第三方包，不依赖本地安装
- 本地测试：pytest
- 核心数据结构：候选组合、任务 bitmask、骑手占用集合
- 核心算法库：贪心、ILP 风格分支定界、统计启发式搜索、LLM 直接推理占位策略

## 当前架构

```text
官方 TSV 输入
  -> solver.py                 在线提交入口，自包含
  -> autosolver_agent.competition
       -> parse_competition_input
       -> compute_metadata
       -> AutoSolverAgent
       -> CompetitionInstance
       -> CandidateBundle
       -> greedy_algorithm
       -> branch_bound_algorithm
       -> heuristic_search_algorithm
       -> llm_direct_reasoning_algorithm
  -> tests/test_competition.py
输出 list[tuple[str, list[str]]]
```

根目录 `solver.py` 是最终提交对象，必须保持自包含。`src/autosolver_agent/competition.py` 是本地开发版本，用于测试、重构和后续策略沉淀。

## 数据流

1. `input_text` 进入 `solve`。
2. 跳过表头，逐行解析候选。
3. 计算元数据：
   - 候选数量
   - 任务数量
   - 骑手数量
   - 分数均值/最小值/最大值
   - 意愿均值/最小值/最大值
   - 最大合单规模
4. 每条候选解析为：
   - `task_ids`
   - `task_id_list_str`
   - `courier_id`
   - `total_score`
   - `willingness`
5. Agent 根据元数据选择算法。
6. 算法选择若干候选。
7. 输出前保证任务和骑手无冲突。

## Agent 决策

当前 Agent 是轻量规则 Agent：

```text
若无候选：使用 greedy
若存在合单候选：使用 heuristic_search
否则：使用 greedy
```

`branch_bound` 保留在算法库中，但不作为默认提交路径。上一轮评测中 tiny/small 样例的 error 很可能来自小规模默认进入分支定界，因此当前 baseline 优先保证稳定通过。

`llm_direct_reasoning_algorithm` 目前不调用外部模型，因为评测环境无法保证网络和 API Key。它保留算法库入口，并退化为确定性启发式策略，便于后续替换。

## 算法库

| 算法 | 当前实现 | 适用场景 |
| --- | --- | --- |
| `greedy` | 按平均分、总分、意愿排序依次选无冲突候选 | 快速 baseline |
| `branch_bound` | 用 bitmask 做任务覆盖搜索，带时间限制和候选裁剪 | 小规模样例 |
| `heuristic_search` | 多个统计排序规则竞速，选择词典序最优结果 | 大规模合单样例 |
| `llm_direct_reasoning` | 评测环境中退化为规则启发式 | 保留 LLM 策略接口 |

## 约束

```text
每个 task_id 最多出现一次
每个 courier_id 最多出现一次
输出候选必须来自输入候选集
运行时间不超过 10 秒/算例
```

## 目标函数

基础词典序目标：

```text
1. 最大化覆盖任务数
2. 最小化 total_score 总和
3. 最大化 willingness 总和或平均值
```

工程实现中可使用排序 key 或大常数目标近似：

```text
objective = BIG_M * covered_task_count - total_score_sum + alpha * willingness_sum
```

## 仓库规范

- `solver.py`：最终提交入口，必须自包含。
- `src/autosolver_agent/competition.py`：比赛格式解析和本地策略实现。
- `docs/0.md`：最新数据约定。
- `docs/contest_plan.md`：任务理解、分工和里程碑。
- `docs/architecture.md`：当前架构与技术约束。
- `docs/baseline_evaluation.md`：baseline 评测结果和后续优化方向。
- `examples/large_seed301.txt`：官方大样例输入。
- `examples/example_solution.py`：官方 baseline。
- `tests/test_competition.py`：解析、Agent 和策略测试。
- `tests/test_solver_submission.py`：提交入口近似测例合法性测试。

## 后续扩展

- 新增 `src/autosolver_agent/evaluation.py`：本地评估覆盖数、总分、意愿和耗时。
- 新增 `src/autosolver_agent/search.py`：bitmask、beam search、局部搜索。
- 新增 `experiments/`：保存策略对比脚本和运行结果。
- 将最终最优策略同步回根目录 `solver.py`。
