# 技术栈与架构规范

## 基本技术栈

- 语言：Python 3.11+
- 提交入口：根目录 `solver.py`
- 依赖策略：提交文件不依赖第三方包，不依赖本地安装
- 本地测试：pytest
- 核心数据结构：候选组合、任务 bitmask、骑手占用集合
- 核心算法库：贪心、ILP 风格分支定界、统计启发式搜索、组件分治 DP、LLM 直接推理占位策略

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
5. Agent 运行多组任务分区策略。
6. 每个任务分区进入多骑手备选分配阶段。
7. 对完整方案计算期望分数，选择最优方案。
8. 输出前保证任务和骑手无冲突。

## Agent 决策

当前 Agent 是确定性的多策略评估 Agent：

```text
解析数据
  -> 计算元数据
  -> 生成多个任务分区候选解：通用启发式、合单优先、最小组数优先、低意愿专科、组件分治 DP
  -> 运行 seeded/global 两类骑手分配器
  -> 计算期望分数与 assigned 成本代理目标
  -> 保留覆盖最多、代理分最低的方案
  -> 在低意愿/骑手稀缺 case 上做策略邻域探索
```

`branch_bound` 保留在算法库中，但不作为默认提交路径。上一轮评测中 tiny/small 样例的 error 很可能来自小规模默认进入分支定界，因此当前 baseline 优先保证稳定通过。

`llm_direct_reasoning_algorithm` 目前不调用外部模型，因为评测环境无法保证网络和 API Key。它保留算法库入口，并退化为确定性启发式策略，便于后续替换。

## 算法库

| 算法 | 当前实现 | 适用场景 |
| --- | --- | --- |
| `greedy` | 按平均分、总分、意愿排序依次选无冲突候选 | 快速 baseline |
| `branch_bound` | 用 bitmask 做任务覆盖搜索，带时间限制和候选裁剪 | 小规模样例 |
| `heuristic_search` | 多个统计排序规则竞速、专科分区、组件分治 DP、局部搜索、多骑手期望评估择优 | 大规模合单样例 |
| `llm_direct_reasoning` | 评测环境中退化为规则启发式 | 保留 LLM 策略接口 |

`heuristic_search` 后接的局部搜索只接受覆盖任务数不下降且总分更低的替换。当前实现尝试替换 1 个或 2 个已选候选，用同一批任务上的更低分候选组合替代，因此优先降低 `total_score`，不牺牲合法性和覆盖率。

多骑手阶段会对同一个 `task_id_list` 输出多个不重复骑手。期望分数按顺序估计：

```text
E = p1 * score1
  + (1-p1) * p2 * score2
  + ...
  + all_failed_probability * reject_penalty
```

这对应题面中“同一订单可同时指派给多位骑手，最先接起订单的骑手获得订单”的机制。

当前专科分区包括：

- `pair_first`：优先选择二任务合单，减少任务组合数量。
- `minimum_group`：尽量接近理论最少组合数，让有限骑手集中给更少组合做备选。
- `low_willingness`：当平均意愿很低时，额外尝试低失败概率优先的二单覆盖，并允许每组更多备选骑手。
- `hybrid_split`：低意愿场景下比较合单和拆单代理成本，只保留确实划算的合单。
- `component_dp`：把高价值合单边切成小组件，组件内用 bitmask DP 求任务分区，再交给全局骑手分配器。

## 当前线上基线

当前提交在官方评测中完成 `10/10`，平均惩罚分数为 `771.89`。核心收益来自：

- 组件分治 DP：减少任务分区被单一贪心顺序锁死的问题。
- 全局骑手分配：从空方案开始按边际收益分配骑手，而不是固定主骑手后补备选。
- 低意愿拆单：当合单成功率不足时，允许拆成单任务来换更低失败惩罚。
- 自评估循环：每个候选分区都经过完整输出级别评分，再决定保留或丢弃。

详情见 [baseline_evaluation.md](baseline_evaluation.md)。

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
- `examples/blackbox_like/`：按官方 case 名称构造的本地近似 TSV 样例。
- `examples/example_solution.py`：官方 baseline。
- `tests/test_competition.py`：解析、Agent 和策略测试。
- `tests/test_example_cases.py`：统一读取样例文件并校验提交输出。
- `tests/test_solver_submission.py`：提交入口小型构造测例合法性测试。

## 后续扩展

- `scripts/evaluate_solver.py`：本地 blackbox-like 期望分数评估。
- 继续增强局部搜索：扩大替换邻域、增加运行时间自适应和 case 类型调参。
- 新增 `experiments/`：保存策略对比脚本和运行结果。
- 将最终最优策略同步回根目录 `solver.py`。
