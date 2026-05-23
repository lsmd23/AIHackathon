# 技术栈与架构规范

## 基本技术栈

- 语言：Python 3.11+
- 项目管理：`pyproject.toml`
- 测试：pytest
- 核心求解：标准库实现 baseline，后续按需要接入 OR-Tools、PuLP、NumPy
- Agent 编排：本仓库自定义轻量 Agent 控制器，后续可接 LLM API
- 代码风格：类型标注、dataclass、纯函数评分、策略接口隔离

## 架构分层

```text
输入数据
  -> models.py      数据结构
  -> scoring.py     评分与可行性判断
  -> strategies/    具体求解策略
  -> solver.py      策略运行与结果选择
  -> agent.py       Agent 决策、迭代与历史反馈
  -> cli.py         命令行入口
输出方案
```

## 策略接口

所有策略实现统一接口：

- 输入：`ProblemInstance`
- 输出：`Solution`
- 元信息：策略名称、耗时、评分

这样可以在同一个 Agent 中横向比较贪心、ILP、启发式搜索和 LLM 辅助策略。

## 当前仓库规范

- `docs/`：方案、架构、比赛记录。
- `examples/`：小型样例和手工构造边界用例。
- `src/autosolver_agent/`：项目源码。
- `tests/`：单元测试。
- `.gitignore`：忽略虚拟环境、缓存、构建产物。

## 后续扩展建议

- 增加 `experiments/` 存放批量实验脚本。
- 增加 `configs/` 存放策略参数。
- 增加 `outputs/` 存放本地运行结果，但默认不提交 Git。
- 增加 `src/autosolver_agent/strategies/ilp.py` 实现精确优化。
- 增加 `src/autosolver_agent/strategies/local_search.py` 做贪心解后处理。
