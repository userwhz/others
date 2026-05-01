---
name: refactoring
description: 重构代码时使用此 skill。强制执行类型提示、小函数拆分、YAGNI、反过度设计、函数式编程、测试通过等纪律。
---

# 重构 Skill

当用户要求"重构"或"refactor"代码时，严格遵循以下规则。

## 规则

### 1. 类型提示 (Type Hints)

所有函数必须有明确的类型提示，包括：

- 每个参数的类型注解
- 返回值类型注解
- 不允许使用 `Any` 除非确实无法确定类型

```python
# Good
def compute_weight(obs: PauliObservable, shot_count: int, epsilon: float) -> float:
    ...

# Bad
def compute_weight(obs, shot_count, epsilon):
    ...
```

### 2. 小函数 (No Large Functions)

- 如果一个函数超过 **30 行** 或缩进超过 **3 层**，拆成更小的函数
- 每个函数只做一件事
- 提取出的小函数应有清晰的单一职责，命名自解释

```python
# Good — split into focused pieces
def is_qubit_wise_commuting(o1: PauliObservable, o2: PauliObservable) -> bool:
    ...

def build_commuting_groups(observables: list[PauliObservable]) -> list[list[PauliObservable]]:
    ...
```

```python
# Bad — one big function doing everything
def process_all(obs):
    for o in obs:
        for q in o.qubits:
            if q.x and not q.z:
                for other in obs:
                    if ...:  # 4 levels of indentation
                        ...
```

### 3. 不处理未发生的问题 (YAGNI)

- 不要为"以后可能会用到"写代码
- 不要加"预留扩展"的参数
- 不要处理当前调用链中不可能出现的状态
- 只实现当前需求明确要求的逻辑

```python
# Bad — handling a case that never happens in practice
def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        # 当前所有调用方保证 v 向量非零，这个分支永远不会执行
        return v
    return v / norm
```

### 4. 厌恶过度设计

- 不要引入不必要的抽象层、接口、基类、设计模式
- 不要用 class 如果函数就够用
- 不要工厂模式如果 `if/else` 只有两个分支
- 不要为了一行代码的不同引入配置项

```python
# Good — simple function
def bernstein_bound(weight: float, n_hits: int, epsilon: float) -> float:
    ...

# Bad — over-engineered
class WeightFunction(ABC):
    @abstractmethod
    def compute(self, weight: float, n_hits: int, epsilon: float) -> float: ...

class BernsteinBound(WeightFunction):
    def compute(self, weight: float, n_hits: int, epsilon: float) -> float: ...
```

### 5. 函数式编程优先

- 优先使用纯函数（无副作用，不修改入参）
- 用 `list`/`dict`/`set` comprehension 替代显式循环
- 用 `map`/`filter`/`reduce` 替代命令式循环
- 用 `functools.partial` 固化参数而非创建闭包
- 避免类状态（成员变量）除非确实需要跨方法共享可变状态

```python
# Good — functional style
weights = {obs: compute_weight(obs, N, eps) for obs in observables}
sorted_obs = sorted(active, key=lambda o: weights[o], reverse=True)

# Bad — imperative style
weights = {}
for obs in observables:
    w = compute_weight(obs, N, eps)
    weights[obs] = w
```

### 6. 测试通过

重构完成后必须运行测试套件确认没有破坏任何功能：

```bash
uv run pytest
```

- 如果测试失败，必须在交付前修复
- 不允许以"测试过时了"为由跳过失败的测试——要么修复代码，要么确认测试确实需要更新并同步更新

## 执行流程

1. 阅读目标文件/函数，理解现有行为
2. 用 git 保存当前状态（确保可回滚）
3. 按上述 6 条规则逐步重构
4. 运行测试确认通过
5. 如测试失败，修复后重新运行直到通过
