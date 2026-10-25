# 自动微分层 — 反向模式自动微分

> 零依赖张量化的 micrograd 架构：每个 `Tensor` 记录运算图，反向传播按拓扑序自动求导。
> 作者：晨星

## 1. 直觉

微积分里求梯度是「手动套公式」，自动微分是「让计算机把任意计算图的导数算对」。本项目用 **micrograd（Karpathy）的标量反向传播架构**扩展为**全张量版本**：前向计算时把运算记录在图上，调用 `backward()` 后沿逆拓扑序把梯度从输出一路传回每个参数。这样写神经网络时只需描述前向，反向传播自动完成——且梯度与解析解逐位一致（梯度检验 ≤1e-6 为硬标准）。

## 2. 数学核心

**链式法则（反向模式）**：设 `y = f(x)`，`L = g(y)`，则

```
∂L/∂x = ∂L/∂y · ∂y/∂x
```

对每个算子实现其「局部雅可比-向量积」（VJP）：前向存 `a, b → out`，反向时 `grad_a += grad_out · ∂out/∂a`。

| 算子 | 局部梯度 |
|------|---------|
| `a + b` | `∂/∂a = 1`, `∂/∂b = 1` |
| `a * b` | `∂/∂a = b`, `∂/∂b = a` |
| `matmul(A, B)` | `∂L/∂A = ∂L/∂out · Bᵀ`, `∂L/∂B = Aᵀ · ∂L/∂out` |
| `softmaxRows(x)` | `∂L/∂xᵢ = Σⱼ Sᵢⱼ(∂L/∂oⱼ − Σₖ Sᵢₖ ∂L/∂oₖ)`（Jacobian-向量积） |
| `sigmoid(x)` | `∂/∂x = s(1−s)` |
| `gelu(x)` | 精确解析导数（tanh 近似形式） |

**梯度检验（黄金标准）**：对每个参数 `θ`，中心差分 `∂f/∂θ ≈ (f(θ+ε) − f(θ−ε)) / (2ε)`，与自动梯度比对，相对误差 `≤ 1e-6`。

## 3. 伪代码

```
class Tensor:
  data: Float64Array           # 数值
  grad: Float64Array = 0       # 累积梯度
  _back: 闭包                 # 反向传播闭包

function backward(t, grad=1):
  topo = 拓扑排序(t)            # 从 t 沿 _prev 深搜
  t.grad = grad
  for node in reversed(topo):   # 逆序
    node._back(node.grad)

# 例：c = (a*b).relu()
c._back = (g) => {
  a.grad += g * b.data        # 乘法局部梯度
  b.grad += g * a.data
}
```

## 4. 复杂度

- 前向：`O(OPS)`（一次遍历计算图）
- 反向：与前向同阶 `O(OPS)`（反向模式对任意输出数只多一次）
- 空间：`O(图节点数)`（缓存中间值）

## 5. 经典论文 / 项目

- Baydin et al. 2018, *Automatic Differentiation in Machine Learning: a Survey*
- Karpathy, micrograd（标量反向传播最小实现，本项目架构蓝本）
- Griewank & Walther, *Evaluating Derivatives*（AD 经典教材）

## 6. 常见坑

- **忘记 `zeroGrad()`**：梯度跨步累积，每个 step 前必须清零。
- **in-place 修改 `data`**：会切断计算图，反向传播失准。
- **softmax 直接算 `exp` 溢出**：必须减行最大值。
- **广播维度错误**：`[r,1] + [r,c]` 与 `[1,c] * [r,c]` 的梯度形状需对齐，本项目统一用 `broadcastAdd`/`broadcastMul`。

## 7. 可验证不变量（`selfcheck.mjs` 实测）

- 梯度检验：matmul / softmaxRows / LayerNorm / GELU / 切片拼接 / 广播 / 一元链，相对误差 `≤ 1e-6`
- 交换律：`(a*b).backward()` 与 `(b*a).backward()` 梯度一致
- 拓扑序正确：`backward()` 每个节点恰好访问一次，无重复累积
- 纯函数组合（exp→log→sqrt→tanh→sigmoid→pow）后梯度仍与数值吻合

---

下一篇：[优化器](../optim.md) · [总索引](../README.md)
