# 优化器层 — SGD / Adam / AdamW / Lion / Muon

> 全部零依赖实现。Muon 用 Newton-Schulz 正交化更新隐藏层权重，对标 2024 年最优优化器理论。
> 作者：晨星

## 1. 直觉

优化器决定「梯度告诉我们要改，但具体怎么改」。SGD 朴素地沿梯度走；Adam 给每个参数自适应学习率；AdamW 把权重衰减从梯度里剥离（修正 Adam 的耦合 bug）；Lion 只用符号方向（内存减半）；**Muon 不直接用梯度，而是把梯度做正交化后再更新**——对隐藏层矩阵，正交化后的更新方向信息量更高，收敛更快。

## 2. 数学核心

**SGD（带动量）**：`v = βv + g; θ ← θ − ηv`

**Adam**（Kingma & Ba 2015）：
```
m = β₁m + (1−β₁)g
v = β₂v + (1−β₂)g²
m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ)
θ ← θ − η·m̂/(√v̂ + ε)
```

**AdamW**（Loshchilov & Hutter 2019）：解耦权重衰减 `θ ← θ − η( m̂/(√v̂+ε) + λθ )`

**Lion**（Chen et al. 2023）：`c = sign(β₁m + (1−β₁)g); θ ← θ − η(c + λθ)`

**Muon**（Jordan et al. 2024）：对二维权重 `G`，先 Newton-Schulz 求其正交近似 `O ≈ NS(G)`，再 `θ ← θ − η·O`。NS 迭代：
```
Z₀ = G / ‖G‖_F
for _ in range(5):
  Z = a Z + b Z Zᵀ Z + c Z Zᵀ Z Zᵀ Z   # 系数 3.4445 / -4.7750 / 2.0315
return Z
```
NS 把任意矩阵推向正交矩阵流形，5 步即可达机器精度（`‖Z Zᵀ − I‖ ≤ 1e-6`）。

## 3. 伪代码（Muon 的 Newton-Schulz 步）

```
function muonStep(G, steps=5):
  Z = G / sqrt(trace(GᵀG) / nRows)
  a, b, c = 3.4445, -4.7750, 2.0315
  for _ in 1..steps:
    A = Z @ Zᵀ
    Z = a*Z + b*A@Z + c*A@A@Z
  return Z                       # ≈ 最近的正交矩阵
```

## 4. 复杂度

| 优化器 | 每步时间 | 额外状态 |
|--------|---------|---------|
| SGD | `O(P)` | `v`（可选） |
| Adam/AdamW | `O(P)` | `m, v` |
| Lion | `O(P)` | `m`（内存减半） |
| Muon | `O(P·k)`（k=5 NS 步 + matmul） | `m, v` + G 的 NS |

## 5. 经典论文

- Kingma & Ba, *Adam: A Method for Stochastic Optimization* (2015)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (2019)
- Chen et al., *Symbolic Discovery of Optimization Algorithms* (Lion, 2023)
- Jordan, Becker, Eless, Mishkin, Lloyd, Garg, Mueller, Kerrigan, Pehlevan, Kirsch, *Muon* (2024)

## 6. 常见坑

- **Adam 权重衰减耦合**：`λ` 应走 AdamW 解耦，否则大学习率下衰减过强。
- **Muon 只适用二维隐藏层矩阵**：对 embedding / 输出头等一维或 `rows>1&&cols>1` 外的张量回退 AdamW（标 `muonExempt`）。
- **NS 系数对归一化敏感**：必须先用 Frobenius 范数归一化再迭代，否则发散。
- **调度器（warmup/cosine）需同步到内部 AdamW 分支**：MiniGPT 的 lm_head 走 Muon 回退路径。

## 7. 可验证不变量（`selfcheck.mjs` 实测）

- 解析解收敛：在凸二次问题上，Adam/AdamW/SGD/Lion 均收敛到理论极小点（残差 `≤ 1e-3`）
- 梯度检验：优化器经 `createOptimizer` 后参数梯度与数值梯度一致
- Muon 正交性：NS 输出 `‖Z Zᵀ − I‖ ≤ 1e-6`
- 确定性：同种子、同初始化的优化轨迹逐位一致
- AdamW 衰减解耦：固定 `g=0` 时参数仍按 `−ηλθ` 收缩

---

返回：[总索引](../README.md)
