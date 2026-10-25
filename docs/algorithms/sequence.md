# 序列模型（src/sequence.mjs）

> 隐马尔可夫模型（前向-后向 / Viterbi / Baum-Welch EM）与束搜索解码，零依赖。
> 作者：晨星

## 统一设计

- 离散观测 HMM：`nStates` 隐状态、`nObs` 观测符号，参数可显式注入或随机行随机化（自动按行归一化为随机随机矩阵）。
- 数值稳定性：前向/后向用 **缩放（scaling）算法**（Rabiner 1989），以 `c[t]` 缩放因子累积对数概率，避免连乘下溢；`logProb = −Σ_t log c[t]`。
- 不变量优先：对数似然单调不减、行归一化概率和为 1、自相似得分=1 等，全部进 `selfCheck`。

## 1. HiddenMarkovModel — 前向-后向（scaled）

- **数学**：观测序列 `O` 下模型的对数概率
  `log P(O) = log Σ_s α_T(s) = −Σ_t log c[t]`
  其中缩放 `α̂_t(s) = α_t(s)/Π_{i≤t} c[i]`，满足每行和 1。
- **后向**：`β̂_t(i) = Σ_j A_ij B_j(O_{t+1}) β̂_{t+1}(j) · c[t]`，不变量 `Σ_i π_i B_i(O_0) β̂_0(i) = 1`。
- **复杂度**：`O(T·N²)`。
- **参考**：Rabiner 1989（HMM 经典 tutorial）。
- **不变量**：`forward`/`backward` 返回的 `logProb` 一致；缩放 `α̂` 每行和 = 1（容差 `1e-12`）。

## 2. Viterbi 解码（最大后验状态路径）

- **数学**：对数域动态规划
  `δ_t(j) = max_i [δ_{t-1}(i) + log A_ij] + log B_j(O_t)`，`ψ_t(j) = argmax_i (·)`；回溯得 `path`，`logProb = max_i δ_T(i)`。
- **复杂度**：`O(T·N²)`。
- **参考**：Viterbi 1967。
- **坑**：朴素连乘下溢 → 一律走对数域。
- **不变量**：`viterbi` 的 `logProb` 应 `≤ forward` 的 `logProb`（一条路径 ≤ 全概率），且 `≥ −∞`；玩具 `toyHmm` 上解码路径稳定可复现。

## 3. Baum-Welch（EM 参数估计）

- **数学**：E 步积累充分统计量 `γ_t(i) = α̂_t(i)β̂_t(i)/c[t]`（对 `i` 求和=1）、`ξ_t(i,j)`（t→t+1 转移）；M 步行归一化更新 `π, A, B`。支持多序列。
- **参考**：Baum et al. 1970（EM 在 HMM 的特例）；`logLikelihood` 单调不增是 EM 定理保证。
- **坑**：分母退化（某状态从未被占据）时保留旧行，避免 NaN；最终 `logLikelihood` 用收敛后参数重算（而非最后一次未计分的 M 步）。
- **不变量**：`llHistory` 单调不减；`transMat`/`emitProb`/`startProb` 每行和 = 1；拟合后 `fitted_=true`。

## 4. beamSearch（确定性束搜索解码）

- **数学**：维护 `beamWidth` 条候选，每步调用 `stepFn(beams)`（接收整个 `beams` 数组，返回每条 beam 的候选列表），按累积对数概率 `logProb` 降序 + 令牌字典序升序确定性剪枝；遇 `endToken` 入完成集，否则截断；最终按 `scoreFn`（缺省 `logProb`）排名。
- **复杂度**：每步 `O(B·C·log B)`，`B=beamWidth`。
- **坑**：非确定性来源（排序并列）→ 用字典序升序做稳定 tie-break，保证同种子逐位可复现。
- **不变量**：返回序列按 `logProb`（或 `scoreFn`）降序；确定性（同输入同输出）；`beamWidth=1` 退化为贪心。

## 基准（seed=42）

| 任务 | 指标 | 结果 |
|------|------|------|
| 缩放前向 vs 暴力前向 | `logProb` 一致性 | ✓（相对误差 ~0） |
| 玩具 HMM 解码 | 路径可复现 | ✓ |
| Baum-Welch | `llHistory` 单调 | ✓ |
| 束搜索 | 返回降序 + 确定性 | ✓ |

## 可验证不变量（selfCheck 全过）

缩放前向对数概率与暴力一致；后向自相似性；Viterbi ≤ 前向且有限；Baum-Welch 对数似然单调、参数行归一；束搜索降序、字典序稳定 tie-break、确定性复现。

---

下一篇：[神经网络与 Transformer](nn.md) · [总索引](../README.md)
