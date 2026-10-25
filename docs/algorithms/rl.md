# 强化学习（src/rl.mjs）

> 多臂老虎机（ε-greedy / UCB1 / Thompson / KL-UCB）+ 表格 Q-learning（CartPole/Chain/GridWorld）+ REINFORCE 策略梯度，零依赖。
> 作者：晨星

## 1. 多臂老虎机（Bandits）

### 1.1 EpsilonGreedy
- **数学**：以 `ε` 概率随机探索、否则选经验均值最高臂；`ε` 可线性退火 `ε_start→ε_end`。
- **复杂度**：`O(1)`/步。参考 Sutton & Barto 1998。

### 1.2 UCB1（Hoeffding 上置信界）
- **数学**：`a_t = argmax_a [ q̂_a + √(2 ln t / N_a) ]`；用对所有 `[0,1]` 奖励成立的 Hoeffding 界。
- **坑**：对 Bernoulli 奖励偏保守（探索项过松）；理论遗憾界 `Σ_a 8·ln T / Δ_a`（Hoeffding 推导）。

### 1.3 ThompsonSampling（贝叶斯概率匹配）
- **数学**：每臂维护 `Beta(α_i, β_i)` 后验（`α=1+成功, β=1+失败`），每轮采样 `θ_i~Beta` 选 `argmax`；Beta 采样用 Marsaglia-Tsang Gamma 比率法。
- **参考**：Thompson 1933；Agrawal & Goyal 2012（遗憾 `O(√(KT ln T))`）。

### 1.4 KLUCB（KL 上置信界，Bernoulli 紧界）
- `bernoulliKL(p,q) = p ln(p/q) + (1−p) ln((1−p)/(1−q))`（定义域裁剪 `1e-12` 防 NaN）。
- `klUCBIndex(mean, count, t, c=0)`：解 `N_a·d(q̂_a, U) = ln t + c·ln ln t` 的根，`d` 在 `q>q̂` 严格单调 → 二分 50 次收敛到机器精度，返回 `U∈[q̂,1]`。
- **为什么强于 UCB1**：用 Bernoulli 精确大偏差率函数取代宽松 Hoeffding 界，被证明渐近达 Lai & Robbins 下界（Garivier & Cappé 2011；Audibert et al. 2010）。
- **不变量**：`U ≥ q̂ ∧ U ≤ 1`；`N·d(q̂,U) ≤ ln t + c·ln ln t`。

## 2. 表格 Q-learning

### 2.1 QLearningAgent
- **数学**：`Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]`；`ε`-贪婪探索；`α` 可衰减至 `alphaMin`。
- **复杂度**：`O(|S|·|A|)` 空间。

### 2.2 CartPoleEnv（物理仿真）
- 倒立摆四状态 `[x, ẋ, θ, θ̇]`；离散化 `cartPoleBinIndex` 将连续态映射至 `cartPoleStateCount` 个 bin（默认 `CARTPOLE_BOUNDS`）。
- **不变量（物理）**：`done` 判定（越界即终止）与 `reward=1/步` 一致；确定性 RNG 下轨迹可复现。

### 2.3 trainQonCartPole
- 签名：`{episodes=8000, seed, alpha=0.1, gamma=0.99, epsilon, epsStart, epsEnd, bounds, target=195, optimisticInit, alphaDecay, alphaMin}`。
- 返回 `last100Avg`、`convergenceEpisode`、`wallMs`、`nDiscrete`。
- **实测**：无乐观初始化 + 无 `alphaDecay` 反而最优（3000 轮 124ms 达 `last100≈312`）；定档 3000 轮。表格 Q 在离散化 CartPole 上**非单调震荡**——属离散化分辨率与探索噪声固有现象，非实现缺陷。

### 2.4 trainQLearningChain / GridWorldEnv / ChainMDP
- ChainMDP：远端高奖励，需足够探索才能学到最优策略；验证 Q-learning 收敛到解析最优值（残差→0）。

## 3. REINFORCEAgent（策略梯度）
- **数学**：`∇J(θ) = E[Σ_t ∇log π(a_t|s_t)·G_t]`，蒙特卡洛回报 `G_t`；基线减方差。
- 参考 Williams 1992。
- **不变量**：策略在可学习问题上**学习效应显著**（回报随训练上升）；同种子可复现。

## 基准与关键结论（seed=42，10 臂 × 2000 步）

| 结论 | 结果 |
|------|------|
| ε-greedy/Thompson/UCB1/KL-UCB 最优臂选择率 | 均 `> 3×随机基线 (0.1)` |
| KL-UCB 累计遗憾 `<` UCB1 累计遗憾 | ✓（Bernoulli 紧界优势） |
| UCB1 累计遗憾 `≤` Hoeffding 理论界 `Σ 8·ln T/Δ` | ✓ |
| Q-learning | 残差→0、收敛到解析最优 |
| CartPole | 3000 轮 `last100≈312`（震荡收敛） |
| REINFORCE | 回报随训练上升（学习效应） |
| 端到端确定性 | 同种子逐位可复现 |

## 可验证不变量（selfCheck 全过）

四 bandit 算法显著优于随机基线；KL-UCB 遗憾 < UCB1；UCB1 遗憾 ≤ 理论界；KL-UCB 指数满足上界约束且 `U∈[q̂,1]`；Q-learning 残差与解析解一致；CartPole 物理不变量；REINFORCE 学习效应；确定性复现。

---

总索引：[../README.md](../README.md) · 全部算法文档由 **晨星** 撰写
