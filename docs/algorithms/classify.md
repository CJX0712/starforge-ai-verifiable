# 监督学习算法全集（src/classify.mjs）

> 12 个算法统一接口：`fit(X, y)` / `predict(X)` / `score(X, y)`，零依赖。
> 作者：晨星

## 统一接口约定

| 方法 | 说明 |
|------|------|
| `fit(X: Mat, y)` | 训练；`X` 为样本×特征矩阵，`y` 为标签数组 |
| `predict(X)` | 返回预测标签 `number[]` |
| `predictProba(X)` | 返回概率（`Softmax`/`NB` 提供） |
| `score(X, y)` | 返回准确率 |

## 1. LogisticRegression（二项）

- **数学**：`ŷ = σ(wᵀx + b)`，`σ` 为 sigmoid；最小化交叉熵 `L = −Σ(y log ŷ + (1−y) log(1−ŷ)) + λ‖w‖₂²`，手写梯度 + Adam 更新。
- **复杂度**：`O(N·d·epochs)`。参考 Nelder & Wedderburn 1972（GLM）。
- **坑**：特征未标准化 → 梯度病态；多分类需 One-vs-Rest。
- **不变量**：梯度与数值梯度一致；`predictProba ∈ [0,1]`。

## 2. SoftmaxRegression（多项）

- **数学**：`p_k = exp(w_kᵀx)/Σⱼ exp(w_jᵀx)`，真正 multinomial softmax，`Σ p_k = 1`。
- **坑**：softmax 必须减行最大值防溢出。
- **不变量**：`predictProba` 每行行和 `= 1`（容差 `1e-12`）。

## 3. Ridge / Lasso（回归）

- **Ridge**：闭式解 `ŵ = (XᵀX + λI)⁻¹ Xᵀy`。
- **Lasso**：`min Σ(y−Xŵ)² + λ‖ŵ‖₁`，坐标下降 + 软阈值 `S(z, λ) = sign(z) max(|z|−λ, 0)`；目标函数随迭代单调不增。
- **不变量**：Ridge 残差正交性 `Xᵀ(y − Xŵ) ≈ 0`（最小二乘一阶条件，实测 `≤ 1e-6`）；Lasso 目标单调。

## 4. kNN（惰性学习）

- **数学**：`k` 近邻投票；`weighted` 用 `1/d` 加权。
- **复杂度**：训练 `O(1)`，预测 `O(N·d)`（暴力）。
- **坑**：维度灾难；必须标准化（本项目在 z-score 后调用）。
- **不变量**：`k=1` 训练集准确率 `= 1.0`。

## 5. 朴素贝叶斯（Gaussian / Multinomial / Bernoulli）

- **数学**：条件独立假设 `P(x|y) = Πⱼ P(xⱼ|y)`；贝叶斯 `argmax_y P(y)ΠⱼP(xⱼ|y)`。
- **不变量**：`predictLogProba` 每行 `exp` 后和 `= 1`（后验归一）。

## 6. DecisionTreeClassifier（CART）

- **数学**：递归二分，Gini 不纯度 `G = 1 − Σ pₖ²`；遍历每特征相邻中点阈值选最优分裂。
- **复杂度**：`O(N·d log N)`。参考 Breiman et al. 1984。
- **坑**：过深过拟合（需 `maxDepth`）；连续特征需排序。
- **不变量**：`maxDepth=1` 恰 2 个叶节点；深树训练集 Gini `< 0.1`。

## 7. AdaBoostClassifier（SAMME）

- **数学**：决策树桩 `F(x) = Σₘ αₘ hₘ(x)`，权重更新 `wᵢ ← wᵢ exp(α·I(yᵢ≠h(xᵢ)))`。
- **参考**：Freund & Schapire 1997。
- **不变量**：moons 上训练准确率 `> 0.9`；集成不低于单树桩。

## 8. SVMClassifier（SMO，OvO 投票）

- **数学**：`min ½‖w‖² s.t. yᵢ(wᵀxᵢ+b)≥1`；核技巧 `K(x,z)`；Platt SMO 两变量闭式解 + KKT 互补松弛。
- **参考**：Cortes & Vapnik 1995；Platt 1998；Hsu & Lin 2002（OvO）。
- **坑**：RBF 需先标准化；`C` 过大过拟合。
- **不变量**：支持向量满足 `y·f(x) ≈ 1`；`α ∈ [0, C]`；线性可分集 KKT 误差小。

## 基准结果（seed=42，Iris/Wine z-score + 30% 测试集）

| 数据集 | Softmax | LogReg | DecTree(d=3/5) | KNN(k=7) | GaussNB | SVM(rbf) |
|--------|---------|--------|----------------|----------|---------|----------|
| Iris acc | ≥0.95 | — | 0.978/0.911 | ≥0.95 | ≥0.95 | ≥0.95 |
| Wine acc | ≥0.90 | — | 0.849* | ≥0.90 | ≥0.90 | ≥0.90 |

> *Wine 单棵 CART 天花板约 0.85：加深至 10 层训练准确率可达 100%，但新增分裂纯属训练集过拟合，测试预测不变（详见 BENCHMARKS.md「单树上限」）。

## 可验证不变量（selfCheck 实测 14 项全过）

残差正交性、Lasso 目标单调、NB 后验归一、`k=1` 训练集全对、决策树叶节点数、AdaBoost moons 训练 `>0.9`、SVM 支持向量 KKT、Softmax 行和=1。

---

下一篇：[聚类与降维](cluster.md) · [总索引](../README.md)
