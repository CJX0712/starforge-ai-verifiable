# 算法文档索引

> 作者：晨星 · StarForge（星炉）
>
> 每个模块的文档遵循统一格式——这是"世界顶级文档"的底线标准：
> **直觉 → 数学核心（公式）→ 伪代码 → 复杂度 → 经典论文 → 常见坑 → 可验证不变量**。
> 最后一节不是装饰：文中每条不变量都在 `src/selfcheck.mjs` 与各模块的 `selfCheck()` 中可执行。

| 模块 | 文档 | 覆盖算法 |
|---|---|---|
| 核心数学 | （见 src/linalg.mjs 头注释） | 矩阵运算、高斯消元、Cholesky、Householder QR、Jacobi 特征分解、SVD、logsumexp |
| 自动微分 | （见 src/autodiff.mjs 头注释） | 反向模式 AD、动态 DAG、拓扑排序、广播、梯度检验 |
| 优化器 | （见 src/optim.mjs 头注释） | SGD 动量、Adam、AdamW、Lion、Muon（Newton-Schulz） |
| 监督学习 | [classify.md](./classify.md) | 逻辑回归、Softmax 回归、Ridge、Lasso、kNN、朴素贝叶斯、CART、AdaBoost、SVM-SMO |
| 聚类降维 | [cluster.md](./cluster.md) | KMeans++、GMM-EM、PCA、层次聚类 |
| 序列模型 | [sequence.md](./sequence.md) | HMM 前向/后向、Viterbi、Baum-Welch、束搜索 |
| 神经网络 | [nn.md](./nn.md) | Linear、LayerNorm、Dropout、Embedding、多头注意力、Transformer、MiniGPT |
| 检索 | [retrieval.md](./retrieval.md) | 分词（CJK）、TF-IDF、BM25、LSA、RRF 融合 |
| 强化学习 | [rl.md](./rl.md) | ε-greedy、UCB1、Thompson 采样、Q-learning、REINFORCE、CartPole |

## 不变量总表（部分）

| 算法 | 不变量 | 类型 |
|---|---|---|
| 全部优化器 | 凸二次问题上收敛到正规方程解析解 | 解析解 vs 数值解 |
| autodiff 每个算子 | 解析梯度 vs 中心差分数值梯度 rel err ≤ 1e-6 | 参照实现 |
| Jacobi 特征分解 | ‖Av−λv‖ ≤ 1e-8，VᵀV=I | 残差 + 正交 |
| Householder QR | QᵀQ=I，QR=A，R 上三角 | 重构 |
| SVD | A≈UΣVᵀ，U 列正交 | 重构（Eckart-Young） |
| Ridge | Xᵀ(y−Xŷ)≈0 | 一阶条件 |
| Lasso | 目标函数迭代单调不增 | 单调性 |
| SVM-SMO | 支持向量 y·f(x)≈1；KKT 互补松弛 | KKT |
| KMeans | SSE 迭代单调不增 | 单调性 |
| GMM-EM | 对数似然单调不减；责任度行和=1 | 单调性 + 守恒 |
| PCA | 主成分正交；解释方差比降序且累计≤1 | 结构 |
| HMM 前向 | α 之和 = P(O)，与暴力枚举一致 | 暴力枚举 |
| Viterbi | 路径合法且概率 ≥ 任一枚举路径 | 最优性 |
| BM25 | 公开算例精确命中（1.4167） | 参照实现 |
| Newton-Schulz | 输出奇异值 ∈ (0.5, 1.6) | 近似正交 |
| Q-learning | Bellman 残差 → 0 | 不动点 |
| MiniGPT | 因果掩码：扰动未来不影响过去；单批量过拟合 loss→0 | 结构 + 极限 |
