# ✦ StarForge 星炉

**零依赖纯 ESM JavaScript 世界顶级 AI 引擎 —— 从数学内核到可验证认知智能体**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Node](https://img.shields.io/badge/node-%3E%3D18-brightgreen)](https://nodejs.org)
[![依赖](https://img.shields.io/badge/dependencies-0-blue)](package.json)
[![作者](https://img.shields.io/badge/作者-晨星-CJX0712-blueviolet)](https://github.com/CJX0712)

> 作者：**晨星**（CJX0712） · 深圳 · 2026
>
> 一句话：**不复用任何 npm 包，但每一项都站在世界顶级成果之上；每个结论都可执行、可复现、可验证。**

---

## 1. 这是什么

StarForge 是一个从第一性原理实现的完整 AI 引擎：线性代数、反向模式自动微分、优化器（含 2025 SOTA 的 Muon）、经典机器学习全集、从零训练的字符级 GPT、BM25/LSA 检索、强化学习，顶端是一个**不调用任何外部大模型 API** 的神经-符号认知智能体——它给出的每个数值结论都附带一张**现场执行的验证证书**。

三个硬承诺：

| 承诺 | 含义 |
|---|---|
| **零依赖零构建** | 纯 ESM `.mjs`，无 npm 包；Node ≥ 18 直接运行；浏览器可 import 同一份代码 |
| **可复现** | 全部随机性经种子注入（`createRng(42)`），同种子逐位相同的结果 |
| **可验证** | 38+ 项核心不变量 + 每模块 `selfCheck()`，集中在 `src/selfcheck.mjs` 单一事实源，测试/CLI/仪表盘三处复用 |

## 2. 站在世界顶级成果之上（零重写，集大成）

| StarForge 组件 | 直接继承的世界顶级成果 |
|---|---|
| `linalg.mjs` 矩阵计算 | Golub & Van Loan《Matrix Computations》；Trefethen & Bau《Numerical Linear Algebra》 |
| `autodiff.mjs` 反向微分 | Karpathy [micrograd](https://github.com/karpathy/micrograd) 的动态 DAG 架构，推广到二维张量 |
| `optim.mjs` Muon | Keller Jordan Muon（2024）+ Moonshot AI《Muon is Scalable》（arXiv:2502.16982）：Newton-Schulz 五次迭代 (3.4445, −4.7750, 2.0315)，0.2·√max(n,m) RMS 缩放 |
| `nn.mjs` Transformer/GPT | Vaswani et al. 2017（注意力）；Radford 2018 / Karpathy [nanoGPT](https://github.com/karpathy/nanoGPT)（GPT 工程范式）；Xiong et al. 2020（pre-norm） |
| `classify.mjs` SVM | Platt《Sequential Minimal Optimization》1998 + Keerthi 改进（KKT 双阈值工作集） |
| `classify.mjs` 树/提升 | Breiman CART 1984；Freund & Schapire AdaBoost 1997 |
| `cluster.mjs` GMM/PCA | Dempster et al. EM 1977；Arthur & Vassilvitskii k-means++ 2007 |
| `sequence.mjs` HMM | Baum et al. 1970（前向-后向/Baum-Welch）；Viterbi 1967；Lowerre 束搜索 1976 |
| `retrieval.mjs` BM25/LSA | Robertson & Zaragoza 2009（Okapi BM25）；Deerwester et al. 1990（LSA）；Cormack et al. 2009（RRF） |
| `rl.mjs` | Thompson 1933；Auer et al. UCB 2002；Watkins & Dayan Q-learning 1992；Williams REINFORCE 1992；Barto et al. CartPole 1983 |

## 3. 架构

```
L7 认知智能体  agent.mjs       配方规划 + 工具流水线 + 验证证书 + 情景/语义记忆
L6 检索/强化学习               BM25·LSA·RRF ｜ Bandit·Q-learning·CartPole
L5 神经网络  nn.mjs           Transformer 全家桶 + MiniGPT（从零训练）
L4 学习算法                    SVM-SMO·CART·AdaBoost·NB·KNN ｜ KMeans·GMM·PCA ｜ HMM
L3 优化器  optim.mjs           SGD·Adam·AdamW·Lion·Muon(Newton-Schulz)
L2 自动微分 autodiff.mjs       反向模式 AD（梯度检验 ≤ 1e-6）
L1 核心数学 linalg.mjs         解方程·Cholesky·QR·Jacobi·SVD·logsumexp
L0 元认知  selfcheck.mjs       不变量的单一事实源（测试/CLI/仪表盘三处复用）
```

完整设计说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)；1024 项微指令分解见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 4. 五分钟上手

```bash
git clone https://github.com/CJX0712/starforge-ai.git
cd starforge-ai

node --test                      # ① 测试套件（node:test，零配置）
node src/cli.mjs selftest        # ② 全引擎不变量自检
node src/cli.mjs bench           # ③ 基准套件 → bench/results.json
node src/cli.mjs demo            # ④ 认知智能体演示
node src/cli.mjs train-gpt       # ⑤ 从零训练 MiniGPT（~数分钟）
node src/cli.mjs serve           # ⑥ Web 仪表盘 → http://localhost:8787
```

认知智能体直接问答：

```bash
node src/cli.mjs ask "对鸢尾花数据集做主成分分析，报告前两个主成分的累计解释方差"
node src/cli.mjs ask "BM25 的原理是什么"
node src/cli.mjs ask "计算 (3+4)*2^3"
```

## 5. 基准摘要

<!-- BENCH-SUMMARY-START -->
（运行 `node src/cli.mjs bench` 后此处填入实际数字；完整版见 [BENCHMARKS.md](BENCHMARKS.md)）
<!-- BENCH-SUMMARY-END -->

核心层不变量自检（`node src/cli.mjs selftest`）：

```
✅ linalg    10/10    ✅ autodiff 13/13    ✅ optim 9/9    ✅ datasets 6/6
```

## 6. 可验证性：不变量即文档

每个算法都携带**可执行**的不变量，而不是文字承诺。例：

```js
// 梯度检验（autodiff 的金标准）：解析梯度 vs 中心差分数值梯度
const r = gradCheckTensor((A) => A.transpose().matmul(A).sum(), A);
// → maxRelErr = 3.2e-09  ≤ 1e-6 ✅

// Ridge 的一阶条件：最小二乘解必须满足 Xᵀ(y − Xŵ) = 0
residualOrthogonality(X, y, w)  // → 4.1e-13 ✅

// GMM-EM 的单调性：对数似然迭代单调不减
// KMeans：SSE 迭代单调不增；HMM 前向：与暴力枚举一致
// BM25：公开算例精确命中 1.4167；Muon：Newton-Schulz 输出奇异值 ∈ (0.5, 1.6)
```

认知智能体把这些不变量延伸到**运行期**：每个数值结论生成时现场签证。示例输出：

```
━━━ 目标: 对鸢尾花数据集做主成分分析，报告前两个主成分的累计解释方差
  ✅ load   → load_dataset
  ✅ scale  → zscore
  ✅ pca    → pca
  ┈┈ 验证证书 ┈┈
     ✅ 主成分正交归一 — max|VᵀV−I| = 3.2e-16
     ✅ 解释方差比性质 — 累计解释方差 = 95.82%
前 2 个主成分累计解释方差 = 95.82%。……
```

## 7. MiniGPT：浏览器里看得见的从零训练

字符级迷你 GPT（词嵌入 + 可学习位置编码 + 2 层 4 头 pre-norm Transformer + 因果掩码），
在莎士比亚 + 爱丽丝梦游仙境混合语料（约 26 万字符）上从零训练：

```bash
node src/cli.mjs train-gpt --preset tiny   # 训练并导出 web/weights.json
node src/cli.mjs serve                     # 仪表盘 → MiniGPT 实验室，浏览器内实时推理
```

训练代码与 nanoGPT 同构（AdamW/Muon + warmup-cosine + 梯度裁剪 + 融合交叉熵），
可无缝迁移到 GPU 框架；零依赖实现的意义在于**全链路可复现、可审计**。

## 8. 目录结构

```
src/          15 个单文件模块（linalg/verify/autodiff/optim/datasets/classify/
              cluster/sequence/nn/retrieval/rl/kb/agent/selfcheck/bench/cli/train-gpt）
test/         node:test 测试套件（与 selfCheck 同源）
docs/         ARCHITECTURE · REPRODUCE · ROADMAP(1024 微指令) · BENCHMARKS
docs/algorithms/  每个模块的论文级算法文档
bench/        基准结果（results.json）
web/          单文件仪表盘 dashboard.html + 训练权重
tools/        gen-data.mjs（数据生成）· serve.mjs（零依赖静态服务器）
data/         UCI/Gutenberg 原始数据 + 生成的 ES 模块
.github/      CI（Node 18/20/22 × 自检/测试/基准）
```

## 9. 文档

| 文档 | 内容 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 七层架构、模块依赖、关键设计决策 |
| [docs/algorithms/](docs/algorithms/) | 每个算法：直觉/数学/伪代码/复杂度/论文/坑/不变量 |
| [BENCHMARKS.md](BENCHMARKS.md) | 全部基准数字与目标值 |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | 复现步骤、确定性声明、数据来源 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 1024 项微指令级操作分解与完成度 |

## 10. 许可证与署名

MIT License · Copyright © 2026 **晨星**（CJX0712）· [github.com/CJX0712/starforge-ai](https://github.com/CJX0712/starforge-ai)

引用本项目时请署名：**晨星 / StarForge（星炉）**。

---

*启明星 —— 天亮前最后还亮着的那颗。夜里把活干完，天亮交出去的东西必须是真的能用。*
