# StarForge 架构文档

> 作者：晨星 · 项目：StarForge（星炉） · 许可证：MIT

## 1. 设计哲学

三条第一性原理，贯穿全部代码：

1. **零重写，集大成。** 不复用任何 npm 包，但每个算法都直接继承世界顶级成果：
   矩阵计算继承 Golub & Van Loan；反向微分继承 Karpathy 的 micrograd 架构并推广到张量；
   Muon 优化器继承 Keller Jordan 的 Newton-Schulz 迭代与 Moonshot 的 RMS 匹配缩放；
   Transformer 继承 Vaswani 的注意力与 Radford/Karpathy 的 GPT 工程范式；
   BM25 继承 Robertson & Zaragoza；SVM 求解继承 Platt 的 SMO；HMM 继承 Baum 的 EM。
2. **可验证优于可解释。** 每个算法至少携带一条**可执行**不变量（而非文字说明），
   集中在 `src/selfcheck.mjs`，被测试套件、CLI 自检、Web 仪表盘三处复用。
3. **确定性即复现。** 所有随机性必须经 `createRng(seed)` 注入；同种子 ⇒ 同结果。
   数据集、初始化、批次采样、环境动力学全部遵守。

## 2. 七层架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│ L7 认知智能体  agent.mjs                                          │
│    意图规划器 → 工具流水线 → 元认知签证 → 答案 + 证书 + 置信度      │
│    记忆：情景（episodic JSONL）+ 语义（fact table）                │
├──────────────────────────────────────────────────────────────────┤
│ L6 检索 / 强化学习                                                │
│    retrieval.mjs: 分词 · TF-IDF · BM25 · LSA(SVD) · RRF 融合      │
│    rl.mjs:        Bandit · UCB1 · Thompson · Q-learning ·         │
│                   REINFORCE · CartPole（经典动力学）              │
├──────────────────────────────────────────────────────────────────┤
│ L5 神经网络  nn.mjs                                               │
│    Linear · LayerNorm · Dropout · Embedding · MultiHeadAttention  │
│    TransformerBlock(pre-norm) · MiniGPT(字符级 GPT) · Trainer     │
├──────────────────────────────────────────────────────────────────┤
│ L4 学习算法                                                       │
│    classify.mjs:  Logistic/Softmax · Ridge/Lasso · KNN · NB ·     │
│                   CART · AdaBoost · SVM-SMO(线性/RBF, OvO)        │
│    cluster.mjs:   KMeans++ · GMM-EM · PCA(Jacobi) · 层次聚类      │
│    sequence.mjs:  HMM 前向/后向/Viterbi/Baum-Welch · 束搜索       │
├──────────────────────────────────────────────────────────────────┤
│ L3 优化器  optim.mjs                                              │
│    SGD(动量/Nesterov) · Adam · AdamW · Lion · Muon(NS 正交化)      │
│    全局梯度裁剪 · warmup+cosine/linear 调度                       │
├──────────────────────────────────────────────────────────────────┤
│ L2 自动微分  autodiff.mjs                                         │
│    动态 DAG · 拓扑排序反向传播 · 广播元素级 · matmul · softmax      │
│    融合交叉熵 · 梯度检验（金标准 rel err ≤ 1e-6）                  │
├──────────────────────────────────────────────────────────────────┤
│ L1 核心数学  linalg.mjs                                           │
│    Mat · 解方程(部分选主元) · Cholesky · Householder QR ·          │
│    Jacobi 特征分解 · SVD · logsumexp · 确定性 RNG                  │
├──────────────────────────────────────────────────────────────────┤
│ L0 元认知  verify.mjs + selfcheck.mjs                             │
│    梯度检验 · 正交性 · 残差 · 单调性 · 概率守恒 —— 单一事实源      │
└──────────────────────────────────────────────────────────────────┘
```

## 3. 模块依赖图

```
selfcheck.mjs ──→ linalg · autodiff · optim · datasets · classify · cluster · sequence · nn · retrieval · rl
cli.mjs       ──→ selfcheck · bench · agent · train-gpt
agent.mjs     ──→ linalg · verify · kb · datasets
                 └─(惰性 import)→ classify · cluster · sequence · retrieval · nn
bench.mjs     ──→(惰性 import)→ 各算法模块的 bench()
train-gpt.mjs ──→ nn · datasets · linalg
dashboard.html──→(浏览器 import)→ 全部 src 模块 + bench/results.json + web/weights.json
```

**惰性 import 的意义**：认知智能体的工具按需加载算法模块——某个模块缺失只影响对应工具，
不拖垮整个引擎；这也是"自治系统不停摆"的工程体现。

## 4. 关键设计决策

| 决策 | 理由 |
|---|---|
| 纯 ESM `.mjs`，零 npm 依赖 | 复现成本归零：Node ≥ 18 即可运行全部功能；浏览器可直接 import 同一份代码 |
| 张量限二维 `[rows, cols]` | 覆盖全部模型所需（批次维循环处理），广播规则简化，梯度实现可被完整检验 |
| LayerNorm/GELU 用原语组合 | 复合运算的反向传播由链式法则自动生成，杜绝手写反向的符号错误 |
| Muon 只作用于二维隐藏矩阵 | 遵循 Jordan/Moonshot 约定：embedding、输出头、归一化增益走 AdamW |
| 智能体用"配方"而非自由规划 | 无外部 LLM 时，确定性配方保证可复现与可审计；配方即文档化的专家流程 |
| 每个数值结论带证书 | 把"可验证"从测试期延伸到运行期：答案的不变量在生成时现场执行 |
| 数据以 ES 模块内置 | `data/*.mjs` 由 `tools/gen-data.mjs` 从公开数据集生成，浏览器/Node 双端零 IO |

## 5. 数据流：一个目标的完整旅程

以 `node src/cli.mjs ask "对鸢尾花数据集做主成分分析"` 为例：

```
目标字符串
  → L4 规划器匹配配方 pca-analysis（确定性）
  → 工具 load_dataset（L5 行动层）→ Iris 150×4
  → 工具 zscore → 标准化矩阵
  → 工具 pca（L4 算法层，协方差 → Jacobi 特征分解）
  → L6 元认知验证器：主成分正交性 + 解释方差比性质 → 验证证书
  → L7 答案模板：中文结论 + 数字 + 证书引用
  → L2 记忆层：写入情景记忆
```

## 6. 文件地图

```
src/
  linalg.mjs      L1 核心数学
  verify.mjs      L0 不变量原语
  autodiff.mjs    L2 自动微分
  optim.mjs       L3 优化器（含 Muon）
  datasets.mjs    数据集/指标/语料
  classify.mjs    L4 监督学习
  cluster.mjs     L4 聚类降维
  sequence.mjs    L4 序列模型
  nn.mjs          L5 神经网络 + MiniGPT
  retrieval.mjs   L6 检索
  rl.mjs          L6 强化学习
  kb.mjs          内置知识库（智能体语义记忆底座）
  agent.mjs       L7 认知智能体
  selfcheck.mjs   L0 不变量单一事实源
  bench.mjs       基准汇总器
  cli.mjs         命令行入口
  train-gpt.mjs   MiniGPT 训练入口
test/             node:test 测试套件（与 selfcheck 同源）
docs/algorithms/  每个模块的算法论文级文档
bench/            基准结果（results.json）
web/              仪表盘 + 训练权重
tools/            gen-data.mjs（数据生成）· serve.mjs（静态服务器）
data/             原始数据（UCI/Gutenberg）与生成的 ES 模块
```
