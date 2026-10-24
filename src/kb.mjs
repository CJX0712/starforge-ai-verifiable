/**
 * @file kb.mjs — StarForge 内置知识库（自描述文档集）
 * 作者：晨星
 *
 * 知识库既是检索模块的评测语料，也是认知智能体的"语义记忆"底座：
 * 智能体可以用自然语言查询本项目自身的任何设计与算法细节。
 */

export const KB_DOCS = [
  {
    id: 'starforge-intro',
    title: 'StarForge 星炉项目总览',
    text:
      'StarForge（星炉）是由晨星发起的零依赖纯 ESM JavaScript 人工智能引擎。' +
      '项目立场是零重写与集大成：不复用任何 npm 包，所有算法从第一性原理实现，' +
      '但每一项都直接继承世界顶级成果——Golub 的矩阵计算、Karpathy 的反向模式自动微分、' +
      'Jordan 的 Muon 优化器、Vaswani 的 Transformer、Robertson 的 BM25。' +
      '引擎覆盖七层：核心数学层、自动微分层、学习算法层、神经网络层、序列模型层、检索层、强化学习层，' +
      '顶端是带验证证书的认知智能体。全部算法携带可执行不变量，测试套件与命令行自检复用同一份代码。',
  },
  {
    id: 'linalg-core',
    title: '核心数学层：线性代数与数值内核',
    text:
      '核心数学层实现稠密矩阵类 Mat（行主序 Float64Array）、矩阵乘法（i-k-j 循环序加零值跳过）、' +
      '高斯消元解线性方程组（部分选主元）、Cholesky 分解、Householder QR 分解、' +
      '循环 Jacobi 旋转求对称矩阵特征分解、经 A 转置 A 特征分解的薄 SVD、logsumexp 稳定技巧。' +
      '不变量包括正交矩阵 Q 转置 Q 等于 I、特征残差趋近机器精度、SVD 重构误差趋近机器精度。' +
      '随机数使用 mulberry32 加 Box-Muller 的确定性生成器，同种子同序列，保证全程可复现。',
  },
  {
    id: 'autodiff-layer',
    title: '自动微分层：反向模式自动微分',
    text:
      '自动微分层把 Karpathy micrograd 的标量级反向模式自动微分推广到二维张量。' +
      '每个 Tensor 记录数据、梯度、父节点和局部反向闭包；backward 方法先拓扑排序，' +
      '把损失梯度种子置一，再逆序执行闭包，沿计算图所有路径累加梯度。' +
      '支持广播元素级运算、矩阵乘法、softmax、融合交叉熵、均方误差、查表与切片拼接。' +
      'LayerNorm 与 GELU 等复合运算由原语自动组合，避免手写反向传播出错。' +
      '正确性金标准是梯度检验：每个算子的解析梯度与中心差分数值梯度的最大相对误差不超过一亿分之一。',
  },
  {
    id: 'optim-layer',
    title: '优化器层：从 AdamW 到 Muon',
    text:
      '优化器层实现 SGD 动量、Adam、AdamW 解耦权重衰减、Lion 符号化更新，' +
      '以及 2025 年最先进的 Muon：对二维隐藏权重矩阵先做 Nesterov 动量，' +
      '再用牛顿-舒尔茨五次迭代把更新矩阵替换为最近的半正交矩阵，迭代系数为 3.4445、-4.7750、2.0315，' +
      '更新按 0.2 倍根号 max(行数,列数) 缩放以匹配均方根；embedding 与输出头等参数回退到 AdamW。' +
      'Muon 在纳米 GPT 速度赛上把训练时间压缩到 AdamW 的百分之六十五，' +
      '月之暗面据此训练了万亿参数的 Kimi K2。本引擎在零依赖 JavaScript 中完整复现了这一优化器族。',
  },
  {
    id: 'classify-algos',
    title: '监督学习算法全集',
    text:
      '监督学习模块实现逻辑回归、多项 softmax 回归、Ridge 闭式解、Lasso 坐标下降、' +
      'k 近邻、高斯/多项式/伯努利朴素贝叶斯、CART 决策树、AdaBoost 决策桩，' +
      '以及 Platt 序贯最小优化 SVM，支持线性核与 RBF 核，多分类用一对一投票。' +
      '基准：鸢尾花数据集标准化后测试准确率不低于 0.95，葡萄酒数据集不低于 0.90，' +
      'XOR 数据集上线性模型约 0.5 而决策树超过 0.95，展示线性不可分问题的处理能力。' +
      '不变量包括最小二乘残差正交性、Lasso 目标单调、支持向量 KKT 条件。',
  },
  {
    id: 'cluster-algos',
    title: '聚类与降维算法',
    text:
      '聚类模块实现 k-means++ 初始化的高斯混合模型期望最大化算法与主成分分析。' +
      'k-means 保证误差平方和迭代单调不增；高斯混合模型用 logsumexp 稳定计算对数似然，' +
      '期望最大化保证对数似然单调不减，责任度每行和为一；主成分分析对协方差矩阵做 Jacobi 特征分解，' +
      '主成分正交归一，解释方差比累计不超过一。基准：三百样本三团簇数据纯度不低于 0.95，' +
      '已知参数合成数据上均值恢复误差小于 0.3。',
  },
  {
    id: 'sequence-algos',
    title: '序列模型：隐马尔可夫模型',
    text:
      '序列模块实现离散发射隐马尔可夫模型：前向算法、后向算法、维特比解码与鲍姆-韦尔奇参数估计。' +
      '前向概率在 log 域计算并与暴力枚举所有路径的结果交叉验证；维特比路径保证是合法路径且概率不低于任何单条枚举路径；' +
      '鲍姆-韦尔奇每轮对数似然单调不减。另提供通用束搜索工具。基准：已知两态参数生成两百步观测，' +
      '估计出的转移矩阵误差小于 0.1，维特比状态恢复准确率超过 0.8。',
  },
  {
    id: 'nn-layer',
    title: '神经网络与 MiniGPT',
    text:
      '神经网络层基于自动微分实现完整模块体系：线性层、层归一化、dropout、嵌入层、' +
      '多头缩放点积注意力（含因果掩码）、GELU 前馈网络、pre-norm 残差 Transformer 块。' +
      'MiniGPT 是字符级迷你 GPT：词嵌入加可学习位置编码，两层四头 Transformer，' +
      '在莎士比亚与爱丽丝梦游仙境混合语料上从零训练。因果掩码不变量：扰动未来 token 不影响之前位置的输出。' +
      '最强正确性证据是单批量过拟合：重复训练同批量两百步损失降到接近零。' +
      '工程参考为 Karpathy 的 nanoGPT 与 minGPT。',
  },
  {
    id: 'retrieval-layer',
    title: '检索层：BM25 与隐语义分析',
    text:
      '检索层实现 TF-IDF 余弦相似度、Okapi BM25 排序函数（k1 等于 1.2，b 等于 0.75，' +
      'Lucene 非负逆文档频率变体）、基于截断 SVD 的隐语义分析稠密检索，' +
      '以及倒数排名融合混合策略。分词支持英文按非字母数字切分与中文二元组滑窗。' +
      'BM25 用公开算例交叉验证：三篇文档查询猫与坐，首篇得分 1.4167。' +
      '混合检索在自建评测集上的前三精度与平均倒数排名不低于任一单项算法。',
  },
  {
    id: 'rl-layer',
    title: '强化学习层',
    text:
      '强化学习模块实现伯努利老虎机环境与 epsilon 贪心、UCB1、汤普森采样三种探索策略，' +
      '表格型 Q 学习（Bellman 方程）、线性 softmax 策略梯度 REINFORCE，' +
      '以及经典控制环境 CartPole（标准动力学参数，欧拉积分，每步奖励一）。' +
      '基准：十臂老虎机两千步累计遗憾 UCB1 显著低于 epsilon 贪心；' +
      'Q 学习在链式马尔可夫过程上 Bellman 残差收敛到 0.01 以下；' +
      'CartPole 状态分箱后训练最后一百集平均步数不低于 195。',
  },
  {
    id: 'agent-layer',
    title: '认知智能体：带验证证书的神经符号架构',
    text:
      '顶层认知智能体不依赖任何外部大模型 API。它由意图规划器把自然语言目标分解为工具调用流水线，' +
      '流水线覆盖数据集加载、标准化、主成分分析、聚类、回归、分类、检索、安全表达式计算。' +
      '每个数值结论都经过元认知验证器检查对应不变量并签发验证证书，' +
      '最终答案附带证书与置信度。情景记忆记录每次运行，语义记忆缓存事实。' +
      '设计哲学：可靠性来自每一层的不变量，而不是来自对黑盒模型的信任。',
  },
  {
    id: 'reproduce-guide',
    title: '复现方法与运行环境',
    text:
      '整个引擎零依赖零构建：只需 Node 18 以上，直接运行 node 命令即可。' +
      '测试套件用 Node 内置 node:test：执行 node --test 或 node --test test/。' +
      '命令行自检执行 node src/cli.mjs selftest 汇总全部模块不变量。' +
      '基准套件执行 node src/cli.mjs bench 生成 bench/results.json。' +
      '智能体演示执行 node src/cli.mjs demo。GPT 训练执行 node src/cli.mjs train-gpt。' +
      '仪表盘执行 node src/cli.mjs serve 后访问本地 8787 端口。' +
      '所有随机性由种子控制，同种子结果完全一致。',
  },
  {
    id: 'datasets-info',
    title: '数据集与语料来源',
    text:
      '鸢尾花数据集来自 UCI 机器学习库，Fisher 1936 年经典数据，一百五十个样本四维特征三类各五十。' +
      '葡萄酒数据集同样来自 UCI，一百七十八个样本十三维化学特征三类分别五十九、七十一、四十八条。' +
      '字符级语料来自古腾堡计划公有领域文本：爱丽丝梦游仙境与莎士比亚全集，' +
      '共约二十六万字符，词表约七十五个字符。' +
      '合成数据包括双月牙、高斯团簇、异或与带交互项回归数据，全部由注入种子决定。',
  },
  {
    id: 'invariants-philosophy',
    title: '不变量哲学：可验证的人工智能',
    text:
      '项目的第一性原理是：每个算法至少携带一条可被独立实现交叉验证的不变量。' +
      '解析解对数值解：线性回归正规方程残差正交、逻辑回归梯度形式。' +
      '暴力枚举对优化算法：隐马尔可夫模型前向概率对所有路径求和、维特比路径与枚举比较。' +
      '参照实现对主实现：BM25 公开算例精确值、micrograd 公开梯度值、sklearn 同分布公式。' +
      '单调性：k-means 误差平方和不增、期望最大化对数似然不减、Lasso 坐标下降目标不增。' +
      '这些不变量集中在 src/selfcheck.mjs，测试套件、命令行自检与 Web 仪表盘三处复用。',
  },
];

/** 检索用：把标题与正文拼成索引文本 */
export function kbDocsForIndex() {
  return KB_DOCS.map((d) => `${d.title}\n${d.text}`);
}

/** 查询 → 相关文档（简单关键词打分，供智能体快速路由） */
export function kbLookup(query, topK = 3) {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  const scored = KB_DOCS.map((d) => {
    const hay = `${d.title} ${d.text}`.toLowerCase();
    let score = 0;
    for (const t of terms) if (t.length > 1 && hay.includes(t)) score += t.length;
    return { doc: d, score };
  });
  return scored.filter((s) => s.score > 0).sort((a, b) => b.score - a.score).slice(0, topK);
}
