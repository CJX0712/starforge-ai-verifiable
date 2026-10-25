/**
 * @file nn.mjs — 神经网络模块体系 + MiniGPT（零依赖纯 ESM）
 * 作者：晨星
 *
 * 设计思想：所有模块严格复用 src/autodiff.mjs 的 Tensor 原语组合而成，
 * 不手写任何反向传播——梯度由 autodiff 动态图自动推导，正确性由
 * gradCheckTensor（中心差分数值梯度，rel err ≤ 1e-6）逐层金标准检验。
 *
 * 组件（每个组件的数学推导见 docs/algorithms/nn.md）：
 * - Module            基类：parameters()/zeroGrad()/train()/eval()/forward()
 * - Linear            y = xW + b，He/Glorot 缩放初始化（Glorot 2010; He 2015）
 * - LayerNorm         行归一化 γ·(x−μ)/√(σ²+ε)+β（Ba 2016），原语组合
 * - Dropout           训练时 x⊙mask（mask ∈ {0, 1/(1−p)}），推理恒等（Srivastava 2014）
 * - Embedding         查表 id→向量，indexRows 实现，weight.muonExempt
 * - MultiHeadAttention 缩放点积注意力 + 因果掩码（Vaswani 2017）
 * - MLP               Linear→GELU→Linear（Hendrycks 2016）
 * - TransformerBlock  pre-norm 残差：x + attn(LN(x)); x + mlp(LN(x))（Xiong 2020）
 * - MiniGPT           token emb + 可学习位置编码 + N×Block + LN + 线性头（Radford 2018）
 * - Trainer           trainModel：AdamW/Muon + warmup/cosine + 梯度裁剪
 * - 序列化            saveModel/loadModel + 纯 JS base64（Node/浏览器双端可用）
 *
 * 参考实现：Karpathy nanoGPT / minGPT（MIT License）——工程结构（pre-norm、
 * 因果掩码、权重绑定省略、GELU MLP）与其一致，但全部算子为零依赖重写。
 */

import { Tensor, concatCols, gradCheckTensor } from './autodiff.mjs';
import { createOptimizer, clipGlobalNorm, makeSchedule } from './optim.mjs';
import { createRng, softmaxArr } from './linalg.mjs';
import { suite } from './verify.mjs';

// ==================== 基础工具 ====================

/** 把 number[][] 或 Tensor 统一为 Tensor */
function asTensor(x) {
  if (x instanceof Tensor) return x;
  if (Array.isArray(x)) return Tensor.from2D(x);
  throw new Error('forward 需要 Tensor 或 number[][]');
}

/** 拍平嵌套目标数组：number[][] → number[]；number[] 原样返回 */
function flattenTargets(y) {
  return Array.isArray(y[0]) ? y.flat() : y;
}

/**
 * 行统计量（每行一个数）：[r,c] → [r,1]。
 * autodiff 的 rowMean() 是**按列**均值（[1,c]），LayerNorm 需要按行均值，
 * 用 transpose→sumRows→transpose 组合得到（反向自动正确）。
 */
function rowMeans(t) {
  return t.transpose().sumRows().transpose().mulScalar(1 / t.cols);
}

/** 单调计时（Node 与浏览器均有 performance） */
const now = () => (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now());

/**
 * 纵向堆叠（行方向拼接）[r₁,c],[r₂,c],… → [r₁+r₂,c]。
 * concatCols 是横向拼接， batch 前向的 logits 汇总需要本工具。
 */
export function stackRows(tensors) {
  if (tensors.length === 1) return tensors[0];
  const rows = tensors.reduce((s, t) => s + t.rows, 0);
  const cols = tensors[0].cols;
  const out = new Tensor([rows, cols], null, { op: 'stackRows' });
  let off = 0;
  for (const t of tensors) {
    for (let i = 0; i < t.rows; i++) for (let j = 0; j < cols; j++) out.data[(off + i) * cols + j] = t.data[i * cols + j];
    off += t.rows;
  }
  out._prev = tensors;
  out._backward = () => {
    let o = 0;
    for (const t of tensors) {
      for (let i = 0; i < t.rows; i++) for (let j = 0; j < cols; j++) t.grad[i * cols + j] += out.grad[(o + i) * cols + j];
      o += t.rows;
    }
  };
  return out;
}

/**
 * 纵向重复（每行复制 times 份）[r,c] → [r·times,c]。
 * MiniGPT 中用于把位置编码 [ctx,d] 扩展成 [B·ctx,d] 与 batch 化 embedding 对齐。
 */
export function repeatRows(t, times) {
  if (times === 1) return t;
  const out = new Tensor([t.rows * times, t.cols], null, { op: 'repeatRows' });
  for (let r = 0; r < times; r++) {
    for (let i = 0; i < t.rows; i++) {
      for (let j = 0; j < t.cols; j++) out.data[(r * t.rows + i) * t.cols + j] = t.data[i * t.cols + j];
    }
  }
  out._prev = [t];
  out._backward = () => {
    for (let r = 0; r < times; r++) {
      for (let i = 0; i < t.rows; i++) {
        for (let j = 0; j < t.cols; j++) t.grad[i * t.cols + j] += out.grad[(r * t.rows + i) * t.cols + j];
      }
    }
  };
  return out;
}

// ==================== Module 基类 ====================

/**
 * 模块基类。约定：
 * - parameters() 只返回**叶子参数**（requiresGrad 且不在计算图中的可训练 Tensor），
 *   顺序确定，供优化器 / 序列化 / 计数复用；
 * - train()/eval() 递归切换子模块（影响 Dropout）；
 * - forward(x) 接受 Tensor 或 number[][]（内部 asTensor 统一）。
 */
export class Module {
  constructor() {
    this.training = true;
    this._modules = [];
  }

  /** 注册子模块（参与 parameters()/train()/eval() 递归） */
  _register(...mods) {
    for (const m of mods) if (m) this._modules.push(m);
    return this;
  }

  /** 叶子参数数组（容器类按注册顺序递归汇总） */
  parameters() {
    const out = [];
    for (const m of this._modules) out.push(...m.parameters());
    return out;
  }

  zeroGrad() {
    for (const p of this.parameters()) p.grad.fill(0);
    return this;
  }

  train() {
    this.training = true;
    for (const m of this._modules) m.train();
    return this;
  }

  eval() {
    this.training = false;
    for (const m of this._modules) m.eval();
    return this;
  }

  forward(_x) {
    throw new Error('Module 子类必须实现 forward(x)');
  }
}

// ==================== 线性层 ====================

/**
 * 全连接层 y = x·W + b。
 * 初始化（Glorot & Bengio 2010；He et al. 2015）：
 *   glorot: std = gain·√(2/(fanIn+fanOut)) —— 保持前向/反向方差一致
 *   he:     std = gain·√(2/fanIn)         —— 适配 ReLU 族激活
 * 偏置零初始化（打破对称性由权重负责，偏置只需提供自由度）。
 */
export class Linear extends Module {
  constructor(inFeatures, outFeatures, { rng = createRng(0), gain = 1, init = 'glorot', bias = true } = {}) {
    super();
    if (!Number.isInteger(inFeatures) || inFeatures < 1) throw new Error(`Linear inFeatures 必须是正整数，收到 ${inFeatures}`);
    if (!Number.isInteger(outFeatures) || outFeatures < 1) throw new Error(`Linear outFeatures 必须是正整数，收到 ${outFeatures}`);
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    const std = gain * (init === 'he' ? Math.sqrt(2 / inFeatures) : Math.sqrt(2 / (inFeatures + outFeatures)));
    this.W = Tensor.randn([inFeatures, outFeatures], rng, std);
    this.b = bias ? new Tensor([1, outFeatures]) : null;
  }

  forward(x) {
    x = asTensor(x);
    if (x.cols !== this.inFeatures) throw new Error(`Linear 输入维度 ${x.cols} ≠ inFeatures ${this.inFeatures}`);
    const y = x.matmul(this.W);
    return this.b ? y.add(this.b) : y;
  }

  parameters() {
    return this.b ? [this.W, this.b] : [this.W];
  }
}

// ==================== 层归一化 ====================

/**
 * LayerNorm（Ba et al. 2016）：对每一行（每个 token 向量）做
 *   y = γ ⊙ (x − μ) / √(σ² + ε) + β,  μ = mean(x), σ² = mean((x−μ)²)
 * γ 初始 1、β 初始 0（恒等初始化，训练更稳）。全部由 autodiff 原语组合，
 * 反向自动正确（数值梯度见 selfCheck）。
 */
export class LayerNorm extends Module {
  constructor(dim, { eps = 1e-5 } = {}) {
    super();
    if (!Number.isInteger(dim) || dim < 1) throw new Error(`LayerNorm dim 必须是正整数，收到 ${dim}`);
    this.dim = dim;
    this.eps = eps;
    this.gamma = new Tensor([1, dim]);
    for (let j = 0; j < dim; j++) this.gamma.set(0, j, 1);
    this.beta = new Tensor([1, dim]);
  }

  forward(x) {
    x = asTensor(x);
    if (x.cols !== this.dim) throw new Error(`LayerNorm 输入维度 ${x.cols} ≠ dim ${this.dim}`);
    const mu = rowMeans(x); // [r,1] 每行均值
    const xc = x.sub(mu);
    const v = rowMeans(xc.pow(2)).addScalar(this.eps).sqrt(); // [r,1] 每行标准差
    return xc.div(v).mul(this.gamma).add(this.beta);
  }

  parameters() {
    return [this.gamma, this.beta];
  }
}

// ==================== Dropout ====================

/**
 * Dropout（Srivastava et al. 2014）：训练时以概率 p 置零、其余放大 1/(1−p)
 * （inverted dropout，推理零成本）；推理时恒等。
 * mask 是 requiresGrad=false 的常量 Tensor，梯度只经 x 回流。
 * `frozen` 为测试钩子：置 true 后复用当前 mask，使数值梯度检验自洽。
 */
export class Dropout extends Module {
  constructor(p, { rng = createRng(0) } = {}) {
    super();
    if (!(p >= 0 && p < 1)) throw new Error(`Dropout p 必须 ∈ [0,1)，收到 ${p}`);
    this.p = p;
    this.rng = rng;
    this.frozen = false;
    this._mask = null;
  }

  forward(x) {
    x = asTensor(x);
    if (!this.training || this.p === 0) return x;
    const reuse = this.frozen && this._mask && this._mask.rows === x.rows && this._mask.cols === x.cols;
    if (!reuse) {
      const keep = 1 / (1 - this.p);
      const data = new Float64Array(x.size);
      for (let i = 0; i < x.size; i++) data[i] = this.rng.next() < 1 - this.p ? keep : 0;
      this._mask = new Tensor(x.shape, data, { op: 'dropoutMask', requiresGrad: false });
    }
    return x.mul(this._mask);
  }

  parameters() {
    return [];
  }
}

// ==================== Embedding ====================

/**
 * 词嵌入：查表 id → weight[id]。weight 为 [vocabSize, dim] 的 N(0,1) 初始化
 * （规模由后续 LayerNorm 归一化兜底，无需精细缩放）。
 * embedding 不适用 Muon 正交化（Moonshot 2025 实践），标记 muonExempt。
 */
export class Embedding extends Module {
  constructor(vocabSize, dim, { rng = createRng(0), scale = 1 } = {}) {
    super();
    if (!Number.isInteger(vocabSize) || vocabSize < 1) throw new Error(`Embedding vocabSize 必须是正整数，收到 ${vocabSize}`);
    this.vocabSize = vocabSize;
    this.dim = dim;
    this.weight = Tensor.randn([vocabSize, dim], rng, scale);
    this.weight.muonExempt = true;
  }

  /** @param {number[]} ids 任意长度的 token id 序列（可重复） */
  forward(ids) {
    if (!Array.isArray(ids)) throw new Error('Embedding.forward 需要 number[]');
    for (const t of ids) {
      if (!Number.isInteger(t) || t < 0 || t >= this.vocabSize) throw new Error(`token id ${t} 越界 [0, ${this.vocabSize})`);
    }
    return this.weight.indexRows(ids);
  }

  parameters() {
    return [this.weight];
  }
}

// ==================== 多头注意力 ====================

/**
 * 多头缩放点积注意力（Vaswani et al. 2017）：
 *   head_h = softmax(Q_h K_hᵀ / √d_k + M) V_h,  Q=xWq, K=xWk, V=xWv
 *   out = concat(head_1..head_H) Wo
 * 因果掩码 M[i][j] = 0 (j≤i) / −∞ (j>i)：上三角置 −∞ 后 softmax 权重**精确为 0**，
 * 因此"过去位置的输出逐位不依赖未来 token"是可验证不变量（见 selfCheck）。
 * 注意：−∞ 掩码是常量 Tensor（requiresGrad=false），经 add 原语融入计算图，
 * 不切断梯度通路；softmaxRows 内部有 max 减法，−∞ 不会产生 NaN。
 */
export class MultiHeadAttention extends Module {
  constructor(dim, heads, { causal = false, rng = createRng(0), bias = true } = {}) {
    super();
    if (!Number.isInteger(dim) || dim < 1) throw new Error(`dim 必须是正整数，收到 ${dim}`);
    if (!Number.isInteger(heads) || heads < 1 || dim % heads !== 0) throw new Error(`heads=${heads} 必须整除 dim=${dim}`);
    this.dim = dim;
    this.heads = heads;
    this.dk = dim / heads;
    this.causal = causal;
    this.Wq = new Linear(dim, dim, { rng, bias });
    this.Wk = new Linear(dim, dim, { rng, bias });
    this.Wv = new Linear(dim, dim, { rng, bias });
    this.Wo = new Linear(dim, dim, { rng, bias });
    this._register(this.Wq, this.Wk, this.Wv, this.Wo);
    this._maskCache = new Map();
  }

  /** 因果掩码常量 [T,T]：上三角 −∞（按 T 缓存复用） */
  _causalMask(T) {
    let m = this._maskCache.get(T);
    if (!m) {
      const data = new Float64Array(T * T);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) data[i * T + j] = j <= i ? 0 : -Infinity;
      m = new Tensor([T, T], data, { op: 'causalMask', requiresGrad: false });
      this._maskCache.set(T, m);
    }
    return m;
  }

  forward(x) {
    x = asTensor(x);
    const T = x.rows;
    if (x.cols !== this.dim) throw new Error(`MHA 输入维度 ${x.cols} ≠ dim ${this.dim}`);
    const q = this.Wq.forward(x);
    const k = this.Wk.forward(x);
    const v = this.Wv.forward(x);
    const scale = 1 / Math.sqrt(this.dk);
    const headOuts = [];
    for (let h = 0; h < this.heads; h++) {
      const qh = q.sliceCols(h * this.dk, (h + 1) * this.dk);
      const kh = k.sliceCols(h * this.dk, (h + 1) * this.dk);
      const vh = v.sliceCols(h * this.dk, (h + 1) * this.dk);
      let scores = qh.matmul(kh.transpose()).mulScalar(scale);
      if (this.causal) scores = scores.add(this._causalMask(T));
      headOuts.push(scores.softmaxRows().matmul(vh));
    }
    const concat = headOuts.length === 1 ? headOuts[0] : concatCols(headOuts);
    return this.Wo.forward(concat);
  }
}

// ==================== MLP ====================

/**
 * 前馈网络 MLP(dim→hidden→dim)：Linear → GELU → Linear（Hendrycks & Gimpel 2016）。
 * GELU 用 tanh 近似，由 autodiff 原语组合（见 Tensor.gelu）。
 * gain 作用于第二层（残差栈输出层缩小初始化是 GPT 稳定训练常用技巧，默认 1）。
 */
export class MLP extends Module {
  constructor(dim, hidden, { rng = createRng(0), gain = 1 } = {}) {
    super();
    this.fc1 = new Linear(dim, hidden, { rng });
    this.act = new Activation('gelu');
    this.fc2 = new Linear(hidden, dim, { rng, gain });
    this._register(this.fc1, this.act, this.fc2);
  }

  forward(x) {
    return this.fc2.forward(this.act.forward(this.fc1.forward(x)));
  }
}

/** 无参数激活模块（relu/gelu/tanh/sigmoid/identity），便于 Sequential 组合 */
export class Activation extends Module {
  constructor(name = 'identity') {
    super();
    if (!['identity', 'relu', 'gelu', 'tanh', 'sigmoid'].includes(name)) throw new Error(`未知激活: ${name}`);
    this.name = name;
  }

  forward(x) {
    x = asTensor(x);
    switch (this.name) {
      case 'relu': return x.relu();
      case 'gelu': return x.gelu();
      case 'tanh': return x.tanh();
      case 'sigmoid': return x.sigmoid();
      default: return x;
    }
  }

  parameters() {
    return [];
  }
}

/** 顺序容器：前向依次穿过各层 */
export class Sequential extends Module {
  constructor(layers = []) {
    super();
    this.layers = layers;
    this._register(...layers);
  }

  forward(x) {
    let h = x;
    for (const l of this.layers) h = l.forward(h);
    return h;
  }
}

/** 顺序容器 + 交叉熵：用于表格数据分类（Iris 基准复用 Trainer） */
export class Classifier extends Module {
  constructor(layers) {
    super();
    this.net = new Sequential(layers);
    this._register(this.net);
    this.config = { type: 'Classifier', layerCount: layers.length };
  }

  forward(x) {
    return this.net.forward(x);
  }

  loss(x, y) {
    return this.forward(x).crossEntropyLogits(flattenTargets(y));
  }

  /** 返回每行 argmax（预测类别） */
  predict(x) {
    const logits = this.forward(x);
    const out = new Array(logits.rows);
    for (let i = 0; i < logits.rows; i++) {
      let best = 0;
      for (let j = 1; j < logits.cols; j++) if (logits.get(i, j) > logits.get(i, best)) best = j;
      out[i] = best;
    }
    return out;
  }
}

// ==================== Transformer Block ====================

/**
 * Pre-Norm Transformer Block（Xiong et al. 2020）：
 *   x = x + dropout(attn(LN₁(x)))
 *   x = x + dropout(mlp(LN₂(x)))
 * pre-norm 使残差通路保持恒等映射，深层堆叠时梯度不衰减、无需 warmup 也能稳定。
 */
export class TransformerBlock extends Module {
  constructor(dim, heads, { causal = false, dropout = 0, rng = createRng(0), hidden = 4 * dim } = {}) {
    super();
    this.dim = dim;
    this.heads = heads;
    this.ln1 = new LayerNorm(dim);
    this.attn = new MultiHeadAttention(dim, heads, { causal, rng });
    this.drop1 = new Dropout(dropout, { rng });
    this.ln2 = new LayerNorm(dim);
    this.mlp = new MLP(dim, hidden, { rng });
    this.drop2 = new Dropout(dropout, { rng });
    this._register(this.ln1, this.attn, this.drop1, this.ln2, this.mlp, this.drop2);
    this.config = { type: 'TransformerBlock', dim, heads, causal, dropout, hidden };
  }

  forward(x) {
    x = asTensor(x);
    let h = x.add(this.drop1.forward(this.attn.forward(this.ln1.forward(x))));
    h = h.add(this.drop2.forward(this.mlp.forward(this.ln2.forward(h))));
    return h;
  }
}

// ==================== MiniGPT ====================

/**
 * MiniGPT：字符/词元级自回归 Transformer 语言模型（Radford et al. 2018 GPT-1 的最小形态）。
 * 结构：token embedding + 可学习位置编码 → N×因果 TransformerBlock → LN → 线性头。
 *
 * forward(ids: number[][]) → logits [B·T, V]：
 *   1. B·T 个 token 一次查表（indexRows）→ [B·T, d]；
 *   2. 位置编码 repeatRows 扩展 B 份后相加；
 *   3. 按序列切片，各自过 blocks（注意力必须按序列独立计算）；
 *   4. LN + 线性头，stackRows 纵向堆叠成 [B·T, V]。
 */
export class MiniGPT extends Module {
  constructor({ vocabSize, dModel, nLayer = 1, nHead = 4, ctx = 64, dropout = 0, hidden = 4 * dModel, rng = createRng(0) } = {}) {
    super();
    if (!Number.isInteger(vocabSize) || vocabSize < 1) throw new Error(`vocabSize 必须是正整数，收到 ${vocabSize}`);
    if (!Number.isInteger(dModel) || dModel < 1) throw new Error(`dModel 必须是正整数，收到 ${dModel}`);
    if (!Number.isInteger(ctx) || ctx < 1) throw new Error(`ctx 必须是正整数，收到 ${ctx}`);
    if (!Number.isInteger(nLayer) || nLayer < 1) throw new Error(`nLayer 必须是正整数，收到 ${nLayer}`);
    this.vocabSize = vocabSize;
    this.dModel = dModel;
    this.nLayer = nLayer;
    this.nHead = nHead;
    this.ctx = ctx;
    this.tokEmb = new Embedding(vocabSize, dModel, { rng });
    // 可学习位置编码 [ctx, dModel]，N(0,1)；属 embedding 家族，Muon 豁免
    this.posEmb = new Tensor([ctx, dModel]);
    for (let i = 0; i < this.posEmb.size; i++) this.posEmb.data[i] = rng.normal();
    this.posEmb.muonExempt = true;
    this.blocks = [];
    for (let i = 0; i < nLayer; i++) {
      this.blocks.push(new TransformerBlock(dModel, nHead, { causal: true, dropout, rng, hidden }));
    }
    this.lnF = new LayerNorm(dModel);
    this.head = new Linear(dModel, vocabSize, { rng });
    this.head.W.muonExempt = true; // 输出头回退 AdamW
    this._register(this.tokEmb, ...this.blocks, this.lnF, this.head);
    this.config = { type: 'MiniGPT', vocabSize, dModel, nLayer, nHead, ctx, dropout, hidden };
  }

  /** 参数顺序：tokEmb → posEmb → blocks → lnF → head（序列化依赖此确定性） */
  parameters() {
    return [
      this.tokEmb.weight,
      this.posEmb,
      ...this.blocks.flatMap((b) => b.parameters()),
      ...this.lnF.parameters(),
      ...this.head.parameters(),
    ];
  }

  /** @param {number[][]} ids 批次，所有序列长度一致且 ≤ ctx */
  forward(ids) {
    if (!Array.isArray(ids) || ids.length === 0) throw new Error('MiniGPT.forward 需要非空 ids: number[][]');
    const B = ids.length;
    const T = ids[0].length;
    if (!Number.isInteger(T) || T < 1) throw new Error(`序列长度必须 ≥ 1，收到 ${T}`);
    if (T > this.ctx) throw new Error(`序列长度 ${T} 超过上下文窗口 ctx=${this.ctx}`);
    const flat = new Array(B * T);
    let idx = 0;
    for (const s of ids) {
      if (!Array.isArray(s) || s.length !== T) throw new Error('批次内所有序列长度必须一致');
      for (const t of s) {
        if (!Number.isInteger(t) || t < 0 || t >= this.vocabSize) throw new Error(`token id ${t} 越界 [0, ${this.vocabSize})`);
        flat[idx++] = t;
      }
    }
    const h = this.tokEmb.forward(flat).add(repeatRows(this.posEmb.sliceRows(0, T), B)); // [B*T, d]
    const outs = [];
    for (let b = 0; b < B; b++) {
      let x = h.sliceRows(b * T, (b + 1) * T);
      for (const blk of this.blocks) x = blk.forward(x);
      outs.push(this.head.forward(this.lnF.forward(x)));
    }
    return stackRows(outs); // [B*T, V]
  }

  /**
   * 语言模型损失（平均交叉熵）。
   * - loss(x, y)：x/y 均为 number[][]，y 展平后与 logits 行对齐；
   * - loss(ids)：单参模式，目标为 ids 右移一位（next-token 预测）。
   */
  loss(x, y) {
    let inp = x;
    let targets;
    if (y === undefined) {
      if (x[0].length < 2) throw new Error('loss(ids) 单参模式要求序列长度 ≥ 2');
      inp = x.map((s) => s.slice(0, -1));
      targets = x.map((s) => s.slice(1));
    } else {
      targets = flattenTargets(y);
      if (targets.length !== x.length * x[0].length) throw new Error(`目标数量 ${targets.length} 与输入 ${x.length}×${x[0].length} 不匹配`);
    }
    return this.forward(inp).crossEntropyLogits(targets);
  }

  /**
   * 自回归生成。返回新生成的 token 序列（不含 prompt）。
   * temperature→0 时退化为贪心；topK>0 时仅在每步概率最高的 K 个候选内采样。
   */
  generate({ prompt, maxNew = 100, temperature = 1.0, topK = 0, rng = createRng(0) } = {}) {
    if (!Array.isArray(prompt) || prompt.length === 0) throw new Error('generate 需要非空 prompt: number[]');
    if (!(temperature > 0)) throw new Error('temperature 必须 > 0');
    const wasTraining = this.training;
    this.eval();
    let seq = prompt.slice(-this.ctx);
    const out = [];
    try {
      for (let n = 0; n < maxNew; n++) {
        const logits = this.forward([seq]);
        const row = Array.from(logits.sliceRows(logits.rows - 1, logits.rows).data);
        if (temperature !== 1) for (let i = 0; i < row.length; i++) row[i] /= temperature;
        if (topK > 0 && topK < row.length) {
          const idx = row.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, topK).map((p) => p[1]);
          const keep = new Set(idx);
          for (let i = 0; i < row.length; i++) if (!keep.has(i)) row[i] = -Infinity;
        }
        const probs = softmaxArr(row);
        const r = rng.next();
        let acc = 0;
        let pick = probs.length - 1;
        for (let i = 0; i < probs.length; i++) {
          acc += probs[i];
          if (r <= acc) { pick = i; break; }
        }
        out.push(pick);
        seq.push(pick);
        if (seq.length > this.ctx) seq = seq.slice(seq.length - this.ctx);
      }
    } finally {
      if (wasTraining) this.train();
    }
    return out;
  }
}

// ==================== Trainer ====================

/**
 * 通用训练循环：任何提供 parameters() 与 loss(x, y) 的模型均可复用。
 * @returns {{history: Array<{step:number,loss:number,time:number}>, finalLoss:number, wallMs:number, steps:number}}
 */
export function trainModel(model, opts = {}) {
  const {
    getBatch,
    epochs = 1,
    stepsPerEpoch = 100,
    lr = 1e-3,
    optimizer = 'adamw',
    schedule = 'cosine',
    warmup = 0,
    clip = 1.0,
    logEvery = 0,
    weightDecay,
  } = opts;
  if (typeof getBatch !== 'function') throw new Error('trainModel 需要 getBatch() → {x, y}');
  const params = model.parameters();
  const opt = createOptimizer(optimizer, params, weightDecay === undefined ? { lr } : { lr, weightDecay });
  const totalSteps = epochs * stepsPerEpoch;
  const lrAt = makeSchedule(schedule, { warmup, total: totalSteps, baseLr: lr });
  const history = [];
  const t0 = now();
  let step = 0;
  let finalLoss = NaN;
  for (let e = 0; e < epochs; e++) {
    for (let s = 0; s < stepsPerEpoch; s++) {
      const batch = getBatch();
      const loss = model.loss(batch.x, batch.y);
      loss.backward();
      if (clip > 0) clipGlobalNorm(params, clip);
      const lrNow = lrAt(step);
      opt.lr = lrNow;
      if (opt._adam) opt._adam.lr = lrNow; // Muon 的 AdamW 回退分支（embedding/输出头）同步调度
      opt.step();
      finalLoss = loss.data[0];
      history.push({ step, loss: finalLoss, time: now() - t0 });
      if (logEvery > 0 && (step % logEvery === 0 || step === totalSteps - 1)) {
        const ms = (now() - t0) / (step + 1);
        console.log(`  [${model.constructor.name} step ${step + 1}/${totalSteps}] loss ${finalLoss.toFixed(4)} (${ms.toFixed(1)} ms/step)`);
      }
      step++;
    }
  }
  return { history, finalLoss, wallMs: now() - t0, steps: totalSteps };
}

// ==================== 参数计数与序列化 ====================

/** 统计模型可训练参数总数 */
export function countParameters(model) {
  return model.parameters().reduce((s, p) => s + p.size, 0);
}

const B64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

/**
 * Float64Array → base64 字符串（纯 JS 实现，不依赖 Node Buffer / 浏览器 atob，
 * 保证同一份 nn.mjs 在 Node 与浏览器双端可用）。
 */
export function encodeWeightsBase64(f64) {
  const bytes = new Uint8Array(f64.buffer, f64.byteOffset, f64.byteLength);
  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < bytes.length ? bytes[i + 1] : 0;
    const b2 = i + 2 < bytes.length ? bytes[i + 2] : 0;
    out += B64_CHARS[b0 >> 2];
    out += B64_CHARS[((b0 & 3) << 4) | (b1 >> 4)];
    out += i + 1 < bytes.length ? B64_CHARS[((b1 & 15) << 2) | (b2 >> 6)] : '=';
    out += i + 2 < bytes.length ? B64_CHARS[b2 & 63] : '=';
  }
  return out;
}

/** base64 字符串 → Float64Array（encodeWeightsBase64 的逆） */
export function decodeWeightsBase64(s) {
  const lookup = new Map();
  for (let i = 0; i < B64_CHARS.length; i++) lookup.set(B64_CHARS[i], i);
  let byteLen = 0;
  for (let i = 0; i < s.length; i += 4) {
    const c0 = lookup.get(s[i]);
    const c1 = lookup.get(s[i + 1]);
    const c2 = s[i + 2] === '=' ? 0 : lookup.get(s[i + 2]);
    const c3 = s[i + 3] === '=' ? 0 : lookup.get(s[i + 3]);
    if (c0 === undefined || c1 === undefined || c2 === undefined || c3 === undefined) throw new Error('非法 base64 字符');
    const pad = (s[i + 2] === '=' ? 1 : 0) + (s[i + 3] === '=' ? 1 : 0);
    byteLen += 3 - pad;
  }
  const bytes = new Uint8Array(byteLen);
  let p = 0;
  for (let i = 0; i < s.length; i += 4) {
    const c0 = lookup.get(s[i]);
    const c1 = lookup.get(s[i + 1]);
    const c2 = s[i + 2] === '=' ? 0 : lookup.get(s[i + 2]);
    const c3 = s[i + 3] === '=' ? 0 : lookup.get(s[i + 3]);
    bytes[p++] = (c0 << 2) | (c1 >> 4);
    if (s[i + 2] !== '=') bytes[p++] = ((c1 & 15) << 4) | (c2 >> 2);
    if (s[i + 3] !== '=') bytes[p++] = ((c2 & 3) << 6) | c3;
  }
  return new Float64Array(bytes.buffer);
}

/**
 * 序列化模型为 JSON 友好对象：
 *   {config, shapes, paramCount, weights: base64(Float64Array)}
 * 浏览器可直接 fetch 后 loadModel，无需 Buffer。
 */
export function saveModel(model) {
  if (!model.config) throw new Error(`${model.constructor.name} 未提供 config，无法序列化`);
  const params = model.parameters();
  const total = params.reduce((s, p) => s + p.size, 0);
  const flat = new Float64Array(total);
  let off = 0;
  for (const p of params) {
    flat.set(p.data, off);
    off += p.size;
  }
  return {
    config: model.config,
    shapes: params.map((p) => [p.shape[0], p.shape[1]]),
    paramCount: total,
    weights: encodeWeightsBase64(flat),
  };
}

/** loadModel(saveModel(m)) → 结构相同、权重逐位一致的新模型（当前支持 MiniGPT） */
export function loadModel(data) {
  if (!data || !data.config) throw new Error('loadModel 需要 saveModel 产生的数据');
  let model;
  if (data.config.type === 'MiniGPT') {
    model = new MiniGPT(data.config);
  } else {
    throw new Error(`loadModel 暂不支持类型: ${data.config.type}`);
  }
  const params = model.parameters();
  if (!data.shapes || data.shapes.length !== params.length) throw new Error('权重形状元数据与模型结构不匹配');
  const flat = decodeWeightsBase64(data.weights);
  let off = 0;
  for (let i = 0; i < params.length; i++) {
    if (data.shapes[i][0] !== params[i].shape[0] || data.shapes[i][1] !== params[i].shape[1]) {
      throw new Error(`第 ${i} 个参数形状不匹配: 期望 [${params[i].shape}]，数据 [${data.shapes[i]}]`);
    }
    params[i].data.set(flat.subarray(off, off + params[i].size));
    off += params[i].size;
  }
  return model;
}

// ==================== 自检（不变量金标准） ====================

/**
 * nn 模块全部不变量自检。每项检查对应一条可被独立交叉验证的性质：
 * 梯度检验（数值 vs 解析）、形状契约、因果性、确定性、过拟合能力、序列化往返。
 */
export function nnSelfCheck() {
  const s = suite('nn');

  const gc = (label, f, t) => {
    const r = gradCheckTensor(f, t);
    s.check(label, () => {
      if (!r.ok) throw new Error(`maxRelErr = ${r.maxRelErr.toExponential(2)}`);
      return `maxRelErr = ${r.maxRelErr.toExponential(1)}`;
    });
  };

  // ---- Linear ----
  {
    const rng = createRng(1);
    const lin = new Linear(5, 3, { rng });
    gc('Linear: 梯度检验（数值梯度金标准）', (x) => lin.forward(x).sum(), Tensor.randn([4, 5], rng));
    s.check('Linear: 形状 / 零偏置 / Glorot 尺度', () => {
      const out = lin.forward(Tensor.randn([7, 5], rng));
      if (out.rows !== 7 || out.cols !== 3) throw new Error(`形状 [${out.shape}]`);
      for (let j = 0; j < 3; j++) if (lin.b.get(0, j) !== 0) throw new Error('偏置未零初始化');
      let m2 = 0;
      for (let i = 0; i < lin.W.size; i++) m2 += lin.W.data[i] ** 2;
      const std = Math.sqrt(m2 / lin.W.size);
      const expect = Math.sqrt(2 / 8);
      if (Math.abs(std - expect) > 0.25 * expect) throw new Error(`W 经验标准差 ${std.toFixed(4)} 偏离 Glorot ${expect.toFixed(4)}`);
      return `std=${std.toFixed(4)} ≈ Glorot ${expect.toFixed(4)}，b=0`;
    });
  }

  // ---- LayerNorm ----
  {
    const rng = createRng(2);
    const ln = new LayerNorm(6);
    gc('LayerNorm: 梯度检验', (x) => ln.forward(x).sum(), Tensor.randn([5, 6], rng));
    s.check('LayerNorm: 输出每行均值 0 方差 1（γ=1,β=0）', () => {
      const x = Tensor.randn([6, 6], rng); // 单位尺度输入
      const y = ln.forward(x);
      for (let i = 0; i < y.rows; i++) {
        let m = 0;
        for (let j = 0; j < y.cols; j++) m += y.get(i, j);
        m /= y.cols;
        if (Math.abs(m) > 1e-12) throw new Error(`行 ${i} 均值 ${m}`);
        let v = 0;
        for (let j = 0; j < y.cols; j++) v += (y.get(i, j) - m) ** 2;
        v /= y.cols;
        // ε=1e-5 使方差系统性为 1/(1+ε) ≈ 1−1e-5，容差按 ε 效应取 1e-4
        if (Math.abs(v - 1) > 1e-4) throw new Error(`行 ${i} 方差 ${v}`);
      }
      return '6 行均值≈0、方差≈1（ε 效应内）';
    });

    s.check('LayerNorm: γ/β 缩放与偏移生效', () => {
      const ln2 = new LayerNorm(4);
      for (let j = 0; j < 4; j++) { ln2.gamma.set(0, j, 2); ln2.beta.set(0, j, 3); }
      const y = ln2.forward(Tensor.randn([3, 4], rng));
      for (let i = 0; i < 3; i++) {
        let m = 0;
        for (let j = 0; j < 4; j++) m += y.get(i, j);
        m /= 4;
        let v = 0;
        for (let j = 0; j < 4; j++) v += (y.get(i, j) - m) ** 2;
        v /= 4;
        if (Math.abs(m - 3) > 1e-10 || Math.abs(v - 4) > 1e-3) throw new Error(`mean=${m.toFixed(6)}, var=${v.toFixed(6)}`);
      }
      return 'γ=2,β=3 → 行均值 3、方差 4';
    });
  }

  // ---- Dropout ----
  {
    const rng = createRng(3);
    s.check('Dropout: eval 模式恒等', () => {
      const d = new Dropout(0.5, { rng });
      d.eval();
      const x = Tensor.randn([4, 4], rng);
      const y = d.forward(x);
      for (let i = 0; i < x.size; i++) if (y.data[i] !== x.data[i]) throw new Error('eval 模式修改了输入');
      return '恒等';
    });
    const d = new Dropout(0.5, { rng });
    d.train();
    s.check('Dropout: 训练取值 ∈ {0, x/(1−p)}', () => {
      const x = Tensor.randn([4, 4], rng);
      const y = d.forward(x);
      for (let i = 0; i < x.size; i++) {
        const ok = y.data[i] === 0 || Math.abs(y.data[i] - x.data[i] * 2) < 1e-12;
        if (!ok) throw new Error(`异常值 ${y.data[i]}`);
      }
      return 'inverted dropout 缩放正确';
    });
    d.frozen = true; // 冻结掩码使数值梯度自洽
    gc('Dropout: 梯度检验（冻结掩码）', (x) => d.forward(x).sum(), Tensor.randn([4, 4], rng));
    d.frozen = false;
    s.check('Dropout: 期望保持 E[x·mask]=x', () => {
      const d2 = new Dropout(0.5, { rng: createRng(8) });
      d2.train();
      const x = Tensor.from2D([[2]]);
      let sum = 0;
      const N = 400;
      for (let i = 0; i < N; i++) sum += d2.forward(x).data[0];
      const mean = sum / N;
      if (Math.abs(mean - 2) > 0.25) throw new Error(`均值 ${mean.toFixed(3)} 偏离 2`);
      return `400 次均值 ${mean.toFixed(3)} ≈ 2`;
    });
  }

  // ---- Embedding ----
  {
    const rng = createRng(4);
    const emb = new Embedding(6, 4, { rng });
    gc('Embedding: 梯度检验（含重复索引累加）', (w) => w.indexRows([0, 2, 2, 1]).sum(), emb.weight);
    s.check('Embedding: 查表返回对应行', () => {
      const y = emb.forward([3, 0, 3]);
      if (y.rows !== 3 || y.cols !== 4) throw new Error(`形状 [${y.shape}]`);
      for (let j = 0; j < 4; j++) if (y.get(0, j) !== emb.weight.get(3, j)) throw new Error('第 0 行 ≠ weight[3]');
      for (let j = 0; j < 4; j++) if (y.get(2, j) !== emb.weight.get(3, j)) throw new Error('第 2 行 ≠ weight[3]');
      if (!emb.weight.muonExempt) throw new Error('embedding 未标记 muonExempt');
      return 'indexRows 语义正确，muonExempt 已设';
    });
  }

  // ---- MultiHeadAttention ----
  {
    const rng = createRng(5);
    const attn = new MultiHeadAttention(8, 2, { causal: false, rng });
    gc('MultiHeadAttention: 梯度检验（双向）', (x) => attn.forward(x).sum(), Tensor.randn([5, 8], rng));
    const causal = new MultiHeadAttention(8, 2, { causal: true, rng });
    gc('MultiHeadAttention: 梯度检验（因果）', (x) => causal.forward(x).sum(), Tensor.randn([5, 8], rng));
    s.check('MultiHeadAttention: 因果不变量（未来扰动不影响过去，逐位）', () => {
      causal.eval();
      const x = Tensor.randn([6, 8], rng);
      const o1 = causal.forward(x);
      const data = x.data.slice();
      for (let i = 4; i < 6; i++) for (let j = 0; j < 8; j++) data[i * 8 + j] += 10; // 扰动最后两行
      const o2 = causal.forward(new Tensor([6, 8], data));
      for (let i = 0; i < 4; i++) for (let j = 0; j < 8; j++) {
        if (o1.get(i, j) !== o2.get(i, j)) throw new Error(`位置 ${i} 输出被未来 token 改变`);
      }
      return '前 4 行逐位不变';
    });
    s.check('MultiHeadAttention: 注意力行和为 1（softmax 概率守恒）', () => {
      const x = Tensor.randn([4, 8], rng);
      const q = attn.Wq.forward(x);
      const k = attn.Wk.forward(x);
      const scores = q.sliceCols(0, 4).matmul(k.sliceCols(0, 4).transpose()).mulScalar(1 / 2).softmaxRows();
      for (let i = 0; i < scores.rows; i++) {
        let sum = 0;
        for (let j = 0; j < scores.cols; j++) sum += scores.get(i, j);
        if (Math.abs(sum - 1) > 1e-12) throw new Error(`行 ${i} 概率和 ${sum}`);
      }
      return '4 行和均为 1';
    });
  }

  // ---- MLP / TransformerBlock ----
  {
    const rng = createRng(6);
    const mlp = new MLP(6, 8, { rng });
    gc('MLP: 梯度检验（GELU 组合）', (x) => mlp.forward(x).sum(), Tensor.randn([4, 6], rng));
    s.check('MLP: 形状 [T,dim]→[T,dim]', () => {
      const y = mlp.forward(Tensor.randn([3, 6], rng));
      if (y.rows !== 3 || y.cols !== 6) throw new Error(`形状 [${y.shape}]`);
      return '[3,6]';
    });
    const blk = new TransformerBlock(6, 3, { causal: true, rng });
    gc('TransformerBlock: 梯度检验（pre-norm 残差）', (x) => blk.forward(x).sum(), Tensor.randn([4, 6], rng));
    s.check('TransformerBlock: 残差恒等路径存在（输入与输出相关）', () => {
      blk.eval();
      const x = Tensor.randn([4, 6], rng);
      const y = blk.forward(x);
      if (y.rows !== 4 || y.cols !== 6) throw new Error(`形状 [${y.shape}]`);
      let dot = 0;
      for (let i = 0; i < x.size; i++) dot += x.data[i] * y.data[i];
      if (dot <= 0) throw new Error('输出与输入无正相关，残差路径可疑');
      return '形状 [4,6]，残差通路正常';
    });
  }

  // ---- MiniGPT ----
  {
    const rng = createRng(7);
    const gpt = new MiniGPT({ vocabSize: 14, dModel: 8, nLayer: 2, nHead: 2, ctx: 10, rng });
    const x = [[1, 5, 9, 2, 7, 3], [0, 4, 4, 8, 6, 1]];
    const y = [[5, 9, 2, 7, 3, 8], [4, 4, 8, 6, 1, 10]]; // 与 x 逐位对齐的 next-token 目标

    s.check('MiniGPT: 前向形状 [B*T, V]', () => {
      const logits = gpt.forward(x);
      if (logits.rows !== 12 || logits.cols !== 14) throw new Error(`形状 [${logits.shape}]`);
      return '[12,14]';
    });

    s.check('MiniGPT: 端到端损失梯度检验（对 token embedding，完整前向）', () => {
      // gradCheckTensor 原地扰动 t.data；loss 每次前向都读 this.tokEmb.weight（即 t）
      const r = gradCheckTensor(() => gpt.loss(x, y), gpt.tokEmb.weight);
      if (!r.ok) throw new Error(`maxRelErr = ${r.maxRelErr.toExponential(2)}`);
      return `maxRelErr = ${r.maxRelErr.toExponential(1)}`;
    });

    s.check('MiniGPT: 参数量与理论公式一致', () => {
      const V = 14, d = 8, L = 2, C = 10, hidden = 4 * d;
      const perBlock = 2 * d + 4 * (d * d + d) + 2 * d + (d * hidden + hidden) + (hidden * d + d);
      const expected = V * d + C * d + L * perBlock + 2 * d + (d * V + V);
      const actual = countParameters(gpt);
      if (actual !== expected) throw new Error(`实际 ${actual} ≠ 理论 ${expected}`);
      return `${actual} = V·d + C·d + L·(12d²+13d) + 2d + dV + V`;
    });

    s.check('MiniGPT: 因果不变量（扰动未来 token，过去 logits 逐位不变）', () => {
      gpt.eval();
      const seq = [1, 5, 9, 2, 7, 3, 8, 0, 4, 6];
      const l1 = gpt.forward([seq]);
      const seq2 = seq.slice();
      for (let i = 5; i < 10; i++) seq2[i] = (seq2[i] + 3) % 14;
      const l2 = gpt.forward([seq2]);
      for (let i = 0; i < 5; i++) for (let j = 0; j < 14; j++) {
        if (l1.get(i, j) !== l2.get(i, j)) throw new Error(`位置 ${i} logits 被未来 token 改变`);
      }
      return '前 5 个位置 logits 逐位一致';
    });

    s.check('MiniGPT: 确定性（同种子两次前向逐位一致）', () => {
      const mk = () => new MiniGPT({ vocabSize: 14, dModel: 8, nLayer: 2, nHead: 2, ctx: 10, rng: createRng(77) });
      const a = mk().forward(x);
      const b = mk().forward(x);
      for (let i = 0; i < a.size; i++) if (a.data[i] !== b.data[i]) throw new Error('两次前向不一致');
      return '168 个 logits 逐位一致';
    });

    s.check('MiniGPT: batch 内序列独立（单条前向 = batch 切片）', () => {
      const full = gpt.forward(x);
      const only = gpt.forward([x[1]]);
      for (let i = 0; i < 6; i++) for (let j = 0; j < 14; j++) {
        if (full.get(6 + i, j) !== only.get(i, j)) throw new Error(`batch 位置 1 的第 ${i} 行不一致`);
      }
      return '序列间无串扰';
    });

    s.check('MiniGPT: 单批过拟合 loss < 0.01（反向传播正确性的最强证据）', () => {
      const m = new MiniGPT({ vocabSize: 24, dModel: 32, nLayer: 1, nHead: 4, ctx: 16, rng: createRng(9) });
      const bx = [[3, 8, 1, 15, 20, 7, 2, 11, 0, 19, 5, 22, 9, 14, 4, 17], [10, 6, 23, 2, 18, 12, 0, 21, 8, 3, 16, 1, 13, 5, 22, 7]];
      const by = bx.map((sq) => sq.slice(1));
      const inp = bx.map((sq) => sq.slice(0, -1));
      const opt = createOptimizer('adamw', m.parameters(), { lr: 5e-3 });
      let last = 0;
      for (let i = 0; i < 200; i++) {
        const loss = m.loss(inp, by);
        loss.backward();
        clipGlobalNorm(m.parameters(), 1.0);
        opt.step();
        last = loss.data[0];
      }
      if (!(last < 0.01)) throw new Error(`200 步后 loss = ${last.toFixed(4)}，未收敛到 0.01 以下`);
      return `200 步 loss → ${last.toExponential(2)}`;
    });

    s.check('Trainer: trainModel 返回结构与损失下降', () => {
      const m = new MiniGPT({ vocabSize: 14, dModel: 8, nLayer: 1, nHead: 2, ctx: 8, rng: createRng(11) });
      const bx = [[1, 5, 9, 2, 7, 3, 8, 0], [4, 4, 8, 6, 1, 12, 2, 9]];
      const by = bx.map((sq) => sq.slice(1));
      const inp = bx.map((sq) => sq.slice(0, -1));
      const res = trainModel(m, { getBatch: () => ({ x: inp, y: by }), stepsPerEpoch: 30, lr: 3e-3, schedule: 'constant', clip: 1.0 });
      if (res.history.length !== 30) throw new Error(`history 长度 ${res.history.length}`);
      for (const h of res.history) {
        if (typeof h.step !== 'number' || typeof h.loss !== 'number' || typeof h.time !== 'number') throw new Error('history 字段缺失');
      }
      if (!(res.finalLoss < res.history[0].loss * 0.8)) throw new Error(`损失未下降: ${res.history[0].loss.toFixed(4)} → ${res.finalLoss.toFixed(4)}`);
      if (!(res.wallMs > 0)) throw new Error('wallMs 异常');
      return `30 步 ${res.history[0].loss.toFixed(3)} → ${res.finalLoss.toFixed(3)}`;
    });

    s.check('generate: 自回归采样（确定性 / 合法 id / 长度）', () => {
      gpt.eval();
      const prompt = [1, 5, 9];
      const a = gpt.generate({ prompt, maxNew: 12, temperature: 0.8, rng: createRng(42) });
      const b = gpt.generate({ prompt, maxNew: 12, temperature: 0.8, rng: createRng(42) });
      if (a.length !== 12) throw new Error(`生成长度 ${a.length}`);
      for (let i = 0; i < 12; i++) if (a[i] !== b[i]) throw new Error('同种子生成不一致');
      for (const t of a) if (!Number.isInteger(t) || t < 0 || t >= 14) throw new Error(`非法 id ${t}`);
      const greedy = gpt.generate({ prompt, maxNew: 6, temperature: 1e-6, rng: createRng(1) });
      if (greedy.length !== 6) throw new Error('低温生成长度错误');
      return '12 token，同种子一致';
    });
  }

  // ---- 序列化 ----
  {
    s.check('base64 权重编解码往返一致（含非 3 字节倍数）', () => {
      const rng = createRng(12);
      const arr = new Float64Array(257);
      for (let i = 0; i < arr.length; i++) arr[i] = rng.normal() * 1e6;
      const s = encodeWeightsBase64(arr);
      const back = decodeWeightsBase64(s);
      if (back.length !== arr.length) throw new Error(`长度 ${back.length} ≠ ${arr.length}`);
      for (let i = 0; i < arr.length; i++) if (back[i] !== arr[i]) throw new Error(`第 ${i} 个浮点数不一致`);
      return `257 个 Float64（2056 字节）往返逐位一致`;
    });

    s.check('saveModel/loadModel 往返：结构与前向逐位一致', () => {
      const m = new MiniGPT({ vocabSize: 20, dModel: 8, nLayer: 2, nHead: 2, ctx: 10, rng: createRng(13) });
      const ids = [[1, 5, 9, 2, 7], [0, 4, 4, 8, 6]];
      const l1 = m.forward(ids);
      const data = saveModel(m);
      if (data.config.type !== 'MiniGPT') throw new Error('config 类型错误');
      if (typeof data.weights !== 'string') throw new Error('weights 不是 base64 字符串');
      const m2 = loadModel(JSON.parse(JSON.stringify(data)));
      const l2 = m2.forward(ids);
      for (let i = 0; i < l1.size; i++) if (l1.data[i] !== l2.data[i]) throw new Error('往返后前向不一致');
      if (countParameters(m2) !== data.paramCount) throw new Error('参数量不一致');
      return `${data.paramCount} 参数，JSON 往返后前向逐位一致`;
    });
  }

  return [s.report()];
}

/** selfCheck 别名（与项目 selfcheck.mjs 的命名约定兼容） */
export const selfCheck = nnSelfCheck;
