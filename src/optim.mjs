/**
 * @file optim.mjs — 优化器：SGD / Adam / AdamW / Lion / Muon
 * 作者：晨星
 *
 * 直接复用世界顶级优化器成果：
 * - Adam（Kingma & Ba, ICLR 2015）
 * - AdamW 解耦权重衰减（Loshchilov & Hutter, ICLR 2019）
 * - Lion（Chen et al., 2023, Google Brain，符号化更新）
 * - Muon（Keller Jordan, 2024；Moonshot AI, arXiv:2502.16982 规模化版本）
 *   —— 对二维隐藏权重矩阵：Nesterov 动量后接 Newton-Schulz 五次迭代正交化，
 *      系数 (3.4445, -4.7750, 2.0315)；embedding/输出头/归一化层回退 AdamW。
 *      更新按 0.2·√max(rows,cols) 缩放（Moonshot RMS 匹配），使 Muon 与 AdamW
 *      可以共用同一数量级的学习率。
 *
 * 不变量：
 * - 凸二次目标上各优化器损失单调下降并收敛到解析解（正规方程）
 * - Newton-Schulz 输出的奇异值落入 (0.5, 1.5)（近似正交）
 */

import { Mat, matMul, transpose, frobeniusNorm } from './linalg.mjs';

export class Optimizer {
  constructor(params, opts = {}) {
    this.params = params;
    this.lr = opts.lr ?? 1e-3;
    this.stepCount = 0;
  }
  step() { this.stepCount++; }
  zeroGrad() { for (const p of this.params) p.grad.fill(0); }
}

/** 带动量（可选 Nesterov）与 L2 权重衰减的 SGD */
export class SGD extends Optimizer {
  constructor(params, opts = {}) {
    super(params, opts);
    this.lr = opts.lr ?? 0.1;
    this.momentum = opts.momentum ?? 0.9;
    this.nesterov = opts.nesterov ?? false;
    this.weightDecay = opts.weightDecay ?? 0;
    this._v = new Map();
  }
  step() {
    this.stepCount++;
    for (const p of this.params) {
      let v = this._v.get(p);
      if (!v) { v = new Float64Array(p.size); this._v.set(p, v); }
      const g = p.grad;
      for (let i = 0; i < p.size; i++) {
        const gi = g[i] + this.weightDecay * p.data[i];
        v[i] = this.momentum * v[i] + gi;
        const upd = this.nesterov ? gi + this.momentum * v[i] : v[i];
        p.data[i] -= this.lr * upd;
      }
    }
  }
}

/** Adam（可选解耦权重衰减即为 AdamW） */
export class Adam extends Optimizer {
  constructor(params, opts = {}) {
    super(params, opts);
    this.lr = opts.lr ?? 1e-3;
    this.betas = opts.betas ?? [0.9, 0.999];
    this.eps = opts.eps ?? 1e-8;
    this.weightDecay = opts.weightDecay ?? 0;
    this.decoupled = opts.decoupled ?? true; // true → AdamW
    this._m = new Map();
    this._v = new Map();
  }
  step() {
    this.stepCount++;
    const t = this.stepCount;
    const [b1, b2] = this.betas;
    const bc1 = 1 - Math.pow(b1, t);
    const bc2 = 1 - Math.pow(b2, t);
    for (const p of this.params) {
      let m = this._m.get(p);
      let v = this._v.get(p);
      if (!m) { m = new Float64Array(p.size); this._m.set(p, m); v = new Float64Array(p.size); this._v.set(p, v); }
      for (let i = 0; i < p.size; i++) {
        const g = p.grad[i] + (this.decoupled ? 0 : this.weightDecay * p.data[i]);
        m[i] = b1 * m[i] + (1 - b1) * g;
        v[i] = b2 * v[i] + (1 - b2) * g * g;
        const mhat = m[i] / bc1;
        const vhat = v[i] / bc2;
        let d = p.data[i];
        if (this.decoupled && this.weightDecay) d -= this.lr * this.weightDecay * d;
        p.data[i] = d - (this.lr * mhat) / (Math.sqrt(vhat) + this.eps);
      }
    }
  }
}

/** Lion：符号化更新 + 解耦权重衰减（Chen et al., 2023） */
export class Lion extends Optimizer {
  constructor(params, opts = {}) {
    super(params, opts);
    this.lr = opts.lr ?? 3e-4;
    this.betas = opts.betas ?? [0.9, 0.99];
    this.weightDecay = opts.weightDecay ?? 0.1;
    this._m = new Map();
  }
  step() {
    this.stepCount++;
    const [b1, b2] = this.betas;
    for (const p of this.params) {
      let m = this._m.get(p);
      if (!m) { m = new Float64Array(p.size); this._m.set(p, m); }
      for (let i = 0; i < p.size; i++) {
        const c = b1 * m[i] + (1 - b1) * p.grad[i];
        const upd = Math.sign(c);
        let d = p.data[i];
        if (this.weightDecay) d -= this.lr * this.weightDecay * d;
        p.data[i] = d - this.lr * upd;
        m[i] = b2 * m[i] + (1 - b2) * p.grad[i];
      }
    }
  }
}

/**
 * Newton-Schulz 五次迭代：把矩阵近似替换为其最近的半正交矩阵。
 * 系数取自 Keller Jordan 的 Muon 参考实现（在零点附近最大化斜率以减少迭代步数）。
 */
export function newtonSchulz(G, steps = 5) {
  const a = 3.4445, b = -4.7750, c = 2.0315;
  const norm = frobeniusNorm(G);
  if (norm < 1e-12) return G.clone();
  let X = matScaleMat(G, 1 / norm);
  let transposed = false;
  if (X.rows > X.cols) { X = transpose(X); transposed = true; }
  for (let s = 0; s < steps; s++) {
    const A = matMul(X, transpose(X));       // X·Xᵀ
    const A2 = matMul(A, A);
    const B = new Mat(A.rows, A.cols);
    for (let i = 0; i < B.size; i++) B.data[i] = b * A.data[i] + c * A2.data[i];
    const BX = matMul(B, X);
    const out = new Mat(X.rows, X.cols);
    for (let i = 0; i < out.size; i++) out.data[i] = a * X.data[i] + BX.data[i];
    X = out;
  }
  return transposed ? transpose(X) : X;
}

function matScaleMat(a, s) {
  const out = new Mat(a.rows, a.cols);
  for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] * s;
  return out;
}

/**
 * Muon 优化器（Jordan 2024 + Moonshot 2025）：
 * - 二维参数（rows>1 且 cols>1，且未标记 muonExempt）：Nesterov 动量 → Newton-Schulz 正交化
 *   → 0.2·√max(rows,cols) 缩放 → 更新
 * - 其余参数（embedding、输出头、归一化增益、偏置）：AdamW 回退
 */
export class Muon extends Optimizer {
  constructor(params, opts = {}) {
    super(params, opts);
    this.lr = opts.lr ?? 2e-3;
    this.momentum = opts.momentum ?? 0.95;
    this.nesterov = opts.nesterov ?? true;
    this.nsSteps = opts.nsSteps ?? 5;
    this.weightDecay = opts.weightDecay ?? 0;
    this._buf = new Map();
    this._adam = new Adam([], { lr: opts.adamLr ?? opts.lr ?? 2e-3, weightDecay: opts.adamWeightDecay ?? 0 });
  }
  step() {
    this.stepCount++;
    const muonParams = [];
    for (const p of this.params) {
      const isMatrix = p.shape[0] > 1 && p.shape[1] > 1 && !p.muonExempt;
      if (isMatrix) muonParams.push(p);
    }
    // Muon 分支
    for (const p of muonParams) {
      let buf = this._buf.get(p);
      if (!buf) { buf = new Float64Array(p.size); this._buf.set(p, buf); }
      const g = p.grad;
      for (let i = 0; i < p.size; i++) buf[i] = this.momentum * buf[i] + g[i];
      const upd = new Float64Array(p.size);
      if (this.nesterov) { for (let i = 0; i < p.size; i++) upd[i] = g[i] + this.momentum * buf[i]; }
      else { for (let i = 0; i < p.size; i++) upd[i] = buf[i]; }
      const O = newtonSchulz(new Mat(p.shape[0], p.shape[1], upd), this.nsSteps);
      const scale = 0.2 * Math.sqrt(Math.max(p.shape[0], p.shape[1]));
      for (let i = 0; i < p.size; i++) {
        let d = p.data[i] - this.lr * scale * O.data[i];
        if (this.weightDecay) d -= this.lr * this.weightDecay * p.data[i];
        p.data[i] = d;
      }
    }
    // AdamW 回退分支
    const others = this.params.filter((p) => !muonParams.includes(p));
    if (others.length) {
      this._adam.params = others;
      this._adam.stepCount = this.stepCount;
      this._adam.step();
    }
  }
}

/** 全局梯度范数裁剪，返回裁剪前的总范数 */
export function clipGlobalNorm(params, maxNorm) {
  let total = 0;
  for (const p of params) for (let i = 0; i < p.size; i++) total += p.grad[i] * p.grad[i];
  total = Math.sqrt(total);
  if (total > maxNorm && total > 0) {
    const s = maxNorm / total;
    for (const p of params) for (let i = 0; i < p.size; i++) p.grad[i] *= s;
  }
  return total;
}

/** 学习率调度：warmup + constant / cosine / linear */
export function makeSchedule(kind = 'cosine', opts = {}) {
  const { warmup = 0, total = 1000, baseLr = 1e-3, minLr = 0 } = opts;
  return (step) => {
    if (warmup > 0 && step < warmup) return (baseLr * (step + 1)) / warmup;
    const t = total > warmup ? Math.min(1, Math.max(0, (step - warmup) / (total - warmup))) : 1;
    if (kind === 'cosine') return minLr + 0.5 * (baseLr - minLr) * (1 + Math.cos(Math.PI * t));
    if (kind === 'linear') return baseLr + (minLr - baseLr) * t;
    return baseLr;
  };
}

/** 工厂：按名字构造优化器 */
export function createOptimizer(name, params, opts = {}) {
  switch (name) {
    case 'sgd': return new SGD(params, opts);
    case 'adam': return new Adam(params, { decoupled: false, ...opts });
    case 'adamw': return new Adam(params, { decoupled: true, ...opts });
    case 'lion': return new Lion(params, opts);
    case 'muon': return new Muon(params, opts);
    default: throw new Error(`未知优化器: ${name}`);
  }
}
