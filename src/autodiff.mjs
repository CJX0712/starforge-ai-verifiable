/**
 * @file autodiff.mjs — 反向模式自动微分（二维张量级）
 * 作者：晨星
 *
 * 设计思想直接复用 Karpathy micrograd 的反向模式 AD 架构（动态 DAG + 每节点闭包
 * 携带局部导数 + 拓扑排序反向传播链式法则），推广到二维张量：
 * - 每个 Tensor 记录 data / grad / 父节点 _prev / 局部反向闭包 _backward
 * - backward() = 拓扑排序 → 种子梯度 1 → 逆序执行闭包，梯度沿所有路径累加（+=）
 * - 复合运算（LayerNorm/GELU 等）由原语自动组合，避免手写反向出错
 *
 * 参考：Baydin et al., Automatic Differentiation in Machine Learning: a Survey (2015)
 *       Karpathy, micrograd (MIT License)
 *
 * 不变量：每个算子的解析梯度必须通过中心差分数值梯度检验（gradCheckTensor，rel err ≤ 1e-6）
 */

import { Mat, matMul, transpose } from './linalg.mjs';

export class Tensor {
  /**
   * @param {[number, number]} shape 二维形状 [rows, cols]
   * @param {Float64Array} [data] 可选数据（不复制）
   */
  constructor(shape, data, opts = {}) {
    if (!Array.isArray(shape) || shape.length !== 2) throw new Error('Tensor 仅支持二维形状 [rows, cols]');
    this.shape = [shape[0], shape[1]];
    this.data = data && data.length === shape[0] * shape[1] ? data : new Float64Array(shape[0] * shape[1]);
    this.grad = new Float64Array(shape[0] * shape[1]);
    this._prev = [];
    this._backward = null;
    this._op = opts.op || 'leaf';
    this.requiresGrad = opts.requiresGrad !== false;
  }

  get rows() { return this.shape[0]; }
  get cols() { return this.shape[1]; }
  get size() { return this.shape[0] * this.shape[1]; }
  get(i, j) { return this.data[i * this.cols + j]; }
  set(i, j, v) { this.data[i * this.cols + j] = v; return this; }

  static zeros(shape) { return new Tensor(shape); }
  static from2D(arr) {
    const t = new Tensor([arr.length, arr[0].length]);
    for (let i = 0; i < arr.length; i++) for (let j = 0; j < arr[0].length; j++) t.set(i, j, arr[i][j]);
    return t;
  }
  static scalar(x) { return new Tensor([1, 1], new Float64Array([x])); }
  static randn(shape, rng, scale = 1) {
    const t = new Tensor(shape);
    for (let i = 0; i < t.size; i++) t.data[i] = rng.normal() * scale;
    return t;
  }

  /** 反向传播：拓扑排序 → 清零梯度 → 种子置 1 → 逆序执行闭包 */
  backward() {
    const topo = [];
    const seen = new Set();
    const build = (t) => {
      if (seen.has(t)) return;
      seen.add(t);
      for (const p of t._prev) build(p);
      topo.push(t);
    };
    build(this);
    for (const t of topo) t.grad.fill(0);
    this.grad[0] = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const t = topo[i];
      if (t._backward) t._backward();
    }
  }

  zeroGrad() { this.grad.fill(0); return this; }

  // ---------- 广播元素级二元运算 ----------
  add(o) { return binary(this, o, (a, b) => a + b, (_a, _b, g) => g, (_a, _b, g) => g, 'add'); }
  sub(o) { return binary(this, o, (a, b) => a - b, (_a, _b, g) => g, (_a, _b, g) => -g, 'sub'); }
  mul(o) { return binary(this, o, (a, b) => a * b, (_a, b, g) => g * b, (a, _b, g) => g * a, 'mul'); }
  div(o) { return binary(this, o, (a, b) => a / b, (_a, b, g) => g / b, (a, b, g) => (-g * a) / (b * b), 'div'); }
  addScalar(s) { return this.add(Tensor.scalar(s)); }
  mulScalar(s) { return this.mul(Tensor.scalar(s)); }

  // ---------- 一元运算 ----------
  neg() { return unary(this, (x) => -x, (_x, g) => -g, 'neg'); }
  exp() { return unary(this, Math.exp, (_x, g, y) => g * y, 'exp'); }
  log() { return unary(this, Math.log, (x, g) => g / x, 'log'); }
  sqrt() { return unary(this, Math.sqrt, (x, g) => g / (2 * Math.sqrt(x)), 'sqrt'); }
  tanh() { return unary(this, Math.tanh, (_x, g, y) => g * (1 - y * y), 'tanh'); }
  sigmoid() { return unary(this, (x) => 1 / (1 + Math.exp(-x)), (_x, g, y) => g * y * (1 - y), 'sigmoid'); }
  relu() { return unary(this, (x) => (x > 0 ? x : 0), (x, g) => (x > 0 ? g : 0), 'relu'); }
  pow(p) { return unary(this, (x) => Math.pow(x, p), (x, g) => g * p * Math.pow(x, p - 1), `pow${p}`); }

  /** tanh 近似的 GELU：0.5x(1+tanh(√(2/π)(x+0.044715x³)))，由原语组合，反向自动正确 */
  gelu() {
    const k = Math.sqrt(2 / Math.PI);
    const inner = this.pow(3).mulScalar(0.044715).add(Tensor.scalar(1)).mulScalar(k).tanh();
    return this.mul(inner.addScalar(1)).mulScalar(0.5);
  }

  // ---------- 矩阵运算 ----------
  matmul(o) {
    if (this.cols !== o.rows) throw new Error(`matmul 维度不匹配: ${this.rows}x${this.cols} @ ${o.rows}x${o.cols}`);
    const r = this.rows, k = this.cols, c = o.cols;
    const out = new Tensor([r, c], null, { op: 'matmul' });
    const A = new Mat(r, k, this.data), B = new Mat(k, c, o.data);
    out.data.set(matMul(A, B).data);
    const self = this;
    out._prev = [this, o];
    out._backward = () => {
      const dOut = new Mat(r, c, out.grad);
      const dA = matMul(dOut, transpose(B));
      const dB = matMul(transpose(A), dOut);
      for (let i = 0; i < r * k; i++) self.grad[i] += dA.data[i];
      for (let i = 0; i < k * c; i++) o.grad[i] += dB.data[i];
    };
    return out;
  }

  transpose() {
    const out = new Tensor([this.cols, this.rows], null, { op: 'T' });
    for (let i = 0; i < this.rows; i++) for (let j = 0; j < this.cols; j++) out.data[j * this.rows + i] = this.data[i * this.cols + j];
    const self = this;
    out._prev = [this];
    out._backward = () => {
      for (let i = 0; i < self.rows; i++) for (let j = 0; j < self.cols; j++) self.grad[i * self.cols + j] += out.grad[j * self.rows + i];
    };
    return out;
  }

  // ---------- 归约 ----------
  sum() {
    const out = Tensor.scalar(0);
    let s = 0;
    for (let i = 0; i < this.size; i++) s += this.data[i];
    out.data[0] = s;
    const self = this;
    out._prev = [this];
    out._backward = () => { const g = out.grad[0]; for (let i = 0; i < self.size; i++) self.grad[i] += g; };
    return out;
  }
  sumRows() {
    const out = new Tensor([1, this.cols], null, { op: 'sumRows' });
    for (let j = 0; j < this.cols; j++) {
      let s = 0;
      for (let i = 0; i < this.rows; i++) s += this.data[i * this.cols + j];
      out.data[j] = s;
    }
    const self = this;
    out._prev = [this];
    out._backward = () => { for (let i = 0; i < self.rows; i++) for (let j = 0; j < self.cols; j++) self.grad[i * self.cols + j] += out.grad[j]; };
    return out;
  }
  rowMean() { return this.sumRows().mulScalar(1 / this.rows); }
  mean() { return this.sum().mulScalar(1 / this.size); }

  // ---------- 结构运算 ----------
  sliceRows(start, end) {
    const r = end - start, c = this.cols;
    const out = new Tensor([r, c], null, { op: 'sliceRows' });
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out.data[i * c + j] = this.data[(start + i) * c + j];
    const self = this;
    out._prev = [this];
    out._backward = () => { for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) self.grad[(start + i) * c + j] += out.grad[i * c + j]; };
    return out;
  }
  sliceCols(start, end) {
    const r = this.rows, c = end - start;
    const out = new Tensor([r, c], null, { op: 'sliceCols' });
    for (let i = 0; i < r; i++) for (let j = start; j < end; j++) out.data[i * c + (j - start)] = this.data[i * this.cols + j];
    const self = this;
    out._prev = [this];
    out._backward = () => { for (let i = 0; i < r; i++) for (let j = start; j < end; j++) self.grad[i * self.cols + j] += out.grad[i * c + (j - start)]; };
    return out;
  }
  /** 按行索引（embedding 查表） */
  indexRows(indices) {
    const c = this.cols, r = indices.length;
    const out = new Tensor([r, c], null, { op: 'indexRows' });
    for (let i = 0; i < r; i++) { const src = indices[i] * c; for (let j = 0; j < c; j++) out.data[i * c + j] = this.data[src + j]; }
    const self = this;
    out._prev = [this];
    out._backward = () => { for (let i = 0; i < r; i++) { const src = indices[i] * c; for (let j = 0; j < c; j++) self.grad[src + j] += out.grad[i * c + j]; } };
    return out;
  }

  // ---------- 概率与损失 ----------
  /** 行 softmax；反向 dx = y ⊙ (dy − Σ(dy⊙y)) */
  softmaxRows() {
    const r = this.rows, c = this.cols;
    const out = new Tensor([r, c], null, { op: 'softmax' });
    const td = this.data, od = out.data;
    for (let i = 0; i < r; i++) {
      let m = -Infinity;
      for (let j = 0; j < c; j++) if (td[i * c + j] > m) m = td[i * c + j];
      let s = 0;
      for (let j = 0; j < c; j++) { od[i * c + j] = Math.exp(td[i * c + j] - m); s += od[i * c + j]; }
      for (let j = 0; j < c; j++) od[i * c + j] /= s;
    }
    const self = this;
    out._prev = [this];
    out._backward = () => {
      for (let i = 0; i < r; i++) {
        let dot = 0;
        for (let j = 0; j < c; j++) dot += out.grad[i * c + j] * od[i * c + j];
        for (let j = 0; j < c; j++) self.grad[i * c + j] += od[i * c + j] * (out.grad[i * c + j] - dot);
      }
    };
    return out;
  }

  /** 融合 softmax + 交叉熵：targets 为每行的类别下标（普通数组，不需要梯度） */
  crossEntropyLogits(targets) {
    const r = this.rows, c = this.cols;
    const out = Tensor.scalar(0);
    const td = this.data;
    let loss = 0;
    for (let i = 0; i < r; i++) {
      let m = -Infinity;
      for (let j = 0; j < c; j++) if (td[i * c + j] > m) m = td[i * c + j];
      let s = 0;
      for (let j = 0; j < c; j++) s += Math.exp(td[i * c + j] - m);
      loss -= Math.log(Math.exp(td[i * c + targets[i]] - m) / s);
    }
    out.data[0] = loss / r;
    const self = this;
    out._prev = [this];
    out._backward = () => {
      const g = out.grad[0] / r;
      for (let i = 0; i < r; i++) {
        let m = -Infinity;
        for (let j = 0; j < c; j++) if (td[i * c + j] > m) m = td[i * c + j];
        let s = 0;
        const p = new Float64Array(c);
        for (let j = 0; j < c; j++) { p[j] = Math.exp(td[i * c + j] - m); s += p[j]; }
        for (let j = 0; j < c; j++) self.grad[i * c + j] += g * (p[j] / s - (j === targets[i] ? 1 : 0));
      }
    };
    return out;
  }

  /** 均方误差（对标量/矩阵均可） */
  mseLoss(target) {
    const out = Tensor.scalar(0);
    const n = this.size;
    let s = 0;
    for (let i = 0; i < n; i++) { const d = this.data[i] - target.data[i]; s += d * d; }
    out.data[0] = s / n;
    const self = this;
    out._prev = [this, target];
    out._backward = () => {
      const g = (2 * out.grad[0]) / n;
      for (let i = 0; i < n; i++) self.grad[i] += g * (self.data[i] - target.data[i]);
    };
    return out;
  }
}

/** 广播规则：[r,c] 与 [r,1]/[1,c]/[1,1]/[r,c] 兼容 */
function outShape(sa, sb) {
  const r = Math.max(sa[0], sb[0]), c = Math.max(sa[1], sb[1]);
  const ok = (s) => (s[0] === r || s[0] === 1) && (s[1] === c || s[1] === 1);
  if (!ok(sa) || !ok(sb)) throw new Error(`广播不兼容: [${sa}] 与 [${sb}]`);
  return [r, c];
}

function binary(a, b, fwd, bwdA, bwdB, op) {
  const [r, c] = outShape(a.shape, b.shape);
  const out = new Tensor([r, c], null, { op });
  const ra = a.rows, ca = a.cols, rb = b.rows, cb = b.cols;
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
    const av = a.data[(ra === 1 ? 0 : i) * ca + (ca === 1 ? 0 : j)];
    const bv = b.data[(rb === 1 ? 0 : i) * cb + (cb === 1 ? 0 : j)];
    out.data[i * c + j] = fwd(av, bv);
  }
  out._prev = [a, b];
  out._backward = () => {
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
      const g = out.grad[i * c + j];
      if (g === 0) continue;
      const ai = ra === 1 ? 0 : i, aj = ca === 1 ? 0 : j, bi = rb === 1 ? 0 : i, bj = cb === 1 ? 0 : j;
      a.grad[ai * ca + aj] += bwdA(a.data[ai * ca + aj], b.data[bi * cb + bj], g);
      b.grad[bi * cb + bj] += bwdB(a.data[ai * ca + aj], b.data[bi * cb + bj], g);
    }
  };
  return out;
}

function unary(a, fwd, bwd, op) {
  const out = new Tensor(a.shape, null, { op });
  for (let i = 0; i < a.size; i++) out.data[i] = fwd(a.data[i]);
  out._prev = [a];
  out._backward = () => {
    for (let i = 0; i < a.size; i++) a.grad[i] += bwd(a.data[i], out.grad[i], out.data[i]);
  };
  return out;
}

/** 横向拼接（列方向），用于多头注意力拼接 */
export function concatCols(tensors) {
  const rows = tensors[0].rows;
  const cols = tensors.reduce((s, t) => s + t.cols, 0);
  const out = new Tensor([rows, cols], null, { op: 'concatCols' });
  let off = 0;
  for (const t of tensors) {
    for (let i = 0; i < rows; i++) for (let j = 0; j < t.cols; j++) out.data[i * cols + off + j] = t.data[i * t.cols + j];
    off += t.cols;
  }
  out._prev = tensors;
  out._backward = () => {
    let o = 0;
    for (const t of tensors) {
      for (let i = 0; i < rows; i++) for (let j = 0; j < t.cols; j++) t.grad[i * t.cols + j] += out.grad[i * cols + o + j];
      o += t.cols;
    }
  };
  return out;
}

/**
 * 张量梯度检验：对 f(t)（返回标量 Tensor）做中心差分数值梯度对比。
 * 这是 autodiff 正确性的硬金标准：maxRelErr ≤ 1e-6。
 */
export function gradCheckTensor(f, t, { eps = 1e-5, tol = 1e-6, floor = 1e-2 } = {}) {
  const n = t.size;
  const numeric = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const orig = t.data[i];
    t.data[i] = orig + eps;
    const lp = f(t).data[0];
    t.data[i] = orig - eps;
    const lm = f(t).data[0];
    t.data[i] = orig;
    numeric[i] = (lp - lm) / (2 * eps);
  }
  const out = f(t);
  out.backward();
  let maxRelErr = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.abs(t.grad[i] - numeric[i]);
    const s = Math.max(Math.abs(t.grad[i]), Math.abs(numeric[i]), floor);
    maxRelErr = Math.max(maxRelErr, d / s);
  }
  return { maxRelErr, ok: maxRelErr <= tol };
}
