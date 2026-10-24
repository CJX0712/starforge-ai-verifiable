/**
 * @file linalg.mjs — StarForge 核心数学层：零依赖线性代数与数值内核
 * 作者：晨星
 *
 * 参考（世界顶级数值线性代数成果，直接复用其算法思想）：
 * - Golub & Van Loan, Matrix Computations（Householder QR、部分选主元高斯消元）
 * - Trefethen & Bau, Numerical Linear Algebra（Jacobi 循环旋转特征分解）
 * - Numerical Recipes（ Jacobi 旋转的 t 公式、logsumexp 稳定技巧）
 *
 * 不变量（可被独立交叉验证的硬金标准）：
 * - 正交矩阵 QᵀQ = I
 * - 对称阵特征向量彼此正交归一，残差 ‖Av−λv‖ ≈ ε
 * - SVD 重构 A ≈ UΣVᵀ（Eckart-Young 最优性）
 * - 解线性方程组后残差 ‖Ax−b‖ ≈ ε
 */

/** 确定性伪随机数生成器（mulberry32 + Box-Muller），所有随机性必须经此注入以保证可复现 */
export function createRng(seed = 42) {
  let a = seed >>> 0;
  const next = () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    seed,
    next,
    uniform: (lo = 0, hi = 1) => lo + (hi - lo) * next(),
    randint: (n) => Math.floor(next() * n),
    normal: () => {
      let u = 0, v = 0;
      while (u === 0) u = next();
      while (v === 0) v = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    },
    shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(next() * (i + 1));
        const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
      }
      return arr;
    },
  };
}

/** 稠密矩阵（行主序 Float64Array） */
export class Mat {
  constructor(rows, cols, data) {
    this.rows = rows;
    this.cols = cols;
    this.data = data && data.length === rows * cols ? data : new Float64Array(rows * cols);
  }
  static zeros(rows, cols) { return new Mat(rows, cols); }
  static identity(n) {
    const m = new Mat(n, n);
    for (let i = 0; i < n; i++) m.data[i * n + i] = 1;
    return m;
  }
  static diag(values) {
    const n = values.length;
    const m = new Mat(n, n);
    for (let i = 0; i < n; i++) m.data[i * n + i] = values[i];
    return m;
  }
  static from2D(arr) {
    const rows = arr.length, cols = arr[0].length;
    const m = new Mat(rows, cols);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) m.data[i * cols + j] = arr[i][j];
    return m;
  }
  get size() { return this.rows * this.cols; }
  get(i, j) { return this.data[i * this.cols + j]; }
  set(i, j, v) { this.data[i * this.cols + j] = v; return this; }
  row(i) {
    const c = this.cols, out = new Float64Array(c);
    for (let j = 0; j < c; j++) out[j] = this.data[i * c + j];
    return out;
  }
  col(j) {
    const r = this.rows, c = this.cols, out = new Float64Array(r);
    for (let i = 0; i < r; i++) out[i] = this.data[i * c + j];
    return out;
  }
  clone() { return new Mat(this.rows, this.cols, this.data.slice()); }
  colSlice(start, end) {
    const out = new Mat(this.rows, end - start);
    for (let i = 0; i < this.rows; i++)
      for (let j = start; j < end; j++) out.set(i, j - start, this.get(i, j));
    return out;
  }
  to2D() {
    const out = [];
    for (let i = 0; i < this.rows; i++) out.push(Array.from(this.row(i)));
    return out;
  }
}

/** 矩阵乘法 C = A·B，i-k-j 循环序 + 零值跳过 */
export function matMul(a, b) {
  if (a.cols !== b.rows) throw new Error(`matMul 维度不匹配: ${a.rows}x${a.cols} @ ${b.rows}x${b.cols}`);
  const r = a.rows, k = a.cols, c = b.cols;
  const out = new Mat(r, c);
  const ad = a.data, bd = b.data, od = out.data;
  for (let i = 0; i < r; i++) {
    const ir = i * k, orow = i * c;
    for (let p = 0; p < k; p++) {
      const av = ad[ir + p];
      if (av === 0) continue;
      const pb = p * c;
      for (let j = 0; j < c; j++) od[orow + j] += av * bd[pb + j];
    }
  }
  return out;
}

export function matVec(a, x) {
  if (a.cols !== x.length) throw new Error('matVec 维度不匹配');
  const out = new Float64Array(a.rows);
  for (let i = 0; i < a.rows; i++) {
    let s = 0;
    for (let j = 0; j < a.cols; j++) s += a.get(i, j) * x[j];
    out[i] = s;
  }
  return out;
}

export function transpose(a) {
  const out = new Mat(a.cols, a.rows);
  for (let i = 0; i < a.rows; i++) for (let j = 0; j < a.cols; j++) out.set(j, i, a.get(i, j));
  return out;
}

export function matAdd(a, b) {
  const out = new Mat(a.rows, a.cols);
  for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] + b.data[i];
  return out;
}
export function matSub(a, b) {
  const out = new Mat(a.rows, a.cols);
  for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] - b.data[i];
  return out;
}
export function matScale(a, s) {
  const out = new Mat(a.rows, a.cols);
  for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] * s;
  return out;
}
export function matHadamard(a, b) {
  const out = new Mat(a.rows, a.cols);
  for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] * b.data[i];
  return out;
}

export function frobeniusNorm(a) {
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i] * a.data[i];
  return Math.sqrt(s);
}
export function l2norm(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s);
}

export function isSymmetric(a, tol = 1e-10) {
  if (a.rows !== a.cols) return false;
  for (let i = 0; i < a.rows; i++)
    for (let j = i + 1; j < a.cols; j++)
      if (Math.abs(a.get(i, j) - a.get(j, i)) > tol) return false;
  return true;
}

/** 高斯消元（部分选主元）解 A·X = B；B 为 Mat（多右端项）或 Float64Array（单右端项） */
export function solveLinear(a, b) {
  const n = a.rows;
  if (a.cols !== n) throw new Error('solveLinear 需要方阵');
  const isVec = b instanceof Float64Array || Array.isArray(b);
  const B = isVec ? new Mat(n, 1, b instanceof Float64Array ? b.slice() : new Float64Array(b)) : b.clone();
  const A = a.clone();
  const m = B.cols;
  const ad = A.data, bd = B.data;
  for (let col = 0; col < n; col++) {
    let piv = col, best = Math.abs(ad[col * n + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(ad[r * n + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-14) throw new Error('矩阵奇异或接近奇异，无法求解');
    if (piv !== col) {
      for (let j = 0; j < n; j++) { const t = ad[col * n + j]; ad[col * n + j] = ad[piv * n + j]; ad[piv * n + j] = t; }
      for (let j = 0; j < m; j++) { const t = bd[col * m + j]; bd[col * m + j] = bd[piv * m + j]; bd[piv * m + j] = t; }
    }
    const d = ad[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const f = ad[r * n + col] / d;
      if (f === 0) continue;
      ad[r * n + col] = 0;
      for (let j = col + 1; j < n; j++) ad[r * n + j] -= f * ad[col * n + j];
      for (let j = 0; j < m; j++) bd[r * m + j] -= f * bd[col * m + j];
    }
  }
  for (let col = n - 1; col >= 0; col--) {
    for (let j = 0; j < m; j++) {
      let s = bd[col * m + j];
      for (let k = col + 1; k < n; k++) s -= ad[col * n + k] * bd[k * m + j];
      bd[col * m + j] = s / ad[col * n + col];
    }
  }
  return isVec ? B.col(0) : B;
}

export function inverse(a) { return solveLinear(a, Mat.identity(a.rows)); }

export function det(a) {
  const n = a.rows;
  const A = a.clone();
  let sign = 1, d = 1;
  for (let col = 0; col < n; col++) {
    let piv = col, best = Math.abs(A.get(col, col));
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(A.get(r, col));
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-300) return 0;
    if (piv !== col) {
      for (let j = 0; j < n; j++) { const t = A.get(col, j); A.set(col, j, A.get(piv, j)); A.set(piv, j, t); }
      sign = -sign;
    }
    d *= A.get(col, col);
    for (let r = col + 1; r < n; r++) {
      const f = A.get(r, col) / A.get(col, col);
      if (f === 0) continue;
      for (let j = col; j < n; j++) A.set(r, j, A.get(r, j) - f * A.get(col, j));
    }
  }
  return sign * d;
}

/** Cholesky 分解 A = L·Lᵀ（A 必须正定） */
export function cholesky(a) {
  const n = a.rows;
  const L = new Mat(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a.get(i, j);
      for (let k = 0; k < j; k++) s -= L.get(i, k) * L.get(j, k);
      if (i === j) {
        if (s <= 0) throw new Error('矩阵非正定，Cholesky 分解失败');
        L.set(i, j, Math.sqrt(s));
      } else {
        L.set(i, j, s / L.get(j, j));
      }
    }
  }
  return L;
}

export function isPositiveDefinite(a) {
  try { cholesky(a); return true; } catch { return false; }
}

/** Householder QR 分解：A = Q·R，Q 为 m×m 正交阵，R 为 m×n 上三角阵 */
export function qr(a) {
  const m = a.rows, n = a.cols;
  const R = a.clone();
  const Q = Mat.identity(m);
  for (let k = 0; k < Math.min(m - 1, n); k++) {
    let xnorm = 0;
    for (let i = k; i < m; i++) xnorm += R.get(i, k) * R.get(i, k);
    xnorm = Math.sqrt(xnorm);
    if (xnorm < 1e-300) continue;
    const alpha = R.get(k, k) > 0 ? -xnorm : xnorm;
    const v = new Float64Array(m - k);
    v[0] = R.get(k, k) - alpha;
    for (let i = k + 1; i < m; i++) v[i - k] = R.get(i, k);
    let vnorm2 = 0;
    for (const t of v) vnorm2 += t * t;
    if (vnorm2 < 1e-300) continue;
    // R ← H·R
    for (let j = 0; j < n; j++) {
      let dot = 0;
      for (let i = k; i < m; i++) dot += v[i - k] * R.get(i, j);
      const f = 2 * dot / vnorm2;
      for (let i = k; i < m; i++) R.set(i, j, R.get(i, j) - f * v[i - k]);
    }
    // Q ← Q·H
    for (let i = 0; i < m; i++) {
      let dot = 0;
      for (let t = k; t < m; t++) dot += Q.get(i, t) * v[t - k];
      const f = 2 * dot / vnorm2;
      for (let t = k; t < m; t++) Q.set(i, t, Q.get(i, t) - f * v[t - k]);
    }
  }
  return { Q, R };
}

/** 对称矩阵特征分解：循环 Jacobi 旋转。返回按特征值降序的 {values, vectors}（列为特征向量） */
export function jacobiEigen(a, { maxSweeps = 100, tol = 1e-12 } = {}) {
  const n = a.rows;
  if (a.cols !== n || !isSymmetric(a, 1e-10)) throw new Error('jacobiEigen 需要对称矩阵');
  const A = a.clone();
  const V = Mat.identity(n);
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A.get(p, q) * A.get(p, q);
    if (Math.sqrt(off) < tol) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A.get(p, q);
        if (Math.abs(apq) < 1e-300) continue;
        const app = A.get(p, p), aqq = A.get(q, q);
        const tau = (aqq - app) / (2 * apq);
        const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          if (k === p || k === q) continue;
          const akp = A.get(k, p), akq = A.get(k, q);
          A.set(k, p, c * akp - s * akq);
          A.set(p, k, A.get(k, p));
          A.set(k, q, s * akp + c * akq);
          A.set(q, k, A.get(k, q));
        }
        A.set(p, p, c * c * app - 2 * s * c * apq + s * s * aqq);
        A.set(q, q, s * s * app + 2 * s * c * apq + c * c * aqq);
        A.set(p, q, 0); A.set(q, p, 0);
        for (let k = 0; k < n; k++) {
          const vkp = V.get(k, p), vkq = V.get(k, q);
          V.set(k, p, c * vkp - s * vkq);
          V.set(k, q, s * vkp + c * vkq);
        }
      }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => A.get(j, j) - A.get(i, i));
  const values = new Float64Array(idx.map((i) => A.get(i, i)));
  const vectors = new Mat(n, n);
  for (let nc = 0; nc < n; nc++)
    for (let r = 0; r < n; r++) vectors.set(r, nc, V.get(r, idx[nc]));
  return { values, vectors };
}

/** 薄 SVD：A = U·Σ·Vᵀ，经 AᵀA 的对称特征分解得到（小规模矩阵足够且稳定） */
export function svd(a, { tol = 1e-12 } = {}) {
  const ata = matMul(transpose(a), a);
  const { values, vectors: Vfull } = jacobiEigen(ata);
  const n = a.cols;
  const sFull = new Float64Array(n);
  for (let i = 0; i < n; i++) sFull[i] = Math.sqrt(Math.max(values[i], 0));
  const smax = sFull[0] || 0;
  let k = 0;
  for (let i = 0; i < n; i++) if (sFull[i] > smax * tol) k++;
  if (k === 0) k = 1;
  const U = new Mat(a.rows, k);
  for (let j = 0; j < k; j++) {
    for (let i = 0; i < a.rows; i++) {
      let sum = 0;
      for (let t = 0; t < n; t++) sum += a.get(i, t) * Vfull.get(t, j);
      U.set(i, j, sum / sFull[j]);
    }
  }
  return { U, s: sFull.slice(0, k), V: Vfull.colSlice(0, k), rank: k };
}

/** 数值稳定 logΣexp */
export function logsumexp(arr) {
  let m = -Infinity;
  for (const x of arr) if (x > m) m = x;
  if (m === -Infinity) return -Infinity;
  let s = 0;
  for (const x of arr) s += Math.exp(x - m);
  return m + Math.log(s);
}

/** 行 softmax（矩阵按行归一化为概率分布） */
export function softmaxRows(a) {
  const out = new Mat(a.rows, a.cols);
  for (let i = 0; i < a.rows; i++) {
    const r = a.row(i);
    const m = Math.max(...r);
    let s = 0;
    for (let j = 0; j < r.length; j++) { const e = Math.exp(r[j] - m); out.set(i, j, e); s += e; }
    for (let j = 0; j < r.length; j++) out.set(i, j, out.get(i, j) / s);
  }
  return out;
}

export function softmaxArr(arr) {
  const m = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

export function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

export function argmax(arr) {
  let bi = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[bi]) bi = i;
  return bi;
}
