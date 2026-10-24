/**
 * @file classify.mjs — StarForge 监督学习算法全集（分类器 + 线性回归）
 * 作者：classify-dev
 *
 * 统一接口约定：
 * - X 一律为 linalg.mjs 的 Mat（行主序 Float64Array），y 为普通数组
 * - 分类器：fit(X, y) → this；predict(X) → 标签数组；score(X, y) → accuracy
 * - 概率接口：二分类 predictProba → Float64Array(n)（P(y=1)）；
 *   多分类 predictProba / predictLogProba → Mat(n×K)，行和为 1
 * - 线性回归（Ridge/Lasso）：predict → Float64Array
 * - 所有随机性经 createRng(seed) 注入 → 同种子完全可复现；零外部依赖
 *
 * 参考（经典论文，按算法在各类 JSDoc 中重复标注）：
 * - Cox 1958 / Nelder & Wedderburn 1972：逻辑回归（广义线性模型 GLM）
 * - Kingma & Ba 2015：Adam 优化器
 * - Hoerl & Kennard 1970：Ridge 回归；Tibshirani 1996：Lasso（软阈值坐标下降）
 * - Cover & Hart 1967：K 近邻；McCallum & Nigam 1998：朴素贝叶斯
 * - Breiman et al. 1984：CART 分类回归树；Freund & Schapire 1997：AdaBoost；
 *   Zhu et al. 2009：SAMME 多类提升
 * - Cortes & Vapnik 1995：支持向量机；Platt 1998：SMO 序列最小优化；
 *   Hsu & Lin 2002：一对一（OvO）多分类投票
 */
import {
  Mat, matMul, matVec, transpose, solveLinear, createRng, sigmoid, argmax, logsumexp,
} from './linalg.mjs';
import { suite, residualOrthogonality, expectClose } from './verify.mjs';
import {
  loadIris, loadWine, makeMoons, makeBlobs, makeXor, makeRegression,
  trainTestSplit, zscoreFit, zscoreApply, metrics,
} from './datasets.mjs';

// ==================== 内部工具 ====================

/** 前置一列 1 作为偏置：X(n×d) → X̃(n×(d+1))，w[0] 即偏置 */
function addBias(X) {
  const out = new Mat(X.rows, X.cols + 1);
  for (let i = 0; i < X.rows; i++) {
    out.set(i, 0, 1);
    for (let j = 0; j < X.cols; j++) out.set(i, j + 1, X.get(i, j));
  }
  return out;
}

/** 排序去重后的类别列表（保证确定性的标签顺序） */
function uniqueClasses(y) {
  return [...new Set(y)].sort((a, b) => a - b);
}

/** 软阈值算子 S(a, γ) = sign(a)·max(|a|−γ, 0)（Lasso 坐标下降的核心） */
function softThreshold(a, gamma) {
  if (a > gamma) return a - gamma;
  if (a < -gamma) return a + gamma;
  return 0;
}

/**
 * 小批量一阶优化步进（Adam / SGD），供逻辑回归与 Softmax 回归共用。
 * 参考：Kingma & Ba 2015（Adam：一阶矩+二阶矩自适应学习率）
 */
function applyUpdate(w, grad, lr, optimizer, st) {
  if (optimizer === 'sgd') {
    for (let i = 0; i < w.length; i++) w[i] -= lr * grad[i];
    return;
  }
  st.t += 1;
  const c1 = 1 - Math.pow(0.9, st.t);
  const c2 = 1 - Math.pow(0.999, st.t);
  for (let i = 0; i < w.length; i++) {
    st.m[i] = 0.9 * st.m[i] + 0.1 * grad[i];
    st.v[i] = 0.999 * st.v[i] + 0.001 * grad[i] * grad[i];
    w[i] -= lr * (st.m[i] / c1) / (Math.sqrt(st.v[i] / c2) + 1e-8);
  }
}

/** Adam 状态初始化 */
function adamState(n) {
  return { m: new Float64Array(n), v: new Float64Array(n), t: 0 };
}

// ==================== 逻辑回归（二分类） ====================

/**
 * 逻辑回归（二分类，手写梯度的小批量 Adam/SGD）
 *
 * 模型：P(y=1|x) = σ(wᵀx̃)，其中 x̃ = [1, x] 已含偏置列
 * 损失（交叉熵 + L2）：L(w) = −(1/n)Σ[y·log p + (1−y)·log(1−p)] + (l2/2)‖w‖₁ₒ₋₍ᵦ₎‖²
 * 梯度：∇_w L = (1/n)X̃ᵀ(p − y) + l2·w（偏置项不正则）
 *
 * 参考：Cox 1958（逻辑回归）；Nelder & Wedderburn 1972（广义线性模型）；
 *       Kingma & Ba 2015（Adam）
 */
export class LogisticRegression {
  constructor({ l2 = 0, lr = 0.1, epochs = 300, batchSize = 32, optimizer = 'adam', seed = 42 } = {}) {
    this.l2 = l2;
    this.lr = lr;
    this.epochs = epochs;
    this.batchSize = batchSize;
    this.optimizer = optimizer;
    this.seed = seed;
    this.w = null; // Float64Array(d+1)，w[0] 为偏置
  }

  /**
   * @param {Mat} X n×d 特征矩阵
   * @param {number[]} y ∈ {0,1} 的标签
   * @returns {this}
   */
  fit(X, y) {
    const n = X.rows;
    const Xa = addBias(X);
    const p = Xa.cols;
    this.w = new Float64Array(p);
    const rng = createRng(this.seed); // 批次洗牌的唯一随机源 → 可复现
    const st = adamState(p);
    const idx = Array.from({ length: n }, (_, i) => i);
    const grad = new Float64Array(p);
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      rng.shuffle(idx);
      for (let start = 0; start < n; start += this.batchSize) {
        const end = Math.min(start + this.batchSize, n);
        const bs = end - start;
        grad.fill(0);
        for (let bi = start; bi < end; bi++) {
          const i = idx[bi];
          let z = 0;
          for (let j = 0; j < p; j++) z += Xa.get(i, j) * this.w[j];
          const err = sigmoid(z) - y[i];
          for (let j = 0; j < p; j++) grad[j] += Xa.get(i, j) * err;
        }
        for (let j = 0; j < p; j++) grad[j] /= bs;
        for (let j = 1; j < p; j++) grad[j] += this.l2 * this.w[j]; // 不惩罚偏置
        applyUpdate(this.w, grad, this.lr, this.optimizer, st);
      }
    }
    return this;
  }

  /** @returns {Float64Array} 每个样本 P(y=1) */
  predictProba(X) {
    const Xa = addBias(X);
    const p = Xa.cols;
    const out = new Float64Array(X.rows);
    for (let i = 0; i < X.rows; i++) {
      let z = 0;
      for (let j = 0; j < p; j++) z += Xa.get(i, j) * this.w[j];
      out[i] = sigmoid(z);
    }
    return out;
  }

  predict(X) {
    return Array.from(this.predictProba(X), (v) => (v >= 0.5 ? 1 : 0));
  }

  score(X, y) {
    return metrics.accuracy(y, this.predict(X));
  }
}

// ==================== Softmax 回归（多分类） ====================

/**
 * Softmax 回归（多项分布 / multinomial logistic regression，真正的多类 softmax）
 *
 * 模型：P(y=k|x) = softmax(Wᵀx̃)_k，W ∈ R^{(d+1)×K}
 * 损失：L(W) = −(1/n)Σ_i log P(y_i|x_i) + (l2/2)‖W‖²（偏置行不正则）
 * 梯度：∇_W L = (1/n)X̃ᵀ(P − Y_onehot) + l2·W
 *
 * 参考：Nelder & Wedderburn 1972；Bishop 2006《Pattern Recognition and ML》§4.3
 */
export class SoftmaxRegression {
  constructor({ l2 = 0, lr = 0.1, epochs = 300, batchSize = 32, optimizer = 'adam', seed = 42 } = {}) {
    this.l2 = l2;
    this.lr = lr;
    this.epochs = epochs;
    this.batchSize = batchSize;
    this.optimizer = optimizer;
    this.seed = seed;
    this.classes = [];
    this.W = null; // Float64Array((d+1)·K)，行主序 (d+1)×K
  }

  fit(X, y) {
    const n = X.rows;
    this.classes = uniqueClasses(y);
    const K = this.classes.length;
    const clsIndex = new Map(this.classes.map((c, i) => [c, i]));
    const Xa = addBias(X);
    const p = Xa.cols;
    const pw = p * K;
    this.W = new Float64Array(pw);
    const rng = createRng(this.seed);
    const st = adamState(pw);
    const idx = Array.from({ length: n }, (_, i) => i);
    const grad = new Float64Array(pw);
    const probs = new Float64Array(K);
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      rng.shuffle(idx);
      for (let start = 0; start < n; start += this.batchSize) {
        const end = Math.min(start + this.batchSize, n);
        const bs = end - start;
        grad.fill(0);
        for (let bi = start; bi < end; bi++) {
          const i = idx[bi];
          // 前向：Z = x̃·W → softmax
          for (let k = 0; k < K; k++) {
            let z = 0;
            for (let j = 0; j < p; j++) z += Xa.get(i, j) * this.W[j * K + k];
            probs[k] = z;
          }
          const mx = Math.max(...probs);
          let s = 0;
          for (let k = 0; k < K; k++) { probs[k] = Math.exp(probs[k] - mx); s += probs[k]; }
          for (let k = 0; k < K; k++) probs[k] /= s;
          // 反向：(P − Y)_k 经 x̃ 外积累积到梯度
          probs[clsIndex.get(y[i])] -= 1;
          for (let j = 0; j < p; j++) {
            const xv = Xa.get(i, j);
            if (xv === 0) continue;
            for (let k = 0; k < K; k++) grad[j * K + k] += xv * probs[k];
          }
        }
        for (let t = 0; t < pw; t++) grad[t] /= bs;
        for (let j = 1; j < p; j++) for (let k = 0; k < K; k++) grad[j * K + k] += this.l2 * this.W[j * K + k];
        applyUpdate(this.W, grad, this.lr, this.optimizer, st);
      }
    }
    return this;
  }

  /** @returns {Mat} n×K 概率矩阵，每行和为 1 */
  predictProba(X) {
    const n = X.rows;
    const K = this.classes.length;
    const Xa = addBias(X);
    const p = Xa.cols;
    const out = new Mat(n, K);
    for (let i = 0; i < n; i++) {
      let mx = -Infinity;
      for (let k = 0; k < K; k++) {
        let z = 0;
        for (let j = 0; j < p; j++) z += Xa.get(i, j) * this.W[j * K + k];
        out.set(i, k, z);
        if (z > mx) mx = z;
      }
      let s = 0;
      for (let k = 0; k < K; k++) { const e = Math.exp(out.get(i, k) - mx); out.set(i, k, e); s += e; }
      for (let k = 0; k < K; k++) out.set(i, k, out.get(i, k) / s);
    }
    return out;
  }

  predict(X) {
    const P = this.predictProba(X);
    return Array.from({ length: X.rows }, (_, i) => this.classes[argmax(P.row(i))]);
  }

  score(X, y) {
    return metrics.accuracy(y, this.predict(X));
  }
}

// ==================== Ridge 回归（闭式解） ====================

/**
 * Ridge 回归（Tikhonov 正则化最小二乘，闭式解）
 *
 * 目标：min_w ‖y − X̃w‖² + λ‖w‖²（偏置不正则）
 * 解：w = (X̃ᵀX̃ + λD)⁻¹X̃ᵀy，D = diag(0,1,1,…,1)
 * 一阶条件（λ=0 时）：X̃ᵀ(y − X̃w) = 0（残差与设计矩阵列正交）
 *
 * 参考：Hoerl & Kennard 1970（Ridge regression: biased estimation for nonorthogonal problems）
 */
export class Ridge {
  constructor({ lambda = 1.0 } = {}) {
    this.lambda = lambda;
    this.w = null; // Float64Array(d+1)
  }

  fit(X, y) {
    const Xa = addBias(X);
    const A = matMul(transpose(Xa), Xa);
    const p = A.rows;
    for (let j = 1; j < p; j++) A.set(j, j, A.get(j, j) + this.lambda); // 偏置列（第 0 个）不正则
    const b = matVec(transpose(Xa), Float64Array.from(y));
    this.w = solveLinear(A, b);
    return this;
  }

  /** @returns {Float64Array} 预测值 */
  predict(X) {
    return matVec(addBias(X), this.w);
  }

  score(X, y) {
    return metrics.r2(Array.from(y), Array.from(this.predict(X)));
  }
}

// ==================== Lasso 回归（坐标下降） ====================

/**
 * Lasso 回归（坐标下降 + 软阈值）
 *
 * 目标：min_w (1/2n)‖y − X̃w‖² + λΣ_{j≥1}|w_j|（即 MSE + λ‖w‖₁，差常数因子 1/2 不影响最优解）
 * 坐标更新（其余坐标固定时的一维精确最小化）：
 *   w_j ← S(ρ_j / z_j, λn / z_j)，其中 ρ_j = Σ_i x̃_ij(y_i − ŷ_i⁽⁻ʲ⁾)，z_j = Σ_i x̃_ij²
 *   S 为软阈值算子；偏置列（j=0）不收缩
 * 坐标下降每步沿坐标精确最小化 → 目标函数值单调不增（selfCheck 验证）
 *
 * 参考：Tibshirani 1996（Regression shrinkage and selection via the lasso）；
 *       Friedman et al. 2010（坐标下降路径算法）
 */
export class Lasso {
  constructor({ lambda = 0.1, maxIter = 1000, tol = 1e-6 } = {}) {
    this.lambda = lambda;
    this.maxIter = maxIter;
    this.tol = tol;
    this.w = null;
    this.lossCurve = []; // 每轮完整坐标扫描后的目标函数值
  }

  /** 目标函数值（1/2n）‖y−X̃w‖² + λ‖w‖₁ */
  objective(X, y, w = this.w) {
    const Xa = addBias(X);
    let s = 0;
    for (let i = 0; i < Xa.rows; i++) {
      let pred = 0;
      for (let j = 0; j < Xa.cols; j++) pred += Xa.get(i, j) * w[j];
      s += (y[i] - pred) ** 2;
    }
    let l1 = 0;
    for (let j = 1; j < w.length; j++) l1 += Math.abs(w[j]);
    return s / (2 * Xa.rows) + this.lambda * l1;
  }

  fit(X, y) {
    const Xa = addBias(X);
    const n = Xa.rows;
    const p = Xa.cols;
    const cols = [];
    const z = new Float64Array(p);
    for (let j = 0; j < p; j++) {
      const c = Xa.col(j);
      cols.push(c);
      let s = 0;
      for (let i = 0; i < n; i++) s += c[i] * c[i];
      z[j] = s;
    }
    const w = new Float64Array(p);
    const pred = new Float64Array(n); // 当前预测 X̃w（随坐标更新增量维护）
    const yv = Float64Array.from(y);
    this.lossCurve = [];
    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxDelta = 0;
      for (let j = 0; j < p; j++) {
        const cj = cols[j];
        // ρ_j = Σ_i x̃_ij·(y_i − ŷ_i + w_j·x̃_ij)（去掉第 j 个坐标贡献的偏残差内积）
        let rho = 0;
        for (let i = 0; i < n; i++) rho += cj[i] * (yv[i] - pred[i] + w[j] * cj[i]);
        let wNew = z[j] > 0 ? rho / z[j] : 0;
        if (j > 0 && z[j] > 0) wNew = softThreshold(wNew, (this.lambda * n) / z[j]);
        const d = wNew - w[j];
        if (d !== 0) {
          for (let i = 0; i < n; i++) pred[i] += cj[i] * d;
          if (Math.abs(d) > maxDelta) maxDelta = Math.abs(d);
          w[j] = wNew;
        }
      }
      this.lossCurve.push(this.objective(X, y, w));
      if (maxDelta < this.tol) break; // 坐标变化足够小 → 收敛
    }
    this.w = w;
    return this;
  }

  predict(X) {
    return matVec(addBias(X), this.w);
  }

  score(X, y) {
    return metrics.r2(Array.from(y), Array.from(this.predict(X)));
  }
}

// ==================== K 近邻 ====================

/**
 * K 近邻分类器（惰性学习，无显式训练）
 *
 * 预测：欧氏距离找 k 个最近邻，多数投票（weighted=true 时按 1/d 加权投票）
 * 不变式：k=1 时训练集最近邻即样本自身 → 训练准确率恒为 1
 *
 * 参考：Cover & Hart 1967（Nearest neighbor pattern classification）
 */
export class KNN {
  constructor({ k = 3, weighted = false } = {}) {
    this.k = k;
    this.weighted = weighted;
    this._rows = [];
    this.y = [];
    this.classes = [];
  }

  fit(X, y) {
    this._rows = Array.from({ length: X.rows }, (_, i) => X.row(i));
    this.y = Array.from(y);
    this.classes = uniqueClasses(y);
    return this;
  }

  predict(X) {
    const k = Math.min(this.k, this._rows.length);
    const out = new Array(X.rows);
    for (let i = 0; i < X.rows; i++) {
      const q = X.row(i);
      // 计算全部距离后按 (距离, 索引) 排序 → 平局确定性取索引小者
      const dists = this._rows.map((r, idx) => {
        let s = 0;
        for (let j = 0; j < q.length; j++) { const d = q[j] - r[j]; s += d * d; }
        return { d: Math.sqrt(s), idx };
      });
      dists.sort((a, b) => a.d - b.d || a.idx - b.idx);
      const votes = new Float64Array(this.classes.length);
      const clsIndex = new Map(this.classes.map((c, i) => [c, i]));
      for (let t = 0; t < k; t++) {
        const { d, idx } = dists[t];
        const w = this.weighted ? 1 / (d + 1e-8) : 1; // 1/d 加权（零距离加 ε 保护）
        votes[clsIndex.get(this.y[idx])] += w;
      }
      out[i] = this.classes[argmax(votes)]; // 平局取类索引小者（argmax 首个最大）
    }
    return out;
  }

  score(X, y) {
    return metrics.accuracy(y, this.predict(X));
  }
}

// ==================== 高斯朴素贝叶斯 ====================

/**
 * 高斯朴素贝叶斯：连续特征，各类特征服从独立高斯分布
 *
 * 后验：P(c|x) ∝ π_c · Π_j N(x_j | μ_cj, σ²_cj)
 * 对数后验：log P(c|x) = log π_c − ½Σ_j[log(2πσ²_cj) + (x_j−μ_cj)²/σ²_cj] + const
 * 方差平滑：σ² ← σ² + varSmoothing·max_j(Var(全体特征 j))（防零方差除零）
 *
 * 参考：McCallum & Nigam 1998（A comparison of event models for naive bayes text classification）
 */
export class GaussianNB {
  constructor({ varSmoothing = 1e-9 } = {}) {
    this.varSmoothing = varSmoothing;
    this.classes = [];
  }

  fit(X, y) {
    const n = X.rows;
    const d = X.cols;
    this.classes = uniqueClasses(y);
    const K = this.classes.length;
    // 全局特征均值/方差（varSmoothing 以此为比例基准）
    const gMean = new Float64Array(d);
    const gVar = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X.get(i, j);
      gMean[j] = s / n;
      let v = 0;
      for (let i = 0; i < n; i++) v += (X.get(i, j) - gMean[j]) ** 2;
      gVar[j] = v / n;
    }
    const eps = this.varSmoothing * Math.max(0, ...gVar);
    this.logPrior = new Float64Array(K);
    this.means = new Mat(K, d);
    this.vars = new Mat(K, d);
    for (let c = 0; c < K; c++) {
      const idx = [];
      for (let i = 0; i < n; i++) if (y[i] === this.classes[c]) idx.push(i);
      this.logPrior[c] = Math.log(idx.length / n);
      for (let j = 0; j < d; j++) {
        let s = 0;
        for (const i of idx) s += X.get(i, j);
        const mu = s / idx.length;
        let v = 0;
        for (const i of idx) v += (X.get(i, j) - mu) ** 2;
        this.means.set(c, j, mu);
        this.vars.set(c, j, v / idx.length + eps);
      }
    }
    return this;
  }

  /** @returns {Mat} n×K 归一化对数后验（每行 exp 后和为 1） */
  predictLogProba(X) {
    const n = X.rows;
    const d = X.cols;
    const K = this.classes.length;
    const out = new Mat(n, K);
    for (let i = 0; i < n; i++) {
      const row = new Float64Array(K);
      for (let c = 0; c < K; c++) {
        let ll = this.logPrior[c];
        for (let j = 0; j < d; j++) {
          const v = this.vars.get(c, j);
          const diff = X.get(i, j) - this.means.get(c, j);
          ll += -0.5 * (Math.log(2 * Math.PI * v) + (diff * diff) / v);
        }
        row[c] = ll;
      }
      const lse = logsumexp(row);
      for (let c = 0; c < K; c++) out.set(i, c, row[c] - lse);
    }
    return out;
  }

  predict(X) {
    const lp = this.predictLogProba(X);
    return Array.from({ length: X.rows }, (_, i) => this.classes[argmax(lp.row(i))]);
  }

  score(X, y) {
    return metrics.accuracy(y, this.predict(X));
  }
}

// ==================== 多项式朴素贝叶斯 ====================

/**
 * 多项式朴素贝叶斯：计数特征（如词频），多项分布事件模型
 *
 * 对数似然项：Σ_j x_ij·log[(fc_cj + α)/(tot_c + α·J)]，fc 为类内特征计数，J 为特征数
 * 后验 ∝ exp(类对数先验 + 上述项)，逐行 logsumexp 归一
 *
 * 参考：McCallum & Nigam 1998（多项模型）
 */
export class MultinomialNB {
  constructor({ alpha = 1 } = {}) {
    this.alpha = alpha;
    this.classes = [];
  }

  fit(X, y) {
    const n = X.rows;
    const d = X.cols;
    this.classes = uniqueClasses(y);
    const K = this.classes.length;
    this.featureLogProb = new Mat(K, d);
    this.classLogPrior = new Float64Array(K);
    for (let c = 0; c < K; c++) {
      let nc = 0;
      const fc = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        if (y[i] !== this.classes[c]) continue;
        nc++;
        for (let j = 0; j < d; j++) fc[j] += X.get(i, j);
      }
      let tot = 0;
      for (let j = 0; j < d; j++) tot += fc[j];
      const denom = tot + this.alpha * d;
      for (let j = 0; j < d; j++) this.featureLogProb.set(c, j, Math.log((fc[j] + this.alpha) / denom));
      this.classLogPrior[c] = Math.log(nc / n);
    }
    return this;
  }

  /** @returns {Mat} n×K 归一化对数后验 */
  predictLogProba(X) {
    const n = X.rows;
    const K = this.classes.length;
    const out = new Mat(n, K);
    for (let i = 0; i < n; i++) {
      const row = new Float64Array(K);
      for (let c = 0; c < K; c++) {
        let s = this.classLogPrior[c];
        for (let j = 0; j < X.cols; j++) s += X.get(i, j) * this.featureLogProb.get(c, j);
        row[c] = s;
      }
      const lse = logsumexp(row);
      for (let c = 0; c < K; c++) out.set(i, c, row[c] - lse);
    }
    return out;
  }

  predict(X) {
    const lp = this.predictLogProba(X);
    return Array.from({ length: X.rows }, (_, i) => this.classes[argmax(lp.row(i))]);
  }

  score(X, y) {
    return metrics.accuracy(y, this.predict(X));
  }
}

// ==================== 伯努利朴素贝叶斯 ====================

/**
 * 伯努利朴素贝叶斯：二值特征（x > binarize 视为 1），伯努利事件模型
 *
 * 类条件概率：p_cj = (类内特征 j 为 1 的计数 + α)/(n_c + 2α)
 * 对数似然项：Σ_j [x̃_ij·log p_cj + (1−x̃_ij)·log(1−p_cj)]
 *
 * 参考：McCallum & Nigam 1998（伯努利模型）
 */
export class BernoulliNB {
  constructor({ alpha = 1, binarize = 0 } = {}) {
    this.alpha = alpha;
    this.binarize = binarize;
    this.classes = [];
  }

  fit(X, y) {
    const n = X.rows;
    const d = X.cols;
    this.classes = uniqueClasses(y);
    const K = this.classes.length;
    this.featureLogProb = new Mat(K, d);
    this.negFeatureLogProb = new Mat(K, d);
    this.classLogPrior = new Float64Array(K);
    for (let c = 0; c < K; c++) {
      let nc = 0;
      const cnt = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        if (y[i] !== this.classes[c]) continue;
        nc++;
        for (let j = 0; j < d; j++) if (X.get(i, j) > this.binarize) cnt[j]++;
      }
      for (let j = 0; j < d; j++) {
        const p = (cnt[j] + this.alpha) / (nc + 2 * this.alpha);
        this.featureLogProb.set(c, j, Math.log(p));
        this.negFeatureLogProb.set(c, j, Math.log(1 - p));
      }
      this.classLogPrior[c] = Math.log(nc / n);
    }
    return this;
  }

  /** @returns {Mat} n×K 归一化对数后验 */
  predictLogProba(X) {
    const n = X.rows;
    const K = this.classes.length;
    const out = new Mat(n, K);
    for (let i = 0; i < n; i++) {
      const row = new Float64Array(K);
      for (let c = 0; c < K; c++) {
        let s = this.classLogPrior[c];
        for (let j = 0; j < X.cols; j++) {
          s += X.get(i, j) > this.binarize
            ? this.featureLogProb.get(c, j)
            : this.negFeatureLogProb.get(c, j);
        }
        row[c] = s;
      }
      const lse = logsumexp(row);
      for (let c = 0; c < K; c++) out.set(i, c, row[c] - lse);
    }
    return out;
  }

  predict(X) {
    const lp = this.predictLogProba(X);
    return Array.from({ length: X.rows }, (_, i) => this.classes[argmax(lp.row(i))]);
  }

  score(X, y) {
    return metrics.accuracy(y, this.predict(X));
  }
}

// ==================== CART 决策树 ====================

/** 树节点（叶节点存类别分布，内部节点存 (特征, 阈值) 二分） */
class TreeNode {
  constructor() {
    this.leaf = false;
    this.feature = -1;
    this.threshold = 0;
    this.left = null;
    this.right = null;
    this.counts = null; // 节点内（加权）类别分布
    this.wsum = 0;
    this.pred = 0; // 叶预测标签（分布 argmax，平局取小类索引）
  }
}

/**
 * CART 分类树（Breiman 1984）
 *
 * 递归二分：对每个连续特征按取值排序，在相邻不同值的中点集合中扫描候选阈值，
 * 选加权不纯度下降最大的 (特征, 阈值)；叶节点存类别分布，预测取 argmax。
 * 不纯度：gini = 1 − Σp_k²（或 entropy = −Σp_k·log p_k）
 * 停止条件：深度达 maxDepth / 样本数不足 minSamplesSplit / 节点纯净 / 无可致纯分裂
 * 支持样本权重（AdaBoost 加权桩用）：不纯度按权重统计。
 *
 * 参考：Breiman, Friedman, Olshen & Stone 1984《Classification and Regression Trees》
 */
export class DecisionTreeClassifier {
  constructor({ maxDepth = 5, minSamplesSplit = 2, criterion = 'gini' } = {}) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.criterion = criterion;
    this.root = null;
    this.classes = [];
  }

  /**
   * @param {Mat} X
   * @param {number[]} y
   * @param {Float64Array} [sampleWeights] 可选样本权重（AdaBoost 用）
   */
  fit(X, y, sampleWeights = null) {
    this.classes = uniqueClasses(y);
    const nClasses = this.classes.length;
    const clsIndex = new Map(this.classes.map((c, i) => [c, i]));
    const yi = Array.from(y, (v) => clsIndex.get(v));
    this._clsIndex = clsIndex;
    this.root = this._build(X, yi, Array.from({ length: X.rows }, (_, i) => i), 0, sampleWeights, nClasses);
    return this;
  }

  _impurity(counts, w) {
    if (w <= 0) return 0;
    if (this.criterion === 'entropy') {
      let e = 0;
      for (const c of counts) if (c > 0) { const p = c / w; e -= p * Math.log(p); }
      return e;
    }
    let s = 0;
    for (const c of counts) { const p = c / w; s += p * p; }
    return 1 - s;
  }

  _build(X, yi, idx, depth, sampleWeights, nClasses) {
    const node = new TreeNode();
    const counts = new Float64Array(nClasses);
    let wsum = 0;
    for (const i of idx) {
      const w = sampleWeights ? sampleWeights[i] : 1;
      counts[yi[i]] += w;
      wsum += w;
    }
    node.counts = counts;
    node.wsum = wsum;
    node.pred = this.classes[argmax(counts)];
    let nonZero = 0;
    for (const c of counts) if (c > 0) nonZero++;
    if (depth >= this.maxDepth || idx.length < this.minSamplesSplit || nonZero <= 1) {
      node.leaf = true;
      return node;
    }
    const best = this._bestSplit(X, yi, idx, counts, wsum, sampleWeights, nClasses);
    if (!best) {
      node.leaf = true;
      return node;
    }
    const leftIdx = [];
    const rightIdx = [];
    for (const i of idx) (X.get(i, best.feature) <= best.threshold ? leftIdx : rightIdx).push(i);
    node.feature = best.feature;
    node.threshold = best.threshold;
    node.left = this._build(X, yi, leftIdx, depth + 1, sampleWeights, nClasses);
    node.right = this._build(X, yi, rightIdx, depth + 1, sampleWeights, nClasses);
    return node;
  }

  /** 扫描全部特征的相邻中点阈值，返回加权不纯度最小的分裂 */
  _bestSplit(X, yi, idx, totalCounts, totalW, sampleWeights, nClasses) {
    const d = X.cols;
    const n = idx.length;
    const parentImp = this._impurity(totalCounts, totalW);
    const left = new Float64Array(nClasses);
    const right = new Float64Array(nClasses);
    let best = null;
    for (let j = 0; j < d; j++) {
      const order = Array.from(idx).sort((a, b) => X.get(a, j) - X.get(b, j) || a - b);
      left.fill(0);
      let lw = 0;
      for (let s = 0; s < n - 1; s++) {
        const i = order[s];
        const wi = sampleWeights ? sampleWeights[i] : 1;
        left[yi[i]] += wi;
        lw += wi;
        if (X.get(i, j) >= X.get(order[s + 1], j)) continue; // 相邻同值不构成分裂
        const rw = totalW - lw;
        let imp = (lw / totalW) * this._impurity(left, lw);
        if (rw > 0) {
          for (let c = 0; c < nClasses; c++) right[c] = totalCounts[c] - left[c];
          imp += (rw / totalW) * this._impurity(right, rw);
        }
        if (!best || imp < best.imp) {
          best = { imp, feature: j, threshold: (X.get(i, j) + X.get(order[s + 1], j)) / 2 };
        }
      }
    }
    if (!best || best.imp >= parentImp - 1e-12) return null; // 无可致纯分裂
    return best;
  }

  _leafFor(X, i) {
    let node = this.root;
    while (!node.leaf) {
      node = X.get(i, node.feature) <= node.threshold ? node.left : node.right;
    }
    return node;
  }

  predict(X) {
    return Array.from({ length: X.rows }, (_, i) => this._leafFor(X, i).pred);
  }

  score(X, y) {
    return metrics.accuracy(y, this.predict(X));
  }

  /** 叶节点数量 */
  countLeaves() {
    const stack = [this.root];
    let cnt = 0;
    while (stack.length) {
      const nd = stack.pop();
      if (nd.leaf) cnt++;
      else { stack.push(nd.left); stack.push(nd.right); }
    }
    return cnt;
  }

  /**
   * 训练集上的加权叶节点 Gini 不纯度：Σ_leaves (n_leaf/N)·gini(leaf)
   * 深树过拟合 → 该值趋近 0（叶纯净）
   */
  giniImpurity(X, y) {
    const N = X.rows;
    if (N === 0) return 0;
    const leafCount = new Map();
    for (let i = 0; i < N; i++) {
      const leaf = this._leafFor(X, i);
      leafCount.set(leaf, (leafCount.get(leaf) || 0) + 1);
    }
    let acc = 0;
    for (const [leaf, cnt] of leafCount) {
      const w = leaf.wsum > 0 ? leaf.wsum : cnt;
      acc += (cnt / N) * this._impurity(leaf.counts, w);
    }
    return acc;
  }
}

// ==================== AdaBoost（SAMME） ====================

/**
 * AdaBoost 分类器（SAMME 多类版，弱学习器为深度 1 决策树桩）
 *
 * 算法：初始化 w_i = 1/n；每轮用加权不纯度训练一个树桩 h_m，
 *   加权错误 err_m = Σw_i·I(h_m≠y_i)，α_m = log((1−err_m)/err_m) + log(K−1)，
 *   w_i ← w_i·exp(α_m·I(h_m≠y_i)) 后归一；
 * 预测：argmax_c Σ_m α_m·I(h_m(x) = c)
 *
 * 参考：Freund & Schapire 1997（A decision-theoretic generalization of on-line learning）；
 *       Zhu et al. 2009（Multi-class AdaBoost / SAMME）
 */
export class AdaBoostClassifier {
  constructor({ nEstimators = 50 } = {}) {
    this.nEstimators = nEstimators;
    this.estimators = []; // {stump, alpha}
    this.classes = [];
  }

  fit(X, y) {
    const n = X.rows;
    this.classes = uniqueClasses(y);
    const K = this.classes.length;
    const w = new Float64Array(n).fill(1 / n);
    this.estimators = [];
    for (let m = 0; m < this.nEstimators; m++) {
      const stump = new DecisionTreeClassifier({ maxDepth: 1, minSamplesSplit: 2, criterion: 'gini' })
        .fit(X, y, w);
      const pred = stump.predict(X);
      let err = 0;
      for (let i = 0; i < n; i++) if (pred[i] !== y[i]) err += w[i];
      if (err <= 1e-10) {
        this.estimators.push({ stump, alpha: 10 }); // 完美弱学习器：给大权重后停止
        break;
      }
      const alpha = Math.log((1 - err) / err) + Math.log(K - 1);
      if (alpha <= 0) break; // 弱学习器不优于随机猜，停止提升
      for (let i = 0; i < n; i++) if (pred[i] !== y[i]) w[i] *= Math.exp(alpha);
      let sw = 0;
      for (let i = 0; i < n; i++) sw += w[i];
      for (let i = 0; i < n; i++) w[i] /= sw;
      this.estimators.push({ stump, alpha });
    }
    return this;
  }

  predict(X) {
    const n = X.rows;
    const K = this.classes.length;
    const scores = Array.from({ length: n }, () => new Float64Array(K));
    const clsIndex = new Map(this.classes.map((c, i) => [c, i]));
    for (const { stump, alpha } of this.estimators) {
      const pred = stump.predict(X);
      for (let i = 0; i < n; i++) scores[i][clsIndex.get(pred[i])] += alpha;
    }
    return Array.from({ length: n }, (_, i) => this.classes[argmax(scores[i])]);
  }

  score(X, y) {
    return metrics.accuracy(y, this.predict(X));
  }
}

// ==================== 支持向量机（SMO） ====================

/**
 * 支持向量机分类器（C-SVM + SMO 求解，多分类 OvO 投票）
 *
 * 二分类对偶问题：max_α Σα_i − ½ΣΣ α_iα_j y_iy_j K(x_i,x_j)  s.t. 0≤α_i≤C, Σα_iy_i=0
 *   y ∈ {−1,+1}（内部把 {0,1} 映射为 {−1,+1}）；决策函数 f(x) = Σα_iy_iK(x_i,x)+b
 * SMO（Platt 1998）：外层找违反 KKT 的样本 i（y_iE_i 越界且 α_i 未触界），
 *   内环选 |E_i−E_j| 最大的 j，两变量闭式解裁剪到 [L,H]，闭式更新 b。
 * 多分类：一对一（OvO）训练 K(K−1)/2 个二分类器，投票取多数（Hsu & Lin 2002）。
 * KKT 最优性：α_i=0 → y_if(x_i)≥1；0<α_i<C → y_if(x_i)=1；α_i=C → y_if(x_i)≤1。
 *
 * 参考：Cortes & Vapnik 1995；Platt 1998（Sequential Minimal Optimization）；
 *       Hsu & Lin 2002（A comparison of methods for multiclass support vector machines）
 */
export class SVMClassifier {
  constructor({ C = 1.0, kernel = 'rbf', gamma = 'scale', tol = 1e-3, maxPasses = 5 } = {}) {
    this.C = C;
    this.kernel = kernel;
    this.gamma = gamma;
    this.tol = tol;
    this.maxPasses = maxPasses;
    this.classes = [];
    this.models = []; // 二分类模型 {alpha, t, rows, b, gamma}（OvO 时附 pair）
  }

  _kernel(a, b, gamma) {
    if (this.kernel === 'linear') {
      let s = 0;
      for (let j = 0; j < a.length; j++) s += a[j] * b[j];
      return s;
    }
    // rbf
    let s = 0;
    for (let j = 0; j < a.length; j++) { const d = a[j] - b[j]; s += d * d; }
    return Math.exp(-gamma * s);
  }

  /** gamma='scale' → 1/(d·Var(X))（sklearn 同款约定）；数值则直接用 */
  _gammaValue(X) {
    if (typeof this.gamma === 'number') return this.gamma;
    const n = X.rows;
    const d = X.cols;
    let mean = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean += X.get(i, j);
    mean /= n * d;
    let v = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) v += (X.get(i, j) - mean) ** 2;
    v /= n * d;
    return v > 0 ? 1 / (d * v) : 1;
  }

  /** SMO 求解单个二分类问题，t ∈ {−1,+1} */
  _smo(X, t) {
    const n = X.rows;
    const C = this.C;
    const tol = this.tol;
    const gamma = this._gammaValue(X);
    const rows = Array.from({ length: n }, (_, i) => X.row(i));
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const k = this._kernel(rows[i], rows[j], gamma);
        K[i * n + j] = k;
        K[j * n + i] = k;
      }
    }
    const alpha = new Float64Array(n);
    const fcache = new Float64Array(n); // f(i) = Σα_j t_j K_ij + b（增量维护）
    let b = 0;
    // 每轮外层循环前全量重建 f 缓存，压制增量累加的浮点漂移
    const rebuildF = () => {
      for (let k = 0; k < n; k++) {
        let s = b;
        for (let j = 0; j < n; j++) if (alpha[j] !== 0) s += alpha[j] * t[j] * K[k * n + j];
        fcache[k] = s;
      }
    };
    const examine = (i) => {
      const Ei = fcache[i] - t[i];
      const ri = t[i] * Ei;
      // KKT 违反判定：负间隔侧 α 未到上界，或正间隔侧 α 未到下界
      if (!((ri < -tol && alpha[i] < C - 1e-12) || (ri > tol && alpha[i] > 1e-12))) return false;
      // 内环：优先在非边界 α 中选 |Ei−Ej| 最大者，否则退化到全体
      let j = -1;
      let bestDelta = -1;
      for (let k = 0; k < n; k++) {
        if (k === i) continue;
        if (alpha[k] > 1e-12 && alpha[k] < C - 1e-12) {
          const dk = Math.abs(Ei - (fcache[k] - t[k]));
          if (dk > bestDelta) { bestDelta = dk; j = k; }
        }
      }
      if (j < 0) {
        for (let k = 0; k < n; k++) {
          if (k === i) continue;
          const dk = Math.abs(Ei - (fcache[k] - t[k]));
          if (dk > bestDelta) { bestDelta = dk; j = k; }
        }
      }
      if (j < 0 || bestDelta <= 1e-12) return false;
      const Ej = fcache[j] - t[j];
      const aiOld = alpha[i];
      const ajOld = alpha[j];
      let L;
      let H;
      if (t[i] !== t[j]) { L = Math.max(0, ajOld - aiOld); H = Math.min(C, C + ajOld - aiOld); }
      else { L = Math.max(0, aiOld + ajOld - C); H = Math.min(C, aiOld + ajOld); }
      if (H - L < 1e-12) return false;
      const eta = 2 * K[i * n + j] - K[i * n + i] - K[j * n + j];
      if (eta >= -1e-12) return false; // 非严格凹 → 跳过
      let aj = ajOld - (t[j] * (Ei - Ej)) / eta;
      if (aj > H) aj = H;
      else if (aj < L) aj = L;
      if (Math.abs(aj - ajOld) < 1e-8) return false;
      const ai = aiOld + t[i] * t[j] * (ajOld - aj);
      const bOld = b;
      const b1 = b - Ei - t[i] * (ai - aiOld) * K[i * n + i] - t[j] * (aj - ajOld) * K[i * n + j];
      const b2 = b - Ej - t[i] * (ai - aiOld) * K[i * n + j] - t[j] * (aj - ajOld) * K[j * n + j];
      if (ai > 1e-12 && ai < C - 1e-12) b = b1;
      else if (aj > 1e-12 && aj < C - 1e-12) b = b2;
      else b = (b1 + b2) / 2;
      alpha[i] = ai;
      alpha[j] = aj;
      const dAi = ai - aiOld;
      const dAj = aj - ajOld;
      const db = b - bOld;
      for (let k = 0; k < n; k++) {
        fcache[k] += t[i] * dAi * K[i * n + k] + t[j] * dAj * K[j * n + k] + db;
      }
      return true;
    };
    let examineAll = true;
    let noChange = 0;
    while (noChange < this.maxPasses) {
      rebuildF();
      let changed = 0;
      if (examineAll) {
        for (let i = 0; i < n; i++) if (examine(i)) changed++;
        examineAll = false;
      } else {
        for (let i = 0; i < n; i++) {
          if (alpha[i] > 1e-12 && alpha[i] < C - 1e-12 && examine(i)) changed++;
        }
        if (changed === 0) examineAll = true;
      }
      if (changed === 0) noChange++;
      else noChange = 0;
    }
    return { alpha, t, rows, b, gamma };
  }

  fit(X, y) {
    this.classes = uniqueClasses(y);
    const K = this.classes.length;
    if (K < 2) throw new Error('SVMClassifier 至少需要 2 个类别');
    this.models = [];
    if (K === 2) {
      const t = y.map((v) => (v === this.classes[1] ? 1 : -1));
      this.models.push(this._smo(X, t));
    } else {
      // OvO：每两类训练一个二分类器（Hsu & Lin 2002）
      for (let a = 0; a < K; a++) {
        for (let bIdx = a + 1; bIdx < K; bIdx++) {
          const rowsIdx = [];
          const t = [];
          for (let i = 0; i < X.rows; i++) {
            if (y[i] === this.classes[a] || y[i] === this.classes[bIdx]) {
              rowsIdx.push(i);
              t.push(y[i] === this.classes[bIdx] ? 1 : -1);
            }
          }
          const Xsub = new Mat(rowsIdx.length, X.cols);
          rowsIdx.forEach((src, dst) => {
            for (let j = 0; j < X.cols; j++) Xsub.set(dst, j, X.get(src, j));
          });
          this.models.push({ ...this._smo(Xsub, t), pair: [this.classes[a], this.classes[bIdx]] });
        }
      }
    }
    return this;
  }

  /** OvO 投票：返回 n×K 票数矩阵 */
  _votes(X) {
    const n = X.rows;
    const K = this.classes.length;
    const votes = Array.from({ length: n }, () => new Float64Array(K));
    for (const m of this.models) {
      const ia = this.classes.indexOf(m.pair[0]);
      const ib = this.classes.indexOf(m.pair[1]);
      for (let i = 0; i < n; i++) {
        const x = X.row(i);
        let s = m.b;
        for (let j = 0; j < m.alpha.length; j++) {
          if (m.alpha[j] !== 0) s += m.alpha[j] * m.t[j] * this._kernel(x, m.rows[j], m.gamma);
        }
        votes[i][s >= 0 ? ib : ia] += 1;
      }
    }
    return votes;
  }

  /**
   * 决策函数值。
   * 二分类：f(x) = Σα_iy_iK(x_i,x)+b（符号即类别）；
   * 多分类：OvO 最高票与次高票的票数差（置信边际）。
   * @returns {Float64Array}
   */
  decisionFunction(X) {
    const n = X.rows;
    const out = new Float64Array(n);
    if (this.models.length === 1) {
      const m = this.models[0];
      for (let i = 0; i < n; i++) {
        const x = X.row(i);
        let s = m.b;
        for (let j = 0; j < m.alpha.length; j++) {
          if (m.alpha[j] !== 0) s += m.alpha[j] * m.t[j] * this._kernel(x, m.rows[j], m.gamma);
        }
        out[i] = s;
      }
      return out;
    }
    const votes = this._votes(X);
    for (let i = 0; i < n; i++) {
      const v = votes[i];
      let best = 0;
      let second = -1;
      for (let c = 1; c < v.length; c++) {
        if (v[c] > v[best]) { second = best; best = c; }
        else if (second < 0 || v[c] > v[second]) second = c;
      }
      out[i] = v[best] - (second >= 0 ? v[second] : 0);
    }
    return out;
  }

  predict(X) {
    if (this.models.length === 1) {
      const f = this.decisionFunction(X);
      return Array.from(f, (v) => (v >= 0 ? this.classes[1] : this.classes[0]));
    }
    return this._votes(X).map((v) => this.classes[argmax(v)]);
  }

  score(X, y) {
    return metrics.accuracy(y, this.predict(X));
  }
}

// ==================== 不变量自检 ====================

/**
 * classify 模块不变量自检（verify.suite 报告数组）
 * 每条检查对应 docs/algorithms/classify.md 中"可验证不变量"一节。
 */
export function selfCheck() {
  const s = suite('classify');

  // 1. Ridge(λ=0)：最小二乘一阶条件 —— 残差与设计矩阵各列正交
  s.check('Ridge(λ=0)：残差正交性 ‖X̃ᵀ(y−X̃ŵ)‖∞ ≈ 0（正规方程一阶条件）', () => {
    const { X, y } = makeRegression({ n: 100, rng: createRng(1) });
    const m = new Ridge({ lambda: 0 }).fit(X, y);
    const orth = residualOrthogonality(addBias(X), y, m.w);
    if (!(orth < 1e-6)) throw new Error(`正交性缺陷 ${orth.toExponential(2)}`);
    return `‖X̃ᵀr‖∞ = ${orth.toExponential(2)}`;
  });

  // 2. Lasso：坐标下降每步沿坐标精确最小化 → 目标值单调不增
  s.check('Lasso 坐标下降：目标函数值随迭代单调不增', () => {
    const { X, y } = makeRegression({ n: 80, rng: createRng(2) });
    const m = new Lasso({ lambda: 0.1, maxIter: 80 }).fit(X, y);
    const curve = m.lossCurve;
    const slack = 1e-9 * (Math.abs(curve[0]) + 1);
    for (let i = 1; i < curve.length; i++) {
      if (curve[i] > curve[i - 1] + slack) {
        throw new Error(`第 ${i} 轮目标值上升: ${curve[i - 1]} → ${curve[i]}`);
      }
    }
    return `${curve.length} 轮扫描: ${curve[0].toFixed(5)} → ${curve[curve.length - 1].toFixed(5)}`;
  });

  // 3. 朴素贝叶斯：predictLogProba 每行 exp 后和 = 1（后验归一）
  s.check('GaussianNB：predictLogProba 每行 exp 后和 = 1', () => {
    const { X, y } = loadIris();
    const nb = new GaussianNB().fit(X, y);
    const lp = nb.predictLogProba(X);
    for (let i = 0; i < X.rows; i++) {
      const row = lp.row(i);
      let sum = 0;
      for (const v of row) sum += Math.exp(v);
      expectClose(sum, 1, 1e-10, `第 ${i} 行`);
    }
    return `${X.rows} 行后验全部归一`;
  });

  s.check('MultinomialNB：predictLogProba 每行 exp 后和 = 1', () => {
    const { X, y } = loadIris(); // Iris 特征非负，可视为计数型数据
    const nb = new MultinomialNB({ alpha: 1 }).fit(X, y);
    const lp = nb.predictLogProba(X);
    for (let i = 0; i < X.rows; i++) {
      const row = lp.row(i);
      let sum = 0;
      for (const v of row) sum += Math.exp(v);
      expectClose(sum, 1, 1e-10, `第 ${i} 行`);
    }
    return `${X.rows} 行后验全部归一`;
  });

  s.check('BernoulliNB：predictLogProba 每行 exp 后和 = 1', () => {
    const { X, y } = loadIris();
    const nb = new BernoulliNB({ alpha: 1 }).fit(X, y);
    const lp = nb.predictLogProba(X);
    for (let i = 0; i < X.rows; i++) {
      const row = lp.row(i);
      let sum = 0;
      for (const v of row) sum += Math.exp(v);
      expectClose(sum, 1, 1e-10, `第 ${i} 行`);
    }
    return `${X.rows} 行后验全部归一`;
  });

  // 4. KNN k=1：训练集最近邻即自身 → 准确率恒 1
  s.check('KNN(k=1)：训练集准确率 = 1.0（最近邻即自身）', () => {
    const { X, y } = makeBlobs({ n: 60, k: 3, rng: createRng(5) });
    const knn = new KNN({ k: 1 }).fit(X, y);
    const acc = knn.score(X, y);
    expectClose(acc, 1, 1e-12);
    return 'acc = 1.0';
  });

  // 5. 决策树结构不变量
  s.check('决策树：maxDepth=1 时恰有两个叶节点', () => {
    const { X, y } = makeMoons({ n: 100, rng: createRng(6) });
    const tree = new DecisionTreeClassifier({ maxDepth: 1 }).fit(X, y);
    expectClose(tree.countLeaves(), 2, 0);
    return '叶节点数 = 2';
  });

  s.check('决策树：深树训练集 Gini 不纯度 < 0.1（过拟合能力）', () => {
    const { X, y } = makeMoons({ n: 120, noise: 0.2, rng: createRng(6) });
    const tree = new DecisionTreeClassifier({ maxDepth: 10 }).fit(X, y);
    const imp = tree.giniImpurity(X, y);
    if (!(imp < 0.1)) throw new Error(`训练 Gini = ${imp.toFixed(4)}`);
    return `训练 Gini = ${imp.toFixed(4)}`;
  });

  // 6. AdaBoost：moons 上训练准确率 > 0.9
  s.check('AdaBoost：moons 训练准确率 > 0.9', () => {
    const { X, y } = makeMoons({ n: 200, noise: 0.2, rng: createRng(42) });
    const ada = new AdaBoostClassifier({ nEstimators: 50 }).fit(X, y);
    const acc = ada.score(X, y);
    if (!(acc > 0.9)) throw new Error(`acc = ${acc.toFixed(4)}`);
    return `acc = ${acc.toFixed(3)}（${ada.estimators.length} 个桩）`;
  });

  // 7. SVM：KKT 条件 —— 支持向量 y·f(x)≈1、互补松弛、α∈[0,C]
  s.check('SVM：线性可分集上支持向量 y·f(x)≈1，KKT 互补松弛，α∈[0,C]', () => {
    const { X, y } = makeBlobs({ n: 40, k: 2, std: 0.5, spread: 8, rng: createRng(7) });
    const svm = new SVMClassifier({ C: 1.0, kernel: 'linear', maxPasses: 20 }).fit(X, y);
    const m = svm.models[0];
    const f = svm.decisionFunction(X);
    let maxMarginErr = 0;
    let maxSlack = 0;
    let nSV = 0;
    for (let i = 0; i < X.rows; i++) {
      const a = m.alpha[i];
      if (a < -1e-12 || a > svm.C + 1e-12) throw new Error(`α_${i} = ${a} 越界 [0, C]`);
      const yf = m.t[i] * f[i];
      maxSlack = Math.max(maxSlack, Math.abs(a * (yf - 1)));
      if (a > 1e-8) {
        nSV++;
        maxMarginErr = Math.max(maxMarginErr, Math.abs(yf - 1));
      }
    }
    if (nSV === 0) throw new Error('未找到支持向量');
    if (!(maxMarginErr < 0.05)) throw new Error(`支持向量 |y·f−1| 最大 ${maxMarginErr.toExponential(2)}`);
    if (!(maxSlack < 0.01)) throw new Error(`互补松弛 |α(y·f−1)| 最大 ${maxSlack.toExponential(2)}`);
    return `${nSV} 个支持向量，|y·f−1| ≤ ${maxMarginErr.toExponential(1)}，|α(y·f−1)| ≤ ${maxSlack.toExponential(1)}`;
  });

  // 8. SoftmaxRegression：predictProba 行和 = 1
  s.check('SoftmaxRegression：predictProba 行和 = 1', () => {
    const { X, y } = loadIris();
    const m = new SoftmaxRegression({ epochs: 100 }).fit(X, y);
    const P = m.predictProba(X);
    for (let i = 0; i < X.rows; i++) {
      const row = P.row(i);
      let sum = 0;
      for (const v of row) sum += v;
      expectClose(sum, 1, 1e-9, `第 ${i} 行`);
    }
    return `${X.rows} 行概率和全部为 1`;
  });

  // 9. 附加：逻辑回归在可分离数据上收敛到 0 训练误差，概率落在 (0,1)
  s.check('LogisticRegression：线性可分玩具集训练准确率 = 1.0，概率 ∈ (0,1)', () => {
    const { X, y } = makeBlobs({ n: 60, k: 2, std: 0.5, spread: 8, rng: createRng(8) });
    const log = new LogisticRegression({ epochs: 300 }).fit(X, y);
    expectClose(log.score(X, y), 1, 1e-12);
    const proba = log.predictProba(X);
    for (const p of proba) {
      if (!(p > 0 && p < 1)) throw new Error(`概率越界: ${p}`);
    }
    return 'acc = 1.0，概率全部 ∈ (0,1)';
  });

  // 10. 附加：三种朴素贝叶斯的 predict 与 predictLogProba argmax 一致
  s.check('朴素贝叶斯：predict = argmax(predictLogProba)（三种模型）', () => {
    const { X, y } = loadIris();
    for (const [name, nb] of [
      ['GaussianNB', new GaussianNB().fit(X, y)],
      ['MultinomialNB', new MultinomialNB().fit(X, y)],
      ['BernoulliNB', new BernoulliNB().fit(X, y)],
    ]) {
      const lp = nb.predictLogProba(X);
      const pred = nb.predict(X);
      for (let i = 0; i < X.rows; i++) {
        if (pred[i] !== nb.classes[argmax(lp.row(i))]) throw new Error(`${name} 第 ${i} 行不一致`);
      }
    }
    return 'Gaussian/Multinomial/Bernoulli 全部一致';
  });

  // 11. 附加：同种子完全可复现
  s.check('可复现性：同种子训练两次，权重逐位一致', () => {
    const { X, y } = makeMoons({ n: 80, rng: createRng(9) });
    const run = () => {
      const m = new SoftmaxRegression({ epochs: 30 }).fit(X, y);
      const l = new LogisticRegression({ epochs: 30 }).fit(X, y);
      return [m.W, l.w];
    };
    const [w1, w2] = run();
    const [w1b, w2b] = run();
    for (let i = 0; i < w1.length; i++) expectClose(w1[i], w1b[i], 0);
    for (let i = 0; i < w2.length; i++) expectClose(w2[i], w2b[i], 0);
    return '两次训练逐位一致';
  });

  return [s.report()];
}

// ==================== 基准 ====================

/**
 * classify 模块基准：Iris / Wine / XOR / 回归 / moons 五组标准场景。
 * @returns {{module: string, results: Array<{name, value, unit, target, ok, detail}>}}
 */
export function bench() {
  const results = [];
  const rec = (name, value, unit, target, ok, detail = '') => {
    results.push({ name, value: Number(value), unit, target, ok: !!ok, detail });
  };

  // ---- 场景一：Iris（zscore 标准化 + 固定划分） ----
  {
    const { X, y } = loadIris();
    const { train, test } = trainTestSplit(X, y, { testRatio: 0.3, seed: 42 });
    const fit = zscoreFit(train.X);
    const Xtr = zscoreApply(train.X, fit);
    const Xte = zscoreApply(test.X, fit);
    const K = 3;
    const yBinTrain = train.y.map((v) => (v === 1 ? 1 : 0));
    const yBinTest = test.y.map((v) => (v === 1 ? 1 : 0));
    const models = [
      ['SoftmaxRegression', () => new SoftmaxRegression({ epochs: 400 }).fit(Xtr, train.y), test.y, K],
      ['LogisticRegression(versicolor-vs-rest)', () => new LogisticRegression({ epochs: 400 }).fit(Xtr, yBinTrain), yBinTest, 2],
      ['DecisionTree', () => new DecisionTreeClassifier({ maxDepth: 5 }).fit(Xtr, train.y), test.y, K],
      ['KNN', () => new KNN({ k: 3 }).fit(Xtr, train.y), test.y, K],
      ['GaussianNB', () => new GaussianNB().fit(Xtr, train.y), test.y, K],
      ['SVM(rbf)', () => new SVMClassifier({ kernel: 'rbf' }).fit(Xtr, train.y), test.y, K],
    ];
    for (const [name, mk, yTrue, k] of models) {
      const m = mk();
      const pred = m.predict(Xte);
      const acc = metrics.accuracy(yTrue, pred);
      const f1 = metrics.macroF1(yTrue, pred, k);
      rec(`iris/${name}/accuracy`, acc, 'acc', '>=0.95', acc >= 0.95, `train=${Xtr.rows}, test=${Xte.rows}`);
      rec(`iris/${name}/macroF1`, f1, 'f1', '>=0.90', f1 >= 0.90);
    }
  }

  // ---- 场景二：Wine（同上） ----
  {
    const { X, y } = loadWine();
    const { train, test } = trainTestSplit(X, y, { testRatio: 0.3, seed: 42 });
    const fit = zscoreFit(train.X);
    const Xtr = zscoreApply(train.X, fit);
    const Xte = zscoreApply(test.X, fit);
    const K = 3;
    const yBinTrain = train.y.map((v) => (v === 1 ? 1 : 0));
    const yBinTest = test.y.map((v) => (v === 1 ? 1 : 0));
    const models = [
      ['SoftmaxRegression', () => new SoftmaxRegression({ epochs: 400 }).fit(Xtr, train.y), test.y, K],
      ['LogisticRegression(class_2-vs-rest)', () => new LogisticRegression({ epochs: 400 }).fit(Xtr, yBinTrain), yBinTest, 2],
      ['DecisionTree', () => new DecisionTreeClassifier({ maxDepth: 5 }).fit(Xtr, train.y), test.y, K],
      ['KNN', () => new KNN({ k: 3 }).fit(Xtr, train.y), test.y, K],
      ['GaussianNB', () => new GaussianNB().fit(Xtr, train.y), test.y, K],
      ['SVM(rbf)', () => new SVMClassifier({ kernel: 'rbf' }).fit(Xtr, train.y), test.y, K],
    ];
    for (const [name, mk, yTrue, k] of models) {
      const m = mk();
      const pred = m.predict(Xte);
      const acc = metrics.accuracy(yTrue, pred);
      const f1 = metrics.macroF1(yTrue, pred, k);
      rec(`wine/${name}/accuracy`, acc, 'acc', '>=0.90', acc >= 0.90, `train=${Xtr.rows}, test=${Xte.rows}`);
      rec(`wine/${name}/macroF1`, f1, 'f1', '>=0.85', f1 >= 0.85);
    }
  }

  // ---- 场景三：XOR（线性不可分基准） ----
  {
    const { X, y } = makeXor({ n: 200, rng: createRng(42) });
    const accLog = new LogisticRegression().fit(X, y).score(X, y);
    rec('xor/LogisticRegression(linear)/accuracy', accLog, 'acc', '≈0.5（线性不可分）',
      Math.abs(accLog - 0.5) <= 0.1, `实际值记录：线性模型无法分离异或`);
    const tree = new DecisionTreeClassifier({ maxDepth: 5 }).fit(X, y);
    const accTree = tree.score(X, y);
    rec('xor/DecisionTree/accuracy', accTree, 'acc', '>=0.95', accTree >= 0.95);
  }

  // ---- 场景四：makeRegression 上的线性回归 R² ----
  {
    const { X, y } = makeRegression({ n: 120, rng: createRng(42) });
    const ridge = new Ridge({ lambda: 1.0 }).fit(X, y);
    const r2Ridge = metrics.r2(y, Array.from(ridge.predict(X)));
    rec('regression/Ridge/R2', r2Ridge, 'R2', '>=0.8', r2Ridge >= 0.8, 'λ=1.0，in-sample R²');
    const lasso = new Lasso({ lambda: 0.01, maxIter: 500 }).fit(X, y);
    const r2Lasso = metrics.r2(y, Array.from(lasso.predict(X)));
    rec('regression/Lasso/R2', r2Lasso, 'R2', '>=0.8', r2Lasso >= 0.8, 'λ=0.01（轻正则），in-sample R²');
  }

  // ---- 场景五：SVM RBF 在 moons 上 ----
  {
    const { X, y } = makeMoons({ n: 200, noise: 0.2, rng: createRng(42) });
    const { train, test } = trainTestSplit(X, y, { testRatio: 0.3, seed: 42 });
    const svm = new SVMClassifier({ kernel: 'rbf' }).fit(train.X, train.y);
    const acc = svm.score(test.X, test.y);
    rec('moons/SVM(rbf)/accuracy', acc, 'acc', '>=0.9', acc >= 0.9, `train=${train.X.rows}, test=${test.X.rows}`);
  }

  return { module: 'classify', results };
}
