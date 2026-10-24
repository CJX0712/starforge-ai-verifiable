/**
 * @file cluster.mjs — StarForge 聚类与降维：KMeans / GMM / PCA / 凝聚层次聚类
 * 作者：晨星
 *
 * 算法谱系（世界经典成果的直接工程化）：
 * - KMeans：Lloyd 1982 的迭代重分配；播种用 Arthur & Vassilvitskii 2007 的 kmeans++
 * - GMM：Dempster, Laird & Rubin 1977 的 EM 算法（观测对数似然单调不减的硬保证）
 * - PCA：Pearson 1901 / Hotelling 1933 的主成分分析，经 Jacobi 对称特征分解求解
 * - Agglomerative：Sokal & Michener 1958 的平均连接（Lance-Williams 距离更新）
 *
 * 可验证不变量（可被独立实现交叉验证的金标准）：
 * - KMeans：每轮 SSE 单调不增；收敛时用最终质心重算分配不变
 * - GMM：logLikelihood_ 序列单调不减；责任度行和 = 1；协方差对称半正定（Cholesky 成功）
 * - PCA：components_ 正交（VᵀV≈I）；解释方差比降序且累计 ≤ 1；重构误差随分量数增加而下降
 */
import {
  Mat, matMul, transpose, solveLinear, cholesky, isPositiveDefinite, isSymmetric,
  jacobiEigen, logsumexp, createRng, argmax,
} from './linalg.mjs';
import { suite } from './verify.mjs';
import { makeBlobs, loadIris, loadWine, metrics } from './datasets.mjs';

const LOG_2PI = Math.log(2 * Math.PI);

/** 两个等长向量（Float64Array）的平方欧氏距离 */
function sqDistVec(a, b) {
  let s = 0;
  for (let j = 0; j < a.length; j++) { const t = a[j] - b[j]; s += t * t; }
  return s;
}

/** 矩阵行两点间的平方距离 */
function sqDistRows(X, i, Y, c) {
  return sqDistVec(X.row(i), Y.row(c));
}

// ==================== KMeans ====================

/**
 * KMeans 聚类（Lloyd 算法 + kmeans++ 播种）。
 *
 * 目标函数：min_{C,μ} Σ_i ‖x_i − μ_{c_i}‖²（SSE / inertia）。
 * kmeans++（Arthur & Vassilvitskii, SODA 2007）以 D² 概率播种，
 * 期望给出 O(log k) 竞争比的初始化，避免随机播种的退化解。
 */
export class KMeans {
  /**
   * @param {object} opts
   * @param {number} opts.k 簇数
   * @param {number} [opts.seed=42] 随机种子（决定 kmeans++ 播种，保证可复现）
   * @param {'kmeans++'|'random'} [opts.init='kmeans++'] 初始化策略
   * @param {number} [opts.maxIter=300] 最大迭代轮数
   */
  constructor({ k, seed = 42, init = 'kmeans++', maxIter = 300 } = {}) {
    if (!Number.isInteger(k) || k < 1) throw new Error('KMeans: k 必须是正整数');
    if (init !== 'kmeans++' && init !== 'random') throw new Error(`KMeans: 未知初始化策略 ${init}`);
    if (!Number.isInteger(maxIter) || maxIter < 1) throw new Error('KMeans: maxIter 必须是正整数');
    this.k = k;
    this.seed = seed;
    this.init = init;
    this.maxIter = maxIter;
    this.labels_ = null;
    this.centroids_ = null;
    this.inertia_ = null;
    this.nIter_ = 0;
    /** 每轮分配后的 SSE 历史（单调不增，可验证） */
    this.inertiaHistory_ = [];
  }

  /** kmeans++ 播种：按 D² 概率依次抽取 k 个质心 */
  _initKmeansPlusPlus(X, rng) {
    const n = X.rows;
    const centroids = [X.row(rng.randint(n))];
    const d2 = new Float64Array(n).fill(Infinity);
    for (let c = 1; c < this.k; c++) {
      const last = centroids[c - 1];
      let total = 0;
      for (let i = 0; i < n; i++) {
        const dd = sqDistVec(X.row(i), last);
        if (dd < d2[i]) d2[i] = dd;
        total += d2[i];
      }
      if (!(total > 0)) {
        // 所有点均与已有质心重合：退化为随机取一点
        centroids.push(X.row(rng.randint(n)));
        continue;
      }
      let r = rng.next() * total;
      let idx = n - 1;
      for (let i = 0; i < n; i++) { r -= d2[i]; if (r <= 0) { idx = i; break; } }
      centroids.push(X.row(idx));
    }
    return centroids;
  }

  /** 随机播种：无放回抽取 k 个样本点 */
  _initRandom(X, rng) {
    const idx = rng.shuffle(Array.from({ length: X.rows }, (_, i) => i));
    return idx.slice(0, this.k).map((i) => X.row(i));
  }

  /** 分配步：每个点归入最近质心；返回 {labels, inertia} */
  _assign(X, centroids) {
    const labels = new Array(X.rows);
    let inertia = 0;
    for (let i = 0; i < X.rows; i++) {
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < this.k; c++) {
        const d = sqDistRows(X, i, centroids, c);
        if (d < bestD) { bestD = d; best = c; }
      }
      labels[i] = best;
      inertia += bestD;
    }
    return { labels, inertia };
  }

  /** 更新步：质心 = 簇内均值；空簇用"离其质心最远的点"填充（Lloyd 的标准修复） */
  _update(X, labels, centroids) {
    const d = X.cols;
    const sums = Array.from({ length: this.k }, () => new Float64Array(d));
    const counts = new Array(this.k).fill(0);
    for (let i = 0; i < X.rows; i++) {
      const c = labels[i];
      counts[c]++;
      const row = X.row(i);
      for (let j = 0; j < d; j++) sums[c][j] += row[j];
    }
    const out = new Mat(this.k, d);
    for (let c = 0; c < this.k; c++) {
      if (counts[c] > 0) {
        for (let j = 0; j < d; j++) out.set(c, j, sums[c][j] / counts[c]);
      } else {
        out.set(c, 0, centroids.get(c, 0));
        for (let j = 1; j < d; j++) out.set(c, j, centroids.get(c, j));
      }
    }
    const empties = [];
    for (let c = 0; c < this.k; c++) if (counts[c] === 0) empties.push(c);
    if (empties.length > 0) {
      // 按到自身质心的距离降序取最远的若干点，填充空簇
      const order = Array.from({ length: X.rows }, (_, i) => i)
        .sort((a, b) => sqDistRows(X, b, centroids, labels[b]) - sqDistRows(X, a, centroids, labels[a]));
      for (let e = 0; e < empties.length && e < order.length; e++) {
        const row = X.row(order[e]);
        for (let j = 0; j < d; j++) out.set(empties[e], j, row[j]);
      }
    }
    return out;
  }

  /**
   * 拟合 KMeans。
   * @param {Mat} X n×d 数据矩阵
   * @returns {{labels_: number[], centroids_: Mat, inertia_: number}}
   */
  fit(X) {
    if (!(X instanceof Mat)) throw new Error('KMeans.fit: X 必须是 Mat');
    if (this.k > X.rows) throw new Error(`KMeans: k=${this.k} 超过样本数 ${X.rows}`);
    const rng = createRng(this.seed);
    let centroids = this.init === 'kmeans++'
      ? Mat.from2D(this._initKmeansPlusPlus(X, rng).map((r) => Array.from(r)))
      : Mat.from2D(this._initRandom(X, rng).map((r) => Array.from(r)));
    let prevLabels = null;
    let labels = null;
    this.inertiaHistory_ = [];
    for (let iter = 0; iter < this.maxIter; iter++) {
      const assigned = this._assign(X, centroids);
      this.inertiaHistory_.push(assigned.inertia);
      labels = assigned.labels;
      centroids = this._update(X, labels, centroids);
      this.nIter_ = iter + 1;
      if (prevLabels && prevLabels.length === labels.length && prevLabels.every((v, i) => v === labels[i])) break;
      prevLabels = labels;
    }
    this.labels_ = labels;
    this.centroids_ = centroids;
    this.inertia_ = this.inertiaHistory_[this.inertiaHistory_.length - 1];
    return { labels_: this.labels_, centroids_: this.centroids_, inertia_: this.inertia_ };
  }

  /**
   * 预测新样本的簇标签。
   * @param {Mat} X n×d
   * @returns {number[]}
   */
  predict(X) {
    if (!this.centroids_) throw new Error('KMeans.predict: 尚未拟合');
    if (X.cols !== this.centroids_.cols) throw new Error('KMeans.predict: 特征维度不匹配');
    return this._assign(X, this.centroids_).labels;
  }

  /** 用当前参数重算 SSE（与 inertia_ 独立复核） */
  score(X) {
    if (!this.centroids_) throw new Error('KMeans.score: 尚未拟合');
    return this._assign(X, this.centroids_).inertia;
  }
}

// ==================== GMM ====================

/**
 * 高斯混合模型（EM 算法，Dempster, Laird & Rubin 1977）。
 *
 * 模型：p(x) = Σ_c w_c · N(x | μ_c, Σ_c)。
 * E 步用 logsumexp 稳定计算责任度 γ_ic ∝ w_c·N(x_i|μ_c,Σ_c)；
 * M 步闭式更新 w_c、μ_c、Σ_c。观测对数似然每轮单调不减。
 */
export class GMM {
  /**
   * @param {object} opts
   * @param {number} opts.k 混合分量数
   * @param {number} [opts.seed=42] 种子（驱动 KMeans 初始化）
   * @param {'full'|'diag'} [opts.covariance='full'] full：每分量 d×d 协方差（Mat[]）；diag：每分量对角方差（Mat k×d）
   * @param {number} [opts.maxIter=200] 最大 EM 轮数
   * @param {number} [opts.tol=1e-6] 对数似然绝对变化小于 tol 即收敛
   * @param {number} [opts.regCovar=1e-6] 加到协方差对角上的正则项，保证正定
   */
  constructor({ k, seed = 42, covariance = 'full', maxIter = 200, tol = 1e-6, regCovar = 1e-6 } = {}) {
    if (!Number.isInteger(k) || k < 1) throw new Error('GMM: k 必须是正整数');
    if (covariance !== 'full' && covariance !== 'diag') throw new Error(`GMM: 未知协方差类型 ${covariance}`);
    if (!Number.isInteger(maxIter) || maxIter < 1) throw new Error('GMM: maxIter 必须是正整数');
    this.k = k;
    this.seed = seed;
    this.covariance = covariance;
    this.maxIter = maxIter;
    this.tol = tol;
    this.regCovar = regCovar;
    this.labels_ = null;
    this.means_ = null;
    this.weights_ = null;
    this.covariances_ = null;
    this.logLikelihood_ = null;
    this.nIter_ = 0;
    /** 每轮 E 步的观测对数似然历史（单调不减，可验证） */
    this.llHistory_ = [];
  }

  /** 单个分量对全部样本的对数高斯密度（logsumexp 稳定的基础） */
  _logGaussian(X, c) {
    const n = X.rows, d = X.cols;
    const mean = this.means_.row(c);
    const out = new Float64Array(n);
    if (this.covariance === 'diag') {
      const varr = this.covariances_.row(c);
      for (let i = 0; i < n; i++) {
        const row = X.row(i);
        let s = 0;
        for (let j = 0; j < d; j++) {
          const t = row[j] - mean[j];
          s += Math.log(varr[j]) + (t * t) / varr[j];
        }
        out[i] = -0.5 * (d * LOG_2PI + s);
      }
      return out;
    }
    const cov = this.covariances_[c];
    const L = cholesky(cov); // 协方差已加 regCovar，必正定
    let logDet = 0;
    for (let i = 0; i < d; i++) logDet += Math.log(L.get(i, i));
    logDet *= 2;
    for (let i = 0; i < n; i++) {
      const diff = X.row(i);
      const y = new Float64Array(d);
      for (let j = 0; j < d; j++) y[j] = diff[j] - mean[j];
      const z = solveLinear(L, y); // L·z = (x−μ)
      let quad = 0;
      for (let j = 0; j < d; j++) quad += z[j] * z[j];
      out[i] = -0.5 * (d * LOG_2PI + logDet + quad);
    }
    return out;
  }

  /**
   * E 步：计算对数联合概率矩阵与责任度。
   * @returns {{logProb: Mat(n×k), resp: Mat(n×k), ll: Float64Array}}
   */
  _eStep(X) {
    const n = X.rows, k = this.k;
    const logDens = [];
    for (let c = 0; c < k; c++) logDens.push(this._logGaussian(X, c));
    const logProb = new Mat(n, k);
    const resp = new Mat(n, k);
    const ll = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const row = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        row[c] = Math.log(this.weights_[c]) + logDens[c][i];
        logProb.set(i, c, row[c]);
      }
      const m = logsumexp(row);
      ll[i] = m;
      if (!Number.isFinite(m)) {
        // 观测在所有分量下概率为 0（理论上被 regCovar 排除）：退化为均匀责任度
        for (let c = 0; c < k; c++) resp.set(i, c, 1 / k);
        continue;
      }
      for (let c = 0; c < k; c++) resp.set(i, c, Math.exp(row[c] - m));
    }
    return { logProb, resp, ll };
  }

  /** M 步：闭式更新权重、均值、协方差 */
  _mStep(X, resp) {
    const n = X.rows, d = X.cols, k = this.k;
    const Nk = new Float64Array(k);
    for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) Nk[c] += resp.get(i, c);
    const weights = new Float64Array(k);
    for (let c = 0; c < k; c++) weights[c] = Nk[c] / n;
    const means = new Mat(k, d);
    for (let c = 0; c < k; c++) {
      if (Nk[c] < 1e-12) {
        // 死分量：保留旧均值
        for (let j = 0; j < d; j++) means.set(c, j, this.means_.get(c, j));
        continue;
      }
      for (let i = 0; i < n; i++) {
        const r = resp.get(i, c);
        if (r === 0) continue;
        const row = X.row(i);
        for (let j = 0; j < d; j++) means.set(c, j, means.get(c, j) + r * row[j]);
      }
      for (let j = 0; j < d; j++) means.set(c, j, means.get(c, j) / Nk[c]);
    }
    if (this.covariance === 'diag') {
      const covs = this.covariances_.clone();
      for (let c = 0; c < k; c++) {
        if (Nk[c] < 1e-12) continue;
        for (let j = 0; j < d; j++) {
          let s = 0;
          for (let i = 0; i < n; i++) {
            const t = X.get(i, j) - means.get(c, j);
            s += resp.get(i, c) * t * t;
          }
          covs.set(c, j, s / Nk[c] + this.regCovar);
        }
      }
      this.covariances_ = covs;
    } else {
      const covs = this.covariances_.map((M) => M.clone());
      for (let c = 0; c < k; c++) {
        if (Nk[c] < 1e-12) continue;
        const S = new Mat(d, d);
        for (let i = 0; i < n; i++) {
          const r = resp.get(i, c);
          if (r === 0) continue;
          const diff = new Float64Array(d);
          const row = X.row(i);
          for (let j = 0; j < d; j++) diff[j] = row[j] - means.get(c, j);
          for (let p = 0; p < d; p++) {
            if (diff[p] === 0) continue;
            const rp = r * diff[p];
            for (let q = p; q < d; q++) {
              S.set(p, q, S.get(p, q) + rp * diff[q]);
              if (q !== p) S.set(q, p, S.get(q, p) + rp * diff[q]);
            }
          }
        }
        for (let p = 0; p < d; p++) {
          S.set(p, p, S.get(p, p) / Nk[c] + this.regCovar);
          for (let q = p + 1; q < d; q++) {
            const v = S.get(p, q) / Nk[c];
            S.set(p, q, v);
            S.set(q, p, v);
          }
        }
        covs[c] = S;
      }
      this.covariances_ = covs;
    }
    this.weights_ = weights;
    this.means_ = means;
  }

  /** KMeans 驱动的 EM 初始化（确定性种子） */
  _initialize(X) {
    const n = X.rows, d = X.cols;
    const km = new KMeans({ k: this.k, seed: this.seed, init: 'kmeans++' });
    km.fit(X);
    const labels = km.labels_;
    this.means_ = km.centroids_.clone();
    this.weights_ = new Float64Array(this.k);
    for (let c = 0; c < this.k; c++) {
      let cnt = 0;
      for (const l of labels) if (l === c) cnt++;
      this.weights_[c] = cnt / n;
      if (this.weights_[c] === 0) this.weights_[c] = 1 / n; // 防御：空簇不给零权重
    }
    if (this.covariance === 'diag') {
      this.covariances_ = new Mat(this.k, d);
      for (let c = 0; c < this.k; c++) {
        let cnt = 0;
        for (const l of labels) if (l === c) cnt++;
        for (let j = 0; j < d; j++) {
          let s = 0;
          for (let i = 0; i < n; i++) if (labels[i] === c) {
            const t = X.get(i, j) - this.means_.get(c, j);
            s += t * t;
          }
          this.covariances_.set(c, j, (cnt > 0 ? s / cnt : 1) + this.regCovar);
        }
      }
    } else {
      this.covariances_ = [];
      for (let c = 0; c < this.k; c++) {
        let cnt = 0;
        for (const l of labels) if (l === c) cnt++;
        const S = new Mat(d, d);
        if (cnt > 1) {
          for (let i = 0; i < n; i++) {
            if (labels[i] !== c) continue;
            const diff = new Float64Array(d);
            for (let j = 0; j < d; j++) diff[j] = X.get(i, j) - this.means_.get(c, j);
            for (let p = 0; p < d; p++) for (let q = p; q < d; q++) {
              S.set(p, q, S.get(p, q) + diff[p] * diff[q]);
              if (q !== p) S.set(q, p, S.get(q, p) + diff[p] * diff[q]);
            }
          }
          for (let p = 0; p < d; p++) {
            for (let q = p; q < d; q++) {
              const v = S.get(p, q) / cnt;
              S.set(p, q, v);
              if (q !== p) S.set(q, p, v);
            }
          }
        } else {
          for (let j = 0; j < d; j++) S.set(j, j, 1);
        }
        for (let j = 0; j < d; j++) S.set(j, j, S.get(j, j) + this.regCovar);
        this.covariances_.push(S);
      }
    }
  }

  /**
   * 拟合 GMM。
   * @param {Mat} X n×d 数据矩阵
   * @returns {{logLikelihood_: number, nIter_: number}} 结构同实例字段
   */
  fit(X) {
    if (!(X instanceof Mat)) throw new Error('GMM.fit: X 必须是 Mat');
    if (this.k > X.rows) throw new Error(`GMM: k=${this.k} 超过样本数 ${X.rows}`);
    this._initialize(X);
    this.llHistory_ = [];
    let prevLL = -Infinity;
    let iter = 0;
    for (; iter < this.maxIter; iter++) {
      const { resp, ll } = this._eStep(X);
      let total = 0;
      for (const v of ll) total += v;
      this.llHistory_.push(total);
      if (iter > 0 && Math.abs(total - prevLL) < this.tol) break;
      this._mStep(X, resp);
      prevLL = total;
    }
    this.nIter_ = iter;
    // 用最终参数再跑一次 E 步，固定 labels_ / logLikelihood_
    const { resp, ll } = this._eStep(X);
    this.labels_ = Array.from({ length: X.rows }, (_, i) => argmax(resp.row(i)));
    let total = 0;
    for (const v of ll) total += v;
    this.logLikelihood_ = total;
    return { labels_: this.labels_, means_: this.means_, weights_: this.weights_, covariances_: this.covariances_, logLikelihood_: this.logLikelihood_, nIter_: this.nIter_ };
  }

  /**
   * 责任度矩阵（后验概率）。
   * @param {Mat} X n×d
   * @returns {Mat} n×k，每行和为 1
   */
  predictProba(X) {
    if (!this.weights_) throw new Error('GMM.predictProba: 尚未拟合');
    if (X.cols !== this.means_.cols) throw new Error('GMM.predictProba: 特征维度不匹配');
    return this._eStep(X).resp;
  }

  /** 硬分配：责任度最大的分量 */
  predict(X) {
    const resp = this.predictProba(X);
    return Array.from({ length: resp.rows }, (_, i) => argmax(resp.row(i)));
  }

  /** 平均对数似然（对数似然 / n） */
  score(X) {
    if (!this.weights_) throw new Error('GMM.score: 尚未拟合');
    const { ll } = this._eStep(X);
    let s = 0;
    for (const v of ll) s += v;
    return s / X.rows;
  }
}

// ==================== PCA ====================

/**
 * 主成分分析（Pearson 1901 / Hotelling 1933；Jolliffe 2002 的现代表述）。
 *
 * 对中心化数据的协方差矩阵做 Jacobi 对称特征分解，取前 k 个特征向量作为主成分。
 * 等价于最大化投影方差 min 重构误差（Eckart-Young 最优低秩近似）。
 */
export class PCA {
  /**
   * @param {object} opts
   * @param {number} opts.nComponents 保留的主成分数（1 ≤ nComponents ≤ min(n, d)）
   */
  constructor({ nComponents } = {}) {
    if (!Number.isInteger(nComponents) || nComponents < 1) throw new Error('PCA: nComponents 必须是正整数');
    this.nComponents = nComponents;
    this.components_ = null;
    this.explainedVariance_ = null;
    this.explainedVarianceRatio_ = null;
    this.mean_ = null;
    this.nFeatures_ = null;
    this.totalVariance_ = null;
  }

  /**
   * 拟合 PCA。
   * @param {Mat} X n×d
   * @returns 实例自身字段集合
   */
  fit(X) {
    if (!(X instanceof Mat)) throw new Error('PCA.fit: X 必须是 Mat');
    const n = X.rows, d = X.cols;
    if (this.nComponents > Math.min(n, d)) {
      throw new Error(`PCA: nComponents=${this.nComponents} 超过 min(n,d)=${Math.min(n, d)}`);
    }
    const mean = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += X.get(i, j);
      mean[j] = s / n;
    }
    // 协方差矩阵（除以 n−1，与 sklearn 的无偏估计一致）；直接累加保证严格对称
    const C = new Mat(d, d);
    for (let i = 0; i < n; i++) {
      const diff = new Float64Array(d);
      for (let j = 0; j < d; j++) diff[j] = X.get(i, j) - mean[j];
      for (let p = 0; p < d; p++) {
        if (diff[p] === 0) continue;
        const dp = diff[p];
        for (let q = p; q < d; q++) {
          C.set(p, q, C.get(p, q) + dp * diff[q]);
          if (q !== p) C.set(q, p, C.get(q, p) + dp * diff[q]);
        }
      }
    }
    const denom = n > 1 ? n - 1 : 1;
    for (let idx = 0; idx < d * d; idx++) C.data[idx] /= denom;

    const { values, vectors } = jacobiEigen(C);
    const evAll = Float64Array.from(values, (v) => Math.max(v, 0)); // 数值误差可能产生极小负值
    let total = 0;
    for (const v of evAll) total += v;

    const k = this.nComponents;
    this.components_ = new Mat(k, d);
    this.explainedVariance_ = new Float64Array(k);
    this.explainedVarianceRatio_ = new Float64Array(k);
    for (let c = 0; c < k; c++) {
      this.explainedVariance_[c] = evAll[c];
      this.explainedVarianceRatio_[c] = total > 0 ? evAll[c] / total : 0;
      for (let j = 0; j < d; j++) this.components_.set(c, j, vectors.get(j, c));
    }
    this.mean_ = mean;
    this.nFeatures_ = d;
    this.totalVariance_ = total;
    return {
      components_: this.components_,
      explainedVariance_: this.explainedVariance_,
      explainedVarianceRatio_: this.explainedVarianceRatio_,
      mean_: this.mean_,
    };
  }

  /**
   * 投影到主成分空间。
   * @param {Mat} X n×d
   * @returns {Mat} n×k
   */
  transform(X) {
    if (!this.components_) throw new Error('PCA.transform: 尚未拟合');
    if (X.cols !== this.nFeatures_) throw new Error('PCA.transform: 特征维度不匹配');
    const n = X.rows, k = this.nComponents, d = this.nFeatures_;
    const Z = new Mat(n, k);
    for (let i = 0; i < n; i++) {
      const row = X.row(i);
      for (let c = 0; c < k; c++) {
        let s = 0;
        for (let j = 0; j < d; j++) s += (row[j] - this.mean_[j]) * this.components_.get(c, j);
        Z.set(i, c, s);
      }
    }
    return Z;
  }

  /**
   * 从主成分坐标重构原始空间（均值加回）。
   * @param {Mat} Z n×k
   * @returns {Mat} n×d
   */
  inverseTransform(Z) {
    if (!this.components_) throw new Error('PCA.inverseTransform: 尚未拟合');
    if (Z.cols !== this.nComponents) throw new Error('PCA.inverseTransform: 主成分数不匹配');
    const n = Z.rows, k = this.nComponents, d = this.nFeatures_;
    const X = new Mat(n, d);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) {
        let s = this.mean_[j];
        for (let c = 0; c < k; c++) s += Z.get(i, c) * this.components_.get(c, j);
        X.set(i, j, s);
      }
    }
    return X;
  }

  /** fit + transform */
  fitTransform(X) {
    this.fit(X);
    return this.transform(X);
  }
}

// ==================== AgglomerativeClustering ====================

/**
 * 自底向上凝聚层次聚类（平均连接 / average linkage，UPGMA）。
 *
 * Lance-Williams 递推：合并 A、B 后，d(A∪B, C) = (|A|·d(A,C) + |B|·d(B,C)) / (|A|+|B|)。
 * 每轮合并全局距离最近的两簇，直至剩下 k 簇。
 */
export class AgglomerativeClustering {
  /**
   * @param {object} opts
   * @param {number} opts.k 目标簇数
   * @param {'average'} [opts.linkage='average'] 连接准则（当前实现平均连接）
   */
  constructor({ k, linkage = 'average' } = {}) {
    if (!Number.isInteger(k) || k < 1) throw new Error('AgglomerativeClustering: k 必须是正整数');
    if (linkage !== 'average') throw new Error(`AgglomerativeClustering: 仅实现 average 连接（收到 ${linkage}）`);
    this.k = k;
    this.linkage = linkage;
    this.labels_ = null;
  }

  /**
   * 拟合凝聚聚类。
   * @param {Mat} X n×d
   * @returns {{labels_: number[]}}
   */
  fit(X) {
    if (!(X instanceof Mat)) throw new Error('AgglomerativeClustering.fit: X 必须是 Mat');
    const n = X.rows;
    if (this.k > n) throw new Error(`AgglomerativeClustering: k=${this.k} 超过样本数 ${n}`);
    const D = new Mat(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = Math.sqrt(sqDistRows(X, i, X, j));
        D.set(i, j, d);
        D.set(j, i, d);
      }
    }
    const members = Array.from({ length: n }, (_, i) => [i]);
    const size = new Array(n).fill(1);
    const alive = new Array(n).fill(true);
    let nClusters = n;
    while (nClusters > this.k) {
      let bi = -1, bj = -1, best = Infinity;
      for (let i = 0; i < n; i++) {
        if (!alive[i]) continue;
        for (let j = i + 1; j < n; j++) {
          if (!alive[j]) continue;
          if (D.get(i, j) < best) { best = D.get(i, j); bi = i; bj = j; }
        }
      }
      if (bi < 0) break; // 理论不可达
      const oldSizeBi = size[bi];
      for (const m of members[bj]) { members[bi].push(m); }
      size[bi] += size[bj];
      alive[bj] = false;
      nClusters--;
      for (let l = 0; l < n; l++) {
        if (!alive[l] || l === bi) continue;
        const d = (oldSizeBi * D.get(bi, l) + size[bj] * D.get(bj, l)) / size[bi];
        D.set(bi, l, d);
        D.set(l, bi, d);
      }
    }
    // 压缩标签到 0..k-1（按代表元升序，保证确定性）
    const reps = [];
    for (let i = 0; i < n; i++) if (alive[i]) reps.push(i);
    const remap = new Map(reps.map((r, idx) => [r, idx]));
    const labels = new Array(n);
    for (const r of reps) for (const m of members[r]) labels[m] = remap.get(r);
    this.labels_ = labels;
    return { labels_: this.labels_ };
  }

  /** 凝聚聚类无 predict（层次结构不可外推），仅返回拟合标签 */
  predict() {
    throw new Error('AgglomerativeClustering 不支持 predict（非参数化模型）');
  }
}

// ==================== 合成数据工具（selfCheck / bench 共用） ====================

/**
 * 生成合成 GMM 数据：两个分离的二维高斯团。
 * @returns {{X: Mat, y: number[], means: number[][], std: number}}
 */
function makeTwoBlobsGauss({ nPer = 200, std = 0.5, seed = 7 } = {}) {
  const rng = createRng(seed);
  const means = [[0, 0], [5, 5]];
  const X = new Mat(2 * nPer, 2);
  const y = new Array(2 * nPer);
  for (let i = 0; i < 2 * nPer; i++) {
    const c = i < nPer ? 0 : 1;
    X.set(i, 0, means[c][0] + rng.normal() * std);
    X.set(i, 1, means[c][1] + rng.normal() * std);
    y[i] = c;
  }
  return { X, y, means, std };
}

/** 在分量排列未知的前提下，估计均值与真实均值的最大误差（取最优排列） */
function meanRecoveryError(estMeans, trueMeans) {
  const k = estMeans.rows;
  const perms = permutations(k);
  let best = Infinity;
  for (const p of perms) {
    let e = 0;
    for (let c = 0; c < k; c++) {
      const row = estMeans.row(c);
      for (let j = 0; j < trueMeans[c].length; j++) e = Math.max(e, Math.abs(row[j] - trueMeans[p[c]][j]));
    }
    best = Math.min(best, e);
  }
  return best;
}

/** 0..k-1 的全部排列 */
function permutations(k) {
  const out = [];
  const cur = [];
  const used = new Array(k).fill(false);
  const rec = () => {
    if (cur.length === k) { out.push(cur.slice()); return; }
    for (let i = 0; i < k; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

// ==================== selfCheck ====================

/** 聚类与降维模块的不变量自检（suite 报告数组） */
export function selfCheck() {
  const reports = [];

  // ---------- KMeans ----------
  {
    const s = suite('kmeans');
    const { X, y } = makeTwoBlobsGauss({ nPer: 100, std: 0.6, seed: 11 });
    const km = new KMeans({ k: 2, seed: 42 });
    km.fit(X);

    s.check('KMeans：每轮 SSE 单调不增', () => {
      const h = km.inertiaHistory_;
      for (let t = 1; t < h.length; t++) {
        if (h[t] > h[t - 1] + 1e-9) throw new Error(`第 ${t} 轮 SSE 增大: ${h[t - 1]} → ${h[t]}`);
      }
      return `${h.length} 轮，SSE ${h[0].toFixed(2)} → ${h[h.length - 1].toFixed(2)}`;
    });

    s.check('KMeans：收敛时标签与质心一致（重算分配不变）', () => {
      const relabels = km.predict(X);
      for (let i = 0; i < relabels.length; i++) {
        if (relabels[i] !== km.labels_[i]) throw new Error(`点 ${i} 重算分配发生变化: ${km.labels_[i]} → ${relabels[i]}`);
      }
      return `${X.rows} 点重算分配全部一致`;
    });

    s.check('KMeans：blobs(300,k=3) purity ≥ 0.95', () => {
      const { X: B, y: by } = datasetsMakeBlobs();
      const km3 = new KMeans({ k: 3, seed: 42 });
      km3.fit(B);
      const purity = clusterPurity(by, km3.labels_, 3);
      if (purity < 0.95) throw new Error(`purity = ${purity.toFixed(4)} < 0.95`);
      return `purity = ${purity.toFixed(4)}`;
    });

    s.check('KMeans：kmeans++ 与 random 初始化均可收敛且结果可复现', () => {
      const a = new KMeans({ k: 3, seed: 42, init: 'kmeans++' });
      const b = new KMeans({ k: 3, seed: 42, init: 'kmeans++' });
      a.fit(X);
      b.fit(X);
      for (let i = 0; i < a.labels_.length; i++) {
        if (a.labels_[i] !== b.labels_[i]) throw new Error('同种子标签不一致');
      }
      const r = new KMeans({ k: 3, seed: 42, init: 'random' });
      r.fit(X);
      if (r.inertia_ > a.inertia_ * 3 + 1) throw new Error('random 初始化未收敛');
      return `kmeans++ SSE=${a.inertia_.toFixed(3)}, random SSE=${r.inertia_.toFixed(3)}`;
    });

    reports.push(s.report());
  }

  // ---------- GMM ----------
  {
    const s = suite('gmm');
    const { X, means: trueMeans } = makeTwoBlobsGauss({ nPer: 200, std: 0.5, seed: 7 });
    const gmm = new GMM({ k: 2, seed: 42, covariance: 'full', maxIter: 200, tol: 1e-8 });
    gmm.fit(X);

    s.check('GMM：对数似然序列单调不减', () => {
      const h = gmm.llHistory_;
      for (let t = 1; t < h.length; t++) {
        if (h[t] < h[t - 1] - 1e-6 * Math.max(1, Math.abs(h[t - 1]))) {
          throw new Error(`第 ${t} 轮 LL 下降: ${h[t - 1].toFixed(6)} → ${h[t].toFixed(6)}`);
        }
      }
      return `${h.length} 轮，LL ${h[0].toFixed(2)} → ${h[h.length - 1].toFixed(2)}`;
    });

    s.check('GMM：责任度行和 = 1', () => {
      const resp = gmm.predictProba(X);
      for (let i = 0; i < resp.rows; i++) {
        let sum = 0;
        for (let c = 0; c < resp.cols; c++) sum += resp.get(i, c);
        if (Math.abs(sum - 1) > 1e-9) throw new Error(`第 ${i} 行责任度和 = ${sum}`);
      }
      return '全部行和 = 1（误差 < 1e-9）';
    });

    s.check('GMM：协方差矩阵对称半正定（Cholesky 成功）', () => {
      for (let c = 0; c < gmm.k; c++) {
        const S = gmm.covariances_[c];
        if (!isSymmetric(S, 1e-10)) throw new Error(`分量 ${c} 协方差不对称`);
        if (!isPositiveDefinite(S)) throw new Error(`分量 ${c} 协方差非正定`);
      }
      return `${gmm.k} 个协方差矩阵全部对称正定`;
    });

    s.check('GMM：合成已知参数数据上均值估计误差 < 0.3', () => {
      const err = meanRecoveryError(gmm.means_, trueMeans);
      if (err >= 0.3) throw new Error(`均值最大误差 ${err.toFixed(4)} ≥ 0.3`);
      return `均值最大误差 = ${err.toFixed(4)}`;
    });

    s.check('GMM：diag 协方差为正且 LL 同样单调', () => {
      const g = new GMM({ k: 2, seed: 3, covariance: 'diag', maxIter: 100, tol: 1e-8 });
      g.fit(X);
      for (let c = 0; c < g.k; c++) for (let j = 0; j < g.covariances_.cols; j++) {
        if (g.covariances_.get(c, j) <= 0) throw new Error(`分量 ${c} 维度 ${j} 方差非正`);
      }
      const h = g.llHistory_;
      for (let t = 1; t < h.length; t++) {
        if (h[t] < h[t - 1] - 1e-6 * Math.max(1, Math.abs(h[t - 1]))) throw new Error(`diag 第 ${t} 轮 LL 下降`);
      }
      return `diag LL ${h[0].toFixed(2)} → ${h[h.length - 1].toFixed(2)}`;
    });

    reports.push(s.report());
  }

  // ---------- PCA ----------
  {
    const s = suite('pca');
    const iris = loadIrisForPca();
    const pca = new PCA({ nComponents: 2 });
    pca.fit(iris.X);

    s.check('PCA：components_ 行正交（V·Vᵀ ≈ I）', () => {
      const G = matMul(pca.components_, transpose(pca.components_));
      let e = 0;
      for (let i = 0; i < G.rows; i++) for (let j = 0; j < G.cols; j++) {
        e = Math.max(e, Math.abs(G.get(i, j) - (i === j ? 1 : 0)));
      }
      if (e > 1e-8) throw new Error(`正交缺陷 ${e.toExponential(2)}`);
      return `正交缺陷 = ${e.toExponential(1)}`;
    });

    s.check('PCA：解释方差比降序且累计 ≤ 1', () => {
      const r = pca.explainedVarianceRatio_;
      for (let i = 1; i < r.length; i++) {
        if (r[i] > r[i - 1] + 1e-12) throw new Error(`解释方差比未降序: ${r[i - 1]} → ${r[i]}`);
      }
      let cum = 0;
      for (const v of r) cum += v;
      if (cum > 1 + 1e-9) throw new Error(`累计解释方差比 ${cum} > 1`);
      return `累计 = ${cum.toFixed(6)}`;
    });

    s.check('PCA：重构误差随 nComponents 增加而下降', () => {
      const errs = [];
      for (const kc of [1, 2, 3]) {
        const p = new PCA({ nComponents: kc });
        const Z = p.fitTransform(iris.X);
        const Xr = p.inverseTransform(Z);
        let e = 0;
        for (let i = 0; i < iris.X.rows; i++) for (let j = 0; j < iris.X.cols; j++) {
          e += (iris.X.get(i, j) - Xr.get(i, j)) ** 2;
        }
        errs.push(Math.sqrt(e / (iris.X.rows * iris.X.cols)));
      }
      if (!(errs[0] > errs[1] && errs[1] > errs[2])) throw new Error(`重构误差未递减: ${errs.map((v) => v.toFixed(4))}`);
      return `RMSE(k=1,2,3) = ${errs.map((v) => v.toFixed(4)).join(', ')}`;
    });

    s.check('PCA：Wine 前 2 主成分累计解释方差（记录实际值）', () => {
      const wine = loadWineForPca();
      const p = new PCA({ nComponents: 2 });
      p.fit(wine.X);
      const cum = p.explainedVarianceRatio_[0] + p.explainedVarianceRatio_[1];
      if (!(cum > 0 && cum <= 1 + 1e-9)) throw new Error(`累计解释方差异常: ${cum}`);
      return `Wine top-2 累计解释方差比 = ${(cum * 100).toFixed(2)}%`;
    });

    reports.push(s.report());
  }

  // ---------- Agglomerative ----------
  {
    const s = suite('agglomerative');
    const { X, y } = makeTwoBlobsGauss({ nPer: 60, std: 0.5, seed: 5 });
    const agg = new AgglomerativeClustering({ k: 2 });
    agg.fit(X);

    s.check('凝聚聚类：blobs 纯度 ≥ 0.95', () => {
      const purity = clusterPurity(y, agg.labels_, 2);
      if (purity < 0.95) throw new Error(`purity = ${purity.toFixed(4)}`);
      return `purity = ${purity.toFixed(4)}`;
    });

    s.check('凝聚聚类：标签恰好 k 个且确定性', () => {
      const set = new Set(agg.labels_);
      if (set.size !== 2) throw new Error(`标签数 ${set.size} ≠ 2`);
      const again = new AgglomerativeClustering({ k: 2 }).fit(X).labels_;
      for (let i = 0; i < again.length; i++) {
        if (again[i] !== agg.labels_[i]) throw new Error('两次拟合标签不一致');
      }
      return '2 个簇，两次拟合完全一致';
    });

    reports.push(s.report());
  }

  return reports;
}

// 复用 datasets.mjs 的官方实现（mms/mods 口径一致）
function clusterPurity(yTrue, yCluster, k) { return metrics.purity(yTrue, yCluster, k); }
function datasetsMakeBlobs() { return makeBlobs({ n: 300, k: 3 }); }
function loadIrisForPca() { return loadIris(); }
function loadWineForPca() { return loadWine(); }

// ==================== bench ====================

/** 聚类与降维模块基准 */
export function bench() {
  const results = [];

  // 1. KMeans on blobs
  {
    const { X, y } = makeBlobs({ n: 300, k: 3 });
    const km = new KMeans({ k: 3, seed: 42 });
    const t0 = now();
    km.fit(X);
    const ms = now() - t0;
    const purity = metrics.purity(y, km.labels_, 3);
    results.push({ name: 'KMeans blobs(300,k=3) 纯度', ok: purity >= 0.95, detail: `purity=${purity.toFixed(4)}, SSE=${km.inertia_.toFixed(2)}, 迭代 ${km.nIter_} 轮, ${ms.toFixed(1)}ms` });
  }

  // 2. GMM on blobs：LL 收敛 + 参数恢复
  {
    const { X, y } = makeBlobs({ n: 300, k: 3 });
    const gmm = new GMM({ k: 3, seed: 42, covariance: 'full', maxIter: 200, tol: 1e-8 });
    gmm.fit(X);
    const h = gmm.llHistory_;
    let mono = true;
    for (let t = 1; t < h.length; t++) if (h[t] < h[t - 1] - 1e-6 * Math.max(1, Math.abs(h[t - 1]))) mono = false;
    const purity = metrics.purity(y, gmm.labels_, 3);
    results.push({
      name: 'GMM blobs 对数似然收敛',
      ok: mono && h[h.length - 1] >= h[0] - 1e-9 && gmm.nIter_ < 200,
      detail: `LL ${h[0].toFixed(2)} → ${h[h.length - 1].toFixed(2)}（${gmm.nIter_} 轮收敛）, purity=${purity.toFixed(4)}`,
    });
    const { X: X2, means: trueMeans } = makeTwoBlobsGauss({ nPer: 200, std: 0.5, seed: 7 });
    const g2 = new GMM({ k: 2, seed: 42, covariance: 'full', maxIter: 200, tol: 1e-8 });
    g2.fit(X2);
    const err = meanRecoveryError(g2.means_, trueMeans);
    results.push({ name: 'GMM 参数恢复（已知均值）', ok: err < 0.3, detail: `均值最大误差 = ${err.toFixed(4)}` });
  }

  // 3. PCA on Iris / Wine
  {
    const iris = loadIris();
    const p1 = new PCA({ nComponents: 2 });
    p1.fit(iris.X);
    const cum1 = p1.explainedVarianceRatio_[0] + p1.explainedVarianceRatio_[1];
    results.push({ name: 'PCA Iris 前 2 主成分累计解释方差', ok: cum1 > 0.9 && cum1 <= 1 + 1e-9, detail: `${(cum1 * 100).toFixed(2)}%` });

    const wine = loadWine();
    const p2 = new PCA({ nComponents: 2 });
    p2.fit(wine.X);
    const cum2 = p2.explainedVarianceRatio_[0] + p2.explainedVarianceRatio_[1];
    results.push({ name: 'PCA Wine 前 2 主成分累计解释方差', ok: cum2 > 0.3 && cum2 <= 1 + 1e-9, detail: `${(cum2 * 100).toFixed(2)}%` });

    // 重构误差递减
    const errs = [1, 2, 3].map((kc) => {
      const p = new PCA({ nComponents: kc });
      const Z = p.fitTransform(iris.X);
      const Xr = p.inverseTransform(Z);
      let e = 0;
      for (let i = 0; i < iris.X.rows; i++) for (let j = 0; j < iris.X.cols; j++) e += (iris.X.get(i, j) - Xr.get(i, j)) ** 2;
      return Math.sqrt(e / (iris.X.rows * iris.X.cols));
    });
    results.push({
      name: 'PCA 重构误差随分量下降（Iris RMSE）',
      ok: errs[0] > errs[1] && errs[1] > errs[2],
      detail: `k=1:${errs[0].toFixed(4)}, k=2:${errs[1].toFixed(4)}, k=3:${errs[2].toFixed(4)}`,
    });
  }

  // 4. 凝聚聚类
  {
    const { X, y } = makeBlobs({ n: 150, k: 3 });
    const agg = new AgglomerativeClustering({ k: 3 });
    const t0 = now();
    agg.fit(X);
    const ms = now() - t0;
    const purity = metrics.purity(y, agg.labels_, 3);
    results.push({ name: 'Agglomerative blobs(150,k=3) 纯度', ok: purity >= 0.9, detail: `purity=${purity.toFixed(4)}, ${ms.toFixed(1)}ms` });
  }

  return { module: 'cluster', results };
}

/** 高精度计时（Node 22 可用 performance.now） */
function now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}
