/**
 * @file sequence.mjs — StarForge 序列模型：离散 HMM（前向/后向/Viterbi/Baum-Welch）与通用束搜索
 * 作者：晨星
 *
 * 算法谱系（世界经典成果的直接工程化）：
 * - 前向/后向算法：Baum & Eagon 1967 / Baum et al. 1970，Rabiner 1989 的缩放技巧
 * - Viterbi 解码：Viterbi 1967（卷积码最大似然序列估计），对数域实现
 * - Baum-Welch 参数估计：Baum et al. 1970（EM 在 HMM 上的特例）
 * - 束搜索：Lowerre 1976（HMM  speech recognition 的 REDUCE 程序）
 *
 * 可验证不变量（可被独立实现交叉验证的金标准）：
 * - 前向对数概率与暴力枚举（所有路径求和）一致（1e-10）
 * - Viterbi 路径概率 ≥ 任何单条枚举路径的概率（最优性）
 * - 缩放后向满足 Σ_i π_i·b_i(o₀)·β̂₀[i] = 1
 * - Baum-Welch 每轮对数似然单调不减；参数矩阵行随机
 * - beamSearch：beamWidth=1 等价贪心；beamWidth ≥ 真宽时找到最优
 */
import { Mat, createRng, argmax, logsumexp } from './linalg.mjs';
import { suite } from './verify.mjs';

/** 从概率向量中按分布抽样（累积和法） */
function sampleFrom(probs, rng) {
  const r = rng.next();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) return i;
  }
  return probs.length - 1;
}

/** 把矩阵每行归一化为概率分布（防御用户输入） */
function normalizeRowsMat(m) {
  const out = m.clone();
  for (let i = 0; i < out.rows; i++) {
    let s = 0;
    for (let j = 0; j < out.cols; j++) {
      const v = out.get(i, j);
      if (!(v >= 0)) throw new Error('概率矩阵含有负元素');
      out.set(i, j, v);
      s += v;
    }
    if (!(s > 0)) throw new Error('概率矩阵存在全零行');
    for (let j = 0; j < out.cols; j++) out.set(i, j, out.get(i, j) / s);
  }
  return out;
}

/** 均匀(0.25,1) 随机行随机矩阵：严格正，避免零概率导致的退化 */
function randomRowStochastic(rng, rows, cols) {
  const m = new Mat(rows, cols);
  for (let i = 0; i < rows; i++) {
    let s = 0;
    for (let j = 0; j < cols; j++) {
      const v = rng.uniform(0.25, 1);
      m.set(i, j, v);
      s += v;
    }
    for (let j = 0; j < cols; j++) m.set(i, j, m.get(i, j) / s);
  }
  return m;
}

// ==================== HiddenMarkovModel ====================

/**
 * 离散发射隐马尔可夫模型。
 *
 * 参数：π（起始概率，Float64Array）、A（转移矩阵 transMat_，行随机 Mat）、
 * B（发射矩阵 emitProb_，行随机 Mat）。λ = (π, A, B)。
 *
 * 概率计算全部经缩放前向/后向（Rabiner 1989 的 c[t] 缩放）或对数域，
 * 长序列下不溢出不消失；前向结果与暴力枚举在 1e-10 内一致。
 */
export class HiddenMarkovModel {
  /**
   * @param {object} opts
   * @param {number} opts.nStates 隐状态数
   * @param {number} opts.nObs 观测符号数（离散发射）
   * @param {number} [opts.seed=42] 随机初始化种子（未显式给参数时生效）
   * @param {number[][]} [opts.transMat] 显式转移矩阵（自动按行归一化）
   * @param {number[][]} [opts.emitProb] 显式发射矩阵（自动按行归一化）
   * @param {number[]} [opts.startProb] 显式起始概率（自动归一化）
   */
  constructor({ nStates, nObs, seed = 42, transMat, emitProb, startProb } = {}) {
    if (!Number.isInteger(nStates) || nStates < 1) throw new Error('HMM: nStates 必须是正整数');
    if (!Number.isInteger(nObs) || nObs < 1) throw new Error('HMM: nObs 必须是正整数');
    this.nStates = nStates;
    this.nObs = nObs;
    this.seed = seed;
    const rng = createRng(seed);
    this.transMat_ = transMat ? normalizeRowsMat(Mat.from2D(transMat)) : randomRowStochastic(rng, nStates, nStates);
    this.emitProb_ = emitProb ? normalizeRowsMat(Mat.from2D(emitProb)) : randomRowStochastic(rng, nStates, nObs);
    if (startProb) {
      const v = Float64Array.from(startProb);
      let s = 0;
      for (const x of v) { if (x < 0) throw new Error('HMM: 起始概率含有负元素'); s += x; }
      if (!(s > 0)) throw new Error('HMM: 起始概率全零');
      this.startProb_ = Float64Array.from(v, (x) => x / s);
    } else {
      this.startProb_ = new Float64Array(nStates).fill(1 / nStates);
    }
    if (this.transMat_.rows !== nStates || this.transMat_.cols !== nStates) throw new Error('HMM: transMat 维度应为 nStates×nStates');
    if (this.emitProb_.rows !== nStates || this.emitProb_.cols !== nObs) throw new Error('HMM: emitProb 维度应为 nStates×nObs');
    this.fitted_ = false;
    this.logLikelihood_ = null;
    this.nIter_ = 0;
    /** 每轮 Baum-Welch 的总对数似然历史（单调不减，可验证） */
    this.llHistory_ = [];
  }

  /** 观测序列合法性检查 */
  _checkObs(obs) {
    if (!Array.isArray(obs) || obs.length === 0) throw new Error('HMM: 观测序列必须是非空数组');
    for (const o of obs) {
      if (!Number.isInteger(o) || o < 0 || o >= this.nObs) throw new Error(`HMM: 观测符号 ${o} 越界（nObs=${this.nObs}）`);
    }
  }

  /**
   * 缩放前向算法（Rabiner 1989）。
   * @param {number[]} obs 观测序列
   * @returns {{logProb: number, alpha: Mat(T×nStates), scales: Float64Array}}
   *   alpha 为缩放后的 α̂（每行和 1）；logProb = −Σ_t log c[t]
   */
  _scaledForward(obs) {
    const T = obs.length, N = this.nStates;
    const A = this.transMat_, B = this.emitProb_, pi = this.startProb_;
    const alpha = new Mat(T, N);
    const scales = new Float64Array(T);
    let s = 0;
    for (let i = 0; i < N; i++) {
      const v = pi[i] * B.get(i, obs[0]);
      alpha.set(0, i, v);
      s += v;
    }
    if (s === 0) return { logProb: -Infinity, alpha, scales };
    const c0 = 1 / s;
    for (let i = 0; i < N; i++) alpha.set(0, i, alpha.get(0, i) * c0);
    scales[0] = c0;
    for (let t = 1; t < T; t++) {
      s = 0;
      for (let j = 0; j < N; j++) {
        let acc = 0;
        for (let i = 0; i < N; i++) acc += alpha.get(t - 1, i) * A.get(i, j);
        const v = acc * B.get(j, obs[t]);
        alpha.set(t, j, v);
        s += v;
      }
      if (s === 0) return { logProb: -Infinity, alpha, scales };
      const c = 1 / s;
      for (let j = 0; j < N; j++) alpha.set(t, j, alpha.get(t, j) * c);
      scales[t] = c;
    }
    let logProb = 0;
    for (let t = 0; t < T; t++) logProb -= Math.log(scales[t]);
    return { logProb, alpha, scales };
  }

  /** 缩放后向算法（与前向共用 scales） */
  _scaledBackward(obs, scales) {
    const T = obs.length, N = this.nStates;
    const A = this.transMat_, B = this.emitProb_;
    const beta = new Mat(T, N);
    for (let i = 0; i < N; i++) beta.set(T - 1, i, scales[T - 1]);
    for (let t = T - 2; t >= 0; t--) {
      for (let i = 0; i < N; i++) {
        let acc = 0;
        for (let j = 0; j < N; j++) acc += A.get(i, j) * B.get(j, obs[t + 1]) * beta.get(t + 1, j);
        beta.set(t, i, acc * scales[t]);
      }
    }
    return beta;
  }

  /**
   * 前向算法（对外接口）。
   * @param {number[]} obs
   * @returns {{logProb: number, alpha: Mat}} alpha 为缩放 α̂（每行和 1）
   */
  forward(obs) {
    this._checkObs(obs);
    const { logProb, alpha } = this._scaledForward(obs);
    return { logProb, alpha };
  }

  /**
   * 后向算法（对外接口）。
   * @param {number[]} obs
   * @returns {Mat} 缩放 β̂（T×nStates）。不变量：Σ_i π_i·b_i(o₀)·β̂₀[i] = 1
   */
  backward(obs) {
    this._checkObs(obs);
    const { scales } = this._scaledForward(obs);
    return this._scaledBackward(obs, scales);
  }

  /**
   * Viterbi 解码（对数域，Viterbi 1967）：最大后验状态路径。
   * @param {number[]} obs
   * @returns {{path: number[], logProb: number}}
   */
  viterbi(obs) {
    this._checkObs(obs);
    const T = obs.length, N = this.nStates;
    const A = this.transMat_, B = this.emitProb_, pi = this.startProb_;
    const delta = new Mat(T, N);
    const psi = new Mat(T, N);
    const logPi = Array.from(pi, (p) => Math.log(p));
    const logB = [];
    for (let j = 0; j < N; j++) logB.push(Array.from(B.row(j), (p) => Math.log(p)));
    for (let i = 0; i < N; i++) delta.set(0, i, logPi[i] + logB[i][obs[0]]);
    for (let t = 1; t < T; t++) {
      for (let j = 0; j < N; j++) {
        let best = -Infinity;
        let bi = 0;
        for (let i = 0; i < N; i++) {
          const v = delta.get(t - 1, i) + Math.log(A.get(i, j));
          if (v > best) { best = v; bi = i; }
        }
        delta.set(t, j, best + logB[j][obs[t]]);
        psi.set(t, j, bi);
      }
    }
    const lastRow = delta.row(T - 1);
    let last = argmax(lastRow);
    const path = new Array(T);
    path[T - 1] = last;
    for (let t = T - 1; t > 0; t--) {
      last = psi.get(t, last);
      path[t - 1] = last;
    }
    return { path, logProb: lastRow[path[T - 1]] };
  }

  /**
   * Baum-Welch 参数估计（EM，Baum et al. 1970），支持多序列。
   * @param {number[]|number[][]} observations 单序列或序列集合
   * @param {{maxIter?: number, tol?: number}} [opts]
   * @returns {{logLikelihood: number, nIter: number}}
   */
  fit(observations, { maxIter = 100, tol = 1e-6 } = {}) {
    const seqs = Array.isArray(observations) && Array.isArray(observations[0]) ? observations : [observations];
    for (const s of seqs) this._checkObs(s);
    if (seqs.length === 0) throw new Error('HMM.fit: 观测序列集合为空');
    this.llHistory_ = [];
    let prevLL = -Infinity;
    let iter = 0;
    for (; iter < maxIter; iter++) {
      let totalLL = 0;
      // —— E 步：跨序列累加充分统计量 ——
      const N = this.nStates;
      const piAcc = new Float64Array(N);
      const numA = new Mat(N, N);
      const denA = new Float64Array(N);
      const numB = new Mat(N, this.nObs);
      const denB = new Float64Array(N);
      for (const obs of seqs) {
        const { logProb, alpha, scales } = this._scaledForward(obs);
        if (!Number.isFinite(logProb)) throw new Error('HMM.fit: 观测序列在当前参数下概率为 0');
        totalLL += logProb;
        const beta = this._scaledBackward(obs, scales);
        const T = obs.length;
        // γ[t][i] = α̂[t][i]·β̂[t][i] / c[t]（理论上对 i 求和为 1）
        const gamma = new Mat(T, N);
        for (let t = 0; t < T; t++) {
          let gsum = 0;
          for (let i = 0; i < N; i++) {
            const g = alpha.get(t, i) * beta.get(t, i) / scales[t];
            gamma.set(t, i, g);
            gsum += g;
          }
          if (gsum > 0) for (let i = 0; i < N; i++) gamma.set(t, i, gamma.get(t, i) / gsum); // 数值安全
        }
        // π 统计量
        for (let i = 0; i < N; i++) piAcc[i] += gamma.get(0, i);
        // γ 统计量
        for (let t = 0; t < T; t++) for (let i = 0; i < N; i++) {
          denB[i] += gamma.get(t, i);
          numB.set(i, obs[t], numB.get(i, obs[t]) + gamma.get(t, i));
        }
        // ξ 统计量（t = 0..T−2）
        for (let t = 0; t < T - 1; t++) {
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              const xi = alpha.get(t, i) * this.transMat_.get(i, j) * this.emitProb_.get(j, obs[t + 1]) * beta.get(t + 1, j);
              numA.set(i, j, numA.get(i, j) + xi);
              denA[i] += xi;
            }
          }
        }
      }
      this.llHistory_.push(totalLL);
      if (iter > 0 && Math.abs(totalLL - prevLL) < tol) break;
      // —— M 步：行归一化更新（分母退化时保留旧行）——
      const nSeq = seqs.length;
      const newPi = new Float64Array(N);
      for (let i = 0; i < N; i++) newPi[i] = piAcc[i] / nSeq;
      let piSum = 0;
      for (const v of newPi) piSum += v;
      if (piSum > 0) for (let i = 0; i < N; i++) newPi[i] /= piSum;
      else newPi.set(this.startProb_);
      const newA = this.transMat_.clone();
      for (let i = 0; i < N; i++) {
        if (denA[i] > 0) {
          for (let j = 0; j < N; j++) newA.set(i, j, numA.get(i, j) / denA[i]);
        } // 否则保留旧行
      }
      const newB = this.emitProb_.clone();
      for (let i = 0; i < N; i++) {
        if (denB[i] > 0) {
          for (let k = 0; k < this.nObs; k++) newB.set(i, k, numB.get(i, k) / denB[i]);
        } // 否则保留旧行
      }
      this.startProb_ = newPi;
      this.transMat_ = newA;
      this.emitProb_ = newB;
      prevLL = totalLL;
    }
    this.nIter_ = iter;
    this.fitted_ = true;
    // logLikelihood_ 必须是"最终参数"的 LL（最后可能还有一次未计分的 M 步）
    const finalLL = this._totalLogProb(seqs);
    this.logLikelihood_ = finalLL;
    return { logLikelihood: finalLL, nIter: this.nIter_ };
  }

  /** 多条序列在当前参数下的对数似然总和（前向算法） */
  _totalLogProb(seqs) {
    let total = 0;
    for (const obs of seqs) total += this._scaledForward(obs).logProb;
    return total;
  }

  /**
   * 从当前模型采样一条观测序列。
   * @param {number} T 序列长度
   * @param {ReturnType<typeof createRng>} [rng] 默认用 this.seed 构造（可复现）
   * @returns {{obs: number[], states: number[]}}
   */
  sample(T, rng = createRng(this.seed)) {
    if (!Number.isInteger(T) || T < 1) throw new Error('HMM.sample: T 必须是正整数');
    const obs = new Array(T);
    const states = new Array(T);
    let s = sampleFrom(this.startProb_, rng);
    states[0] = s;
    obs[0] = sampleFrom(this.emitProb_.row(s), rng);
    for (let t = 1; t < T; t++) {
      s = sampleFrom(this.transMat_.row(s), rng);
      states[t] = s;
      obs[t] = sampleFrom(this.emitProb_.row(s), rng);
    }
    return { obs, states };
  }

  /** 单条路径的对数概率（含起始、转移、发射） */
  _pathLogProb(obs, path) {
    const A = this.transMat_, B = this.emitProb_;
    let lp = Math.log(this.startProb_[path[0]]) + Math.log(B.get(path[0], obs[0]));
    for (let t = 1; t < obs.length; t++) {
      lp += Math.log(A.get(path[t - 1], path[t])) + Math.log(B.get(path[t], obs[t]));
    }
    return lp;
  }

  /**
   * 暴力枚举参考实现：对所有 nStates^T 条路径求和（金标准，仅供交叉验证）。
   * @param {number[]} obs
   * @returns {number} log P(obs)
   */
  bruteForceLogProb(obs) {
    this._checkObs(obs);
    const T = obs.length;
    const logs = [];
    const path = new Array(T);
    const rec = (t) => {
      if (t === T) { logs.push(this._pathLogProb(obs, path)); return; }
      for (let i = 0; i < this.nStates; i++) { path[t] = i; rec(t + 1); }
    };
    rec(0);
    return logsumexp(logs);
  }

  /** 暴力枚举参考实现：所有单条路径中的最大对数概率（Viterbi 最优性金标准） */
  bruteForceMaxPathLogProb(obs) {
    this._checkObs(obs);
    const T = obs.length;
    let best = -Infinity;
    const path = new Array(T);
    const rec = (t) => {
      if (t === T) { best = Math.max(best, this._pathLogProb(obs, path)); return; }
      for (let i = 0; i < this.nStates; i++) { path[t] = i; rec(t + 1); }
    };
    rec(0);
    return best;
  }
}

// ==================== beamSearch ====================

/**
 * 通用束搜索（Lowerre 1976）。
 *
 * 每一步对当前 beam 中每个前缀调用 stepFn 获取候选扩展，按累积对数概率
 * 剪枝保留 beamWidth 个最优假设；遇到 endToken 的候选进入完成集。
 * 返回得分最高的 top-beamWidth 条完整序列。
 *
 * @param {object} opts
 * @param {(beams: Array<{tokens: any[], logProb: number, state: any}>) => Array<Array<{token: any, logProb: number, state: any}>>} opts.stepFn
 *   输入当前 beam（前缀数组），返回与 beam 逐对应的候选扩展列表；扩展的 logProb 为增量对数概率
 * @param {number} [opts.beamWidth=4] 束宽
 * @param {number} [opts.maxLen=20] 最大扩展步数（不含起始符）
 * @param {any} opts.startToken 起始符（首条 beam 的 tokens 为 [startToken]）
 * @param {any} [opts.endToken] 结束符；候选 token 等于 endToken 时该序列进入完成集
 * @param {(seq: {tokens: any[], logProb: number, state: any}) => number} [opts.scoreFn] 完成序列的最终排序分（默认 logProb）
 * @returns {Array<{tokens: any[], logProb: number, state: any}>} 按得分降序的完整序列
 */
export function beamSearch({ stepFn, beamWidth = 4, maxLen = 20, startToken, endToken, scoreFn } = {}) {
  if (typeof stepFn !== 'function') throw new Error('beamSearch: stepFn 必须是函数');
  if (!Number.isInteger(beamWidth) || beamWidth < 1) throw new Error('beamSearch: beamWidth 必须是正整数');
  if (!Number.isInteger(maxLen) || maxLen < 1) throw new Error('beamSearch: maxLen 必须是正整数');
  let beams = [{ tokens: [startToken], logProb: 0, state: undefined }];
  const completed = [];
  for (let step = 0; step < maxLen && beams.length > 0; step++) {
    const exts = stepFn(beams);
    const next = [];
    for (let bi = 0; bi < beams.length; bi++) {
      const list = exts[bi] || [];
      for (const e of list) {
        const cand = {
          tokens: [...beams[bi].tokens, e.token],
          logProb: beams[bi].logProb + (Number.isFinite(e.logProb) ? e.logProb : 0),
          state: e.state,
        };
        if (endToken !== undefined && e.token === endToken) completed.push(cand);
        else next.push(cand);
      }
    }
    // 确定性剪枝：累积对数概率降序，令牌字典序升序
    next.sort((a, b) => (b.logProb - a.logProb) || compareTokens(a.tokens, b.tokens));
    beams = next.slice(0, beamWidth);
  }
  // 排名：完成集优先（未遇到 endToken 时回退到截断序列）
  const pool = completed.length > 0 ? completed : beams;
  const scored = pool.map((seq) => ({ ...seq, _score: scoreFn ? scoreFn(seq) : seq.logProb }));
  scored.sort((a, b) => (b._score - a._score) || compareTokens(a.tokens, b.tokens));
  return scored.slice(0, beamWidth).map(({ tokens, logProb, state }) => ({ tokens, logProb, state }));
}

/** 令牌序列的确定性字典序比较（混合类型安全） */
function compareTokens(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const sa = String(a[i]);
    const sb = String(b[i]);
    if (sa < sb) return -1;
    if (sa > sb) return 1;
  }
  return a.length - b.length;
}

// ==================== selfCheck ====================

/** 构造一个小的确定性玩具 HMM（2 状态、3 符号），参数手工指定 */
function toyHmm() {
  return new HiddenMarkovModel({
    nStates: 2,
    nObs: 3,
    transMat: [[0.7, 0.3], [0.4, 0.6]],
    emitProb: [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]],
    startProb: [0.6, 0.4],
  });
}

/** 序列模型模块的不变量自检（suite 报告数组） */
export function selfCheck() {
  const reports = [];

  // ---------- 前向 / 后向 ----------
  {
    const s = suite('hmm-forward-backward');
    const cases = [
      { obs: [0, 1, 2], hmm: toyHmm() },
      { obs: [2, 2, 1, 0, 1], hmm: toyHmm() },
      { obs: [1, 0, 2, 2, 0, 1], hmm: toyHmm() },
      {
        obs: [0, 2, 1, 1, 0],
        hmm: new HiddenMarkovModel({
          nStates: 3, nObs: 3, seed: 9,
          transMat: [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]],
          emitProb: [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]],
          startProb: [0.5, 0.3, 0.2],
        }),
      },
    ];

    s.check('前向对数概率与暴力枚举一致（T≤6，1e-10）', () => {
      let maxErr = 0;
      for (const { obs, hmm } of cases) {
        const { logProb } = hmm.forward(obs);
        const bf = hmm.bruteForceLogProb(obs);
        maxErr = Math.max(maxErr, Math.abs(logProb - bf));
      }
      if (maxErr > 1e-10) throw new Error(`最大偏差 ${maxErr.toExponential(2)} > 1e-10`);
      return `4 个例子最大偏差 = ${maxErr.toExponential(2)}`;
    });

    s.check('缩放后向不变量：Σ_i π_i·b_i(o₀)·β̂₀[i] = 1', () => {
      let maxErr = 0;
      for (const { obs, hmm } of cases) {
        const beta = hmm.backward(obs);
        let sum = 0;
        for (let i = 0; i < hmm.nStates; i++) {
          sum += hmm.startProb_[i] * hmm.emitProb_.get(i, obs[0]) * beta.get(0, i);
        }
        maxErr = Math.max(maxErr, Math.abs(sum - 1));
      }
      if (maxErr > 1e-9) throw new Error(`不变量偏差 ${maxErr.toExponential(2)}`);
      return `最大偏差 = ${maxErr.toExponential(2)}`;
    });

    s.check('缩放 α̂ 每行和为 1（无下溢）', () => {
      for (const { obs, hmm } of cases) {
        const { alpha } = hmm.forward(obs);
        for (let t = 0; t < alpha.rows; t++) {
          let s2 = 0;
          for (let i = 0; i < alpha.cols; i++) s2 += alpha.get(t, i);
          if (Math.abs(s2 - 1) > 1e-9) throw new Error(`t=${t} 行和 = ${s2}`);
        }
      }
      return '全部行和 = 1';
    });

    reports.push(s.report());
  }

  // ---------- Viterbi ----------
  {
    const s = suite('hmm-viterbi');
    const cases = [
      { obs: [0, 1, 2], hmm: toyHmm() },
      { obs: [2, 2, 1, 0, 1], hmm: toyHmm() },
      { obs: [1, 0, 2, 2, 0, 1], hmm: toyHmm() },
    ];

    s.check('Viterbi 路径合法且概率 ≥ 任何单条枚举路径', () => {
      for (const { obs, hmm } of cases) {
        const { path, logProb } = hmm.viterbi(obs);
        if (path.length !== obs.length) throw new Error('路径长度错误');
        for (const st of path) {
          if (!Number.isInteger(st) || st < 0 || st >= hmm.nStates) throw new Error(`非法状态 ${st}`);
        }
        const bfBest = hmm.bruteForceMaxPathLogProb(obs);
        if (logProb < bfBest - 1e-9) throw new Error(`Viterbi ${logProb} < 枚举最优 ${bfBest}`);
        // 同时验证路径自身概率自洽
        const own = hmm._pathLogProb(obs, path);
        if (Math.abs(own - logProb) > 1e-9) throw new Error(`路径概率不自洽: ${own} vs ${logProb}`);
      }
      return '3 个例子全部满足最优性';
    });

    s.check('Viterbi 与暴力枚举最优路径概率精确相等', () => {
      for (const { obs, hmm } of cases) {
        const { logProb } = hmm.viterbi(obs);
        const bfBest = hmm.bruteForceMaxPathLogProb(obs);
        if (Math.abs(logProb - bfBest) > 1e-10) throw new Error(`偏差 ${Math.abs(logProb - bfBest).toExponential(2)}`);
      }
      return '3 个例子偏差 < 1e-10';
    });

    reports.push(s.report());
  }

  // ---------- Baum-Welch ----------
  {
    const s = suite('hmm-baum-welch');
    const truth = new HiddenMarkovModel({
      nStates: 2, nObs: 3, seed: 123,
      transMat: [[0.9, 0.1], [0.15, 0.85]],
      emitProb: [[0.8, 0.15, 0.05], [0.05, 0.15, 0.8]],
      startProb: [0.5, 0.5],
    });
    const rng = createRng(7);
    const seqs = Array.from({ length: 4 }, () => truth.sample(120, rng).obs);

    s.check('Baum-Welch：每轮 LL 单调不减（多序列）', () => {
      const hmm = new HiddenMarkovModel({ nStates: 2, nObs: 3, seed: 42 });
      const { logLikelihood, nIter } = hmm.fit(seqs, { maxIter: 60, tol: 1e-7 });
      const h = hmm.llHistory_;
      for (let t = 1; t < h.length; t++) {
        if (h[t] < h[t - 1] - 1e-6 * Math.max(1, Math.abs(h[t - 1]))) {
          throw new Error(`第 ${t} 轮 LL 下降: ${h[t - 1].toFixed(6)} → ${h[t].toFixed(6)}`);
        }
      }
      return `${nIter} 轮，LL ${h[0].toFixed(2)} → ${logLikelihood.toFixed(2)}`;
    });

    s.check('Baum-Welch：估计参数行随机', () => {
      const hmm = new HiddenMarkovModel({ nStates: 2, nObs: 3, seed: 42 });
      hmm.fit(seqs, { maxIter: 60, tol: 1e-7 });
      const check = (m, name) => {
        for (let i = 0; i < m.rows; i++) {
          let s2 = 0;
          for (let j = 0; j < m.cols; j++) {
            if (m.get(i, j) < -1e-12) throw new Error(`${name}(${i},${j}) 为负`);
            s2 += m.get(i, j);
          }
          if (Math.abs(s2 - 1) > 1e-9) throw new Error(`${name} 第 ${i} 行和 = ${s2}`);
        }
      };
      check(hmm.transMat_, 'transMat');
      check(hmm.emitProb_, 'emitProb');
      let piSum = 0;
      for (const v of hmm.startProb_) { if (v < -1e-12) throw new Error('startProb 为负'); piSum += v; }
      if (Math.abs(piSum - 1) > 1e-9) throw new Error(`startProb 和 = ${piSum}`);
      return 'A、B 行和=1，π 和=1';
    });

    s.check('Baum-Welch：已知参数下恢复转移矩阵（排列对齐误差 < 0.1）', () => {
      const hmm = new HiddenMarkovModel({ nStates: 2, nObs: 3, seed: 42 });
      hmm.fit(seqs, { maxIter: 100, tol: 1e-8 });
      const err = bestPermError(hmm.transMat_, truth.transMat_);
      if (err >= 0.1) throw new Error(`转移矩阵最大误差 ${err.toFixed(4)} ≥ 0.1`);
      return `转移矩阵最大误差 = ${err.toFixed(4)}`;
    });

    s.check('Baum-Welch：单序列与多序列接口一致性（LL 可加）', () => {
      const hmm = new HiddenMarkovModel({ nStates: 2, nObs: 3, seed: 42 });
      hmm.fit(seqs, { maxIter: 40, tol: 1e-9 });
      let sum = 0;
      for (const obs of seqs) sum += hmm.forward(obs).logProb;
      if (Math.abs(sum - hmm.logLikelihood_) > 1e-6) throw new Error(`LL 不加和: ${sum} vs ${hmm.logLikelihood_}`);
      return `4 条序列 LL 加和一致（${sum.toFixed(2)}）`;
    });

    reports.push(s.report());
  }

  // ---------- beamSearch ----------
  {
    const s = suite('beam-search');

    // 确定性玩具格子：start → {a: 0, b: −0.1}；a → END(−10)；b → END(0)
    const stepFn = (beams) => beams.map((beam) => {
      const last = beam.tokens[beam.tokens.length - 1];
      if (last === '<s>') return [{ token: 'a', logProb: 0, state: 1 }, { token: 'b', logProb: -0.1, state: 1 }];
      if (last === 'a') return [{ token: '<e>', logProb: -10, state: 2 }];
      return [{ token: '<e>', logProb: 0, state: 2 }];
    });

    s.check('beamSearch：beamWidth=1 等价贪心', () => {
      const top = beamSearch({ stepFn, beamWidth: 1, maxLen: 2, startToken: '<s>', endToken: '<e>' });
      if (top.length !== 1) throw new Error(`返回 ${top.length} 条`);
      const seq = top[0];
      const greedyTokens = ['<s>', 'a', '<e>'];
      const greedyLogProb = 0 + (-10);
      for (let i = 0; i < greedyTokens.length; i++) {
        if (seq.tokens[i] !== greedyTokens[i]) throw new Error(`贪心路径不一致: ${seq.tokens} vs ${greedyTokens}`);
      }
      if (Math.abs(seq.logProb - greedyLogProb) > 1e-12) throw new Error(`贪心概率不一致: ${seq.logProb}`);
      return `贪心路径 ${seq.tokens.join('→')}，logProb=${seq.logProb}`;
    });

    s.check('beamSearch：beamWidth ≥ 真宽时找到全局最优', () => {
      const top = beamSearch({ stepFn, beamWidth: 2, maxLen: 2, startToken: '<s>', endToken: '<e>' });
      if (top[0].tokens[1] !== 'b') throw new Error(`未找到最优路径: ${top[0].tokens}`);
      if (Math.abs(top[0].logProb - (-0.1)) > 1e-12) throw new Error(`最优概率错误: ${top[0].logProb}`);
      // 窄束确实次优（证明该玩具需要宽度 2）
      const narrow = beamSearch({ stepFn, beamWidth: 1, maxLen: 2, startToken: '<s>', endToken: '<e>' })[0];
      if (!(narrow.logProb < top[0].logProb)) throw new Error('窄束未体现次优性');
      return `beamWidth=2 找到最优 ${top[0].tokens.join('→')}（logProb=${top[0].logProb} > 窄束 ${narrow.logProb}）`;
    });

    s.check('beamSearch：返回序列按得分降序且数量 ≤ beamWidth', () => {
      const top = beamSearch({ stepFn, beamWidth: 2, maxLen: 2, startToken: '<s>', endToken: '<e>' });
      for (let i = 1; i < top.length; i++) {
        if (top[i].logProb > top[i - 1].logProb + 1e-12) throw new Error('未按得分降序');
      }
      return `返回 ${top.length} 条完整序列`;
    });

    reports.push(s.report());
  }

  return reports;
}

/** 两个矩阵在最优行列排列下的最大逐元素误差（标签切换金标准） */
function bestPermError(est, truth) {
  const n = est.rows;
  const perms = permute(n);
  let best = Infinity;
  for (const p of perms) {
    let e = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      e = Math.max(e, Math.abs(est.get(i, j) - truth.get(p[i], p[j])));
    }
    best = Math.min(best, e);
  }
  return best;
}

/** 0..n-1 的全部排列 */
function permute(n) {
  const out = [];
  const cur = [];
  const used = new Array(n).fill(false);
  const rec = () => {
    if (cur.length === n) { out.push(cur.slice()); return; }
    for (let i = 0; i < n; i++) {
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

/** 准确率（按最优标签映射对齐，处理标签切换） */
function bestMappingAccuracy(yTrue, yPred, k) {
  const perms = permute(k);
  let best = 0;
  for (const p of perms) {
    // p[c] = 预测簇 c 应对齐到的真实类别
    let ok = 0;
    for (let i = 0; i < yTrue.length; i++) if (p[yPred[i]] === yTrue[i]) ok++;
    best = Math.max(best, ok / yTrue.length);
  }
  return best;
}

// ==================== bench ====================

/** 序列模型模块基准 */
export function bench() {
  const results = [];

  // 已知参数的两状态 HMM：强自转移 + 强区分发射
  const truth = new HiddenMarkovModel({
    nStates: 2, nObs: 3, seed: 2024,
    transMat: [[0.9, 0.1], [0.1, 0.9]],
    emitProb: [[0.85, 0.1, 0.05], [0.05, 0.1, 0.85]],
    startProb: [0.5, 0.5],
  });
  const rng = createRng(42);
  const { obs, states } = truth.sample(200, rng);

  // 1. Baum-Welch 参数恢复
  const est = new HiddenMarkovModel({ nStates: 2, nObs: 3, seed: 1 });
  const t0 = now();
  const { logLikelihood, nIter } = est.fit(obs, { maxIter: 100, tol: 1e-8 });
  const ms = now() - t0;
  const h = est.llHistory_;
  let mono = true;
  for (let t = 1; t < h.length; t++) if (h[t] < h[t - 1] - 1e-6 * Math.max(1, Math.abs(h[t - 1]))) mono = false;
  const transErr = bestPermError(est.transMat_, truth.transMat_);
  const emitErr = bestPermError(est.emitProb_, truth.emitProb_);
  results.push({ name: 'Baum-Welch 转移矩阵最大误差', ok: transErr < 0.15 && mono, detail: `${transErr.toFixed(4)}（LL ${h[0].toFixed(1)}→${logLikelihood.toFixed(1)}，${nIter} 轮，${ms.toFixed(1)}ms）` });
  results.push({ name: 'Baum-Welch 发射矩阵最大误差', ok: emitErr < 0.2, detail: `${emitErr.toFixed(4)}` });

  // 2. Viterbi 状态恢复
  const { path } = est.viterbi(obs);
  const acc = bestMappingAccuracy(states, path, 2);
  results.push({ name: 'Viterbi 状态恢复准确率', ok: acc > 0.8, detail: `${(acc * 100).toFixed(2)}%（标签切换已对齐）` });

  // 3. forward vs 暴力枚举：一致性 + 耗时
  const small = obs.slice(0, 8);
  const tf = now();
  const fwd = est.forward(small).logProb;
  const tfMs = now() - tf;
  const tb = now();
  const bf = est.bruteForceLogProb(small);
  const tbMs = now() - tb;
  results.push({
    name: 'forward vs 暴力枚举（T=8）',
    ok: Math.abs(fwd - bf) < 1e-10,
    detail: `logProb 偏差 ${Math.abs(fwd - bf).toExponential(1)}；forward ${tfMs.toFixed(3)}ms vs 枚举 ${tbMs.toFixed(3)}ms（${(tbMs / Math.max(tfMs, 1e-6)).toFixed(0)}×）`,
  });

  return { module: 'sequence', results };
}

/** 高精度计时 */
function now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}
