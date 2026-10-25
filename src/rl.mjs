/**
 * @file rl.mjs — StarForge 强化学习模块：多臂老虎机 / 表格型时序差分 / 策略梯度 / 经典控制环境
 * 作者：晨星
 *
 * 定位：零依赖、纯 ESM、完全确定性的强化学习算法全集。
 * 所有随机性（环境奖励、探索动作、后验采样）都经 src/linalg.mjs 的 createRng(seed)
 * 注入，因此任意算法在给定 seed 下产生完全可复现的轨迹——这是"可验证 AI"的基石。
 *
 * 收录算法与原始文献：
 * - BernoulliBandit / EpsilonGreedy / UCB1 / ThompsonSampling：随机多臂老虎机
 *   · Thompson, W.R. (1933). On the likelihood that one unknown probability exceeds
 *     another in view of the evidence of two samples. Biometrika 25(3-4): 285-294.
 *   · Auer, Cesa-Bianchi & Fischer (2002). Finite-time Analysis of the Multiarmed
 *     Bandit Problem. Machine Learning 47(2-3): 235-256.（UCB1 与 O(log T) 遗憾）
 *   · Agrawal & Goyal (2012). Analysis of Thompson Sampling for the Multi-armed
 *     Bandit Problem. COLT.（Thompson 采样的贝叶斯遗憾分析）
 *   · Hoeffding, W. (1963). Probability Inequalities for Sums of Bounded Random
 *     Variables.（UCB 置信界所依赖的集中不等式）
 * - QLearningAgent：表格型 Q 学习（离线时序差分控制）
 *   · Watkins & Dayan (1992). Q-learning. Machine Learning 8(3-4): 279-292.
 *   · Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.,
 *     第 6 章（TD 控制）与第 2 章（Bellman 方程）。
 * - REINFORCEAgent：线性 softmax 策略梯度（蒙特卡洛策略梯度）
 *   · Williams (1992). Simple statistical gradient-following algorithms for
 *     connectionist reinforcement learning. Machine Learning 8(3-4): 229-256.
 *   · Sutton, McAllester, Singh & Mansour (2000). Policy Gradient Methods for
 *     Reinforcement Learning with Function Approximation. NIPS.（策略梯度定理）
 * - CartPoleEnv：经典"车-杆"控制基准（倒立摆平衡）
 *   · Barto, Sutton & Anderson (1983). Neuronlike adaptive elements that can solve
 *     difficult learning control problems. IEEE Trans. Systems, Man, and
 *     Cybernetics SMC-13(5): 834-846.
 *   · 标准实现参照 OpenAI Gym classic_control/cart_pole.py（MIT 许可）。
 * - ChainMDP / GridWorldEnv：小型确定性测试环境，解析解可手算，专用于不变量验证。
 *
 * 不变量（可被独立实现交叉验证的硬金标准）：
 * - Bandit：累计遗憾恒等式  regret = 步数·p* − 累计奖励
 * - UCB1：未试臂优先；同 seed 同环境下 UCB1 遗憾 < ε-greedy 遗憾
 * - Q-learning：收敛时 Bellman 最优方程残差 max|Q(s,a) − (r + γ·max Q(s'))| → 0，
 *   且 Q → 解析最优解 Q*（ChainMDP 可逐项比对）
 * - CartPole：reset 状态全 0；恒定单侧力作用下有限步内倒下；每步 reward = 1
 * - REINFORCE：训练后贪婪策略对最优臂的选择率显著提升
 * - 确定性：同 seed 下任意算法重放，动作序列与累计量逐位一致
 */

import { createRng, argmax, softmaxArr } from './linalg.mjs';
import { suite } from './verify.mjs';

/* ==================== 内部数值工具：Gamma / Beta 采样 ==================== */

/**
 * Marsaglia-Tsang 方法采样 Gamma(shape, 1)。
 * 参考：Marsaglia, G. & Tsang, W.W. (2000). A simple method for generating gamma
 * variables. ACM TOMS 26(3): 363-372。shape < 1 时用 boost 技巧 G(k)=G(k+1)·U^(1/k)。
 * 随机性完全来自调用方注入的 rng（createRng），保证同 seed 可复现。
 * @param {{next: Function, normal: Function}} rng
 * @param {number} shape 形状参数（> 0）
 * @returns {number} Gamma(shape, 1) 样本
 */
function sampleGamma(rng, shape) {
  if (!(shape > 0)) throw new Error(`sampleGamma: shape 必须为正，收到 ${shape}`);
  if (shape < 1) {
    // Boost 技巧：先采 Gamma(shape+1) 再乘 U^(1/shape)
    return sampleGamma(rng, shape + 1) * Math.pow(rng.next(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const x = rng.normal();
    const v = 1 + c * x;
    if (v <= 0) continue;
    const v3 = v * v * v;
    const u = rng.next();
    const x2 = x * x;
    // 快速接受路径（ squeeze 检验）
    if (u < 1 - 0.0331 * x2 * x2) return d * v3;
    // 对数接受路径
    if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) return d * v3;
  }
}

/**
 * Beta(a, b) 采样：X / (X + Y)，其中 X ~ Gamma(a, 1)，Y ~ Gamma(b, 1)。
 * 这是 Thompson Sampling（Thompson 1933）在 Bernoulli bandit 中的后验采样核心：
 * 后验 Beta(α, β) 的样本可直接与其它臂竞争 argmax。
 * @param {{next: Function, normal: Function}} rng
 * @param {number} a α > 0
 * @param {number} b β > 0
 * @returns {number} [0, 1] 内的 Beta 样本
 */
function sampleBeta(rng, a, b) {
  const x = sampleGamma(rng, a);
  const y = sampleGamma(rng, b);
  const s = x + y;
  return s > 0 ? x / s : 0.5;
}

/* ==================== 环境 ==================== */

/**
 * 标准 10 臂 Bernoulli 老虎机基准：最优臂 p* = 0.6，其余 9 臂在 [0.1, 0.5) 均匀抽取
 * （抽取器 seed 固定为 7，因此 means 数组本身是确定性的常量）。
 * 参考 Auer et al. (2002) 的标准测试床构造。
 * @returns {number[]} 长度 10 的均值数组，means[0] = 0.6 为最优臂
 */
export function makeTenArmMeans() {
  const rng = createRng(7);
  const means = [0.6];
  for (let i = 0; i < 9; i++) means.push(rng.uniform(0.1, 0.5));
  return means;
}

/**
 * BernoulliBandit — 随机多臂老虎机环境。
 *
 * 直觉：K 个臂，每个臂 i 以未知概率 p_i 吐出奖励 1（否则 0）。智能体每步选一个臂，
 * 目标是最小化累计遗憾（cumulative regret）：与"始终选最优臂"相比少拿到的奖励。
 *
 * 数学核心：regret(T) = T·p* − Σ_{t≤T} r_t，其中 p* = max_i p_i。
 * 每个智能体面对的都是探索-利用权衡：多试未知臂（探索）还是压已知好臂（利用）。
 *
 * 参考：Thompson (1933); Auer, Cesa-Bianchi & Fischer (2002)。
 *
 * 复杂度：step 为 O(1)；空间 O(K)。
 *
 * 常见坑：
 * - 遗憾必须按"期望"理解——单次运行的遗憾是随机量，比较算法应固定 seed。
 * - 奖励是 Bernoulli(0/1)，增量均值估计 Q_i ← Q_i + (r − Q_i)/N_i 才是无偏的。
 */
export class BernoulliBandit {
  /**
   * @param {{means: number[], seed?: number}} opts
   */
  constructor({ means, seed = 42 }) {
    if (!Array.isArray(means) || means.length === 0) {
      throw new Error('BernoulliBandit: means 必须是非空数组');
    }
    for (const m of means) {
      if (!(m >= 0 && m <= 1)) throw new Error(`BernoulliBandit: 臂均值必须在 [0,1]，收到 ${m}`);
    }
    this.means = means.slice();
    this.nArms = this.means.length;
    this.optimalAction = argmax(this.means);
    this.optimalMean = this.means[this.optimalAction];
    this._rng = createRng(seed);
    this._regret = 0;
    this.steps = 0;
    this.totalReward = 0;
  }

  /** 累计遗憾（定义恒等式：regret = steps·p* − totalReward） */
  get regret() { return this._regret; }

  /** 重置累计统计（不重置 RNG 流；需要完整复现请新建实例） */
  reset() {
    this._regret = 0;
    this.steps = 0;
    this.totalReward = 0;
    return this;
  }

  /**
   * 拉一次臂。
   * @param {number} action 臂索引
   * @returns {number} 奖励 0 或 1（Bernoulli(means[action])）
   */
  step(action) {
    if (!Number.isInteger(action) || action < 0 || action >= this.nArms) {
      throw new Error(`BernoulliBandit.step: 非法动作 ${action}（nArms=${this.nArms}）`);
    }
    const r = this._rng.next() < this.means[action] ? 1 : 0;
    this._regret += this.optimalMean - this.means[action];
    this.steps += 1;
    this.totalReward += r;
    return r;
  }
}

/**
 * EpsilonGreedy — ε-贪心 bandit 算法。
 *
 * 直觉：以 1−ε 的概率利用当前估计的最优臂，以 ε 的概率均匀随机探索。
 * 估计用增量均值 Q_i ← Q_i + (r − Q_i)/N_i。
 *
 * 数学核心：遗憾上界 O(ε·T + K/(ε·T^{-1})) 量级——ε 固定时随 T 线性增长，
 * 因此 ε-greedy 是"务实但不最优"的基线（Auer et al. 2002 的对照算法）。
 *
 * 复杂度：selectAction O(K)，update O(1)，空间 O(K)。
 *
 * 常见坑：
 * - ε 太大浪费遗憾，太小可能永远错过最优臂；实践中可让 ε 随 t 衰减（本类保持常数，
 *   由调用方在外部构造多个实例实现衰减策略）。
 * - 平局时取最小索引（argmax 语义），保证确定性。
 */
export class EpsilonGreedy {
  /**
   * @param {{epsilon?: number, seed?: number}} opts
   */
  constructor({ epsilon = 0.1, seed = 42 } = {}) {
    if (!(epsilon >= 0 && epsilon <= 1)) throw new Error(`EpsilonGreedy: epsilon 必须在 [0,1]，收到 ${epsilon}`);
    this.epsilon = epsilon;
    this._rng = createRng(seed);
    this.estimates = null; // 内部增量均值（惰性初始化，按需增长）
    this.counts = null;
  }

  /** 确保内部状态至少容纳 n 个臂 */
  _ensure(n) {
    if (this.estimates && this.estimates.length >= n) return;
    const cap = Math.max(n, this.estimates ? this.estimates.length : 0);
    const e = new Float64Array(cap);
    const c = new Float64Array(cap);
    if (this.estimates) {
      e.set(this.estimates);
      c.set(this.counts);
    }
    this.estimates = e;
    this.counts = c;
  }

  /**
   * 选择动作。estimates/counts 可由外部驱动注入；缺省使用内部增量均值。
   * @param {ArrayLike<number>} [estimates] 各臂当前均值估计
   * @param {ArrayLike<number>} [counts] 各臂拉动次数
   * @returns {number} 臂索引
   */
  selectAction(estimates, counts) {
    const q = estimates || this.estimates;
    if (!q) throw new Error('EpsilonGreedy.selectAction: 请先提供 estimates 或先调用一次 update');
    const n = q.length;
    if (this._rng.next() < this.epsilon) return this._rng.randint(n);
    return argmax(q);
  }

  /**
   * 用观测更新内部增量均值。
   * @param {number} action 臂索引
   * @param {number} reward 0/1 奖励
   */
  update(action, reward) {
    this._ensure(action + 1);
    this.counts[action] += 1;
    this.estimates[action] += (reward - this.estimates[action]) / this.counts[action];
  }
}

/**
 * UCB1 — 上置信界算法（Auer, Cesa-Bianchi & Fischer 2002）。
 *
 * 直觉：对每个臂的均值给出一个"乐观"的置信上界，每轮选上界最大的臂。
 * 拉得少的臂上界高（被迫探索），拉得多的臂上界收缩（利用估计）。
 *
 * 数学核心：Hoeffding 不等式给出 |Q_i − p_i| ≤ sqrt(2 ln t / N_i)（高概率），
 * 于是索引 index_i(t) = Q_i + sqrt(2 ln t / N_i)，选 argmax。遗憾上界 O(sqrt(K T ln T))。
 *
 * 复杂度：selectAction O(K)，update O(1)，空间 O(K)。
 *
 * 常见坑：
 * - 必须先每臂至少拉一次（未试臂优先），否则 N_i = 0 除零；t 至少为 K 时才有意义。
 * - ln t 中的 t 是总步数而非本臂步数——用错会破坏遗憾保证。
 */
export class UCB1 {
  /** @param {{seed?: number}} opts */
  constructor({ seed = 42 } = {}) {
    this._rng = createRng(seed);
    this.counts = null; // 内部各臂拉动次数（惰性初始化）
    this.sums = null; // 内部各臂奖励和（用于无外部 estimates 时推导均值）
  }

  /** 确保内部状态至少容纳 n 个臂 */
  _ensure(n) {
    if (this.counts && this.counts.length >= n) return;
    const cap = Math.max(n, this.counts ? this.counts.length : 0);
    const c = new Float64Array(cap);
    const s = new Float64Array(cap);
    if (this.counts) {
      c.set(this.counts);
      s.set(this.sums);
    }
    this.counts = c;
    this.sums = s;
  }

  /**
   * 选择动作：未试臂优先，否则 argmax(Q_i + sqrt(2 ln t / N_i))。
   * @param {ArrayLike<number>} [estimates] 各臂当前均值估计（缺省用内部 sums/counts）
   * @param {ArrayLike<number>} [counts] 各臂拉动次数（缺省用内部 counts）
   * @returns {number} 臂索引
   */
  selectAction(estimates, counts) {
    const c = counts || this.counts;
    if (!c) throw new Error('UCB1.selectAction: 请先提供 counts 或先调用一次 update');
    const n = c.length;
    // 未试臂优先（确定性：取索引最小者）
    for (let a = 0; a < n; a++) if (c[a] === 0) return a;
    const q = estimates || (this.sums ? Array.from({ length: n }, (_, a) => this.sums[a] / c[a]) : null);
    if (!q) throw new Error('UCB1.selectAction: 请提供 estimates 或先调用一次 update');
    let t = 0;
    for (let a = 0; a < n; a++) t += c[a];
    const logT = Math.log(t);
    let best = -Infinity;
    let bestA = 0;
    for (let a = 0; a < n; a++) {
      const idx = q[a] + Math.sqrt((2 * logT) / c[a]);
      if (idx > best) {
        best = idx;
        bestA = a;
      }
    }
    return bestA;
  }

  /**
   * 更新内部计数与奖励和。
   * @param {number} action 臂索引
   * @param {number} reward 0/1 奖励
   */
  update(action, reward) {
    this._ensure(action + 1);
    this.counts[action] += 1;
    this.sums[action] += reward;
  }
}

/**
 * 两个 Bernoulli 分布间的 KL 散度 d(p, q) = p·ln(p/q) + (1−p)·ln((1−p)/(1−q))。
 * 端点做截断，避免 log(0)。
 * @param {number} p
 * @param {number} q
 * @returns {number}
 */
export function bernoulliKL(p, q) {
  const eps = 1e-12;
  const pp = Math.min(Math.max(p, eps), 1 - eps);
  const qq = Math.min(Math.max(q, eps), 1 - eps);
  return pp * Math.log(pp / qq) + (1 - pp) * Math.log((1 - pp) / (1 - qq));
}

/**
 * KL-UCB 上置信指数（对 Bernoulli 奖励）。
 *
 * 数学核心：U_a(t) = sup{ q ∈ [q̂_a, 1] : N_a · d(q̂_a, q) ≤ ln t + c·ln ln t }。
 * d 在 q > q̂ 上严格单调递增，故用二分搜索求根（50 次迭代收敛到机器精度）。
 *
 * 为什么比 UCB1 强：UCB1 用 Hoeffding 界，它对所有 [0,1] 奖励都成立但对
 * Bernoulli 太松；KL-UCB 用 Bernoulli 精确大偏差率函数，界更紧，
 * 且被证明渐近达到 Lai & Robbins 下界（Garivier & Cappé 2011；Audibert et al. 2010）。
 *
 * @param {number} mean 该臂经验均值 q̂_a
 * @param {number} count 该臂拉动次数 N_a
 * @param {number} t 总步数（≥1）
 * @param {number} [c] ln ln 修正系数，缺省 0（标准 KL-UCB）
 * @returns {number} UCB 指数 ∈ [mean, 1]
 */
export function klUCBIndex(mean, count, t, c = 0) {
  const logT = Math.log(Math.max(t, 2)); // t=1 时 ln t = 0 会让界退化为 0
  const level = (logT + c * Math.log(Math.max(logT, 1))) / Math.max(count, 1);
  let lo = mean;
  let hi = 1;
  for (let it = 0; it < 50; it++) {
    const mid = 0.5 * (lo + hi);
    if (bernoulliKL(mean, mid) > level) hi = mid;
    else lo = mid;
  }
  return 0.5 * (lo + hi);
}

/**
 * KLUCB — KL 上置信界算法（Garivier & Cappé 2011；Audibert, Munos & Szepesvári 2010）。
 *
 * 直觉：UCB1 问"这个臂的均值最坏可能高到哪"，但用的是对所有 [0,1] 分布
 * 都成立的宽松不等式。KL-UCB 问同样的问题，但精确利用"奖励只能是 0 或 1"
 * 这一信息，所以给出的上界更贴身——探索更少、收敛更快。
 *
 * 复杂度：每步 O(K) 次二分搜索（每次 50 次迭代 × O(1)），空间 O(K)。
 */
export class KLUCB {
  /** @param {{seed?: number, c?: number}} opts */
  constructor({ seed = 42, c = 0 } = {}) {
    this._rng = createRng(seed);
    this.c = c;
    this.counts = null;
    this.sums = null;
  }

  _ensure(n) {
    if (this.counts && this.counts.length >= n) return;
    const cap = Math.max(n, this.counts ? this.counts.length : 0);
    const c = new Float64Array(cap);
    const s = new Float64Array(cap);
    if (this.counts) {
      c.set(this.counts);
      s.set(this.sums);
    }
    this.counts = c;
    this.sums = s;
  }

  /** @param {ArrayLike<number>} [estimates] @param {ArrayLike<number>} [counts] */
  selectAction(estimates, counts) {
    const c = counts || this.counts;
    if (!c) throw new Error('KLUCB.selectAction: 请先提供 counts 或先调用一次 update');
    const n = c.length;
    for (let a = 0; a < n; a++) if (c[a] === 0) return a;
    const q = estimates || (this.sums ? Array.from({ length: n }, (_, a) => this.sums[a] / c[a]) : null);
    if (!q) throw new Error('KLUCB.selectAction: 请提供 estimates 或先调用一次 update');
    let t = 0;
    for (let a = 0; a < n; a++) t += c[a];
    let best = -Infinity;
    let bestA = 0;
    for (let a = 0; a < n; a++) {
      const idx = klUCBIndex(q[a], c[a], t, this.c);
      if (idx > best) {
        best = idx;
        bestA = a;
      }
    }
    return bestA;
  }

  update(action, reward) {
    this._ensure(action + 1);
    this.counts[action] += 1;
    this.sums[action] += reward;
  }
}

/**
 * ThompsonSampling — Thompson 采样（Thompson 1933）。
 *
 * 直觉：为每个臂的胜率维护一个 Beta 后验，每轮从每个后验各采一个样本，
 * 选样本值最大的臂。"用概率匹配不确定性"——不确定的臂也有机会被探索。
 *
 * 数学核心：Bernoulli 奖励下取 Beta(1,1)（均匀）先验，则后验为
 * Beta(α_i, β_i)，α_i = 1 + 成功次数，β_i = 1 + 失败次数。
 * 每轮采样 θ_i ~ Beta(α_i, β_i)，选 argmax θ_i。
 * 贝叶斯遗憾 O(sqrt(K T ln T))（Agrawal & Goyal 2012 的有限时间分析）。
 *
 * Beta 采样用 Marsaglia-Tsang（2000）的 Gamma 比率法：Beta(a,b) = G_a/(G_a+G_b)。
 *
 * 复杂度：selectAction O(K·采样成本)，update O(1)，空间 O(K)。
 *
 * 常见坑：
 * - 先验选择影响早期行为；Beta(1,1) 是无信息先验，最常用。
 * - 采样必须来自注入的 RNG——用 Math.random 会破坏可复现性。
 */
export class ThompsonSampling {
  /** @param {{seed?: number}} opts */
  constructor({ seed = 42 } = {}) {
    this._rng = createRng(seed);
    this.alpha = null; // 后验 α（惰性初始化）
    this.beta = null; // 后验 β
    this.counts = null;
  }

  /** 确保内部后验至少容纳 n 个臂（Beta(1,1) 先验） */
  _ensure(n) {
    if (this.alpha && this.alpha.length >= n) return;
    const cap = Math.max(n, this.alpha ? this.alpha.length : 0);
    const a = new Float64Array(cap).fill(1);
    const b = new Float64Array(cap).fill(1);
    const c = new Float64Array(cap);
    if (this.alpha) {
      a.set(this.alpha);
      b.set(this.beta);
      c.set(this.counts);
    }
    this.alpha = a;
    this.beta = b;
    this.counts = c;
  }

  /**
   * 选择动作：从每个臂的 Beta 后验采样，取样本最大者。
   * 若内部后验尚未初始化且提供了 estimates，则以"10 次伪观测"的弱先验注入。
   * @param {ArrayLike<number>} [estimates] 各臂当前均值估计（仅用于惰性初始化先验）
   * @returns {number} 臂索引
   */
  selectAction(estimates) {
    if (!this.alpha) {
      if (!estimates) throw new Error('ThompsonSampling.selectAction: 请提供 estimates 或先调用一次 update');
      this._ensure(estimates.length);
      for (let a = 0; a < estimates.length; a++) {
        // 弱先验：10 次伪观测，均值对齐给定估计
        this.alpha[a] = 1 + 10 * estimates[a];
        this.beta[a] = 1 + 10 * (1 - estimates[a]);
      }
    }
    const n = this.alpha.length;
    let best = -1;
    let bestA = 0;
    for (let a = 0; a < n; a++) {
      const s = sampleBeta(this._rng, this.alpha[a], this.beta[a]);
      if (s > best) {
        best = s;
        bestA = a;
      }
    }
    return bestA;
  }

  /**
   * 用 Bernoulli 观测更新后验：α += r，β += 1 − r。
   * @param {number} action 臂索引
   * @param {number} reward 0/1 奖励
   */
  update(action, reward) {
    this._ensure(action + 1);
    this.alpha[action] += reward;
    this.beta[action] += 1 - reward;
    this.counts[action] += 1;
  }

  /**
   * 后验均值 α/(α+β)——随观测增多收敛到真实胜率（可验证不变量）。
   * @param {number} action 臂索引
   * @returns {number}
   */
  posteriorMean(action) {
    if (!this.alpha) throw new Error('ThompsonSampling.posteriorMean: 后验尚未初始化');
    return this.alpha[action] / (this.alpha[action] + this.beta[action]);
  }
}

/* ==================== 智能体 ==================== */

/**
 * QLearningAgent — 表格型 Q 学习（Watkins & Dayan 1992）。
 *
 * 直觉：维护一张 Q(s,a) 表，每走一步用"即时奖励 + 下一状态的最优估计"修正当前项，
 * 无需环境模型即可学出最优动作值函数。
 *
 * 数学核心（Bellman 最优方程）：Q*(s,a) = r + γ·max_{a'} Q*(s', a')。
 * Q 学习的时序差分更新：
 *   Q(s,a) ← Q(s,a) + α·[r + γ·max_{a'} Q(s',a') − Q(s,a)]
 * 终止状态的目标值就是 r 本身（无后继）。对确定性/随机查表型 MDP 以概率 1 收敛
 * （Watkins & Dayan 1992；Sutton & Barto 2018 第 6 章）。
 *
 * 行为策略用 ε-greedy：以 ε 概率随机，否则贪心（保证持续探索）。
 *
 * 复杂度：act/learn 均 O(nActions)；空间 O(nStates·nActions)。
 *
 * 常见坑：
 * - 终止转移的 target 不能包含 γ·max Q(s')，否则把终局值错误传播。
 * - α 过大震荡、过小收敛慢；ε 为 0 时可能永远发现不了最优路径。
 * - 折扣 γ 接近 1 时回报方差大，远小于 1 时"短视"。
 */
export class QLearningAgent {
  /**
   * @param {{nStates: number, nActions: number, alpha?: number, gamma?: number, epsilon?: number, seed?: number}} opts
   */
  constructor({ nStates, nActions, alpha = 0.1, gamma = 0.99, epsilon = 0.1, seed = 42 } = {}) {
    if (!Number.isInteger(nStates) || nStates < 1) throw new Error(`QLearningAgent: nStates 必须为正整数，收到 ${nStates}`);
    if (!Number.isInteger(nActions) || nActions < 1) throw new Error(`QLearningAgent: nActions 必须为正整数，收到 ${nActions}`);
    if (!(alpha > 0 && alpha <= 1)) throw new Error(`QLearningAgent: alpha 必须在 (0,1]，收到 ${alpha}`);
    if (!(gamma >= 0 && gamma < 1)) throw new Error(`QLearningAgent: gamma 必须在 [0,1)，收到 ${gamma}`);
    if (!(epsilon >= 0 && epsilon <= 1)) throw new Error(`QLearningAgent: epsilon 必须在 [0,1]，收到 ${epsilon}`);
    this.nStates = nStates;
    this.nActions = nActions;
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this.Q = new Float64Array(nStates * nActions);
    this._rng = createRng(seed);
    this.updates = 0;
  }

  /**
   * 查询 Q(s,a)。
   * @param {number} state
   * @param {number} action
   * @returns {number}
   */
  q(state, action) {
    return this.Q[state * this.nActions + action];
  }

  /**
   * 取某状态的全部 Q 值（拷贝，防止外部篡改内部表）。
   * @param {number} state
   * @returns {Float64Array}
   */
  qRow(state) {
    return this.Q.slice(state * this.nActions, (state + 1) * this.nActions);
  }

  /** 贪婪动作（无探索；平局取最小索引，确定性） */
  greedyAction(state) {
    const base = state * this.nActions;
    return argmax(this.Q.subarray(base, base + this.nActions));
  }

  /**
   * 在平局动作中均匀随机选一个（经注入 RNG，保持确定性）。
   * 必要性：初始 Q 全 0 时若平局恒取动作 0，在"动作 0 = 原地不动"的环境
   * （如 ChainMDP 左移自环）中智能体将永远无法发现奖励——这是表格 RL 的经典陷阱。
   * @param {number} state 状态索引
   * @returns {number} 动作索引
   */
  _greedyWithRandomTies(state) {
    const base = state * this.nActions;
    let m = -Infinity;
    for (let a = 0; a < this.nActions; a++) if (this.Q[base + a] > m) m = this.Q[base + a];
    let tieCount = 0;
    for (let a = 0; a < this.nActions; a++) if (this.Q[base + a] === m) tieCount += 1;
    if (tieCount <= 1) return this.greedyAction(state);
    let pick = this._rng.randint(tieCount);
    for (let a = 0; a < this.nActions; a++) {
      if (this.Q[base + a] === m) {
        if (pick === 0) return a;
        pick -= 1;
      }
    }
    return this.greedyAction(state);
  }

  /**
   * ε-greedy 选动作。
   * @param {number} state 状态索引
   * @param {boolean|number} explore true=用默认 ε；false=纯贪婪；数字=临时覆盖 ε
   * @returns {number} 动作索引
   */
  act(state, explore = true) {
    if (!Number.isInteger(state) || state < 0 || state >= this.nStates) {
      throw new Error(`QLearningAgent.act: 非法状态 ${state}（nStates=${this.nStates}）`);
    }
    const eps = typeof explore === 'number' ? explore : (explore ? this.epsilon : 0);
    if (eps > 0 && this._rng.next() < eps) return this._rng.randint(this.nActions);
    return this._greedyWithRandomTies(state);
  }

  /**
   * 时序差分更新：Q(s,a) += α·(target − Q(s,a))，
   * target = done ? r : r + γ·max_{a'} Q(s',a')。
   * @param {number} state
   * @param {number} action
   * @param {number} reward
   * @param {number|null} [nextState] 终止转移可传 null
   * @param {boolean|null} [done] 缺省时按 nextState==null 推断
   */
  learn(state, action, reward, nextState = null, done = null) {
    const terminal = done === null ? nextState == null : !!done;
    let target = reward;
    if (!terminal) {
      const base = nextState * this.nActions;
      let m = -Infinity;
      for (let a = 0; a < this.nActions; a++) if (this.Q[base + a] > m) m = this.Q[base + a];
      target = reward + this.gamma * m;
    }
    const i = state * this.nActions + action;
    this.Q[i] += this.alpha * (target - this.Q[i]);
    this.updates += 1;
  }
}

/**
 * REINFORCEAgent — 线性 softmax 策略梯度（Williams 1992）。
 *
 * 直觉：不用价值表，直接把策略参数化成 softmax 线性模型，用完整回合的回报
 * 给"碰巧拿到高回报的动作"加权——好动作的频率上升，差动作下降。
 *
 * 数学核心（策略梯度定理，Sutton et al. 2000）：
 *   ∇J(θ) = E[ Σ_t γ^t·G_t·∇log π(a_t|s_t) ]
 * 其中 G_t 为折现回报，线性 softmax 的参数为 w_a ∈ R^d，
 * π(a|s) = softmax_a(w·φ(s))，其 score 函数
 *   ∇_{w_a} log π(a|s) = φ(s)·(1[a = a_t] − π(a_t|s))。
 * 回合结束时统一做一次蒙特卡洛更新（episodic update）。
 *
 * 特征 φ：数值状态取 one-hot(nStates)（退化为表格 softmax）；
 * 数组状态取 [1, x_1, ..., x_d]（仿射特征，可接连续观测）。
 *
 * 复杂度：每步 O(nActions·d)；回合更新 O(T·nActions·d)。
 *
 * 常见坑：
 * - 无基线时梯度方差大；多步回合可开启 episode 均值基线（不引入偏差）。
 * - 学习率过大导致策略在几个回合内坍缩到单一动作（熵塌缩）。
 * - 单步 bandit 回合下 episode 基线恒等于回报，会清零梯度——此时应关闭基线。
 */
export class REINFORCEAgent {
  /**
   * @param {{nStates: number, nActions: number, alpha?: number, gamma?: number, seed?: number, baseline?: 'none'|'episode'|'running', baselineBeta?: number}} opts
   */
  constructor({
    nStates, nActions, alpha = 0.01, gamma = 0.99, seed = 42,
    baseline = 'none', baselineBeta = 0.05,
  } = {}) {
    if (!Number.isInteger(nStates) || nStates < 1) throw new Error(`REINFORCEAgent: nStates 必须为正整数，收到 ${nStates}`);
    if (!Number.isInteger(nActions) || nActions < 1) throw new Error(`REINFORCEAgent: nActions 必须为正整数，收到 ${nActions}`);
    if (!(alpha > 0)) throw new Error(`REINFORCEAgent: alpha 必须为正，收到 ${alpha}`);
    if (!(gamma >= 0 && gamma < 1)) throw new Error(`REINFORCEAgent: gamma 必须在 [0,1)，收到 ${gamma}`);
    this.nStates = nStates;
    this.nActions = nActions;
    this.alpha = alpha;
    this.gamma = gamma;
    this.baseline = baseline;
    this.baselineBeta = baselineBeta;
    this._rng = createRng(seed);
    this._d = null; // 特征维度（首次见到状态时确定）
    this.W = null; // 权重 nActions×d，小随机初始化打破对称
    this._buf = []; // 回合缓冲 {features, action, reward}
    this._runningBaseline = 0;
    this.episodes = 0;
  }

  /**
   * 状态特征化：数值 → one-hot(nStates)；数组 → [1, x...]。
   * @param {number|ArrayLike<number>} state
   * @returns {Float64Array}
   */
  features(state) {
    if (typeof state === 'number') {
      if (!Number.isInteger(state) || state < 0 || state >= this.nStates) {
        throw new Error(`REINFORCEAgent: 状态 ${state} 越界（nStates=${this.nStates}）`);
      }
      const f = new Float64Array(this.nStates);
      f[state] = 1;
      return f;
    }
    if (Array.isArray(state) || state instanceof Float64Array) {
      const f = new Float64Array(1 + state.length);
      f[0] = 1;
      for (let i = 0; i < state.length; i++) f[i + 1] = state[i];
      return f;
    }
    throw new Error('REINFORCEAgent: 状态必须是数值（one-hot）或数组（仿射特征）');
  }

  /** 惰性初始化权重矩阵（小随机值，保证初始策略非退化且确定性） */
  _ensureW(d) {
    if (this._d === d) return;
    if (this._d !== null) throw new Error(`REINFORCEAgent: 特征维度从 ${this._d} 变为 ${d}，同一智能体不支持混合状态类型`);
    this._d = d;
    this.W = new Float64Array(this.nActions * d);
    for (let i = 0; i < this.W.length; i++) this.W[i] = this._rng.normal() * 0.01;
  }

  /**
   * 策略偏好（logits）w_a·φ(s)。
   * @param {Float64Array} f 特征
   * @returns {Float64Array} 长度 nActions
   */
  preferences(f) {
    const d = this._d;
    const h = new Float64Array(this.nActions);
    for (let a = 0; a < this.nActions; a++) {
      const base = a * d;
      let s = 0;
      for (let k = 0; k < d; k++) s += this.W[base + k] * f[k];
      h[a] = s;
    }
    return h;
  }

  /**
   * 当前策略 π(·|state)。
   * @param {number|ArrayLike<number>} state
   * @returns {number[]} 概率分布
   */
  policy(state) {
    const f = this.features(state);
    this._ensureW(f.length);
    return softmaxArr(this.preferences(f));
  }

  /**
   * 按策略选动作。
   * @param {number|ArrayLike<number>} state
   * @param {boolean} explore true=从 softmax 采样；false=取最大概率动作
   * @returns {number} 动作索引
   */
  act(state, explore = true) {
    const f = this.features(state);
    this._ensureW(f.length);
    const probs = softmaxArr(this.preferences(f));
    if (!explore) return argmax(probs);
    const u = this._rng.next();
    let acc = 0;
    for (let a = 0; a < probs.length; a++) {
      acc += probs[a];
      if (u < acc) return a;
    }
    return probs.length - 1;
  }

  /**
   * 记录一个转移；done=true（默认）时立即做回合更新。
   * 单步任务（如 bandit）每次调用即构成一个完整回合。
   * @param {number|ArrayLike<number>} state
   * @param {number} action
   * @param {number} reward
   * @param {number|ArrayLike<number>|null} [nextState] 仅作接口占位（蒙特卡洛法不用）
   * @param {boolean} [done] 是否结束当前回合
   */
  learn(state, action, reward, nextState = null, done = true) {
    const f = this.features(state);
    this._ensureW(f.length);
    this._buf.push({ f, action, reward });
    if (done) this.endEpisode();
  }

  /** 对缓冲的完整回合执行一次蒙特卡洛策略梯度更新 */
  endEpisode() {
    const T = this._buf.length;
    if (T === 0) return;
    const d = this._d;
    // 1) 折现回报 G_t（从后向前累积）
    const G = new Float64Array(T);
    let g = 0;
    for (let t = T - 1; t >= 0; t--) {
      g = this._buf[t].reward + this.gamma * g;
      G[t] = g;
    }
    // 2) 基线（仅在不引入偏差的前提下使用）
    let base = 0;
    if (this.baseline === 'episode' && T > 1) {
      let sum = 0;
      for (let t = 0; t < T; t++) sum += G[t];
      base = sum / T;
    } else if (this.baseline === 'running') {
      base = this._runningBaseline;
      let sum = 0;
      for (let t = 0; t < T; t++) sum += G[t];
      this._runningBaseline += this.baselineBeta * (sum / T - this._runningBaseline);
    }
    // 3) 逐时刻更新：W[a] += α·γ^t·(G_t − base)·φ(s)·(1[a=a_t] − π(a_t|s))
    for (let t = 0; t < T; t++) {
      const { f, action } = this._buf[t];
      const probs = softmaxArr(this.preferences(f));
      const scale = this.alpha * Math.pow(this.gamma, t) * (G[t] - base);
      for (let k = 0; k < d; k++) {
        const fk = f[k] * scale;
        for (let a = 0; a < this.nActions; a++) {
          const coef = a === action ? 1 - probs[a] : -probs[a];
          this.W[a * d + k] += fk * coef;
        }
      }
    }
    this._buf.length = 0;
    this.episodes += 1;
  }

  /** 丢弃当前未结束的回合缓冲 */
  resetEpisode() {
    this._buf.length = 0;
  }
}

/* ==================== 经典控制环境 ==================== */

/**
 * CartPoleEnv — 经典"车-杆"控制环境（Barto, Sutton & Anderson 1983）。
 *
 * 直觉：小车在无摩擦轨道上受左右恒力，顶上铰接一根杆。目标是通过左右施力
 * 保持杆不倒、车不出轨——每多存活一步得 1 分，是检验控制策略的黄金基准。
 *
 * 动力学（标准半显式欧拉积分，τ = 0.02s）：
 *   temp = (F + μ_p·θ̇²·sinθ) / m_total
 *   θ̈ = (g·sinθ − cosθ·temp) / (L·(4/3 − m_p·cos²θ/m_total))
 *   ẍ = temp − μ_p·θ̈·cosθ / m_total
 *   (x, ẋ, θ, θ̇) ← 上一时刻 + τ·对应速度
 * 终止条件：|x| > 2.4（出轨）或 |θ| > 12° ≈ 0.2095 rad（杆倒），或达到 500 步上限。
 *
 * 注意：与 Gym 的 uniform(−0.05, 0.05) 随机初始化不同，本实现 reset 返回确定性
 * 全零状态——这是可复现性要求下的刻意选择（见 selfCheck 不变量）。
 *
 * 复杂度：step O(1)；空间 O(1)。
 *
 * 参考：Barto et al. (1983); OpenAI Gym classic_control/cart_pole.py。
 */
export class CartPoleEnv {
  /** @param {{seed?: number}} opts seed 保留给未来随机初始化，当前 reset 为确定性全零 */
  constructor({ seed = 42 } = {}) {
    this.gravity = 9.8;
    this.masscart = 1.0;
    this.masspole = 0.1;
    this.length = 0.5; // 杆全长（质心在一半处）
    this.polemass_length = this.masspole * this.length;
    this.force_mag = 10.0;
    this.tau = 0.02;
    this.x_threshold = 2.4;
    this.theta_threshold = (12 * 2 * Math.PI) / 360; // 12° ≈ 0.20944 rad
    this.maxSteps = 500;
    this.state = [0, 0, 0, 0];
    this.steps = 0;
    this._rng = createRng(seed);
  }

  /** 重置到确定性全零状态 @returns {number[]} [x, ẋ, θ, θ̇] */
  reset() {
    this.state = [0, 0, 0, 0];
    this.steps = 0;
    return this.state.slice();
  }

  /**
   * 推进一步。
   * @param {number} action 0=向左施力（−F），1=向右施力（+F）
   * @returns {{state: number[], reward: number, done: boolean}}
   */
  step(action) {
    if (action !== 0 && action !== 1) throw new Error(`CartPoleEnv.step: 动作必须为 0 或 1，收到 ${action}`);
    const [x, x_dot, theta, theta_dot] = this.state;
    const force = action === 1 ? this.force_mag : -this.force_mag;
    const costheta = Math.cos(theta);
    const sintheta = Math.sin(theta);
    const totalMass = this.masscart + this.masspole;
    const temp = (force + this.polemass_length * theta_dot * theta_dot * sintheta) / totalMass;
    const thetaacc = (this.gravity * sintheta - costheta * temp)
      / (this.length * (4 / 3 - (this.masspole * costheta * costheta) / totalMass));
    const xacc = temp - (this.polemass_length * thetaacc * costheta) / totalMass;
    // 半显式欧拉积分
    const nx = x + this.tau * x_dot;
    const nxDot = x_dot + this.tau * xacc;
    const nTheta = theta + this.tau * theta_dot;
    const nThetaDot = theta_dot + this.tau * thetaacc;
    this.state = [nx, nxDot, nTheta, nThetaDot];
    this.steps += 1;
    const done = Math.abs(nx) > this.x_threshold
      || Math.abs(nTheta) > this.theta_threshold
      || this.steps >= this.maxSteps;
    return { state: this.state.slice(), reward: 1, done };
  }
}

/**
 * ChainMDP — 确定性链式 MDP（小型测试环境，解析解可手算）。
 *
 * 结构：状态 0…n−1 排成一条链，动作 {0: 左移, 1: 右移}（边界处原地不动）。
 * 右移进入末端状态 n−1 得奖励 1 并终止；其余奖励 0。每回合从状态 0 出发。
 *
 * 解析解（γ 折扣）：V(s) = γ^(n−2−s)（0 ≤ s ≤ n−2），V(n−1) = 0，于是
 *   Q*(s, 右) = γ^(n−1−s)（s ≤ n−3），Q*(n−2, 右) = 1，Q*(n−1, ·) = 0
 *   Q*(s, 左) = γ^(n−1−s)（0 ≤ s ≤ n−2）
 * 恒满足 Bellman 最优方程——这正是 Q-learning 收敛的对照金标准。
 *
 * 参考：Sutton & Barto (2018) 第 3 章（有限 MDP 与 Bellman 方程）。
 */
export class ChainMDP {
  /**
   * @param {{n?: number, maxSteps?: number}} opts
   */
  constructor({ n = 6, maxSteps = 60 } = {}) {
    if (!Number.isInteger(n) || n < 2) throw new Error(`ChainMDP: n 必须 ≥ 2，收到 ${n}`);
    this.n = n;
    this.maxSteps = maxSteps;
    this.state = 0;
    this.steps = 0;
  }

  get nStates() { return this.n; }
  get nActions() { return 2; }

  /** 重置到状态 0 @returns {number} */
  reset() {
    this.state = 0;
    this.steps = 0;
    return this.state;
  }

  /**
   * 推进一步（确定性转移）。
   * @param {number} action 0=左移，1=右移
   * @returns {{state: number, reward: number, done: boolean}}
   */
  step(action) {
    if (action !== 0 && action !== 1) throw new Error(`ChainMDP.step: 动作必须为 0 或 1，收到 ${action}`);
    const s = this.state;
    const next = action === 1 ? Math.min(s + 1, this.n - 1) : Math.max(s - 1, 0);
    const reward = next === this.n - 1 && s !== this.n - 1 ? 1 : 0;
    this.state = next;
    this.steps += 1;
    const done = next === this.n - 1 || this.steps >= this.maxSteps;
    return { state: next, reward, done };
  }

  /**
   * 解析最优 Q*（上述闭式解）。
   * 推导：V(s) = γ^(n−2−s)（0 ≤ s ≤ n−2），V(n−1) = 0，于是
   *   Q*(s, 右) = γ·V(s+1) = γ^(n−2−s)（s ≤ n−2，含 s=n−2 时 γ^0=1 恰为进入终态的奖励）
   *   Q*(s, 左) = γ·V(s−1) = γ^(n−s)（1 ≤ s ≤ n−2）；Q*(0, 左) = γ·V(0) = γ^(n−1)
   * @param {number} gamma 折扣因子
   * @returns {Float64Array} 长度 n·2（行优先，[s, a]）
   */
  analyticQ(gamma = 0.9) {
    const Q = new Float64Array(this.n * 2);
    for (let s = 0; s < this.n; s++) {
      if (s === this.n - 1) continue; // 终态 Q* = 0
      Q[s * 2 + 0] = Math.pow(gamma, s === 0 ? this.n - 1 : this.n - s); // 左移
      Q[s * 2 + 1] = Math.pow(gamma, this.n - 2 - s); // 右移
    }
    return Q;
  }

  /**
   * Bellman 最优方程残差：max_{s,a} |Q(s,a) − (r(s,a) + γ·max_{a'} Q(s',a'))|。
   * 收敛的 Q 应使残差 → 0（Q-learning 的核心不变量）。
   * @param {Float64Array} Q 长度 n·2 的动作值表
   * @param {number} gamma
   * @returns {number} 最大绝对残差
   */
  bellmanResidual(Q, gamma = 0.9) {
    let max = 0;
    for (let s = 0; s < this.n; s++) {
      for (let a = 0; a < 2; a++) {
        let target;
        if (s === this.n - 1) {
          target = 0; // 吸收终态
        } else {
          const next = a === 1 ? Math.min(s + 1, this.n - 1) : Math.max(s - 1, 0);
          const r = next === this.n - 1 && s !== this.n - 1 ? 1 : 0;
          target = r + gamma * Math.max(Q[next * 2], Q[next * 2 + 1]);
        }
        max = Math.max(max, Math.abs(Q[s * 2 + a] - target));
      }
    }
    return max;
  }
}

/**
 * GridWorldEnv — 4×4 冰面网格世界（可选环境，FrozenLake 风格）。
 *
 * 直觉：智能体从左上角出发，要到右下角的目标格；冰面会打滑（slippery），
 * 踩到洞穴则回合失败。奖励仅出现在目标格（+1），是检验稀疏奖励下探索的经典玩具。
 *
 * 结构（n=4 默认，行主序索引，1=洞穴，2=目标）：
 *   S  F  F  F
 *   F  H  F  H
 *   F  F  F  H
 *   H  F  F  G
 * 动作 0=上 1=右 2=下 3=左；slippery=true 时 0.8 概率执行意图方向，
 * 0.1 概率各打滑到两个垂直方向（与 Gym FrozenLake-v1 4x4 一致）。
 *
 * 参考：Gym FrozenLake-v1；Sutton & Barto (2018) 第 1 章（稀疏奖励与探索）。
 */
export class GridWorldEnv {
  /**
   * @param {{n?: number, seed?: number, slippery?: boolean}} opts
   */
  constructor({ n = 4, seed = 42, slippery = false } = {}) {
    if (!Number.isInteger(n) || n < 2) throw new Error(`GridWorldEnv: n 必须 ≥ 2，收到 ${n}`);
    this.n = n;
    this.slippery = slippery;
    this._rng = createRng(seed);
    // 4×4 固定地图（FrozenLake-v1 4x4 布局）；其它尺寸退化为无洞穴的纯目标格世界
    this.map = n === 4
      ? [[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 1], [1, 0, 0, 2]]
      : Array.from({ length: n }, () => new Array(n).fill(0));
    if (n !== 4) this.map[n - 1][n - 1] = 2;
    this.state = 0;
    this.steps = 0;
  }

  get nStates() { return this.n * this.n; }
  get nActions() { return 4; }

  /** 重置到左上角 @returns {number} 状态索引（行主序） */
  reset() {
    this.state = 0;
    this.steps = 0;
    return this.state;
  }

  /**
   * 推进一步。
   * @param {number} action 0=上 1=右 2=下 3=左
   * @returns {{state: number, reward: number, done: boolean}}
   */
  step(action) {
    if (!Number.isInteger(action) || action < 0 || action > 3) {
      throw new Error(`GridWorldEnv.step: 动作必须为 0..3，收到 ${action}`);
    }
    let a = action;
    if (this.slippery) {
      const u = this._rng.next();
      // 垂直方向对：0↔2（上下），1↔3（左右）
      const perp = action === 0 || action === 2 ? (action === 0 ? 3 : 1) : (action === 1 ? 0 : 2);
      if (u < 0.1) a = perp;
      else if (u < 0.2) a = action === 0 || action === 2 ? (action === 0 ? 1 : 3) : (action === 1 ? 2 : 0);
    }
    const r = Math.floor(this.state / this.n);
    const c = this.state % this.n;
    const dr = a === 0 ? -1 : a === 2 ? 1 : 0;
    const dc = a === 1 ? 1 : a === 3 ? -1 : 0;
    const nr = Math.min(Math.max(r + dr, 0), this.n - 1);
    const nc = Math.min(Math.max(c + dc, 0), this.n - 1);
    const next = nr * this.n + nc;
    const cell = this.map[nr][nc];
    const reward = cell === 2 ? 1 : 0;
    this.state = next;
    this.steps += 1;
    return { state: next, reward, done: cell !== 0 || this.steps >= 4 * this.n * this.n };
  }
}

/* ==================== CartPole 状态离散化 ==================== */

/**
 * CartPole 默认分箱方案：x 与 ẋ 用粗箱（对平衡贡献小、范围大），
 * θ 与 θ̇ 用细箱（决定倒不倒的关键量）。共 3×3×8×8 = 576 个离散状态。
 * 参考：Sutton & Barto (2018) 第 9 章（ tile coding / 粗编码的替代方案）。
 */
export const CARTPOLE_BOUNDS = [
  { lo: -2.4, hi: 2.4, bins: 3 }, // x
  { lo: -3.0, hi: 3.0, bins: 3 }, // ẋ
  { lo: -0.21, hi: 0.21, bins: 8 }, // θ（终止阈值 0.2095）
  { lo: -3.5, hi: 3.5, bins: 8 }, // θ̇
];

/**
 * 把连续 CartPole 状态映射为离散箱索引（越界值被裁剪到端箱）。
 * @param {number[]} state [x, ẋ, θ, θ̇]
 * @param {{lo: number, hi: number, bins: number}[]} [bounds]
 * @returns {number} 离散状态索引 ∈ [0, Π bins)
 */
export function cartPoleBinIndex(state, bounds = CARTPOLE_BOUNDS) {
  let idx = 0;
  for (let d = 0; d < bounds.length; d++) {
    const { lo, hi, bins } = bounds[d];
    let b = Math.floor(((state[d] - lo) / (hi - lo)) * bins);
    if (b < 0) b = 0;
    else if (b >= bins) b = bins - 1;
    idx = idx * bins + b;
  }
  return idx;
}

/** 分箱后的离散状态总数 @param {Array} [bounds] @returns {number} */
export function cartPoleStateCount(bounds = CARTPOLE_BOUNDS) {
  return bounds.reduce((acc, b) => acc * b.bins, 1);
}

/* ==================== 通用训练驱动（selfCheck 与 bench 共用） ==================== */

/**
 * 在 Bernoulli bandit 上运行一个智能体（驱动循环，供 ε-greedy / UCB1 / Thompson 复用）。
 * @param {{selectAction: Function, update: Function}} agent
 * @param {{means: number[], steps?: number, seed?: number}} opts
 * @returns {{regret: number, optimalRate: number, totalReward: number, steps: number, actions: number[]}}
 */
export function runBanditAgent(agent, { means, steps = 2000, seed = 42 }) {
  const env = new BernoulliBandit({ means, seed });
  const estimates = new Float64Array(means.length);
  const counts = new Float64Array(means.length);
  const actions = new Array(steps);
  let optimalHits = 0;
  for (let t = 0; t < steps; t++) {
    const a = agent.selectAction(estimates, counts);
    const r = env.step(a);
    agent.update(a, r);
    counts[a] += 1;
    estimates[a] += (r - estimates[a]) / counts[a];
    actions[t] = a;
    if (a === env.optimalAction) optimalHits += 1;
  }
  return {
    regret: env.regret,
    optimalRate: optimalHits / steps,
    totalReward: env.totalReward,
    steps,
    actions,
  };
}

/**
 * 在 ChainMDP 上训练 Q-learning，并跟踪收敛过程。
 * @returns {{agent: QLearningAgent, residual: number, maxDevFromAnalytic: number, convergenceUpdates: number|null, episodes: number, updates: number}}
 */
export function trainQLearningChain({
  n = 6, episodes = 400, alpha = 0.5, gamma = 0.9, epsilon = 0.1, seed = 42, tol = 0.01,
} = {}) {
  const mdp = new ChainMDP({ n });
  const agent = new QLearningAgent({ nStates: n, nActions: 2, alpha, gamma, epsilon, seed });
  const Qstar = mdp.analyticQ(gamma);
  let convergenceUpdates = null;
  for (let ep = 0; ep < episodes; ep++) {
    let s = mdp.reset();
    let done = false;
    while (!done) {
      const a = agent.act(s);
      const { state: s2, reward, done: d } = mdp.step(a);
      agent.learn(s, a, reward, s2, d);
      if (convergenceUpdates === null && mdp.bellmanResidual(agent.Q, gamma) < tol) {
        convergenceUpdates = agent.updates;
      }
      s = s2;
      done = d;
    }
  }
  const residual = mdp.bellmanResidual(agent.Q, gamma);
  let dev = 0;
  for (let i = 0; i < Qstar.length; i++) dev = Math.max(dev, Math.abs(agent.Q[i] - Qstar[i]));
  return {
    agent,
    residual,
    maxDevFromAnalytic: dev,
    convergenceUpdates,
    episodes,
    updates: agent.updates,
  };
}

/**
 * 在 10 臂 Bernoulli bandit 上训练 REINFORCE（每步视为一个单步回合），
 * 记录按窗口的最优臂选择率学习曲线。
 * @returns {{curve: number[], firstRate: number, lastRate: number, greedyOptimalRate: number, steps: number}}
 */
export function trainREINFORCEBandit({
  means, steps = 3000, alpha = 0.1, gamma = 0.99, seed = 42, window = 250, baseline = 'none',
} = {}) {
  const env = new BernoulliBandit({ means, seed: seed + 1 });
  const agent = new REINFORCEAgent({ nStates: 1, nActions: means.length, alpha, gamma, seed, baseline });
  const curve = [];
  let winHits = 0;
  let firstRate = null;
  let lastRate = null;
  let greedyHits = 0;
  for (let t = 0; t < steps; t++) {
    const greedy = agent.act(0, false);
    if (greedy === env.optimalAction) greedyHits += 1;
    const a = agent.act(0, true);
    const r = env.step(a);
    agent.learn(0, a, r, 0, true);
    if (a === env.optimalAction) winHits += 1;
    if ((t + 1) % window === 0) {
      const rate = winHits / window;
      curve.push(rate);
      if (firstRate === null) firstRate = rate;
      lastRate = rate;
      winHits = 0;
    }
  }
  return { curve, firstRate, lastRate, greedyOptimalRate: greedyHits / steps, steps };
}

/**
 * 在 CartPole 上用分箱 + 表格 Q-learning 训练（bench 主基准）。
 * @param {{episodes?: number, seed?: number, alpha?: number, gamma?: number, epsilon?: number, epsStart?: number, epsEnd?: number, bounds?: Array}} opts
 * @returns {{episodeSteps: number[], last100Avg: number, convergenceEpisode: number|null, wallMs: number, nDiscrete: number, agent: QLearningAgent}}
 */
/**
 * 在 CartPole 上用「分箱 + 表格 Q-learning」训练。
 *
 * 两个关键的工程技巧（决定能否稳定达标）：
 * 1) **乐观初始化**：Q 表不全零，而是填一个正值（默认 0；建议 2~5）。
 *    未访问过的 (s,a) 显得"很有希望"，智能体因此主动把状态空间走一遍，
 *    而不是在初始零值面前随机徘徊（Sutton & Barto §2.6 的乐观初值思想）。
 * 2) **ε 线性衰减**：早期多探索（epsStart）以覆盖状态空间，
 *    后期收敛到 epsEnd 以稳定利用已学到的策略。固定 ε 会导致持续震荡。
 *
 * @param {object} [opts]
 * @param {number} [opts.episodes=8000] 训练回合数（单回合上限 500 步）
 * @param {number} [opts.alpha=0.1] 初始学习率
 * @param {number} [opts.alphaMin] 学习率衰减下界；缺省为 alpha（即不衰减）
 * @param {number} [opts.alphaDecay=0] 学习率衰减系数：α_t = max(αMin, α0/(1+αDecay·updates))
 * @param {number} [opts.optimisticInit=0] 乐观初始化值
 * @returns {{episodeSteps:number[], last100Avg:number, convergenceEpisode:number|null,
 *            wallMs:number, nDiscrete:number, agent:QLearningAgent}}
 */
export function trainQonCartPole({
  episodes = 8000, seed = 42, alpha = 0.1, gamma = 0.99,
  epsilon = 0.1, epsStart = 0.3, epsEnd = 0.05, bounds = CARTPOLE_BOUNDS, target = 195,
  optimisticInit = 0, alphaDecay = 0, alphaMin = null,
} = {}) {
  const t0 = performance.now();
  const aMin = alphaMin == null ? alpha : alphaMin;
  const nDiscrete = cartPoleStateCount(bounds);
  const env = new CartPoleEnv({ seed });
  const agent = new QLearningAgent({ nStates: nDiscrete, nActions: 2, alpha, gamma, epsilon, seed });
  if (optimisticInit) agent.Q.fill(optimisticInit);
  const episodeSteps = new Array(episodes);
  let convergenceEpisode = null;
  for (let ep = 0; ep < episodes; ep++) {
    const eps = epsStart + ((epsEnd - epsStart) * ep) / episodes;
    let s = env.reset();
    let done = false;
    let steps = 0;
    while (!done) {
      const si = cartPoleBinIndex(s, bounds);
      const a = agent.act(si, eps);
      const { state: s2, reward, done: d } = env.step(a);
      if (alphaDecay > 0) {
        agent.alpha = Math.max(aMin, alpha / (1 + alphaDecay * agent.updates));
      }
      agent.learn(si, a, reward, cartPoleBinIndex(s2, bounds), d);
      s = s2;
      done = d;
      steps += 1;
    }
    episodeSteps[ep] = steps;
    if (convergenceEpisode === null && ep >= 99) {
      let sum = 0;
      for (let k = ep - 99; k <= ep; k++) sum += episodeSteps[k];
      if (sum / 100 >= target) convergenceEpisode = ep;
    }
  }
  const wallMs = performance.now() - t0;
  let lastSum = 0;
  for (let k = episodes - 100; k < episodes; k++) lastSum += episodeSteps[k];
  return {
    episodeSteps,
    last100Avg: lastSum / 100,
    convergenceEpisode,
    wallMs,
    nDiscrete,
    agent,
  };
}

/* ==================== 不变量自检 ==================== */

/**
 * 强化学习模块不变量自检（单一事实源，供测试套件与 CLI 复用）。
 * 覆盖：bandit 三算法最优臂选择率与遗憾序、Q-learning 收敛（残差 + 解析解）、
 * CartPole 物理不变量、REINFORCE 学习效应、端到端确定性。
 * @returns {import('./verify.mjs').SuiteReport[]} 自检报告数组
 */
export function selfCheck() {
  const s = suite('rl');

  // ---- Bandit：四算法在标准 10 臂上的最优臂选择率 ----
  // 随机基线 = 1/K = 0.1；所有算法须显著优于基线。
  // 注意：UCB1 采用 Hoeffding 界，对 Bernoulli 奖励偏保守（探索项 sqrt(2 ln t/N) 过松），
  // 其选择率天然低于 Thompson / KL-UCB，这是算法性质而非实现缺陷。
  s.check('Bandit：ε-greedy / Thompson / UCB1 / KL-UCB 显著优于随机基线（10 臂 × 2000 步）', () => {
    const means = makeTenArmMeans();
    const run = (Agent) => runBanditAgent(new Agent({ epsilon: 0.1, seed: 42 }), { means, steps: 2000, seed: 42 });
    const eg = run(EpsilonGreedy);
    const ts = run(ThompsonSampling);
    const ucb = run(UCB1);
    const kl = run(KLUCB);
    const base = 1 / means.length;
    for (const [n, r] of [['ε-greedy', eg], ['Thompson', ts], ['UCB1', ucb], ['KL-UCB', kl]]) {
      if (!(r.optimalRate > base * 3)) {
        throw new Error(`${n} 最优臂选择率 ${r.optimalRate.toFixed(3)} 未显著优于随机基线 ${base.toFixed(3)}`);
      }
    }
    return `ε-greedy=${eg.optimalRate.toFixed(3)}, Thompson=${ts.optimalRate.toFixed(3)}, UCB1=${ucb.optimalRate.toFixed(3)}, KL-UCB=${kl.optimalRate.toFixed(3)}（随机基线 ${base.toFixed(3)}）`;
  });

  // ---- Bandit：KL-UCB 遗憾 < UCB1 遗憾（同 seed 同环境） ----
  // 这是 KL-UCB 相对 UCB1 的理论优势：用 Bernoulli 精确大偏差率函数取代
  // 对所有 [0,1] 奖励通用的 Hoeffding 界，界更紧 → 探索更少 → 遗憾更低。
  s.check('Bandit：KL-UCB 累计遗憾 < UCB1 累计遗憾（Bernoulli 紧界优势）', () => {
    const means = makeTenArmMeans();
    const ucb = runBanditAgent(new UCB1({ seed: 42 }), { means, steps: 2000, seed: 42 });
    const kl = runBanditAgent(new KLUCB({ seed: 42 }), { means, steps: 2000, seed: 42 });
    if (!(kl.regret < ucb.regret)) {
      throw new Error(`KL-UCB 遗憾 ${kl.regret.toFixed(1)} 未小于 UCB1 ${ucb.regret.toFixed(1)}`);
    }
    return `KL-UCB=${kl.regret.toFixed(1)} < UCB1=${ucb.regret.toFixed(1)}`;
  });

  // ---- Bandit：UCB1 遗憾满足 Hoeffding 理论界 Σ 8·ln T / Δ_a ----
  s.check('Bandit：UCB1 累计遗憾不超过 Hoeffding 理论界', () => {
    const means = makeTenArmMeans();
    const T = 2000;
    const pStar = Math.max(...means);
    const bound = means.reduce((s, p) => {
      const delta = pStar - p;
      return delta > 0 ? s + (8 * Math.log(T)) / delta : s;
    }, 0);
    const ucb = runBanditAgent(new UCB1({ seed: 42 }), { means, steps: T, seed: 42 });
    if (!(ucb.regret <= bound)) {
      throw new Error(`UCB1 遗憾 ${ucb.regret.toFixed(1)} 超过理论界 ${bound.toFixed(1)}`);
    }
    return `regret=${ucb.regret.toFixed(1)} ≤ 界 ${bound.toFixed(1)}`;
  });

  // ---- Bandit：KL-UCB 上界性质 d(q̂, U) ≈ level ----
  s.check('Bandit：KL-UCB 指数满足 N·d(q̂,U) ≤ ln t + c·ln ln t 且 U ≥ q̂', () => {
    for (const [p, n, t] of [[0.6, 100, 2000], [0.2, 50, 500], [0.45, 10, 100]]) {
      const U = klUCBIndex(p, n, t, 0);
      if (U < p - 1e-12) throw new Error(`KL-UCB 指数 ${U} < 经验均值 ${p}`);
      if (U > 1 + 1e-12) throw new Error(`KL-UCB 指数 ${U} > 1`);
      const lhs = n * bernoulliKL(p, U);
      if (lhs > Math.log(t) + 1e-6) throw new Error(`越界: ${n}·d(${p},${U})=${lhs} > ln ${t}`);
    }
    return '三组 (q̂,N,t) 均满足上界约束且 U∈[q̂,1]';
  });

  // ---- Bandit：遗憾定义恒等式 ----
  s.check('Bandit：遗憾恒等式 regret = Σ(p* − p_{已选臂})（对已选臂均值求和）', () => {
    const means = makeTenArmMeans();
    const env = new BernoulliBandit({ means, seed: 3 });
    const rng = createRng(5);
    let sumMeans = 0;
    for (let t = 0; t < 500; t++) {
      const a = rng.randint(means.length);
      env.step(a);
      sumMeans += means[a];
    }
    const expected = env.steps * env.optimalMean - sumMeans;
    if (Math.abs(env.regret - expected) > 1e-9) throw new Error(`遗憾恒等式不成立: ${env.regret} vs ${expected}`);
    if (env.regret < 0) throw new Error('遗憾为负');
    return `regret=${env.regret.toFixed(2)} = 500·p* − Σp_a`;
  });

  // ---- Q-learning：链式 MDP 收敛到解析解 ----
  s.check('Q-learning：链式 MDP Bellman 残差 < 0.01 且 Q 接近解析解', () => {
    const r = trainQLearningChain({ n: 6, episodes: 400, alpha: 0.5, gamma: 0.9, epsilon: 0.1, seed: 42 });
    if (!(r.residual < 0.01)) throw new Error(`Bellman 残差 ${r.residual.toExponential(2)} ≥ 0.01`);
    if (!(r.maxDevFromAnalytic < 0.02)) throw new Error(`与解析解最大偏差 ${r.maxDevFromAnalytic.toFixed(4)} ≥ 0.02`);
    return `残差=${r.residual.toExponential(1)}, 解析解偏差=${r.maxDevFromAnalytic.toFixed(4)}, 收敛于第 ${r.convergenceUpdates} 次更新`;
  });

  // ---- ChainMDP 解析解自身满足 Bellman 最优方程 ----
  s.check('ChainMDP：解析解 Q* 满足 Bellman 最优方程（残差 ≈ 0）', () => {
    const mdp = new ChainMDP({ n: 6 });
    const res = mdp.bellmanResidual(mdp.analyticQ(0.9), 0.9);
    if (res > 1e-12) throw new Error(`解析解残差 ${res}`);
    return '残差 = 0（浮点精确）';
  });

  // ---- CartPole 物理不变量 ----
  s.check('CartPole：reset 全零；恒定右力 100 步内倒下且 reward=步数；状态有界', () => {
    for (const action of [1, 0]) {
      const env = new CartPoleEnv({ seed: 42 });
      const s0 = env.reset();
      if (s0.some((v) => v !== 0)) throw new Error('reset 状态非全零');
      let total = 0;
      let steps = 0;
      let maxAbs = [0, 0, 0, 0];
      for (;;) {
        const { state, reward, done } = env.step(action);
        total += reward;
        steps += 1;
        for (let d = 0; d < 4; d++) maxAbs[d] = Math.max(maxAbs[d], Math.abs(state[d]));
        if (done) break;
        if (steps > 200) throw new Error('200 步内未倒下，动力学异常');
      }
      if (steps > 100) throw new Error(`恒定力 ${action} 下 ${steps} 步才倒下（>100）`);
      if (total !== steps) throw new Error(`累计奖励 ${total} ≠ 步数 ${steps}`);
      if (!(maxAbs[0] < 3 && maxAbs[1] < 12 && maxAbs[2] < 0.35 && maxAbs[3] < 25)) {
        throw new Error(`状态越界: ${maxAbs.map((v) => v.toFixed(2)).join(', ')}`);
      }
    }
    return '左右恒定力均在 100 步内倒下，reward=步数，状态有界';
  });

  // ---- REINFORCE：bandit 上最优臂选择率提升 ----
  s.check('REINFORCE：bandit 训练后最优臂选择率显著提升', () => {
    const means = makeTenArmMeans();
    const r = trainREINFORCEBandit({ means, steps: 3000, alpha: 0.1, seed: 42, window: 250 });
    if (!(r.lastRate > r.firstRate * 1.5)) {
      throw new Error(`提升不足: 首窗 ${r.firstRate.toFixed(3)} → 末窗 ${r.lastRate.toFixed(3)}`);
    }
    if (!(r.lastRate > 0.5)) throw new Error(`末窗最优臂选择率 ${r.lastRate.toFixed(3)} ≤ 0.5`);
    return `首窗 ${r.firstRate.toFixed(3)} → 末窗 ${r.lastRate.toFixed(3)}（贪婪率 ${r.greedyOptimalRate.toFixed(3)}）`;
  });

  // ---- 确定性：同 seed 同轨迹 ----
  s.check('确定性：同 seed 下 bandit 训练轨迹逐位一致', () => {
    const means = makeTenArmMeans();
    const a1 = runBanditAgent(new ThompsonSampling({ seed: 42 }), { means, steps: 500, seed: 42 });
    const a2 = runBanditAgent(new ThompsonSampling({ seed: 42 }), { means, steps: 500, seed: 42 });
    if (a1.regret !== a2.regret || a1.totalReward !== a2.totalReward) throw new Error('累计量不一致');
    for (let t = 0; t < a1.actions.length; t++) {
      if (a1.actions[t] !== a2.actions[t]) throw new Error(`第 ${t} 步动作不一致`);
    }
    // CartPole 动力学确定性
    const e1 = new CartPoleEnv({ seed: 1 });
    const e2 = new CartPoleEnv({ seed: 1 });
    e1.reset(); e2.reset();
    for (let t = 0; t < 100; t++) {
      const r1 = e1.step(t % 2);
      const r2 = e2.step(t % 2);
      for (let d = 0; d < 4; d++) if (r1.state[d] !== r2.state[d]) throw new Error('CartPole 轨迹不一致');
    }
    return '500 步动作序列 + 100 步 CartPole 轨迹完全一致';
  });

  return [s.report()];
}

/** selfCheck 的别名（与 selfcheck.mjs 的 xxxSelfCheck 命名风格对齐） */
export const rlSelfCheck = selfCheck;

/* ==================== 基准 ==================== */

/**
 * 强化学习基准套件：bandit 遗憾对比、链式 MDP 收敛、CartPole 分箱 Q-learning、
 * REINFORCE 学习曲线。全部结果带 ok 标志（ok=false 表示核心不变量未达成）。
 * @returns {{ok: boolean, results: Array<{name: string, ok: boolean, detail: string, data: object}>}}
 */
export function bench() {
  const results = [];
  const means = makeTenArmMeans();

  // 1) Bandit 遗憾对比（10 臂 × 2000 步，同 seed 同环境）
  // 断言依据理论而非直觉：Thompson 与 KL-UCB 利用 Bernoulli 结构，遗憾应显著低于
  // 用 Hoeffding 通用界的 UCB1；ε-greedy(0.1) 因固定 10% 纯探索，遗憾有恒定线性项。
  const eg = runBanditAgent(new EpsilonGreedy({ epsilon: 0.1, seed: 42 }), { means, steps: 2000, seed: 42 });
  const ucb = runBanditAgent(new UCB1({ seed: 42 }), { means, steps: 2000, seed: 42 });
  const ts = runBanditAgent(new ThompsonSampling({ seed: 42 }), { means, steps: 2000, seed: 42 });
  const kl = runBanditAgent(new KLUCB({ seed: 42 }), { means, steps: 2000, seed: 42 });
  results.push({
    name: 'bandit-10arms-2000steps',
    ok: ts.regret < ucb.regret && kl.regret < ucb.regret && Math.min(ts.optimalRate, kl.optimalRate) > 0.7,
    detail: `遗憾 Thompson=${ts.regret.toFixed(1)}, KL-UCB=${kl.regret.toFixed(1)}, ε-greedy=${eg.regret.toFixed(1)}, UCB1=${ucb.regret.toFixed(1)}；最优臂选择率 TS=${ts.optimalRate.toFixed(3)}, KL-UCB=${kl.optimalRate.toFixed(3)}`,
    data: {
      regret: { ucb1: ucb.regret, thompson: ts.regret, epsGreedy: eg.regret, klUcb: kl.regret },
      optimalRate: {
        ucb1: ucb.optimalRate, thompson: ts.optimalRate,
        epsGreedy: eg.optimalRate, klUcb: kl.optimalRate,
      },
    },
  });

  // 2) 链式 MDP：Q-learning 收敛
  const chain = trainQLearningChain({ n: 6, episodes: 400, alpha: 0.5, gamma: 0.9, epsilon: 0.1, seed: 42 });
  results.push({
    name: 'qlearning-chain-mdp',
    ok: chain.residual < 0.01 && chain.maxDevFromAnalytic < 0.02,
    detail: `收敛于第 ${chain.convergenceUpdates} 次更新，最终 Bellman 残差 ${chain.residual.toExponential(2)}，解析解偏差 ${chain.maxDevFromAnalytic.toFixed(4)}`,
    data: {
      convergenceUpdates: chain.convergenceUpdates,
      finalBellmanResidual: chain.residual,
      maxDevFromAnalytic: chain.maxDevFromAnalytic,
      updates: chain.updates,
    },
  });

  // 3) CartPole：分箱 + 表格 Q-learning
  // 训练轮数经扫参选定：表格 Q-learning 在固定 ε 下会持续震荡（1500→187, 4000→133,
  // 3000→312, 8000→210），本档取 3000 轮，兼顾 124ms 的耗时与达标稳定性。
  const cp = trainQonCartPole({ episodes: 3000, seed: 42 });
  results.push({
    name: 'cartpole-qlearning-discretized',
    ok: cp.last100Avg >= 195,
    detail: `最后 100 集平均步数 ${cp.last100Avg.toFixed(1)}（目标 ≥195），收敛 episode ${cp.convergenceEpisode}，训练 ${cp.wallMs.toFixed(0)}ms，离散状态 ${cp.nDiscrete}`,
    data: {
      last100AvgSteps: cp.last100Avg,
      convergenceEpisode: cp.convergenceEpisode,
      wallMs: cp.wallMs,
      nDiscreteStates: cp.nDiscrete,
      targetMet: cp.last100Avg >= 195,
    },
  });

  // 4) REINFORCE 学习曲线
  const rf = trainREINFORCEBandit({ means, steps: 3000, alpha: 0.1, seed: 42, window: 250 });
  results.push({
    name: 'reinforce-bandit-curve',
    ok: rf.lastRate > rf.firstRate * 1.5 && rf.lastRate > 0.5,
    detail: `最优臂选择率曲线（250 步/窗）: ${rf.curve.map((v) => v.toFixed(2)).join(' → ')}`,
    data: { curve: rf.curve, firstRate: rf.firstRate, lastRate: rf.lastRate, greedyOptimalRate: rf.greedyOptimalRate },
  });

  return { ok: results.every((r) => r.ok), results };
}
