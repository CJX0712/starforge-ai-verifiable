/**
 * @file verify.mjs — 不变量验证器：StarForge 的"元认知"层
 * 作者：晨星
 *
 * 理念：每个算法至少携带一条可被独立实现交叉验证的不变量（解析解 vs 数值解、
 * 暴力枚举 vs 优化算法、参照实现 vs 主实现）。本模块把这些不变量标准化，
 * 测试套件、CLI 自检、智能体验证证书三处复用同一份代码。
 *
 * 参考：micrograd 的梯度检验传统（中心差分 vs 解析梯度）；
 *       Ribbon 检验/基于性质的测试（property-based testing）思想。
 */

/** 数值梯度（中心差分），作为解析梯度的独立参照实现 */
export function numericGradient(f, x, eps = 1e-5) {
  const n = x.length;
  const g = new Array(n);
  for (let i = 0; i < n; i++) {
    const xp = Array.from(x); xp[i] += eps;
    const xm = Array.from(x); xm[i] -= eps;
    g[i] = (f(xp) - f(xm)) / (2 * eps);
  }
  return g;
}

/** 解析梯度 vs 数值梯度；返回 {maxRelErr, ok, analytic, numeric} */
export function gradCheck(f, x, grad, { eps = 1e-5, tol = 1e-6, floor = 1e-2 } = {}) {
  const numeric = numericGradient(f, x, eps);
  const analytic = grad(x);
  let maxRelErr = 0;
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(analytic[i] - numeric[i]);
    const s = Math.max(Math.abs(analytic[i]), Math.abs(numeric[i]), floor);
    maxRelErr = Math.max(maxRelErr, d / s);
  }
  return { maxRelErr, ok: maxRelErr <= tol, analytic, numeric };
}

/** 数组间最大相对误差（带下限保护） */
export function maxRelErr(a, b, floor = 1e-8) {
  let max = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    const s = Math.max(Math.abs(a[i]), Math.abs(b[i]), floor);
    max = Math.max(max, d / s);
  }
  return max;
}

export function assertTrue(cond, msg) {
  if (!cond) throw new Error(`不变量违反: ${msg}`);
}

export function expectClose(actual, expected, tol = 1e-8, msg = '') {
  const err = Math.abs(actual - expected);
  if (!(err <= tol)) throw new Error(`期望 ${expected}，实际 ${actual}（误差 ${err.toExponential(2)} > 容差 ${tol.toExponential(2)}）${msg ? ' | ' + msg : ''}`);
}

/** 特征分解残差：max_i ‖A·v_i − λ_i·v_i‖₂ */
export function eigenResidual(a, values, vectors) {
  let max = 0;
  for (let k = 0; k < values.length; k++) {
    const v = vectors.col(k);
    const av = new Float64Array(a.rows);
    for (let i = 0; i < a.rows; i++) {
      let s = 0;
      for (let j = 0; j < a.cols; j++) s += a.get(i, j) * v[j];
      av[i] = s;
    }
    let e = 0;
    for (let i = 0; i < a.rows; i++) e = Math.max(e, Math.abs(av[i] - values[k] * v[i]));
    max = Math.max(max, e);
  }
  return max;
}

/** 正交性缺陷：max |VᵀV − I| */
export function orthogonalityDefect(v, tol = 1e-10) {
  const vt = transposeOf(v);
  const g = matMulOf(vt, v);
  let max = 0;
  for (let i = 0; i < g.rows; i++)
    for (let j = 0; j < g.cols; j++)
      max = Math.max(max, Math.abs(g.get(i, j) - (i === j ? 1 : 0)));
  return max;
}

// 延迟引入避免循环依赖的小工具（仅内部使用）
function transposeOf(a) {
  const out = new (a.constructor)(a.cols, a.rows);
  for (let i = 0; i < a.rows; i++) for (let j = 0; j < a.cols; j++) out.set(j, i, a.get(i, j));
  return out;
}
function matMulOf(a, b) {
  const out = new (a.constructor)(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++)
    for (let p = 0; p < a.cols; p++) {
      const av = a.get(i, p);
      if (av === 0) continue;
      for (let j = 0; j < b.cols; j++) out.set(i, j, out.get(i, j) + av * b.get(p, j));
    }
  return out;
}

/** 线性回归正规方程残差：‖Xᵀ(y−Xŵ)‖∞（最小二乘的一阶条件） */
export function residualOrthogonality(X, y, w) {
  const r = new Float64Array(X.rows);
  for (let i = 0; i < X.rows; i++) {
    let p = 0;
    for (let j = 0; j < X.cols; j++) p += X.get(i, j) * w[j];
    r[i] = y[i] - p;
  }
  let max = 0;
  for (let j = 0; j < X.cols; j++) {
    let s = 0;
    for (let i = 0; i < X.rows; i++) s += X.get(i, j) * r[i];
    max = Math.max(max, Math.abs(s));
  }
  return max;
}

/**
 * 检查套件：收集多项检查结果，供测试 / CLI 自检 / 智能体证书复用。
 * 用法：const s = suite('pca'); s.check('正交性', () => {...返回 detail}); s.report()
 */
export function suite(name) {
  const checks = [];
  return {
    check(label, fn) {
      try {
        const detail = fn();
        checks.push({ label, ok: true, detail: detail === undefined ? '' : String(detail) });
      } catch (e) {
        checks.push({ label, ok: false, detail: String(e && e.message ? e.message : e) });
      }
    },
    report() {
      return { name, checks, passed: checks.every((c) => c.ok), passedCount: checks.filter((c) => c.ok).length, total: checks.length };
    },
  };
}
