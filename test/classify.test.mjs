/**
 * @file classify.test.mjs — 监督学习模块测试（node:test）
 * 作者：晨星
 *
 * 运行：node --test test/
 * 不变量主体在 src/classify.mjs 的 selfCheck()（单一事实源），
 * 本文件负责驱动自检，并补充「接口契约」与「可复现性」测试。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { selfCheck, bench } from '../src/classify.mjs';
import {
  LogisticRegression, SoftmaxRegression, Ridge, Lasso, KNN,
  GaussianNB, MultinomialNB, BernoulliNB, DecisionTreeClassifier,
  AdaBoostClassifier, SVMClassifier,
} from '../src/classify.mjs';
import { loadIris, trainTestSplit, zscoreFit, zscoreApply, metrics } from '../src/datasets.mjs';

test('classify 不变量全部通过', () => {
  const reports = selfCheck();
  const failed = [];
  let passed = 0;
  let total = 0;
  for (const r of reports) {
    passed += r.passedCount;
    total += r.total;
    for (const c of r.checks) if (!c.ok) failed.push(`${r.name} / ${c.label}: ${c.detail}`);
  }
  assert.equal(failed.length, 0, `失败的不变量:\n${failed.join('\n')}`);
  console.log(`  不变量: ${passed}/${total} 通过`);
});

test('classify 基准全部达标', () => {
  const { results } = bench();
  const bad = results.filter((r) => !r.ok);
  assert.equal(bad.length, 0, `未达标项:\n${bad.map((b) => `${b.name}=${b.value} (${b.target}) ${b.detail}`).join('\n')}`);
  console.log(`  基准: ${results.length}/${results.length} 达标`);
});

test('接口契约：分类器统一提供 fit/predict/score', () => {
  const { X, y } = loadIris();
  const { train, test: te } = trainTestSplit(X, y, { testRatio: 0.3, seed: 42 });
  const fit = zscoreFit(train.X);
  const Xtr = zscoreApply(train.X, fit);
  const Xte = zscoreApply(te.X, fit);
  const factories = [
    () => new SoftmaxRegression({ epochs: 30 }),
    () => new DecisionTreeClassifier({ maxDepth: 3 }),
    () => new KNN({ k: 5 }),
    () => new GaussianNB(),
    () => new SVMClassifier({ kernel: 'rbf', C: 2 }),
  ];
  for (const mk of factories) {
    const m = mk().fit(Xtr, train.y);
    const pred = m.predict(Xte);
    assert.equal(pred.length, Xte.rows, '预测数量须等于样本数');
    for (const p of pred) assert.ok(Number.isInteger(p) && p >= 0 && p < 3, '预测须为合法类别');
    const acc = m.score(Xte, te.y);
    assert.ok(acc >= 0 && acc <= 1, '准确率须在 [0,1]');
  }
});

test('接口契约：概率输出合法（行和=1，元素∈[0,1]）', () => {
  const { X, y } = loadIris();
  const { train } = trainTestSplit(X, y, { testRatio: 0.3, seed: 42 });
  const Xtr = zscoreApply(train.X, zscoreFit(train.X));
  const sm = new SoftmaxRegression({ epochs: 30 }).fit(Xtr, train.y);
  const P = sm.predictProba(Xtr);
  for (let i = 0; i < P.rows; i++) {
    let s = 0;
    for (let j = 0; j < P.cols; j++) {
      const v = P.get(i, j);
      assert.ok(v >= -1e-12 && v <= 1 + 1e-12, '概率须在 [0,1]');
      s += v;
    }
    assert.ok(Math.abs(s - 1) < 1e-9, `第 ${i} 行概率和须为 1，实际 ${s}`);
  }
  const nb = new GaussianNB().fit(Xtr, train.y);
  const LP = nb.predictLogProba(Xtr);
  for (let i = 0; i < LP.rows; i++) {
    let s = 0;
    for (let j = 0; j < LP.cols; j++) {
      assert.ok(LP.get(i, j) <= 1e-12, '对数概率须 ≤ 0');
      s += Math.exp(LP.get(i, j));
    }
    assert.ok(Math.abs(s - 1) < 1e-6, `朴素贝叶斯第 ${i} 行后验和须为 1，实际 ${s}`);
  }
});

test('接口契约：二分类概率 ∈ [0,1] 且与 predict 一致', () => {
  const { X, y } = loadIris();
  const yBin = y.map((v) => (v === 0 ? 1 : 0));
  const lr = new LogisticRegression({ epochs: 60 }).fit(X, yBin);
  const p = lr.predictProba(X);
  const pred = lr.predict(X);
  for (let i = 0; i < p.length; i++) {
    assert.ok(p[i] >= 0 && p[i] <= 1, '概率越界');
    assert.equal(pred[i], p[i] > 0.5 ? 1 : 0, 'predict 须与 0.5 阈值一致');
  }
});

test('正则化单调性：Lasso 的 λ 越大，系数越稀疏', () => {
  const { X, y } = loadIris();
  const Xtr = zscoreApply(X, zscoreFit(X));
  const sparsity = [];
  for (const lambda of [1e-4, 1e-2, 1, 10]) {
    const w = new Lasso({ lambda, maxIter: 300 }).fit(Xtr, y).w;
    sparsity.push(Array.from(w).filter((v) => Math.abs(v) < 1e-8).length);
  }
  for (let i = 1; i < sparsity.length; i++) {
    assert.ok(sparsity[i] >= sparsity[i - 1], `λ 增大后零系数不应减少: ${sparsity.join(' → ')}`);
  }
  console.log(`  Lasso 零系数个数随 λ: ${sparsity.join(' → ')}`);
});

test('确定性：同种子训练两次结果逐位一致', () => {
  const { X, y } = loadIris();
  const Xtr = zscoreApply(X, zscoreFit(X));
  const a = new SoftmaxRegression({ epochs: 20, seed: 7 }).fit(Xtr, y).predict(Xtr);
  const b = new SoftmaxRegression({ epochs: 20, seed: 7 }).fit(Xtr, y).predict(Xtr);
  assert.deepEqual(a, b, '同种子训练结果必须一致');
});

test('AdaBoost 集成不低于单棵树桩', () => {
  const { X, y } = loadIris();
  const Xtr = zscoreApply(X, zscoreFit(X));
  const yBin = y.map((v) => (v === 0 ? 1 : 0));
  const boost = new AdaBoostClassifier({ nEstimators: 20 }).fit(Xtr, yBin);
  const accBoost = boost.score(Xtr, yBin);
  const stump = new DecisionTreeClassifier({ maxDepth: 1 }).fit(Xtr, yBin);
  const accStump = stump.score(Xtr, yBin);
  assert.ok(accBoost >= accStump - 1e-12, `集成 ${accBoost} 不应低于树桩 ${accStump}`);
  console.log(`  AdaBoost=${accBoost.toFixed(4)} ≥ 树桩=${accStump.toFixed(4)}`);
});

test('kNN 惰性学习：k=1 时训练集准确率=1', () => {
  const { X, y } = loadIris();
  const Xtr = zscoreApply(X, zscoreFit(X));
  assert.equal(new KNN({ k: 1 }).fit(Xtr, y).score(Xtr, y), 1);
});
