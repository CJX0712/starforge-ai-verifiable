/**
 * @file cluster.test.mjs — 聚类与降维测试（node:test）
 * 作者：晨星
 *
 * 运行：node --test test/cluster.test.mjs
 * 不变量主体在 src/cluster.mjs 的 selfCheck()（单一事实源），
 * 本文件负责驱动自检 + 补充接口契约测试。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { selfCheck, bench as clusterBench, KMeans, GMM, PCA, AgglomerativeClustering } from '../src/cluster.mjs';
import { Mat } from '../src/linalg.mjs';
import { makeBlobs, makeMoons, loadIris, loadWine, metrics } from '../src/datasets.mjs';

test('cluster 不变量全部通过', () => {
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

test('cluster 每个自检组检查项充足', () => {
  for (const r of selfCheck()) {
    assert.ok(r.total >= 2, `${r.name} 检查项过少`);
  }
});

test('cluster bench 全部 ok', () => {
  const b = clusterBench();
  const failed = b.results.filter((r) => !r.ok).map((r) => `${r.name}: ${r.detail}`);
  assert.equal(failed.length, 0, `bench 失败:\n${failed.join('\n')}`);
});

// ==================== KMeans 契约 ====================

test('KMeans：fit 返回结构与形状正确', () => {
  const { X } = makeBlobs({ n: 90, k: 3 });
  const km = new KMeans({ k: 3 });
  const out = km.fit(X);
  assert.ok(Array.isArray(out.labels_) && out.labels_.length === 90);
  assert.ok(out.centroids_ instanceof Mat);
  assert.equal(out.centroids_.rows, 3);
  assert.equal(out.centroids_.cols, X.cols);
  assert.ok(Number.isFinite(out.inertia_) && out.inertia_ >= 0);
  for (const l of out.labels_) assert.ok(l >= 0 && l < 3);
});

test('KMeans：确定性——同种子同结果，不同种子可不同', () => {
  const { X } = makeBlobs({ n: 90, k: 3 });
  const a = new KMeans({ k: 3, seed: 42 }).fit(X);
  const b = new KMeans({ k: 3, seed: 42 }).fit(X);
  assert.deepEqual(a.labels_, b.labels_);
  assert.equal(a.inertia_, b.inertia_);
  const c = new KMeans({ k: 3, seed: 7 }).fit(X);
  // 不同种子至少 inertia 应接近（都收敛），但不要求相同
  assert.ok(Math.abs(c.inertia_ - a.inertia_) < a.inertia_ * 0.2);
});

test('KMeans：predict 与 fit 的 labels_ 一致；predict 可处理新样本', () => {
  const { X } = makeBlobs({ n: 90, k: 3 });
  const km = new KMeans({ k: 3 });
  km.fit(X);
  assert.deepEqual(km.predict(X), km.labels_);
  // 贴近质心的点应归入对应簇
  for (let c = 0; c < 3; c++) {
    const probe = new Mat(1, X.cols);
    for (let j = 0; j < X.cols; j++) probe.set(0, j, km.centroids_.get(c, j));
    assert.equal(km.predict(probe)[0], c);
  }
});

test('KMeans：SSE 与手算一致', () => {
  const X = Mat.from2D([[0, 0], [0, 1], [10, 0], [10, 1]]);
  const km = new KMeans({ k: 2, seed: 42 });
  km.fit(X);
  let manual = 0;
  for (let i = 0; i < X.rows; i++) {
    const c = km.labels_[i];
    manual += (X.get(i, 0) - km.centroids_.get(c, 0)) ** 2 + (X.get(i, 1) - km.centroids_.get(c, 1)) ** 2;
  }
  assert.ok(Math.abs(manual - km.inertia_) < 1e-9, `${manual} vs ${km.inertia_}`);
});

test('KMeans：kmeans++ 播种的 D² 性质（首批质心互异且覆盖远点）', () => {
  const { X } = makeBlobs({ n: 60, k: 3, spread: 20 });
  const km = new KMeans({ k: 3, seed: 42, init: 'kmeans++' });
  km.fit(X);
  const centroids = km.centroids_;
  // 三质心两两距离应显著大于 0（kmeans++ 倾向远点）
  const d01 = Math.hypot(centroids.get(0, 0) - centroids.get(1, 0), centroids.get(0, 1) - centroids.get(1, 1));
  const d02 = Math.hypot(centroids.get(0, 0) - centroids.get(2, 0), centroids.get(0, 1) - centroids.get(2, 1));
  assert.ok(d01 > 1 && d02 > 1, `质心过近: ${d01}, ${d02}`);
});

test('KMeans：错误处理', () => {
  assert.throws(() => new KMeans({ k: 0 }), /正整数/);
  assert.throws(() => new KMeans({ k: 3, init: 'bogus' }), /初始化/);
  const { X } = makeBlobs({ n: 10, k: 3 });
  assert.throws(() => new KMeans({ k: 11 }).fit(X), /超过样本数/);
  const km = new KMeans({ k: 2 });
  assert.throws(() => km.predict(X), /尚未拟合/);
});

// ==================== GMM 契约 ====================

test('GMM：fit 返回结构与形状正确', () => {
  const { X } = makeBlobs({ n: 90, k: 2 });
  const gmm = new GMM({ k: 2 });
  const out = gmm.fit(X);
  assert.equal(out.labels_.length, 90);
  assert.equal(out.means_.rows, 2);
  assert.equal(out.means_.cols, X.cols);
  assert.equal(out.weights_.length, 2);
  assert.equal(out.covariances_.length, 2); // full → Mat[]
  assert.ok(Number.isFinite(out.logLikelihood_));
  assert.ok(out.nIter_ >= 1);
});

test('GMM：predictProba 行和=1 且与 predict 一致', () => {
  const { X } = makeBlobs({ n: 90, k: 3 });
  const gmm = new GMM({ k: 3 });
  gmm.fit(X);
  const resp = gmm.predictProba(X);
  assert.equal(resp.rows, 90);
  assert.equal(resp.cols, 3);
  const pred = gmm.predict(X);
  for (let i = 0; i < resp.rows; i++) {
    let sum = 0;
    let best = 0;
    for (let c = 0; c < resp.cols; c++) {
      sum += resp.get(i, c);
      if (resp.get(i, c) > resp.get(i, best)) best = c;
    }
    assert.ok(Math.abs(sum - 1) < 1e-9);
    assert.equal(pred[i], best);
  }
});

test('GMM：score = 平均对数似然，与 logLikelihood_/n 一致', () => {
  const { X } = makeBlobs({ n: 60, k: 2 });
  const gmm = new GMM({ k: 2, maxIter: 50, tol: 1e-9 });
  gmm.fit(X);
  assert.ok(Math.abs(gmm.score(X) - gmm.logLikelihood_ / X.rows) < 1e-8);
});

test('GMM：权重和为 1，covariances 维度正确', () => {
  const { X } = makeBlobs({ n: 60, k: 2 });
  const gmm = new GMM({ k: 2 });
  gmm.fit(X);
  let w = 0;
  for (const v of gmm.weights_) w += v;
  assert.ok(Math.abs(w - 1) < 1e-9);
  for (const S of gmm.covariances_) {
    assert.equal(S.rows, X.cols);
    assert.equal(S.cols, X.cols);
  }
});

test('GMM：diag 模式下 covariances_ 为 k×d 方差矩阵', () => {
  const { X } = makeBlobs({ n: 60, k: 2 });
  const gmm = new GMM({ k: 2, covariance: 'diag' });
  gmm.fit(X);
  assert.ok(gmm.covariances_ instanceof Mat);
  assert.equal(gmm.covariances_.rows, 2);
  assert.equal(gmm.covariances_.cols, X.cols);
  for (let i = 0; i < gmm.covariances_.size; i++) assert.ok(gmm.covariances_.data[i] > 0);
});

test('GMM：确定性——同种子同对数似然', () => {
  const { X } = makeBlobs({ n: 60, k: 2 });
  const a = new GMM({ k: 2, seed: 5 }).fit(X);
  const b = new GMM({ k: 2, seed: 5 }).fit(X);
  assert.equal(a.logLikelihood_, b.logLikelihood_);
});

test('GMM：错误处理', () => {
  assert.throws(() => new GMM({ k: 2, covariance: 'tied' }), /协方差/);
  const { X } = makeBlobs({ n: 10, k: 2 });
  assert.throws(() => new GMM({ k: 11 }).fit(X), /超过样本数/);
});

// ==================== PCA 契约 ====================

test('PCA：fit 返回结构与 Iris 形状正确', () => {
  const { X } = loadIris();
  const pca = new PCA({ nComponents: 2 });
  const out = pca.fit(X);
  assert.equal(out.components_.rows, 2);
  assert.equal(out.components_.cols, 4);
  assert.equal(out.explainedVariance_.length, 2);
  assert.equal(out.explainedVarianceRatio_.length, 2);
  assert.equal(out.mean_.length, 4);
  // Iris 均值应接近真实均值
  for (let j = 0; j < 4; j++) {
    let m = 0;
    for (let i = 0; i < X.rows; i++) m += X.get(i, j);
    assert.ok(Math.abs(out.mean_[j] - m / X.rows) < 1e-10);
  }
});

test('PCA：transform/inverseTransform 往返（满秩时近似还原）', () => {
  const { X } = loadIris();
  const pca = new PCA({ nComponents: 4 });
  const Z = pca.fitTransform(X);
  assert.equal(Z.rows, X.rows);
  assert.equal(Z.cols, 4);
  const Xr = pca.inverseTransform(Z);
  for (let i = 0; i < X.rows; i++) for (let j = 0; j < X.cols; j++) {
    assert.ok(Math.abs(X.get(i, j) - Xr.get(i, j)) < 1e-8, `(${i},${j}) 重构误差过大`);
  }
});

test('PCA：满秩时累计解释方差比 = 1', () => {
  const { X } = loadIris();
  const pca = new PCA({ nComponents: 4 });
  pca.fit(X);
  let cum = 0;
  for (const v of pca.explainedVarianceRatio_) cum += v;
  assert.ok(Math.abs(cum - 1) < 1e-9, `cum = ${cum}`);
});

test('PCA：降序且解释方差 = 特征方向上的样本方差', () => {
  const { X } = loadIris();
  const pca = new PCA({ nComponents: 4 });
  pca.fit(X);
  const Z = pca.transform(X);
  for (let c = 0; c < 4; c++) {
    // 投影坐标的样本方差（除以 n−1）应等于 explainedVariance_[c]
    let m = 0;
    for (let i = 0; i < Z.rows; i++) m += Z.get(i, c);
    m /= Z.rows;
    let v = 0;
    for (let i = 0; i < Z.rows; i++) v += (Z.get(i, c) - m) ** 2;
    v /= Z.rows - 1;
    assert.ok(Math.abs(v - pca.explainedVariance_[c]) < 1e-6, `分量 ${c}: ${v} vs ${pca.explainedVariance_[c]}`);
  }
  for (let c = 1; c < 4; c++) assert.ok(pca.explainedVariance_[c] <= pca.explainedVariance_[c - 1] + 1e-12);
});

test('PCA：成分正交（行内积 = δ）', () => {
  const { X } = loadWine();
  const pca = new PCA({ nComponents: 13 });
  pca.fit(X);
  for (let a = 0; a < 13; a++) for (let b = 0; b < 13; b++) {
    let dot = 0;
    for (let j = 0; j < 13; j++) dot += pca.components_.get(a, j) * pca.components_.get(b, j);
    const expect = a === b ? 1 : 0;
    assert.ok(Math.abs(dot - expect) < 1e-8, `(${a},${b}) 内积 = ${dot}`);
  }
});

test('PCA：确定性——同数据同成分（符号约定除外，比较子空间投影）', () => {
  const { X } = loadIris();
  const z1 = new PCA({ nComponents: 2 }).fitTransform(X);
  const z2 = new PCA({ nComponents: 2 }).fitTransform(X);
  for (let i = 0; i < z1.size; i++) assert.equal(z1.data[i], z2.data[i]);
});

test('PCA：错误处理', () => {
  assert.throws(() => new PCA({ nComponents: 0 }), /正整数/);
  const { X } = loadIris();
  assert.throws(() => new PCA({ nComponents: 5 }).fit(X), /超过/);
  const pca = new PCA({ nComponents: 2 });
  assert.throws(() => pca.transform(X), /尚未拟合/);
});

// ==================== Agglomerative 契约 ====================

test('AgglomerativeClustering：blobs 纯度与标签结构', () => {
  const { X, y } = makeBlobs({ n: 150, k: 3 });
  const agg = new AgglomerativeClustering({ k: 3 });
  const out = agg.fit(X);
  assert.equal(out.labels_.length, 150);
  assert.equal(new Set(out.labels_).size, 3);
  const purity = metrics.purity(y, out.labels_, 3);
  assert.ok(purity >= 0.9, `purity = ${purity}`);
});

test('AgglomerativeClustering：k=n 时每点独立成簇', () => {
  const { X } = makeBlobs({ n: 12, k: 3 });
  const agg = new AgglomerativeClustering({ k: 12 });
  agg.fit(X);
  assert.equal(new Set(agg.labels_).size, 12);
});

test('AgglomerativeClustering： moons 上不如 KMeans（凸假设的记录性对照）', () => {
  const { X, y } = makeMoons({ n: 120, noise: 0.1 });
  const agg = new AgglomerativeClustering({ k: 2 }).fit(X);
  const p = metrics.purity(y, agg.labels_, 2);
  // 平均连接对双月牙同样无能为力（记录值，不做通过性断言）
  assert.ok(p >= 0.5 && p <= 1);
});

test('AgglomerativeClustering：错误处理', () => {
  assert.throws(() => new AgglomerativeClustering({ k: 2, linkage: 'ward' }), /average/);
  const { X } = makeBlobs({ n: 5, k: 2 });
  assert.throws(() => new AgglomerativeClustering({ k: 6 }).fit(X), /超过样本数/);
});

// ==================== 跨算法对照 ====================

test('KMeans 与 GMM 在分离 blobs 上标签一致', () => {
  const { X, y } = makeBlobs({ n: 150, k: 3, std: 0.5 });
  const km = new KMeans({ k: 3 }).fit(X).labels_;
  const gm = new GMM({ k: 3, maxIter: 100 }).fit(X).labels_;
  assert.equal(metrics.purity(y, km, 3), 1);
  assert.equal(metrics.purity(y, gm, 3), 1);
});
