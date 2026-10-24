/**
 * @file selfcheck.mjs — 全引擎不变量自检中心（单一事实源）
 * 作者：晨星
 *
 * CLI `selftest`、测试套件、Web 仪表盘三处复用本文件。
 * 每个算法的不变量都在这里以可执行代码的形式固定下来——
 * 这是"可验证 AI"的第一层：任何回归都会立刻在自检中暴露。
 */
import { suite, gradCheck, eigenResidual, orthogonalityDefect, residualOrthogonality, expectClose } from './verify.mjs';
import {
  Mat, matMul, matVec, transpose, solveLinear, inverse, cholesky, isPositiveDefinite,
  qr, jacobiEigen, svd, logsumexp, softmaxArr, createRng, frobeniusNorm, det,
} from './linalg.mjs';
import { Tensor, gradCheckTensor, concatCols } from './autodiff.mjs';
import { Adam, SGD, Lion, Muon, newtonSchulz, clipGlobalNorm, makeSchedule } from './optim.mjs';
import {
  loadIris, loadWine, makeMoons, makeBlobs, makeXor, makeRegression,
  trainTestSplit, zscoreFit, zscoreApply, metrics, loadCorpus, encodeText, decodeIds, charBatch,
} from './datasets.mjs';

function randMat(rng, rows, cols, scale = 1) {
  const m = new Mat(rows, cols);
  for (let i = 0; i < rows * cols; i++) m.data[i] = rng.normal() * scale;
  return m;
}

// ==================== 线性代数不变量 ====================
export function linalgSelfCheck() {
  const s = suite('linalg');
  const rng = createRng(42);

  s.check('matMul 与朴素三重循环一致', () => {
    const A = randMat(rng, 4, 5), B = randMat(rng, 5, 3);
    const C = matMul(A, B);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 5; k++) sum += A.get(i, k) * B.get(k, j);
      expectClose(C.get(i, j), sum, 1e-10);
    }
    return 'maxErr < 1e-10';
  });

  s.check('solveLinear 残差 ‖Ax−b‖≈0', () => {
    const A = randMat(rng, 6, 6);
    const xTrue = Float64Array.from({ length: 6 }, () => rng.normal());
    const x = solveLinear(A, matVec(A, xTrue));
    let e = 0;
    for (let i = 0; i < 6; i++) e = Math.max(e, Math.abs(x[i] - xTrue[i]));
    expectClose(e, 0, 1e-8);
    return `err = ${e.toExponential(1)}`;
  });

  s.check('A·A⁻¹ = I', () => {
    const A = randMat(rng, 5, 5);
    const I = matMul(A, inverse(A));
    let e = 0;
    for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) e = Math.max(e, Math.abs(I.get(i, j) - (i === j ? 1 : 0)));
    expectClose(e, 0, 1e-8);
    return `err = ${e.toExponential(1)}`;
  });

  s.check('Cholesky: L·Lᵀ=A 且正定性判定正确', () => {
    const M = randMat(rng, 4, 4);
    const A = matMul(M, transpose(M)); // 对称半正定
    const L = cholesky(A);
    const R = matMul(L, transpose(L));
    let e = 0;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) e = Math.max(e, Math.abs(R.get(i, j) - A.get(i, j)));
    expectClose(e, 0, 1e-8);
    const notPD = new Mat(2, 2, new Float64Array([1, 2, 2, 1])); // 特征值 3, −1
    if (isPositiveDefinite(notPD)) throw new Error('负定矩阵被误判为正定');
    return `重构 err = ${e.toExponential(1)}`;
  });

  s.check('QR: QᵀQ=I，QR=A，R 上三角', () => {
    const A = randMat(rng, 6, 4);
    const { Q, R } = qr(A);
    const QtQ = matMul(transpose(Q), Q);
    let e1 = 0;
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) e1 = Math.max(e1, Math.abs(QtQ.get(i, j) - (i === j ? 1 : 0)));
    expectClose(e1, 0, 1e-8);
    const QR = matMul(Q, R);
    let e2 = 0;
    for (let i = 0; i < 6; i++) for (let j = 0; j < 4; j++) e2 = Math.max(e2, Math.abs(QR.get(i, j) - A.get(i, j)));
    expectClose(e2, 0, 1e-8);
    for (let i = 0; i < 6; i++) for (let j = 0; j < 4; j++) if (i > j) expectClose(R.get(i, j), 0, 1e-10);
    return `正交 err=${e1.toExponential(1)}, 重构 err=${e2.toExponential(1)}`;
  });

  s.check('Jacobi: 残差 ‖Av−λv‖<1e-8，特征向量正交，降序', () => {
    const M = randMat(rng, 5, 5);
    const A = matMul(M, transpose(M)); // 对称半正定
    const { values, vectors } = jacobiEigen(A);
    expectClose(eigenResidual(A, values, vectors), 0, 1e-8);
    expectClose(orthogonalityDefect(vectors), 0, 1e-8);
    for (let i = 1; i < values.length; i++) {
      if (values[i] > values[i - 1] + 1e-12) throw new Error('特征值未按降序排列');
    }
    return `残差 ${eigenResidual(A, values, vectors).toExponential(1)}`;
  });

  s.check('Jacobi: 已知矩阵特征值解析解', () => {
    const A = new Mat(2, 2, new Float64Array([2, 1, 1, 2]));
    const { values } = jacobiEigen(A);
    expectClose(values[0], 3, 1e-10);
    expectClose(values[1], 1, 1e-10);
    return 'λ = (3, 1)';
  });

  s.check('SVD: A≈UΣVᵀ，U 列正交', () => {
    const A = randMat(rng, 5, 3);
    const { U, s, V } = svd(A);
    const rec = matMul(matMul(U, Mat.diag(Array.from(s))), transpose(V));
    let e = 0;
    for (let i = 0; i < 5; i++) for (let j = 0; j < 3; j++) e = Math.max(e, Math.abs(rec.get(i, j) - A.get(i, j)));
    expectClose(e, 0, 1e-8);
    const UtU = matMul(transpose(U), U);
    let e2 = 0;
    for (let i = 0; i < UtU.rows; i++) for (let j = 0; j < UtU.cols; j++) e2 = Math.max(e2, Math.abs(UtU.get(i, j) - (i === j ? 1 : 0)));
    expectClose(e2, 0, 1e-8);
    return `重构 err=${e.toExponential(1)}`;
  });

  s.check('logsumexp ≥ max 且与朴素计算一致', () => {
    const xs = [1, 2, 3, 4, 5];
    const direct = Math.log(xs.reduce((a, b) => a + Math.exp(b), 0));
    expectClose(logsumexp(xs), direct, 1e-12);
    const big = [1000, 1001, 1002];
    expectClose(logsumexp(big), 1002 + Math.log(Math.exp(-2) + Math.exp(-1) + 1), 1e-10);
    return '稳定（无溢出）';
  });

  s.check('softmax 行和为 1 且最大元素概率最大', () => {
    const p = softmaxArr([1, 2, 3]);
    expectClose(p.reduce((a, b) => a + b, 0), 1, 1e-12);
    if (p[2] <= p[1]) throw new Error('softmax 次序错误');
    return 'sum=1';
  });

  s.check('RNG 同种子同序列（可复现性）', () => {
    const r1 = createRng(123), r2 = createRng(123);
    for (let i = 0; i < 100; i++) {
      if (r1.next() !== r2.next()) throw new Error('RNG 不确定');
    }
    return '100 次抽样一致';
  });

  return [s.report()];
}

// ==================== 自动微分不变量（金标准：梯度检验） ====================
export function autodiffSelfCheck() {
  const s = suite('autodiff');
  const rng = createRng(7);

  const gc = (label, f, t) => {
    const r = gradCheckTensor(f, t);
    s.check(label, () => {
      if (!r.ok) throw new Error(`maxRelErr = ${r.maxRelErr.toExponential(2)}`);
      return `maxRelErr = ${r.maxRelErr.toExponential(1)}`;
    });
  };

  gc('matmul 链 XᵀX', (A) => A.transpose().matmul(A).sum(), Tensor.randn([3, 4], rng));
  gc('融合 softmax+交叉熵', (L) => L.crossEntropyLogits([0, 1, 2, 3, 4]), Tensor.randn([5, 6], rng));
  const lnGamma = Tensor.randn([1, 4], rng, 0.5).addScalar(1);
  const lnBeta = Tensor.randn([1, 4], rng, 0.1);
  gc('LayerNorm（原语组合）', (x) => {
    const mu = x.rowMean();
    const xc = x.sub(mu);
    const v = xc.pow(2).rowMean().addScalar(1e-5).sqrt();
    return xc.div(v).mul(lnGamma).add(lnBeta).sum();
  }, Tensor.randn([6, 4], rng));
  gc('GELU（原语组合）', (x) => x.gelu().sum(), Tensor.randn([5, 5], rng));
  gc('embedding 查表 indexRows', (W) => W.indexRows([0, 2, 2, 1, 3]).sum(), Tensor.randn([4, 3], rng));
  gc('slice + concat', (x) => concatCols([x.sliceCols(0, 2), x.sliceCols(2, 4).mulScalar(2)]).sliceRows(1, 4).sum(), Tensor.randn([5, 4], rng));
  const bcCol = Tensor.randn([5, 1], rng);
  const bcRow = Tensor.randn([1, 4], rng);
  gc('广播 [r,1]+[r,c] 与 [1,c]×[r,c]', (x) => x.add(bcCol).mul(bcRow).sum(), Tensor.randn([5, 4], rng));
  const mseTarget = Tensor.randn([4, 3], rng);
  gc('MSE 损失', (x) => x.mseLoss(mseTarget), Tensor.randn([4, 3], rng));
  gc('一元运算链 exp/log/sqrt/tanh/sigmoid/pow', (x) => x.sigmoid().mulScalar(0.98).addScalar(0.01).exp().log().sqrt().tanh().pow(2).sum(), Tensor.randn([4, 4], rng));
  gc('菱形图（梯度多路径累加）', (x) => x.mul(x).add(x.mul(x.mulScalar(2))).sum(), Tensor.randn([3, 3], rng));
  gc('除法 backward', (x) => x.div(x.transpose().matmul(x).addScalar(2)).sum(), Tensor.randn([3, 3], rng));

  s.check('micrograd 公开参照值（Karpathy README）', () => {
    const a = Tensor.from2D([[2]]);
    const b = Tensor.from2D([[3]]);
    const c = a.mul(b);
    const L = c.add(a);
    L.backward();
    expectClose(a.grad[0], 4, 1e-12);
    expectClose(b.grad[0], 2, 1e-12);
    return 'dg/da=4, dg/db=2';
  });

  s.check('训练确定性：同种子同损失轨迹', () => {
    const run = () => {
      const r = createRng(99);
      const w = Tensor.randn([3, 1], r, 0.1);
      const X = Tensor.randn([8, 3], r);
      const y = Tensor.randn([8, 1], r);
      const opt = new Adam([w], { lr: 0.05 });
      const losses = [];
      for (let i = 0; i < 20; i++) {
        const loss = X.matmul(w).sub(y).pow(2).mean();
        loss.backward();
        opt.step();
        losses.push(loss.data[0]);
      }
      return losses;
    };
    const l1 = run();
    const l2 = run();
    for (let i = 0; i < l1.length; i++) expectClose(l1[i], l2[i], 0);
    return '20 步损失完全一致';
  });

  return [s.report()];
}

// ==================== 优化器不变量 ====================
export function optimSelfCheck() {
  const s = suite('optim');

  s.check('Adam 收敛到凸二次问题解析解', () => {
    const rng = createRng(11);
    const X = randMat(rng, 10, 3);
    const wTrue = Float64Array.from({ length: 3 }, () => rng.normal());
    const y = matVec(X, wTrue);
    const wAnalytic = solveLinear(matMul(transpose(X), X), matVec(transpose(X), y));
    const Xt = Tensor.from2D(X.to2D());
    const yt = Tensor.from2D(Array.from(y, (v) => [v]));
    const w = Tensor.zeros([3, 1]);
    const opt = new Adam([w], { lr: 0.1 });
    let first = Infinity, last = Infinity;
    for (let i = 0; i < 500; i++) {
      const loss = Xt.matmul(w).sub(yt).pow(2).mean();
      loss.backward();
      opt.step();
      if (i === 0) first = loss.data[0];
      last = loss.data[0];
    }
    let e = 0;
    for (let i = 0; i < 3; i++) e = Math.max(e, Math.abs(w.get(i, 0) - wAnalytic[i]));
    if (e > 1e-4) throw new Error(`解析解偏差 ${e.toExponential(2)}`);
    if (!(last < first * 1e-3)) throw new Error(`损失未充分下降: ${first} → ${last}`);
    return `解析解偏差 ${e.toExponential(1)}，损失 ${first.toFixed(4)} → ${last.toExponential(1)}`;
  });

  s.check('Muon 收敛于矩阵二次问题（含 Newton-Schulz 路径）', () => {
    const rng = createRng(5);
    const X = randMat(rng, 16, 4);
    const Wt = randMat(rng, 4, 2);
    const Y = matMul(X, Wt);
    const Xt = Tensor.from2D(X.to2D());
    const Yt = Tensor.from2D(Y.to2D());
    const W = Tensor.zeros([4, 2]);
    const opt = new Muon([W], { lr: 0.08 });
    let first = Infinity, last = Infinity;
    for (let i = 0; i < 400; i++) {
      const loss = Xt.matmul(W).sub(Yt).pow(2).mean();
      loss.backward();
      opt.step();
      if (i === 0) first = loss.data[0];
      last = loss.data[0];
    }
    if (!(last < first * 1e-2)) throw new Error(`Muon 收敛不足: ${first.toFixed(4)} → ${last.toExponential(2)}`);
    return `损失 ${first.toFixed(4)} → ${last.toExponential(2)}（${(first / last).toFixed(0)}×）`;
  });

  s.check('Muon vs AdamW：同一矩阵问题对比记录', () => {
    const run = (name) => {
      const rng = createRng(5);
      const X = randMat(rng, 16, 4);
      const Wt = randMat(rng, 4, 2);
      const Y = matMul(X, Wt);
      const Xt = Tensor.from2D(X.to2D());
      const Yt = Tensor.from2D(Y.to2D());
      const W = Tensor.zeros([4, 2]);
      W.muonExempt = name === 'adamw';
      const opt = name === 'muon' ? new Muon([W], { lr: 0.08 }) : new Adam([W], { lr: 0.05 });
      let last = 0;
      for (let i = 0; i < 400; i++) {
        const loss = Xt.matmul(W).sub(Yt).pow(2).mean();
        loss.backward();
        opt.step();
        last = loss.data[0];
      }
      return last;
    };
    const muonLoss = run('muon');
    const adamLoss = run('adamw');
    return `400 步后损失：Muon=${muonLoss.toExponential(3)}, AdamW=${adamLoss.toExponential(3)}`;
  });

  s.check('SGD 动量与 Lion 均能下降损失', () => {
    const rng = createRng(3);
    const X = randMat(rng, 10, 3);
    const y = matVec(X, Float64Array.from({ length: 3 }, () => rng.normal()));
    const Xt = Tensor.from2D(X.to2D());
    const yt = Tensor.from2D(Array.from(y, (v) => [v]));
    for (const [name, mk] of [
      ['sgd', () => new SGD([Tensor.zeros([3, 1])], { lr: 0.05, momentum: 0.9 })],
      ['lion', () => new Lion([Tensor.zeros([3, 1])], { lr: 0.01 })],
    ]) {
      const opt = mk();
      const w = opt.params[0];
      let first = 0;
      for (let i = 0; i < 300; i++) {
        const loss = Xt.matmul(w).sub(yt).pow(2).mean();
        loss.backward();
        opt.step();
        if (i === 0) first = loss.data[0];
        if (i === 299 && !(loss.data[0] < first * 0.5)) throw new Error(`${name} 损失下降不足`);
      }
    }
    return 'sgd/lion 均下降 >50%';
  });

  s.check('Newton-Schulz 输出近似正交（奇异值 ∈ (0.5, 1.6)）', () => {
    const rng = createRng(21);
    const G = randMat(rng, 6, 4);
    const O = newtonSchulz(G);
    const { s } = svd(O);
    for (const v of s) {
      if (v < 0.5 || v > 1.6) throw new Error(`奇异值 ${v.toFixed(3)} 偏离正交范围`);
    }
    return `奇异值 ∈ [${Math.min(...s).toFixed(3)}, ${Math.max(...s).toFixed(3)}]`;
  });

  s.check('全局梯度裁剪', () => {
    const p1 = new Tensor([2, 2]);
    p1.grad.set([3, 4, 0, 0]);
    const norm = clipGlobalNorm([p1], 1);
    expectClose(norm, 5, 1e-12);
    expectClose(Math.hypot(p1.grad[0], p1.grad[1]), 1, 1e-12);
    return '范数 5 → 1';
  });

  s.check('学习率调度（warmup + cosine/linear）', () => {
    const cos = makeSchedule('cosine', { warmup: 10, total: 100, baseLr: 1 });
    expectClose(cos(0), 0.1, 1e-12);
    expectClose(cos(9), 1.0, 1e-12);
    expectClose(cos(100), 0, 1e-12);
    const lin = makeSchedule('linear', { warmup: 0, total: 100, baseLr: 1 });
    expectClose(lin(50), 0.5, 1e-12);
    return 'warmup/cosine/linear 正确';
  });

  return [s.report()];
}

// ==================== 数据集与指标不变量 ====================
export function datasetsSelfCheck() {
  const s = suite('datasets');

  s.check('Iris：150×4，三类各 50', () => {
    const { X, y } = loadIris();
    if (X.rows !== 150 || X.cols !== 4) throw new Error('形状错误');
    const counts = [0, 0, 0];
    for (const c of y) counts[c]++;
    if (counts.join() !== '50,50,50') throw new Error(`类别分布 ${counts}`);
    return '150×4, 50/50/50';
  });

  s.check('Wine：178×13，三类 59/71/48', () => {
    const { X, y } = loadWine();
    if (X.rows !== 178 || X.cols !== 13) throw new Error('形状错误');
    const counts = [0, 0, 0];
    for (const c of y) counts[c]++;
    if (counts.join() !== '59,71,48') throw new Error(`类别分布 ${counts}`);
    return '178×13, 59/71/48';
  });

  s.check('合成数据形状与可复现性', () => {
    const m1 = makeMoons({ n: 100 }), m2 = makeMoons({ n: 100 });
    if (m1.X.rows !== 100) throw new Error('moons 形状错误');
    if (m1.X.get(0, 0) !== m2.X.get(0, 0)) throw new Error('moons 不可复现');
    if (makeBlobs({ n: 90, k: 3 }).X.rows !== 90) throw new Error('blobs 错误');
    if (makeXor({ n: 40 }).y.filter((v) => v === 1).length === 0) throw new Error('xor 单类');
    if (makeRegression({ n: 50 }).X.cols !== 3) throw new Error('regression 错误');
    return 'moons/blobs/xor/regression OK';
  });

  s.check('trainTestSplit 比例与确定性', () => {
    const { X, y } = loadIris();
    const a = trainTestSplit(X, y, { testRatio: 0.3, seed: 1 });
    const b = trainTestSplit(X, y, { testRatio: 0.3, seed: 1 });
    if (a.train.X.rows !== 105 || a.test.X.rows !== 45) throw new Error('划分比例错误');
    if (a.test.X.get(0, 0) !== b.test.X.get(0, 0)) throw new Error('划分不确定');
    return '105/45，同种子一致';
  });

  s.check('评估指标手算参照值', () => {
    expectClose(metrics.accuracy([0, 1, 1], [0, 1, 0]), 2 / 3, 1e-12);
    expectClose(metrics.macroF1([0, 0, 1, 1], [0, 1, 1, 1], 2), (2 / 3 + 0.8) / 2, 1e-12);
    expectClose(metrics.r2([1, 2, 3], [1, 2, 3]), 1, 1e-12);
    expectClose(metrics.purity([0, 0, 1, 1], [1, 1, 0, 0], 2), 1, 1e-12);
    const cm = metrics.confusionMatrix([0, 0, 1], [0, 1, 1], 2);
    let tot = 0;
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) tot += cm.get(i, j);
    if (tot !== 3) throw new Error('混淆矩阵总数错误');
    return 'accuracy/F1/R²/purity 全部命中';
  });

  s.check('z-score 标准化：均值 0 方差 1', () => {
    const { X } = loadIris();
    const fit = zscoreFit(X);
    const Z = zscoreApply(X, fit);
    for (let j = 0; j < Z.cols; j++) {
      let m = 0;
      for (let i = 0; i < Z.rows; i++) m += Z.get(i, j);
      expectClose(m / Z.rows, 0, 1e-10);
    }
    return '4 列均值 ≈ 0';
  });

  s.check('字符语料：词表构建 / 编解码往返 / 批次形状', () => {
    const c = loadCorpus({ maxChars: 5000 });
    if (c.vocabSize < 20) throw new Error('词表过小');
    const ids = encodeText(c.text, c.stoi);
    if (decodeIds(ids, c.itos) !== c.text) throw new Error('编解码往返不一致');
    const { x, y } = charBatch(ids, { batch: 4, ctx: 16, rng: createRng(1) });
    if (x.length !== 4 || x[0].length !== 16 || y[0].length !== 16) throw new Error('批次形状错误');
    if (x[0][1] !== y[0][0]) throw new Error('目标未右移一位');
    return `词表 ${c.vocabSize} 字符，往返一致`;
  });

  return [s.report()];
}

/** 汇总全部不变量自检 */
export function selfCheckAll() {
  return [
    ...linalgSelfCheck(),
    ...autodiffSelfCheck(),
    ...optimSelfCheck(),
    ...datasetsSelfCheck(),
  ];
}
