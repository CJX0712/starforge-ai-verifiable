/**
 * @file datasets.mjs — 数据集加载、合成数据、评估指标、字符级语料
 * 作者：晨星
 *
 * 数据不变量：
 * - Iris：150 行 × 4 特征，三类各 50 条（Fisher 1936 的经典设定）
 * - Wine：178 行 × 13 特征，三类分别 59/71/48 条
 * - 合成数据（moons/blobs/XOR）由注入的种子完全决定 → 可复现
 */
import { Mat, createRng } from './linalg.mjs';
import { IRIS_ROWS } from '../data/iris.mjs';
import { WINE_ROWS } from '../data/wine.mjs';
import { CORPUS_TEXT } from '../data/corpus.mjs';

const IRIS_CLASSES = { 'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2 };

/** 鸢尾花数据集：{X: Mat(150×4), y: number[150] ∈ {0,1,2}} */
export function loadIris() {
  return {
    X: Mat.from2D(IRIS_ROWS.map((r) => r.slice(0, 4))),
    y: IRIS_ROWS.map((r) => IRIS_CLASSES[r[4]]),
    featureNames: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    targetNames: ['setosa', 'versicolor', 'virginica'],
  };
}

/** 葡萄酒数据集：{X: Mat(178×13), y: number[178] ∈ {0,1,2}} */
export function loadWine() {
  return {
    X: Mat.from2D(WINE_ROWS.map((r) => r.slice(1))),
    y: WINE_ROWS.map((r) => r[0] - 1),
    featureNames: [
      'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
      'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue',
      'od280_od315_of_diluted_wines', 'proline',
    ],
    targetNames: ['class_1', 'class_2', 'class_3'],
  };
}

/** 双月牙数据集（sklearn make_moons 同分布公式） */
export function makeMoons({ n = 200, noise = 0.2, rng = createRng(42) } = {}) {
  const nOut = Math.floor(n / 2);
  const nIn = n - nOut;
  const X = Mat.zeros(n, 2);
  const y = new Array(n);
  for (let i = 0; i < nOut; i++) {
    const t = (Math.PI * i) / nOut;
    X.set(i, 0, Math.cos(t));
    X.set(i, 1, Math.sin(t));
    y[i] = 0;
  }
  for (let i = 0; i < nIn; i++) {
    const t = (Math.PI * i) / nIn;
    X.set(nOut + i, 0, 1 - Math.cos(t));
    X.set(nOut + i, 1, 0.5 - Math.sin(t));
    y[nOut + i] = 1;
  }
  for (let i = 0; i < n; i++) {
    X.set(i, 0, X.get(i, 0) + rng.normal() * noise);
    X.set(i, 1, X.get(i, 1) + rng.normal() * noise);
  }
  return { X, y };
}

/** 高斯团簇数据集 */
export function makeBlobs({ n = 300, k = 3, std = 0.8, spread = 6, rng = createRng(42) } = {}) {
  const X = Mat.zeros(n, 2);
  const y = new Array(n);
  const centers = [];
  for (let c = 0; c < k; c++) {
    const ang = (2 * Math.PI * c) / k;
    centers.push([spread * Math.cos(ang), spread * Math.sin(ang)]);
  }
  for (let i = 0; i < n; i++) {
    const c = i % k;
    X.set(i, 0, centers[c][0] + rng.normal() * std);
    X.set(i, 1, centers[c][1] + rng.normal() * std);
    y[i] = c;
  }
  return { X, y };
}

/** XOR 异或数据集（线性不可分的经典基准） */
export function makeXor({ n = 200, noise = 0.25, rng = createRng(42) } = {}) {
  const X = Mat.zeros(n, 2);
  const y = new Array(n);
  for (let i = 0; i < n; i++) {
    const q = i % 4;
    const sx = q < 2 ? 1 : -1;
    const sy = q % 2 === 0 ? 1 : -1;
    X.set(i, 0, sx + rng.normal() * noise);
    X.set(i, 1, sy + rng.normal() * noise);
    y[i] = sx * sy > 0 ? 1 : 0;
  }
  return { X, y };
}

/** 带交互项的回归数据：y = 3x₀ − 2x₁ + 1.5x₀x₁ + 0.5x₂ + ε */
export function makeRegression({ n = 120, noise = 0.15, rng = createRng(42) } = {}) {
  const X = Mat.zeros(n, 3);
  const y = new Array(n);
  for (let i = 0; i < n; i++) {
    const x0 = rng.normal(), x1 = rng.normal(), x2 = rng.normal();
    X.set(i, 0, x0); X.set(i, 1, x1); X.set(i, 2, x2);
    y[i] = 3 * x0 - 2 * x1 + 1.5 * x0 * x1 + 0.5 * x2 + rng.normal() * noise;
  }
  return { X, y };
}

/** 按种子可复现的 train/test 划分 */
export function trainTestSplit(X, y, { testRatio = 0.3, seed = 42 } = {}) {
  const rng = createRng(seed);
  const idx = rng.shuffle(Array.from({ length: X.rows }, (_, i) => i));
  const nTest = Math.round(X.rows * testRatio);
  const testIdx = idx.slice(0, nTest);
  const trainIdx = idx.slice(nTest);
  const sub = (indices) => {
    const M = new Mat(indices.length, X.cols);
    const yy = new Array(indices.length);
    indices.forEach((src, dst) => {
      for (let j = 0; j < X.cols; j++) M.set(dst, j, X.get(src, j));
      yy[dst] = y[src];
    });
    return { X: M, y: yy };
  };
  return { train: sub(trainIdx), test: sub(testIdx) };
}

/** z-score 标准化（按列） */
export function zscoreFit(X) {
  const mean = new Float64Array(X.cols);
  const std = new Float64Array(X.cols);
  for (let j = 0; j < X.cols; j++) {
    let s = 0;
    for (let i = 0; i < X.rows; i++) s += X.get(i, j);
    mean[j] = s / X.rows;
    let v = 0;
    for (let i = 0; i < X.rows; i++) v += (X.get(i, j) - mean[j]) ** 2;
    std[j] = Math.sqrt(v / X.rows) || 1;
  }
  return { mean, std };
}

export function zscoreApply(X, fit) {
  const out = new Mat(X.rows, X.cols);
  for (let i = 0; i < X.rows; i++)
    for (let j = 0; j < X.cols; j++) out.set(i, j, (X.get(i, j) - fit.mean[j]) / fit.std[j]);
  return out;
}

/** 评估指标集合 */
export const metrics = {
  accuracy(yTrue, yPred) {
    let ok = 0;
    for (let i = 0; i < yTrue.length; i++) if (yTrue[i] === yPred[i]) ok++;
    return ok / yTrue.length;
  },
  confusionMatrix(yTrue, yPred, k) {
    const M = new Mat(k, k);
    for (let i = 0; i < yTrue.length; i++) M.set(yTrue[i], yPred[i], M.get(yTrue[i], yPred[i]) + 1);
    return M;
  },
  /** 宏平均 F1（各类 F1 的算术平均，零除记 0） */
  macroF1(yTrue, yPred, k) {
    let sum = 0;
    for (let c = 0; c < k; c++) {
      let tp = 0, fp = 0, fn = 0;
      for (let i = 0; i < yTrue.length; i++) {
        if (yPred[i] === c && yTrue[i] === c) tp++;
        else if (yPred[i] === c) fp++;
        else if (yTrue[i] === c) fn++;
      }
      const prec = tp + fp === 0 ? 0 : tp / (tp + fp);
      const rec = tp + fn === 0 ? 0 : tp / (tp + fn);
      sum += prec + rec === 0 ? 0 : (2 * prec * rec) / (prec + rec);
    }
    return sum / k;
  },
  mse(yTrue, yPred) {
    let s = 0;
    for (let i = 0; i < yTrue.length; i++) s += (yTrue[i] - yPred[i]) ** 2;
    return s / yTrue.length;
  },
  rmse(yTrue, yPred) { return Math.sqrt(this.mse(yTrue, yPred)); },
  /** 决定系数 R²：1 = 完美预测，0 = 与均值预测持平 */
  r2(yTrue, yPred) {
    const mu = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
    let ssRes = 0, ssTot = 0;
    for (let i = 0; i < yTrue.length; i++) {
      ssRes += (yTrue[i] - yPred[i]) ** 2;
      ssTot += (yTrue[i] - mu) ** 2;
    }
    return 1 - ssRes / ssTot;
  },
  /** 聚类纯度：多数类一致比例，∈ [1/k, 1] */
  purity(yTrue, yCluster, k) {
    const M = this.confusionMatrix(yTrue, yCluster, k);
    let ok = 0;
    for (let j = 0; j < k; j++) {
      let best = 0;
      for (let i = 0; i < k; i++) best = Math.max(best, M.get(i, j));
      ok += best;
    }
    return ok / yTrue.length;
  },
};

// ---------- 字符级语料 ----------

const _corpusCache = new Map();

/** 加载字符级语料并构建词表；词表按字符排序保证确定性 */
export function loadCorpus({ maxChars } = {}) {
  const key = maxChars ?? 'full';
  if (_corpusCache.has(key)) return _corpusCache.get(key);
  const text = maxChars ? CORPUS_TEXT.slice(0, maxChars) : CORPUS_TEXT;
  const chars = Array.from(new Set(text)).sort();
  const stoi = new Map(chars.map((c, i) => [c, i]));
  const corpus = { text, vocab: chars, vocabSize: chars.length, stoi, itos: chars };
  _corpusCache.set(key, corpus);
  return corpus;
}

export function encodeText(text, stoi) {
  const out = [];
  for (const ch of text) {
    const id = stoi.get(ch);
    if (id !== undefined) out.push(id);
  }
  return out;
}

export function decodeIds(ids, itos) {
  return ids.map((i) => itos[i]).join('');
}

/**
 * 字符级批次采样：随机起点切出 ctx 长度的输入与右移一位的目标。
 * 返回 {x: number[][] (batch×ctx), y: number[][]}。
 */
export function charBatch(ids, { batch = 32, ctx = 64, rng = createRng(42) } = {}) {
  const x = [];
  const y = [];
  const maxStart = ids.length - ctx - 1;
  for (let b = 0; b < batch; b++) {
    const ix = rng.randint(maxStart);
    x.push(ids.slice(ix, ix + ctx));
    y.push(ids.slice(ix + 1, ix + ctx + 1));
  }
  return { x, y };
}
