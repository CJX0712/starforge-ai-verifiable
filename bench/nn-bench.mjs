/**
 * @file bench/nn-bench.mjs — 神经网络与 MiniGPT 训练基准（独立可运行）
 * 作者：晨星
 *
 * 设计目标：在 CI 友好的时间内给出可复现的性能数字，并验证「反向传播正确性」
 * 这一最关键的不变量（过拟合单批 → loss ≈ 0）。
 *
 * 运行：
 *   node bench/nn-bench.mjs                 # tiny 预设（默认，< 3 min）
 *   node bench/nn-bench.mjs --preset small # 更大模型，更长训练
 *   node bench/nn-bench.mjs --json out.json# 将结果写入 JSON
 *
 * 引用：
 *   Karpathy, nanoGPT / minGPT — Transformer 的工程化参考实现
 *   Vaswani et al. 2017, Attention Is All You Need
 *   Jordan et al. 2024, Muon: An optimizer for hidden layers in neural networks
 */
import { writeFileSync } from 'node:fs';
import { MiniGPT, trainModel, countParameters, Linear, LayerNorm }
  from '../src/nn.mjs';
import { createRng } from '../src/linalg.mjs';
import { loadCorpus, encodeText, charBatch, loadIris, zscoreFit, zscoreApply, trainTestSplit, metrics }
  from '../src/datasets.mjs';
import { Tensor } from '../src/autodiff.mjs';

const PRESETS = {
  tiny: { dModel: 32, nLayer: 2, nHead: 4, ctx: 64, batch: 16, steps: 300, lr: 1e-3, optimizer: 'adamw' },
  small: { dModel: 48, nLayer: 2, nHead: 4, ctx: 64, batch: 16, steps: 800, lr: 1e-3, optimizer: 'adamw' },
};

function parseArgs(argv) {
  const a = { preset: 'tiny', json: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--preset') a.preset = argv[++i];
    else if (argv[i] === '--json') a.json = argv[++i];
    else if (argv[i] === '--steps') a.steps = parseInt(argv[++i], 10);
  }
  return a;
}

/** 计算困惑度（perplexity）= exp(cross-entropy) */
function perplexity(loss) { return Math.exp(loss); }

/** 实验 1：标准训练（tiny/small 预设） */
async function runTrain(cfg, corpusIds, stoi) {
  const model = new MiniGPT({
    vocabSize: corpusIds.vocabSize, dModel: cfg.dModel, nLayer: cfg.nLayer,
    nHead: cfg.nHead, ctx: cfg.ctx,
  });
  const nParams = countParameters(model);
  const rng = createRng(42);
  const { history, finalLoss, wallMs } = trainModel(model, {
    getBatch: () => charBatch(corpusIds.ids, { batch: cfg.batch, ctx: cfg.ctx, rng }),
    epochs: 1, stepsPerEpoch: cfg.steps, lr: cfg.lr, optimizer: cfg.optimizer,
    schedule: 'cosine', warmup: Math.min(50, cfg.steps >> 2), clip: 1.0, logEvery: 0,
  });
  const initialLoss = history.length ? history[0].loss : NaN;
  // 生成三个温度的采样样本
  const samples = [0.5, 0.8, 1.0].map((temperature) => {
    const prompt = encodeText('To be, or not to be', stoi);
    const out = model.generate({ prompt, maxNew: 120, temperature, rng: createRng(7) });
    return { temperature, text: 'To be, or not to be' + out.map((i) => corpusIds.itos[i]).join('') };
  });
  return {
    preset: cfg, nParams, initialLoss, finalLoss,
    perplexity: perplexity(finalLoss), wallMs, stepsPerSec: cfg.steps / (wallMs / 1000), samples,
  };
}

/** 实验 2：AdamW vs Muon 对比实验（各 200 步） */
async function runOptimizerCompare(corpusIds) {
  const mk = () => new MiniGPT({ vocabSize: corpusIds.vocabSize, dModel: 32, nLayer: 2, nHead: 4, ctx: 64 });
  const rngA = createRng(42);
  const rngM = createRng(42);
  const run = (opt, rng) => {
    const m = mk();
    const r = trainModel(m, {
      getBatch: () => charBatch(corpusIds.ids, { batch: 16, ctx: 64, rng }),
      epochs: 1, stepsPerEpoch: 200, lr: opt === 'muon' ? 2e-3 : 1e-3, optimizer: opt,
      schedule: 'const', clip: 1.0, logEvery: 0,
    });
    return r.finalLoss;
  };
  const adamw = run('adamw', rngA);
  const muon = run('muon', rngM);
  return { adamwFinalLoss: adamw, muonFinalLoss: muon, muonBetter: muon < adamw };
}

/** 实验 3：过拟合单批（反向传播正确性最强证据） */
async function runOverfit(corpusIds) {
  const model = new MiniGPT({ vocabSize: corpusIds.vocabSize, dModel: 32, nLayer: 1, nHead: 4, ctx: 32 });
  const batch = [corpusIds.ids.slice(0, 32)];
  const { finalLoss } = trainModel(model, {
    getBatch: () => ({ x: batch, y: batch }),
    epochs: 1, stepsPerEpoch: 300, lr: 0.05, optimizer: 'adamw',
    schedule: 'const', clip: 1.0, logEvery: 0,
  });
  return { overfitFinalLoss: finalLoss, ok: finalLoss < 0.05 };
}

/** 实验 4：Iris MLP 分类（softmax 回归 + 隐藏层） */
async function runIrisMlp() {
  const { X, y } = loadIris();
  const Z = zscoreApply(X, zscoreFit(X));
  const { train, test } = trainTestSplit(Z, y, { testRatio: 0.3, seed: 42 });

  // 用 nn.mjs 的组件搭一个 MLP(4→16→3)
  const rng = createRng(42);
  const l1 = new Linear(4, 16, { rng });
  const l2 = new Linear(16, 3, { rng });
  const params = [...l1.parameters(), ...l2.parameters()];
  const forward = (x) => {
    const h = l1.forward(x).relu();
    return l2.forward(h);
  };
  const { createOptimizer } = await import('../src/optim.mjs');
  const opt = createOptimizer('adamw', params, { lr: 1e-2 });
  let lastAcc = 0;
  const trainX = train.X.to2D();
  const testX = test.X.to2D();
  for (let s = 0; s < 400; s++) {
    const logits = forward(trainX);
    const loss = logits.crossEntropyLogits(train.y);
    loss.backward();
    opt.lr = 1e-2;
    opt.step();
    // 评估
    const pred = argmaxRows(forward(testX), 3);
    lastAcc = metrics.accuracy(test.y, pred);
  }
  return { irisMlpAccuracy: lastAcc, ok: lastAcc >= 0.95 };
}

function argmaxRows(m, k) {
  const out = new Array(m.rows);
  for (let i = 0; i < m.rows; i++) {
    let best = 0; let bv = -Infinity;
    for (let j = 0; j < k; j++) { const v = m.get(i, j); if (v > bv) { bv = v; best = j; } }
    out[i] = best;
  }
  return out;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const cfg = { ...(PRESETS[args.preset] || PRESETS.tiny) };
  if (args.steps) cfg.steps = args.steps;
  console.log(`[nn-bench] 预设 ${args.preset} | dModel=${cfg.dModel} nLayer=${cfg.nLayer} steps=${cfg.steps}`);

  const corpus = loadCorpus();
  const ids = encodeText(corpus.text, corpus.stoi);
  const corpusIds = { ids, vocabSize: corpus.vocabSize, itos: corpus.itos };

  console.log('[nn-bench] 加载语料：', ids.length, '字符，词表', corpusIds.vocabSize);

  const trainRes = await runTrain(cfg, corpusIds, corpus.stoi);
  console.log(`[nn-bench] 训练: 初始 loss=${trainRes.initialLoss.toFixed(4)} → 最终 ${trainRes.finalLoss.toFixed(4)} | 困惑度 ${trainRes.perplexity.toFixed(2)} | ${trainRes.wallMs.toFixed(0)}ms | ${trainRes.stepsPerSec.toFixed(1)} step/s`);
  for (const s of trainRes.samples) console.log(`  T=${s.temperature}: ${s.text.slice(0, 60)}...`);

  const optCmp = await runOptimizerCompare(corpusIds);
  console.log(`[nn-bench] 优化器对比(200步): AdamW=${optCmp.adamwFinalLoss.toFixed(4)} vs Muon=${optCmp.muonFinalLoss.toFixed(4)} | Muon 更优=${optCmp.muonBetter}`);

  const overfit = await runOverfit(corpusIds);
  console.log(`[nn-bench] 过拟合单批: loss=${overfit.overfitFinalLoss.toExponential(2)} | ok=${overfit.ok}`);

  const iris = await runIrisMlp();
  console.log(`[nn-bench] Iris MLP(4→16→3): 测试 acc=${iris.irisMlpAccuracy.toFixed(4)} | ok=${iris.ok}`);

  const result = {
    module: 'nn',
    preset: args.preset,
    config: cfg,
    train: trainRes,
    optimizerCompare: optCmp,
    overfit: overfit,
    irisMlp: iris,
    ok: trainRes.finalLoss < 4 && overfit.ok && iris.ok,
  };

  if (args.json) {
    writeFileSync(args.json, JSON.stringify(result, null, 2));
    console.log(`[nn-bench] 结果已写入 ${args.json}`);
  }
  return result;
}

main().then((r) => { process.exit(r.ok ? 0 : 1); }).catch((e) => { console.error(e); process.exit(1); });
