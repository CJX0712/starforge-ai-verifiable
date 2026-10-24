/**
 * @file train-gpt.mjs — MiniGPT 训练入口（CLI train-gpt 子命令）
 * 作者：晨星
 *
 * 从零训练一个字符级迷你 GPT 并导出权重（web/weights.json，供 Web 仪表盘推理）。
 * 训练目标：在莎士比亚 + 爱丽丝梦游仙境混合语料上最小化交叉熵。
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRng } from './linalg.mjs';
import { loadCorpus, encodeText, charBatch } from './datasets.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const PRESETS = {
  tiny: { dModel: 32, nLayer: 2, nHead: 4, ctx: 64, batch: 16, steps: 300, lr: 1e-3, optimizer: 'adamw' },
  small: { dModel: 48, nLayer: 2, nHead: 4, ctx: 64, batch: 16, steps: 800, lr: 1e-3, optimizer: 'adamw' },
};

export async function trainGptCli({ preset = 'tiny', steps, root: rootDir = root } = {}) {
  const { MiniGPT, trainModel, saveModel, countParameters } = await import('./nn.mjs');
  const cfg = { ...(PRESETS[preset] || PRESETS.tiny) };
  if (steps) cfg.steps = steps;

  const corpus = loadCorpus();
  const ids = encodeText(corpus.text, corpus.stoi);
  const model = new MiniGPT({
    vocabSize: corpus.vocabSize,
    dModel: cfg.dModel,
    nLayer: cfg.nLayer,
    nHead: cfg.nHead,
    ctx: cfg.ctx,
  });
  const nParams = countParameters(model);
  console.log(`词表 ${corpus.vocabSize} 字符 | 参数量 ${nParams.toLocaleString()} | 语料 ${ids.length.toLocaleString()} 字符`);
  console.log(`配置: dModel=${cfg.dModel}, nLayer=${cfg.nLayer}, nHead=${cfg.nHead}, ctx=${cfg.ctx}, batch=${cfg.batch}, steps=${cfg.steps}, ${cfg.optimizer}`);

  const rng = createRng(42);
  const t0 = Date.now();
  const { history, finalLoss } = trainModel(model, {
    getBatch: () => charBatch(ids, { batch: cfg.batch, ctx: cfg.ctx, rng }),
    epochs: 1,
    stepsPerEpoch: cfg.steps,
    lr: cfg.lr,
    optimizer: cfg.optimizer,
    schedule: 'cosine',
    warmup: Math.min(50, cfg.steps >> 2),
    clip: 1.0,
    logEvery: 50,
  });
  const wall = ((Date.now() - t0) / 1000).toFixed(1);
  const initialLoss = history.length ? history[0].loss : finalLoss;

  const samples = [0.5, 0.8, 1.0].map((temperature) => {
    const prompt = encodeText('To be, or not to be', corpus.stoi);
    const out = model.generate({ prompt, maxNew: 200, temperature, rng: createRng(7) });
    return { temperature, text: 'To be, or not to be' + out.map((i) => corpus.itos[i]).join('') };
  });

  const data = saveModel(model);
  const outPath = join(rootDir, 'web', 'weights.json');
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify(data));

  console.log(`\n训练完成：${cfg.steps} 步 | loss ${initialLoss.toFixed(4)} → ${finalLoss.toFixed(4)} | 困惑度 ${Math.exp(finalLoss).toFixed(2)} | 耗时 ${wall}s | ${(Number(wall) / cfg.steps * 1000).toFixed(0)}ms/步`);
  console.log(`权重已写入 ${outPath}（${(JSON.stringify(data).length / 1024).toFixed(0)} KB）`);
  for (const s of samples) console.log(`\n--- 温度 ${s.temperature} ---\n${s.text}`);
  return { finalLoss, wall, samples, outPath, nParams, config: cfg };
}
