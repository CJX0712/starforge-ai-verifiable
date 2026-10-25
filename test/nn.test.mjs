/**
 * @file nn.test.mjs — 神经网络与 MiniGPT 模块测试（node:test）
 * 作者：晨星
 *
 * 运行：node --test test/
 * 不变量主体在 src/nn.mjs 的 selfCheck()（单一事实源），
 * 本文件负责驱动自检，并补充架构契约、过拟合能力与权重序列化测试。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  Module, Linear, LayerNorm, Dropout, Embedding, MultiHeadAttention,
  MLP, TransformerBlock, MiniGPT, trainModel, countParameters,
  encodeWeightsBase64, decodeWeightsBase64, saveModel, loadModel, selfCheck,
} from '../src/nn.mjs';
import { Tensor } from '../src/autodiff.mjs';
import { createRng } from '../src/linalg.mjs';

test('nn 不变量全部通过', () => {
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

test('架构契约：所有 Module 子类可枚举参数并清零梯度', () => {
  const rng = createRng(42);
  const mods = [
    new Linear(6, 4, { rng }),
    new LayerNorm(6),
    new Dropout(0.2, { rng }),
    new Embedding(10, 6, { rng }),
    new MLP(6, 12, { rng }),
    new MultiHeadAttention(8, 2, { rng }),
    new TransformerBlock(8, 2, { rng }),
    new MiniGPT({ vocabSize: 20, dModel: 8, nLayer: 1, nHead: 2, ctx: 16, rng }),
  ];
  for (const m of mods) {
    assert.ok(m instanceof Module, '须继承 Module');
    const params = m.parameters();
    assert.ok(Array.isArray(params), 'parameters() 须返回数组');
    for (const p of params) assert.ok(p instanceof Tensor, '参数须为 Tensor');
    // Dropout 无可训练参数（它只做随机置零），其余模块都必须有参数
    if (!(m instanceof Dropout)) assert.ok(params.length > 0, `${m.constructor.name} 须有可训练参数`);
    m.zeroGrad();
    for (const p of params) assert.ok(p.grad.every((v) => v === 0), '梯度须清零');
  }
});

test('架构契约：train()/eval() 切换影响 Dropout 行为', () => {
  const rng = createRng(1);
  const d = new Dropout(0.5, { rng });
  const x = new Tensor([8, 8], new Float64Array(64).fill(1));
  d.eval();
  const ye = Array.from(d.forward(x).data);
  assert.ok(ye.every((v) => v === 1), 'eval 模式须恒等');
  d.train();
  let diff = false;
  for (let i = 0; i < 20; i++) {
    const yt = Array.from(d.forward(x).data);
    if (yt.some((v) => v !== 1)) { diff = true; break; }
  }
  assert.ok(diff, 'train 模式须产生随机置零/缩放');
});

test('线性层输出形状与仿射变换正确性', () => {
  const rng = createRng(3);
  const lin = new Linear(3, 2, { rng });
  const x = new Tensor([4, 3], Float64Array.from({ length: 12 }, (_, i) => i * 0.1));
  const y = lin.forward(x);
  assert.deepEqual(y.shape, [4, 2]);
  // 手工核对第一行：y = x·W + b
  for (let j = 0; j < 2; j++) {
    let s = lin.b.data[j];
    for (let k = 0; k < 3; k++) s += x.data[k] * lin.W.data[k * 2 + j];
    assert.ok(Math.abs(y.data[j] - s) < 1e-12, '线性层输出与手工计算不符');
  }
});

test('LayerNorm：归一化后逐行均值≈0、方差≈1', () => {
  const rng = createRng(11);
  const ln = new LayerNorm(5);
  const x = new Tensor([6, 5], Float64Array.from({ length: 30 }, () => rng.normal() * 3 + 2));
  const y = ln.forward(x);
  for (let i = 0; i < 6; i++) {
    let m = 0;
    for (let j = 0; j < 5; j++) m += y.data[i * 5 + j];
    m /= 5;
    let v = 0;
    for (let j = 0; j < 5; j++) v += (y.data[i * 5 + j] - m) ** 2;
    v /= 5;
    assert.ok(Math.abs(m) < 1e-6, `第 ${i} 行均值 ${m} 应 ≈0`);
    assert.ok(Math.abs(v - 1) < 1e-5, `第 ${i} 行方差 ${v} 应 ≈1`);
  }
});

test('Dropout：训练模式随机置零，评估模式恒等', () => {
  const rng = createRng(5);
  const d = new Dropout(0.5, { rng });
  const x = new Tensor([4, 4], new Float64Array(16).fill(1));
  d.train();
  const yt = d.forward(x);
  let zeros = 0;
  for (const v of yt.data) if (v === 0) zeros++;
  assert.ok(zeros > 0, '训练模式应有置零');
  d.eval();
  const ye = d.forward(x);
  for (const v of ye.data) assert.equal(v, 1, '评估模式须恒等');
});

test('MiniGPT：因果掩码保证未来 token 不影响过去位置的 logits', () => {
  const rng = createRng(42);
  const g = new MiniGPT({ vocabSize: 12, dModel: 8, nLayer: 2, nHead: 2, ctx: 16, rng });
  g.eval();
  const base = [1, 2, 3, 4, 5];
  const a = g.forward([base]).data;
  const perturbed = [1, 2, 3, 4, 5, 9, 9, 9]; // 仅在尾部追加未来 token
  const b = g.forward([perturbed]).data;
  for (let i = 0; i < base.length; i++) {
    for (let v = 0; v < 12; v++) {
      assert.ok(Math.abs(a[i * 12 + v] - b[i * 12 + v]) < 1e-12,
        `第 ${i} 位置受未来 token 影响`);
    }
  }
});

test('过拟合能力：单批小样本训练后损失趋近 0（反向传播正确性的最强证据）', () => {
  const rng = createRng(42);
  const g = new MiniGPT({ vocabSize: 10, dModel: 16, nLayer: 1, nHead: 2, ctx: 8, rng });
  const batch = [[1, 2, 3, 4, 5, 6, 7, 8]];
  const r = trainModel(g, {
    getBatch: () => ({ x: batch, y: batch }),
    epochs: 1, stepsPerEpoch: 300, lr: 0.05, optimizer: 'adamw',
    schedule: 'const', logEvery: 1000,
  });
  assert.ok(r.finalLoss < 0.05, `过拟合后损失应 <0.05，实际 ${r.finalLoss}`);
  console.log(`  过拟合: 初始 ${r.history[0].loss.toFixed(4)} → 最终 ${r.finalLoss.toFixed(6)}`);
});

test('参数量：MiniGPT 参数量等于各子模块参数之和，且同配置可复现', () => {
  const V = 10; const D = 8; const L = 2; const ctx = 12;
  const g = new MiniGPT({ vocabSize: V, dModel: D, nLayer: L, nHead: 2, ctx, rng: createRng(0) });
  const have = countParameters(g);

  // 逐子模块累加：与 parameters() 的枚举结果交叉验证
  let sum = 0;
  for (const p of g.parameters()) sum += p.size;
  assert.equal(have, sum, '参数量须等于枚举参数的规模之和');

  // 理论下界：至少包含 token 嵌入 + 位置嵌入 + 输出头
  const lower = V * D + ctx * D + (D * V + V);
  assert.ok(have > lower, `参数量 ${have} 应大于嵌入与输出头合计 ${lower}`);

  const g2 = new MiniGPT({ vocabSize: V, dModel: D, nLayer: L, nHead: 2, ctx, rng: createRng(0) });
  assert.equal(countParameters(g2), have, '同配置模型参数量须一致');
  console.log(`  参数量: ${have}（嵌入+输出头下界 ${lower}）`);
});

test('权重序列化：base64 编解码往返一致', () => {
  const rng = createRng(9);
  const g = new MiniGPT({ vocabSize: 8, dModel: 8, nLayer: 1, nHead: 2, ctx: 8, rng });
  const snap = saveModel(g);
  const g2 = loadModel(snap);
  const a = g.parameters().map((p) => Array.from(p.data));
  const b = g2.parameters().map((p) => Array.from(p.data));
  assert.equal(a.length, b.length, '参数组数须一致');
  for (let i = 0; i < a.length; i++) assert.deepEqual(a[i], b[i], `第 ${i} 组参数不一致`);
});

test('纯 JS base64：随机数据编解码往返一致', () => {
  const rng = createRng(4);
  for (const n of [0, 1, 2, 3, 7, 100]) {
    const src = new Float64Array(n);
    for (let i = 0; i < n; i++) src[i] = rng.normal() * 10;
    const back = decodeWeightsBase64(encodeWeightsBase64(src));
    assert.equal(back.length, n);
    for (let i = 0; i < n; i++) assert.equal(back[i], src[i], `第 ${i} 个元素不一致`);
  }
});
