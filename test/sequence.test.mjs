/**
 * @file sequence.test.mjs — 序列模型模块测试（node:test）
 * 作者：晨星
 *
 * 运行：node --test test/
 * 不变量主体在 src/sequence.mjs 的 selfCheck()（单一事实源），
 * 本文件负责驱动自检，并补充接口契约与参照值测试。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { HiddenMarkovModel, beamSearch, selfCheck, bench } from '../src/sequence.mjs';

test('sequence 不变量全部通过', () => {
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

test('sequence 基准全部达标', () => {
  const { results } = bench();
  const bad = results.filter((r) => !r.ok);
  assert.equal(bad.length, 0, `未达标项:\n${bad.map((b) => `${b.name}: ${b.detail}`).join('\n')}`);
  console.log(`  基准: ${results.length}/${results.length} 达标`);
});

test('HMM：前向对数概率与暴力枚举一致（T≤6，容差 1e-10）', () => {
  const hmm = new HiddenMarkovModel({ nStates: 2, nObs: 3, seed: 1 });
  for (let t = 0; t < 20; t++) {
    const obs = Array.from({ length: 1 + (t % 6) }, () => Math.floor(Math.random() * 3));
    const f = hmm.forward(obs).logProb;
    const b = hmm.bruteForceLogProb(obs);
    assert.ok(Math.abs(f - b) < 1e-10, `T=${obs.length}: forward=${f} vs 暴力=${b}`);
  }
});

test('HMM：Viterbi 路径合法且概率不低于任何枚举路径', () => {
  const hmm = new HiddenMarkovModel({ nStates: 2, nObs: 3, seed: 5 });
  const obs = [0, 1, 2, 1, 0];
  const { path, logProb } = hmm.viterbi(obs);
  assert.equal(path.length, obs.length, '路径长度须等于观测长度');
  for (const s of path) assert.ok(s >= 0 && s < 2, '状态须合法');
  const best = hmm.bruteForceMaxPathLogProb(obs); // 返回最优对数概率（数字）
  assert.ok(logProb >= best - 1e-9, `Viterbi=${logProb} 应 ≥ 枚举最优=${best}`);
  // 交叉验证：Viterbi 概率应等于其自身路径的直接打分
  assert.ok(Math.abs(logProb - hmm._pathLogProb(obs, path)) < 1e-12, 'Viterbi 分数须与路径打分一致');
});

test('HMM：Baum-Welch 训练后转移矩阵保持行随机', () => {
  const hmm = new HiddenMarkovModel({ nStates: 3, nObs: 4, seed: 42 });
  hmm.fit([[0, 1, 2, 3, 0, 1], [2, 3, 0, 1, 2]], { maxIter: 30 });
  for (let i = 0; i < hmm.transMat_.rows; i++) {
    let s = 0;
    for (let j = 0; j < hmm.transMat_.cols; j++) s += hmm.transMat_.get(i, j);
    assert.ok(Math.abs(s - 1) < 1e-9, `第 ${i} 行和为 ${s}，应为 1`);
  }
  let sp = 0;
  for (const v of hmm.startProb_) sp += v;
  assert.ok(Math.abs(sp - 1) < 1e-9, `初始分布和为 ${sp}`);
});

test('束搜索：beamWidth=1 等价贪心；宽度足够时找到全局最优', () => {
  // 确定性格子：贪心会陷入局部次优（stepFn 接收全部 beams，返回每个 beam 的候选列表）
  const stepFn = (beams) => beams.map((b) => {
    if (b.tokens.length >= 3) return [{ token: 'E', logProb: 0 }]; // 显式结束
    if (b.tokens.length <= 1) {
      return [
        { token: 'a', logProb: Math.log(0.6), state: 'trap' },
        { token: 'b', logProb: Math.log(0.4), state: 'good' },
      ];
    }
    const isTrap = b.state === 'trap';
    return [
      { token: 'x', logProb: Math.log(isTrap ? 0.1 : 0.9), state: b.state },
      { token: 'y', logProb: Math.log(0.05), state: b.state },
    ];
  });
  // 以 endToken 显式结束，否则所有候选会在 maxLen 步后被清空
  const greedy = beamSearch({ stepFn, beamWidth: 1, maxLen: 4, startToken: null, endToken: 'E' });
  const wide = beamSearch({ stepFn, beamWidth: 8, maxLen: 4, startToken: null, endToken: 'E' });
  assert.ok(greedy.length > 0 && wide.length > 0, '应返回至少一个完整序列');
  const gs = greedy[0].tokens.filter((t) => t !== null).join('');
  const ws = wide[0].tokens.filter((t) => t !== null).join('');
  assert.equal(gs, 'axE', `贪心应落入陷阱，实际 ${gs}`);
  assert.equal(ws, 'bxE', `宽度足够应找到全局最优，实际 ${ws}`);
  assert.ok(wide[0].logProb > greedy[0].logProb, '全局最优概率须更高');
});

test('束搜索：返回序列按对数概率降序', () => {
  const stepFn = (beams) => beams.map((b) => {
    if (b.tokens.length >= 2) return [{ token: 'E', logProb: 0 }];
    return [
      { token: 'p', logProb: Math.log(0.5) },
      { token: 'q', logProb: Math.log(0.3) },
    ];
  });
  const out = beamSearch({ stepFn, beamWidth: 4, maxLen: 3, startToken: null, endToken: 'E' });
  assert.ok(out.length > 0, '应返回结果');
  for (let i = 1; i < out.length; i++) {
    assert.ok(out[i - 1].logProb >= out[i].logProb - 1e-12, '结果须按概率降序');
  }
});
