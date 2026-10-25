/**
 * @file rl.test.mjs — 强化学习模块测试（node:test）
 * 作者：晨星
 *
 * 运行：node --test test/
 * 不变量主体在 src/rl.mjs 的 selfCheck()（单一事实源），
 * 本文件负责驱动自检，并补充遗憾序、物理动力学与收敛性的交叉验证。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  makeTenArmMeans, BernoulliBandit, EpsilonGreedy, UCB1, KLUCB, ThompsonSampling,
  QLearningAgent, REINFORCEAgent, CartPoleEnv, ChainMDP, GridWorldEnv,
  CARTPOLE_BOUNDS, cartPoleBinIndex, cartPoleStateCount,
  runBanditAgent, trainQLearningChain, trainREINFORCEBandit, trainQonCartPole,
  bernoulliKL, klUCBIndex, selfCheck, bench,
} from '../src/rl.mjs';

test('rl 不变量全部通过', () => {
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

test('rl 基准全部达标', () => {
  const { results } = bench();
  const bad = results.filter((r) => !r.ok);
  assert.equal(bad.length, 0, `未达标项:\n${bad.map((b) => `${b.name}: ${b.detail}`).join('\n')}`);
  console.log(`  基准: ${results.length}/${results.length} 达标`);
});

test('Bernoulli 老虎机：奖励服从给定均值（大数定律）', () => {
  const means = [0.2, 0.8];
  const env = new BernoulliBandit({ means, seed: 42 });
  const N = 20000;
  let s = 0;
  for (let i = 0; i < N; i++) s += env.step(0);
  assert.ok(Math.abs(s / N - 0.2) < 0.02, `经验均值 ${(s / N).toFixed(4)} 应 ≈ 0.2`);
});

test('老虎机：遗憾定义恒等式成立且非负', () => {
  const means = makeTenArmMeans();
  const env = new BernoulliBandit({ means, seed: 3 });
  let sumMeans = 0;
  for (let t = 0; t < 1000; t++) {
    const a = t % means.length;
    env.step(a);
    sumMeans += means[a];
  }
  const expected = env.steps * env.optimalMean - sumMeans;
  assert.ok(Math.abs(env.regret - expected) < 1e-9, '遗憾恒等式不成立');
  assert.ok(env.regret >= 0, '遗憾须非负');
});

test('UCB1：未试臂优先被选择', () => {
  const a = new UCB1({ seed: 1 });
  a.update(0, 1);
  const counts = [1, 0, 0];
  assert.equal(a.selectAction([0.1, 0.9, 0.5], counts), 1, '应先选未试过的臂 1');
});

test('KL-UCB：指数 ∈ [q̂,1] 且随 N 增大而下降（探索收紧）', () => {
  const u1 = klUCBIndex(0.5, 5, 1000, 0);
  const u2 = klUCBIndex(0.5, 500, 1000, 0);
  assert.ok(u1 >= 0.5 && u1 <= 1, '指数须在 [q̂,1]');
  assert.ok(u2 >= 0.5 && u2 <= 1, '指数须在 [q̂,1]');
  assert.ok(u2 < u1, `样本越多上界越紧：N=500 (${u2.toFixed(4)}) 应 < N=5 (${u1.toFixed(4)})`);
  console.log(`  KL-UCB 收紧: N=5 → ${u1.toFixed(4)}, N=500 → ${u2.toFixed(4)}`);
});

test('Bernoulli KL 散度：非负且 p=q 时为 0', () => {
  assert.ok(Math.abs(bernoulliKL(0.3, 0.3)) < 1e-15, 'p=q 时 KL 须为 0');
  for (const [p, q] of [[0.2, 0.7], [0.9, 0.1], [0.5, 0.51]]) {
    assert.ok(bernoulliKL(p, q) > 0, `KL(${p},${q}) 须为正`);
  }
});

test('QLearningAgent：确定性格子上收敛到解析最优 Q', () => {
  // 3 状态链：s0 --(+1)→ s1 --(+1)→ s2(终止)；gamma=0.9
  const gamma = 0.9;
  const agent = new QLearningAgent({ nStates: 3, nActions: 1, alpha: 0.5, gamma, epsilon: 0, seed: 1 });
  for (let ep = 0; ep < 500; ep++) {
    for (let s = 0; s < 2; s++) agent.learn(s, 0, 1, s + 1, s + 1 === 2);
  }
  // V(s2)=0, V(s1)=1, V(s0)=1+0.9*1=1.9
  const q1 = agent.q(1, 0);
  const q0 = agent.q(0, 0);
  assert.ok(Math.abs(q1 - 1) < 1e-3, `Q(s1,a) 应 ≈1，实际 ${q1}`);
  assert.ok(Math.abs(q0 - (1 + gamma * 1)) < 1e-3, `Q(s0,a) 应 ≈1.9，实际 ${q0}`);
});

test('CartPole 物理：初始状态为零，恒力必倒且状态有界', () => {
  const env = new CartPoleEnv({ seed: 1 });
  const s0 = env.reset();
  for (const v of s0) assert.equal(v, 0, 'reset 后状态须全零');
  for (const force of [0, 1]) {
    const e = new CartPoleEnv({ seed: 7 });
    e.reset();
    let steps = 0;
    let done = false;
    while (!done && steps < 200) {
      const r = e.step(force);
      done = r.done;
      steps += 1;
      for (const v of r.state) assert.ok(Number.isFinite(v), '状态须有限');
    }
    assert.ok(done, `恒定动作 ${force} 应在 200 步内终止`);
  }
});

test('确定性：同 seed 下 Bandit 与 CartPole 轨迹逐位一致', () => {
  const means = makeTenArmMeans();
  const a = runBanditAgent(new ThompsonSampling({ seed: 42 }), { means, steps: 300, seed: 42 });
  const b = runBanditAgent(new ThompsonSampling({ seed: 42 }), { means, steps: 300, seed: 42 });
  assert.equal(a.regret, b.regret);
  assert.deepEqual(a.actions, b.actions);

  const run = () => {
    const e = new CartPoleEnv({ seed: 5 });
    e.reset();
    const traj = [];
    for (let t = 0; t < 50; t++) traj.push(...e.step(t % 2).state);
    return traj;
  };
  assert.deepEqual(run(), run(), 'CartPole 轨迹须可复现');
});
