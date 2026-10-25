/**
 * @file agent.test.mjs — 认知智能体模块测试（node:test）
 * 作者：晨星
 *
 * 运行：node --test test/
 * 覆盖：任务规划与配方匹配、工具注册表、验证证书、记忆、确定性。
 * 智能体采用神经-符号架构：规划（符号）→ 执行（神经/数值）→ 验证（不变量）。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  safeEval, AgentMemory, ToolRegistry, Verifier, certificateText,
  StarAgent, createDefaultAgent,
} from '../src/agent.mjs';
import { Mat } from '../src/linalg.mjs';

/** 构造 2×3 的标准正交行（单位矩阵的前两行），用于 PCA 证书的正例 */
function buildOrthonormalRows() {
  const m = new Mat(2, 3);
  m.set(0, 0, 1);
  m.set(1, 1, 1);
  return m;
}

/* ---------------- safeEval：沙箱式表达式求值（无 eval，白名单） ---------------- */

test('safeEval：基本算术与运算符优先级', () => {
  assert.equal(safeEval('1 + 2 * 3'), 7, '乘法优先于加法');
  assert.equal(safeEval('(1 + 2) * 3'), 9, '括号改变优先级');
  assert.equal(safeEval('10 / 4'), 2.5);
  assert.equal(safeEval('7 % 3'), 1);
});

test('safeEval：拒绝危险输入（代码注入 / 全局访问）', () => {
  for (const bad of ['process.exit(1)', 'require("fs")', 'globalThis', 'a; b']) {
    assert.throws(() => safeEval(bad), /不允许|非法|拒绝|不支持|错误|安全|未知/i, `应拒绝: ${bad}`);
  }
  // 危险输入绝不能返回数字结果
  for (const bad of ['constructor', 'prototype', '__proto__']) {
    let returned = true;
    try { safeEval(bad); } catch { returned = false; }
    assert.equal(returned, false, `${bad} 必须抛错而非返回结果`);
  }
});

test('safeEval：数学函数与常量', () => {
  assert.ok(Math.abs(safeEval('sqrt(16)') - 4) < 1e-12);
  assert.ok(Math.abs(safeEval('log(exp(2))') - 2) < 1e-12);
});

/* ---------------- ToolRegistry：工具注册与调用 ---------------- */

test('ToolRegistry：注册后可调用，未知工具返回 ok=false 而非抛错', async () => {
  const reg = new ToolRegistry();
  reg.register('double', { description: 'x2', handler: async ({ x }) => x * 2 });
  const r = await reg.call('double', { x: 21 });
  assert.equal(r.ok, true);
  assert.equal(r.output, 42);

  const missing = await reg.call('nope', {});
  assert.equal(missing.ok, false);
  assert.ok(missing.error, '须携带错误信息');
});

test('ToolRegistry：处理器抛异常被捕获为 ok=false', async () => {
  const reg = new ToolRegistry();
  reg.register('boom', { description: '爆炸', handler: async () => { throw new Error('boom!'); } });
  const r = await reg.call('boom', {});
  assert.equal(r.ok, false);
  assert.match(r.error, /boom/);
});

/* ---------------- Verifier：不变量验证证书 ---------------- */

test('Verifier：对合法 PCA 产物签发通过证书', () => {
  const v = new Verifier();
  // 构造 2×3 的标准正交行向量作为主成分
  const { Mat } = globalThis.__sfMat || {};
  const comps = buildOrthonormalRows();
  const cert = v.verify('pca', {
    components: comps,
    explainedVarianceRatio: Float64Array.from([0.6, 0.4]),
  });
  assert.ok(Array.isArray(cert.checks) && cert.checks.length > 0, '证书须含检查项');
  assert.equal(cert.passed, true, `正交行应通过验证: ${JSON.stringify(cert.checks)}`);
});

test('Verifier：对非正交产物判不通过', () => {
  const v = new Verifier();
  const bad = buildOrthonormalRows();
  // 破坏正交性：让第二行与第一行相同
  for (let j = 0; j < bad.cols; j++) bad.set(1, j, bad.get(0, j));
  const cert = v.verify('pca', {
    components: bad,
    explainedVarianceRatio: Float64Array.from([0.6, 0.4]),
  });
  assert.equal(cert.passed, false, '非正交主成分必须判不通过');
});

test('certificateText：输出人类可读文本且带状态图标', () => {
  const txt = certificateText({
    kind: 'pca', ok: true,
    checks: [{ label: '正交性', ok: true, detail: '6.7e-16' }],
  });
  assert.ok(typeof txt === 'string' && txt.length > 0);
  assert.ok(txt.includes('✅'), '通过的检查须显示 ✅');
  assert.ok(txt.includes('正交性'), '须含检查项名称');
  assert.equal(certificateText(null), '（无证书）');
});

/* ---------------- AgentMemory：情景记忆 ---------------- */

test('AgentMemory：记录情景记忆并导出合法 JSONL', () => {
  const mem = new AgentMemory();
  const n = mem.record({ goal: '测试', ok: true });
  assert.equal(n, 1, 'record 应返回记录条数');
  const jsonl = mem.exportJSONL();
  assert.ok(jsonl.includes('测试'), '导出内容须含记录');
  assert.doesNotThrow(() => JSON.parse(jsonl.split('\n')[0]), '每行须为合法 JSON');
});

test('AgentMemory：语义记忆可存取与检索', () => {
  const mem = new AgentMemory();
  mem.rememberFact('iris.accuracy', 0.9778, 'bench');
  const f = mem.recallFact('iris.accuracy');
  assert.equal(f.value, 0.9778);
  assert.equal(f.source, 'bench');
  assert.equal(mem.recallFact('missing'), undefined, '未知键应返回 undefined');
  // 情景检索
  mem.record({ goal: '主成分分析' });
  assert.equal(mem.searchEpisodic('主成分').length, 1);
  assert.equal(mem.searchEpisodic('聚类').length, 0);
});

/* ---------------- StarAgent：端到端规划与执行 ---------------- */

test('StarAgent：未知目标回退安全，不抛异常', async () => {
  const agent = await createDefaultAgent();
  const r = await agent.run('今天天气怎么样？');
  assert.ok(r, '须返回结果对象');
  assert.ok(typeof r.answer === 'string' && r.answer.length > 0, '须给出非空回答');
});

test('StarAgent：PCA 配方端到端跑通并签发证书', async () => {
  const agent = await createDefaultAgent();
  const r = await agent.run('对鸢尾花数据集做主成分分析，报告前两个主成分的累计解释方差');
  assert.ok(Array.isArray(r.trace) && r.trace.length > 0, '须有执行轨迹');
  assert.ok(Array.isArray(r.certificates), '须有验证证书');
  assert.ok(/方差|主成分|PCA/i.test(r.answer), `回答须含分析结论: ${r.answer.slice(0, 80)}`);
  console.log(`  PCA 配方: ${r.trace.length} 步 / ${r.certificates.length} 张证书`);
});

test('StarAgent：分类配方端到端跑通', async () => {
  const agent = await createDefaultAgent();
  const r = await agent.run('用鸢尾花数据集训练一个分类器并报告准确率');
  assert.ok(typeof r.answer === 'string' && r.answer.length > 0);
  assert.ok(r.trace.length > 0, '须有执行轨迹');
});

test('确定性：同目标两次执行结果一致', async () => {
  const g = '对鸢尾花数据集做主成分分析';
  const a1 = await createDefaultAgent();
  const a2 = await createDefaultAgent();
  const r1 = await a1.run(g);
  const r2 = await a2.run(g);
  assert.equal(r1.answer, r2.answer, '同目标回答须一致');
});
