/**
 * @file retrieval.test.mjs — 检索模块测试（node:test）
 * 作者：晨星
 *
 * 运行：node --test test/
 * 不变量主体在 src/retrieval.mjs 的 selfCheck()（单一事实源），
 * 本文件负责驱动自检，并补充排序质量与中文分词的契约测试。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  tokenize, chunkText, TfIdfIndex, BM25Index, LsaIndex,
  reciprocalRankFusion, hybridSearch, precisionAtK, recallAtK, reciprocalRank,
  selfCheck, bench,
} from '../src/retrieval.mjs';

test('retrieval 不变量全部通过', () => {
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

test('retrieval 基准全部达标', () => {
  const { results } = bench();
  const bad = results.filter((r) => !r.ok);
  assert.equal(bad.length, 0, `未达标项:\n${bad.map((b) => `${b.name}: ${b.detail}`).join('\n')}`);
  console.log(`  基准: ${results.length}/${results.length} 达标`);
});

test('BM25：公开算例逐项精确值（硬标准，容差 1e-3）', () => {
  const idx = new BM25Index([
    'the cat sat on the mat',
    'the dog sat on the log',
    'cats and dogs are friends',
  ]);
  // Doc1: cat 贡献 0.9578，sat 贡献 0.4590，合计 1.4167
  assert.ok(Math.abs(idx.explain('cat sat', 0).score - 1.4167) < 1e-3);
  assert.ok(Math.abs(idx.explain('cat sat', 1).score - 0.4590) < 1e-3);
  assert.ok(Math.abs(idx.explain('cat sat', 2).score - 0.0) < 1e-12);
  assert.ok(Math.abs(idx.avgdl - 17 / 3) < 1e-12);
});

test('BM25：词频饱和——重复词带来的增益递减', () => {
  const d1 = 'cat cat cat cat cat sat';
  const d2 = 'cat sat';
  const idx = new BM25Index([d1, d2]);
  const s1 = idx.explain('cat', 0).score;
  const s2 = idx.explain('cat', 1).score;
  const ratio = s1 / s2;
  // 词频从 1 增到 5，分数增长应显著低于 5 倍
  assert.ok(ratio > 1 && ratio < 5, `饱和比应在 (1,5)，实际 ${ratio.toFixed(3)}`);
  console.log(`  TF 饱和: 5 次重复 / 1 次 = ${ratio.toFixed(3)} 倍（线性应为 5.0）`);
});

test('BM25：文档长度归一化——b=0 时长度不影响打分', () => {
  const short = 'cat';
  const long = 'cat ' + 'filler '.repeat(50);
  const idx0 = new BM25Index([short, long], { b: 0 });
  const d = Math.abs(idx0.explain('cat', 0).score - idx0.explain('cat', 1).score);
  assert.ok(d < 1e-9, `b=0 时长短文档分数应相同，实际差 ${d}`);
});

test('RRF：融合分等于各排名 1/(k+rank) 之和', () => {
  const rankings = [[1, 2, 3], [3, 1, 4]];
  const k = 60;
  const fused = reciprocalRankFusion(rankings, { k });
  const expected = new Map();
  for (const r of rankings) {
    r.forEach((doc, i) => {
      expected.set(doc, (expected.get(doc) || 0) + 1 / (k + i + 1));
    });
  }
  for (const { docIndex, score } of fused) {
    assert.ok(Math.abs(score - expected.get(docIndex)) < 1e-15,
      `文档 ${docIndex} 融合分 ${score} ≠ 手算 ${expected.get(docIndex)}`);
  }
  for (let i = 1; i < fused.length; i++) {
    assert.ok(fused[i - 1].score >= fused[i].score - 1e-15, '融合结果须按分数降序');
  }
});

test('指标函数：P@K / Recall@K / RR 的边界与正确性', () => {
  assert.equal(precisionAtK([1, 2, 3], [1], 3), 1 / 3);
  assert.equal(precisionAtK([1, 2, 3], [], 3), 0);
  assert.equal(recallAtK([1, 2, 3], [1, 9], 3), 1 / 2);
  assert.equal(recallAtK([1, 2, 3], [1], 3), 1);
  assert.equal(reciprocalRank([2, 1, 3], [1]), 1 / 2);
  assert.equal(reciprocalRank([2, 3], [1]), 0);
});

test('中文分词：CJK 二元组滑窗约定稳定', () => {
  assert.deepEqual(tokenize('机器学习'), ['机器', '器学', '学习']);
  assert.deepEqual(tokenize('猫'), ['猫']);
  const t = tokenize('你好world世界');
  assert.ok(t.includes('你好') && t.includes('world') && t.includes('世界'));
});

test('chunkText：分块拼接覆盖原文且重叠生效', () => {
  const text = Array.from({ length: 500 }, (_, i) => `w${i}`).join(' ');
  const chunks = chunkText(text, { size: 60, overlap: 15 });
  assert.ok(chunks.length > 1, '应产生多个块');
  const toksAll = text.split(/\s+/);
  const toksChunk = chunks.join(' ').split(/\s+/);
  assert.ok(toksChunk.length >= toksAll.length, '拼接后须覆盖全部 token');
  for (const w of toksAll) assert.ok(toksChunk.includes(w), `缺失 token ${w}`);
});

test('检索合理性：相关查询的最高命中即为标注文档', () => {
  const docs = [
    '深度学习引擎与自动微分',
    '意大利面煮八分钟口感劲道',
    '宋朝交子是世界上最早的纸币',
  ];
  const idx = new BM25Index(docs);
  assert.equal(idx.search('自动微分 深度学习', { topK: 1 })[0].docIndex, 0);
  assert.equal(idx.search('意大利面 煮面', { topK: 1 })[0].docIndex, 1);
  assert.equal(idx.search('宋朝 纸币', { topK: 1 })[0].docIndex, 2);
});
