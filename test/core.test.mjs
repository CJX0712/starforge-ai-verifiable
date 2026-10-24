/**
 * @file core.test.mjs — 核心层测试（node:test）
 * 作者：晨星
 *
 * 运行：node --test test/
 * 不变量主体在 src/selfcheck.mjs（单一事实源），本文件负责驱动与补充参照值测试。
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { selfCheckAll, linalgSelfCheck, autodiffSelfCheck, optimSelfCheck, datasetsSelfCheck } from '../src/selfcheck.mjs';

test('核心四层不变量全部通过', () => {
  const reports = selfCheckAll();
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

test('每个自检组都非空', () => {
  for (const group of [linalgSelfCheck, autodiffSelfCheck, optimSelfCheck, datasetsSelfCheck]) {
    const reports = group();
    assert.ok(reports.length >= 1);
    for (const r of reports) assert.ok(r.total >= 3, `${r.name} 检查项过少`);
  }
});
