/**
 * @file bench.mjs — 基准套件汇总器
 * 作者：晨星
 *
 * 每个算法模块导出 bench() → { module, results: [{name, value, unit, target, ok, detail}] }，
 * 本模块负责动态汇总、渲染表格并落盘 JSON（供 CI 与 Web 仪表盘消费）。
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const MODULES = ['classify', 'cluster', 'sequence', 'nn', 'retrieval', 'rl'];

export async function runAllBench({ quick = false } = {}) {
  const results = [];
  for (const name of MODULES) {
    try {
      const mod = await import(`./${name}.mjs`);
      if (typeof mod.bench === 'function') {
        const r = await mod.bench({ quick });
        results.push({ module: name, ...r });
      } else {
        results.push({ module: name, error: '未导出 bench()' });
      }
    } catch (e) {
      results.push({ module: name, error: `模块加载失败: ${e.message}` });
    }
  }
  const summary = { total: 0, passed: 0 };
  for (const r of results) {
    for (const x of r.results || []) {
      summary.total++;
      if (x.ok) summary.passed++;
    }
  }
  return {
    generatedAt: new Date().toISOString(),
    node: process.version,
    quick,
    summary,
    results,
  };
}

function pad(s, w) {
  s = String(s);
  let len = 0;
  for (const ch of s) len += ch.charCodeAt(0) > 127 ? 2 : 1;
  return s + ' '.repeat(Math.max(0, w - len));
}

function fmtValue(v) {
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return String(v);
  return Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? v.toExponential(2) : v.toFixed(4);
}

/** 控制台报告（CJK 宽度感知对齐） */
export function formatReport(report) {
  const lines = [];
  let allOk = true;
  for (const r of report.results) {
    if (r.error) {
      allOk = false;
      lines.push(`❌ ${pad(r.module, 12)} 加载失败: ${r.error}`);
      continue;
    }
    lines.push(`${'─'.repeat(3)} ${r.module} ${'─'.repeat(3)}`);
    for (const x of r.results || []) {
      if (!x.ok) allOk = false;
      const mark = x.ok ? '✅' : '❌';
      lines.push(`  ${mark} ${pad(x.name, 38)} ${pad(fmtValue(x.value), 12)} ${pad(x.unit, 10)} 目标 ${x.target || '-'}`);
      if (x.detail) lines.push(`     ${x.detail}`);
    }
  }
  lines.push('');
  lines.push(`${report.summary.passed}/${report.summary.total} 项基准${allOk ? ' 全部达标 ✅' : ' 存在未达标 ❌'}`);
  return lines.join('\n');
}

/** 写 bench/results.json */
export function saveReport(report, path = join(root, 'bench', 'results.json')) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(report, null, 2));
  return path;
}
