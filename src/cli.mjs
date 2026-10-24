#!/usr/bin/env node
/**
 * @file cli.mjs — StarForge 命令行入口
 * 作者：晨星
 *
 * 用法：node src/cli.mjs <command> [options]
 *   selftest                    全引擎不变量自检（核心层 + 各算法模块）
 *   bench [--quick] [--json p]  基准套件（默认写入 bench/results.json）
 *   demo                        认知智能体演示（3 个代表性目标）
 *   ask "目标"                  认知智能体单次问答
 *   train-gpt [--preset tiny]   训练 MiniGPT 并导出权重到 web/weights.json
 *   serve [--port 8787]         启动 Web 仪表盘服务器
 *   info                        引擎信息
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { selfCheckAll } from './selfcheck.mjs';
import { runAllBench, formatReport } from './bench.mjs';
import { createDefaultAgent, runDemo, certificateText } from './agent.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const cmd = args[0] || 'info';
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : undefined;
};

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const DIM = '\x1b[2m';
const BOLD = '\x1b[1m';
const RESET = '\x1b[0m';

function pad(s, w) {
  s = String(s);
  let len = 0;
  for (const ch of s) len += ch.charCodeAt(0) > 127 ? 2 : 1;
  return s + ' '.repeat(Math.max(0, w - len));
}

async function cmdSelfTest() {
  const groups = selfCheckAll();
  const moduleNames = ['classify', 'cluster', 'sequence', 'nn', 'retrieval', 'rl'];
  for (const name of moduleNames) {
    try {
      const mod = await import(`./${name}.mjs`);
      if (typeof mod.selfCheck === 'function') groups.push(...mod.selfCheck());
    } catch (e) {
      groups.push({ name, checks: [{ label: '模块加载', ok: false, detail: e.message }], passed: false, passedCount: 0, total: 1 });
    }
  }
  console.log(`${BOLD}StarForge 不变量自检${RESET}${DIM}（每个算法携带的可执行金标准）${RESET}\n`);
  let allOk = true;
  for (const g of groups) {
    const ok = g.passed;
    if (!ok) allOk = false;
    console.log(`${ok ? GREEN + '✅' : RED + '❌'} ${pad(g.name, 22)} ${g.passedCount}/${g.total}`);
    if (!ok) for (const c of g.checks) if (!c.ok) console.log(`     ${RED}└─ ${c.label}: ${c.detail}${RESET}`);
  }
  const total = groups.reduce((s, g) => s + g.total, 0);
  const passed = groups.reduce((s, g) => s + g.passedCount, 0);
  console.log(`\n${allOk ? GREEN : RED}${BOLD}总计: ${passed}/${total} 项不变量${allOk ? ' 全部通过 ✅' : ' 存在失败 ❌'}${RESET}`);
  process.exitCode = allOk ? 0 : 1;
}

async function cmdBench() {
  const quick = args.includes('--quick');
  const jsonPath = flag('--json') || join(root, 'bench', 'results.json');
  console.log(`${BOLD}StarForge 基准套件${RESET}${quick ? DIM + '（quick 模式）' + RESET : ''}\n`);
  const report = await runAllBench({ quick });
  console.log(formatReport(report));
  mkdirSync(dirname(jsonPath), { recursive: true });
  writeFileSync(jsonPath, JSON.stringify(report, null, 2));
  console.log(`\n${DIM}结果已写入 ${jsonPath}${RESET}`);
  process.exitCode = report.results.every((r) => !r.error && (r.results || []).every((x) => x.ok)) ? 0 : 1;
}

async function cmdDemo() {
  const agent = await createDefaultAgent();
  const results = await runDemo(agent);
  for (const r of results) {
    console.log(`${BOLD}━━━ 目标: ${r.goal}${RESET}`);
    console.log(`${DIM}配方: ${r.plan}（${r.intent}） | 置信度: ${(r.confidence * 100).toFixed(1)}%${RESET}`);
    for (const t of r.trace) {
      const mark = t.status === 'ok' ? GREEN + '✅' : t.status === 'fallback' ? '⚠️ ' : RED + '❌';
      console.log(`  ${mark} ${pad(t.step, 10)} → ${t.tool} ${DIM}${JSON.stringify(t.args)}${RESET}`);
      if (t.error) console.log(`     ${RED}${t.error}${RESET}`);
    }
    for (const cert of r.certificates) {
      console.log(`  ${DIM}┈┈ 验证证书 ┈┈${RESET}`);
      for (const line of certificateText(cert).split('\n')) console.log(`     ${line}`);
    }
    console.log(`\n${r.answer}\n`);
  }
}

async function cmdAsk() {
  const goal = args.slice(1).join(' ');
  if (!goal) {
    console.error('用法: node src/cli.mjs ask "目标"');
    process.exitCode = 1;
    return;
  }
  const agent = await createDefaultAgent();
  const r = await agent.run(goal);
  console.log(`${BOLD}目标${RESET}: ${goal}`);
  console.log(`${DIM}配方: ${r.plan} | 置信度: ${(r.confidence * 100).toFixed(1)}%${RESET}`);
  for (const t of r.trace) {
    const mark = t.status === 'ok' ? GREEN + '✅' : t.status === 'fallback' ? '⚠️ ' : RED + '❌';
    console.log(`  ${mark} ${t.step} → ${t.tool}`);
    if (t.error) console.log(`     ${RED}${t.error}${RESET}`);
  }
  for (const cert of r.certificates) {
    console.log(`  ${DIM}┈┈ 验证证书 ┈┈${RESET}`);
    for (const line of certificateText(cert).split('\n')) console.log(`     ${line}`);
  }
  console.log(`\n${r.answer}`);
}

async function cmdTrainGpt() {
  const preset = flag('--preset') || 'tiny';
  const steps = flag('--steps');
  console.log(`${BOLD}训练 MiniGPT${RESET}${DIM}（preset=${preset}，从零开始，无框架）${RESET}`);
  const { trainGptCli } = await import('./train-gpt.mjs');
  await trainGptCli({ preset, steps: steps ? parseInt(steps, 10) : undefined, root });
}

async function cmdServe() {
  const port = parseInt(flag('--port') || '8787', 10);
  const { startServer } = await import('../tools/serve.mjs');
  await startServer(port);
  console.log(`${GREEN}${BOLD}StarForge 仪表盘${RESET} → http://localhost:${port}`);
  console.log(`${DIM}（零依赖静态服务器，Ctrl+C 停止）${RESET}`);
}

function cmdInfo() {
  console.log(`${BOLD}StarForge（星炉）${RESET} — 零依赖纯 ESM JavaScript 世界顶级 AI 引擎`);
  console.log(`${DIM}作者: 晨星 | GitHub: CJX0712${RESET}`);
  console.log(`
${BOLD}七层架构${RESET}
  L0 核心数学    src/linalg.mjs     矩阵/特征分解/SVD/数值稳定
  L1 自动微分    src/autodiff.mjs   反向模式 AD（micrograd 张量化）
  L2 优化器      src/optim.mjs      AdamW/Lion/Muon(Newton-Schulz)
  L3 学习算法    src/classify.mjs src/cluster.mjs src/sequence.mjs
  L4 神经网络    src/nn.mjs         Transformer/MiniGPT 从零训练
  L5 检索        src/retrieval.mjs  BM25/LSA/RRF
  L6 强化学习    src/rl.mjs         Bandit/Q-learning/CartPole
  L7 认知智能体  src/agent.mjs      神经-符号 + 验证证书
  元认知        src/selfcheck.mjs  全部不变量的单一事实源

${BOLD}命令${RESET}
  selftest   全引擎不变量自检        bench   基准套件
  demo       智能体演示              ask     智能体问答
  train-gpt  训练 MiniGPT            serve   Web 仪表盘
`);
}

const COMMANDS = {
  selftest: cmdSelfTest,
  bench: cmdBench,
  demo: cmdDemo,
  ask: cmdAsk,
  'train-gpt': cmdTrainGpt,
  serve: cmdServe,
  info: cmdInfo,
};

const fn = COMMANDS[cmd];
if (!fn) {
  console.error(`未知命令: ${cmd}`);
  cmdInfo();
  process.exitCode = 1;
} else {
  fn().catch((e) => {
    console.error(RED + `命令执行失败: ${e.stack || e.message}` + RESET);
    process.exitCode = 1;
  });
}
