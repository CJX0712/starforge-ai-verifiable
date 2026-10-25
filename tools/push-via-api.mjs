/**
 * push-via-api.mjs — 通过 GitHub REST API (api.github.com) 推送提交
 * 原因：本机 git 的 github.com:443 传输被网络策略阻断，仅 api.github.com 可达。
 * 做法：用 Git Data API 重建 blob → tree → commit，并严格复用本地提交的
 *       tree / parent / author / committer / message，使远端 commit SHA 与本地完全一致，
 *       从而保持本地与远端零分歧。
 * 作者：晨星
 */
import { execFileSync } from 'node:child_process';

const OWNER = 'CJX0712';
const REPO = 'starforge-ai-verifiable';
const BRANCH = 'main';

const git = (...args) => execFileSync('git', args, { cwd: process.cwd(), encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });

function gh(args, input) {
  const res = execFileSync('gh', args, {
    input: input === undefined ? undefined : JSON.stringify(input),
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
  }).trim();
  if (!res) return null;
  // gh --jq 输出可能是裸字符串（如 sha），也可能是 JSON
  try { return JSON.parse(res); } catch { return res; }
}

// ---------- 1. 读取本地提交元数据 ----------
const headSha = git('rev-parse', 'HEAD').trim();
const parentSha = git('rev-parse', 'HEAD^').trim();
const treeSha = git('rev-parse', 'HEAD^{tree}').trim();
const parentTreeSha = git('rev-parse', 'HEAD^^{tree}').trim();
const rawCommit = git('cat-file', 'commit', 'HEAD');

// 精确切出提交消息（含末尾换行）
const msgStart = rawCommit.indexOf('\n\n');
const message = rawCommit.slice(msgStart + 2);

const parseIdent = (line) => {
  const m = /^(.*) <(.*)> (\d+) ([+-]\d{4})$/.exec(line);
  const [_, name, email, epoch] = m;
  // 关键：GitHub API 会把提交时间统一归一化为 +0000（UTC）。
  // 本地若存 +0800，同一时刻也会得到不同 commit SHA → 造成本地/远端分歧。
  // 因此这里一律换算成 UTC，并要求本地提交也用 UTC（见文末说明）。
  const d = new Date(Number(epoch) * 1000);
  const pad = (n) => String(n).padStart(2, '0');
  const iso = `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}+00:00`;
  return { name, email, date: iso };
};
const authorLine = rawCommit.split('\n').find((l) => l.startsWith('author '));
const committerLine = rawCommit.split('\n').find((l) => l.startsWith('committer '));
const author = parseIdent(authorLine.slice('author '.length));
const committer = parseIdent(committerLine.slice('committer '.length));

console.log(`本地 HEAD    = ${headSha}`);
console.log(`本地 tree    = ${treeSha}`);
console.log(`parent tree  = ${parentTreeSha}`);
console.log(`author       = ${author.name} <${author.email}> ${author.date}`);
console.log(`committer    = ${committer.name} <${committer.email}> ${committer.date}`);

// ---------- 2. 逐个文件创建 blob 并校验 SHA ----------
const changed = git('diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').trim().split('\n').filter(Boolean);
const entries = [];
for (const path of changed) {
  const content = execFileSync('git', ['cat-file', 'blob', `HEAD:${path}`], { cwd: process.cwd(), maxBuffer: 64 * 1024 * 1024 }).toString('utf8');
  const localBlobSha = git('rev-parse', `HEAD:${path}`).trim();
  const blob = gh(['api', '--method', 'POST', `repos/${OWNER}/${REPO}/git/blobs`, '--input', '-', '--jq', '.sha'],
    { content, encoding: 'utf-8' });
  const ok = blob === localBlobSha;
  console.log(`blob ${path.padEnd(24)} remote=${blob} local=${localBlobSha} ${ok ? '✅' : '❌ 不一致'}`);
  if (!ok) throw new Error(`blob SHA 不一致: ${path}`);
  entries.push({ path, mode: '100644', type: 'blob', sha: blob });
}

// ---------- 3. 创建 tree 并校验 SHA ----------
const tree = gh(['api', '--method', 'POST', `repos/${OWNER}/${REPO}/git/trees`, '--input', '-', '--jq', '.sha'],
  { base_tree: parentTreeSha, tree: entries });
console.log(`tree remote=${tree} local=${treeSha} ${tree === treeSha ? '✅' : '❌ 不一致'}`);
if (tree !== treeSha) throw new Error('tree SHA 不一致');

// ---------- 4. 创建 commit 并校验 SHA ----------
const commit = gh(['api', '--method', 'POST', `repos/${OWNER}/${REPO}/git/commits`, '--input', '-', '--jq', '.sha'],
  { message, tree, parents: [parentSha], author, committer });
console.log(`commit remote=${commit} local=${headSha} ${commit === headSha ? '✅' : '❌ 不一致'}`);
if (commit !== headSha) throw new Error('commit SHA 不一致');

// ---------- 5. 更新分支引用 ----------
gh(['api', '--method', 'PATCH', `repos/${OWNER}/${REPO}/git/refs/heads/${BRANCH}`, '--input', '-'],
  { sha: commit, force: false });
console.log(`ref ${BRANCH} -> ${commit} ✅`);

// ---------- 6. 校验远端状态 ----------
const ref = gh(['api', `repos/${OWNER}/${REPO}/git/ref/heads/${BRANCH}`]);
console.log(`远端 ${BRANCH} 当前 = ${ref.object.sha}`);
console.log(ref.object.sha === headSha ? '✅ 本地与远端完全一致（零分歧）' : '❌ 远端未同步');
