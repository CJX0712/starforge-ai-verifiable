#!/usr/bin/env node
/**
 * @file tools/serve.mjs — 零依赖静态服务器（Web 仪表盘专用）
 * 作者：晨星
 *
 * 用法：node tools/serve.mjs [--port 8787]
 * 只服务项目根目录内的文件，带路径穿越防护；不引入任何 npm 依赖。
 */
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { join, dirname, extname, normalize } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.md': 'text/markdown; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
};

export async function startServer(port = 8787) {
  const server = createServer(async (req, res) => {
    try {
      const urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
      let p = normalize(urlPath).replace(/^(\.\.[/\\])+/, '');
      if (p === '/' || p === '\\' || p === '.') p = '/web/dashboard.html';
      const file = join(root, p);
      if (!file.startsWith(root)) {
        res.writeHead(403, { 'Content-Type': 'text/plain' });
        res.end('Forbidden');
        return;
      }
      const data = await readFile(file);
      res.writeHead(200, {
        'Content-Type': MIME[extname(file).toLowerCase()] || 'application/octet-stream',
        'Cache-Control': 'no-store',
      });
      res.end(data);
    } catch {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('404 Not Found');
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '127.0.0.1', resolve);
  });
  return server;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const i = process.argv.indexOf('--port');
  const port = i >= 0 ? parseInt(process.argv[i + 1], 10) : 8787;
  startServer(port).then((s) => {
    console.log(`StarForge 仪表盘 → http://localhost:${port}（零依赖静态服务器，Ctrl+C 停止）`);
    process.on('SIGINT', () => s.close(() => process.exit(0)));
  }).catch((e) => {
    console.error(`启动失败: ${e.message}`);
    process.exit(1);
  });
}
