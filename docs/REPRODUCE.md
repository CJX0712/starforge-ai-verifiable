# 复现指南（Reproduce）

> 作者：晨星 · StarForge（星炉） · MIT License

## 1. 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| Node.js | ≥ 18（推荐 22 LTS） | 唯一硬依赖；无 npm 包、无构建步骤 |
| 操作系统 | Windows / macOS / Linux 均可 | 纯 ESM + `node:` 内置模块 |
| 网络 | 仅首次获取数据需要 | 数据集已内置为 ES 模块，运行期零网络 |

## 2. 五分钟复现全链路

```bash
git clone https://github.com/CJX0712/starforge-ai.git
cd starforge-ai

# ① 测试套件（node:test，零配置）
node --test

# ② 不变量自检：每个算法的可执行金标准
node src/cli.mjs selftest

# ③ 基准套件（分类/聚类/序列/检索/强化学习，约 1-3 分钟）
node src/cli.mjs bench

# ④ 认知智能体演示
node src/cli.mjs demo
node src/cli.mjs ask "BM25 的原理是什么"

# ⑤ 从零训练 MiniGPT（tiny 预设约 2-5 分钟，取决于机器）
node src/cli.mjs train-gpt --preset tiny

# ⑥ Web 仪表盘（架构图 / 自检 / 基准 / GPT 实验室 / 智能体）
node src/cli.mjs serve
# 浏览器打开 http://localhost:8787
```

预期输出：`selftest` 显示全部不变量通过；`bench` 的基准达标率见 `BENCHMARKS.md`；
`train-gpt` 输出损失曲线（从 ≈ ln(词表大小) 下降）与三个温度的生成样本。

## 3. 确定性声明

- 所有随机性经 `src/linalg.mjs` 的 `createRng(seed)`（mulberry32 + Box-Muller）注入。
- 数据划分、模型初始化、批次采样、聚类初始化、老虎机/环境动力学全部使用固定种子（默认 42）。
- **同种子 ⇒ 逐位相同的结果**（测试 `autodiff/训练确定性` 专门守护这一点）。
- 未固定种子的入口：Web 仪表盘 GPT 实验室的采样温度交互（推理期体验功能，不影响基准）。

## 4. 数据来源与再生成

| 数据 | 来源 | 许可证 | 再生成 |
|---|---|---|---|
| 鸢尾花（150×4） | UCI ML Repository，R.A. Fisher (1936) | CC BY 4.0 | `node tools/gen-data.mjs` |
| 葡萄酒（178×13） | UCI ML Repository，Forina et al. | CC BY 4.0 | 同上 |
| 字符语料（≈26 万字符） | Project Gutenberg：爱丽丝梦游仙境 + 莎士比亚全集 | 公有领域 | 同上 |

`tools/gen-data.mjs` 读取 `data/*.data` 与 `data/corpus.txt`，生成浏览器友好的
`data/iris.mjs` / `data/wine.mjs` / `data/corpus.mjs`。仓库中已包含生成产物，
**重新生成不是运行前提**。

## 5. 基准结果对照

`node src/cli.mjs bench` 生成 `bench/results.json`，人类可读摘要见 `BENCHMARKS.md`。
所有基准的目标值与实际值都记录在案；若你的运行结果与文档存在偏差，请带上
Node 版本、操作系统与 `results.json` 提 issue。

## 6. CI 对照

GitHub Actions（`.github/workflows/ci.yml`）在 Node 18/20/22 三个版本上执行：
不变量自检 → 测试套件 → quick 基准。CI 绿 ⇔ 本机复现应一致。

## 7. 常见问题

**Q: `node --test` 报错？** 确认 Node ≥ 18；本机可用 `node --version` 检查。

**Q: 智能体某些工具报"模块加载失败"？** 对应算法模块缺失或被裁剪，其余功能不受影响
（工具注册表按需加载的设计本意）。

**Q: MiniGPT 生成质量？** 这是零依赖 JS 中的字符级迷你模型（数万参数），
定位是"从零训练全链路可复现"的教学-研究载体，不是生产级语言模型；
架构与训练代码与 nanoGPT 同构，可无缝迁移到 GPU 框架。

**Q: 如何在自己的项目里使用单个算法？** 每个 `src/*.mjs` 都是独立单文件，
直接 `import { KMeans } from './src/cluster.mjs'` 即可，无传递依赖负担
（模块间依赖仅限 `linalg/autodiff/optim/verify/datasets` 五个核心文件）。
