# 神经网络与 Transformer（src/nn.mjs）

> 张量化自动微分（micrograd 风格反向模式 AD，见 autodiff.md）+ Module 系统 + 注意力/Transformer/MiniGPT，零依赖。
> 作者：晨星

## 1. Module 与张量计算图

- 所有层继承 `Module`，统一 `forward(x)`、`parameters()`、`train()/eval()`、`zeroGrad()`。
- 复用 `autodiff.mjs` 的 `Tensor`：前向记录计算图，反向累积 `.grad`；`crossEntropyLogits` 直接对 logits 做 softmax+NLL（数值稳定，减最大值）。
- `stackRows`/`repeatRows`：批量序列拼接/位置编码广播的纯张量原语。

## 2. 基础层

| 层 | 数学 | 关键不变量 |
|----|------|-----------|
| `Linear(d_in, d_out)` | `y = xW + b`，Glorot `U(−1/√fan, ·)` 初始化、`b=0` | 梯度检验 `maxRelErr ≈ 1e-7`；零偏置 |
| `LayerNorm(d)` | `y = γ·(x−μ)/√(σ²+ε) + β` | `γ=1,β=0` 时每行均值 0、方差 1（ε 效应内 `≤1e-4`） |
| `Dropout(p)` | 训练期概率 `p` 置零并缩放 `1/(1−p)`；推理期恒等 | 同种子掩码可复现 |
| `Embedding(V, d)` | 查表 `E[id]`；属 embedding 家族，`muonExempt=true` | 梯度仅落在该行 |
| `MultiHeadAttention(d, h)` | 缩放点积注意力 `softmax(QKᵀ/√d_k)V`，因果 `causal=true` 掩码 | 因果掩码下位置 `t` 看不到 `>t` |
| `MLP` / `Activation` | 两层线性 + 非线性（gelu/relu） | 形状传播 |
| `Sequential` / `Classifier` | 串联 / 末层接交叉熵 | `forward` 形状一致 |

## 3. TransformerBlock

- **结构**：`h = x + Dropout(Attn(LN1(x)))`；`h = h + Dropout(MLP(LN2(h)))`（Pre-LN 残差）。
- 参数 `dim, heads, causal, dropout, hidden=4·dim`；`config` 记录可序列化元信息。

## 4. MiniGPT（字符级自回归语言模型）

- **结构**（Radford et al. 2018 GPT-1 最小形态）：token embedding + 可学习位置编码 `[ctx,d]` → `nLayer` 个因果 `TransformerBlock` → `LN` → 线性头 `[d, V]`。
- **forward(ids: number[][])**：`B·T` 个 token 查表 → 加 `repeatRows(posEmb[:T], B)` → 按序列切片各自过 blocks → LN + 头 → `stackRows` 成 `[B·T, V]`。
- **loss(x, y)**：`x,y` 为 `number[][]` 时逐元素对齐；单参 `loss(ids)` 目标为右移一位（next-token）。
- **generate**：`temperature→0` 退化为贪心，`topK>0` 仅在前 K 候选采样；`eval()` 模式生成。
- **Muon 豁免**：token/位置 embedding 与输出头 `muonExempt=true`，回退 AdamW（这些矩阵非方阵、按列正交化无意义）。
- **参考**：Vaswani et al. 2017（Attention is All You Need）；Radford et al. 2018。
- **复杂度为**：`O(T²·d·nLayer)`（因果注意力）。

## 5. 训练与复用工具

- `trainModel(model, opts)`：小批量 SGD/AdamW/Muon 驱动，返回 loss 历史。
- `countParameters(model)`：遍历 `parameters()` 累加 `size`（结构化断言，不依赖具体形状）。
- `encodeWeightsBase64` / `decodeWeightsBase64`：权重序列化为 `Float64` base64（确定性、可 diff）。
- `saveModel` / `loadModel`：JSON 结构含 `config` 与 base64 权重，跨进程复现。

## 基准（seed=42，详见 BENCHMARKS.md）

统一基准套件（`node src/cli.mjs bench`，CI 入口，quick 模式约 2 秒）：

| 基准项 | 结果 | 目标 |
|--------|------|------|
| 单批过拟合（反向传播金标准） | 3.9592 → 3.89e-4 | <1e-3 ✅ |
| 因果掩码不变量（改动位置 ≥5，位置 <5 的 logits 偏差） | 0.0（逐位一致） | <1e-12 ✅ |
| AdamW vs Muon（同预算） | AdamW 2.5e-3 / Muon 1.5e-3 | 两者均下降 >40% ✅ |
| 权重 base64 往返（257 个 Float64） | 0 | ==0 ✅ |
| Iris MLP（4→16→3）测试准确率 | 0.9556 | ≥0.95 ✅ |

完整规模训练基准（`node bench/nn-bench.mjs`，约 5 分钟）：

| 任务 | 指标 | 结果 |
|------|------|------|
| 过拟合单批 | loss → 2.08e-6 | ✅ |
| AdamW vs Muon（各 200 步） | 最终 loss | 2.7504 vs 2.6648（Muon 更优） |
| 莎士比亚字符级 MiniGPT | tiny 预设 300 步 | loss 4.6001 → 2.7239，困惑度 15.24 |

## 可验证不变量（selfCheck 全过，梯度检验金标准）

- `Linear`：中心差分 vs 解析梯度 `maxRelErr ≈ 1e-7`，零偏置、Glorot 尺度。
- `LayerNorm`：每行均值≈0、方差≈1（ε 效应内），`γ/β` 缩放生效。
- `MultiHeadAttention`：梯度检验通过；因果掩码下三角为 0。
- `MiniGPT`：forward 形状 `[B·T, V]`；`loss(ids)` 右移目标；`generate` 长度受控。

---

下一篇：[检索与排序](retrieval.md) · [总索引](../README.md)
