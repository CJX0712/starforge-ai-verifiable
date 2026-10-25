# 检索与排序（src/retrieval.mjs）

> TF-IDF / BM25 / LSA 三套检索 + RRF 混合融合 + 评测指标，零依赖。
> 作者：晨星

## 0. 预处理

- `tokenize(text)`：小写 + 正则分词（英文/数字），可扩展。
- `chunkText(text, {size, overlap})`：滑动窗口切块，覆盖 `size`、相邻块重叠 `overlap`；不变量：相邻块重叠 `= overlap`、文本全覆盖。

## 1. TfIdfIndex（向量空间模型）

- **数学**：`w(t,d) = tf(t,d)·idf(t)`，`idf(t) = log((N+1)/(df(t)+1)) + 1`（平滑，非负）；检索 `score = cos(q, d) = q·d / (‖q‖·‖d‖)`。
- **复杂度**：构建 `O(Σ|d|)`，查询 `O(|vocab|)`。
- **参考**：Salton et al. 1975（SMART）。
- **坑**：未登录词直接丢弃；空查询返回全 0。
- **不变量**：`scoreAll(docs[i])[i] === 1`（自相似）；余弦得分 `∈ [−1,1]`，本实现非负。

## 2. BM25Index（Okapi BM25）

- **数学**：
  `score(D,Q) = Σ_{t∈Q} idf(t)· f(t,D)·(k1+1) / (f(t,D) + k1·(1−b + b·|D|/avgdl))`
  - `idf_lucene(t) = log(1 + (N−df(t)+0.5)/(df(t)+0.5))`（默认，非负）
  - `idf_rsj(t) = log((N−df(t)+0.5)/(df(t)+0.5))`（原式，可负）
- **参数**：`k1≥0`（词频饱和速率，0→布尔权重）、`b∈[0,1]`（长度归一化强度）、`idfVariant`。
- **复杂度**：构建 `O(Σ|d|)`，查询 `O(|Q|·命中文档)`。
- **参考**：Robertson & Zaragoza 2009。
- **坑**：`k1<0`/`b∉[0,1]`/未知 `idfVariant` 直接抛错；`df=0` 未登录词跳过；仅返回 `score>0`。
- **不变量**：`k1=0` 退化为二元权重；`idf_lucene` 恒非负；长文档经 `b` 归一后权重回落。

## 3. LsaIndex（潜在语义分析）

- **数学**：文档-词项矩阵 `X` → SVD `X = UΣVᵀ`（复用 `linalg.jacobiSVD`）→ 取前 `k` 维语义空间 `X_k = U_k Σ_k`；查询向量投影后算余弦。
- **参考**：Deerwester et al. 1990（LSA）。
- **坑**：需要语料足够大；`k` 过大过拟合、过小丢语义。
- **不变量**：降维重构误差随 `k` 增大而下降（Eckart-Young）；同义改写经共现结构可桥接到同一语义向量。

## 4. reciprocalRankFusion + hybridSearch（RRF 混合检索）

- **数学**（Cormack et al. 2009）：`score(d) = Σ_{r∈排名列表} 1/(k + rank(d))`，`k=60`；只用名次不用分数，天然免疫 BM25/LSA 分数量纲差异。
- **混合**：`hybridSearch(bm25Results, lsaResults)` = RRF 融合两路 `docIndex` 排名，取 `topK`。
- **复杂度**：`O(R·log R)`，`R` 为候选数。
- **不变量**：同文档跨列表重复名次只计一次；融合分按名次降序、字典序 tie-break。

## 5. 评测指标

- `precisionAtK`：topK 命中比例 `∈[0,1]`；多数查询仅 1 篇相关时上限 `1/k`（天花板现象）。
- `recallAtK`：`命中相关数 / 相关总数`，归一化掉相关集规模影响——**本评测以 Recall@K 为主质量门槛**。
- `reciprocalRank`：首个相关文档名次倒数 `1/rank`（MRR 单查询项）。

## 基准（seed=42）

| 任务 | 指标 | 结果 |
|------|------|------|
| TF-IDF 公开算例 | 公式逐项精确值 | ✓ |
| BM25 公开算例 | 公式逐项精确值 | ✓ |
| LSA | 重构误差随 k 下降、同义桥接 | ✓ |
| TF-IDF / LSA 自相似 | `score=1` | ✓ |
| RRF | 手算值一致 | ✓ |
| chunkText | 覆盖 + 重叠 | ✓ |

## 可验证不变量（selfCheck 全过）

分词约定、TF-IDF 公式值、BM25 公开算例逐项精确值、TF 饱和性、`idf_lucene` 非负、LSA 重构误差递减、LSA 同义桥接、TF-IDF/LSA 自相似=1、RRF 手算值、chunkText 覆盖与重叠。

---

下一篇：[强化学习](rl.md) · [总索引](../README.md)
