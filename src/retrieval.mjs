/**
 * @file retrieval.mjs — StarForge 检索层：分词 / TF-IDF / BM25 / LSA / RRF 混合检索
 * 作者：晨星
 *
 * 定位：世界顶级 AI 引擎的"记忆检索"原语。纯 ESM、零外部依赖、完全确定性
 * （无随机性注入点——同输入必同输出），每个算法携带可被独立实现交叉验证的不变量。
 *
 * 算法出处（直接复用经典公式，不做无依据的"魔改"）：
 * - 分词约定：CJK 二元组（bigram）滑窗 + 英文按非字母数字切分
 *   （Manning, Raghavan & Schütze 2008, *Introduction to Information Retrieval*, 第 2 章）
 * - TF-IDF：Salton & Buckley (1988), Term-weighting approaches in automatic textual retrieval
 *   （idf = log((N+1)/(df+1)) + 1，sklearn smooth_idf 风格，保证非负）
 * - BM25：Robertson & Zaragoza (2009), The Probabilistic Relevance Framework: BM25 and Beyond
 *   （Okapi BM25；默认采用 Lucene 经典实现的非负 idf 变体 log(1 + (N−df+0.5)/(df+0.5))）
 * - LSA：Deerwester, Dumais, Furnas, Landauer & Harshman (1990), Indexing by Latent
 *   Semantic Analysis（TF-IDF 矩阵的截断 SVD，查询投影后 cosine 相似度）
 * - RRF：Cormack, Clarke & Buettcher (2009), Reciprocal Rank Fusion outperforms
 *   Condorcet and individual Rank Learning Methods（score(d) = Σ_r 1/(k + rank_r(d))）
 * - 评测：Manning et al. (2008) 第 8 章（P@k / MRR）
 *
 * 交叉验证不变量（在 selfCheck() 中固化为可执行断言）：
 * - BM25 公开算例逐项精确值（cat 贡献 0.9578、sat 贡献 0.4590、BM25(D1)=1.4167）
 * - TF 饱和性：k1 效应下同一文档查询词出现 2 次 vs 1 次，分数比 < 1.5
 * - Lucene idf 非负；RSJ idf 在 df=N 时可为负（Robertson 原文行为）
 * - LSA 重构误差随 nComponents 增加严格下降（Eckart–Young 1936 最优性）
 * - LSA 同义桥接：与查询无语词交集的语义相关文档 cosine > 0（TF-IDF 恒为 0）
 * - RRF 融合分手工计算值；chunkText 全覆盖 + 重叠；tokenize 中英混合约定
 */
import { Mat, matMul, matSub, transpose, svd, frobeniusNorm, l2norm } from './linalg.mjs';
import { suite, expectClose } from './verify.mjs';

/* ==================== 1. 分词与分块 ==================== */

/**
 * CJK 表意文字区（扩展 A + 基本区 + 兼容表意）。韩文/日文假名等按"非字母数字"切分，
 * 不生成二元组——这是明确的约定，见 docs/algorithms/retrieval.md。
 */
const CJK_RANGE = '\\u3400-\\u4DBF\\u4E00-\\u9FFF\\uF900-\\uFAFF';
const CJK_CHAR_RE = new RegExp(`[${CJK_RANGE}]`);
/** 词元扫描：CJK 连续段 | 英文/数字词（允许词内部 ' ’ -，如 don't / state-of-the-art） */
const TOKEN_SCAN_RE = new RegExp(`[${CJK_RANGE}]+|[A-Za-z0-9]+(?:['\\u2019-][A-Za-z0-9]+)*`, 'g');
/** 分块单元扫描：单个 CJK 字 | 英文/数字词 | 任意非空白字符（标点独立成单元，保持归属） */
const UNIT_SCAN_RE = new RegExp(`[${CJK_RANGE}]|[A-Za-z0-9]+(?:['\\u2019-][A-Za-z0-9]+)*|\\S`, 'g');

/**
 * 分词：中英混合文本 → 词元数组（确定性，无外部依赖）。
 *
 * 约定（本实现的"契约"，测试与文档均以此为准）：
 * 1. 英文/数字：按非字母数字边界切分，默认转小写；词内部的 ' ’ - 不切开
 *    （"don't" → ["don't"]，"GPT-4" → ["gpt-4"]，"2024" → ["2024"]）。
 * 2. CJK：对每一段连续 CJK 字符串做长度为 2 的滑窗二元组（bigram）；
 *    单字 CJK 段退化为单字词元（避免单字词丢失）。
 * 3. 中英混合：各自处理，按原文从左到右的顺序输出，互不干扰。
 *    例："你好world世界" → ["你好", "world", "世界"]（不做跨语言边界的二元组）。
 * 4. 空白与标点（含全角标点）一律作为分隔符丢弃。
 *
 * @param {string} text 输入文本
 * @param {{lowercase?: boolean}} [options] lowercase 默认 true，仅作用于英文词元
 * @returns {string[]} 词元数组（保留原文顺序，可重复）
 */
export function tokenize(text, { lowercase = true } = {}) {
  if (typeof text !== 'string') throw new TypeError(`tokenize 需要字符串输入，收到 ${typeof text}`);
  const out = [];
  const re = new RegExp(TOKEN_SCAN_RE.source, 'g');
  let m;
  while ((m = re.exec(text)) !== null) {
    const tok = m[0];
    if (CJK_CHAR_RE.test(tok[0])) {
      if (tok.length === 1) out.push(tok); // 单字 CJK 段：退化为单字词元
      else for (let i = 0; i + 1 < tok.length; i++) out.push(tok.slice(i, i + 2)); // 二元组滑窗
    } else {
      out.push(lowercase ? tok.toLowerCase() : tok);
    }
  }
  return out;
}

/**
 * 文本分块：按"单元"（英文词 / 单个 CJK 字 / 单个标点）贪心装包，输出 string[]。
 *
 * 契约：
 * - 每个 chunk 是在不切断单元的前提下尽量装满 size 个字符（贪心），
 *   因此 chunk 长度可能略超 size（超幅不超过一个单元的长度，长单词例外见文档"常见坑"）。
 * - overlap 为尽力而为的重叠字符数（受单元粒度限制，实际重叠 ∈ (overlap−单元长, overlap]）。
 * - 不变量：所有 chunk 按序拼接后，原文的每个单元都至少出现在一个 chunk 中（全覆盖）；
 *   当 overlap ≥ 1 时相邻 chunk 边界处至少共享 1 个字符（分块长度 > 1 个单元时）。
 *
 * @param {string} text 原文
 * @param {{size?: number, overlap?: number}} [options] size 默认 300，overlap 默认 50（0 ≤ overlap < size）
 * @returns {string[]} chunk 数组（按原文顺序；空文本返回 []）
 */
export function chunkText(text, { size = 300, overlap = 50 } = {}) {
  if (typeof text !== 'string') throw new TypeError(`chunkText 需要字符串输入，收到 ${typeof text}`);
  if (!Number.isInteger(size) || size < 1) throw new RangeError(`size 必须是正整数，收到 ${size}`);
  if (!Number.isInteger(overlap) || overlap < 0) throw new RangeError(`overlap 必须是非负整数，收到 ${overlap}`);
  if (overlap >= size) throw new RangeError(`overlap(${overlap}) 必须小于 size(${size})，否则分块无法前进`);
  const units = [];
  const re = new RegExp(UNIT_SCAN_RE.source, 'g');
  let m;
  while ((m = re.exec(text)) !== null) units.push({ start: m.index, end: m.index + m[0].length });
  const chunks = [];
  let i = 0;
  while (i < units.length) {
    // 贪心：从单元 i 开始装包，直到跨度达到 size 或文本耗尽
    let j = i + 1;
    while (j < units.length && units[j - 1].end - units[i].start < size) j++;
    chunks.push(text.slice(units[i].start, units[j - 1].end));
    if (j === units.length) break; // 已覆盖到文本末尾，无需再造 chunk
    // 回退：下一个 chunk 起点尽可能靠前，使重叠跨度 ≤ overlap（单元粒度下尽力而为）
    let next = j;
    while (next > i + 1 && units[j - 1].end - units[next - 1].start <= overlap) next--;
    i = next; // next ≥ i+1，保证至少前进一个单元，循环必终止
  }
  return chunks;
}

/* ==================== 2. TF-IDF 索引 ==================== */

/** 内部：对文档数组做分词并统计 df / 词表（确定性：词表按字典序） */
function analyzeDocs(docs) {
  const docTokens = docs.map((d) => tokenize(String(d ?? '')));
  const df = new Map();
  for (const toks of docTokens) {
    for (const t of new Set(toks)) df.set(t, (df.get(t) ?? 0) + 1);
  }
  const vocab = [...df.keys()].sort();
  return { docTokens, df, vocab, N: docs.length };
}

/** 内部：把稀疏权重 Map 收敛为 topK 排序结果（仅保留正分文档） */
function rankTopK(scores, topK = 5, eps = 1e-12) {
  if (!Number.isInteger(topK) || topK < 1) throw new RangeError(`topK 必须是正整数，收到 ${topK}`);
  return scores
    .map((score, docIndex) => ({ docIndex, score }))
    .filter((r) => r.score > eps)
    .sort((a, b) => b.score - a.score || a.docIndex - b.docIndex)
    .slice(0, topK);
}

/**
 * TF-IDF 向量空间索引（Salton & Buckley 1988）。
 *
 * - 词频 tf 用原始计数（经典 VSM 做法，未做次线性缩放——见文档"常见坑"）。
 * - idf = log((N+1)/(df+1)) + 1（sklearn smooth_idf 风格，恒 ≥ 1，非负）。
 * - 文档/查询表示为 tf·idf 权重向量，相似度为 cosine。
 * - search 仅返回得分 > 0 的文档（未命中任何查询词的文档不进结果），
 *   按分数降序、同分按 docIndex 升序（确定性）。
 */
export class TfIdfIndex {
  /**
   * @param {string[]} docs 文档数组
   */
  constructor(docs = []) {
    this.build(docs);
  }

  /** 构建 / 重建索引 */
  build(docs) {
    if (!Array.isArray(docs)) throw new TypeError(`build 需要字符串数组，收到 ${typeof docs}`);
    this.docs = docs.map(String);
    const { docTokens, df, vocab, N } = analyzeDocs(this.docs);
    this.N = N;
    this.df = df;
    this.vocab = vocab;
    this.termIndex = new Map(vocab.map((t, i) => [t, i]));
    this.idf = vocab.map((t) => Math.log((N + 1) / (df.get(t) + 1)) + 1);
    // 文档稀疏向量：Map<词元, tf·idf>
    this.docVecs = docTokens.map((toks) => {
      const tf = new Map();
      for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
      const w = new Map();
      for (const [t, f] of tf) w.set(t, f * this.idf[this.termIndex.get(t)]);
      return w;
    });
    this.norms = this.docVecs.map((w) => l2norm(Float64Array.from(w.values())));
    return this;
  }

  /**
   * 查询的 TF-IDF 稀疏向量（未登录词直接丢弃）。
   * @returns {Map<string, number>} 词元 → tf·idf
   */
  vectorize(text) {
    const toks = tokenize(text);
    const tf = new Map();
    for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
    const w = new Map();
    for (const [t, f] of tf) {
      const ti = this.termIndex.get(t);
      if (ti !== undefined) w.set(t, f * this.idf[ti]);
    }
    return w;
  }

  /**
   * 全部文档的 cosine 得分数组（下标即 docIndex）。
   * 不变量：对任意文档 i，scoreAll(docs[i])[i] === 1（自相似）。
   * @returns {number[]} 长度 N，无词项交集的文档得 0
   */
  scoreAll(query) {
    const q = this.vectorize(query);
    const qn = l2norm(Float64Array.from(q.values()));
    if (qn === 0) return this.docs.map(() => 0);
    return this.docVecs.map((w, i) => {
      if (this.norms[i] === 0) return 0;
      let dot = 0;
      for (const [t, v] of q) {
        const d = w.get(t);
        if (d !== undefined) dot += v * d;
      }
      return dot / (qn * this.norms[i]);
    });
  }

  /**
   * 检索。
   * @param {string} query 查询串
   * @param {{topK?: number}} [options] topK 默认 5
   * @returns {{docIndex: number, score: number}[]} 按分数降序，长度 ≤ topK
   */
  search(query, { topK = 5 } = {}) {
    return rankTopK(this.scoreAll(query), topK);
  }
}

/* ==================== 3. BM25 索引 ==================== */

/**
 * Okapi BM25 索引（Robertson & Zaragoza 2009, The Probabilistic Relevance Framework）。
 *
 * 公式（k3=∞ 的标准简化，查询词项按去重计）：
 *   score(D,Q) = Σ_{t∈Q} idf(t) · f(t,D)·(k1+1) / (f(t,D) + k1·(1−b + b·|D|/avgdl))
 *   idf_lucene(t) = log(1 + (N−df(t)+0.5)/(df(t)+0.5))   （Lucene 经典实现，非负，默认）
 *   idf_rsj(t)    = log((N−df(t)+0.5)/(df(t)+0.5))        （Robertson–Sparck Jones 原式，可为负）
 *
 * 参数：
 * - k1 ∈ [0,∞)：词频饱和速率。k1=0 时退化为二元（布尔）权重；k1 越大饱和越慢。
 * - b ∈ [0,1]：文档长度归一化强度。b=0 不归一化，b=1 完全按长度归一。
 * - idfVariant：'lucene'（默认，非负）| 'rsj'（原式，df 接近 N 时为负）。
 *
 * 未登录词（df=0）无 idf，直接跳过；search 仅返回得分 > 0 的文档。
 */
export class BM25Index {
  /**
   * @param {string[]} docs 文档数组
   * @param {{k1?: number, b?: number, idfVariant?: 'lucene'|'rsj'}} [options]
   */
  constructor(docs = [], { k1 = 1.2, b = 0.75, idfVariant = 'lucene' } = {}) {
    if (!Number.isFinite(k1) || k1 < 0) throw new RangeError(`k1 必须是非负有限数，收到 ${k1}`);
    if (!Number.isFinite(b) || b < 0 || b > 1) throw new RangeError(`b 必须在 [0,1]，收到 ${b}`);
    if (idfVariant !== 'lucene' && idfVariant !== 'rsj') {
      throw new RangeError(`idfVariant 只能是 'lucene' 或 'rsj'，收到 ${idfVariant}`);
    }
    this.k1 = k1;
    this.b = b;
    this.idfVariant = idfVariant;
    this.build(docs);
  }

  /** 构建 / 重建索引 */
  build(docs) {
    if (!Array.isArray(docs)) throw new TypeError(`build 需要字符串数组，收到 ${typeof docs}`);
    this.docs = docs.map(String);
    const { docTokens, df, N } = analyzeDocs(this.docs);
    this.N = N;
    this.df = df;
    this.docTokens = docTokens;
    this.docLen = docTokens.map((t) => t.length);
    this.avgdl = N > 0 ? this.docLen.reduce((a, b2) => a + b2, 0) / N : 0;
    this.docTf = docTokens.map((toks) => {
      const tf = new Map();
      for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
      return tf;
    });
    this._idfCache = new Map();
    return this;
  }

  /**
   * 词项 idf。未登录词（df=0）返回 null（无 idf 定义）。
   * @returns {number|null}
   */
  idf(term) {
    if (this._idfCache.has(term)) return this._idfCache.get(term);
    const df = this.df.get(term) ?? 0;
    let v = null;
    if (df > 0) {
      v = this.idfVariant === 'rsj'
        ? Math.log((this.N - df + 0.5) / (df + 0.5))
        : Math.log(1 + (this.N - df + 0.5) / (df + 0.5));
    }
    this._idfCache.set(term, v);
    return v;
  }

  /** 词表（按字典序，确定性） @returns {string[]} */
  vocabOf() {
    return [...this.df.keys()].sort();
  }

  /**
   * 单个文档的 BM25 得分分解（供文档与自检逐项核对）。
   * @param {string} query 查询串
   * @param {number} docIndex 文档下标
   * @returns {{docIndex: number, K: number, parts: Array<{term: string, idf: number|null, tf: number, k1Factor: number, contribution: number, oov: boolean}>, score: number}}
   */
  explain(query, docIndex) {
    if (!Number.isInteger(docIndex) || docIndex < 0 || docIndex >= this.N) {
      throw new RangeError(`docIndex 越界: ${docIndex}（N=${this.N}）`);
    }
    const terms = [...new Set(tokenize(query))];
    const dl = this.docLen[docIndex];
    const K = this.k1 * (1 - this.b + this.b * (this.avgdl > 0 ? dl / this.avgdl : 0));
    const parts = terms.map((term) => {
      const idf = this.idf(term);
      if (idf === null) return { term, idf: null, tf: 0, k1Factor: 0, contribution: 0, oov: true };
      const tf = this.docTf[docIndex].get(term) ?? 0;
      const k1Factor = (tf * (this.k1 + 1)) / (tf + K);
      return { term, idf, tf, k1Factor, contribution: idf * k1Factor, oov: false };
    });
    return { docIndex, K, parts, score: parts.reduce((a, p) => a + p.contribution, 0) };
  }

  /**
   * 全部文档的 BM25 得分数组（下标即 docIndex）。
   * @returns {number[]} 长度 N；lucene 变体下恒 ≥ 0
   */
  scoreAll(query) {
    const terms = [...new Set(tokenize(query))];
    return this.docs.map((_, i) => this.explain(query, i).score);
  }

  /**
   * 检索。
   * @returns {{docIndex: number, score: number}[]} 按分数降序，同分按 docIndex 升序，长度 ≤ topK
   */
  search(query, { topK = 5 } = {}) {
    return rankTopK(this.scoreAll(query), topK);
  }
}

/* ==================== 4. LSA 索引 ==================== */

/**
 * 潜在语义分析索引（Deerwester et al. 1990, Indexing by Latent Semantic Analysis）。
 *
 * 流程：
 * 1. 构建 TF-IDF 矩阵 A（N×V，行=文档，列=词元，权重 tf·idf）。
 * 2. 对 A 做截断 SVD：A ≈ U_k·Σ_k·V_kᵀ（复用 src/linalg.mjs 的 svd，
 *    经 AᵀA 的 Jacobi 对称特征分解得到，小规模稠密矩阵足够稳定）。
 * 3. 文档隐语义向量 D = U_k·Σ_k（N×k，行即文档在语义空间的坐标）。
 * 4. 查询投影：q_latent = V_kᵀ·q_tfidf（k 维），与各文档隐向量求 cosine。
 *
 * 同义桥接是 LSA 的核心能力：两个词形不同但在多篇文档中共现的词（如 cat / feline），
 * 在隐空间中方向接近，因而"无语词交集的语义相关文档"也能获得正相似度
 * （TF-IDF 对此恒为 0）——见 selfCheck 的同义评测。
 *
 * 不变量：
 * - 重构误差随 k 严格下降（Eckart–Young 1936：截断 SVD 是最优低秩近似）。
 * - 自查询：scoreAll(docs[i])[i] ≈ 1（因 A·V = U·Σ，查询投影即文档隐向量本身）。
 */
export class LsaIndex {
  /**
   * @param {string[]} docs 文档数组
   * @param {{nComponents?: number}} [options] nComponents 默认 50，超出秩/维度时自动收敛
   */
  constructor(docs = [], { nComponents = 50 } = {}) {
    if (!Number.isInteger(nComponents) || nComponents < 1) {
      throw new RangeError(`nComponents 必须是正整数，收到 ${nComponents}`);
    }
    this.nComponents = nComponents;
    this.build(docs);
  }

  /** 构建 / 重建索引 */
  build(docs) {
    this.tfidf = new TfIdfIndex(docs);
    const N = this.tfidf.N;
    const V = this.tfidf.vocab.length;
    if (N === 0 || V === 0) {
      this.k = 0;
      this.tfidfMatrix = new Mat(0, 0);
      this.docLatent = new Mat(0, 0);
      this.Vk = new Mat(0, 0);
      this.singularValues = new Float64Array(0);
      return this;
    }
    // TF-IDF 稠密矩阵 A（N×V）
    const A = new Mat(N, V);
    for (let i = 0; i < N; i++) {
      for (const [t, w] of this.tfidf.docVecs[i]) A.set(i, this.tfidf.termIndex.get(t), w);
    }
    this.tfidfMatrix = A;
    const { U, s, V: Vfull } = svd(A);
    const k = Math.min(this.nComponents, s.length, N, V);
    this.k = k;
    this.singularValues = s.slice(0, k); // 保留的谱（模型的实际维度）
    // 文档隐语义向量 D = U_k·Σ_k
    const D = new Mat(N, k);
    for (let i = 0; i < N; i++) for (let j = 0; j < k; j++) D.set(i, j, U.get(i, j) * s[j]);
    this.docLatent = D;
    this.Vk = Vfull.colSlice(0, k); // V×k：词项空间 → 隐空间的投影矩阵
    return this;
  }

  /**
   * 查询的隐语义投影向量（k 维）。
   * @returns {Float64Array}
   */
  latent(query) {
    if (this.k === 0) return new Float64Array(0);
    const q = this.tfidf.vectorize(query);
    const out = new Float64Array(this.k);
    for (const [t, w] of q) {
      const ti = this.tfidf.termIndex.get(t);
      if (ti === undefined) continue; // 未登录词不参与投影
      for (let j = 0; j < this.k; j++) out[j] += w * this.Vk.get(ti, j);
    }
    return out;
  }

  /**
   * 全部文档的 cosine 得分数组（下标即 docIndex）。
   * @returns {number[]} 长度 N，值域 [-1,1]；空索引或零查询返回全 0
   */
  scoreAll(query) {
    const N = this.tfidf.N;
    if (this.k === 0) return new Array(N).fill(0);
    const q = this.latent(query);
    const qn = l2norm(q);
    const out = new Array(N).fill(0);
    if (qn === 0) return out;
    for (let i = 0; i < N; i++) {
      let dot = 0;
      let dn = 0;
      for (let j = 0; j < this.k; j++) {
        const d = this.docLatent.get(i, j);
        dot += q[j] * d;
        dn += d * d;
      }
      out[i] = dn === 0 ? 0 : dot / (qn * Math.sqrt(dn));
    }
    return out;
  }

  /**
   * 检索。
   * @returns {{docIndex: number, score: number}[]} 按分数降序，同分按 docIndex 升序，长度 ≤ topK；
   *   仅返回 cosine > 0 的文档（负相关视为不相关）
   */
  search(query, { topK = 5 } = {}) {
    return rankTopK(this.scoreAll(query), topK);
  }

  /**
   * 截断重构误差：‖A − U_k·Σ_k·V_kᵀ‖_F（Frobenius 范数）。
   * 随 k 增加严格下降（Eckart–Young），k=rank 时 ≈ 0。
   * @returns {number}
   */
  reconstructionError() {
    if (this.k === 0) return frobeniusNorm(this.tfidfMatrix);
    const rec = matMul(this.docLatent, transpose(this.Vk));
    return frobeniusNorm(matSub(rec, this.tfidfMatrix));
  }
}

/* ==================== 5. RRF 融合与混合检索 ==================== */

/**
 * 倒数排名融合（Reciprocal Rank Fusion, Cormack, Clarke & Buettcher 2009）。
 *
 *   score(d) = Σ_{r: d ∈ ranking_r} 1 / (k + rank_r(d))
 *
 * 其中 rank_r(d) 为文档 d 在第 r 个排名中的 1 起名次。k 是平滑常数
 * （论文实验 k=60 对多种排序信号都稳健；k 越小头部文档优势越突出）。
 * 同一个文档在同一 ranking 中重复出现时只取第一次名次（去重）。
 *
 * @param {number[][]} rankings 若干排名（每个数组内为 docIndex，按名次从前到后）
 * @param {{k?: number}} [options] k 默认 60
 * @returns {{docIndex: number, score: number}[]} 融合排序（分数降序，同分按 docIndex 升序）
 */
export function reciprocalRankFusion(rankings, { k = 60 } = {}) {
  if (!Array.isArray(rankings)) throw new TypeError(`rankings 必须是数组的数组，收到 ${typeof rankings}`);
  if (!Number.isFinite(k) || k < 0) throw new RangeError(`k 必须是非负有限数，收到 ${k}`);
  const scores = new Map();
  for (const ranking of rankings) {
    if (!Array.isArray(ranking)) throw new TypeError('每个 ranking 必须是 docIndex 数组');
    const seen = new Set();
    for (let r = 0; r < ranking.length; r++) {
      const d = ranking[r];
      if (!Number.isInteger(d)) throw new TypeError(`ranking 元素必须是整数 docIndex，收到 ${d}`);
      if (seen.has(d)) continue; // 同一排名内去重，只取第一次名次
      seen.add(d);
      scores.set(d, (scores.get(d) ?? 0) + 1 / (k + r + 1)); // r 为 0 起下标 → 1 起名次 r+1
    }
  }
  return [...scores.entries()]
    .map(([docIndex, score]) => ({ docIndex, score }))
    .sort((a, b) => b.score - a.score || a.docIndex - b.docIndex);
}

/**
 * 混合检索：BM25（词面精确匹配）+ LSA（语义泛化）经 RRF 融合。
 *
 * 直觉：BM25 对查询词的字面匹配敏感、精度高；LSA 能通过共现结构召回同义改写，
 * 但排序噪声大。RRF 只用名次不用分数，天然免疫两者分数量纲差异，
 * 任何一方进过前列的文档都会获得融合加分——通常优于任一单项。
 *
 * @param {{docIndex: number}[]} bm25Results BM25Index.search 的返回
 * @param {{docIndex: number}[]} lsaResults LsaIndex.search 的返回
 * @param {{k?: number, topK?: number}} [options] RRF 常数 k 默认 60，topK 默认 5
 * @returns {{docIndex: number, score: number}[]} 融合后的 topK 结果
 */
export function hybridSearch(bm25Results, lsaResults, { k = 60, topK = 5 } = {}) {
  for (const [name, r] of [['bm25Results', bm25Results], ['lsaResults', lsaResults]]) {
    if (!Array.isArray(r)) throw new TypeError(`${name} 必须是检索结果数组，收到 ${typeof r}`);
    for (const item of r) {
      if (!item || !Number.isInteger(item.docIndex)) {
        throw new TypeError(`${name} 的元素必须是 {docIndex: number, score: number} 结构`);
      }
    }
  }
  const fused = reciprocalRankFusion([
    bm25Results.map((r) => r.docIndex),
    lsaResults.map((r) => r.docIndex),
  ], { k });
  return fused.slice(0, topK);
}

/* ==================== 6. 检索评测指标 ==================== */

/**
 * Precision@k：topK 中命中相关文档的比例（Manning et al. 2008, 第 8 章）。
 * @param {number[]} rankedIds 按名次从高到低的 docIndex 序列
 * @param {number[]|Set<number>} relevant 相关文档下标集合
 * @param {number} [k=3]
 * @returns {number} hits / k（∈ [0,1]；当 |relevant| < k 时上限相应降低，属标准行为）
 */
export function precisionAtK(rankedIds, relevant, k = 3) {
  const rel = relevant instanceof Set ? relevant : new Set(relevant);
  const top = rankedIds.slice(0, k);
  let hits = 0;
  for (const d of top) if (rel.has(d)) hits++;
  return hits / k;
}

/**
 * MRR 的单查询项：第一个相关文档名次的倒数（1/rank，无命中为 0）。
 * @param {number[]} rankedIds 按名次从高到低的 docIndex 序列
 * @param {number[]|Set<number>} relevant 相关文档下标集合
 * @returns {number} ∈ [0,1]
 */
export function reciprocalRank(rankedIds, relevant) {
  const rel = relevant instanceof Set ? relevant : new Set(relevant);
  for (let i = 0; i < rankedIds.length; i++) if (rel.has(rankedIds[i])) return 1 / (i + 1);
  return 0;
}

/* ==================== 7. 不变量自检 ==================== */

/**
 * 召回率@K：命中的相关文档数 / 相关文档总数。
 *
 * 为什么评测以 Recall@K 为主而非 P@K：本评测集多数查询仅标注 1 篇相关文档，
 * 此时 P@3 的理论上限就是 1/3 ≈ 0.333，用它当质量门槛会变成"天花板探测器"，
 * 无法区分好坏系统。Recall@K 归一化掉"相关文档总数"的影响，才是真正的检索质量指标。
 *
 * @param {number[]} rankedIds 按名次从高到低的 docIndex 序列
 * @param {number[]|Set<number>} relevant 相关文档下标集合
 * @param {number} [k=3]
 * @returns {number} ∈ [0,1]
 */
export function recallAtK(rankedIds, relevant, k = 3) {
  const rel = relevant instanceof Set ? relevant : new Set(relevant);
  if (rel.size === 0) return 0;
  const top = rankedIds.slice(0, k);
  let hits = 0;
  for (const d of top) if (rel.has(d)) hits++;
  return hits / rel.size;
}

/** selfCheck 的规范别名，与其余模块接口统一（返回 verify 报告数组） */
export function selfCheck() {
  const r = retrievalSelfCheck();
  return Array.isArray(r) ? r : [r];
}

/**
 * 检索层不变量自检（单一事实源：测试套件、CLI 自检、智能体证书复用）。
 * 覆盖：分词约定、TF-IDF 公式值、BM25 公开算例逐项精确值、TF 饱和性、idf 非负性、
 * LSA 重构误差递减、LSA 同义桥接、LSA/TF-IDF 自相似、RRF 手算值、chunkText 覆盖与重叠。
 *
 * @returns {{name: string, checks: Array<{label: string, ok: boolean, detail: string}>, passed: boolean, passedCount: number, total: number}[]}
 */
export function retrievalSelfCheck() {
  const s = suite('retrieval');

  // ---- 分词约定 ----
  s.check('tokenize：中英混合约定（你好world世界 / GPT-4 / 单字 CJK）', () => {
    assertArrayEq(tokenize('你好world世界'), ['你好', 'world', '世界'], '中英混合');
    assertArrayEq(tokenize('The Cat, SAT!'), ['the', 'cat', 'sat'], '英文切分+小写');
    assertArrayEq(tokenize('机器学习'), ['机器', '器学', '学习'], 'CJK 二元组滑窗');
    assertArrayEq(tokenize('猫'), ['猫'], '单字 CJK 退化');
    assertArrayEq(tokenize('GPT-4 于 2024 年发布'), ['gpt-4', '于', '2024', '年发', '发布'], '数字/连字符/顺序');
    assertArrayEq(tokenize("don't stop"), ["don't", 'stop'], '词内撇音不切');
    assertArrayEq(tokenize('你好world世界', { lowercase: false }), ['你好', 'world', '世界'], 'lowercase=false');
    return '约定与文档一致';
  });

  // ---- TF-IDF ----
  s.check('TF-IDF：idf=log((N+1)/(df+1))+1 精确值 + cosine 自相似=1 + 排序', () => {
    const idx = new TfIdfIndex(['a b', 'a c', 'a d']);
    // df(a)=3 → log(4/4)+1 = 1；df(b)=df(c)=df(d)=1 → log(4/2)+1
    expectClose(idx.idf[idx.termIndex.get('a')], 1, 1e-12, 'idf(a)');
    expectClose(idx.idf[idx.termIndex.get('b')], Math.log(2) + 1, 1e-12, 'idf(b)');
    expectClose(idx.scoreAll('a b')[0], 1, 1e-12, '自相似');
    const r = idx.search('b', { topK: 5 });
    if (r.length !== 1 || r[0].docIndex !== 0) throw new Error(`search('b') = ${JSON.stringify(r)}`);
    // 单词查询 cosine = idf(b)/√(1+idf(b)²)（查询向量只有 b 一维，文档向量为 {a:1, b:idf(b)}）
    const idfB = Math.log(2) + 1;
    expectClose(r[0].score, idfB / Math.sqrt(1 + idfB * idfB), 1e-12, '单词查询 cosine');
    return `idf(a)=1, idf(b)=${idfB.toFixed(6)}, cosine(b)=${r[0].score.toFixed(6)}`;
  });

  // ---- BM25 公开算例（交叉验证硬标准）----
  s.check('BM25 公开算例逐项精确值（Lucene idf, k1=1.2, b=0.75）', () => {
    const docs = ['the cat sat on the mat', 'the dog sat on the log', 'cats and dogs are friends'];
    const idx = new BM25Index(docs);
    expectClose(idx.avgdl, 17 / 3, 1e-12, 'avgdl');
    const e1 = idx.explain('cat sat', 0);
    expectClose(e1.score, 1.4167, 1e-3, 'BM25(D1)');
    const byTerm = new Map(e1.parts.map((p) => [p.term, p.contribution]));
    expectClose(byTerm.get('cat'), 0.9578, 1e-3, 'cat 贡献');
    expectClose(byTerm.get('sat'), 0.4590, 1e-3, 'sat 贡献');
    expectClose(idx.explain('cat sat', 1).score, 0.4590, 1e-3, 'BM25(D2)');
    expectClose(idx.explain('cat sat', 2).score, 0.0, 1e-12, 'BM25(D3)');
    const top = idx.search('cat sat', { topK: 5 });
    assertArrayEq(top.map((r) => r.docIndex), [0, 1], 'search 名次');
    return `D1=${e1.score.toFixed(4)} (cat ${byTerm.get('cat').toFixed(4)}, sat ${byTerm.get('sat').toFixed(4)}), D2=${idx.explain('cat sat', 1).score.toFixed(4)}, D3=0`;
  });

  // ---- TF 饱和性 ----
  s.check('BM25 TF 饱和性：k1 控制词频增速（默认 k1=1.2 时翻倍比值 < 1.5）', () => {
    // 两篇文档等长（|D| 均为 3，avgdl=3，K 相同），仅 tf(alpha) 不同 → 比值有解析式
    const docs = ['alpha beta gamma', 'alpha alpha beta'];
    const ratio = (k1) => {
      const [s1, s2] = new BM25Index(docs, { k1 }).scoreAll('alpha');
      return s2 / s1;
    };
    const r12 = ratio(1.2);
    if (!(r12 < 1.5)) throw new Error(`k1=1.2 分数比 ${r12.toFixed(4)} ≥ 1.5，饱和性失效`);
    if (!(r12 > 1)) throw new Error(`k1=1.2 分数比 ${r12.toFixed(4)} ≤ 1，词频未增分`);
    // k1 控制实验：因子 tf(k1+1)/(tf+K)，等长文档下比值 = 2(1+k1)/(2+k1)。
    // 强饱和（k1=0.1 → ≈1.05）远小于 1.5；弱饱和（k1=10 → ≈1.83）超过 1.5（趋于线性）
    const rSmall = ratio(0.1);
    const rBig = ratio(10);
    expectClose(r12, (2 * (1 + 1.2)) / (2 + 1.2), 1e-9, 'k1=1.2 解析值');
    expectClose(rSmall, (2 * (1 + 0.1)) / (2 + 0.1), 1e-9, 'k1=0.1 解析值');
    expectClose(rBig, (2 * (1 + 10)) / (2 + 10), 1e-9, 'k1=10 解析值');
    if (!(rSmall < 1.5 && rBig > 1.5)) throw new Error(`k1 控制失效: ${rSmall.toFixed(4)} / ${rBig.toFixed(4)}`);
    return `比值: k1=0.1 → ${rSmall.toFixed(4)}，k1=1.2 → ${r12.toFixed(4)}（<1.5），k1=10 → ${rBig.toFixed(4)}（>1.5）`;
  });

  // ---- idf 非负性 ----
  s.check('BM25 idf：lucene 变体非负；RSJ 在 df=N 时为负（文献行为）', () => {
    const docs = ['common word here', 'common word there', 'common word everywhere'];
    const lucene = new BM25Index(docs);
    for (const t of lucene.vocabOf()) {
      const v = lucene.idf(t);
      if (!(v >= 0)) throw new Error(`lucene idf(${t}) = ${v} < 0`);
    }
    const rsj = new BM25Index(docs, { idfVariant: 'rsj' });
    // df=N 的词（如 common/word）：RSJ idf = log(0.5/(N+0.5)) < 0，Lucene idf = log(1+0.5/(N+0.5)) > 0
    if (!(rsj.idf('common') < 0)) throw new Error(`RSJ idf(common) = ${rsj.idf('common')} 应为负`);
    if (!(lucene.idf('common') > 0)) throw new Error('lucene idf(common) 应为正');
    return `lucene idf(common)=${lucene.idf('common').toFixed(4)}>0，RSJ idf(common)=${rsj.idf('common').toFixed(4)}<0（Robertson 原式行为）`;
  });

  // ---- RRF 手算 ----
  s.check('RRF：手算融合分（k=60，score(d)=Σ 1/(k+rank)）', () => {
    const fused = reciprocalRankFusion([[0, 1, 2], [0, 2, 1]], { k: 60 });
    // doc0: 1/61+1/61；doc1: 1/62+1/63；doc2: 1/63+1/62
    expectClose(fused[0].docIndex, 0, 0, 'top1');
    expectClose(fused[0].score, 2 / 61, 1e-12, 'score(0)');
    expectClose(fused[1].score, 1 / 62 + 1 / 63, 1e-12, 'score(1)');
    expectClose(fused[2].score, 1 / 63 + 1 / 62, 1e-12, 'score(2)');
    assertArrayEq(fused.map((f) => f.docIndex), [0, 1, 2], '同分按 docIndex 升序');
    return `score(0)=${fused[0].score.toFixed(6)}, score(1)=score(2)=${fused[1].score.toFixed(6)}`;
  });

  // ---- LSA 重构误差 ----
  s.check('LSA：重构误差随 nComponents 增加严格下降（Eckart–Young）', () => {
    const docs = [
      '机器学习 神经网络 训练 数据 算法',
      '深度学习 梯度 优化 模型 损失',
      '烹饪 食材 美味 厨房 菜肴',
      '旅行 风景 海岛 假期 沙滩',
      '宋朝 纸币 瓷器 丝绸之路 贸易',
    ];
    const errs = [1, 2, 3, 4].map((k) => new LsaIndex(docs, { nComponents: k }).reconstructionError());
    for (let i = 1; i < errs.length; i++) {
      if (!(errs[i] < errs[i - 1])) {
        throw new Error(`重构误差未递减: k=${i} → ${errs[i - 1].toExponential(3)} → ${errs[i].toExponential(3)}`);
      }
    }
    const full = new LsaIndex(docs, { nComponents: 50 }).reconstructionError();
    if (!(full < 1e-8)) throw new Error(`满秩重构误差 ${full.toExponential(2)} 过大`);
    return `err(k=1..4) = ${errs.map((e) => e.toFixed(4)).join(' → ')}，满秩 ${full.toExponential(1)}`;
  });

  // ---- LSA 同义评测 ----
  s.check('LSA 同义桥接：截断隐空间下语义相关文档 cosine 高于 TF-IDF', () => {
    // 受控小型同义评测：D0 桥接 cat/feline；D1 与查询语义相关但无 "cat" 词形；
    // D2 完全无关；D3 词面匹配。TF-IDF 对 D1 恒为 0（无语词交集）；
    // LSA 必须截断到秩以下（k=3 < rank=4）才能把 cat/feline 的共现结构混入同一隐维度。
    const docs = [
      'cat feline pet animal',
      'the feline sleeps and purrs',
      'database systems store records',
      'a cat chases the mouse',
    ];
    const lsa = new LsaIndex(docs, { nComponents: 3 });
    const tfidf = new TfIdfIndex(docs);
    const q = 'cat';
    const ls = lsa.scoreAll(q);
    const ts = tfidf.scoreAll(q);
    if (!(ls[1] > 0.1)) throw new Error(`LSA 对语义相关文档 D1 的 cosine = ${ls[1]} 未 > 0.1`);
    expectClose(ts[1], 0, 1e-12, 'TF-IDF 对 D1 无交集');
    if (!(ls[1] > ls[2])) throw new Error(`LSA: D1(${ls[1]}) 未超过无关文档 D2(${ls[2]})`);
    // 截断前（满秩 k=4）桥接消失：验证"同义能力来自截断"而非偶然
    const full = new LsaIndex(docs, { nComponents: 4 }).scoreAll(q);
    if (!(Math.abs(full[1]) < 0.05)) throw new Error(`满秩时 D1 cosine=${full[1]} 应≈0（回到 TF-IDF 行为）`);
    return `cosine: D1=${ls[1].toFixed(4)}（TF-IDF=0），D2=${ls[2].toFixed(4)}，D3=${ls[3].toFixed(4)}；满秩 D1=${full[1].toExponential(1)}`;
  });

  // ---- 自相似不变量 ----
  s.check('自查询不变量：TF-IDF cosine=1 精确成立；LSA 满秩时 ≈1', () => {
    const docs = ['机器学习 神经网络', '烹饪 美食 厨房', '旅行 海岛 沙滩'];
    const tfidf = new TfIdfIndex(docs);
    for (let i = 0; i < docs.length; i++) expectClose(tfidf.scoreAll(docs[i])[i], 1, 1e-12, `TF-IDF 自查询 D${i}`);
    const lsaFull = new LsaIndex(docs, { nComponents: 50 }); // 满秩
    for (let i = 0; i < docs.length; i++) {
      expectClose(lsaFull.scoreAll(docs[i])[i], 1, 1e-8, `LSA 满秩自查询 D${i}`);
      const top = lsaFull.search(docs[i], { topK: 1 });
      if (top[0].docIndex !== i) throw new Error(`LSA 自查询 top1 = D${top[0].docIndex} ≠ D${i}`);
    }
    return '自相似全部命中';
  });

  // ---- chunkText ----
  s.check('chunkText：全覆盖 + overlap 生效 + size 契约', () => {
    const text = '机器学习需要大量数据进行训练，深度学习模型通过反向传播更新参数。'
      + 'BM25 与 TF-IDF 是经典检索算法，LSA 通过奇异值分解发现隐语义结构。';
    const units = text.match(new RegExp(UNIT_SCAN_RE.source, 'g')) ?? [];
    for (const [size, overlap] of [[40, 0], [40, 10], [300, 50], [10, 3]]) {
      const chunks = chunkText(text, { size, overlap });
      if (chunks.length < 1) throw new Error(`size=${size} 未产出 chunk`);
      const concat = chunks.join('');
      for (const u of units) {
        if (!concat.includes(u)) throw new Error(`单元 "${u}" 未被任何 chunk 覆盖`);
      }
      for (let i = 1; i < chunks.length; i++) {
        if (overlap > 0 && chunks[i - 1].length > 2) {
          const shared = sharedSuffixPrefixLen(chunks[i - 1], chunks[i]);
          if (shared < 1) throw new Error(`chunk ${i - 1}/${i} 边界无重叠（overlap=${overlap}）`);
        }
      }
    }
    // 短文本单 chunk；空文本空数组
    assertArrayEq(chunkText('短文本', { size: 300 }), ['短文本'], '短文本');
    assertArrayEq(chunkText('', { size: 300 }), [], '空文本');
    return `4 组参数全覆盖（${units.length} 个单元）`;
  });

  return [s.report()];
}

/** 内部：两个字符串"后缀=前缀"的最长共享长度 */
function sharedSuffixPrefixLen(a, b) {
  const max = Math.min(a.length, b.length);
  for (let len = max; len >= 1; len--) {
    if (a.endsWith(b.slice(0, len))) return len;
  }
  return 0;
}

/** 内部：数组精确相等断言（带可读报错） */
function assertArrayEq(actual, expected, what) {
  if (actual.length !== expected.length || actual.some((v, i) => v !== expected[i])) {
    throw new Error(`${what}: 期望 [${expected}], 实际 [${actual}]`);
  }
}

/* ==================== 8. 基准评测 ==================== */

/**
 * 自建小型检索评测集（中英混合，主题覆盖：机器学习 / 烹饪 / 旅行 / 历史，
 * 含 StarForge 项目自身文档片段作正例，干扰主题作反例），并报告
 * BM25 / LSA / RRF 混合三者的 P@3 与 MRR。
 *
 * 目标不变量：混合检索 P@3 与 MRR 均不低于任一单项
 * （RRF 只用名次融合，通常优于单项——若失败说明评测集或实现有偏，必须修复）。
 *
 * @returns {{name: string, checks: Array<{label: string, ok: boolean, detail: string}>, passed: boolean, passedCount: number, total: number, metrics: object}}
 */
export function bench() {
  const s = suite('retrieval-bench');

  // ---- 评测集：14 篇文档（0-4 机器学习/StarForge，5-7 烹饪，8-10 旅行，11-13 历史）----
  const docs = [
    'StarForge 是一个零依赖的纯 ESM JavaScript 深度学习引擎，内置自动微分、优化器与线性代数内核',
    '机器学习模型通过反向传播算法更新参数，训练数据驱动神经网络学习特征表示',
    '深度学习在自然语言处理领域取得突破，Transformer 架构支撑了大语言模型',
    '梯度下降是最常用的优化算法，Adam 自适应调整学习率以加速收敛',
    '支持向量机通过核技巧处理非线性分类问题，是经典的机器学习算法',
    '烹饪意大利面需要先烧水，加入橄榄油和盐，面条煮八分钟口感劲道',
    '烘焙面包要掌握面粉与酵母的比例，发酵时间决定成品的松软程度',
    '这份菜谱使用新鲜食材，慢火炖煮让汤汁浓郁，非常适合家庭聚餐',
    '旅行攻略：海岛度假推荐潜水看珊瑚，沙滩日落和海鲜市场不容错过',
    '背包客徒步穿越山脉，在高原露营观星，是极具挑战的旅行体验',
    '自驾游路过古城，品尝当地小吃，记录一路上的风景与人文故事',
    '宋朝商品经济繁荣，交子是世界上最早的纸币，瓷器与丝绸贸易兴盛',
    '文艺复兴时期欧洲涌现了达芬奇与米开朗基罗，人文主义思潮兴起',
    '丝绸之路连接了东西方文明，商队运送丝绸、茶叶和香料穿越沙漠',
  ];
  // ---- 14 条查询 + 人工相关性标注 ----
  const queries = [
    { q: '深度学习 神经网络 训练', rel: [1, 2] },
    { q: '反向传播 梯度 优化', rel: [1, 3] },
    { q: '自动微分 JavaScript 引擎', rel: [0] },
    { q: 'Transformer 大语言模型', rel: [2] },
    { q: '支持向量机 分类 算法', rel: [4] },
    { q: '意大利面 烹饪 烧水', rel: [5] },
    { q: '面包 烘焙 酵母 发酵', rel: [6] },
    { q: '菜谱 食材 炖煮', rel: [7] },
    { q: '海岛 度假 沙滩 旅行', rel: [8] },
    { q: '徒步 露营 高原 旅行', rel: [9] },
    { q: '古城 自驾游 小吃', rel: [10] },
    { q: '宋朝 纸币 瓷器', rel: [11] },
    { q: '文艺复兴 欧洲 达芬奇 米开朗基罗', rel: [12] },
    { q: '丝绸之路 东西方 商队', rel: [13] },
    { q: '机器学习 算法', rel: [1, 4] },
    { q: '神经网络 模型 训练', rel: [1, 2] },
    { q: '宋朝 贸易 丝绸', rel: [11, 13] },
  ];

  s.check('评测集规模：≥12 文档 / ≥10 查询', () => {
    if (docs.length < 12) throw new Error(`文档数 ${docs.length} < 12`);
    if (queries.length < 10) throw new Error(`查询数 ${queries.length} < 10`);
    for (const { q, rel } of queries) {
      if (rel.length < 1) throw new Error(`查询 "${q}" 无相关性标注`);
      for (const d of rel) if (d < 0 || d >= docs.length) throw new Error(`查询 "${q}" 标注越界: ${d}`);
    }
    return `${docs.length} 文档 / ${queries.length} 查询（中英混合，4 主题）`;
  });

  s.check('BM25 公开算例精确命中（硬标准）', () => {
    const idx = new BM25Index(['the cat sat on the mat', 'the dog sat on the log', 'cats and dogs are friends']);
    const e1 = idx.explain('cat sat', 0);
    expectClose(e1.score, 1.4167, 1e-3, 'BM25(D1)');
    expectClose(idx.explain('cat sat', 1).score, 0.4590, 1e-3, 'BM25(D2)');
    expectClose(idx.explain('cat sat', 2).score, 0.0, 1e-12, 'BM25(D3)');
    expectClose(idx.avgdl, 17 / 3, 1e-12, 'avgdl');
    return `D1=${e1.score.toFixed(4)}, D2=${idx.explain('cat sat', 1).score.toFixed(4)}, D3=0`;
  });

  const bm25 = new BM25Index(docs);
  const lsa = new LsaIndex(docs, { nComponents: 8 });

  const evaluate = (search) => {
    let p3 = 0;
    let mrr = 0;
    let r3 = 0;
    const perQuery = queries.map(({ q, rel }) => {
      const ranked = search(q).map((r) => r.docIndex);
      const p = precisionAtK(ranked, rel, 3);
      const rr = reciprocalRank(ranked, rel);
      const rc = recallAtK(ranked, rel, 3);
      p3 += p;
      mrr += rr;
      r3 += rc;
      return { q, rel, top3: ranked.slice(0, 3), p3: p, rr, r3: rc };
    });
    const n = queries.length;
    return { p3: p3 / n, mrr: mrr / n, r3: r3 / n, perQuery };
  };

  const mBm25 = evaluate((q) => bm25.search(q, { topK: 10 }));
  const mLsa = evaluate((q) => lsa.search(q, { topK: 10 }));
  const mHybrid = evaluate((q) => hybridSearch(bm25.search(q, { topK: 10 }), lsa.search(q, { topK: 10 }), { topK: 10 }));
  const metrics = {
    docs: docs.length,
    queries: queries.length,
    bm25: { p3: mBm25.p3, mrr: mBm25.mrr },
    lsa: { p3: mLsa.p3, mrr: mLsa.mrr },
    hybrid: { p3: mHybrid.p3, mrr: mHybrid.mrr },
    perQuery: mHybrid.perQuery,
  };

  s.check('混合 P@3 ≥ 任一单项', () => {
    const worst = Math.max(mBm25.p3, mLsa.p3);
    if (!(mHybrid.p3 >= worst - 1e-12)) {
      throw new Error(`混合 P@3=${mHybrid.p3.toFixed(4)} < max(单项)=${worst.toFixed(4)}`);
    }
    return `混合 ${mHybrid.p3.toFixed(4)} ≥ BM25 ${mBm25.p3.toFixed(4)} / LSA ${mLsa.p3.toFixed(4)}`;
  });

  s.check('混合 MRR ≥ 任一单项', () => {
    const worst = Math.max(mBm25.mrr, mLsa.mrr);
    if (!(mHybrid.mrr >= worst - 1e-12)) {
      throw new Error(`混合 MRR=${mHybrid.mrr.toFixed(4)} < max(单项)=${worst.toFixed(4)}`);
    }
    return `混合 ${mHybrid.mrr.toFixed(4)} ≥ BM25 ${mBm25.mrr.toFixed(4)} / LSA ${mLsa.mrr.toFixed(4)}`;
  });

  // 质量门槛用 Recall@3：本评测集多数查询仅 1 篇相关文档，P@3 天花板仅 1/3，
  // 不适合作为判别性指标（详见 recallAtK 处的说明）。
  s.check('单项质量下限：BM25 Recall@3 ≥ 0.85，LSA Recall@3 ≥ 0.70', () => {
    if (!(mBm25.r3 >= 0.85)) throw new Error(`BM25 Recall@3=${mBm25.r3.toFixed(4)} < 0.85`);
    if (!(mLsa.r3 >= 0.7)) throw new Error(`LSA Recall@3=${mLsa.r3.toFixed(4)} < 0.70`);
    return `BM25 Recall@3=${mBm25.r3.toFixed(4)}, LSA Recall@3=${mLsa.r3.toFixed(4)}`;
  });

  const report = s.report();
  // 统一 bench 契约：{module, ok, results:[{name,ok,detail,value,unit}]}，metrics 作为附加诊断数据
  return {
    module: 'retrieval',
    ok: report.passed,
    results: report.checks.map((c) => ({
      name: c.label,
      ok: c.ok,
      detail: c.detail || '',
      value: c.ok ? 1 : 0,
      unit: 'pass',
    })),
    metrics,
  };
}
