# 线性代数与数值内核（src/linalg.mjs）

> StarForge 的 L1 核心数学层：稠密矩阵、直接法解方程、矩阵分解与数值稳定原语，全部零依赖实现。
> 本文件所有类名、函数名与源码严格一致，文末不变量可在 `src/selfcheck.mjs` 的 `linalgSelfCheck()` 中逐条对上。

作者：晨星

---

## 0. 确定性随机源 `createRng(seed)`

### 直觉

整个引擎**不允许出现 `Math.random()`**。所有随机性（数据划分、权重初始化、批次洗牌、聚类播种、环境动力学）必须从同一个可复现的水龙头里放出来——`createRng(seed)` 就是这个水龙头。它像一个"可倒带的骰子"：同一个种子必然吐出同一串数字。

### 数学核心

- 整数哈希用 **mulberry32**：32 位状态 `a`，每次 `a += 0x6D2B79F5`，再做三轮 xor-shift 混淆，最后除以 $2^{32}$ 归一到 $[0,1)$。
- 正态分布用 **Box–Muller**：取两个独立的 $u, v \sim U(0,1)$，返回 $\sqrt{-2\ln u}\cdot\cos(2\pi v)$。
- 洗牌用 **Fisher–Yates**（从后往前，逐个与 $[0,i]$ 内的随机位置交换）。

```
function createRng(seed = 42):
  a ← seed >>> 0
  next():
    a ← (a + 0x6D2B79F5) | 0
    t ← imul(a ^ (a >>> 15), 1 | a)
    t ← (t + imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 2^32
  uniform(lo, hi) ← lo + (hi - lo) * next()
  normal()        ← √(-2·ln u)·cos(2π v)      # u, v ← next()，且 u,v ≠ 0
  randint(n)      ← floor(next() * n)
  shuffle(arr)    ← Fisher–Yates，反向遍历
```

### 复杂度

| 操作 | 时间 | 空间 |
|---|---|---|
| `next()` / `uniform()` / `randint()` | $O(1)$ | $O(1)$ |
| `normal()` | $O(1)$（期望，拒绝采样 u≠0） | $O(1)$ |
| `shuffle(arr)` | $O(n)$ | $O(1)$ 原地 |

### 经典论文

- Mulberry32 属于 **Lehmer / LCG 族**伪随机数生成器：Lehmer, D.H. (1951), *Mathematical methods in large-scale computing units*, Symp. on Large-Scale Digital Calculating Machinery。
- Box, G.E.P. & Muller, M.E. (1958), *A Note on the Generation of Random Normal Deviates*, Annals of Mathematical Statistics 29(2): 610–611。
- Fisher–Yates 洗牌：Fisher, R.A. & Yates, F. (1938), *Statistical Tables for Biological, Agricultural and Medical Research*；Durstenfeld, R. (1964), *Algorithm 235: Random permutation*, CACM 7(7)。

### 常见坑

1. **不要用 `Math.random()` 做实验**。一次替换就会让所有基准失去可比性，也会让"同种子同结果"的自检失败。
2. **`normal()` 必须拒绝 $u = 0$**，`ln 0 = -∞` 会直接污染整条数据流；源码用 `while (u === 0) u = next()` 处理。
3. **种子要按用途区分**（如数据集 seed 与模型 seed 分开），否则两处随机流会互相"推动"，改一个实验参数会牵连到不相干的模块。
4. **`reset()` 不等于重新播种**。例如 `BernoulliBandit.reset()` 只清零统计量、不重置 RNG 流；需要完整复现请新建实例。
5. **顺序即依赖**：任何一次 `next()` 调用顺序的改变都会改变后续全部结果。新增随机调用点必须放在既有调用点之后，并重新记录基准。

### 可验证不变量

| 检查项（`linalgSelfCheck()`） | 期望 |
|---|---|
| `RNG 同种子同序列（可复现性）` | 两个 `createRng(123)` 连续 100 次 `next()` 逐一相等 |

---

## 1. 稠密矩阵 `Mat` 与基础运算

### 直觉

`Mat` 就是把一张二维表格压成一条**行主序 `Float64Array`**：第 `i` 行第 `j` 列躺在下标 `i * cols + j`。为什么压平？因为一条连续内存讓矩阵乘法的内层循环可以线性扫描，缓存命中率远高于"数组的数组"；`Float64Array` 还保证全引擎统一的双精度语义。

矩阵乘法 `matMul` 用 **i-k-j 循环序**并**跳过零元素**：稀疏矩阵（如 one-hot 特征、树模型的分块矩阵）能白捡大量加速。

### 数学核心

$$C = A\cdot B,\quad C_{ij} = \sum_{k=1}^{K} A_{ik}B_{kj}$$

i-k-j 序的增量写法（每个 $A_{ik}$ 只取一次，对 $B$ 的第 $k$ 行做乘加）：

$$C_{i,:} \mathrel{+}= A_{ik}\cdot B_{k,:}\quad (k = 1\ldots K)$$

其余原语：`transpose`（转置）、`matAdd` / `matSub` / `matScale`（逐元素）、`matHadamard`（Hadamard 积 $A\odot B$）、`frobeniusNorm`（$\|A\|_F = \sqrt{\sum A_{ij}^2}$）、`l2norm`（向量 $\|x\|_2$）、`matVec`（矩阵-向量积）。

```
function matMul(A, B):                 # A: m×k, B: k×n
  C ← zeros(m, n)
  for i in 0..m-1:
    for p in 0..k-1:
      av ← A[i, p];  if av == 0: continue      # 零值跳过
      for j in 0..n-1:
        C[i, j] += av * B[p, j]                 # B 的第 p 行线性扫描
  return C

function matVec(A, x):                 # A: m×n
  y ← zeros(m)
  for i in 0..m-1:
    for j in 0..n-1:  y[i] += A[i, j] * x[j]
  return y
```

### 复杂度

| 操作 | 时间 | 空间 |
|---|---|---|
| `matMul(A:m×k, B:k×n)` | $O(mkn)$；稀疏时按非零元数显著下降 | $O(mn)$ |
| `matVec` | $O(mn)$ | $O(m)$ |
| `transpose` / `matAdd` / `matSub` / `matScale` / `matHadamard` | $O(mn)$ | $O(mn)$ |
| `row(i)` / `col(j)` / `clone()` / `colSlice()` | $O(n)$ / $O(m)$ / $O(mn)$ / $O(r\cdot k)$ | 同阶 |
| `Mat` 常驻内存 | — | $O(mn)$ 个 float64 |

### 经典论文

- Golub, G.H. & Van Loan, C.F., *Matrix Computations*, 4th ed., Johns Hopkins University Press（矩阵乘法循环序与分块的缓存分析，第 1 章）。
- Strassen, V. (1969), *Gaussian Elimination is not Optimal*, Numerische Mathematik 13: 354–356（注：本实现用朴素三重循环，未采用 Strassen；小规模矩阵下朴素法常数更小）。

### 常见坑

1. **行主序别搞反**：`Mat.from2D(arr)` 要求 `arr` 是矩形，`get/set` 一律 `(i, j)` 行列序；从别的库迁移数据时最容易在转置上出错。
2. **`col()` 比 `row()` 慢**：按列取数在行主序下是跨步访问。热路径（如 PCA 的协方差累加）应改写成一次性遍历。
3. **`matMul` 的零值跳过只在精确 `=== 0` 时生效**。$10^{-17}$ 这种"数值零"不会被跳过，别指望它对稠密结果矩阵加速。
4. **`Float64Array` 与 `Array` 语义不同**：`new Float64Array(n)` 已清零，且越界写静默失败（`Mat` 内部用 `Float64Array`，务必用 `get/set`）。
5. **`data` 可能共享**：`new Mat(rows, cols, data)` 只在 `data.length === rows*cols` 时复用，否则新建；要独立副本请用 `clone()`。

### 可验证不变量

| 检查项（`linalgSelfCheck()`） | 容差 |
|---|---|
| `matMul 与朴素三重循环一致` | 逐元素 `expectClose(..., 1e-10)` |

---

## 2. 解线性方程组 `solveLinear` / `inverse` / `det`

### 直觉

解 $Ax = b$ 最直接的办法是高斯消元：把矩阵一步步"下三角化"，再回代。但纯高斯消元遇到主元接近 0 就会把误差放大到天上去。加一步**部分选主元**（每列挑绝对值最大的行换上来）就把这个致命伤堵住了——这是数值线性代数里性价比最高的一行代码。`inverse(A)` 就是解 $AX = I$；`det(A)` 顺路把消元后的对角元乘起来（换行要翻符号）。

### 数学核心

部分选主元 LU：对每一列 `col`，选 $p = \arg\max_{r \ge col}|A_{r,col}|$，交换第 `col` 行与第 `p` 行，然后

$$f = \frac{A_{r,col}}{A_{col,col}},\qquad A_{r,:} \mathrel{-}= f\cdot A_{col,:},\quad B_{r,:} \mathrel{-}= f\cdot B_{col,:}$$

回代（支持多右端项 $B$ 为矩阵）：

$$x_{col} = \frac{1}{A_{col,col}}\Big(B_{col} - \sum_{k > col} A_{col,k}\,x_k\Big)$$

行列式 $\det A = (-1)^{\text{swap}}\prod_{col} A_{col,col}$。

一阶条件（最小二乘的金标准）：$X^\top (y - X\hat w) = 0$，由 `verify.mjs` 的 `residualOrthogonality` 度量。

```
function solveLinear(A, B):            # A: n×n, B: n×m（或向量）
  for col in 0..n-1:
    piv ← argmax_{r ≥ col} |A[r, col]|               # 部分选主元
    if |A[piv, col]| < 1e-14: throw '矩阵奇异或接近奇异'
    swap rows col, piv in A and B
    for r in col+1..n-1:
      f ← A[r, col] / A[col, col]
      A[r, col:] -= f * A[col, col:]
      B[r, :]    -= f * B[col, :]
  for col in n-1 .. 0:                                # 回代
    B[col, :] ← (B[col, :] - Σ_{k>col} A[col,k]·B[k,:]) / A[col, col]
  return B
```

### 复杂度

| 操作 | 时间 | 空间 |
|---|---|---|
| `solveLinear(A:n×n, B:n×m)` | $O(n^3 + n^2 m)$ | $O(n^2 + nm)$（原地） |
| `inverse(A)` | $O(n^3)$（解 $n$ 个右端项） | $O(n^2)$ |
| `det(A)` | $O(n^3)$ | $O(n^2)$ |

### 经典论文

- Wilkinson, J.H. (1961), *Error Analysis of Direct Methods of Matrix Inversion*, J. ACM 8(3): 281–330（部分选主元的向后稳定性分析）。
- Golub & Van Loan, *Matrix Computations*, §3.4（高斯消元与选主元）。
- Higham, N.J. (2002), *Accuracy and Stability of Numerical Algorithms*, SIAM, §9。

### 常见坑

1. **永远不要真的去算 $A^{-1}$ 再乘 $b$**。数值误差更大且慢 3 倍；直接 `solveLinear(A, b)`。
2. **`Ridge` 里的 $\lambda$ 只加在非偏置列**（源码：`for (let j = 1; ...)`），把偏置也正则会让模型整体平移。
3. **奇异性阈值 $10^{-14}$ 是"接近奇异"而非"数学奇异"**。条件数很大的矩阵（如高度相关特征）会通过阈值但结果仍不可信，需配合正则化。
4. **`det` 会溢出**：$n$ 大时 $\prod$ 对角元可能超出 double 范围，判断奇异性应看选主元时的主元绝对值而非 `det` 是否接近 0。
5. **多右端项要一次传矩阵**，比循环调用单向量少做一次 $O(n^3)$ 消元。

### 可验证不变量

| 检查项（`linalgSelfCheck()`） | 容差 |
|---|---|
| `solveLinear 残差 ‖Ax−b‖≈0` | `expectClose(err, 0, 1e-8)` |
| `A·A⁻¹ = I` | `expectClose(err, 0, 1e-8)` |

---

## 3. Cholesky 分解 `cholesky` / `isPositiveDefinite`

### 直觉

如果一个对称矩阵是"正定"的（可以想象成一个碗口向上的抛物面），就能把它拆成一个下三角矩阵 $L$ 与其转置的乘积 $A = LL^\top$。这相当于对矩阵开平方。有了它，高斯分布采样、GMM 的密度计算都能一步到位。反过来，**能不能成功分解就是正定性的判定**——不用去算特征值。

### 数学核心

逐元素递推（$i$ 从 0 到 $n-1$，$j$ 从 0 到 $i$）：

$$s = A_{ij} - \sum_{k < j} L_{ik}L_{jk},\qquad L_{ij} = \begin{cases} \sqrt{s}, & i = j\\[4pt] s / L_{jj}, & i > j \end{cases}$$

若 $i = j$ 时 $s \le 0$，矩阵非正定，抛错（这正是 `isPositiveDefinite` 的实现：try/catch）。

对数行列式与二次型（GMM 用）：

$$\log|\Sigma| = 2\sum_i \log L_{ii},\qquad (x-\mu)^\top\Sigma^{-1}(x-\mu) = \|z\|^2,\ \ Lz = x - \mu$$

```
function cholesky(A):                  # A 对称正定 n×n
  L ← zeros(n, n)
  for i in 0..n-1:
    for j in 0..i:
      s ← A[i, j] - Σ_{k<j} L[i,k]·L[j,k]
      if i == j:
        if s <= 0: throw '矩阵非正定，Cholesky 分解失败'
        L[i, j] ← √s
      else:
        L[i, j] ← s / L[j, j]
  return L
```

### 复杂度

| 操作 | 时间 | 空间 |
|---|---|---|
| `cholesky(A:n×n)` | $O(n^3/3)$ | $O(n^2)$ |
| `isPositiveDefinite` | 同 `cholesky`（失败即提前抛错） | 同 |

### 经典论文

- Cholesky, A.-L. (1910), *Sur la résolution numérique des systèmes d'équations linéaires*（原始手稿方法）。
- Golub & Van Loan, *Matrix Computations*, §4.2；Trefethen, L.N. & Bau, D. (1997), *Numerical Linear Algebra*, SIAM, 讲稿 23。

### 常见坑

1. **对称性不是自动保证的**。浮点累加会让 $A_{ij}$ 与 $A_{ji}$ 差一个 ulp；PCA/GMM 的协方差累加应对称写入（`cluster.mjs` 中同时写 `S(p,q)` 与 `S(q,p)`）。
2. **半正定不等于正定**。$s = 0$ 会触发抛错；实务上加一个 `regCovar`（GMM 默认 `1e-6`）到对角线即可。
3. **`isPositiveDefinite` 用异常做控制流**，在热路径里反复 try/catch 有开销；判定一次后应缓存结论。
4. **别用 Cholesky 判"接近正定"**：只要有一个 $s \le 0$ 就整体失败，无法给出"有多不正定"的度量。要看程度请用特征分解。
5. **尺度敏感**：特征量纲差 $10^6$ 倍时，浮点舍入会让本该正定的矩阵失败。先做 z-score 标准化。

### 可验证不变量

| 检查项（`linalgSelfCheck()`） | 容差 |
|---|---|
| `Cholesky: L·Lᵀ=A 且正定性判定正确` | 重构 `expectClose(err, 0, 1e-8)`；负定矩阵 $\begin{pmatrix}1&2\\2&1\end{pmatrix}$ 必须被判为非正定 |

---

## 4. Householder QR 分解 `qr`

### 直觉

QR 把一个矩阵拆成"一个正交阵 $Q$ 乘一个上三角阵 $R$"。正交阵就是"只旋转、不拉伸"的变换，它不会把误差放大——这是 QR 比高斯消元更稳的根本原因。Householder 的做法是：对每一列，找一面"镜子"（一次反射），把该列对角线以下的部分一次性照到零。比逐个元素消零的 Givens 旋转更少循环。

### 数学核心

对第 $k$ 列 $x = R_{k:,k}$，取 $\alpha = -\operatorname{sign}(R_{kk})\|x\|_2$（符号与 $R_{kk}$ 相反以避免相减抵消），令

$$v = x - \alpha e_1,\qquad H = I - \frac{2vv^\top}{v^\top v}$$

$H$ 是对称正交的反射矩阵。左乘 $H$ 把 $x$ 变为 $\alpha e_1$（下段全零）：

$$R \leftarrow H R,\qquad Q \leftarrow QH$$

源码用 $\beta = 2 / \|v\|^2$ 的**紧凑形式**（不显式构造 $H$），每次更新 $R$ 的一列、$Q$ 的一行都是一次点积加一次乘加。

```
function qr(A):                        # A: m×n，返回 Q: m×m, R: m×n
  R ← copy(A);  Q ← I_m
  for k in 0..min(m-1, n)-1:
    x ← R[k:, k];  xnorm ← ‖x‖₂
    if xnorm < 1e-300: continue                     # 整列为零，跳过
    α ← -xnorm if R[k,k] > 0 else +xnorm            # 防抵消
    v ← x - α·e₁;  β ← 2 / ‖v‖²
    for j in 0..n-1:                                # R ← H·R
      dot ← Σ_{i≥k} v[i-k]·R[i, j]
      for i in k..m-1: R[i, j] -= β·dot·v[i-k]
    for i in 0..m-1:                                # Q ← Q·H
      dot ← Σ_{t≥k} Q[i, t]·v[t-k]
      for t in k..m-1: Q[i, t] -= β·dot·v[t-k]
  return {Q, R}
```

### 复杂度

| 操作 | 时间 | 空间 |
|---|---|---|
| `qr(A:m×n)`（返回完整 $Q$：$m\times m$） | $O(m^2 n)$ | $O(m^2 + mn)$ |

### 经典论文

- Householder, A.S. (1958), *Unitary Triangularization of a Nonsymmetric Matrix*, J. ACM 5(4): 339–342。
- Golub & Van Loan, *Matrix Computations*, §5.2（Householder QR 与向后稳定性）。
- Trefethen & Bau, *Numerical Linear Algebra*, 讲稿 7–10。

### 常见坑

1. **$\alpha$ 的符号必须与 $R_{kk}$ 相反**。取同号会让 $v_0 = R_{kk} - \alpha$ 变成两个相近数相减，灾难性抵消，分解精度崩塌。
2. **整列为零要 `continue`**（源码 `xnorm < 1e-300`），否则 $\|v\|^2 = 0$ 除零得到 NaN。秩亏矩阵会得到含零行的 $R$。
3. **$Q$ 是 $m\times m$ 的完整正交阵**，不是薄 $Q$。大规模场景下内存是 $m^2$，若只需解方程应改走 `solveLinear`。
4. **QR 不保证 $R$ 对角元为正**，需要规范正交基时要手动翻符号。
5. **反射是逐列累积的**，中途修改 $R$ 会破坏后续列；不要在分解过程中读写中间结果。

### 可验证不变量

| 检查项（`linalgSelfCheck()`） | 容差 |
|---|---|
| `QR: QᵀQ=I，QR=A，R 上三角` | $Q^\top Q = I$ 误差 `1e-8`；$QR = A$ 误差 `1e-8`；$R$ 下三角元素 `expectClose(..., 0, 1e-10)` |

---

## 5. 对称矩阵特征分解 `jacobiEigen`

### 直觉

对称矩阵一定可以被"转到"对角形：找一组互相垂直的方向（特征向量），使得矩阵在这些方向上只做伸缩（特征值）。Jacobi 的做法极其朴素——反复挑一对"没对齐"的坐标轴 $(p,q)$，在这两个方向构成的平面内转一个小角度，把它们的交叉项 $A_{pq}$ 转成 0。转完一轮再来一轮，交叉项的总量越来越小，最终收敛。它慢，但**精度高、天生并行、且对重特征值极其稳健**——小规模矩阵（PCA 的协方差通常 $d \le 100$）它是首选。

### 数学核心

对 $(p, q)$ 平面，令 $\tau = \dfrac{A_{qq} - A_{pp}}{2A_{pq}}$，取较小根的 $t$ 公式避免抵消：

$$t = \frac{\operatorname{sign}(\tau)}{|\tau| + \sqrt{1 + \tau^2}},\qquad c = \frac{1}{\sqrt{1+t^2}},\qquad s = tc$$

旋转后

$$A'_{pp} = c^2A_{pp} - 2scA_{pq} + s^2A_{qq},\quad A'_{qq} = s^2A_{pp} + 2scA_{pq} + c^2A_{qq},\quad A'_{pq} = A'_{qp} = 0$$

其余行列按二维旋转更新；特征向量矩阵 $V$ 同步右乘同一旋转。收敛判据用**非对角元素的 Frobenius 范数**：

$$\sqrt{\sum_{p<q} A_{pq}^2} < \text{tol}$$

最后按特征值**降序**排列（`values` 与 `vectors` 的列同步重排），这是 PCA 取前 $k$ 个主成分的前提。

```
function jacobiEigen(A, {maxSweeps=100, tol=1e-12}):
  require isSymmetric(A, 1e-10)
  V ← I_n;  A ← copy(A)
  for sweep in 0..maxSweeps-1:
    off ← √(Σ_{p<q} A[p,q]²)
    if off < tol: break
    for p in 0..n-1:
      for q in p+1..n-1:
        if |A[p,q]| < 1e-300: continue
        τ ← (A[q,q] - A[p,p]) / (2·A[p,q])
        t ← sign(τ) / (|τ| + √(1+τ²));  c ← 1/√(1+t²);  s ← t·c
        更新 A 的第 p、q 行与列（对称写回），A[p,q] ← A[q,p] ← 0
        更新 V 的第 p、q 列
  idx ← argsort(diag(A), descending)
  return {values: diag(A)[idx], vectors: V[:, idx]}
```

### 复杂度

| 操作 | 时间 | 空间 |
|---|---|---|
| 单次 sweep（$n^2/2$ 个旋转，每个 $O(n)$） | $O(n^3)$ | — |
| `jacobiEigen`（$S$ 轮 sweep，典型 $S \approx 6\sim 10$） | $O(S n^3)$ | $O(n^2)$ |

### 经典论文

- Jacobi, C.G.J. (1846), *Über ein leichtes Verfahren, die in der Theorie der Säcularstörungen vorkommenden Gleichungen numerisch aufzulösen*, Crelle's Journal 30: 51–94。
- Golub & Van Loan, *Matrix Computations*, §8.5（Jacobi 方法的精度优势：相对误差优于 QR 迭代）。
- Demmel, J. & Veselić, K. (1992), *Jacobi's method is more accurate than QR*, SIAM J. Matrix Anal. Appl. 13(4): 1204–1245。
- Numerical Recipes, 3rd ed., §11.1（$t$ 公式的稳定写法）。

### 常见坑

1. **必须先确认对称性**。源码对非对称矩阵直接抛错（`isSymmetric(a, 1e-10)`），但容差是固定的——浮点累加产生的 $10^{-16}$ 级非对称不会触发。
2. **收敛判据用非对角范数而非"本轮有无变化"**。后者在重特征值附近会过早退出。
3. **`maxSweeps = 100` 是保险丝**。正常 5–10 轮就收敛；如果你的矩阵跑到 100 轮，说明条件数极差或 tol 定得太小。
4. **特征值要自己排序**。Jacobi 输出顺序不定，PCA 依赖降序，源码在末尾做了 `sort` + 同步重排列。
5. **负特征值要截断**。协方差矩阵理论上半正定，数值上可能出现 $-10^{-16}$；PCA 用 `Math.max(v, 0)` 处理，否则开根号得到 NaN。

### 可验证不变量

| 检查项（`linalgSelfCheck()`） | 容差 |
|---|---|
| `Jacobi: 残差 ‖Av−λv‖<1e-8，特征向量正交，降序` | `eigenResidual` `1e-8`；`orthogonalityDefect` `1e-8`；降序判定 |
| `Jacobi: 已知矩阵特征值解析解` | $\begin{pmatrix}2&1\\1&2\end{pmatrix}$ 的 $\lambda = (3, 1)$，`expectClose(..., 1e-10)` |

---

## 6. 薄 SVD `svd`

### 直觉

SVD 说：任何矩阵都可以写成「旋转 → 拉伸 → 再旋转」三步：$A = U\Sigma V^\top$。它同时给出矩阵的秩、最优低秩压缩（Eckart–Young）、伪逆、以及 LSA 的隐语义空间。本实现走**正规方程路线**：先算 $A^\top A$（对称半正定），用上面的 Jacobi 分解得到 $V$ 与 $\sigma^2$，再由 $u_i = A v_i / \sigma_i$ 反推 $U$。

### 数学核心

$$A^\top A = V\Sigma^2 V^\top \Rightarrow \sigma_i = \sqrt{\lambda_i},\qquad U_{:,i} = \frac{A\,V_{:,i}}{\sigma_i}$$

数值秩截断（避免 $\sigma_i = 0$ 除零）：

$$k = \#\{i : \sigma_i > \sigma_{\max}\cdot \text{tol}\},\quad \text{tol} = 10^{-12};\quad k \ge 1$$

低秩最优性（**Eckart–Young**）：

$$\min_{\operatorname{rank}(B)\le k}\|A - B\|_F = \|A - U_k\Sigma_k V_k^\top\|_F = \sqrt{\sum_{i > k}\sigma_i^2}$$

```
function svd(A, {tol=1e-12}):          # A: m×n，薄 SVD
  {values, vectors} ← jacobiEigen(Aᵀ·A)          # 对称 → Jacobi
  sFull[i] ← √max(values[i], 0)
  k ← #{i : sFull[i] > max(sFull)·tol};  if k == 0: k ← 1
  U ← zeros(m, k)
  for j in 0..k-1:
    U[:, j] ← (A · vectors[:, j]) / sFull[j]
  return {U, s: sFull[0:k], V: vectors[:, 0:k], rank: k}
```

### 复杂度

| 操作 | 时间 | 空间 |
|---|---|---|
| `svd(A:m×n)`（含 $A^\top A$、Jacobi、$U$ 反推） | $O(mn^2 + S n^3)$（$S$ 为 Jacobi sweep 数） | $O(mn + n^2)$ |

### 经典论文

- Eckart, C. & Young, G. (1936), *The approximation of one matrix by another of lower rank*, Psychometrika 1(3): 211–218。
- Golub, W. & Kahan, W. (1965), *Calculating the singular values and pseudo-inverse of a matrix*, SIAM J. Numer. Anal. 2(2): 205–224。
- Golub & Van Loan, *Matrix Computations*, §8.6。
- Deerwester et al. (1990), *Indexing by Latent Semantic Analysis*, JASIS 41(6)（本实现的下游用途，见 `docs/algorithms/retrieval.md`）。

### 常见坑

1. **$A^\top A$ 会把条件数平方**。$\kappa(A^\top A) = \kappa(A)^2$，极端病态矩阵会损失一半有效位数。小规模、良态矩阵（LSA 的 TF-IDF 矩阵）完全够用；病态场景应改用双边 Jacobi 或 Golub–Kahan。
2. **薄 SVD 的 $U$ 只有 $k$ 列**，不是 $m\times m$。需要完整正交基时不能直接用。
3. **$\sigma_i = 0$ 必须排除**，否则 $U$ 反推除零；源码用相对阈值 $\sigma_{\max}\cdot 10^{-12}$ 而非绝对阈值。
4. **`rank` 是数值秩**，不是数学秩。改变 `tol` 会直接改变 LSA 的隐空间维度与同义桥接效果。
5. **`V` 用 `colSlice` 返回的是副本**，修改它不会影响原特征向量矩阵——这点是安全的，但也意味着大矩阵上会有额外拷贝开销。

### 可验证不变量

| 检查项（`linalgSelfCheck()`） | 容差 |
|---|---|
| `SVD: A≈UΣVᵀ，U 列正交` | 重构误差 `expectClose(..., 0, 1e-8)`；$U^\top U = I$ 误差 `1e-8` |

---

## 7. 数值稳定原语 `logsumexp` / `softmaxRows` / `softmaxArr` / `sigmoid` / `argmax`

### 直觉

概率模型里到处都是 $\log\sum\exp$：HMM 的前向、softmax 的分母、朴素贝叶斯的后验归一化。直接算会**上溢**（$\exp(1000) = \infty$）。技巧只有一句话：**先减去最大值再算**，$\exp(x - m) \le 1$，永远不溢出，最后把 $m$ 加回来。这是全引擎最常被复用的一行数学。

### 数学核心

$$\mathrm{logsumexp}(x) = m + \log\sum_i e^{x_i - m},\qquad m = \max_i x_i$$

softmax（逐行）：

$$\mathrm{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}},\qquad m = \max_j z_j$$

sigmoid：$\sigma(x) = 1/(1 + e^{-x})$；`argmax` 返回**首个**最大元素的下标（平局取小索引，保证确定性）。

```
function logsumexp(x):
  m ← max(x);  if m == -∞: return -∞
  return m + log(Σ exp(x_i - m))

function softmaxRows(A):              # 逐行归一化
  for i in rows:
    m ← max_j A[i, j]
    e_j ← exp(A[i, j] - m);  s ← Σ e_j
    out[i, j] ← e_j / s
  return out
```

### 复杂度

| 操作 | 时间 | 空间 |
|---|---|---|
| `logsumexp(x:n)` | $O(n)$（两趟扫描） | $O(1)$ |
| `softmaxRows(A:m×n)` | $O(mn)$ | $O(mn)$ |
| `softmaxArr` / `sigmoid` / `argmax` | $O(n)$ | $O(n)$ / $O(1)$ / $O(1)$ |

### 经典论文

- 减去最大值的稳定技巧见：Numerical Recipes, 3rd ed., §10.6；以及 Press et al. 关于浮点误差的一般性讨论。
- softmax / log-sum-exp 的关系见：Bishop, C.M. (2006), *Pattern Recognition and Machine Learning*, §4.3.4（softmax 的归一化与过参数化说明）。
- Blanchard, P., Higham, D.J. & Higham, N.J. (2021), *Accurately computing the log-sum-exp and softmax functions*, IMA J. Numer. Anal. 41(4)。

### 常见坑

1. **必须先减最大值再取 $\exp$**，不要"先算 $\exp$ 再除"——$\exp(1000)$ 直接 `Infinity`，$\infty/\infty = $ NaN，整批梯度全废。
2. **全 $-\infty$ 输入要特判**：`logsumexp` 源码对 $m = -\infty$ 直接返回 $-\infty$（GMM 的 `_eStep` 也据此退化为均匀责任度）。
3. **`softmaxRows` 内部用 `Math.max(...r)` 展开数组**，行数/列数极大时会触发栈上限；超大批次应改写成循环取最大值。
4. **softmax 对输入平移不变**（减去常数结果不变），但对**缩放**敏感；温度参数要显式除以 `temperature`，不要靠改 logits 尺度。
5. **`argmax` 的平局语义是"取首个"**。所有依赖确定性的地方（KNN 投票、聚类标签压缩）都依赖这一条，不要换成随机打破平局。

### 可验证不变量

| 检查项（`linalgSelfCheck()`） | 容差 |
|---|---|
| `logsumexp ≥ max 且与朴素计算一致` | 与朴素 $\log\sum e^{x}$ 比对 `1e-12`；`[1000,1001,1002]` 不溢出，比对 `1e-10` |
| `softmax 行和为 1 且最大元素概率最大` | 概率和 `expectClose(..., 1, 1e-12)`；$p_2 > p_1$ |

---

## 附：本模块不变量总览（`src/selfcheck.mjs` → `linalgSelfCheck()`）

| # | 检查项名称 | 关键容差 |
|---|---|---|
| 1 | `matMul 与朴素三重循环一致` | 1e-10 |
| 2 | `solveLinear 残差 ‖Ax−b‖≈0` | 1e-8 |
| 3 | `A·A⁻¹ = I` | 1e-8 |
| 4 | `Cholesky: L·Lᵀ=A 且正定性判定正确` | 1e-8 |
| 5 | `QR: QᵀQ=I，QR=A，R 上三角` | 1e-8 / 1e-10 |
| 6 | `Jacobi: 残差 ‖Av−λv‖<1e-8，特征向量正交，降序` | 1e-8 |
| 7 | `Jacobi: 已知矩阵特征值解析解` | 1e-10 |
| 8 | `SVD: A≈UΣVᵀ，U 列正交` | 1e-8 |
| 9 | `logsumexp ≥ max 且与朴素计算一致` | 1e-12 / 1e-10 |
| 10 | `softmax 行和为 1 且最大元素概率最大` | 1e-12 |
| 11 | `RNG 同种子同序列（可复现性）` | 100 次抽样逐一相等 |

实跑结果：`node src/cli.mjs selftest` → `✅ linalg 11/11`。
