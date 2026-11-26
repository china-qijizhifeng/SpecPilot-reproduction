# 投机解码（Speculative Decoding）无损拒绝采样笔记

本文系统整理 `temp > 0`（随机采样）场景下，链式与树状（Draft Tree）投机解码如何保证 **100% 数学无损**、何时需要草稿分布 $q$、工程实现细节，以及 Controller 如何验收无损性。

---

## 1. 核心总览：两条无损路线的本质区别

在投机解码的验证阶段，Target 模型通过一次并行前向算出每个候选节点 $u$ 处的目标分布 $p_u(\cdot)$（经过 `temperature`、`top-k`、`top-p` 归一化）。要保证最终输出严格服从 $p_u$，有且仅有两条标准路线：

| 维度 | 路线 A：Target-Only（不用 $q$，等价 $q=1$） | 路线 B：带 $q$ 的标准拒绝采样（$\min(1, p/q)$） |
|---|---|---|
| **对草稿生成方式的要求** | **无任何要求**（确定性 top-$k$、beam、随机采都行） | **必须严格按 $q$ 随机采样**，且验证端拿到完全一致的 $q$ |
| **是否允许自己编造 $q$** | 不需要 $q$ | **绝对不允许**（$q_{\text{假}} \neq q_{\text{真}}$ 直接导致有损） |
| **无损性条件** | 对**任意给定候选集 $S$** 都有 $P(Y=x\mid S)=p(x)$ | 对随机候选集取期望 $\mathbb{E}_{S\sim q}[P(Y=x\mid S)]=p(x)$ |
| **单层接收率** | $\sum_{i=1}^k p(c_i)$（确定性候选集的**理论上限**） | 单链 $\sum_x \min(p(x), q(x)) = 1 - \text{TVD}(p, q)$ |
| **适用场景** | 确定性分支树（Medusa / EAGLE-2 / DFlash 树） | Drafter 自身随机采样且输出完整 $q$（Leviathan / SpecInfer） |
| **SGLang 对应核函数** | `sgl_kernel.tree_speculative_sampling_target_only`（`draft_probs=0`） | `chain_speculative_sampling_triton`（传入 `draft_probs=q`） |

**核心判断原则**：
- 只有当 draft token **真的是按分布 $q$ 随机采样出来的**，才能用 $\min(1, p/q)$；
- 只要 draft token 是确定性挑出来的（`argmax`、top-$k$ 建树、beam search）或拿不到精确 $q$，为了严格无损就**只能走 Target-Only**。

---

## 2. 路线 A：Draft Tree 的 Target-Only（不用 $q$）无损采样

### 2.1 为什么整棵树无损只取决于每个单节点？

Target 模型带着 tree attention mask 一次前向，会同时算出树上所有节点 $u$ 处的 $p_u(\cdot) = P_{\text{target}}(\cdot \mid \text{从根到 } u \text{ 的路径})$。

从根节点 $u=0$ 出发：
- 在当前节点 $u$ 按 $p_u$ 选出一个 token $y$；
- 若 $y$ 恰好等于 $u$ 的某个儿子 $c_i$，指针跳到 $c_i$，直接复用已算好的 $p_{c_i}$ 继续下一步；
- 若 $y$ 不在 $u$ 的儿子中，输出 $y$ 作为本步最后一个 bonus token 并停止。

由概率链式法则 $P(y_1, y_2, \dots) = p_0(y_1)\cdot p_{y_1}(y_2)\cdots$，只要**每个单节点的输出分布严格等于 $p_u$**，整棵树走出来的序列就 100% 无损。

### 2.2 三种完全等价的视角

设当前节点 $u$ 有 $k$ 个互不相同的候选儿子 $\{c_1, c_2, \dots, c_k\}$，目标分布为 $p$。以下三种写法**数学上 100% 等价**：

1. **视角 1：直接从 $p$ 采样再比对**
   - 直接采 $y \sim p$；若 $y = c_i$ 则进入子节点 $c_i$，否则把 $y$ 当 bonus 输出并停下。
   - 显然无损，但缺点是每个节点都要对完整词表（如 $V=248{,}320$）做一次 `multinomial(p)`，扫 $M$ 遍大词表太慢。

2. **视角 2：重排 $[0, 1)$ 区间一次掷飞镖**
   - 离散分布采样时，词表在 $[0, 1)$ 上的排列顺序无关紧要。
   - 故意把 $k$ 个儿子排在 $[0, 1)$ 最前面，总长度 $\sum_{i=1}^k p(c_i)$，其余词排在末尾：
     ```text
     |<-- p(c1) -->|<-- p(c2) -->| ... |<-- p(ck) -->|<--- 其余词，总长 1 - Σp(ci) --->|
     0           p(c1)       p(c1)+p(c2)           Σp(ci)                             1
     ```
   - 扔一支飞镖 $U \sim U(0,1)$：落进 $c_i$ 就走 $c_i$，落进末尾就从剩余词按比例采 bonus。

3. **视角 3：逐兄弟条件抛硬币 $u_i \sim U(0,1)$（CUDA 核函数实际写法）**
   - 记测完前 $i-1$ 个儿子后的剩余概率质量为 $R_{i-1} = 1 - \sum_{j=1}^{i-1} p(c_j)$（$R_0 = 1$）。
   - 对第 $i$ 个儿子 $c_i$，抽独立 $u_i \sim U(0,1)$：
     - 若 $u_i < \dfrac{p(c_i)}{R_{i-1}}$，**接受 $c_i$** 并进入子节点；
     - 否则拒绝，将 $p(c_i)$ 清零，更新 $R_i = R_{i-1} - p(c_i)$，继续测下一个兄弟。
   - 无条件接受 $c_i$ 的概率：
     $$P(\text{前 } i-1 \text{ 个全拒}) \times P(\text{接受 } c_i \mid \text{前 } i-1 \text{ 个全拒}) = R_{i-1} \cdot \frac{p(c_i)}{R_{i-1}} = p(c_i)$$
   - 若 $k$ 个兄弟全拒（概率 $R_k$），抽 $u_{\text{final}} \sim U(0,1)$ 从残差分布 $p_{\text{res}}(x) = \dfrac{\mathbb{I}[x \notin \{c_1,\dots,c_k\}]\cdot p(x)}{R_k}$ 采 bonus：
     $$P(\text{输出树外词 } x) = R_k \cdot \frac{p(x)}{R_k} = p(x)$$
   - **工程优势**：前面被接受的节点只需读 **1 个标量** $p(c_i)$，整个验证步只在最后停下的 **1 个节点** 扫一次全词表采 bonus！

### 2.3 数值例子

设词表 $\{A, B, C, D\}$，$p = (0.4, 0.3, 0.2, 0.1)$，树上挂了两个儿子 $c_1=A, c_2=B$：
1. 测 $A$：$u_1 < 0.4$ 则收 $A$ $\implies P(A) = \mathbf{0.4}$。若拒（概率 $0.6$），清零 $A$ 得 $p^{(1)} = (0, 0.5, \frac{1}{3}, \frac{1}{6})$。
2. 测 $B$：$u_2 < 0.5$ 则收 $B$ $\implies P(B) = 0.6 \times 0.5 = \mathbf{0.3}$。若拒（概率 $0.3$），清零 $B$ 得 $p^{(2)} = (0, 0, \frac{2}{3}, \frac{1}{3})$。
3. 全拒采 bonus：$P(C) = 0.3 \times \frac{2}{3} = \mathbf{0.2}$，$P(D) = 0.3 \times \frac{1}{3} = \mathbf{0.1}$。

最终分布严格等于 $(0.4, 0.3, 0.2, 0.1)$。

### 2.4 两个重要性质

- **即使 $S$ 是随机生成的，Target-Only 依然无损**：由全概率公式 $P(Y=x) = \mathbb{E}_S[P(Y=x\mid S)] = \mathbb{E}_S[p(x)] = p(x)$（只是此时接收率次优，没利用 $S$ 来自 $q$ 的信息）。
- **重复兄弟 token 安全**：只要拒绝 $c_i$ 时将 `p[c_i]` 原地置 0，就算树中出现重复兄弟，第二次测到时其剩余概率为 0，不会被重复接受，依然严格无损。

---

## 3. 路线 B：带 $q$ 的标准拒绝采样（Leviathan '23 / Chen '23）

### 3.1 算法与无损证明（单链）

前提：Drafter 真的按分布 $q$ 随机采样出 $c \sim q$。
1. 抽 $u \sim U(0,1)$，若 $u < \min\big(1, \frac{p(c)}{q(c)}\big)$ 则接受 $c$；
2. 若拒绝，则从归一化残差分布采 bonus：
   $$p_{\text{res}}(x) = \frac{\max\big(0, p(x) - q(x)\big)}{\sum_z \max\big(0, p(z) - q(z)\big)}$$

记总拒绝概率为 $\beta = 1 - \sum_z \min(p(z), q(z)) = \sum_z \max(0, p(z) - q(z))$。对任意词 $x$：
- 若 $p(x) \le q(x)$：$\max(0, p(x)-q(x))=0$，只能通过“被采且被接受”输出，概率 $= q(x)\cdot \frac{p(x)}{q(x)} = p(x)$。
- 若 $p(x) > q(x)$：接受概率为 $1$，总输出概率 $= q(x)\cdot 1 + \beta \cdot \frac{p(x)-q(x)}{\beta} = p(x)$。

### 3.2 为什么高温下带 $q$ 接受率远高于 Target-Only？

设 $p = q = (0.25, 0.25, 0.25, 0.25)$（高温平坦分布，且 drafter 完美预测）：
- **Target-Only（单链）**：猜 1 个词再独立从 $p$ 抽，撞上的概率只有 **25%**；
- **带 $q$ 拒绝采样**：$\min(1, 0.25/0.25) = \mathbf{100\%}$ 直接全收！

但注意：**如果 $c$ 是确定性挑的（或按另一个分布 $q_{\text{真}}$ 采的），却拿 $q_{\text{假}}$ 代入 $\min(1, p/q_{\text{假}})$**，上面证明中的 $q(x)$ 无法与分母约掉，分布立刻失真（有损）。

---

## 4. 社区代表工作与本仓库实测

- **代表论文**：SpecInfer (ASPLOS '24)、Sequoia (NeurIPS '24)、EAGLE / EAGLE-2 / EAGLE-3 (ICML '24 / EMNLP '24)。
- **SGLang 实现**：
  - [dflash_utils.py](file:///gpfs/users/jixinpeng/SpecDec/SpecPilot/v2_runtime/round6_blind/worktrees/r6_integrated/python/sglang/srt/speculative/dflash_utils.py#L976-L1118)（链式非贪心）
  - [eagle_utils.py](file:///gpfs/users/jixinpeng/SpecDec/SpecPilot/v2_runtime/round6_blind/worktrees/r6_integrated/python/sglang/srt/speculative/eagle_utils.py#L814-L886)（EAGLE 树非贪心）
  - [dflash_worker_v2.py](file:///gpfs/users/jixinpeng/SpecDec/SpecPilot/v2_runtime/temp07_test/wt_tree_sampling/python/sglang/srt/speculative/dflash_worker_v2.py#L3132-L3218)（本仓库 DFlash 28 节点树非贪心 `SP_OPT_DFLASH_TREE_SAMPLING=1`）
- **Qwen3.8-27B `temperature=0.7` 实测（H100 单卡）**：
  - Stock DFlash W8 链：`177.84 tok/s`，$\tau = 4.43$（`1.000×`）
  - R6 仅算子 + W12 链：`229.93 tok/s`，$\tau = 4.53$（`1.293×`）
  - **R6 + 28 节点无损树拒绝采样**：**`262.48 tok/s`**，$\tau = 5.87$（**`1.476×`，`+47.6%`**，相比贪心树仅增加 `0.04 ms/step` 开销，且 20 万次蒙特卡洛最大单项分布误差 `0.00096`）。

---

## 5. Controller 如何对 `temp > 0` 做无损验收

由于换了树结构后随机数消耗顺序不同，两次运行必然生成不同的 token 序列，无法逐 token 比对。Controller 应采用 **3 层验收**：

1. **L1 — 非贪心轨迹确定性 Token Forcing**（验算子 / KV / GDN 状态）：
   沿固定的 `temp=0.7` 轨迹跑真实 verify 路径，核对 $\text{KL}(p_{\text{ref}} \| p_{\text{cand}})$ 与 `argmax` 大间隔异常率（零随机性）。
2. **L2 — 端到端随机轨迹鞅检验（Martingale CLT + Randomized PIT）**（验采样器本身）：
   用参考模型对候选引擎实际生成的整条序列做一次 prefill，得到每步真实 $p_t = p_{\text{ref}}(\cdot \mid y_{<t})$。只要每步 $y_t \sim p_t$，无论序列怎么变，以下统计量均严格成立：
   - **熵校准**：$Z_{\text{NLL}} = \dfrac{\sum_t (-\log p_t(y_t) - H(p_t))}{\sqrt{\sum_t \text{Var}_{p_t}(-\log p_t)}} \xrightarrow{d} \mathcal{N}(0,1)$（瞬间抓出退化贪心或乱码）；
   - **Top-1 校准**：$Z_{\text{top1}} = \dfrac{\sum_t (\mathbb{I}[y_t=\arg\max p_t] - \max p_t)}{\sqrt{\sum_t \max p_t (1 - \max p_t)}} \xrightarrow{d} \mathcal{N}(0,1)$；
   - **随机化 PIT + KS 检验**：$U_t = F_t^-(y_t) + v_t\cdot p_t(y_t) \stackrel{\text{i.i.d.}}{\sim} \text{Uniform}(0,1)$（抓整条分布形状畸变、温度/top-p 偏移）。
3. **L3 — 单前缀高频抽样 $\chi^2$ 检验 + 核函数蒙特卡洛单测**：
   对 2~3 个高熵前缀打 1,000 次短请求核对直方图，配合 20 万次核函数单测覆盖边界条件。
