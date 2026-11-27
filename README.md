# SpecPilot v2 — Autonomous Algorithm Discovery & Systems Optimization

复现并演进 *SpecPilot: Autonomous Algorithm Discovery and Systems Optimization for Speculative Decoding*（arXiv:2611.02501v1）的**模型/硬件/温度通用的全盲测自主优化闭环系统**。

GitHub：[china-qijizhifeng/SpecPilot-reproduction](https://github.com/china-qijizhifeng/SpecPilot-reproduction)

---

## 核心成果总览

所有优化均由默认关闭（default-OFF）的 `SP_OPT_*` 开关隔离，经**独立 2×H100 物理互换 Controller**（T1 / T1-S 无损门禁 + T2 30-req 配对 CI）独立验收；研究角色无权修改 `controller/`、`eval/sealed/` 或 `ledger/`。

### 1. 主线全盲测：Qwen3.8-27B BF16 + DFlash2（H100 80GB，Greedy `temp = 0`）

全部 6 轮均在**严格盲测**下由自主 Agent 独立发现（审计 **0 次联网、0 次读禁区/论文/作者 `final`**），经 Dev 30-req 2048-cap + Sealed 30-req 严格未见集（2048 & 4096-cap Forcing）双重验收：

| 轮次 / 对照 | 盲测独立发现的核心机制 | Dev 30-req 2048-cap 配对加速比 (`tok/s`, $\tau$) | Sealed 30-req 未见集 (`tok/s`, $\tau$) | Forcing 大 margin 事件 (Dev / 2048 / 4096) | T4 Memory 逐字节还原 | 状态 |
|---|---|---|---|---|---|---|
| **Baseline** | Stock SGLang v0.5.20 + DFlash2 (`AR = 51.6 tok/s`) | `1.00×` (`190.4`, $\tau=4.65$) | `1.00×` (`173.7`, $\tau=4.26$) | `0` / `0` / `0` | — | 基线 |
| **§6.1 消融（无 CM 单 Agent）** | `block_size=16` + 切换 FlashInfer GDN 后端（3 轮串行盲搜） | `1.1173× [1.081, 1.159]` (`209.8`, $\tau=5.30$) | — | `1` (`p=0.981`) | — | 对照基线 |
| **Round 1** | Zero-snapshot CUDA GDN + `b=12` + 28-node Best-first Tree | `1.2960× [1.271, 1.322]` (`248.5`, $\tau=6.70$) | — | `0` | `r1_tree_gdn_b12` (`true`) | `PROMOTED` |
| **Round 2** | 无损 12-bit BF16 线层/LM-head + FP8 Drafter + GDN Tree v2 | `1.5561× [1.516, 1.600]` (`304.6`, $\tau=6.77$) | — | `0` | `r2_w12_gdnv2_fp8draft` (`true`) | `PROMOTED` |
| **Round 3** | All-MMA Tree-WY GDN + Fused GQA Suffix Tree Attn | `1.6260× [1.585, 1.672]` (`319.8`, $\tau=6.83$) | `1.5745× [1.544, 1.608]` (`279.1`, $\tau=5.95$) | `0` / `0` / `0` | `r3_tree_wy_attn` (`true`) | `PROMOTED` + `SEALED` |
| **Round 4（初版 → 修复）** | Fast `sm_90` W6-v2 两级 Lookahead + 自适应 `M_q=16`；初版被 T1 拦 `15` 事件 → 盲测改为**精确 FP32 subwarp 反向三角求解** | `1.7068× [1.657, 1.758]` (`327.0`, $\tau=10.05$) | `1.6700× [1.627, 1.715]` (`292.1`, $\tau=8.70$) | `0` / `0` / `0` | `r4_w6v2_adaptive_fixed` (`true`) | `PROMOTED` + `SEALED` |
| **Round 5** | Always-On 2-Pass (`MODE=2`) + 消除 2-pass `.item()` GPU-CPU 同步 | `1.7313× [1.701, 1.765]` (`335.0`, $\tau=12.20$) | `1.6488× [1.601, 1.696]` (`290.5`, $\tau=10.54$) | `0` / `0` / `0` | `r5_mode2_nosync` (`true`) | `PROMOTED` + `SEALED` |
| **Round 6（P0+P1 破局轮）** | `SP_OPT_LA_PIPELINE_FUSE=1`（空闲气泡 `-91.7%`）+ `SP_OPT_ADAPTIVE_LA_CONF=1`（零同步 EWMA 自适应门控） | **`1.7741× [1.738, 1.813]`** (**`342.5`**, $\tau=12.07$, **`6.64×` AR**) | **`1.7036× [1.653, 1.754]`** (**`299.0`**, $\tau=10.43$) | **`0` / `0` / `0`** | **`r6_la_pipeline_ewma` (`true`)** | **`PROMOTED` + `SEALED`** |

---

### 2. 跨模型 / 跨架构实战：Qwen3.5-35B-A3B-FP8（256 专家稀疏 MoE + 原生 MTP `NEXTN`）

| 轮次 | 核心优化机制 | Dev 30-req 配对加速比 (`tok/s`, $\tau$) | FP32 Token Forcing (事件 / 不匹配率) | 状态 |
|---|---|---|---|---|
| **Baseline** | Stock SGLang + 原生 MTP (`AR = 279 tok/s`) | `1.00×` (`507.9`, $\tau=3.41$) | `12` / `1.2763%` | 基线 |
| **Round 1** | SM90 W8A16 FP8 MoE + Shared-MLP 融合（92–99.6% HBM 上限）+ FP8 MTP `lm_head` + `draft_extend` CUDA Graph + 零同步 EWMA 自适应步数（成功拦住 256 专家 MoE 宽树负迁移 `0.893×`） | **`1.2771× [1.2591, 1.2958]`** (**`636.5`**, $\tau=4.34$, **`2.28×` AR**) | **`8` < Stock `12`** (`p=0.9998`) / `1.2145%` | **`PROMOTED`** |

---

### 3. 随机采样扩展：`temperature = 0.7` 无损优化与 T1-S 黑盒验收（Qwen3.8-27B + DFlash2）

新增 **T1-S 黑盒分布无损验收**（[v2/t1s_verify.py](v2/t1s_verify.py)：M1~M4 逐 token Top-20 分布 TVD/logprob 对比 FP32 Reference + M5 鞅检验 $Z_{\text{NLL}}/Z_{\text{top1}}$ & 随机化 PIT-KS + Stock 基线缓存），对 Leviathan 链与 Target-Only 树统一兼容（30 题 × 1024 tokens，`T=0.7, top_p=1.0, top_k=-1`）：

| 方案（均为无记忆纯盲测，除末尾对照） | 核心机制 | 吞吐 (`tok/s`, $\tau$) | 2×H100 配对加速比 (95% CI) | T1-S 分布指标 (`mean_tvd` / `PIT-KS p`) | 状态 |
|---|---|---|---|---|---|
| **Stock（原生 Leviathan 链）** | 原生 DFlash2 W8 链 | `187.7`, $\tau=4.63$ | `1.0000×` | `0.005332` / `p=0.526` | 基线（已缓存） |
| **Blind R1** | 12-bit 无损压缩 + FP32 GDN 融合 + Draft FP8 | `219.8`, $\tau=4.58$ | `1.1807× [1.1521, 1.2098]` | `0.005573` / `p=0.681` | `PROMOTED` |
| **Blind R2** | + 全量 12-bit (`out_proj/o_proj/lm_head`) + 零 SMEM 融合核 | `231.1`, $\tau=4.56$ | `1.2337× [1.2128, 1.2548]` | **`0.005265`** / **`p=0.911`** | `PROMOTED` |
| **Blind R3（当前盲测最佳）** | + 并行 Overflow CTA（消 185 次核启动）+ 在线 Softmax/胶水融合 | **`249.5`**, $\tau=4.72$ | **`1.3277× [1.2982, 1.3595]`** | **`0.005319`** / `p=0.618` | **`PROMOTED`** |
| *（对照上限）`r6_tree_sampling_t07`* | *R6 算子 + WY 树 GDN + `M=28, D=12` 宽深树 Target-Only* | **`279.9`**, $\tau=6.15$ | **`1.5089× [1.4724, 1.5443]`** | `0.005359` / `p=0.238` | `PROMOTED` |

> 完整实验报告与技术笔记：[SpecPilot v2 全盲测实验报告](docs/experiments/specpilot-v2-blind-r1-r4-20261125.md) · [投机解码无损拒绝采样笔记](docs/notes/speculative-rejection-sampling.md)

---

## SpecPilot v2 系统架构（[v2/](v2/)）

采用**探索与裁判物理隔离**、**模型 / 投机算法 / GPU / 温度全通用**的模块化设计（详见 [v2/README.md](v2/README.md)）：

1. **双端三因子 Phase 0 Ceiling（[v2/phase0/](v2/phase0/)）**：
   - `floors.py`：支持纯 Attention、混合 GDN/Attn 及 **Sparse MoE**，内置多卡预设（H100 / H200 / B200 / RTX PRO 6000 / Ascend 910B2C），显式量化 `T_step - T_prof` 系统气泡；
   - `tau_headroom.py`：同时支持 `temp=0` 精确重放与 `temp>0` Target-Only Best-First 树期望接受长度（含重复兄弟去重）。
2. **无记忆原生探索引擎（[v2/agents/](v2/agents/)、[v2/sandbox/](v2/sandbox/)）**：
   - 持久多卡探索池（2–8 秒零冷启动）+ 隔离工作树 + Easy/Hard 分层 Tier-3 筛查（`bench_1gpu.py`）；
   - **Priority 0 规则**：先修复/证伪上一轮 T1 被拒候选，再开新方向；
   - **3 小时深挖预算 + 严格三态结案协议**：仅允许 `PROMOTABLE`、`PROVEN_FALSIFIED`（必须实测 $\tau$ 与 Tier-1 对齐 `< 5%` 且核逼近下界）或 `STEPPING_STONE_HANDOFF`（保留中继工作树接力），配合 **Plateau Escape & Co-Design** 跨模块协同协议。
3. **独立双卡互换 Controller（[v2/controller_verify_2gpu.py](v2/controller_verify_2gpu.py)、[v2/t1s_verify.py](v2/t1s_verify.py)、[v2/controller_sealed_final.py](v2/controller_sealed_final.py)）**：
   - `--temperature 0` 走 T1 FP32 `argmax` Token Forcing；`--temperature > 0` 自动切 T1-S（M1~M5 分布 Token Forcing + 鞅/PIT-KS + Stock 缓存）+ 双卡互换配对 CI。
4. **可选 5 轴双层 Structured Memory（[v2/memory/](v2/memory/)，默认关闭）**：
   - 覆盖 `model / spec / hardware / workload / dataset` 5 轴上下文指纹与 `transfer_scope`，自动分级 `code_direct / code_adapt / pattern_transfer / blocked`（默认隐藏 `blocked`、Stage-1 1 行索引 → Stage-2 按需展开 + 溯源日志），用于跨场景迁移对照实验。

---

## 快速自检

```bash
# 运行 SpecPilot v2 全部 CPU 单元测试（17 项）
python3 -m unittest discover -s v2/tests -p 'test_*.py' -v

# 查看可选 5 轴 Structured Memory 摘要索引
python3 v2/memory/store.py index
```

> 历史 v1 探索记录与交接日志保留在 [HANDOFF.md](HANDOFF.md)、[PLANS.md](PLANS.md) 及 `docs/experiments/` 中存档。
