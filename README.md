# SpecPilot v2 — Autonomous Algorithm Discovery & Systems Optimization

复现并演进 *SpecPilot: Autonomous Algorithm Discovery and Systems Optimization for Speculative Decoding*（arXiv:2611.02501v1）的**全盲测自主优化闭环系统**。

GitHub：[china-qijizhifeng/SpecPilot-reproduction](https://github.com/china-qijizhifeng/SpecPilot-reproduction)

---

## 核心成果总览（H100 80GB，Qwen3.8-27B BF16 + DFlash2，单请求 Greedy）

全部 6 轮优化均在**严格盲测**下由自主 Agent 独立发现并实现（审计 **0 次联网、0 次读禁区/论文/作者 `final`**），所有改动由 default-OFF `SP_OPT_*` 开关隔离，经**隔离 2×H100 双卡互换 Controller（T1 FP32 LM-head Token Forcing + T2 30-req 2048-cap 配对 CI）**与 **Phase 4 Sealed Final（30 条严格未见请求 + 4096-cap Forcing）** 独立验收：

| 轮次 / 对照 | 盲测独立发现的核心机制 | Dev 30-req 2048-cap 配对加速比 (`tok/s`, $\tau$) | Sealed 30-req 未见集 (`tok/s`, $\tau$) | Forcing 大 margin 事件 (Dev / 2048 / 4096) | T4 Memory 逐字节还原 | 状态 |
|---|---|---|---|---|---|---|
| **Baseline** | Stock SGLang v0.5.20 + DFlash2 (`AR = 51.6 tok/s`) | `1.00×` (`190.4`, $\tau=4.65$) | `1.00×` (`173.7`, $\tau=4.26$) | `0` / `0` / `0` | — | 基线 |
| **§6.1 消融（无 CM 单 Agent）** | `block_size=16` + 切换 FlashInfer GDN 后端（3 轮串行盲搜） | `1.1173× [1.081, 1.159]` (`209.8`, $\tau=5.30$) | — | `1` (`p=0.981`) | — | 对照基线 |
| **Round 1** | Zero-snapshot CUDA GDN + `b=12` + 28-node Best-first Tree | `1.2960× [1.271, 1.322]` (`248.5`, $\tau=6.70$) | — | `0` | `r1_tree_gdn_b12` (`true`) | `PROMOTED` |
| **Round 2** | 无损 12-bit BF16 线层/LM-head（独立重发现 W12）+ FP8 Drafter + GDN Tree v2 | `1.5561× [1.516, 1.600]` (`304.6`, $\tau=6.77$) | — | `0` | `r2_w12_gdnv2_fp8draft` (`true`) | `PROMOTED` |
| **Round 3** | All-MMA Tree-WY GDN + Fused GQA Suffix Tree Attn | `1.6260× [1.585, 1.672]` (`319.8`, $\tau=6.83$) | `1.5745× [1.544, 1.608]` (`279.1`, $\tau=5.95$) | `0` / `0` / `0` | `r3_tree_wy_attn` (`true`) | `PROMOTED` + `SEALED` |
| **Round 4（初版 → 修复）** | Fast `sm_90` W6-v2 两级 Lookahead + 自适应 `M_q=16`；初版被 T1 拦截 `15` 事件 → 盲测定位深树 TF32 Neumann 相消并改为**精确 FP32 subwarp 反向三角求解** | `1.7068× [1.657, 1.758]` (`327.0`, $\tau=10.05$) | `1.6700× [1.627, 1.715]` (`292.1`, $\tau=8.70$) | `0` / `0` / `0` | `r4_w6v2_adaptive_fixed` (`true`) | `PROMOTED` + `SEALED` |
| **Round 5** | Always-On 2-Pass (`MODE=2`) + 消除 2-pass `.item()` GPU-CPU 同步 | `1.7313× [1.701, 1.765]` (`335.0`, $\tau=12.20$) | `1.6488× [1.601, 1.696]` (`290.5`, $\tau=10.54$) | `0` / `0` / `0` | `r5_mode2_nosync` (`true`) | `PROMOTED` + `SEALED` |
| **Round 6（P0+P1 破局轮）** | `SP_OPT_LA_PIPELINE_FUSE=1`（消除 Stage-2 D2H 停顿与胶水，空闲气泡 `-91.7%`）+ `SP_OPT_ADAPTIVE_LA_CONF=1`（零同步 EWMA 自适应门控） | **`1.7741× [1.738, 1.813]`** (**`342.5`**, $\tau=12.07$, **`6.64×` AR**) | **`1.7036× [1.653, 1.754]`** (**`299.0`**, $\tau=10.43$) | **`0` / `0` / `0`** | **`r6_la_pipeline_ewma` (`true`)** | **`PROMOTED` + `SEALED`** |

完整实验记录、NorthJob UUID 索引与消融分析见 [SpecPilot v2 全盲测实验报告](docs/experiments/specpilot-v2-blind-r1-r4-20261125.md)。

---

## SpecPilot v2 系统架构与核心能力

代码全部收敛在 [v2/](v2/)（详见 [v2/README.md](v2/README.md)），采用**探索与裁判物理隔离**、**模型与硬件无关**的均衡全栈设计：

1. **双端三因子 Phase 0 Ceiling（[v2/phase0/](v2/phase0/)）**：
   - `floors.py`：支持纯 Attention、混合 GDN/Attn 及 **Sparse MoE** 模型，内置多卡预设（H100 / H200 / B200 / RTX PRO 6000 / Ascend 910B2C），显式量化 `T_step - T_prof` 系统气泡与 Overlap 空间；
   - `tau_headroom.py`：模型无关的精确重放（exact replay）接受率上界分析。
2. **盲测 Master + 多路并行 Direction + 持久探索池（[v2/agents/](v2/agents/)、[v2/sandbox/](v2/sandbox/)）**：
   - 8×H100 持久探索池（2–8 秒零冷启动周转）+ 纯 Python 隔离工作树；
   - **Priority 0 规则**：上一轮高加速但 T1 Token Forcing 被拒的候选必须先诊断修复或证伪，再开新方向；
   - **Plateau Escape & Co-Design 协议**：增量收益 `< 3%` 时强制做假设/耦合审计并支持跨模块联合重构；
   - **Easy / Hard 分层 Tier-3（`bench_1gpu.py`）**：零额外耗时拆出高/低接受率双桶，防止 Dev 过拟合。
3. **独立双卡互换 Controller 与 Phase 4 Sealed Final（[v2/controller_verify_2gpu.py](v2/controller_verify_2gpu.py)、[v2/controller_sealed_final.py](v2/controller_sealed_final.py)）**：
   - 独立 2×H100 节点运行两轮物理 GPU 互换配对 CI + FP32 LM-head Token Forcing（2048 & 4096 cap），研究角色无权修改 `controller/`、`eval/sealed/` 或 `ledger/`。
4. **双层 Structured Memory（[v2/memory/](v2/memory/)）**：
   - **正向条目（`distill.py`）**：传递依赖链 + 从干净 `stock` 逐字节 `TREE_SHA` 还原校验（6 代全部 `byte_exact_match = true`）；
   - **条件负向错题本 + 4 维上下文指纹（`store.py`）**：绑定 `model / hardware / workload / dataset` 四维指纹与物理瓶颈签名，支持 **Stage-1 1 行摘要索引 (`index`) → Stage-2 按需展开 (`show --log-provenance`)**，跨实验自动标出条件漂移防止负迁移。

---

## 当前状态（已完成 / 正在做 / 接下来）

- **已完成**：
  - **`Qwen3.8-27B BF16 + DFlash2`（H100，Greedy `temp=0`）全盲测 Round 1–6 闭环**：Dev **`1.7741×` / `342.46 tok/s`**（`6.64×` AR），Sealed **`1.7036×` / `298.95 tok/s`**，`0` 大 margin 事件，6 代 Memory 逐字节一致；
  - **论文 §6.1 无 Cost Model 单 Agent 消融**：`1.1173×`（`209.79 tok/s`），与 SpecPilot 首轮 `1.2960×` / R6 `1.7741×` 形成清晰无重叠 CI 对照；
  - **跨模型/跨架构实战验证（`Qwen3.5-35B-A3B-FP8` 256 专家稀疏 MoE + 原生 MTP `NEXTN`）Round 1**：2×H100 Controller 验收达 **`1.2771× [1.2591, 1.2958]`**（**`636.46 tok/s`** vs Stock MTP `507.92 tok/s`，`2.28×` AR，$\tau: 3.41 \to 4.34$；FP32 Token Forcing 不匹配率 `1.2145%` < Stock `1.2763%`，大 margin 事件 **`8` 少于 Stock 的 `12`**，`event_p = 0.9998`，**`promoted = true`**）；
  - **`temperature > 0`（Target-Only）全链路无损支持与盲测验证（Qwen3.8-27B，`T=0.7`）**：
    - 新增 [v2/t1s_verify.py](v2/t1s_verify.py)（T1-S 分布 Token Forcing M1~M5：`TVD > 0.05` 事件、Mean/P99 TVD、`dlogp`、随机性冒烟 + 鞅检验 $Z_{\text{NLL}}/Z_{\text{top1}}$ + PIT-KS，以及 Stock 基线缓存）与 `tau_headroom.py` 树期望接受长度；
    - 锁定官方 `sgl_kernel.tree_speculative_sampling_target_only`（`draft_probs=0, threshold=1.0`，链/树统一无损，20 万次蒙特卡洛最大单项误差 `0.00096`），`r6_tree_sampling_t07` 经 2×H100 Controller 验收达 **`1.5089× [1.4724, 1.5443]`**（`279.94 tok/s`，T1-S 全 5 项门禁通过）；
    - **无记忆纯盲测 `temp=0.7` Round 1–4**：R1 **`1.1807× [1.1521, 1.2098]`**（`219.77 tok/s`）、R2 **`1.2337× [1.2128, 1.2548]`**（`231.11 tok/s`）、R3 **`1.3277× [1.2982, 1.3595]`**（**`249.53 tok/s`**，纯链 + 并行 Overflow CTA + 在线 Softmax 融合，`mean_tvd=0.005319` < Stock `0.005332`，PIT-KS `p=0.618`），R4 Target-Only 4 路树 Co-Design **`1.2808× [1.2042, 1.3498]`**（`M=16` 树，$\tau: 4.15 \to 5.03$，`z_nll=+0.49`，PIT-KS `p=0.726`），全部 **`promoted = true`**；
  - **原生探索能力与通用 5 轴可选 Memory 升级**：`direction_prompt.md` / `master_prompt.md` 落地 3 小时深挖预算 + 三态结案协议（`PROMOTABLE` / `PROVEN_FALSIFIED` / `STEPPING_STONE_HANDOFF`）+ 实测/理论 $\tau$ 对齐检查；`v2/memory/store.py` 升级为 5 轴通用迁移引擎（默认关闭、支持 `code_direct / code_adapt / pattern_transfer / blocked` 自动隔离）；**17/17 单元测试全部通过**。
- **正在做**：`temp = 0`（Dense + Sparse MoE）与 `temp > 0`（T1-S 通用黑盒无损验证，不限内部走 Leviathan 链或 Target-Only / Sequoia 树）全套系统均已收口并同步。
- **接下来（可选）**：继续推进 `Qwen3.5-35B-A3B-FP8` Round 2 或扩展大 Batch（`bs=8/16`）场景验证。

---

## 快速自检

```bash
# 运行 SpecPilot v2 全部 CPU 单元测试（17 项）
python3 -m unittest discover -s v2/tests -p 'test_*.py' -v

# 查看本地双层 Structured Memory 1 行摘要索引
python3 v2/memory/store.py index
```

> 历史 v1 探索记录与 Ascend 轨道日志保留在 [HANDOFF.md](HANDOFF.md)、[PLANS.md](PLANS.md) 及 `docs/experiments/` 中存档。
