# SpecPilot v2 全盲测自主优化闭环记录（Round 1–4，2026-11-25）

## 1. 目标与系统架构

在单卡 H100（`Qwen3.8-27B BF16 + DFlash2 + SGLang v0.5.20`，单请求 greedy 解码）上，完整复现 *SpecPilot: Autonomous Algorithm Discovery and Systems Optimization for Speculative Decoding*（arXiv:2611.02501v1）的核心自主研发闭环：

- **Phase 0（硬件精确三因子 Ceiling）**：[v2/phase0/floors.py](../../v2/phase0/floors.py)、[v2/phase0/measure_stock.py](../../v2/phase0/measure_stock.py)。实测 AR = `51.61 tok/s`（`19.38 ms`），Stock = `194.37 tok/s`（$\tau=4.82$，$T_{\text{step}}=24.79\text{ ms}$，目标权重流占比 `87.8%`）。
- **Phase 1–2（严格盲测 Master + 多路并行 Direction + 持久探索池）**：
  - 8×H100 NorthJob 持久探索池（[v2/sandbox/pool_worker.py](../../v2/sandbox/pool_worker.py)、[v2/sandbox/run_gpu.py](../../v2/sandbox/run_gpu.py)、[v2/sandbox/worktree.py](../../v2/sandbox/worktree.py)、[v2/sandbox/northjob_pool.py](../../v2/sandbox/northjob_pool.py)），GPFS 轮询延迟 2–8s，支持 3-tier 阶梯验证（Tier 1 算子正确性 → Tier 2 微基准 → Tier 3 1-GPU A/B）。
  - 盲测隔离审计：Round 1–4 全部研究角色 0 次联网搜索、0 次读取禁区（`docs/`、`HANDOFF.md`、`PLANS.md`、`README.md`、`eval/sealed/`、`memory/`、`ledger/`、作者 `final` 实现）。
  - Priority 0 协议（[v2/agents/master_prompt.md](../../v2/agents/master_prompt.md)）：若上一轮出现高加速（`t2_pass = true`）但 T1 Token Forcing 被拒（`t1_pass = false`）的候选，下一轮优先诊断修复或证伪，闭环后再重测 Ceiling 决定新方向。
- **Phase 3（隔离 2×H100 独立 Controller T1 + T2）**：[v2/controller_verify_2gpu.py](../../v2/controller_verify_2gpu.py)，30 条 dev 请求、2048 cap、两轮物理 GPU 互换配对 CI（`controller.stats.stratified_geomean_ci`）+ FP32 LM-head teacher-forced Token Forcing（$\Delta = 0.25$）。不修改 `controller/`、`eval/`、`ledger/`。
- **Phase 4（T3 Sealed Final + T4 Structured Memory）**：
  - [v2/controller_sealed_final.py](../../v2/controller_sealed_final.py)：30 条严格未见请求（`indices 6..11`，与 dev `0..5` 互不相交）配对测速 + 未见集 2048-cap & 4096-cap 双重 FP32 Token Forcing。
  - [v2/memory/distill.py](../../v2/memory/distill.py)：从干净 `stock` 沿 `requires` 依赖链逐字节还原各代引擎并校验 `TREE_SHA`。

## 2. 独立 Controller 与 Sealed Final 验收总表

| 轮次 | 盲测独立发现机制（全部 default-OFF `SP_OPT_*`） | Dev 30-req 2048-cap 配对加速比 (`tok/s`, $\tau$) | Dev T1 Forcing (不匹配率 / 大 margin 事件) | Sealed 30-req 未见集 (`tok/s`, $\tau$) | Sealed Forcing 2048 / 4096 事件 | T4 Memory 逐字节还原 | 判决 |
|---|---|---|---|---|---|---|---|
| **Baseline** | Stock SGLang + DFlash2 | `1.00×` (`190.41`, $\tau=4.65$) | `0.7824%` / `0` | `1.00×` (`174.84`, $\tau=4.26$) | `0` / `0` | — | 基线 |
| **Round 1** | Zero-snapshot CUDA GDN + `block_size=12` + 28-node Best-first Tree | `1.2960× [1.2712, 1.3219]` (`248.49`, $\tau=6.70$) | `0.7463%` / `0` | — | — | `r1_tree_gdn_b12` (`true`) | `PROMOTED` |
| **Round 2** | 无损 12-bit BF16 线层（重发现 W12）+ GDN Tree v2 + FP8 Drafter + 12-bit Stream-K `lm_head` | `1.5561× [1.5161, 1.5996]` (`304.58`, $\tau=6.77$) | `0.7095%` / `0` | — | — | `r2_w12_gdnv2_fp8draft` (`true`) | `PROMOTED` |
| **Round 3** | All-MMA Tree-WY GDN + Fused GQA Suffix Tree Attention | `1.6260× [1.5848, 1.6717]` (`319.77`, $\tau=6.83$) | `0.7378%` / `0` | `1.5745× [1.5440, 1.6080]` (`279.12`, $\tau=5.95$) | `0` / `0` | `r3_tree_wy_attn` (`true`) | `PROMOTED` + `SEALED` |
| **Round 4（初版，被拒）** | Fast `sm_90` W6-v2 两级 Lookahead + 非对称 `M_q=16` 自适应 | `1.7312× [1.6901, 1.7736]` (`331.87`, $\tau=10.28$) | `0.9653%` / **`15` (`p=1.87e-5`)** | — | — | 未入库 | **`REJECTED (T1)`** |
| **Round 4（修复版）** | 同上 + **精确 FP32 subwarp 反向三角 WY 求解** + `M_q=16` CUDA Graph 树缓冲捕获修复 | **`1.7068× [1.6574, 1.7580]`** (**`326.97`**, $\tau=10.05$, `6.34×` AR) | **`0.6739%` / `0` (`p=1.0`)** | **`1.6700× [1.6273, 1.7148]`** (**`292.05`**, $\tau=8.70$) | **`0` / `0` (`p=1.0`)** | **`r4_w6v2_adaptive_fixed` (`true`)** | **`PROMOTED` + `SEALED`** |
| **Round 5（收官轮）** | Always-On 2-Pass (`MODE=2`) + 消除 2-pass `.item()` GPU-CPU 同步（`SP_OPT_W6_CACHE_NUM_TOK_CPU=1`）；同步在硬件屋顶证伪 D1/D2/D3/D5 四条路径 | **`1.7313× [1.7006, 1.7654]`** (**`334.96`**, $\tau=12.20$, `6.49×` AR) | **`0.8057%` / `0` (`p=1.0`)** | `1.6488× [1.6005, 1.6964]` (`290.52`, $\tau=10.54$) | **`0` / `0` (`p=1.0`)** | **`r5_mode2_nosync` (`true`)** | **`PROMOTED` + `SEALED` + 停机收官** |

## 3. Round 4 Token Forcing 拦截与闭环修复记录

- **拦截实证**：`r4-blind-d6-w6v2-adaptive` 在 Dev Controller T2 测得 `1.7312×`（`331.87 tok/s`），但 T1 Token Forcing 捕获到 **15 个大 margin 事件（$\Delta > 0.25$，`event_p = 1.87e-5`）** 并严格拒绝晋升。
- **根因定位与消融**：
  1. **主因**（`gdn_verify_fast_cuda.py:1708-1770`）：`gdn_verify_fast_tree_wy_cuda_kernel` 使用 5 步 TF32 Neumann 级数倍增 $(I-L)(I+L^2)(I+L^4)(I+L^8)(I+L^{16})$ 求 $(I+L)^{-1}$。在 R3 中树深 $\le 11$ 误差仅 `~0.014`；在 R4 两级 W6 Lookahead 中 32 节点树深达 **31**，$L^8 / L^{16}$ 暴涨至 $\sim 1.65\times 10^5$ 并在 TF32 下灾难性相消，产生高达 `38.04` 的 logit margin 偏差。关闭 `SP_OPT_GDN_TREE_WY=0` 后大 margin 事件立即从 `16 → 0`。
  2. **次因**（`hybrid_linear_attn_backend.py:626-662`）：`M_q=16` 时 `_capture_metadata` 未绑定与 `_replay_metadata` 一致的 `_w6_mq_bufs_cache[16]`。
- **修复与复验**：改用精确 FP32 subwarp 反向三角求解（8 lanes/row，循环内 0 次 `__syncthreads()`，Tier-1 depth=31 最坏误差 `0.000488`，Tier-2 延迟 `27.23 us` 较 `tree_v2` 快 24%）并修复 `_capture_metadata` 绑定。修复后 `d6_fix` 关掉全部开关 30/30 prompts 逐 token 复现 `stock`，并在独立 2×H100 Controller Dev（`0.6739%` / `0` events）与 Sealed Final 2048（`0.9042%` / `0` events）和 4096（`0.9175%` / `0` events）全线通过。
- **T4 Memory 传递校验**：`stock → r1_tree_gdn_b12 → r2_w12_gdnv2_fp8draft → r3_tree_wy_attn → r4_w6v2_adaptive_fixed` 重建 `TREE_SHA = a6b67a779cd996531cd2ac306425efcd57d6a59df2995f26fa1e92201ceaa273`，`byte_exact_match = true`。

## 4. 关键 NorthJob 任务索引与本地证据路径

- Dev Controller R1：`job-772a4591ebc8-20261124211302`（`v2_runtime/controller/r1-blind-integrated-v1-a02/verdict.json`）
- Dev Controller R2：`job-13e54d87cf52-20261125012349`（`v2_runtime/controller/r2-blind-integrated-v1-a02/verdict.json`）
- Dev Controller R3：`job-3c8ae6b43f8e-20261125113335`（`v2_runtime/controller/r3-blind-integrated-v1/verdict.json`）
- Sealed Final R3：`job-30bdbd5a6d33-20261125121511`（`v2_runtime/sealed/r3/sealed_verdict.json`）
- Dev Controller R4（初版被拒）：`job-dd5567e768ef-20261125143820`（`v2_runtime/controller/r4-blind-d6-w6v2-adaptive/verdict.json`）
- Dev Controller R4（修复晋升）：`job-9f2490ce5d07-20261125162408`（`v2_runtime/controller/r4-blind-d6-fix-v1/verdict.json`）
- Sealed Final R4（修复通过）：`sp-v2-sealed-r4-fix-20261125-a01`（`v2_runtime/sealed/r4_fix/sealed_verdict.json`）
- Dev Controller §6.1 消融（Single-Agent 无 Cost Model）：`job-5e164dc49eee-20261125183458`（`v2_runtime/controller/ablation-single-nocm-v1/verdict.json`）
- Dev Controller R5（晋升）：`job-1e8c09a76071-20261125205337`（`v2_runtime/controller/r5-blind-mode2-nosync-v1/verdict.json`）
- Sealed Final R5（通过）：`job-3c4406f256ea-20261125210932`（`v2_runtime/sealed/r5/sealed_verdict.json`）
- 本地单元测试：`python3 -m unittest discover -s v2/tests -p 'test_*.py' -v`（5/5 通过）。

## 6. Round 5 物理屋顶收敛与有据停机结论

- **4 条硬件/算法轴全部到达 H100 Pareto 边界**：
  - **Exact 12-bit W12 目标 GEMM**：CUDA Graph 内实测 `15.38 ms`（`~2.45 TB/s` = H100 `2.91 TB/s` 流带宽屋顶的 84%），D1 优化仅剩 `0.70 ms` 空间且 Tier-3 净收益为负（`0.9794×`）；D3 无损 sub-12-bit Huffman 编码（0 bit 误差、`10.74 b/param`）在 SM90 `M=32` 受 SMEM bank conflict 与 INT32 ALU 限制慢于 12-bit SWAR（`0.605×`）。
  - **Lookahead 量化与非 GEMM**：D2 实证 W4/W5 一致率掉至 `88.4%/92.8%`（vs W6 `97.4%`），W6 为 Pareto 最优；D5 实证非 GEMM 无损空间 `< 0.3 ms`，仅提取了 3 行 `dflash_worker_v2.py:2703` `.item()` GPU-CPU 同步修复。
- **Dev vs Sealed 泛化对比（Phase 4 价值实证）**：
  - 在接受率较高的 Dev 集（stock $\tau=4.65$）上，R5 `MODE=2` 推高 $\tau \to 12.20$ 并取得最高 Dev 成绩 **`1.7313× [1.7006, 1.7654]`**（`334.96 tok/s`，`6.49×` AR，`0` events）。
  - 在更难的 Sealed 未见集（stock $\tau=4.26$）上，R5 仍 100% 无损通过（2048 & 4096 均 `0` events），加速比 **`1.6488× [1.6005, 1.6964]`**（`290.52 tok/s`），与 R4 自适应 `MODE=1` 的 **`1.6700× [1.6273, 1.7148]`**（`292.05 tok/s`）CI 重叠且略低——说明 `MODE=1` 在低接受率域更稳健，`MODE=2` 在高接受率域更快；两者共用同一套 `r5_integrated` 引擎（仅运行时环境变量切换）。
  - 传递记忆校验：`stock → r1 → r2 → r3 → r4 → r5_mode2_nosync` 重建 `TREE_SHA = 8e5151207c22171ecfb21a4574c2fc1d26a43263cadfda98eba0f734cd720cb1`，`byte_exact_match = true`。Blind Round-5 Master 正式建议停止继续盲搜并封板收官。

## 5. 论文 §6.1 对照消融：无 Cost Model 单 Agent vs SpecPilot 完整架构

为检验 **Phase 0 硬件精确三因子 Ceiling** 与 **Master + 多路并行 Direction** 架构的因果贡献，在同一干净 `stock` 引擎起点、同一严格盲测约束（审计 0 联网、0 读禁区与 `v2/phase0`）下，运行单 Agent 串行消融 `Ablation-Single-NoCM`（GPU slot 0，3 次串行尝试）：

- **Attempt 1**：`SP_OPT_DFLASH_BLOCK_SIZE=16` → Tier-3 `1.0387×`（`202.11 tok/s`，10/10 exact match）
- **Attempt 2**：`+ SP_OPT_GDN_REPLAYSSM=1` → Tier-3 `0.9549×`（回退）
- **Attempt 3**：`+ SP_OPT_GDN_VERIFY_FLASHINFER=1` → Tier-3 `1.0576×`（`205.74 tok/s`，FP32 等价 `max_diff_s=7.15e-7`）

将消融基线的最强方案（Attempt 3）送入**同一套 2×H100 独立 Dev Controller（30-req 2048-cap 双卡互换 + FP32 Token Forcing）**，同口径对比如下：

| 系统配置 | 自主发现机制范围 | Dev 30-req 2048-cap 配对加速比 (95% CI) | 吞吐 (`tok/s`) | $\tau$ | T1 Forcing (`rate` / `events`) | Controller 判决 |
|---|---|---|---:|---:|---|---|
| **无 Cost Model 单 Agent 消融** | `block_size=16` + 切换 FlashInfer GDN 后端（局部配置/后端切换） | **`1.1173× [1.0809, 1.1590]`** | `209.79` | `5.30` | `0.7045%` / `1` (`p=0.981`) | `PROMOTED` |
| **SpecPilot 完整架构（仅 Round 1）** | Zero-snapshot CUDA GDN + `b=12` + 28-node Best-first Tree | **`1.2960× [1.2712, 1.3219]`**（较消融 **`+16.0%`**，CI 无重叠） | `248.49` | `6.70` | `0.7463%` / `0` (`p=1.0`) | `PROMOTED` |
| **SpecPilot 完整架构（Round 1–4）** | + 无损 12-bit W12 + FP8 Drafter + 精确 FP32 subwarp Tree-WY + W6 两级 Lookahead | **`1.7068× [1.6574, 1.7580]`**（较消融 **`+52.8%`**） | **`326.97`** | **`10.05`** | `0.6739%` / `0` (`p=1.0`) | `PROMOTED` + `SEALED` (`1.6700×`) |

结论：单 Agent 在没有 Cost Model 引导时并非无效（真实拿到 `1.1173×` 有效无损提速），但其探索停留在局部配置与现成后端切换；引入 Phase 0 三因子屋顶与 Master/多路并行架构后，系统首轮即突破至树验证与定制 CUDA kernel（`1.2960×`），并在 4 轮内直击权重带宽与深树验证瓶颈达到 `1.7068×`（未见集 `1.6700×`）。

> 注：`v2_runtime/` 中的工作树、原始日志、池收件箱与 tarball 属于本地运行产物，已列入 `.gitignore` 不提交公开仓库；`controller/`、`eval/sealed/`、`ledger/` 未修改。
