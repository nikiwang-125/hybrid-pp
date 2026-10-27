# 研究计划清单

## 阶段 0：环境（租卡第一天）
- [ ] 租一台 Linux 机器：2×3090/4090/A100，内存 ≥ 64GB，磁盘 ≥ 100GB，驱动 ≥ 570
- [ ] `bash scripts/setup_env.sh`
- [ ] `bash scripts/check_env.sh`：确认 `vllm.general_plugins` 里有 `hybrid_pp`

## 阶段 1：跑通并验证正确性（`scripts/day1.sh`）
- [ ] 不打补丁的 PP=2 确实失败（验证 findings.md 里的 bug 1 和 bug 2）
- [ ] 打补丁后，PP=2 和 PP=1 的输出完全一致
- [ ] 不均匀切分（20,36）的输出也一致
- [ ] PP=4（需要 4 卡）：验证 findings.md 里的问题 3（没有 attention 层的 stage）

## 阶段 2：读代码（对照 docs/findings.md 和 third_party/vllm）
- [ ] `nemotron_h.py`：模型结构和 PP 切分
- [ ] `worker/worker.py`、`executor/executor_base.py`：显存测算和取最小值
- [ ] `models/mamba_cache.py`：Mamba state 的管理
- [ ] `engine/llm_engine.py`、`core/scheduler.py`：virtual engine 和 micro-batch 调度
- [ ] 产出：调用链笔记和图

## 阶段 3：motivation 实验（最关键）
- [ ] 图 1：三类层的耗时随上下文长度的变化（PP=1，`runner --profile`，上下文从 1K 到 128K）
- [ ] 图 2：各 stage 的显存构成，以及取最小值造成的浪费（`analyze_memory`）
- [ ] 图 3：均匀切分时各 stage 的耗时和空泡（PP=2，`runner --profile` 和 `analyze_profile`）
- [ ] 图 4：不同负载下最优静态切分不同，用错切分会损失多少吞吐（`runner` 扫描切分 × 负载）
- [ ] 🚩 判断点：如果图 4 显示一种静态切分就能通吃各种负载，先停下来重新讨论方向

## 阶段 4：相关工作
- [ ] OpWeave（2609.14237）读全文：是否覆盖 SSM？是否随负载动态调整？
- [ ] KV-Pipe（2608.15943）、Megatron-LM PR #8366、AVMP（2605.22416）、Dual Pools（2610.15755）
- [ ] 再搜一轮 MLSys/OSDI/SOSP/EuroSys/ASPLOS
- [ ] 产出：一张对比表

## 阶段 5：方法设计（等阶段 3 数据出来）
- [ ] A：负载感知的动态切分（运行时迁移层）
- [ ] B：stage 之间不对称的显存管理（打破 min()）
- [ ] C：感知层类型的 micro-batch 组合

## 阶段 6：实现和评估
- [ ] Baselines：vLLM 均匀 PP、静态最优切分、TP、最新版 vLLM
- [ ] 负载：ShareGPT、LongBench、混合负载
- [ ] 扩大规模：4×A100 80G，跑 12B v2 / 47B / 56B

## 阶段 7：写作
