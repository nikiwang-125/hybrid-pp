# 显存可变性、重配置成本基线与通信模型（2026-11-11，slave21 + slave20）

环境：2×(2×RTX 3090 24GB)，容器 nikihpp，torch 2.7.1+cu128，vLLM 0.10.1.1（V0），NCCL 2.25.1。
Ray 集群两节点在线（192.168.0.21 / .20，共 4 GPU）。原始数据：`results/reconfig/`、`results/comm/`，日志 `logs/reconfig.log`、`logs/comm.log`。

## 1. 显存分配方式与可变性（按已安装 vLLM 源码核对）

源码路径前缀：`/usr/local/lib/python3.12/dist-packages/vllm/`

| 内容 | 何时分配 | 谁管理 | 大小由什么决定 | 运行中能否缩放/归还 |
|---|---|---|---|---|
| 权重 | 启动 `Worker.load_model`（worker/worker.py:200） | sleep mode 下在 `weights` 标签池 | 本 stage 的层 | sleep 可整体卸到 CPU 再恢复；**不能**借此加入本 stage 原本没有的层 |
| Attention KV | 启动 `initialize_cache → _init_cache_engine`（worker.py:313）一次性分配 | 每个 VE 一个 `CacheEngine`，每个本地 attention 层一个 `torch.zeros` 张量（worker/cache_engine.py:46,96） | `num_gpu_blocks // PP`，固定 | **不能缩放**。scheduler 释放 block 只是逻辑空闲，显存不还给分配器。腾出空间 = 释放张量 → 重新申请（服务中断，KV 内容丢失）|
| Mamba 状态 | **惰性**：首次 forward（即启动时 profile_run）在 `NemotronHForCausalLM.forward` 创建（model_executor/models/nemotron_h.py:583-599） | `MambaCacheManager`，conv/temporal 两个整块张量（mamba_cache.py:46-53） | `max_num_seqs`（graph 模式按 capture size 补齐）× 本 stage Mamba 层数，**与实际 batch 无关** | 不在任何 sleep 标签池中，sleep **不释放**；无缩放接口 |
| 激活/工作空间、graph 池、CUDA context | 运行时 | PyTorch 缓存分配器 / CUDA | profile 峰值；graph capture sizes | sleep 后仍常驻（见下：约 1.0 GiB）|

结论：KV 池确实是启动时一次性预留的固定空间。“把一部分 KV 空间改放权重”在现有代码中只能是 **释放 KV 张量 → 加载新层权重 → 以更小 block 数重建 KV**，且 KV/Mamba 缓存索引、CUDA graph（固化了 KV/Mamba 地址）都要重建。

## 2. 重配置成本基线

### 2.1 微基准（`reconfig_cost micro`，GPU0，3 轮取平均，首轮热身已丢弃）

场景：一张卡上有 6 GiB KV 池（2 个 attention 层），要接收 3 个新层；在途 32 条请求（attention 每条 4096 token）。

| 移入 | 权重 | ① 释放 KV 池 | ②a 从 safetensors 加载* | ②b 从 pinned 内存加载 | ③ 重建 KV 池 | 在途状态 | ④a 状态→另一 GPU | ④b 经 CPU 往返 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 个 attention 层 | 315 MiB | 6.3 ms | 202 ms | 13.7 ms | 7.1 ms | 1536 MiB | 155 ms | 1044 ms |
| 3 个 Mamba 层 | 844 MiB | 4.8 ms | 267 ms | 47.1 ms | 6.5 ms | 487 MiB | 48 ms | 331 ms |
| 3 个 MLP 层 | 804 MiB | 6.3 ms | 351 ms | 44.4 ms | 6.6 ms | 0 | — | — |

\* 文件已在 page cache 中，不是冷磁盘读取。

- KV 池释放与重建本身只需毫秒级；成本主要在**权重搬运**与**在途状态迁移**。
- Attention 状态随上下文线性增长（此例 1.5 GiB），Mamba 状态与上下文无关但按槽位固定；Mamba/MLP 层权重约为 attention 层的 2.7 倍。
- 卡间状态复制约 10 GB/s，暂未确认是否走 P2P。
- 未计入：scheduler/block table 重建、跨机迁移（跨机带宽见第 3 节，约 1.2 GB/s，1.5 GiB 约需 1.3 s）。

### 2.1b CUDA graph 重新捕获（来自 PP=4 实测 server.log，16 次启动 × 4 段）

V0 在 `worker/model_runner.py:1441` 对**每个 VE** 分别捕获 graph，且 graph 固化了该 VE 的 KV/Mamba 张量地址。PP=4、capture sizes [16,8,4,2,1] 时每段捕获 4×5 = 20 个 graph。

| 项 | 实测 |
|---|---|
| 每段 graph 捕获 | **7–14 s**（多数 7–9 s），各段并行 |
| graph 额外显存 | 0.06–0.10 GiB/段 |
| init engine（profile + 建 KV + warmup，含捕获） | 17.8–19.4 s |
| 对照：PP=1、4 个 capture size | 2 s |

含义：只要重建 KV 池，graph 就失效，需重捕获。**一次布局切换即使不重启引擎，仅 graph 重捕获就约 7–14 s 中断**，远大于释放/重建 KV（毫秒级）与权重搬运（百毫秒级），成为切换成本的主项。若切换期间改用 eager 运行，可避开中断，但 decode 每段会慢 2–4 倍（见 modeling_20261111.md）。

### 2.2 vLLM sleep mode 与整引擎重启（PP=1，GPU0，max_model_len 4096，max_num_seqs 8）

| 操作 | 耗时 | 显存 |
|---|---:|---|
| 整引擎启动（重启上界） | 60.0 s（权重加载 5.0 s，显存 profiling 26.3 s，KV+graph 4 s） | 醒着 20.66 GiB |
| sleep level 1（权重备份到 CPU、释放 KV） | 首次 11.1 s，之后 0.94–0.97 s | 睡眠时仍占 1.00 GiB |
| wake 权重（16.58 GiB 拷回） | 0.736 s（约 22.5 GiB/s） | 17.59 GiB |
| wake KV（仅重新映射，内容不保留） | 0.4 ms | 20.66 GiB |

3 轮恢复后 greedy 输出与 sleep 前一致。进程在写出结果后退出阶段发生 segfault（exit 139），不影响上述数据，但需另查。

含义：
- 整机重启式切换 ≈ 1 分钟中断，主要耗在显存 profiling，而非权重加载。
- sleep/wake 能在 ~1 s 级释放并恢复权重与 KV，但 KV 内容被丢弃（在途请求需重算），Mamba 状态与其他约 1 GiB 常驻，且无法改变层归属。它是“释放/恢复”机制，不是“布局切换”机制。

## 3. 通信模型（stage 边界张量：hidden_states + residual，各 [T, 4480] bf16）

`hybrid_pp/comm_bench.py`，Ray 两 actor 组 NCCL；ping-pong 半往返（排除下游未就绪等待）与 barrier 后单向两种计时，50 次中位数。

| token 数 | 数据量 | 机内 | 跨机 |
|---:|---:|---:|---:|
| 1 (decode) | 17.5 KiB | 0.27 ms | 0.25 ms |
| 8 | 140 KiB | 0.09 ms | 0.27 ms |
| 32 | 560 KiB | 0.09 ms | 0.63 ms |
| 512 | 8.75 MiB | 0.57 ms | 7.9 ms |
| 2048 | 35 MiB | 2.1 ms | 31.3 ms |
| 8192 | 140 MiB | 8.2 ms | 124.9 ms |

拟合 t = a + bytes/BW（≥1 MiB）：机内 a≈0.11 ms，BW≈18.6 GB/s；跨机 a≈0.12 ms，BW≈1.18 GB/s。
跨机走 NCCL socket（`NCCL_SOCKET_IFNAME=eno12399np0`，10 GbE，无 InfiniBand），已达线速约 94%。

含义：PP=4 默认映射下 stage1→stage2 跨机。decode 跨机 < 1 ms，可忽略；但 **prefill 每 2048 token 的 chunk 跨机传输约 31 ms**，与单 stage 计算时间同量级，切分搜索必须把跨机边界的 prefill 通信计入，且跨机边界位置会影响长输入负载下的最优切分。
单向与半往返结果一致（差 <5%），说明孤立测量中没有混入等待；真实流水线中还有 vLLM 每步经 gloo 发送的元数据，以及计算-通信重叠，需用真实 trace 校准。
