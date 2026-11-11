# 显存分配与重配置：第一轮代码核查（2026-11-10）

## 证据分级

- **本地代码/归档确认**：当前插件、runner、历史 server.log 与 trace 可直接检查。
- **版本源码确认**：通过网页读取 vLLM `v0.10.1.1` 标签上游源码，不能代替容器安装文件校验。
- **待实测**：释放多少、耗时多少、Mamba 状态在 sleep 中的实际处理、完整切换正确性。

## 最重要的结论

不能把旧版本描述为“没有考虑过回收显存”。v0.10.1.1 的 Worker 已有 sleep/wake_up：启用 sleep mode 后，权重与 KV 使用带标签的池。allocator 会解除物理映射，恢复时重新映射；sleep 的 offload_tags 指备份哪些标签，并不是只释放这些标签。
这说明可利用现成机制做基线；它没有证明能运行中缩小任意一段 KV 池、改变 PP 层归属或保留全部有效状态。

依据：[Worker](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.1.1/vllm/worker/worker.py) 的 load_model、initialize_cache、sleep、wake_up；[CuMemAllocator](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.1.1/vllm/device_allocator/cumem.py) 的 sleep、wake_up、use_memory_pool。

本仓库的 server.py/serve.sh 默认命令不传 --enable-sleep-mode，所检查的 graphlong 归档命令也没传。需在线确认配置默认值、插件及实际分配器，不能据此声称既有实验已经测了 sleep 能力。

## 分配与可变性初表

| 内容 | 已查到的行为 | 暂不能声称 |
|---|---|---|
| 权重 | 启动加载；开启 sleep 时使用 weights 标签池 | wake_up 可以自动增添本 stage 原先没有的层 |
| Attention KV | CacheEngine 按本地 attention 层循环创建 torch.zeros 张量；每 VE 有缓存实例 | 整卡只有一个巨大张量；逻辑 free block 就把显存归还给权重 |
| Mamba 状态 | 本地笔记与 v0.10.1 API 中按 max_num_seqs 分配，graph 模式可能补齐 | 与实际执行 batch 一样大；sleep 自动正确保存所有请求状态 |
| 激活/工作空间 | 先作为峰值与剩余项实测 | startup activation_gib 就是纯激活 |
| graph/通信缓冲 | 需要运行时独立核对 | sleep 后设备上所有其他显存也一定释放 |

KV 依据：[CacheEngine](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.1.1/vllm/worker/cache_engine.py)。
Mamba 依据：本地 docs/vllm_callchain.md，以及[相邻版本 v0.10.1 API 源码](https://docs.vllm.ai/en/v0.10.1/api/vllm/model_executor/models/mamba_cache.html)；v0.10.1.1 该文件网页获取失败，必须再核对实际安装文件。

Mamba 管理器在何时被惰性创建、是否位于带标签的 allocator 上下文，直接影响 sleep 行为。列为优先验证项，不能从类名或张量存在推断。

## 现有测量能提供的直接证据

results/pp4_graphlong/default/16384-128：四 rank 的 Mamba 实际张量记录分别为
680525824、510394368、510394368、595460096 字节。
默认切分 rank0/rank3 没有 attention，因此 stage_mem 打印的约 13.82/13.89 GiB KV **预算**不等于实际分配了同样大的 KV 张量。
需要将预算、最终 block 数、张量分配量、有效使用量分别记账。

## 第一批微基准设计（待 GPU 环境恢复后执行）

先用隔离进程、无在途请求，保留原服务；各机制分开运行，避免缓存分配器与失败操作污染后续轮次。

| 对照 | 目的 | 必记内容 |
|---|---|---|
| 普通张量池：无引用后释放、empty_cache、重新分配 | 测普通分配器操作基线 | allocated/reserved/free、分配/释放/清零分别耗时 |
| 旧版 CuMemAllocator 标签池 sleep/wake | 测物理解除映射/恢复 | CPU 备份与恢复成本、实际回收字节、是否保持地址 |
| 同版本 vLLM 小规模实际引擎 | 检查机制在真实权重、KV、Mamba 下是否适用 | 各状态归属、graph 兼容性、恢复后输出 |

合成张量不是移动真实层的实验，只能按明确字节量测 allocator 操作。权重写入分别标明预先驻留 CPU、其他 GPU 或磁盘来源；不能把省略传输的合成实验称为权重迁移。
每个操作前后明确同步边界；用 host wall time 记录可能含 CPU/驱动的操作，用 CUDA event 补充设备复制时间，两者不相加冒充不同成本。
先试少量大小与重复轮次，记录首轮和热身后分布，不预设耗时。

## 进一步布局切换的条件

单卡完成缓存空间回收并不代表完成流水线切换。后续需明确：排空或同步点、源/目标层权重、KV/Mamba 有效状态迁移或重算、缓存索引与层归属更新、graph 处理、恢复输出正确性。
最终布局容量与过渡峰值容量分别检查；阶段操作能否重叠尚未知，因此不能用简单求和声称完整切换中断。

本轮不测完整动态系统，不声称已有迁移收益；先用离线切分验证建立收益依据，再判断是否值得投入在线机制。
