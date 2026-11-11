# slave21 单机阶段交付（2026-11-10）

## 完成情况

| 任务 | 本轮结果 | 仍需验证 |
|---|---|---|
| 当前环境核查 | 主机名 slave21；只读采集脚本和环境 JSON 已生成 | 当前进程命名空间无 GPU 设备、torch/vLLM；Docker socket 拒绝读取。不能据此认定物理 GPU 损坏 |
| profiling 字段 | 新增 CPU 调度信息、逐序列长度、VE、run ID、hostname、窗口标记；CPU 测试通过 | 实际 V0 worker/graph 路径与开销仍待 GPU 验收 |
| 显存容量工具 | 无 GPU 可运行；已输出默认切分示例；读取 checkpoint header 对账 | runtime/过渡峰值、有效状态与实际分配仍需在线测量 |
| 单卡重配置 | 两种分配器独立子进程脚本、CPU→GPU 复制计时与地址/内容检查已实现 | 本轮运行返回 unavailable，没有 GPU 时间数据 |
| 旧数据建模 | 步骤 CSV、阶段/层类型查表、整组 4K 诊断留出及误差分解 | 尚不能指导新切分，更不是最终独立验证 |

未启动或停止容器/Ray，未连接 slave20，未运行全模型 GPU 推理，保留原有 plugin.py 改动。

## 显存工具与对账

`python3 -m hybrid_pp.capacity` 接受层切分、保留请求的上下文长度、实际 Mamba 槽位数、分页大小和可选全局池 block 数。

- 权重：复用已有解析公式，声明 TP=1、bf16 权重/conv、fp32 SSM。另读 safetensors header（共 341 个张量），不加载大权重数据。
- KV：每个请求分别向上取整分页；有效 KV 与预分配池分开。V0 的全局池按 VE 除以 PP 后取整，工具反映该截断；VE 各自容量仍需另外校验。
- Mamba：按实际槽位数计算，不能用本轮 batch 替代。graph 补齐槽位必须从运行时取得后传入。
- runtime peak 和 transition peak 未测则保留 null，不给出“肯定装得下”的结论。

示例：default 切分、保留四条 16K 请求、Mamba 分配 16 个槽位（GiB）：

| rank | 权重解析估算 | Mamba 分配 | 有效分页 KV |
|---|---:|---:|---:|
| 0 | 5.955 | 0.634 | 0 |
| 1 | 5.611 | 0.475 | 0.500 |
| 2 | 5.611 | 0.475 | 0.500 |
| 3 | 5.942 | 0.555 | 0 |

这不是显存峰值，也不是实际缓存池总量。输入示例未指定池容量和额外 runtime，因此工具没有给出可行性认证。

用四种 graphlong 负载、每种四个 rank 的归档记录核对：16 项 Mamba 张量分配字节均与公式完全一致。

另发现旧 trace `stage_param_bytes` 在 profiler._open 中只统计内部模型；启动日志是较大统计范围。配置中单个词表矩阵为 `131072 × 4480 × 2 = 1174405120 B = 1.09375 GiB`，即外层 lm_head 的大小。该范围差异能解释之前权重数字的主要差额；解析权重与 checkpoint 总量仍有细小差异，不当作实际分配字节相等。

## profiling 的新增字段及窗口

`profile_features.schedule_features` 只读 CPU SequenceData，不读取 token 文本、不把 CUDA tensor 转回 CPU。记录 VE、每序列长度、已计算 token、调度 query/context 和实际序列长度列表。
调度信息与执行信息分开：多步执行、graph padding 仍需核对；保留状态数量没有可靠来源，明确记录 unknown，而不伪造为当前 batch。

`HYBRID_PP_RUN_ID` 写入 meta；`HYBRID_PP_WINDOW_FILE` 可指向由驱动维护的 warmup/main/cooldown 文件。不设置或不可读就写 unmarked。
新增 profile_window_client.py：对已启动服务先发送 warmup，再标记主窗口并按并发上限发请求，最后标记 cooldown。它是非流式测量，只报告 E2E 和服务返回的 usage，不声称测得 TTFT/TPOT，也不与历史 vllm bench 直接对比。
窗口文件读取是可选额外开销，必须做启用/关闭的 matched 对照。单机 worker 要能读取同一路径。

## 旧数据整理与初版查表

产物位于 `results/single_host_20261110/modeling_final/`：

- steps.csv：所有被接纳 trace 的逐步骤数据，保留 mixed、graph、phase、原始路径和 split。
- stage_lookup.json：仅训练数据建表；按实际 batch/上下文/query 分桶，并匹配层组成、并发上限与状态槽配置。留出观测另存 holdout_observations.json。
- layer_type_lookup.json：仅 eager 有逐层观测，汇总同一步内各类型层的平均时间；graph 不伪造逐层数据。
- holdout_predictions.json / summary.json：诊断预测与误差。

暂排除三处审计异常单元；每个 trace 明确跳过最初两步作为探索用热身规则，不删其余离群点。旧 trace 没主窗口标记，此规则不能保证只留下稳态样本。
4K 输入实验整组不参与 lookup 建表，mixed 步骤不进入纯阶段模型。全局 rank 不等于相同层组成，因此训练候选必须同时匹配 stage_pattern、静态并发与槽位设置。

最新诊断结果（对每个特征组的中位时间预测，各组等权）：

| 范围 | 组数 | 绝对相对误差中位数 |
|---|---:|---:|
| prefill | 32 | 0.46% |
| decode | 36 | 6.98% |
| 总体 | 68 | 1.00% |

总体平均误差约 5.81%，最差组约 42.06%，不能只看总体中位数。prefill 小误差可能来自固定 chunk 尺寸的重复覆盖，不能解释成任意长度的外推能力。
初版混合不同静态配置时 decode 中位误差约 58%，据此纠正了分组口径。由于这些诊断已用于修改，4K 不能再视为未碰过的最终验证集。新的独立运行与新切分验证仍需预先留出。
lookup 是探索基线，不用于宣称最优切分、端到端延迟精度或在线收益。

## 单卡重配置脚本

`scripts/memory_reconfig_bench.py` 不启动模型，分配小型 uint8 合成张量：

1. 普通 torch allocator：保留 weights，释放 KV 并 empty_cache，申请替代权重缓冲，从 pinned CPU 复制，随后恢复 KV。
2. vLLM 0.10.1.1 CuMemAllocator：weights/KV 标签池，sleep 备份 weights 并解除映射，使用释放空间申请替代缓冲并复制，释放替代缓冲后分别 wake weights/KV。

两种策略语义不同：cumem 还包含 weights CPU 备份，不能简单拿总耗时判定谁更快。记录每阶段 wall time、复制的 CUDA event 时间、allocated/reserved/free，保留首轮及后续轮次。恢复后检查原 weights 地址和全部内容。
每种机制用独立子进程；GPU 空闲状态不明或发现已有计算进程就跳过，不主动停其他任务。该检查不是资源锁，实际执行仍应使用已分配给自己的空闲 GPU。
本轮实际执行结果两种均为 unavailable（torch 未安装），详见 reconfig_attempt/summary.json；GPU 分支尚未实机通过。

## 验证及后续运行

8 项 CPU 单元测试通过：分页/预分配区别、按槽位计算、非法输入、chunked prefill/decode 信息、未知字段、worker 元数据传递与清理、warmup/main 请求窗口转换，以及训练/留出数据隔离。另对 16 项历史 Mamba 分配做精确对账。Python 编译和 git diff 空白检查通过。

在原实验容器内先执行只读检查（不需要 slave20）：

```bash
cd /workspace/hybrid-pp
python3 scripts/collect_single_host.py --out results/slave21_runtime/environment.json
```

确认容器有 torch/vLLM、GPU 为自己可用的空闲设备后运行小型基准：

```bash
python3 scripts/memory_reconfig_bench.py --device 0 --out results/slave21_reconfig
```

输出路径必须尚不存在。不要直接用旧 pp4.sh 启动缺少 slave20 的双机实验。
后续先小规模验证 profiling 新字段与开销，使用明确的主窗口；再固定单机模型/配置做代表负载补测。另一台机器恢复后才补跨机通信和完整四阶段验证。
