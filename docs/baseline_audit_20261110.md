# 第一阶段：环境与已有数据核查（2026-11-10）

## 当前状态与证据边界

已完成本地文件、代码、PP=4 原始数据的离线核查；未启动推理、未变更已有补丁、未覆盖旧结果。
当前会话 `nvidia-smi` 无法连接 NVIDIA 驱动；`docker ps` 返回 Docker socket 权限不足。
因此本报告不是硬件在线验收，尚不能确认两台机器当前容器、GPU 空闲状态、代码一致性和网络拓扑。

## 1. 实验基础

| 项目 | 已确认内容 | 证据 |
|---|---|---|
| 主仓库 | `/home/mzzguest1/niki/hybrid-pp` | 本地目录 |
| 当前 HEAD | `d6e4593437f4806d2b7f3c524e9fc22ce2a409de` | git rev-parse |
| 引擎 | vLLM 0.10.1.1 / V0 | 旧 server.log 的初始化行；Dockerfile |
| 模型 | NVIDIA-Nemotron-Nano-9B-v2 | args.json、server.log |
| 本地配置快照 | `6533e8de2c68e4536bf7c411d7a3ce5734111476` | HF 缓存目录；历史运行 revision=None，不能据此保证当时加载了同一快照 |
| 精度 | bf16 权重；Mamba SSM 指定 float32 | 初始化日志、serve 命令 |
| 并行 | PP=4、TP=1 | 初始化日志、args.json |
| GPU 型号 | 归档 trace 的全部 meta 均为 RTX 3090 | 历史测量，不代表当前在线硬件验收 |
| 双机组织 | 脚本设计为 slave21 head + slave20 worker，Ray | scripts/ray_up.sh；当前映射仍需在线确认 |
| 默认切分 | `[14,14,14,14]`；attention 分布 `[0,2,2,0]` | trace meta 的层范围与 stage_pattern |
| 既有正确性 | PP=1 与 PP=4 的 10 个测试提示输出相同 | logs/pp4_correctness.log；只证明这批样例 |

工作区原本已有改动：`hybrid_pp/plugin.py` 修改了 Ray 环境变量传递；`scripts/pp4.sh`、`pp4_layers.sh`、`plot_pp4_layers.py` 未跟踪。本轮保留这些内容。
复现必须保存实际文件、哈希及 diff，不能只写一个 HEAD。审计工具已保存本轮源码快照；这不能反向证明历史运行代码完全一致。

## 2. 已有 PP=4 数据清单

用 `scripts/audit_baseline.py` 对三个目录做逐行 JSONL 扫描，输出 inventory.json / inventory.csv / environment.json、当前源码快照与 tracked diff。

| 数据组 | 审计单元 | bench.json | trace 文件 |
|---|---:|---:|---:|
| results/pp4 | 13 | 12 | 26 |
| results/pp4_graphlong | 4 | 4 | 16 |
| results/pp4_layers | 16 | 16 | 64 |
| 合计 | 33 | 32 | 106 |

这里的单元含一个已失败残留目录，不等于 33 次成功实验。
共解析 215859 条 step：纯 prefill/eager 5337、纯 decode/eager 133225、混合/eager 27392、纯 decode/graph 49905。
未发现 JSON 行解析失败或同一 trace 内重复 step ID；成功的 profiling 单元具有四个 rank 的 meta。
这些是结构完整性检查，不等于计时方法和预测误差已经通过验证。

### 需隔离的三处

1. `results/pp4/profile/18-10-10-18/8192-128`：有 FAILED 标记，同时存在 completed=32 和四 rank trace。
2. `results/pp4/profile/default/16384-128`：同上。
3. `results/pp4/profile_failed_permission/default/1024-128`：无 bench、无 args，仅两个 rank 的 trace。

前两处可能是重跑后的历史标记，但当前文件不能独立建立执行先后和来源一致性。保留原件，暂不作为模型校准或验证样本；不删除 FAILED 来制造成功状态。

## 3. 可以复用什么

- pp4_layers：default 切分，concurrency=1/4/16/32；输入标签为 1024/4096/16384/32768。适合探索 eager 层级时间与实际 batch、ctx_sum、ctx_max 的关系。
- pp4_graphlong：default、concurrency=16、四种长度，实际存在 graph decode。可对照 graph/eager 的执行差异。
- pp4/sweep：三种切分、两种负载，六份 completed=64 的 benchmark，可作为历史候选对照。
- 这些实验未证明有独立重复测量；不能用单次小幅差异证明优化有效。不同数据组的输出长度、并发、插桩、graph 上限不同，不能直接相减当作 graph 加速收益。

历史 sweep 输出吞吐（tok/s，单次观测）：

| 切分 | 1024:256 | 16384:128 |
|---|---:|---:|
| default | 242.93 | 37.42 |
| 18,10,10,18 | 237.68 | 35.77 |
| 16,12,12,16 | 247.47 | 36.43 |

已有样本不足以证明非均分切分稳定优于默认方案，也没有证明值得在线切换。

## 4. 建模前必须修正的口径

1. `ctx_sum/ctx_max` 不是逐请求历史分布。当前 trace 缺每请求 query 长度、已有 context、virtual_engine、请求/微批次关联和保留状态数量。混合步骤不能当纯 prefill 或纯 decode。
2. CUDA event 区间可能包含 CPU 发射不及时导致的 GPU 空闲；层耗时不是纯 kernel 耗时。CPU recv/send 区间含等待，不能拟合成纯链路传输。
3. 每张 GPU 的 t_start_ms 使用本地首个事件为零点，不能直接拼成跨 GPU 同一时间轴。wall 也尚无跨机器时钟校准。
4. bench 有初始单请求测试，profiler 没有明确标记主 benchmark 窗口。需增加窗口/阶段标记，不能默认全部 trace 都属于主测试。
5. 部分首次 decode 的层耗时明显偏大，需明确热身定义，不能任意删异常点，也不能直接作为稳态样本。
6. startup 的 weights_gib 与 trace 的 stage_param_bytes 不是同一口径。后续离线核查定位到主要统计范围差异：profiler 累加内部 NemotronHModel 的参数，外层 lm_head 不在其中。当前配置一个 lm_head 为 1.09375 GiB，可解释 rank0 约 5.96 与 4.86 GiB 的主要差额；不应直接叫碎片。详见 single_host_progress_20261110.md。
7. startup 的 activation_gib 不应直接等同纯激活，可能包含惰性初始化状态等；Mamba 状态需独立对账，防止重复计入。
8. 负载目录是目标长度标签，实际输入 token 总量以 bench 和步骤记录为准。

## 5. 下一批最小实验的执行顺序

先在原运行容器做只读在线核查：两端记录 nvidia-smi、topo、torch/vllm/transformers/ray 版本、插件入口、实际源码哈希、模型配置哈希、Ray 资源和 rank→机器/GPU UUID 对应关系。
不要直接运行会创建/删除容器、启停 Ray 的脚本充当环境检查。

环境可用后依次进行：

1. 固定模型快照、代码快照、default 切分与全部服务参数，先小规模正确性复验。
2. 补齐 profiling 字段与测试窗口标记；检查插桩开销，matched eager/graph 配置对照。
3. 先选短/长输入和小/大工作量四个代表点；建模前指定留出配置，独立运行重复测量，不先全组合扫描。
4. 同步记录缓存预算、实际张量字节、有效缓存量、torch allocated/reserved 与设备 free；注明采样同步成本。
5. 做无在途请求的显存重配置微基准；具体机制见 memory_reconfiguration_audit_20261110.md。

先不提供“已验证可运行”的新 GPU 命令：当前会话无法执行原容器，因此启动路径与参数支持尚未在线确认。现有 pp4.sh 是历史入口，不应未经核查整批重跑。

## 6. 审计复现

```bash
cd /home/mzzguest1/niki/hybrid-pp
python3 scripts/audit_baseline.py --out results/audit_20261110_v2
```

工具只读原结果；输出目录必须不存在，避免覆盖审计产物。范围只包括 PP=4 三组，不包括已有 PP=2 全部数据。
