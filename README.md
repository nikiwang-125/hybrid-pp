# hybrid-pp

研究 Mamba-Transformer 混合模型（Nemotron-H / Nemotron Nano v2）在**流水线并行（PP）推理**中，由层间异构导致的 stage 计算不均衡和显存利用率低的问题。

- 推理框架：**vLLM v0.10.1.1，V0 引擎**（`VLLM_USE_V1=0`）
- 主力模型：`nvidia/NVIDIA-Nemotron-Nano-9B-v2`（56 层，4 个 attention 层，支持 128K 上下文）
- 不改 vLLM 源码，所有改动通过 vLLM 插件（`vllm.general_plugins`）注入

> ⚠️ 原版 vLLM v0.10.1.1 的 NemotronH 在 PP>1 时跑不通，本仓库的插件自带修复。详见 [docs/findings.md](docs/findings.md)。

## 快速开始（Linux + NVIDIA GPU）

```bash
git clone <this repo> && cd hybrid-pp
bash scripts/setup_env.sh                  # conda 环境 + vllm + 本仓库 + 下载模型
conda activate vllm010
mkdir -p logs && bash scripts/day1.sh 2>&1 | tee logs/day1.log   # 第一天全流程验证
```

`day1.sh` 依次运行：环境检查 → 静态 stage 分析 → 逐个证实 3 个上游 bug → 打补丁的 PP=2 与 PP=1 输出对比 → 小规模切分扫描 → profiling。跑完后把 `logs/day1.log` 和 `results/day1/` 带回来分析。

## 仓库结构

| 路径 | 作用 |
|---|---|
| `hybrid_pp/plugin.py` | vLLM 插件入口，按环境变量启用各补丁 |
| `hybrid_pp/patches/nemotron_h_pp.py` | 修复 NemotronH 的 PP（加载权重不跳过其他 stage 的层、遍历了所有层、residual 被清空） |
| `hybrid_pp/patches/stage_memory.py` | 每个 stage 打印显存明细；允许没有 attention 层的 stage 启动 |
| `hybrid_pp/profiler.py` | 用 CUDA event 记录每步、每层的 GPU 耗时（需要 `--enforce-eager`） |
| `hybrid_pp/model_info.py` | 不需要 GPU：分析任意切分下各 stage 的层构成和显存估算 |
| `hybrid_pp/correctness.py` | 对比 PP=1 和 PP=N 的 greedy 输出 |
| `hybrid_pp/runner.py` | 批量实验：切分 × 负载，调用 `vllm bench serve` |
| `hybrid_pp/analyze_memory.py` | 从 server.log 算出各 stage 的 KV 预算、实际使用量和浪费量 |
| `hybrid_pp/analyze_profile.py` | 每类层的耗时曲线、各 stage 的耗时和忙碌率，并画图 |
| `scripts/` | 环境安装、环境检查、手动起服务、第一天流程 |
| `docs/findings.md` | 读 vLLM 源码发现的问题（带行号） |
| `docs/plan.md` | 研究计划清单 |

## 常用命令

```bash
# 静态分析（不需要 GPU）
python -m hybrid_pp.model_info nvidia/NVIDIA-Nemotron-Nano-9B-v2 --pp 4
python -m hybrid_pp.model_info nvidia/NVIDIA-Nemotron-Nano-9B-v2 --pp 2 --candidates 4 --step 2

# 手动起服务（任意参数会透传给 vllm serve）
PP=2 PARTITION=24,32 bash scripts/serve.sh

# 切分扫描
python -m hybrid_pp.runner --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 --pp 2 \
  --partitions 24,32 26,30 28,28 30,26 32,24 \
  --workloads 1024:256 8192:256 32768:128 --out results/sweep_pp2
python -m hybrid_pp.analyze_memory results/sweep_pp2/*/server.log

# 逐层 profiling
python -m hybrid_pp.runner --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 --pp 1 --profile \
  --workloads 1024:128 4096:128 16384:128 65536:64 --max-model-len 131072 \
  --concurrency 8 --num-prompts 32 --out results/layers_pp1
python -m hybrid_pp.analyze_profile results/layers_pp1/default/*/profile --out results/layers_pp1/summary
```

## 插件环境变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `HYBRID_PP_FIX_LOAD` | 1 | NemotronH load_weights 修复（Bug 3） |
| `HYBRID_PP_FIX` | 1 | NemotronH forward 修复（Bug 1、2）；设为 `layers_only` 只修 Bug 1 |
| `HYBRID_PP_FIX_NOATTN` | 1 | 没有 attention 层的 stage 不参与 KV block 的 min() |
| `HYBRID_PP_STAGE_MEM_LOG` | 1 | 每个 stage 打印一行 `[hybrid_pp] stage_mem {...}` |
| `HYBRID_PP_PROFILE_DIR` | 未设置 | 设置后开启 profiling，结果写到这个目录 |
| `HYBRID_PP_PROFILE_LAYERS` | 1 | 设为 0 只记录整步耗时，开销更小 |
