# vLLM v0.10.1.1 源码发现（与 PP + 混合模型相关）

读的是 tag `v0.10.1.1`。行号以该 tag 为准。标 ⏳ 的是读代码得出、还没在 GPU 上验证的结论，`scripts/day1.sh` 会验证。

## 1. NemotronH 的 PP 有四个 bug，层层遮挡

四个 bug 依次触发，前一个会挡住后一个，所以 `day1.sh` 第 3a–3c 步每次多修一个，逐个在真机上证实（2×RTX 4090，2026-10-28）：

| 步骤 | 打开的修复 | 现象 | 证实 |
|---|---|---|---|
| 3a | 无（原版） | 启动加载权重时 `KeyError: 'layers.49.mixer.down_proj.weight'` | Bug 3 ✅ |
| 3b | Bug 3 | 启动 profile run 时 `ValueError: too many values to unpack (expected 2)` | Bug 1 ✅ |
| （旧 3c/4） | Bug 3 + 1 + 2 | 非首 stage 启动 profile run 时 `AssertionError`（`intermediate_tensors` 是 None） | Bug 4 ✅ |
| 3c | Bug 3 + 1 + 4 | 能启动，但输出从开头就和 PP=1 不一样 | Bug 2 ⏳ |
| 4 | 全部 | 和 PP=1 **10/10 完全一致** | 修复正确 ✅ |

### Bug 4：中间结果工厂函数名字写错，静默返回 None ✅

`NemotronHForCausalLM.__init__` 里写的是 `self.make_empty_intmd_tensors = ...`，而 V0 的 `model_runner.py`（1321 行 profile run、1422 行 CUDA graph capture）调用的是 `self.model.make_empty_intermediate_tensors(...)`。按理说名字不对应该报 `AttributeError`，但 NemotronH 继承了 `SupportsPP`，`interfaces.py` 310 行有一个同名的 Protocol 空方法（函数体只有 `...`），调用它返回 `None`。所以不会报错，非首 stage 拿到 `intermediate_tensors=None`，直到 forward 里的 `assert` 才挂。修复：在 `NemotronHForCausalLM` 上补一个正确名字的方法，转调 `make_empty_intmd_tensors`。

### Bug 3：load_weights 不跳过其他 stage 的层 ✅

文件：`nemotron_h.py`，`NemotronHModel.load_weights`（约 428–470 行）。每个 stage 都会收到完整的 checkpoint，但这个函数从不调用 `is_pp_missing_parameter`，直接 `params_dict[name]`。stage 0 读到第 28 层以后的权重时就会 `KeyError`。其他模型都会先 `if is_pp_missing_parameter(name, self): continue`。

### 顺带发现：embedding 和 lm_head 每个 stage 都有一份

`NemotronHModel.__init__` 无条件创建 `embed_tokens`，`NemotronHForCausalLM.__init__` 无条件创建 `lm_head`，都没有按 rank 换成 `PPMissingLayer`。Nano 9B v2 的词表是 131072 × 4480，每份约 1.1GB，所以**每个 stage 都多占约 1.1GB 用不到的权重**（非首 stage 用不到 embedding，非末 stage 用不到 lm_head）。这和层间异构无关，但会挤占 KV 预算。论文里如果拿 vLLM 做 baseline，要注意这一点，或者先修掉它再比较。`model_info` 的权重估算已经按这种实际行为计算。

### Bug 1 和 Bug 2：forward

文件：`vllm/model_executor/models/nemotron_h.py`，`NemotronHModel.forward`（约 369–426 行）

```python
residual = None                      # 400 行：无条件覆盖
for i in range(len(self.layers)):    # 402 行：遍历全部层
    ...
    hidden_states, residual = layer(positions=..., hidden_states=..., ...)
```

- **Bug 1：遍历了所有层。** `make_layers`（`models/utils.py` 626 行）给不属于本 stage 的层放的是 `PPMissingLayer`，它的 forward 返回第一个 kwarg，也就是 `positions`。`positions` 是长度等于 token 数的一维 tensor，`hidden_states, residual = positions` 会报 `ValueError: too many values to unpack (expected 2)`。启动时的 profile run 就会用假数据跑一次 forward，所以服务在启动阶段就会挂掉。别的模型都写成 `self.layers[self.start_layer:self.end_layer]`。
- **Bug 2：residual 被清空。** 非首 stage 读到上一个 stage 传来的 `residual` 后，又被第 400 行置成 None，残差流就断了，结果是错的。
- v0.10.2 和 v0.11.0 中仍有同样的代码。
- 修复：`hybrid_pp/patches/nemotron_h_pp.py`（通过 vLLM 插件打补丁，不改 vLLM 源码）。
- Bug 1 会挡住 Bug 2，所以第 3c 步用 `HYBRID_PP_FIX=layers_only` 只修 Bug 1，单独证实 Bug 2。
- 结论：**vLLM 虽然声明 NemotronH 支持 `SupportsPP`，但 PP>1 实际上跑不通。** 可以在论文里写成"现有框架对混合模型的 PP 支持不成熟"的一个证据。

## 2. KV block 数在各 stage 之间取最小值

- `vllm/worker/worker.py` 的 `determine_num_available_blocks`（约 230–299 行）：每个 worker 各自做显存 profiling，用 `available_kv_bytes // cache_block_size` 算出 block 数。
- `cache_block_size` 只统计**本 stage 的 attention 层**（`worker/cache_engine.py` 的 `get_cache_block_size` → `get_num_layers_by_block_type(attention)`）。
- `vllm/executor/executor_base.py` 第 106 行：`a = min([r[0] for r in results])`。所有 stage 都只能用最小的那个 block 数。
- **后果**：attention 层少的 stage，每个 block 很小，能放很多 block，但只能用其他 stage 的最小值，**显存被浪费**。这正是 motivation 里说的"显存利用率低"，可以直接测量：用 `hybrid_pp.analyze_memory` 看每个 stage 的 `unused`。

## 3. 没有 attention 层的 stage 会导致启动失败 ⏳

- 如果某个 stage 一个 attention 层都没有：`cache_block_size == 0` → 该 worker 返回 0 个 block → `min()` = 0 → `raise_if_cache_size_invalid` 报错 "No available memory for the cache blocks"（`worker.py` 576 行）。
- Nano 9B v2 的 attention 层位于第 14/21/30/39 层，**PP=4 均匀切分（14/14/14/14）时，stage 0 和 stage 3 都没有 attention 层**，所以启动会失败。
- Megatron-LM 也遇到过同样的问题（PR #8366）。
- 修复：`hybrid_pp/patches/stage_memory.py` 让这种 stage 不参与 `min()`。

## 4. V0 的其他相关细节

- V0 的 PP 只能通过 `AsyncLLMEngine`（也就是 `vllm serve`）使用；离线的 `LLM` 类在 PP>1 时会直接报错（`engine/llm_engine.py` 1140 行）。所以所有实验都要走 server 加 HTTP 请求。
- V0 PP 有 `pp_size` 个 virtual engine，每个都有自己的 scheduler；单个序列能用的最大 token 数是 `block_size * (num_gpu_blocks // pp_size)`（`worker.py` 580 行）。
- 在 v0.10.1.1 中，Mamba 模型默认会回退到 V0（`arg_utils.py` 约 1428 行，"V1 mamba models are unoptimized"）。我们显式设置 `VLLM_USE_V1=0`。
- V0 的 Mamba state 由 `MambaCacheManager` 管理，在第一次 forward 时（也就是显存 profiling 时）按 `max_num_seqs` 分配，大小是**本 stage 的 Mamba 层数 × max_num_seqs × 每层 state**。所以它算在 profiling 的 "activation peak" 里，会挤占 KV 预算。
- NemotronH 不支持 prefix caching（`nemotron_h.py` 541 行有 assert）。

## 5. 各模型的层分布

| 模型 | 层数 | Mamba | Attention | MLP | 上下文 |
|---|---|---|---|---|---|
| Nemotron Nano 9B v2 | 56 | 27 | 4（第 14/21/30/39 层） | 25 | 128K |
| Nemotron Nano 12B v2 | 62 | 28 | 6 | 28 | 128K |
| Nemotron-H 8B Base | 52 | 24 | 4 | 24 | 8K |

运行 `python -m hybrid_pp.model_info <model> --pp N` 可以看到任意切分下每个 stage 的层分布和显存估算。
