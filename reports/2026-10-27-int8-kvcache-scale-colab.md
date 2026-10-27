---
feature_ids: []
topics: [int8-kv-cache, paged-attention, quantization]
doc_kind: experiment-report
created: 2026-10-27
---

# Colab 实验：不同 KV INT8 动态量化粒度的 decode 对照

## 目标与结论

在 vLLM 的真实模型运行中捕获 attention 输入，以 native FlashAttention 输出为基准，比较两种容易混淆的 KV scale 定义，并检查动态量化 Q 的影响。实验使用 Qwen/Qwen2.5-1.5B-Instruct、BF16 和 Google Colab A100 40GB。测试样本较小，结果用于定位误差，不代表完整基准精度或性能。

本次真实 decode 的 84 次层调用中，KV 使用 `per_channel`、Q 保持浮点、先反量化再做 attention 时，最大相对 L2 误差为 4.876%；`per_token_head` 同条件下最大为 32.423%。两者的均值分别为 1.485% 和 3.077%。这说明在本次捕获的数据上，单看平均值不足以发现最差层问题，也不能据此断言某一种粒度在所有模型和输入上更优。

## 两种 scale 的定义

设某个请求有效 KV 张量为 `[T, H_kv, D]`，K 和 V 分别计算 scale，使用对称 absmax INT8：`s = max(abs(x)) / 127`，`q = clamp(round(x / s), -127, 127)`，`x_hat = q * s`。实际实现还需处理全零向量，避免 scale 为零。

| 粒度 | reduce 维度 | scale 形状 | 含义与使用 |
|---|---|---|---|
| `per_channel` | T | `[H_kv, D]` | 同一 head 的同一特征通道跨 token 共享 scale；若用实际整数内积，scale 与 D 有关，不能简单从 Q·K 的 D 归约中整体提出。 |
| `per_token_head` | D | `[T, H_kv, 1]` | 每个 token、每个 KV head 的 D 维向量各一个 scale；Q 和 K 的内积可在 D 归约后再乘该 token/head 的 K scale。 |

此处的 `per_channel` 使用当前捕获的有效 token 计算 scale，是实验中的动态参考实现。实际在线 decode 若采用跨 token 的动态 scale，新增 token 可能改变旧 scale，必须设计重标定、重编码或分组策略；不能直接假定每步免费更新。`per_token_head` 的每个 token scale 可以在写入时确定，但需要额外存储 scale。两者不能只比较量化后 KV 主体的字节数，还要考虑 scale 的 dtype、写入和加载成本。

Q 的 `per_tensor` 为一次调用的整个 Q 张量计算一个动态 scale；它不改变上述 KV scale 的定义。本实验分别保留 Q 浮点和量化 Q，K/V 始终独立计算 scale。

## 环境、数据与过程

- 模型：`Qwen/Qwen2.5-1.5B-Instruct`，BF16；vLLM 0.29.0，FlashAttention 2。调用路径为 `FlashAttentionImpl.forward` → `flash_attn_varlen_func` → `torch.ops._vllm_fa2_C.varlen_fwd`。
- 固定 prompt：`What is 17 * 23? Reply with the integer only.`；捕获配置 `max_model_len=256`、`max_tokens=4`。捕获到 8 个 step，每步 28 层：前 4 步为 warmup，第 5 步是 prefill（Q 长度 16），最后 3 步是单请求 decode（Q 长度 1，KV 长度分别 17、18、19）。后续 decode 数值表只统计最后 3 步、共 84 层调用。
- FP replay 覆盖全部 224 次捕获调用，PyTorch 参考与 native 输出的最大相对 L2 为 0.002802、均值 0.001400，满足 5% 的实验门槛。这个参考实现本身并非位级一致，且该统计包括 warmup 和 prefill。
- 先用捕获的有效 KV 检查重建误差；再比较 decode attention 输出；最后通过 vLLM 整网路径验证执行与小样本任务行为。实际 GPU 执行验证了相关测试 24 项通过，以及窗口分发修复后的 4 项测试通过。
- 数值实验代码来源提交 `0c8ee09`；整网运行的窗口分发修复 `a28c78e`；MMLU 脚本 `5b4de68`。可复现 Colab：[INT8 KV per-token-head 对照实验](https://colab.research.google.com/drive/1zzCG4KWUckp9VUiR-VQ9TfiElHvTysk3)。

## Stage 2：KV 重建误差

下表使用最后捕获步骤的有效 KV 汇总，显示相对 L2；越小表示 KV 本身重建更接近浮点，不直接等同于最终 attention 输出或模型精度。

| KV 粒度 | 相对 L2 |
|---|---:|
| per_tensor | 0.01080 |
| per_head | 0.00964 |
| per_token | 0.01057 |
| per_channel | 0.00232 |
| per_token_head | 0.00940 |

## Stage 4：真实 decode 输出数值

以 native FlashAttention 输出为基准，统计 84 次层调用的相对 L2。`KV 反量化` 保留浮点 Q；`模拟整数计算` 将 Q 也做 per-tensor 动态量化，使用浮点运算模拟整数值的计算，softmax 保持 FP32，默认不量化概率。后者不是实际 INT8 GEMM kernel，也不能用来推断加速。

| KV 粒度 | KV 反量化：最大 | KV 反量化：均值 | Q 量化 + 模拟整数：最大 | Q 量化 + 模拟整数：均值 |
|---|---:|---:|---:|---:|
| per_head | 0.231377 | 0.040203 | 0.231704 | 0.048392 |
| per_channel | 0.048762 | 0.014849 | 0.086295 | 0.029836 |
| per_token_head | 0.324235 | 0.030772 | 0.274808 | 0.040548 |

相对于 5% 的输出误差实验门槛，`per_channel` 的 KV 反量化路径在本次 84 次调用中最大值低于门槛，`per_token_head` 超过门槛。加入 Q 量化后，即使 per-channel 最大值也超过 5%。某组加入 Q 量化后最大误差下降并不意味着 Q 量化必然改善精度：误差可能在单个样本中相互抵消。

### 逐层分布和超门槛次数

复核下载的 decode-only JSON，按 84 个层调用统计，5% 是本实验的相对 L2 门槛（不是行业通用标准）：

| KV 粒度及 Q 路径 | 中位数 | 超过 5% 的调用数 | 观察 |
|---|---:|---:|---|
| per_channel，Q 浮点 | 1.339% | 0/84 | 本样本没有调用超门槛。 |
| per_token_head，Q 浮点 | 1.932% | 6/84 | 6 次全部集中在 3 个 decode step 的第 0、1 层。 |
| per_channel，Q per-tensor | 2.684% | 8/84 | 超门槛不仅出现在第 0 层。 |
| per_token_head，Q per-tensor | 3.047% | 13/84 | Q 量化后，超门槛调用扩展到更多层。 |

这里的 84 次来自一个 prompt 的 3 个连续 decode step 和 28 层，彼此并非独立样本，因此不能将上述比例外推成总体失效率。该分布表明，在本案例中值得分别研究早期层的 K 量化误差和 Q per-tensor 的影响。

### 最差层诊断

所有上述最大误差来自第一个 decode 步骤、KV 长度 17 的第 0 层。以该层 PyTorch 浮点参考为比较基准，分别只量化 K、只量化 V、同时量化 K/V（Q 保持浮点）：

| KV 粒度 | 仅 K 相对 L2 | 仅 V 相对 L2 | K/V 同时相对 L2 |
|---|---:|---:|---:|
| per_channel | 0.048466 | 0.005040 | 0.048752 |
| per_token_head | 0.323089 | 0.015446 | 0.324195 |

该层浮点参考相对 native 的 L2 为 0.000452。per-token-head 的仅 K 量化还观察到 logit 最大绝对变化 38.797、softmax 最大绝对变化 0.9685。因此，在这个具体层和输入上，输出误差主要与 K 量化改变 attention 权重有关；这不足以证明全模型的普遍原因，也不能直接断言由某个异常值导致。下一步应检查对应 Q/K 的值、scale 分布以及更多层和序列。

## 整网路径与小样本行为

在修复前，整网量化路径没有拦截任何 attention 调用。原因是 vLLM 将无窗口限制表示为 `[-1, -1]`，旧代码只识别 `(-1, -1)`，导致错误回退到 native。针对列表和元组新增测试，先复现失败再修复；Colab 上修复后相关测试 4 项全部通过。修复后自造的 4 题 smoke 测试中，各量化组记录到 588 次量化 PyTorch 调用、0 次 fallback；该计数包括 warmup，不能把 588 次全称为 decode。

smoke 题只是算术、物理、编程、地理各 1 题，虽然输出表使用了 AIME、GPQA、HumanEval、MMLU 的名称，但它们不是这些正式数据集，不能当作这些 benchmark 的分数。其用途是验证整网执行及输出行为。

### 真实 MMLU 小样本

使用 Hugging Face `cais/mmlu` 的 `all/test`，固定随机种子 27，不放回抽取 10 条：行号 `1071, 1211, 3217, 4168, 4541, 4685, 7862, 10624, 11491, 13504`；当次数据 fingerprint 为 `436299c1c09696bb`。两种粒度使用相同题目，`max_model_len=1024`、`max_new_tokens=8`，量化组 Q 为 per-tensor、KV 使用 absmax。

| KV 粒度 | 原生 FP | KV 量化后反量化 | Q/KV 量化 + 模拟整数 |
|---|---:|---:|---:|
| per_token_head | 5/10 | 5/10 | 5/10 |
| per_channel | 5/10 | 5/10 | 5/10 |

四个量化组各记录 224 次 PyTorch 量化调用、0 次 fallback；计数包括模型预热。进一步按 prompt 对齐逐题 completion：per-token-head 两条量化路径均与 native 10/10 完全相同；per-channel 两条量化路径均在行号 1071 由 native 的 B 变为 D，其他 9 题相同。该题两种答案都错误，因此尽管输出改变，三条路径仍都为 5/10。这正说明总分相同并不代表逐题输出完全相同，更不能证明在完整 MMLU 或其他任务上不存在精度损失。仅 10 条题目、输出长度上限和答案提取规则都会限制结论；应扩大样本、核对逐题答案及截断、增加多个 prompt 与模型再做判断。

## 结果文件与复现

上述结果实际在 [Colab 运行会话](https://colab.research.google.com/drive/1C3YctDNXDAU4mussCqVlwIqNv0Y1ZlKj) 中通过终端执行；该会话所在 Notebook 出现自动保存冲突。另存的 [可复现 Notebook](https://colab.research.google.com/drive/1zzCG4KWUckp9VUiR-VQ9TfiElHvTysk3) 已保存新增步骤，但其单元格尚未整本运行，不应把它当作带有本次输出的运行记录。

当次运行目录为 `/content/int8-kvcache-experiments/20261027T094313Z-49e274/`；其关键文件如下，目录随 Colab 运行时结束可能被清理。结果包包含 JSON、日志和 decode-only 捕获，已确认下载并校验完整性，工作区副本为 `reports/artifacts/2026-10-27/int8-kvcache-results.zip`；SHA-256 为 `8f7dcfb076b793f2b39ee35daf23649817271d5355b7f4a774680cf3d5795d96`。该 ZIP 未提交至 Git。完整的原始捕获 `qwen-real-kv.pt` 不在结果包内。

- `vllm_kv_cache_calibration-20261027T095446761292Z.json`：KV 重建统计。
- `stage3_replay-20261027T095514723081Z.json`：224 层次调用的 FP 重放。
- `stage4_capture_numerics-20261027T095905166019Z.json`：84 次真实 decode 层调用的数值对比；`qwen-decode-only.pt` 为对应捕获。
- `stage5_vllm_dynamic_int8_scorecard-20261027T101415136199Z.json` 和 `stage5_vllm_dynamic_int8_scorecard-20261027T101747436144Z.json`：两组真实 MMLU 逐题结果及抽样信息。
- `logs/unit_tests.log`、`logs/window_dispatch_red.log`、`logs/window_dispatch_green.log` 和 `logs/worst_decode_layer_diagnostic.log`：测试和最差层诊断证据。

可复现 Notebook 将代码固定到 `5b4de68` 并提供捕获、decode 筛选、精度和小样本对比步骤；实际重新运行受 Colab GPU、模型下载、软件包及数据源版本等条件影响，结果不保证完全一致。

## 后续工作

1. 针对最差层统计 Q/K 极值、每种 scale、量化前后 logits 和 softmax，扩展到多条请求与更长上下文，确认误差是否可重复。
2. 分别比较 Q 浮点、Q per-tensor、仅 K 量化、仅 V 量化，记录逐层误差分位数，而不只看均值和最大值。
3. 扩大真实数据集样本并保留逐题答案、输出截断信息，必要时多 seed 和多个模型；把整网质量与单层误差分开分析。
4. 设计实际的 paged INT8 KV 存储及 scale 元数据，再实现 fused kernel，并单独测量显存、吞吐和延迟。当前模拟路径不支持性能收益结论。
