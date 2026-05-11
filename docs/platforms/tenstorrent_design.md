# SGLang on Tenstorrent — Adaptation Design Doc

> **⚠️ Superseded for the P1 implementation path by**
> [`docs/superpowers/specs/2026-05-11-sglang-tenstorrent-p1-design.md`](../superpowers/specs/2026-05-11-sglang-tenstorrent-p1-design.md).
>
> Several assumptions in this document were revised after hardware survey:
> - ETH fabric (QSFP-DD 800GbE) exists and is used for inter-card traffic — original doc assumed PCIe-only.
> - Device 1 sits in PCIe Gen3 x1 by motherboard limitation, forcing ETH-only collectives.
> - vLLM-fork validation step removed (already proven working with Qwen3-1.7B).
> - Target tightened to TP=2 Llama-3.1-8B BF16 in-tree, MLX-style integration.
>
> Read this doc only for broader subsystem mapping. Implementation must follow the superseding spec.
>
> **状态**: 调研 (Research, not implementation).
> **目标读者**: 准备投入开发的工程师 / 决策者。
> **日期**: 2026-05-11
> **目标硬件**: 2× Tenstorrent Blackhole p150a (本机 `01:00.0` / `06:00.0`).

本文档评估把 SGLang 适配到 Tenstorrent Blackhole 加速卡所需的工作量、技术路径、关键风险,并给出分阶段实施建议。**它不是一份施工蓝图,而是一份决策文档** — 重点在于"做不做、怎么做、谁来做"。

---

## 0. TL;DR

- **可行,但成本高**: 最小可用的 SGLang+TT 端到端推理(单卡、单一 Llama-7B、eager 模式),预计 **3-5 月** 由 2 名熟悉 SGLang 内部 + ttnn 的工程师完成。生产可用(多卡、性能合理、支持 MLA/MoE/Speculative)再加 **6-9 月**。
- **不要走"伪装成 torch 后端"的路径**: ttnn 没有 `torch.Tensor` 兼容,无法注册为 PyTorch backend(不像 NPU 用 `torch_npu`)。**必须走 MLX 那条路** — 整层替换 model runner / KV cache / worker。
- **建议先行**: 用 Tenstorrent 官方 vLLM fork 在 p150a 上跑通 Llama,**先验证硬件与 ttnn 在 Blackhole 上的稳定性**。如果连 vLLM fork 都跑不通,SGLang 适配项目应推迟。
- **最大风险**: Blackhole 在 ttnn 上的成熟度远低于 Wormhole。许多算子需自行补完,而这不是 SGLang 的事情。

---

## 1. 当前环境盘点

### 1.1 硬件
```
01:00.0 Processing accelerators: Tenstorrent Inc Blackhole
06:00.0 Processing accelerators: Tenstorrent Inc Blackhole
```
- 两张卡通过 PCIe 连接,**没有 NVLink 等价物**;p150a 之间的通信走 PCIe 或主板上的 Ethernet bridge(若有 — 需查 SKU)。
- 每卡 28 GB GDDR6,约能装下 FP16 Llama-13B 或 INT8 Llama-30B 单卡。

### 1.2 软件栈现状
| 组件 | 已装 | 备注 |
|---|---|---|
| KMD (`tt-kmd`) | ✅ 2.8.0 | 驱动 |
| `tt-smi` | ✅ 5.0.1 | 诊断 |
| `tt-flash` | ✅ 3.6.5 | 固件刷写 |
| `tt-umd` | ✅ 0.9.4 | 用户态驱动 |
| `tt-metal` (C++) | ❌ 未装 | 核心运行时,需源码构建,~1-2h |
| `ttnn` (Python) | ❌ 未装 | 神经网络算子库 |
| `tt-torch` / `tt-mlir` | ❌ 未装 | PyTorch 前端编译器 |
| PyTorch | ❌ 未装 | |

**结论**: 还没到能写代码的程度 — 当前只是驱动 + 诊断工具。SGLang 适配开工前,需要先把 tt-metal 完整构建链跑通,并能在 p150a 上独立跑一个 ttnn matmul demo。

---

## 2. Tenstorrent 软件栈:三种集成路径的选择

### 2.1 Path A: 通过 `tt-torch` / PJRT 让 PyTorch 跑在 TT
- **机制**: tt-torch 把 PyTorch 计算图编译到 tt-mlir,最终在 tt-metal 上执行。
- **优点**: SGLang 的模型层(Llama、Qwen 等 `nn.Module`)理论上可以"零改动"运行。
- **缺点**:
  - tt-torch 截至 2026 年初处于 alpha,**Blackhole 支持极不稳定**。
  - 不支持 dynamic shape / KV cache 这类增量写入语义。
  - 性能远低于直接用 ttnn。
- **判断**: **不推荐作为主路径**,但可作为模型 forward 的早期调试工具。

### 2.2 Path B: 直接用 `ttnn` 重写关键层(类似 MLX 后端的做法)
- **机制**: 加载权重 → 用 `ttnn.Tensor` 持有 → 用 `ttnn.matmul / ttnn.softmax / ttnn.experimental.rotary` 等算子重写 attention / MLP / RoPE / RMSNorm。
- **优点**: 性能上限最高;贴近 Tenstorrent 官方做法(他们的 vLLM fork、tt-transformers 都走这条)。
- **缺点**: SGLang 中每个 layer 都要重写一份 TT 版本。
- **判断**: **主路径**。即使后续 tt-torch 成熟,核心 attention 仍建议保留 ttnn 手写版本。

### 2.3 Path C: 不动 SGLang,改用 Tenstorrent 官方 vLLM
- **链接**: `github.com/tenstorrent/vllm` (官方 fork)
- **优点**: 立刻能跑(单卡 Llama-7B),用来验证硬件 + 软件栈。
- **缺点**: 不是 SGLang;失去 RadixAttention、grammar、structured output、disaggregation 等 SGLang 特性。
- **判断**: **作为 Phase 0 验证步骤,不作为最终目标**。

---

## 3. SGLang 适配契约 (Integration Contract)

SGLang 的平台抽象已经为非 CUDA 后端做好了基础设施。详见 [`python/sglang/srt/platforms/interface.py`](../../python/sglang/srt/platforms/interface.py) 和 [`device_mixin.py`](../../python/sglang/srt/platforms/device_mixin.py)。

### 3.1 插件注册
通过 `setuptools` entry point 注册:
```toml
# pyproject.toml of the TT backend package
[project.entry-points."sglang.srt.platforms"]
tenstorrent = "sglang_tenstorrent:activate_tt_platform"
```
`activate_tt_platform()` 检测 p150a 存在则返回 `"sglang_tenstorrent.TTSRTPlatform"`,否则返回 `None`。

### 3.2 必须实现的工厂方法 (`SRTPlatform` 子类)
| 方法 | 用途 | TT 处理建议 |
|---|---|---|
| `get_default_attention_backend()` | 返回 attention backend 名 | `"tenstorrent"`(自注册) |
| `get_graph_runner_cls()` | 返回 decode 期图运行器 | `TTEagerRunner`(类似 `cpu_graph_runner.py`,无图捕获,纯 eager) |
| `get_mha_kv_pool_cls()` | MHA KV pool 类 | **stub**(因为 ttnn 不是 torch tensor,见 §5.2) |
| `get_mla_kv_pool_cls()` | MLA KV pool 类 | stub 或 raise(Phase 1 不做) |
| `get_nsa_kv_pool_cls()` | NSA KV pool(V3.2) | raise(Phase 1 不做) |

### 3.3 关键能力旗标
| 旗标 | 应返回 |
|---|---|
| `support_cuda_graph()` | `False` |
| `support_piecewise_cuda_graph()` | `False` |
| `supports_fp8()` | `False`(Blackhole 数据格式不是标准 IEEE FP8) |
| `is_pin_memory_available()` | `True`(主机端 pinned 仍可用) |

### 3.4 已有的两种集成模式
| 模式 | 代表 | 特征 |
|---|---|---|
| **A: 标准 runner + 子系统替换** | NPU(`hardware_backend/npu`) | model 仍以 `torch.Tensor` 居住,通过 `torch_npu` 把 device 标记为 `"npu"`。仅替换 attention、KV pool 实现、通信器。 |
| **B: 整层替换** | MLX(`hardware_backend/mlx`) | model 不用 PyTorch,KV cache 是 `_DummyKVCache` 占位,worker 是自定义 `MlxTpModelWorker`。 |

**Tenstorrent 必须走 B**。原因:ttnn 没有 PyTorch backend(不像 `torch_npu`、`torch_xla`)。`torch.Tensor` 不能放到 TT 卡上。

---

## 4. 模块映射表 (SGLang → Tenstorrent)

按"改动难度"和"是否必须 Phase 1 完成"排序:

| SGLang 模块 | 难度 | Phase | TT 替换策略 | 关键文件 |
|---|---|---|---|---|
| **Platform 类注册** | 易 | 1 | 实现 `TTSRTPlatform(SRTPlatform)` + entry point | 新建 `sglang_tenstorrent/platform.py` |
| **Worker / TpModelWorker** | 中 | 1 | 仿照 `MlxTpModelWorker`,绕过 PyTorch model forward | `hardware_backend/mlx/tp_worker.py` 参考 |
| **Model loader** | 中 | 1 | 从 HF safetensors 直接加载到 `ttnn.Tensor`(可参考 tt-transformers) | 不复用 `model_loader/loader.py`(它假设 torch) |
| **Attention backend** | **高** | 1 | 新建 `TTAttnBackend(AttentionBackend)`,用 `ttnn.transformer.scaled_dot_product_attention` 或手写 ttnn matmul + softmax;Blackhole 上可能要自己补 paged attention | `layers/attention/base_attn_backend.py`(契约),`torch_native_backend.py`(最简参考) |
| **KV Cache** | **高** | 1 | 不用 `MHATokenToKVPool`(它是 torch tensor 池)。自管 `ttnn.Tensor` 池,暴露 `set_kv_buffer / get_kv_buffer` 接口供 attention 调 | `mem_cache/memory_pool.py` 是契约,但实现要在 ttnn 侧重写 |
| **Radix tree / chunk cache** | 易 | 1 | **可复用** — 它是纯 Python + CPU 索引(token_id 列表 → slot 索引),不碰 device tensor | `mem_cache/radix_cache.py` |
| **Scheduler / batching** | 易 | 1 | **可复用**(可能要 MLX 风格的 scheduler_mixin 微调) | `managers/scheduler.py` 参考 mlx mixin |
| **Sampler** | 易 | 1 | 用 `sampling_backend=pytorch` fallback(把 logits 从 ttnn 拷回 host 后用 torch 采样) | `layers/sampler.py` |
| **RMSNorm / RoPE / Linear** | 中 | 1 | 每个 op 出一个 ttnn 包装版本;模型 forward 用这些包装替代 `nn.Module` | `layers/layernorm.py`, `layers/rotary_embedding/`, `layers/linear.py` 仅作契约参考 |
| **Distributed (TP)** | **高** | 2 | ttnn 有 `mesh_device` 概念用于多卡 fabric,但不暴露 `torch.distributed.Backend`。需要 host 端协同(类似 MLX 的 `ProcessGroup` over Gloo + ttnn fabric 处理设备侧 all-reduce) | `distributed/parallel_state.py`, `distributed/device_communicators/` |
| **Graph runner / CUDA graph** | 中 | 2 | 仿 `cpu_graph_runner.py` 写 `TTEagerRunner`。ttnn 自带 program cache,本身有图缓存效果 | `model_executor/cpu_graph_runner.py` 模板 |
| **Speculative decoding** | 高 | 3 | 取决于 attention backend 是否支持非对齐 KV 索引;EAGLE worker 需要单独 TT 适配 | `speculative/eagle_*.py` |
| **MoE / EP** | 高 | 3 | Blackhole 上 ttnn 的 grouped matmul 成熟度未知,需先 benchmark | `layers/moe/` |
| **MLA (DeepSeek)** | 高 | 3 | MLA 的 KV 形状不同,要额外的 pool 子类 | `mem_cache/memory_pool.py` MLA 部分 |
| **Quantization (W8A8/W4A16)** | 中 | 2 | Blackhole 支持 BFP8 / BFP4(自有低精度格式),不是 GPTQ/AWQ。要单独 quantizer | `layers/quantization/` |

### 4.1 不需要改的(SGLang 核心已抽象好)
- Request 调度、HTTP/gRPC server、tokenizer、grammar、structured output、JSON mode、function calling、prefix cache 索引层、disaggregation 的控制面、benchmark 工具。

### 4.2 必须改但容易遗漏的 CUDA leakage 点
即使在已抽象的边界外,SGLang 核心仍有 ~15 处硬编码 `torch.cuda.*` 调用,需要在 `apply_server_args_defaults()` 中关掉对应特性,或加 `current_platform.is_cuda()` 守卫:
- `model_runner.py:1787,1807,1269` — `torch.cuda.empty_cache/synchronize`
- `managers/schedule_batch.py:342` — `torch.cuda.current_device()`
- `server_args.py:1844,2957` — `torch.cuda.get_device_capability()`(SM 检测)
- `utils/offloader.py`, `disaggregation/common/staging_buffer.py` — CUDA stream / IPC (建议直接 disable offloader + disaggregation 在 TT 上)
- `utils/profile_utils.py`, `utils/nvtx_pytorch_hooks.py` — profile / nvtx
- `utils/multi_stream_utils.py` — overlap streams
- `managers/scheduler_dp_attn_mixin.py` — NCCL overlap tuning

修复策略:大部分加 `if is_cuda():` 即可,或填充 `DeviceMixin` 的 [Planned] 方法。

---

## 5. 关键现实约束 (必须正视)

### 5.1 ttnn 在 Blackhole 上的成熟度
- Tenstorrent 自己的 demo 矩阵中,Wormhole 上 Llama/Mistral/Falcon 都有完整支持,Blackhole 上覆盖少得多。
- ttnn 算子在 Blackhole 上偶有 "not implemented on Blackhole" 或性能严重退化。
- **必须先做**: clone tt-metal,跑 `tt-metal/models/demos/llama3/` 的 Blackhole 版本,看哪些 op 缺失。这是 Phase 0 的硬门槛。

### 5.2 ttnn 不是 torch 设备
- `torch.zeros(..., device="tenstorrent")` 不存在。
- 不能像 NPU 那样 `model.to("npu")` 把整个 `nn.Module` 搬到卡上 — 必须**逐层用 ttnn API 重写 forward**。
- 这是 MLX 后端选 "Pattern B (整层替换)" 的根本原因,Tenstorrent 同理。

### 5.3 两卡之间的通信
- p150a 是 PCIe 卡,两张卡之间**没有 NVLink/IB 等价物**。
- ttnn 的 `mesh_device` 概念主要是为 n300 / Galaxy 这种有 Ethernet fabric 的板子设计的。
- 两张 p150a 跑 TP=2 时,all-reduce 走 PCIe → host → PCIe,延迟极高。
- **建议**: Phase 1 只做单卡,TP=2 延后到 Phase 2 并先 benchmark 通信开销。

### 5.4 没有 CUDA graph 等价物
- ttnn 有 program cache (host 端缓存编译后的程序),提供类似的"避免重复编译"效果,但**不是图捕获**。
- SGLang 的 decode 高吞吐严重依赖 CUDA graph 消除 host overhead。
- 在 TT 上,首次实现会显著慢于理论值。后续优化方向是利用 ttnn `tracing` 特性(beta)。

### 5.5 数据类型
- Blackhole 原生支持: BF16、BFP8(blockwise FP8,不兼容 IEEE-FP8)、BFP4。
- **不支持**: 标准 FP8(E4M3/E5M2)、INT4 GPTQ/AWQ 格式。
- 推理量化必须重新走 BFP8 路径,**不能直接用 SGLang 的 FP8/AWQ quantizer**。

---

## 6. 分阶段实施计划

### Phase 0 — 验证可行性 (1-2 周, 1 人)
**Exit criteria**: 在 p150a 上用 Tenstorrent vLLM fork 跑通 Llama-3.1-8B,QPS > 0,生成结果可读。
- 装 tt-metal、ttnn、tt-vllm。
- 跑 Blackhole 上的 Llama demo(`tt-metal/models/demos/llama3/`)。
- 如果 Blackhole 上连 demo 都跑不通,**项目应暂停**,先等 ttnn for Blackhole 成熟。

### Phase 1 — 最小可用 (3-5 月, 2 人)
**Exit criteria**: SGLang 单卡 eager 模式跑 Llama-7B,通过 SGLang 自带的 `bench_serving.py` 短 prompt 测试。
- 实现 `TTSRTPlatform` 插件 + entry point。
- 实现 `TTAttnBackend`(eager, prefill + decode 分开)。
- 实现 `TTKVPool`(ttnn tensor 池,vLLM 那种 paged 布局)。
- 实现 `TTEagerRunner`(无图捕获)。
- 实现 model loader(safetensors → ttnn weights),仅 Llama 架构。
- 替换 RMSNorm / RoPE / Attention / MLP linear 为 ttnn 版本。
- 守卫 §4.2 中所有 CUDA leakage 点。
- Sampler 走 pytorch fallback(host 侧采样)。

### Phase 2 — 性能与多卡 (3-4 月, 2 人)
- 双卡 TP=2(评估是否值得)。
- ttnn tracing / program cache 调优,消除 host overhead。
- BFP8 量化路径(Llama-13B / 70B)。
- Continuous batching 验证。
- Prefix cache (RadixAttention) 端到端打通。

### Phase 3 — 高级特性 (按需)
- MLA / DeepSeek 系列(KV pool 子类)。
- MoE / EP(Blackhole 上 grouped matmul 可行性先于实现)。
- Speculative decoding(EAGLE)。
- Structured output 已经在框架层,无需额外工作。

---

## 7. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| Blackhole ttnn 不支持关键算子 | 高 | 阻塞 | Phase 0 提前排雷;无法解决就回退到 Wormhole 平台 |
| 性能远低于 NVIDIA 同价位卡 | 中-高 | 削弱项目价值 | Phase 1 后给客观对比;接受 TT 作为"低 TCO 而非高峰值"定位 |
| 双卡通信开销 > 单卡内存收益 | 中 | TP=2 没意义 | Phase 1 收尾时实测;若是则保持单卡部署 |
| tt-metal API 不稳定,频繁 breaking change | 高 | 维护成本 | 锁定 tt-metal release branch,不跟 main |
| SGLang upstream 的 CUDA leakage 加重 | 中 | 持续 rebase 成本 | 用插件/entry point 而非 in-tree fork,降低耦合 |
| 没人懂 ttnn | 高 | 进度无法保证 | 招/培训至少 1 人深入 ttnn(读 tt-metal 源码、参与 Discord) |

---

## 8. 待决策的开放问题

需要在动工前由项目所有者决定:

1. **In-tree vs Out-of-tree?** 推荐 OOT(独立 pip 包 `sglang-tenstorrent`,通过 entry point 注册)。Mainline merge 等到稳定后再说,避免污染上游 CI。
2. **支持 PyPI 发布还是仅内部?** 影响版本锁定策略。
3. **目标模型范围?** Llama-only 是最小集;若必须支持 Qwen2/DeepSeek/Mixtral 等,Phase 1 周期翻倍。
4. **是否需要 disaggregation (PD 分离)?** 推荐 **No**(它依赖 CUDA IPC / NIXL)。
5. **谁负责 ttnn 侧?** 需要至少 1 人能独立写 ttnn 算子(不只是调用)。
6. **测试覆盖目标?** 推荐复用 SGLang `test/srt/` 的服务级测试,而不是 op 级单测(那个量太大)。

---

## 9. 推荐下一步

按顺序:
1. **本周**: 完成 Phase 0 — 装 tt-metal,跑通 Tenstorrent vLLM fork 上的 Llama-3.1-8B。**这是 go/no-go 关口**。
2. **下一步**: 如果 Phase 0 通过,做一个 1-week spike:把 `TTSRTPlatform` skeleton + 一个 hardcoded 的 `TTAttnBackend.forward_decode()` 实现出来,跑通 batch=1, seq_len=1 的 decode step。验证集成契约真的可用。
3. **再下一步**: 根据 spike 的实际工作量,重新校准 Phase 1 估算。本文档的 3-5 月是**乐观估计**,基于 ttnn Blackhole 支持没大坑的假设。

---

## 附录 A — 文件路径速查

| 内容 | 路径 |
|---|---|
| Platform 抽象 | `python/sglang/srt/platforms/interface.py` |
| Device mixin | `python/sglang/srt/platforms/device_mixin.py` |
| MLX 后端(整层替换参考) | `python/sglang/srt/hardware_backend/mlx/` |
| NPU 后端(标准 runner 参考) | `python/sglang/srt/hardware_backend/npu/` |
| Attention 契约 | `python/sglang/srt/layers/attention/base_attn_backend.py` |
| Attention 注册 | `python/sglang/srt/layers/attention/attention_registry.py` |
| 最简 attention 参考实现 | `python/sglang/srt/layers/attention/torch_native_backend.py` |
| KV pool 契约 | `python/sglang/srt/mem_cache/memory_pool.py` |
| NPU KV pool 改写参考 | `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` |
| Eager graph runner 参考 | `python/sglang/srt/model_executor/cpu_graph_runner.py` |
| Distributed 入口 | `python/sglang/srt/distributed/parallel_state.py` |
| Sampler | `python/sglang/srt/layers/sampler.py` |

## 附录 B — Tenstorrent 资源

| 资源 | 用途 |
|---|---|
| `github.com/tenstorrent/tt-metal` | 底层 C++ + ttnn |
| `github.com/tenstorrent/vllm` | 参考实现,Llama on TT |
| `github.com/tenstorrent/tt-transformers` | 官方 transformer 模型库 |
| `github.com/tenstorrent/tt-torch` | PyTorch 前端(alpha) |
| Tenstorrent Discord | 算子问题、Blackhole 支持状态 |
