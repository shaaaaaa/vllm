# DSA KV-Cache / Indexer Bundle Pool 跨仓需求文档

## 1. 文档信息

- 主要贡献者：sth4nthL、BrokenDuskL
- 涉及仓库：vLLM、vLLM Ascend、LMCache、LMCache Ascend
- 状态：首版能力已实现，进入正确性、性能和硬件回归阶段
- 适用模型：以 GLM-5.1 / DSA 为首个落地模型，接口不得绑定单一模型名称

本文定义 DSA KV cache、Lightning Indexer cache、共享 bundle pool、P 节点
layerwise prefill、LMCache 持久化、NPU 异步传输和 decode offload 的统一需求。
实现细节见：

- `docs/design/prefill_layer_block_pool.md`
- `vllm_ascend/distributed/kv_transfer/sparse_offload/DESIGN.md`
- `vllm_ascend/distributed/kv_transfer/sparse_offload/INDEXER.md`
- `vllm_ascend/distributed/kv_transfer/sparse_offload/INTEGRATION.md`
- `LMCache/docs/design/decode_offload_rebuild_notes.md`

## 2. 背景与问题

DSA 将注意力拆成两个阶段：

1. Lightning Indexer 使用较小的 index key 对全部历史 token 打分并选择 Top-K。
2. Sparse Flash Attention 只读取 Top-K token 对应的 MLA latent KV。

因此系统中同时存在两类生命周期、形状和容量不同的 KV cache：

- Indexer cache：体积较小，但需要覆盖完整上下文，供全历史打分。
- Latent KV cache：体积较大，prefill 时生成，decode 时只需按 Top-K 稀疏读取。

如果两组 cache 独立按最坏情况预留，会产生显存碎片和不可复用空间；如果 P 节点
保留所有层的完整 prefill KV，则超长上下文很快耗尽 NPU 显存。系统需要共享 bundle
pool 和 layerwise transfer，在保持 DSA 精度与原有 D 节点行为的前提下提高可支持的
上下文长度和 P 节点吞吐。

## 3. 目标

### 3.1 必须实现

- KV cache 与 indexer cache 使用统一 bundle 抽象进行容量核算和原子分配。
- P 节点仅保留当前流水线所需的两个 layer bank，其余层由 LMCache 持久化。
- 当前层计算完成后，save 当前层和 load 下一层可在独立 NPU stream 上并发提交。
- load/save 应尽可能与当前层 HCOM all-reduce 重叠，不得在窗口前引入 host 同步。
- D 节点和非 P 节点保持原有 full-resident KV cache、调度和 decode 路径。
- 标准 PIECEWISE graph 必须可用；不满足回放语义的 graph 模式必须启动时报错。
- cache miss、取消、抢占、异常和进程退出不得造成 bank 覆盖、悬空地址或静默错误。

### 3.2 非目标

- 首版不支持 pipeline parallel，P 节点要求 `pipeline_parallel_size == 1`。
- 首版不支持 P 节点 prefix caching。
- 首版不通过请求参数区分模式，节点身份由部署配置决定。
- 首版不要求同一个 P 实例在 prefill 完成后切换成 full-resident decode。
- Decode offload 是独立能力，不得改变原有 decode attention 的数据来源和精度。

## 4. KV-Cache / Indexer Bundle Pool 需求

### BP-1：统一 bundle 几何

- bundle 必须表达 latent block 与 indexer block 的等价容量关系。
- GLM-5.1 当前关系为 `1 bundle = 2 latent blocks = 9 indexer blocks`。
- 几何关系必须由 cache spec 计算并在启动时校验，不能依靠层名或固定地址猜测。
- 不满足整除、对齐或 dtype/element-size 契约时必须 fail fast。

### BP-2：原子分配和释放

- 同一请求所需的 latent/indexer bundle 必须以一致的 bundle ID 分配。
- 任一 group 分配失败时不得留下另一 group 的部分分配。
- 请求完成、取消、抢占和异常退出必须只释放一次。
- block 0 保持保留语义，不得进入普通可分配容量。

### BP-3：类型化身份

- `FULL_PARENT` 与 `PREFILL_CHILD` 必须是显式类型，不能通过整数范围推断。
- P 节点所有请求使用 `PREFILL_CHILD`，默认节点使用 `FULL_PARENT`。
- scheduler、block table、worker 和 allocator 之间必须保留类型信息。
- 为未来 mixed allocation 保留协议边界，但首版不启用同实例混合分配。

### BP-4：共享 slab

- worker 每种物理 view 只允许一个全局连续 slab。
- latent/indexer view 必须从同一份已核算的物理存储派生，不能重复收费。
- TP 各 rank 必须对 slab 大小、bundle 数和 bank 数达成一致。
- 最大可用显存必须按两个 layer bank 计费，而不是按完整层数计费。

### BP-5：容量与公平性

- capacity 计算必须包含对齐、保留块、两个 bank 和 graph workspace。
- `max_model_len`、`max_num_batched_tokens` 和 `max_num_seqs` 的组合必须在启动时验证。
- 多请求并发不得让单个请求越过自身 bundle lease 访问其他请求数据。
- 释放后的 bundle 可以复用，但复用前必须满足 NPU event 和 host source 生命周期约束。

## 5. Layerwise Prefill 需求

### LP-1：节点身份

- 使用 `VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=true|false` 标识 P 节点。
- 默认值为 `false`，只接受严格布尔值。
- 不得使用 `kv_role`、请求字段或 DP rank 推断 P 节点身份。
- P 节点必须验证 LMCache connector 支持 layerwise transfer window。

### LP-2：双 bank 时序

- layer N 使用 `N % 2` 对应的计算 bank。
- 当前层 KV 稳定后，store stream 提交 save(N)。
- load stream 同时向另一 bank 提交 load(N+1)。
- load(N+1) 覆盖目标 bank 前必须等待该 bank 上一次 save 完成。
- layer N+1 消费数据前必须等待 load(N+1) 完成。
- 两个 bank 足以保证 save 当前层与 load 下一层访问不同 bank；不得为轮转形式本身增加无收益 bank。

### LP-3：LMCache 持久化

- latent 和 indexer group 都必须完成逐层保存与加载。
- prefill 完成回复前，所有必须持久化的 store future 必须完成或明确失败。
- cache miss 仍需执行 bank reuse fence，不能因为没有 load 数据而跳过旧 save 等待。
- 最后一层 load 的 pinned host source 必须在 NPU load stream 完成后才能释放。
- abort 路径必须先同步并失效旧 generation/event，再释放 MemoryObj。

### LP-4：多请求

- 一个 chunk 中可以包含多个请求，每个请求只保存本轮实际 prefill 的连续 token 区间。
- slot mapping 必须按请求、group 和 bank 生成，不能把 batch 总 token 数误认为单请求范围。
- latent/indexer 的 token 区间必须一致；不一致时必须拒绝执行。
- 不得因为一个请求的尾部 token 与另一个请求同批调度而错误截断 LMCache 保存范围。

## 6. HCOM 与 NPU 传输重叠需求

### OV-1：提交顺序

- KV 数据稳定后、HCOM all-reduce 提交前，完成 save/load 的设备任务入队。
- HCOM 提交后才允许进行可能阻塞 host 的持久化提交和 backpressure 处理。
- pre-HCOM callback 只允许固定地址查询、event 依赖和 native kernel launch。
- pointer table、native state 和 host MemoryObj 必须在 forward 前预创建或预热。

### OV-2：stream 与 core

- compute、load、save 使用显式 stream/event 依赖。
- dense layer copy 默认限制为 8 个 AIV，避免无必要占满 vector core。
- HCOM 使用 AIC/MIX_AIC 不代表天然可并发；必须以设备 timeline 和 makespan 验证。
- 不得用独立 event 区间拼接出虚假的 overlap 百分比。

### OV-3：性能验收

- 输出 all-reduce-only、load-only、save-only、strict-serial 和 combined 时间。
- combined 应明显小于 strict-serial；不要求写死 all-reduce 的绝对耗时。
- 传输 payload 应按可配置带宽模型构造，并报告真实 byte 数。
- 生产 profile 必须覆盖首尾及全部 chunk，确认传输没有被 host 空洞延迟到 HCOM 之后。

## 7. Graph 兼容需求

- 标准 PIECEWISE 使用 `vllm::mla_forward` runtime split，每次 replay 必须执行 transfer callback。
- P 节点不得启用会绕过逐层 callback 的 FULL graph。
- P 节点不得启用当前 staged SFA cross-layer save 路径；该路径会在 forward 后保存，双 bank 数据可能已被覆盖。
- graph capture/replay 中 slot mapping tensor 地址必须稳定，step 间只更新内容。
- D 节点已有 staged SFA graph、scratch 和 decode 行为不得因 P 节点功能发生回归。

## 8. Lightning Indexer 与 Sparse NPU Cache 需求

- Lightning Indexer 必须对完整历史 token 执行打分，并输出确定的 Top-K token 索引。
- indexer key 与 MLA latent 是不同 cache group，不得混用 shape、dtype 或 slot mapping。
- sparse NPU cache 只物化本轮 Top-K latent，不改变 indexer 的全历史可见性。
- union、intersection、空洞查找和 remap 必须保持 token 顺序、去重和 padding 语义。
- resident sparse scratch 可以跨 decode step 复用，但请求结束或 generation 变化后必须失效。
- sparse cache miss 不得返回部分历史数据并继续计算；无法满足完整输入时必须明确 fallback 或失败。

## 9. Decode Offload 需求

- decode backup window 是原 full-resident decode KV 的旁路备份，不是 attention 数据源。
- window size 必须是 latent block size 的整数倍。
- prefill 结束时需要保存不足一个完整 window 的 prompt tail。
- 仅在绝对 window 边界提交保存，不能按局部 batch 长度重新计数。
- async save 可以在后台执行，但请求结束前必须提供完成、失败和未完成状态。
- decode offload 不得改变 top-k 选择、原 decode block table 或 logits 精度。

## 10. 配置与兼容性需求

- 新能力默认关闭，非 P 节点不改变显存布局和运行时路径。
- P 节点必须同时满足：PP=1、LMCache layerwise、KV producer/consumer 能力、latent/indexer group 可用。
- MultiConnector 必须把 save/load/finish 只路由给声明支持 transfer window 的 child。
- 同一个 child 必须同时满足 layerwise transfer 和 indexer group 契约，不能由两个 child 分别拼出 capability。
- 配置不完整时启动失败，不允许退化成可能产生错误 KV 的单 bank 或 legacy 路径。

## 11. 可观测性需求

- 日志能够关联 request ID、rank、chunk、layer、bank 和 cache group。
- 调试计时只记录 host 边界时间点，不得通过 NPU 同步改变被测关键路径。
- profile 必须能够区分计算、HCOM、single-layer load/save、host idle 和最终持久化。
- 默认关闭详细诊断，开启后不得改变正确性和 stream 依赖。

## 12. 跨仓职责

| 仓库 | 主要职责 |
|---|---|
| vLLM | bundle pool、typed block、capacity、scheduler、connector window API |
| vLLM Ascend | 双 bank KV view、SFA/linear hook、graph admission、NPU metadata |
| LMCache | layerwise retrieve/store generator、MemoryObj 生命周期、持久化和 abort |
| LMCache Ascend | NPU direct load/save、stream/event、AIV kernel 和 HCOM overlap 验证 |

## 13. 验收标准

- 相同输入下，开启和关闭 layerwise cache 的模型输出满足既定精度阈值。
- latent/indexer bundle 分配、释放、抢占、取消和复用单测全部通过。
- cache hit、cache miss、部分尾块、多个请求、多个 TP rank 和两个 group 全部覆盖。
- P 节点显存随两个 bank 计费，支持明显长于 full-resident 基线的 prompt。
- D 节点 prefill/decode、staged SFA graph 和 speculative decode 回归通过。
- NPU profile 证明 load/save 与 HCOM 至少部分重叠，且 combined 小于 strict-serial。
- 正常完成和 abort 后无 pinned host 泄漏、UAF、stale event 或重复释放。
- 四仓的纯 CPU 单测、NPU 单卡测试和 TP8 HCCL 集成测试均有可复现命令。

