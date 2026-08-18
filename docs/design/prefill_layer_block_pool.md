# Layerwise Prefill KV-Cache Bundle Pool 设计与实施计划

## 1. 目标与首版边界

在不把 vLLM、vLLM Ascend 和 LMCache 完整改造成 HMA 的前提下，为 PD 分离中的 P 节点提供 layerwise prefill KV cache：模型执行时只在 NPU 上保留流水线正在使用的两个 layer bank，历史 KV 从 LMCache 逐层加载，新生成的 KV 逐层保存到 LMCache。

首版边界如下：

- 仅 P 节点启用；P 节点完成 prefill 并确认 KV 已持久化到 LMCache 后结束请求。
- D 节点保持现有 full-resident 路径、内存布局和调度行为，不引入本地 `PREFILL_CHILD -> FULL_PARENT` 迁移。
- P 节点上的所有请求都使用 `PREFILL_CHILD`，不增加请求级开关，首版不支持同一实例内 mixed allocation。
- 内部 block identity 仍保留 `FULL_PARENT/PREFILL_CHILD` 类型边界，allocator 也保留 parent reservation 抽象，避免未来支持 mixed allocation 时依赖数值范围猜测或再次改协议。
- 必须支持现有 PIECEWISE graph；不能以关闭 graph 作为功能前提。
- 首版不支持 prefix caching。

节点身份和功能开关使用同一个布尔环境变量：

```bash
VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE=true
```

- 默认值为 `false`。
- `true` 表示该实例是 P 节点，并启用 global slab、child allocator 和双 bank 流水线。
- `false` 表示非 P 节点，完整保留旧路径。
- 值按大小写不敏感的 `true/false` 解析，其他值在启动时直接报错。
- 不使用 `kv_role` 或 `kv_rank` 推断 P 节点。
- 在 P 节点启动时断言 `pipeline_parallel_size == 1`；非 P 节点不受影响。
- TP 保持支持；MTP/speculative 路径不增加限制，使用现有行为并做回归测试。

## 2. 物理布局和地址模型

现有 DSA shared bundle 的物理复用关系保持不变：

```text
1 bundle = 2 latent blocks
         或 9 indexer blocks
```

设：

- `C`：现有 allocator 的可分配 parent bundle 数，不含 block 0；
- `L`：当前 worker 实际拥有的本地 DSA layer pair 数；
- `K = L * C`：global slab 的可分配 child bundle 总数；
- `physical_slot`：与逻辑层无关的物理 child 槽，范围为 `[0, L)`；
- `parent_bundle_id`：范围为 `[1, C]`。

global slab 中的 child bundle 使用稠密编号：

```text
child_bundle_id = physical_slot * C + parent_bundle_id
global child bundle capacity = K = L * C
```

`child_bundle_id == 0` 永久保留为 null/padding，不为每个 `physical_slot` 额外保留 null hole。latent/indexer block ID 和 byte offset 必须通过 global `DSASharedBlockLayout` helper 计算，不能复用 legacy `slot_count` 做裸算术，尤其要保证 indexer NOPE/PE 分区在 global slab 中的 offset 正确。

`physical_slot` 不绑定逻辑 layer。逻辑层 `l` 在运行时选择该请求两个 bank 中的一个；同一个 child 可在生命周期不重叠时服务于任意逻辑层。

## 3. 核心不变量

1. parent bundle 只能处于 `FREE`、`FULL_LATENT`、`FULL_INDEXER`、`PREFILL_RESERVED` 之一。
2. `PREFILL_RESERVED` parent 从 full-parent allocator 中移除，其 `L` 个 child 仅由 `PrefillLayerBundlePool` 管理。
3. 一个 child bundle 只能由 latent 或 indexer 其中一种 owner 使用。
4. Prefill 分配优先填充已有 `PREFILL_RESERVED` parent，仅在现有 child 不足时 reserve 新 parent。
5. 两个 bank 和 latent/indexer 两个 KV group 必须事务式分配；任一步失败都回滚本次全部分配。
6. 每次 request arena 分配产生递增 `allocation_generation`。异步完成必须匹配 `(request_id, allocation_generation, logical_layer, kv_group, bank, save_job_id)`，旧 generation 的回调不得影响已复用资源。
7. `load_done` 前 attention 不得访问对应 bank。
8. D2H 将数据交给 CPU `MemoryObj` 后产生 `source_done`；`source_done` 后 NPU child/bank 可复用。
9. 后端存储提交完成后产生 `persist_done`；P 请求只有在全部必需数据 `persist_done` 后才成功完成。
10. parent 内所有 child 的 allocator refcount 都归零且 NPU source lease 都结束后即可归还；不需要等待已由 CPU `MemoryObj` 独立持有的数据完成持久化。
11. feature 关闭时不得改变现有 allocator、KV tensor 数量、block table、LMCache 地址或 graph 行为。

## 4. 实施步骤

### 4.1 扩展 DSA shared allocator

在 `DSASharedBundleAllocator` 上增加独立的 parent reserve/release API，并新增 `PrefillLayerBundlePool`：

- parent 展开为 `L` 个 layer-agnostic child bundle；
- child 维护 `FREE/LATENT/INDEXER`、block refcount、source lease 和 allocation generation；
- 多个 P 请求可使用同一 reserved parent 的不同 child；
- 使用已 reserve parent 优先的 packing 策略，降低内部碎片；
- parent 的所有 child allocator refcount 归零且最后一个 source lease 释放后归还 parent；
- reserve、双 bank、两个 KV group 的分配和回滚由 scheduler/core 统一执行，worker 不在 forward 中修改 allocator 所有权。

请求在每个 bank 中需要容纳完整的 attention 可见前缀，而不是只容纳当前 chunk。单请求容量为：

```text
required_child_bundles =
    2 * (ceil(required_latent_blocks / 2)
       + ceil(required_indexer_blocks / 9))
```

多请求按上述值求和。若单请求超过 `L * C`，调度前直接返回明确容量错误，不回退到 `FULL_PARENT`；普通排队和调度压力仍按现有 scheduler 机制处理。

### 4.2 分配并注册 global slab

P 节点 KV cache 初始化时：

- 一次性分配满足 global layout 总字节数的 aligned `uint8 global_raw`；
- 保留 backing allocation 和 aligned view 的强引用，避免对齐 slice 隐藏 owner 后被回收；
- 对整个 `global_raw` 只调用一次 global reshape，生成 `global_latent_k_nope`、`global_latent_k_pe` 和 `global_indexer`；
- 禁止先 reshape 每层 raw 再 `cat`。indexer PE 区域依赖整个 slab 的 NOPE 页面总数，逐层拼接会得到错误布局；
- 所有本地 DSA attention layer 绑定同一个稳定 global cache base；
- LMCache 注册同一 global base 时处理重复 layer 指针，不重复计算或重复注册错误的 page range；
- 保持 2 MiB 对齐，并验证 NPU operator 对 block 第一维和 page stride 的要求。

非 P 节点继续使用现有 per-layer raw tensor 路径。

### 4.3 类型化 block identity

逻辑资源身份使用：

```text
(allocation_mode, logical_block_id)
allocation_mode = FULL_PARENT | PREFILL_CHILD
```

首版 mode 由节点级配置决定：P 节点恒为 `PREFILL_CHILD`，非 P 节点恒为 `FULL_PARENT`，没有请求参数。mode 随 scheduler 输出、cached request state 和 connector lease 明确传递；禁止根据整数范围猜测 mode。只有 worker 在提交 kernel/connector 地址前将逻辑 identity lowering 为 global physical block ID。

### 4.4 双 bank 和 graph-compatible metadata

每个活跃 P 请求预留两个完整 bank：

```text
逻辑层 l：
  current compute/save bank = l % 2
  next prefetch bank = (l + 1) % 2
```

当前同一 KV group 内各层共享一份 attention metadata，不能直接表达双 bank 轮转，因此需要增加逐层 lowering：

- 在 P 节点的标准 PIECEWISE 路径中，以现有 `vllm::mla_forward` 为逐层边界；
- 不新增 graph split point；`mla_forward` 的运行时回调在每次 replay 都会执行。P 节点拒绝 FULL/FULL_DECODE_ONLY 以及 staged cross-layer SFA graph，后者会绕过或延后逐层窗口；
- 在每层执行前，将逻辑 block table lowering 到预分配、固定地址的 current-layer scratch table；
- `slot_mapping` 必须使用完全相同的 mode、bank 和 global layout 转换；
- graph capture 后不新建 tensor、不更换 storage、不改变地址，只更新固定 tensor 的内容；
- 更新过程必须批量化，禁止在 hot path 中逐 block `.item()` 或逐块 KV gather/copy；
- D 节点现有 cross-layer staged decode graph 及其 `sfa_lmcache_retrieve` 切分保持不变。

### 4.5 LMCache load/save 流水线

单层稳态流水为：

```text
bank[l % 2]：等待 load_done，执行当前层 attention；attention 完成后异步 D2H 保存当前层 KV
bank[(l + 1) % 2]：等待该 bank 上 save(l - 1) 完成后，异步 H2D 预取下一层历史 KV
o_proj pre-reduce window：本地 GEMM 后提交 save(l) 与 load(l + 1)，随后提交 HCOM all-reduce
```

约束如下：

- `save(l)` 和 `load(l + 1)` 在同一个逐层窗口内提交到独立 store/load stream；二者均与随后提交的 HCOM all-reduce 并发。
- 必须关闭 fused matmul-allreduce；本地 o_proj GEMM 与 HCOM 之间需要保留显式回调边界。
- `load(l + 1)` 覆盖另一 bank 前必须等待该 bank 的 `save_done(l - 1)`；下一层消费前必须等待 `load_done(l + 1)`。没有 cache hit/load 时，下一层仍必须等待该 bank 的旧 `save_done`。
- 保留 LMCache Ascend 现有 dense-direct D2H 路径，不新增 NPU staging buffer，也不新增一次 D2D 拷贝；
- D2H 完成产生 `source_done`，释放该 bank 的 NPU source lease；
- CPU `MemoryObj` 的 backend commit 独立产生 `persist_done`；
- 所有 layer 和 KV group 持久化成功后，P 请求才向上层报告完成，之后 D 节点继续走现有路径；
- pending persistence 使用有界 host 队列。达到 byte/job 上限时对 layer pipeline 施加 backpressure，禁止无限累计 CPU 内存或丢弃数据；
- 调度阶段宣称命中的缓存范围必须是所有必需 layer/KV group 的共同完整前缀；逐层 load 与已宣称范围不一致时有限重试，仍失败则终止 P 请求，不静默补算；
- 持久化失败时 P 请求失败，不通知 D 节点数据可用；重试从 CPU `MemoryObj` 进行，不重新占用 NPU child。

### 4.6 完成、取消、抢占和异常

- 正常完成：等待全部 `persist_done`，最后一个不足 window 的片段也必须提交。
- 取消：停止提交后续 layer，等待已发起 D2H 到达 `source_done` 后释放 child/parent，取消尚未提交的持久化任务。
- 抢占：按取消路径清理；重新调度时从第 0 层重新执行，不从 LMCache 中的半成品 layer 恢复。
- load/D2H 异常：同步或 fence 对应 stream 后再释放 source；不得在 DMA 仍可能访问时复用 child。
- persist 异常：保留 CPU object 供有限重试；最终失败则请求失败。
- 所有完成回调都校验 allocation generation，防止 ABA 和迟到回调。

### 4.7 涉及仓库

- `vllm`：parent/child allocator、事务式 scheduler allocation、类型化 block metadata、generation 和 lease 生命周期；
- `vllm-ascend`：P-node 环境配置、global slab、双 bank、block-table/slot-mapping lowering、PIECEWISE graph 接入及 o_proj pre-reduce 回调；
- `LMCache`：layerwise source/persist 两阶段完成协议、bounded pending-save/backpressure、最终持久化 barrier；
- `LMCache-Ascend`：global block 地址计算、direct H2D/D2H 事件和 CPU `MemoryObj` ownership。

## 5. 测试计划

### 5.1 Layout 和 allocator 单测

- 对所有 `physical_slot`、首尾 parent、latent intra-offset、indexer NOPE/PE 边界做正反向映射测试；
- block 0 始终映射为 global 0，且不进入可分配区；
- global child ID 覆盖 `[1, L * C]`，无 hole、重复或越界；
- global block ID、view 和 raw byte offset 一致；
- latent 每 bundle 2 blocks、indexer 每 bundle 9 blocks，owner 不重叠；
- 优先填充已 reserved parent，最后一个 child 释放后 parent 正确归还；
- 两个 KV group、两个 bank 中任一步失败时全部回滚；
- 重复释放、错误 owner、越界 ID、迟到 generation completion 被检测；
- global slab backing owner、2 MiB 对齐及重复 LMCache pointer 注册正确。

### 5.2 Scheduler 和容量测试

- 每个 bank 都按完整可见前缀分配，不按当前 chunk 少算；
- 严格验证 `2 * (ceil(latent/2) + ceil(indexer/9))` 容量公式；
- 覆盖总容量边界 `K-1/K/K+1`、单请求超限和多请求共享 parent；
- 长 prompt 跨多个 chunked-prefill chunk；
- 抢占后 generation 更新并从第 0 层重算；
- P 节点 `pipeline_parallel_size != 1` 启动失败，非 P 节点不受影响；
- 环境变量默认 false、显式 true/false 以及非法值解析行为明确。

### 5.3 Block table 和 graph 测试

- 双 bank 的 layer-to-bank 轮转、global block table 和 `slot_mapping` 一致；
- padding、不同 block-table 长度、batch 行重排和 chunked prefill 正确；
- PIECEWISE capture/replay 前后 tensor storage 地址不变；
- graph replay 多轮更新 metadata 内容正确，不读取上一请求残留；
- feature 关闭时 graph partition 和 metadata 与当前路径一致；
- `MTP=2` 回归通过，不因本功能增加额外限制；
- D 节点 cross-layer staged decode graph 回归通过。

### 5.4 异步和故障注入测试

- `save(l)`、`load(l + 1)` 与当前层 o_proj HCOM 可同时在途，并验证 load 覆盖前等待 `save_done(l - 1)`；
- `load_done` 前 attention 被正确阻塞；
- `source_done` 后 NPU child 可复用，`persist_done` 前 P 请求仍不可成功完成；
- parent 在所有 source lease 结束后可释放，即使 host persistence 仍在进行；
- source/persist 乱序完成不会提前释放或错误提交；
- pending host 队列达到上限后正确 backpressure；
- load miss、D2H 失败、persist 失败、取消、抢占及进程异常均无泄漏和悬挂 DMA；
- 旧 generation completion 在 child 被新请求复用后不会产生 ABA 释放。

## 6. NPU 集成与 Profile

至少覆盖：单请求、多请求、长 prompt、跨 chunk、容量边界、MTP=2、TP 多卡，以及不同完成乱序和故障注入场景。

正确性基准：

- 输出 token 与当前路径一致；
- 保存到 LMCache 的 latent/indexer KV 与当前路径逐项一致；
- P 请求返回成功后，D 节点使用现有路径能够完整加载和继续执行。

Profile 必须确认：

- 没有新增 NPU staging buffer 或逐层 KV D2D/gather/copy；
- H2D、D2H 与 o_proj HCOM 形成重叠窗口；
- global slab 和 metadata lowering 未引入额外全局同步；
- graph replay 地址稳定；
- feature 关闭时无可测行为或性能回退；
- feature 开启后，相比当前 layerwise prefill，TTFT 回退不超过 5%，吞吐回退不超过 3%。

## 7. 可观测性

至少增加以下统计：

```text
parent bundle 总数、空闲数、PREFILL_RESERVED 数
child bundle 总数、空闲数、LATENT/INDEXER 占用数
reserved parent 利用率和内部碎片率
每请求两个 bank 的 bundle 数与 generation
load_done、source_done、persist_done 等待时间
H2D、attention、D2H overlap 时间
pending host save jobs/bytes 与 backpressure 次数
因 source lease 延迟回收的 child/parent 数
load retry、persist retry、失败和 stale completion 数
```

## 8. 验收标准

1. 环境变量为 false 时，普通路径的行为、内存布局、graph 和性能不变。
2. 环境变量为 true 时，P 节点所有请求使用 `PREFILL_CHILD`，非 P/D 节点保持旧路径。
3. 所有 live allocation 的物理 child block 不重叠，容量计算与双 bank 公式一致。
4. global latent/indexer view、LMCache 地址和 null block 映射正确。
5. PIECEWISE graph capture/replay 正常，运行期无 tensor 地址变化。
6. 每层本地 o_proj GEMM 后、HCOM 前提交当前层 D2H 和下一层 H2D；两类传输可与 HCOM 并发，且没有新增 NPU D2D 搬运。
7. `source_done` 与 `persist_done` 生命周期分离正确，不发生覆盖、提前完成或 ABA 释放。
8. P 请求仅在所有 KV 持久化成功后完成；失败、取消和抢占路径不泄漏资源。
9. 输出 token 和 LMCache 中的 KV 与基准一致，D 节点现有路径无需修改即可继续处理。
10. TTFT、吞吐和 feature-off 回归满足第 6 节阈值。

## 9. 后续扩展（不属于首版）

- 同一实例或同一 batch 内 `FULL_PARENT` 与 `PREFILL_CHILD` mixed allocation；
- 请求级 allocation mode override；
- `pipeline_parallel_size > 1`；
- prefix caching 与 parent/child hash ownership；
- 在不改变类型化 block identity 和 parent reservation 协议的前提下评估上述能力。
