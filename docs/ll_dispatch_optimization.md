# LL Dispatch 低时延优化记录

> 分支 `kernel/ll-ndr-ab-per-core-qp`，记录截至 `cde48af`。
> 本文汇总**已完成**、**待完成**、**已否决**的全部优化点，含算法协议、分核与任务划分、count 链、接收 consume、布局与缓存策略。

---

## 1. 基线与口径

### 1.1 当前 shape

| 项 | 值 |
|---|---|
| 并行 | EP32（2 节点 × 16 rank），`kLlPhysicalRanksPerHost = 16` |
| 专家 | 总专家 384，`localExperts = 12` |
| 每卡 token | `bs = 32`（恒定，不是 512） |
| topk | `k = 8` |
| 核 | 56 个 AIV：28 发送（S）+ 28 接收（R），`aivUsedAllToAll_ = aivUsedCumSum_ = coreNum/2` |
| 计数记录 | `rscvStatusNum_ = localExperts × ranks = 384`，每条 32B，GM 步长 512B |
| 单 token wire | `tokenStride ≈ 4KB`，每 480B 后插 32B flag |
| 小 batch 特例 | `bs < 512` ⇒ `useAllCoreCount_ = true`，56 核都参与计数 |

### 1.2 打点口径

阶段（`stages.csv` 列）与参考 word 的对应：

```
kernel            = w21 - w0        count_receive   = w11 - w10
before_finalize   = w57 - w0        count_prefix    = w13 - w12
count_ready_poll  = w25 - w24       count_complete  = w15 - w14
cumsum_join       = w27 - w26       prefix_ready    = w29 - w28
receiver          = w19 - w18       recv_packet_poll= w33
recv_load_consume = w34             recv_loop_clear = w35 - w32
sender            = w9  - w8        send_packets    = w39 - w38
send_drain_metadata = w40 - w39     recv_first_packet = w42 - w32
recv_count_arrive = w46 - w10       recv_count_poll_iter = w47 - w24
```

计数列（`stages.csv` 末尾 5 列，`a2b0cb9` 起）：

```
w37 receive_task_count      w41 receive_probe_batch_count  w43 send_route_count
w48 receive_first_hop_tokens  w49 receive_relay_members
```

**纪律**：跨轮次的绝对值不可比（S 侧 `sender` 会抖动 ±6µs）；只能看轮内相对量（如 `receiver` 极差、`receiver − count`）。

---

## 2. 优化点总表

状态：**已生效**（含提交号）/ **待做**（按优先级 P1→P3）/ **降级** / **否决**（原因见 §3）。

### A. 算法 / 协议

| 编号 | 优化点 | 做法 | 依据 | 预期 | 状态 |
|---|---|---|---|---|---|
| A1 | 每源一份"专家计数向量" | 发送端把对某目标 rank 的 `[q0..q11]`（12×2B）压成一条 32B 槽；接收端读 32×32B=1KB 算全前缀 | 屏障里仍含一轮跨核发布（`cumsum_join` 1.7~10.1）；记录数 384 | −3~5µs | 待做 P3 |
| A2 | 计数嵌入 payload 首包 | 取消独立 count 帧，接收端收包即统计 | 与 A1 重叠，需原子或扫描 | 大但周期长 | 待做 P3 |
| A3 | per-expert 远端原子计数 | `aclshmemx_roce_atomic_add` | 固定 `qp_idx=0` 会串行 | — | 否决 |
| A4 | cumsum 收窄到 8 核 | 参考 `moe_distribute_dispatch_v2_full_mesh.h` 的 `CUMSUM_MAX_CORE_NUM` | 记录数已只剩 384，屏障已 ~10µs | — | 否决 |
| A5 | wire 改 DeepEP 式密排 + 去 flag | `[source][slot]` 段内 token 密排 + epilogue scatter | 失去 `(source,dest,expert,q)` 第三方可寻址 → relay 不可行 | — | 否决 |

### B. 分核与任务划分

| 编号 | 优化点 | 做法 | 依据 | 预期 | 状态 |
|---|---|---|---|---|---|
| B1 | count 按记录条数均分 | `SplitToCore(rscvStatusNum_, 28)` | 每记录代价恒定（等 ready + 32B 求和），跨核极差仅 2.3µs | — | 保持 |
| B2 | 接收按前缀 token 数均分 | `Build` 取 `total/recvCores`，差 ≤1 | `w37 = 9~10`，token 数本就均等 | — | 保持 |
| B3 | 接收任务改等步长交错 | 核 r 取全局行 `{r, r+C, r+2C, …}`，每核拿各源的 1/28 | 原连续切片让整段继承晚到源与 relay 量 | `receiver` 极差 21.7→11.3 | **已生效** `33cc038` |
| B4 | 按 relay 量加权切分 | 发送端把每条记录的成员数写进计数记录 word1，`Build` 按 `q+members` 装箱 | `w34 ≈ 6.0 + 1.1×w49` | 仅 −2~3µs（B3 已吃掉大半） | 暂缓 |
| B5 | S 核空闲期参与 count / 接收 | 发完即从共享未完成队列领取任务 | S 侧余量 5~10µs | −2~5µs | 待做 P2 |
| B6 | 发送按 route 而非 token 切 | `bs/sendCores` 余数导致 4 核拿 2 token | `w43 = 16/8`，2-token 核 `sender` 慢 ~5µs | S 非瓶颈 | 待做 P3 |

### C. count 链

| 编号 | 优化点 | 做法 | 依据 | 预期 | 状态 |
|---|---|---|---|---|---|
| C1 | count 链探针 w44–w47 | 首帧/末帧、首次 ready、首轮扫描 | 使 count 阶段可拆解 | 诊断 | **已生效** `d850cc1` |
| C2 | cumsum 片段记分板紧凑化 | 28×28 转置矩阵 → 每核一个 32B 槽；连续 896B burst；槽回收移到屏障后 | 原方案每核写 896B 整行、读自己一列 28 条分散 line | 去冗余搬运 | **已生效** `3374ba9` |
| C3 | 计数记录读侧走共享 L2 | 去掉 `windowInstatusFp32Tensor_` 的 `CACHE_MODE_DISABLE` | 写者是 RDMA/MTE3，读者绕 L2 时盲等 | `recv_count_arrive` 3.9~16.1→**1.1~1.5**；屏障 19.5/28.5→**9.2~11.6** | **已生效** `5aff591` |
| C4 | cumsum 槽读侧走共享 L2 | 去掉 `selfRankWinInGMTensor_` 的 `CACHE_MODE_DISABLE` | 写者是本地 MTE3（落 L2 脏行），读者绕 L2 读到未回写 HBM | `cumsum_join` 17~40→**1.2~4.1** | **已生效** `f6699a3` |
| C5 | 完成 flag 并入 cumsum 槽 word2 | 删 `cumsumFlagOffset` | 该转置结构同时保护片段槽回收；合并后回收跨核可见 → 落后核读到 0 → 死锁；需 per-call epoch | 仅 0.5~1µs | 否决 |
| C6 | 计数记录 512B→32B 紧凑 | 16 条/512B line | 384 条从 384 条 line 降到 24 条 | −1µs | 待做 P3 |
| C7 | GetCumSum 轮询次数探针 | 新增一个计数 word | 用于区分"发布晚"与"看到晚" | 诊断 | 待做（可选） |

### D. 接收 consume / relay

| 编号 | 优化点 | 做法 | 依据 | 预期 | 状态 |
|---|---|---|---|---|---|
| D1 | **relay 去复制** | 同一份 payload 只落一份（每 `(source,dest-host)` 到达区），各专家槽只存 `(source,q)` 引用；consume 按引用读一次再散写 | `w34 ≈ 5.8 + 1.1×w49`，`w49` 0~12，`w37`/`w41` 无关；`w49-w62` 才是可省次数 | `w34` max 19→~6，`receiver` 24.7→**~13**，R kernel→~33 | 探针 `w62 receive_relay_distinct_ranks` 已加，待一轮打点确认前提 |
| D2 | 同 record 多 token 合并搬运 | 同一 record 的 q 个 token 在 peerMem 连续，一次 n-burst `DataCopy` 替代每 token 一次 `DataCopyPad` | 现在每 token 一次 4KB 拷贝 | −1~3µs | 待做 P2 |
| D3 | 消费顺序 token 行序 → record 序 | 同 record 连续到达、写同一 region | 减少稀疏批次的固定开销 | −1~3µs | 待做 P2 |
| D4 | `recv_tasks` 二分改前向扫描 | 交错后行号跨步 28，二分 9 步 | 实测 0.44→2.0µs | −1µs | 待做 P3 |
| D5 | `ClearFlags` 与 consume 重叠 | 增量清 / 延迟到窗口下次复用 | 实测仅 0.5~0.9µs | — | 降级 |
| D6 | relay 成员写合并 | read index 指向连续 rank 时合并成一次 n-burst | 现在每成员一次 `Packet` | −1~2µs | 待做 P3 |

### E. 发送侧

| 编号 | 优化点 | 做法 | 依据 | 预期 | 状态 |
|---|---|---|---|---|---|
| E1 | q 批量预计算 | 每 8 条路由一次 V→S，替代每条一次 | 每条路由 `0.84µs`，其中 q ≈0.3 | −3µs（S） | **已回退**，重做需整块 dst，见 §3 |
| E2 | `send_drain_metadata` 与 payload 重叠 | 提前 quiet | 10~14µs 主要是 RDMA quiet（网络完成等待） | — | 不可压 |
| E3 | SIMT 16-lane 拼 WQE + AICore 门铃 | 照抄 DeepEP-Ascend | S 侧 30~43µs 且已批量 post，下发非瓶颈 | ≈0 | 否决 |
| E4 | 探针 w44/w45 定位首帧/末帧 | 已完成 | — | 诊断 | **已生效** |

### F. 布局 / 缓存

| 编号 | 优化点 | 做法 | 依据 | 预期 | 状态 |
|---|---|---|---|---|---|
| F1 | payload 读侧走 L2 | 去掉 `IssueLoad` 的 hint | 流式读（4KB×9/核），污染 L2 挤掉计数记录 | `w33`+1.2、`w34`+1.1、`receiver`+3.1 | **已回退** `f00fa1c` |
| F2 | flag 保持与 payload 同视图 | `ReadServerDedupFlags` / `ClearFlags` 与 payload 一致绕 L2 | 二者同事务写入 | — | 保持 |
| F3 | flag 粒度每 480B → 每 N token | 一条 record 的连续 token 共享一个 marker | wire 开销 6.25%→0.8%，flag 写/清 ↓16× | −2~4µs | 待做 P2 |

### G. 工具与采样

| 编号 | 优化点 | 做法 | 依据 | 状态 |
|---|---|---|---|---|
| G1 | 宽表追加计数列 | `stages.csv` 末尾加 w37/w41/w43/w48/w49 | 不必再翻原始 sheet | **已生效** `a2b0cb9` |
| G2 | 采样回退 exp100 | `torch.exp` 前缀 100 次，`SAMPLING_PROTOCOL=initial-reduce-exp100-batch-v3` | 设备忙等可直接用 torch 表达 | **已生效** `5042d5a` |
| G3 | 宽表追加形状列 | 加 `w52 EP` / `w53 active_aiv` / `w54 BS` | 避免本轮"bs 口径误判" | 待做 P3 |

### 关键结论修正

早期判断的"计数小包风暴 / 每 rank 384 帧 / 到达慢"**全部错误**。实测：

- `ls_count_assignment` 显示跨机目的地一帧承载一个远端 rank 的全部 12 个专家，每 rank 只有 32~48 帧，且 `send_count_last_frame ≈ 0.025µs`；
- 真正的 count 屏障来自**读侧缓存可见性**：数据早就到了，读侧看不见；
- 三处同类问题已修两处（cumsum 槽、计数记录），合计贡献约 35µs。

---

## 3. 已否决 / 已回退

| 提交 | 内容 | 回退原因 | 结论 |
|---|---|---|---|
| `5ffcd2a` → `f00fa1c` | payload 读侧也走 L2 | payload 是**流式读**（每 token 4KB 只读一次，28 核合计约 1.1MB），走 L2 不省延迟却污染 L2，挤掉计数记录/cumsum/relay 目标行：`w33` +1.2、`w34` +1.1、`receiver` +3.1µs | 判据：**小且反复轮询 → 走 L2；大且只读一次 → 绕 L2** |
| `1525c46` + `05da97d` → `6d6ed21` + `cde48af` | S 侧 q 批量预计算（每 8 条路由一次 V→S） | ① `Mins` 要求双操作数同 dtype（传 float 视图编译失败）；② 把 `ReduceSum` 目标从整块缓冲改成 32B 槽后**运行卡死**，怀疑 ReduceSum 写入超过 1 个元素、覆盖相邻槽 → 产生重复 q → 接收端永远等不到某个槽 | 设备侧向量 API 改动必须拆成最小可验证步；具体见 §9 |

### 明确不做

| 项 | 原因 |
|---|---|
| 照抄 DeepEP-Ascend 的 SIMT 16-lane 拼 WQE + AICore 门铃 | 我们 S 侧 30~43µs 且已批量 post，下发不是瓶颈，收益 ≈ 0 |
| per-expert 远端原子计数 | `aclshmemx_roce_atomic_add` 固定 `qp_idx = 0`，会串行；"每源一条计数向量"零原子即可达到同样效果 |
| 直接把 wire 改成 DeepEP 式密排 + 去 flag | 密排会失去 `(source, dest, expert, q)` 的第三方可寻址性 → **relay 不可实现**，必须先砍掉 first-hop |
| cumsum 收窄到 8 核（参考 full_mesh 的 `CUMSUM_MAX_CORE_NUM`） | 记录数已只有 384，屏障已降到 ~10µs，收益不足 |

---

## 4. 算法 / 协议层

### 4.1 当前协议事实

```
发送：token → Classify → 本 host 每条路由各一份 direct；每远端 host 合并成 1 个 first-hop 包
      (source, dest, expert, q) → DispatchServerDedupDispatchOffset 纯函数算出槽地址
计数：每 (expert, source) 一条 32B 记录，步长 512B；接收端按记录轮询 ready → 求段内前缀
      → PublishRecvCntSum（28×32B）→ GetCumSum → WaitCumSumFlag 屏障
接收：Build 任务表（按前缀 token 数）→ Probe（flag 位图）→ IssueLoad → ConsumeLoaded
      ├─ DIRECT：写 expandX / tokenType / destinationIndex
      └─ FirstHop：按 read index 对每个成员再 Packet 一份（relay）
```

### 4.2 与 DeepEP-Ascend 的取舍（结论）

| 维度 | DeepEP-Ascend | 本仓 | 结论 |
|---|---|---|---|
| wire 槽 | `[source_rank][slot]` 段内 token 密排 | `(expert, source, q)` 固定预留 + 每 480B flag | 我们的布局是 **relay 的必要前提** |
| 输出 | 按本地专家密排（epilogue scatter） | 按 token 行散写 | 都要一次重排 |
| 就绪判定 | 每专家远端原子 + `psum_per_rank` reduction | 每 32B 记录 ready flag 轮询 | 我们无原子，靠 flag |
| 容量 | `ranks × max_tokens × bytes` | `experts × bs × stride`（= localExperts 倍） | EP32 约 402MB，4GB 窗口够用，**容量不是问题** |
| 跨机字节 | 每 token ≈ distinct rank 数（~7.7 份） | 每 token 每远端 host 1 份（1~2 份） | **我们省 2.5~4×** |
| 单份字节 | ≈7296B（98% 有效） | ≈4096/8192B（91%） | 我们多约 10%（480+32 flag + 512 对齐） |

### 4.3 待做：A0 每源一份"专家计数向量"（降级为低优先）

- 做法：发送端把对某目标 rank 的 `[q0..q11]`（12×2B）压成一条 32B 槽；接收端读 32×32B=1KB 就能算出全部前缀，删掉 `PublishRecvCntSum` / `GetCumSum` 两轮跨核发布。
- 收益重估：**3~5µs**（原估 −28µs）。原因：屏障的真正来源已被 `5aff591` 修掉，剩下的是"等最慢记录 + 一轮发布"，A0 只能拿掉后者。
- 风险：改 host 布局 + 发送端 + 接收端三处，收益与成本不匹配，**排在 relay 之后**。

---

## 5. 分核与任务划分

| 项 | 现状 | 已被验证的结论 |
|---|---|---|
| 发送分核 | 按 token 切 `bs/sendCores`（`bs=32` 时前 4 核各 2 个） | `w43 = 16/8`，2-token 核 `sender` 慢约 5µs；S 非瓶颈，暂不动 |
| 计数分核 | `SplitToCore(384, 28)` **按记录条数**均分 | **正确**：每条记录代价恒定（等 ready + 32B 求和），跨核极差仅 2.3µs |
| 接收分核 | `Build` 按前缀 token 数均分（`total/28`，差 ≤1） | token 数本来均等（`w37 = 9~10`），不均来自 relay；**按记录条数切反而会引入 100 倍不均** |
| 接收任务映射 | `33cc038` 起改为**等步长交错** | 已解决"整段继承晚到源"；`receiver` 极差减半 |
| relay 量分布 | 交错后 `w49` 仍 0~12（Σ=96，均值 3.4） | 只有 ~9 个 token/核，成员数在分配时不可知 → 需要发送端发布或消除复制 |

### 待做：relay 去复制（当前最高优先）

- 现象：`w34 ≈ 5.8 + 1.1 × w49`（µs），relay 成员数是 consume 唯一解释量；`w37`/`w41` 与 `w34` 无关。
- 前提待确认：relay 成员是 `Packet(...)` 把 UB 里的 `wire` 写到目标 rank 的专家槽，属于必要落地。只有当多个成员指向**同一 destinationRank**（只是 expert 不同）时，才能改成"写一次 + 目标 rank 本地散写"。
- 判据：`w49 - w62`（`w62 = receive_relay_distinct_ranks`）即可省次数；按最慢核算 `(w49max - w62max) × 1.1µs ≥ 3µs` 才值得动协议，否则 D1 直接否决。
- 做法：同一份 payload 只落一份（per `(source, dest-host)` 到达区），各专家槽只存 `(source, q)` 引用；consume 时按引用读一次、再散写到 `expandX`。
- 预期：`w34` 最大 19 → **~6**，`receiver` 24.7 → **~13**，R kernel → **~33**。
- 风险：改动 consume 读路径与 relay 语义（相当于把 host 内去重从 dest 侧移到消费侧），属协议级；需保证 `ClearFlags`、窗口复用、magic 交替仍然正确。

---

## 6. count 链细节

当前结构（`bs=32`，28 接收核）：

```
count_receive (2.5~11.5)  轮询 384 条记录 → 稠密读 → GatherSum → 发布 28×32B 片段
count_prefix  (2.2~11.1)  GetCumSum 等齐 28 个片段 → 算全局前缀
count_complete(1.0~3.0)   WaitCumSumFlag 转置屏障 + receiver0 写 expertTokenNumsOut
屏障 = ready_poll + cumsum_join ≈ 9.2~11.6µs（28 核极差 2.4µs）
```

| # | 项 | 状态 | 说明 |
|---|---|---|---|
| 6.1 | 计数记录 512B 步长 → 32B 紧凑（16 条/line） | 待做 | 384 条从 384 条 line 降到 24 条；`recv_count_poll_iter` 已只有 0.6~0.8µs，纯收益 ~1µs，**低优先** |
| 6.2 | 完成 flag 并入 cumsum 槽 word2（原 C1） | **否决** | 屏障的转置结构同时承担"保护片段槽回收"的作用；合并后回收变成跨核可见 → 落后核读到 0 → 死锁。加 per-call epoch 才可行，收益仅 0.5~1µs |
| 6.3 | 计数嵌入 payload 首包，取消独立 count 帧 | 待做（长期） | 需要接收端在收包时统计（原子或扫描），与 A0 重叠 |
| 6.4 | S 核空闲期参与 count | 待做 | S 侧有 5~10µs 余量，可借 4 个核给 count 分核（28→32 源一一对应） |

---

## 7. 接收 consume 与 relay

| 项 | 现状 | 结论 |
|---|---|---|
| `w33` 等包 | 交错后 1.6~11.8µs | 等最后一批 payload 到达，受 S 侧发送窗口限制 |
| `w34` consume | 5.8~19.0µs，`≈ 4.5 + 1.2×w49` | **relay 复制是唯一解释量** |
| `recv_tasks` | 0.44 → **2.0µs** | 交错带来的二分搜索成本，可优化（前向扫描/行号表） |
| `ClearFlags` | ≈0.5~0.9µs | 便宜，**D2/B3 降级** |
| `recv_loop_clear` | 8~21µs | = poll + consume + clear，不是独立开销 |
| 同 record 多 token 合并搬运 | 待做 | 现在每 token 一次 `DataCopyPad`；同一 record 的 q 个 token 在 peerMem 连续，可 n-burst 一次读 |
| 消费顺序改 record 序 | 待做 | 同 record 连续到达、写同一 region；与交错叠加需实测 |

---

## 8. 存储布局与缓存 hint 判据

| 数据 | 体量 | 访问模式 | 正确 hint |
|---|---|---|---|
| cumsum 片段槽 | 28×32B | 本地 MTE3 写，28 核反复轮询 | **走 L2**（`f6699a3`） |
| 计数记录 | 384×32B | RDMA/MTE3 写，反复轮询 | **走 L2**（`5aff591`） |
| payload 包 | 每 token ~4KB ×9 | RDMA 写，每包只读一次 | **绕 L2**（`f00fa1c`） |
| 就绪 flag（480B 后 32B） | 每 token blockCount×32B | 与 payload 同事务写入 | **保持与 payload 一致（绕 L2）** |

判据：**写者走缓存、读者绕缓存 = 读到陈旧数据**；反复轮询的小对象必须与写者同视图，大流式对象则要绕开以避免污染。

---

## 9. 方法论与纪律

1. **设备侧无法本地编译**。本机只有 CPU 回归（Python 解析/表格/契约），任何涉及 `Mins/ReduceSum/Abs`、缓冲区尺寸、事件同步的改动都必须拆成最小可验证步，一次只改一处语义。
2. 本仓的 `Abs(float 视图) → Mins(int 视图) → ReduceSum(subnormal 位模式)` 是**刻意技巧**：
   - `Sub` 得到有符号 int 差（如 -3 = `0xFFFFFFFD`）；
   - `Abs` 走 float 视图（IEEE abs 只清符号位）把它变成 `0x7FFFFFFD`；
   - `Mins` 走 int 视图夹成 1，`0` 保持 `0`；
   - `ReduceSum` 在 `CTRL[60]=0` 下按 subnormal 位模式累加，得到"不同条数"，`index - 其它 = q`。
   - 任何"顺手简化"都会破坏它（`Mins` 一旦传入 float 视图直接编译失败）。
3. `ReduceSum` 的目标不要用"32B 槽位打包"：它可能写出 1 个以上元素，会覆盖相邻槽（`1525c46` 卡死的头号嫌疑）。
4. **先证伪再优化**：本轮两次最大的收益都来自"证明等待不是计算"（`ready_poll + cumsum_join` 恒定、最慢核自己也在等），而不是来自任何微优化。
5. 每轮打点必须标注环境；跨轮只比相对量。

---

## 10. 执行顺序

| 优先级 | 编号 | 理由 |
|---|---|---|
| **P1** | D1（relay 去复制） | 唯一还剩 8µs 量级的项，直接决定 R 侧长尾；先跑 `w62` 确认 `w49 > w62` |
| **P2** | E1（先用 word 75 判定）→ B5 / D2 / D3 / F3 | 都在 2~5µs，且互相不冲突 |
| **P3** | A1 / C6 / D4 / G3 / B6 | 单项 1~3µs，或属诊断完善 |
| 不做 | A3 / A4 / A5 / C5 / E3 / F1 / D5 / E2 | 原因见 §3 |

---

## 11. 复现

```bash
# 跑完一轮后，在工程目录离线重导出（只用标准库，不加载 CANN/Torch）
python tests/ll_rdma/analyze.py --analyze-only --run-dir <run 目录> --out <新目录>

# 生成 stages.csv（含 §1.2 的 33 阶段 + 5 个计数列）
# 原始 w0..w63 在导出的「原始Cycle」sheet，或 dispatch/rank_XXX/sample_*.json 的 raw[core][slot][word]
```

判读要点：
- count 是否仍是屏障 → 看 `ready_poll + cumsum_join` 的**极差**（应 ≤3µs）；
- consume 是否仍不均 → 把 `w34` 对 `w49` 做散点（应强相关）；
- 是否有"发布了但看不见" → 某核 `cumsum_join` 远大于 `max(ready_poll) - 自己 ready_poll`。
