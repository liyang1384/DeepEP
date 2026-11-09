> 当前默认身份34、直方图身份35：dispatch按GM stage容量批处理，combine回传批处理并将尾批CQ移至M5之后。见[COMBINE_BATCH_RETURN.md](COMBINE_BATCH_RETURN.md)。下方旧身份为历史。

# Count源隔离与quiet审计（默认30，直方图31）

对照ec15f2e（默认28/直方图29）。用户明确要求按算子数学依赖分析，不能把共用暂存区造成的
实现依赖当作count→payload的固有依赖。本次消除该共享：SendCnt不再执行quiet。

## 算子依赖与实现

- 源端payload不消费本端count发送完成状态；接收端看到SO尾FLAG后才能读count并计算prefix。
  count的源GM与payload源GM之间没有数学依赖，因此可以并行。
- 每个对称bank的1280KiB处复用已删除配对路由留下的空闲区，按目标rank分配独立count源槽。
  槽长`(E/P+1)*32B`，总长`(E+P)*32B<=40960B`；高于count/prefix及combine控制，低于2MiB payload。
  每目标只有一个核写；多轮使用不同槽，不再使用payload的每核128KiB Stage。
- UB→独立GM仍保留MTE3_S，随后一次整帧put，立即继续下一目标统计或本机统计。
  put返回后SDK UB scratch已退休；S_V只保护Scalar scratch再次给Vector使用，不等待网络完成。
- 接收端SO/完整flag检查、count快照汇合、旧flag清理、双bank及combine_reuse保持。

## CQ生命周期与去重

锁定HNS SDK的quiet调用poll_cq推进CQ/SQ consumer index；这不是接收端prefix的数学依赖，
但不应在不处理队列生命周期的情况下永久弃置CQ。本次把未退休count的回收移到本核业务结束：
SDK在SQ/CQ接近满时也会主动poll；这里不声称必须每调用显式quiet，而是选择本核末尾完成回收，
不把最后一次调用的源释放依赖于未来调用或销毁路径。普通put自身会ring doorbell，NIC进展不依赖quiet。

1. 每核只记录自己发布的远端任务，最多15位（EP256、16核），不新增UB/lane/QP。
2. Sender已有payload batch对同peer、同物理核QP执行quiet后，标记对应count已退休，避免重复quiet。
   依据同QP的CQ目标head覆盖已提交count，不推导跨QP或跨WQE的接收发布顺序。
3. 仅剩的count在本核业务结束后quiet。零payload的S核、空任务R核也覆盖，不依赖“总有后续payload”。
4. 先执行既有PIPE_ALL，再复用已退休的角色UB：活跃S核用outBuf、无payload的S核用原count scratch，
   R核用已消费的prefixBuf（多机最小128B）。不使用R核Reset前失效的TBuf，不覆盖packet trace。
5. 回收发生在w21之前，因此计入最终drain阶段和核总耗时；SendCnt变短不等于完整task同比变短。

count TX在本核结束前已退休；下次调用与shape变化不会覆盖未完成的count源。接收侧bank复用仍依赖
同通信器、同调用次序、单stream及所有源（含零）的发布，不由源区隔离自动免除这些条件。

## 其他quiet检查

| 位置 | 数学依赖/实现原因 | 本次处理 |
|---|---|---|
| DirectCountFrame立即quiet | 无count→payload数学依赖，原来用于共享Stage复用 | 删除；独立源区+末尾CQ回收 |
| payload batch同peer quiet与count末尾quiet | 同QP前序count已被payload quiet覆盖 | 合并，不重复调用 |
| dispatch满批/尾批quiet | 有限GM payload环复用/算子完成；不是数学上每批必须阻塞 | 保留；进一步隔离或扩大payload源池是单独改造，不能直接删 |
| combine Packet每包quiet | 当前单GM Stage反复覆写；不是数学上每包必须quiet | 保留；需要证明所有结果源/批次可保持到回收后，不能把本次小count源区方案机械用于大payload |
| legacy CountFrame远端body/commit两次quiet | 活跃多机count已走DirectCountFrame；当前调用仅用于同机MTE，不执行远端quiet分支 | 无活跃开销，不把死分支删除计作性能收益 |
| Records远端quiet | 当前count/local-ready调用属于本机，远端分支不在这些活跃路径 | 保留兼容helper，不声称已消除真实调用 |

没有删除SDK内部quiet，也没有新增全rank屏障/ACK。SDK源码和数学算法不变。

## 检查及目标测量

448项CPU回归通过、2项缺少Torch跳过。新增生产helper模型在NIC延迟读取期间覆写全部payload Stage
及原UB，检查count源保持；覆盖EP32/64/128/256、16..72核模型、同核多目标、payload quiet合并、
无payload回收。72核模型仍不代表SDK允许72核跨机（现有每peer56QP约束保持）。
未执行CANN编译或NPU正确性/性能，不能预先宣称10us收益。

所有rank更新当前分支完整wheel，不混装旧身份。环境变量保持：默认OFF身份30，直方图ON身份31。
两机更新后运行（NODE=0/1，MASTER=节点0地址；多机另加相同--num-nodes）：

```bash
git fetch origin && git switch kernel/ll-ndr-ab-per-core-qp && git pull --ff-only &&
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" count-isolated-001 \
  --mode perf --tokens 1 8 32 256 --capacity 256 --hidden 3584 --topk 8 --experts 384 --dtype fp8
```

对照ec15f2e保持同一SDK、shape/seed/采样方式。逐核诊断另用新run名及
`--mode reference --reference-iters 5`；同时看count_send、最终drain和完整OFF dispatch。
原始数据留目标机，单个回传报告仍<=60000字节。
