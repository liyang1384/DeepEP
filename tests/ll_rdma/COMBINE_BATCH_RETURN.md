# Combine回传批处理（默认34/直方图35）

分支kernel/ll-ndr-ab-per-core-qp。第一项dispatch满stage单独提交4c12865（默认32/直方图33）；
本次combine提交以它为直接对照。第二项dispatch依赖只审计，见DISPATCH_DEPENDENCY_AUDIT.md。

## 改动与真实收益口径

原DIRECT和M4 partial跨机回传均调用Packet，每包Stage复制、PIPE_ALL、put、quiet。
现在两条路径共用已验证的GM批次生命周期：

- 每核128KiB stage按完整包槽使用，包准备好即暂存/提交，不等待下一条归约准备。
- 每个包仍是一次完整WQE，保留原SO尾FLAG与接收检查；每批每peer仅一次quiet。
- 当前输出V→MTE3与GM源MTE3_S依赖保留，远端路径不再调用Packet的全PIPE_ALL排空。
- M4将所有包提交后即可进入M5。未满尾批GM保持不变，不以CQ完成阻塞M5；
  batch状态由外层kernel持有，不随临时Partial对象销毁。
- 本核最终PIPE_ALL后用已退休的resident arena前128B作为SDK scratch回收尾批CQ。
  M4输出事件仍交给M5按槽退休，不在M4末尾额外排空本机输出lane。
- 本机MTE回传、terminal本地暂存、ready/复用协议、q、权重、所有权、归约及Cast次序保持。
  无新增UB/lane/GM分配或全rank屏障；SDK不改。

这次减少的是逐包等待与quiet调用，增加在途度，**没有减少put次数或通信字节数**。
H3584的BF16 combine包长是7680B，每stage可放17包；一个满批的quiet数等于其中不同peer数，
不是固定17次。如果17包都去同一个peer，则17次quiet降为1次；peer各不相同则调用数不降，
但这些put已经并行在途，不再每发一包就等其完成。
SDK内部SQ/CQ接近满时仍可poll，不能保证硬件无限排队。

## 为什么没有直接把跨机包拼成一个put

当前多机G16去重路径中，对一个源token、一个非空远端主机：只有一个专家贡献时返回DIRECT，
多个专家贡献时已在该主机聚合为一个partial返回。它已经做了逐token/主机的数学聚合。
若要进一步减少WQE，需要把多个token的结果合成消息。现有目的槽按`token*K+topk`排列，
同一发送核的多个结果通常不连续；中间槽可能由其他producer写入，不能直接连续put覆盖空洞。
进一步合包需要新的返回消息布局/接收分发及窗口复用设计，不能冒充为此次批处理已实现。

## 生命周期与测试

1. Enqueue先把完整包复制到本核独占GM槽，MTE3_S后put。SDK只借用UB的前128B，put返回后可复用。
2. Stage未满时后续归约/打包可继续，旧GM槽不覆写；stage满后退休所有涉及peer再从槽0重用。
3. peer记录使用256位按rank位图，覆盖超过32包和peer255；不再用包序号作32位移位。
4. M4所有put在M5之前已提交，NIC进展不依赖接收者主动quiet，因此不会形成“大家都等未提交末包”的环。
5. M5不写RDMA stage，可以复用M4 UB。最终CQ退休前该GM源仍保持，后续算子不会覆盖未完成源。

CPU整体回归449项通过、2项Torch缺失跳过；新增回传模型最初发现模拟peer地址空间重叠，
修正测试桩后相关检查通过。最终版本的298项针对性回归通过（与整体重叠，不相加）。
新增生产回传helper模型覆盖450组：1/2/8输出lane、512B至128KiB包、最多256槽、
满批/尾批/零任务、peer16/31/32/63/127/255、延迟DMA/NIC、破坏性SDK scratch以及M5覆写全部M4 UB。
尚未执行CANN编译、NPU正确性或性能，不能据CPU检查宣称加速。

## 目标机测试

全rank更新当前完整wheel，编译身份固定35。
保持相同 SDK、shape 与采样方式比较4c12865与当前版，不混装不同身份/源码。两机NODE=0/1，MASTER为节点0地址：

```bash
git fetch origin && git switch kernel/ll-ndr-ab-per-core-qp && git pull --ff-only &&
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" full-stage-combine-001 \
  --mode perf --tokens 1 8 32 256 --capacity 256 --hidden 3584 --topk 8 --experts 384 --dtype fp8
```

该配置是FP8 dispatch+BF16 combine。多机另加所有节点相同的--num-nodes；E须能整除实际EP。
不使用旧wheel的--skip-build，不重建SDK。比较完整OFF dispatch/combine；reference模式只诊断dispatch，
不能用它生成combine内部阶段结论。原始数据保留目标机，回传单个<=60000字节报告。
