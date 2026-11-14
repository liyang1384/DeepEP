# LL Dispatch 优化移植与验证说明

本文记录在本仓库（`lueelu/DeepEP:feat/roce-support` 固定基线）上复现
《Dispatch 优化点与代码复现说明（单文件版）》**validated** 优化组合的结果、
构建身份和验证边界。实现以该说明第 4–10、14、20–21 节为准；第 11/22 节的
自有 Device RDMA 候选与第 12 节的 TileXR 未移植项不在本分支范围内。

## 1. 基线与分支

| 项 | 值 |
| --- | --- |
| 固定上游 | `1d0e13c92d1234cf8e359f83032bbd9bc335d830`（`lueelu/DeepEP:feat/roce-support`） |
| 分支 | `kernel/ll-dispatch-c-api-20261112` |
| 协议版本 | 固定 `36`（不再按 histogram 宏宣告 34/35） |
| SHMEM | 上游 gitlink `b41a8fa560b03ddfd9001052cd6002c6a28de809`（IPv6 v2 已包含） |

## 2. 实现的优化

1. **CountSlots 统一硬件直方图**（第 4 节）
   - `device/ll_count_histogram.h`：一次 `asc_frequency_histogram_bin0` 扫描，
     先 clamp 再偏移，尾部按剩余元素/字节更新 mask，只读 `histogram[i+1]`。
   - `dispatch_server_dedup_counts.h`：remote/local 共用 `CountSlots`，删除
     `CountByVectorRange`/`CountByExpert` 与 `DEEPEP_LL_COUNT_HISTOGRAM` 分叉；
     零计数专家仍发布 ready=1/count=0。
2. **同机 CountFrame 512B DataAsFlag**（第 5 节）
   - `common/ll_count_frame_layout.h`：每块 480B/15 条 32B record + 480..511B
     八个 `0x3f800000`；`CountWireBytes`/`CountWireRecordOffset` 与
     `LlCountFrame{sourceOffset,wireBytes,flagWords}`。
   - 发送端一次完整 block 发布；`PackLocalCountWire` 从高记录向低记录反向搬移。
   - `common/ll_count_read_plan.h`：`expectedFlagSum += length * frame.flagWords`
     取代按记录数判断 ready。
3. **Count / 全 Dispatch C API**（第 6–7 节）
   - 新增 `device/ll_count_c_api.h`、`device/ll_dispatch_c_api.h`：raw
     `__ubuf__`/`__gm__` 指针、显式 DMA pitch、事件、整数 count/浮点 ready 分离。
   - `ll_transport.h`、`dispatch_server_dedup_{counts,send,recv,magic,
     receive_packet,receive_tasks}.h`、`dispatch_dedup_mxfp8_quant.h` 迁移到显式
     UB 布局（`LlDispatchC::UbLayout`，190KiB 上限）与 role 生命周期；
     `dispatch_server_dedup_kernel.h` 删除 `TPipe*`，Count drain 后再 reset。
   - Combine 侧新增 `combine/low_latency/ll_transport_tensor_compat.h`，保持共享
     transport 的 Tensor 调用可编译。
4. **计数与 payload 重叠**（第 8 节）
   - `ll_count_frame_transport.h`：`CountRetirement` 位图，按同 peer 的 payload
     quiet 覆盖较早 count 请求；count TX 源区与 payload stage 隔离。
   - `ll_dispatch_rdma_batch.h`：128KiB stage 内按容量满批回收，按 peer 位图
     quiet，不再假定 8 包或 `1<<packetIndex`。
5. **跨机 DIRECT 优先**（第 9 节）
   - `common/ll_send_order.h` 增加 `remoteFirst`：ownGroup 放最后，其余组保持
     原 half-interleaving；Sender 仅在 `fullMeshFirstHop_`（Group16 且 RDMA）时开启。
6. **时钟对齐**（第 10 节）
   - 新增 `runtime/clock_alignment.{h,cpp}`：四时间戳、单 AIV `sleep_at`、
     1.5MiB 控制区、既有 QP0；`_C` 导出 `low_latency_clock_sync`/
     `low_latency_clock_sleep_at`。
   - 测试入口：`tests/ll_rdma/one_click.py` + `tests/ll_stage_timer/run.py`，
     显式 `--clock-align`/`--no-clock-align`。
7. **构建与入口**
   - `cmake/device/CMakeLists.txt`：LL 源文件拆 plain/reference object 目标，
     保留 direct/clos 与其余 kernel；`DEEPEP_LL_STAGE_TIMER=ON` 直接报错。
   - `build_support.py`/`CMakeLists.txt`：从环境读取并记录
     `DEEPEP_LL_REFERENCE_PROFILE`/`DEEPEP_LL_STAGE_TIMER`，wheel 身份写入
     `_build_info.json`（含 SHMEM `build` 身份）。
   - `low_latency.cpp` 入口使用编译器 intrinsic `set_ffts_base_addr` 与 `trap()`，
     保持单 LL 翻译单元与 launch ABI。

## 3. 构建

环境：aarch64 openEuler、CANN 9.2.0、Python 3.12、SHMEM `b41a8fa` 预编译 SDK。

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
export DEEPEP_NPU_ARCH=Ascend950
export SHMEM_ROOT=<project>/third-party/shmem/install/shmem
export DEEPEP_BUILD_MEGAMOE=OFF DEEPEP_MEGAMOE_TIMER=OFF DEEPEP_BUILD_EP_GMM_FUSED=OFF
export DEEPEP_LL_STAGE_TIMER=OFF DEEPEP_LL_REFERENCE_PROFILE=OFF   # ON 用于诊断
python3 -m pip wheel . -w wheels --no-build-isolation --no-deps -v
```

两种 variant 使用独立 build 指纹目录；`_build_info.json` 的
`ll_reference_profile`/`ll_stage_timer` 与 `_C` 导出的 capability 必须一致。

## 4. 已验证内容

| 层 | 证据 |
| --- | --- |
| 编译 | 同一源码树分别以 `DEEPEP_LL_REFERENCE_PROFILE=OFF/ON` 完整构建 wheel，包含 direct/clos/UBX/其余 kernel，无 error |
| 产物 | OFF/ON wheel 均安装并 import 成功；`low_latency_core_protocol=36`、`implementation=ndr-hns1825`、`rdma_enabled=True`、`stage_timer_enabled=False`、`clock_alignment_version=1` |
| 身份 | `_build_info.json` 含 `source_sha256`、SHMEM revision + `build`（`target=Ascend950`、`cann=9.2.0`）；ON wheel `ll_reference_profile=True`、`ll_stage_timer=False` |
| 布局一致性 | ON wheel 的 `low_latency_reference_profile_layout()` 与 `deep_ep/utils/ll_reference_profile_parse.py` 的 schema/slots/fields/640 words 完全一致（含 `w62=receive_relay_distinct_ranks`） |
| Host/CPU | `tests/ll_dispatch/host_layout_test.cpp`：512B DataAsFlag 算术、remote-direct/local `LlCountFrame`、read-plan run 重建与 `expectedFlagSum`、remote-first 排列是 own-group-last 的双射、stagger 轮转 |

Host 测试运行方式：

```bash
g++ -std=c++17 -O2 -I csrc tests/ll_dispatch/host_layout_test.cpp -o /tmp/ll_host_layout
/tmp/ll_host_layout     # ll host layout checks passed
```

## 5. 未覆盖内容与边界

- 本环境没有 RoCE 业务网络，**未执行**双机 EP32/BS32/FP8 OFF A/B、正确性 ST
  和 performance 采样；因此不引用说明第 17 节的任何微秒收益，也不声称已验证性能。
- 说明第 11/22 节的**自有 Device RDMA 候选后端未纳入本分支**：validated 组合使用
  SHMEM Device 后端，候选在冻结时仍待设备验收；沿用其代码需要新的性能与正确性证据。
- 说明第 12 节的 TileXR 方向（多 SGE、超前 relay、双 probe、元数据预取、CQ 批量）
  在第 12 节明确为“尚未移植”，本分支不包含。
- 参考 profile 的 **count 微段（mode 4/27 段）实现未包含在说明正文**。本移植保留
  count 调用点的全部边界 API，并以 mode 1 编译为 no-op：ON wheel 产出说明第 20 节
  定义 schema-4 的 29 段标准记录（parser 校验通过），但不提供 count 子段拆解。
  其余第 13 节已回退方向均未恢复。
- 说明第 15.3 节的 SHMEM IPv6/affinity 来源补丁未重放：上游 gitlink `b41a8fa`
  本身已包含 `support IPv6 links in v2`，本环境使用按该 revision 记录的预编译 SDK。
