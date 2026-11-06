# LL dispatch C++/MPI 运行手册

不经过 Python/torch，用 MPI 做主机侧协调、ACL + `deepep::Session` 建对称共享内存域，直接调用 dispatch kernel。
结构对齐参考实现 `Ziha0-Zeng/DeepEP:tests/ll_cpp_bench`。

| 文件 | 作用 |
| --- | --- |
| `main.cpp` | MPI rank/world → ACL 设备 → `Session(mpi=true)` → 分配 GM → 调 dispatch → 校验 |
| `ll_dispatch_c_api.h` | 与 pybind 入口对应的 4 个 host 函数声明 |
| `ll_dispatch_host.cpp` | 布局 / `CommArgs` / tiling / `EpServerDedupKernelArgs` / launch |
| `CMakeLists.txt` | 独立构建，不并入主工程 |

kernel、布局、tiling 算法与 Python 路径完全同一套；差别只是跳过 torch，用 `aclrtMalloc` 申请 GM。

---

## 1. 拓扑约束（先确认再动手）

| 约束 | 说明 |
| --- | --- |
| `world ∈ {16, 32, 64, 128, 256}` | 必须是 2 的幂；EP64 = 4 台 × 16 |
| 每台 16 个**连续** rank | `kLlPhysicalRanksPerHost = 16`，kernel 用 `rank / 16` 判断同机 |
| 每 rank 8 GiB 工作区 | 与 Python 路径一致；EP64 时每台 16 × 8 GiB HBM |
| `experts % world == 0` | 384 对 16/32/64 都成立；EP128 时 384/128 = 3 |
| `tokens ≤ capacity ≤ 256` | 默认 `capacity = tokens` |
| 仅 Ascend950 | 运行时会拒绝其它 SoC |

`capacity` 只影响 tiling 的 `numMaxTokensPerRank`（>32 会走去重通路），输出行数按
`rows = world × tokens × min(topk, experts/world)`：

- EP16 / tokens=32 / topk=8 / experts=384 → `rows = 4096`
- EP64 / 同上 → `rows = 12288`

## 2. 前置检查（每台都做）

```bash
npu-smi info | head -20                 # 需要能看到 16 张卡
mpirun --version                        # OpenMPI
ls third-party/shmem/install/shmem/lib  # SHMEM SDK 已构建
ssh 10.0.0.12 hostname                  # 4 台互相免密
ip -o -4 addr | grep 10.0.0             # 找到管理网卡名，例如 bond0
```

4 台的仓库路径、wheel 版本、`third-party/shmem` 必须一致（推荐共享目录或同镜像）。

## 3. 构建

```bash
cd /path/to/DeepEP-Huawei
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source tests/ll_rdma/env.sh            # 把 third-party/shmem/lib 加进 LD_LIBRARY_PATH
unset SHMEM_ROOT CATLASS_ROOT

cmake -S tests/ll_cpp_dispatch -B build/ll_cpp_dispatch \
      -DDEEPEP_PACKAGE_DIR=$(python -c "import deep_ep,os;print(os.path.dirname(deep_ep.__file__))")
cmake --build build/ll_cpp_dispatch -j
```

产物：`build/ll_cpp_dispatch/ll_cpp_dispatch`（RPATH 已指向 `DEEPEP_PACKAGE_DIR`，`libdeepep_kernels.so` 就在那里）。

## 4. 环境变量

每个节点都要有，跨机由 `mpirun -x` 显式透传：

| 变量 | 值 | 作用 |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | `set_env.sh` 提供 | CANN 头/库 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,...,15` | 每台 16 张卡；程序用 `device = local_rank` |
| `LD_LIBRARY_PATH` | `<repo>/third-party/shmem/install/shmem/lib:$LD_LIBRARY_PATH` | `libshmem.so` |
| `TMPDIR` / `TMP` / `TEMP` | `<repo>/.tmp` | 编译/运行时临时目录 |
| `ASCEND_WORK_PATH` | `<repo>/.logs/ascend` | 落盘日志 |
| `ASCEND_PROCESS_LOG_PATH` | 同上 | plog 位置 |
| `SHMEM_LOG_PATH` | `<repo>/.logs/shmem` | SHMEM 日志 |

**不要设** `MASTER_ADDR`、`MASTER_PORT`、`HCCL_SOCKET_IFNAME`、`HCCL_IF_IP`：
`Session(mpi=true)` 走 `ACLSHMEMX_INIT_WITH_MPI` + `MPI_COMM_WORLD`，**不建 HCCL 域**。

## 5. hostfile

`~/hostfile-ep64`：

```
10.0.0.11 slots=16
10.0.0.12 slots=16
10.0.0.13 slots=16
10.0.0.14 slots=16
```

- hostfile 只写“机器 + 每台允许几个进程”，**不含 rank 号**；rank 由 `mpirun` 按书写顺序、每台填满 16 个 slot 顺延分配 —— 第 1 行得 rank 0..15，第 2 行得 16..31。
- `slots=16` 必须写，否则默认每台 1 个 slot，`-np 64` 直接失败。
- 行数 × 16 = world：4 行 = EP64，2 行 = EP32，8 行 = EP128。

## 6. 启动前验证 rank 分布

```bash
mpirun --allow-run-as-root -np 64 -hostfile ~/hostfile-ep64 \
  --map-by ppr:16:node --bind-to none hostname | sort | uniq -c
```

期望 4 行、每行计数 16。若未按行连续，补 `--rank-by node`；要直接看映射表加 `--display-map`。

## 7. 运行 EP64（4 台 × 16 卡）

只在 node0 上执行：

```bash
NIC=bond0          # 换成第 2 步查到的那块内网网卡

mpirun --allow-run-as-root -np 64 -hostfile ~/hostfile-ep64 \
  --map-by ppr:16:node --bind-to none \
  --mca pml ob1 --mca btl self,tcp --mca btl ^openib \
  --mca btl_tcp_if_include $NIC --mca oob_tcp_if_include $NIC \
  -x LD_LIBRARY_PATH -x PATH -x ASCEND_HOME_PATH -x ASCEND_RT_VISIBLE_DEVICES \
  -x TMPDIR -x TMP -x TEMP -x ASCEND_WORK_PATH -x ASCEND_PROCESS_LOG_PATH -x SHMEM_LOG_PATH \
  /path/to/DeepEP-Huawei/build/ll_cpp_dispatch/ll_cpp_dispatch \
  --tokens 32 --hidden 3584 --topk 8 --experts 384 --capacity 32 \
  --dtype bf16 --backend rdma --warmup 5 --iters 20 --timeout 300
```

先单机自检可把 hostfile 换成 1 行、`-np 16`，其余不变。

## 8. 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--tokens` | 32 | 每卡输入 token 数 b |
| `--hidden` | 3584 | h |
| `--topk` | 8 | k |
| `--experts` | 384 | 全局专家数 e，须被 world 整除 |
| `--capacity` | = tokens | 每卡容量上限 M，`b ≤ M ≤ 256` |
| `--dtype` | bf16 | `bf16` / `fp16`（FP8 暂未接） |
| `--backend` | rdma | 对应 `DEEPEP_LL_BACKEND`；`mte` 仅做同机对照 |
| `--warmup` / `--iters` | 5 / 20 | 计时轮数（`aclrtEvent`，不含主机侧准备） |
| `--device-base` | 0 | 设备号偏移；`device = device-base + local_rank` |
| `--timeout` | 300 | Session 超时秒数 |

## 9. 输出与判据

rank0 打印：

```
tokens=32 hidden=3584 topk=8 experts=384 world=64 capacity=32 backend=rdma rows=12288
dispatch avg=xx.xxus over 20 iters (warmup 5)
expert counts total=256 expect=256 mismatches=0
```

三条同时成立才算通过：逐专家计数与 CPU 参考完全一致（`mismatches=0`）、
总和 `= tokens × topk = 256`、全 rank `MPI_MIN` 汇总通过。否则任一 rank 报错并 `MPI_Abort`。

## 10. 排错

| 现象 | 原因 / 处理 |
| --- | --- |
| `each host must own 16 consecutive ranks` | hostfile 顺序或 `--map-by` 不对；回到第 6 步验证 |
| `MPI_Init failed` / 卡在 `mpirun` 连接阶段 | 免密 ssh 不通，或改用 `--mca plm_rsh_agent "ssh -p 22"` |
| 卡在 SHMEM 初始化无输出 | `-x LD_LIBRARY_PATH` 漏透传，远端找不到 `libshmem.so` |
| `aclrtSetDevice failed` | `ASCEND_RT_VISIBLE_DEVICES` 少于 16 项，或该卡被占用 |
| `This runtime build is qualified only for Ascend950` | 跑在非 A5 机器上 |
| `LL Host/device core protocol mismatch` | `libdeepep_kernels.so` 与头文件不同一次构建；重建 wheel 或重编样例 |
| `dispatch shape exceeds topology, UB or workspace limits` | `experts % world != 0`、`topk > experts` 或 `capacity > 256` |
| `dispatch expert count mismatch` | 看 `first mismatch` 的 `rank/local_expert/got/want`：got 全 0 说明该卡没收到数据（多为跨机 RDMA 或组网问题） |
| 内存不足 | 每 rank 8 GiB，EP64 每台 128 GiB；先降 `iters` 不改工作区 |

## 11. 已知限制

- 只跑 dispatch，不跑 combine。
- 只支持 `bf16/fp16`；FP8 预量化需额外接 `scales` / `output_scales`。
- 计时段只覆盖 device 端连续 dispatch，不含主机准备；与 Python 的 33 阶段 reference 表不可直接比。
- 不设 HCCL，因此不能把这套流程的耗时与 HCCL 组网路径的耗时混用比较。
