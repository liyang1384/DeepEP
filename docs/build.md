<!--
Copyright (c) 2026, Lu Lu
Modified by Joyce_An 2026
-->

# 构建、打包与安装

wheel 包含 Host 运行时、Notify 及 HT/LL dispatch/combine kernel。输入和执行约束见 [Elastic 接口](elastic.md)，资源与事件见 [API 指南](api.md)。
MegaMoE 和 EpGmmFused 通过独立开关选择构建，见下文实验组件说明。
包名为 `ascend-deepep`，导入名仍为 `deep_ep`；不要与其他提供 `deep_ep` 的包装在同一环境。

## 前置条件

- Linux、Python ≥3.11、C++17 编译器、CMake ≥3.20，以及 Make 或 Ninja；构建工具须在 PATH 中。
- CANN 9.2.0（原资格基线）或用户确认支持的9.3.0，与配套驱动/框架，目标Ascend950。9.3.0的本次timer验证范围见[LL阶段打点](../tests/ll_stage_timer/README.md)。
- 由本仓子模块构建或使用外部固定提交的 SHMEM SDK（见 `dependencies.lock.json`），安装根包含 `include/`、
  `lib/libshmem.so` 以及 `src/device`、`src/device_simt` 等 Device 头文件。
- 普通（HT）dispatch/combine 依赖 SHMEM 的 shared-jetty 支持，包括 Host 配置接口和 Device 共享队列状态；使用锁文件固定的 `cann/shmem` dev 提交，不直接跟随分支最新版本。
- CANN 安装中须包含 `tools/bisheng_compiler/bin/bisheng`，Device 使用 `-xasc` 编译。
- 普通 CPU 测试仅需 `requirements-dev.txt`；wheel 构建依赖固定在 `pyproject.toml`。
- wheel 构建不自动下载/编译外部 SDK，不将驱动/CANN/SHMEM 打入 wheel。SDK 及其传递依赖必须由部署环境提供。

## 可选实验组件

MegaMoE 的构建和多机测试入口位于 `experiments/megamoe/scripts/`，
命令、依赖和环境配置见 [MegaMoE 使用说明](../experiments/megamoe/README.md#2-编译安装)。
EpGmmFused 通过 `DEEPEP_BUILD_EP_GMM_FUSED=ON` 构建 AllToAllV → GMM 和 GMM → AllToAllV，
需要单独准备 CatCCOS、其 CATLASS 依赖及 SHMEM SDK，见 [EpGmmFused 使用说明](../experiments/ep_gmm_fused/README.md#构建)。
基础包使用下面的通用构建流程，两个实验组件默认均关闭。

## 分步骤构建（外部 SDK、打包或 CI）

先激活已安装匹配 Torch/torch_npu 的 Python 环境。以下命令在仓库根目录执行，使用仓内固定版本依赖：

```bash
source /path/to/ascend-toolkit/set_env.sh
git submodule update --init --recursive
unset SHMEM_ROOT CATLASS_ROOT  # 使用仓内依赖；外部 SDK 用户见下文。
python scripts/build_shmem.py
export DEEPEP_NPU_ARCH=Ascend950
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"

python -m pip install -r requirements-build.txt
```

主构建默认查找 `third-party/shmem/install/shmem`；MegaMoE 启用时默认查找 `third-party/catlass`。
准备脚本默认要求 SHMEM 源码干净且提交与锁文件一致；已有匹配 SDK 经完整身份校验后复用，身份不一致时要求显式 `--rebuild`。不再应用或接受 ROCE QP 扩容补丁，旧扩容 SDK 不能复用；更新源码不会自动删除或重建现有 SDK 产物。
旧 QP 扩容 SDK 的精确撤销、Git 更新及显式重建步骤见 [原生 SHMEM 迁移](NATIVE_SHMEM_MIGRATION.md)。
更换 SDK 前停止使用它的作业，之后重建 wheel。完整边界及离线依赖见 [第三方依赖](../third-party/README.md)。

已有外部 SDK 时，加载 CANN 环境、设置 `DEEPEP_NPU_ARCH=Ascend950` 并安装 `requirements-build.txt` 后，跳过子模块准备和 SHMEM 构建，改用：

```bash
export SHMEM_ROOT=/path/to/pinned/shmem-sdk
source "$SHMEM_ROOT/set_env.sh"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
# 外部 SDK 必须已由匹配构建流程生成完整身份；只验证，不补写或改写 manifest。
python scripts/record_shmem_sdk.py --sdk "$SHMEM_ROOT" --check
# 启用 MegaMoE 时显式设置 CATLASS_ROOT=/path/to/pinned/catlass；CATLASS已移出子模块。
```

sdist 不携带子模块，解包后构建需显式设置 SHMEM_ROOT，以及启用 MegaMoE 时所需的 CATLASS_ROOT；EpGmmFused 的外部源码路径按其构建说明设置。
下方自动化验证入口会将原 checkout 的默认依赖路径固定为绝对路径，确保 sdist 重建使用相同 SDK。

版本声明由 SDK 提供方负责，脚本校验固定公共头文件、记录头文件及库的校验和；不把“头文件一致”等同于“二进制来源可信”。
基础构建对 SDK 身份或校验和不匹配发出警告，并记录实际头文件和库的校验和；缺少必需文件仍会失败。
MegaMoE 构建及仓内 SDK 复用保持严格校验，SDK 必须与锁文件匹配，不能修改 manifest 绕过检查。

`setup.py` 元数据查询不需要 SDK/框架。实际构建只经 `build_support.py` 调用根 CMake。
不同工具链、SDK、Python 环境和构建模式使用不同指纹目录；源码哈希仅用于标识产物，
修改 Python、测试或单个 kernel 不切换整个构建目录。`DEEPEP_BUILD_MEGAMOE` 默认 OFF；
ON 的构建与使用方式见 [MegaMoE 文档](../experiments/megamoe/README.md)。
安装产物没有构建机绝对 RPATH，使用部署环境的 SDK 路径；加载时核对实际 SHMEM 二进制身份。

## 编译与安装

Host 编译器优先使用 `CXX` 指定的可执行文件；未设置时依次查找 PATH 中的
`c++`、`g++`、`clang++`、`bisheng++`、`bisheng`，再查找 CANN 的
`tools/bisheng_compiler/bin`。自动探测会跳过无法执行 `--version` 的候选；
显式指定的 `CXX` 无效则报错。保留 `c++ -> ccache` 等符号链接的调用名称，
不要把 `CXX` 设成裸 `ccache`。编译器是否满足 C++17 和链接要求仍由 CMake/实际构建验证。
日志会输出选中的 Host 编译器路径，Device kernel 继续使用 CANN Bisheng。

单机编译并安装仍使用：

```bash
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

这条命令会编译 Host 和 Device、生成 wheel 并安装到当前 Python 环境。
`requirements-build.txt` 只安装构建依赖，不能替代上述命令；修改 kernel 或 Python 包代码后重新执行上述命令。
只改 `tests/` 或文档时，更新共享源码即可，无需重新编译安装。

多机共享目录只需在一台机器生成 wheel，再在各节点的 Python 环境安装同一个文件：

```bash
python -m pip wheel -v --no-build-isolation --no-deps . -w /path/to/shared/wheels
python -m pip install --force-reinstall --no-deps /path/to/shared/wheels/实际文件名.whl
```

`pip wheel` 只编译打包，不安装。第一条仅执行一次，第二条在每个独立 Python 环境执行；
使用本次生成的实际文件名，不要硬编码旧版本，也不要在多台机器同时编译同一共享 checkout。
每台运行节点都需要加载匹配的 CANN/SHMEM 环境并设置运行时共享 endpoint，详见
[单机/多机测试](../tests/README.md)。

若需要同时交付源码包与 wheel，可在安装构建依赖后执行
`python -m build --sdist --wheel --no-isolation`，产物位于 `dist/`；该命令同样不会安装。

以下选项同时作用于 Host 和 Device；默认 Release、4 个编译任务：

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=8
export DEEPEP_BUILD_TYPE=RelWithDebInfo  # Release / RelWithDebInfo / Debug
```

支持 Ninja、Unix Makefiles 等单配置生成器。不同构建模式保留各自的缓存。
并行度改变不会切换缓存；内外层并行任务可能重叠，内存不足时降低并行度。
增量复用要求保留同一 checkout 的 `build/`；从新的源码包目录构建会使用新的缓存。

工程分工：

- `build_support.py`：环境校验、工具链缓存、调用 CMake、写入 `_build_info.json`。
- 根 `CMakeLists.txt`：Host 封装、Python 扩展 `_C.so`、能力声明、链接及安装。
- `cmake/Device.cmake`：每次进入独立 Device 构建，由子工程判断是否需要重新编译。
- `cmake/device/CMakeLists.txt`：Bisheng 源文件列表及编译选项，生成 `libdeepep_kernels.so`。
- `cmake/Dependencies.cmake`：Host/Device 共用的 CANN 和 SHMEM 库定位。

新增算子涉及的源码注册和检查项见[开发指南](development.md)。
编译器依赖信息负责追踪公共头文件；无改动、仅改 Python/测试时，不重新编译原生目标。
修改一个 kernel 时仅编译受影响的对象并重新链接，修改公共头文件时重编依赖它的对象。

wheel 中 `_C.so` 和 `libdeepep_kernels.so` 放在同一目录，通过 `$ORIGIN` 查找。
安装阶段会清除 setuptools 构建目录里遗留的旧 `libdeepep_notify.so`；包校验拒绝混入旧库。
CPU 构建预检和打包回归位于 `tests/runtime/test_build.py`，检查配置、缓存键、构建命令与产物要求；
这些测试不执行真实 Bisheng 编译，也不替代 Device 增量构建和设备验收。

## 运行期 JIT（仅 LL dispatch）

`DEEPEP_LL_JIT=ON` 时，LL dispatch 入口不再编入 `libdeepep_kernels.so`，改由 Host 首次调用时用
DeepJIT 现场编译；combine 仍走 Ahead-of-Time。两条路径编译同一个
`csrc/kernels/dispatch/low_latency/dispatch_entry.h`，通信协议、tiling 与 ABI 不变。

- 构建期需要 `torch`/`torch_npu`（DeepJIT 的 Ascend kernel 句柄使用 torch_npu 的 stream 类型），
  Host 模块会链接 `torch_npu`；wheel 额外携带 `deep_ep/_kernel_src/csrc` 供运行期编译使用。
- 运行期仍需 `ASCEND_HOME_PATH`（含 Bisheng）与 SHMEM SDK 头文件。SDK 根在构建期记录；
  部署位置变化时用 `DEEPEP_LL_JIT_SHMEM_ROOT` 指向提供 `lib/libshmem.so` 的安装目录。
- 首次调用交给 DeepJIT 的源码是一小段宏定义加一行
  `#include <kernels/dispatch/low_latency/dispatch_entry.h>`：宏把 B/H/K/E、输入模式、
  E4M3 与 dedup 决策以编译期常量写进翻译单元，因此每个不同 case 各有一份 kernel，
  缓存键也随之不同（磁盘目录名带 `b.._h.._k.._e.._n<cores>_w<world>..` 等字段；每 rank 容量只通过
  dedup 位体现，容量不同但决策相同的两次调用共用同一份 kernel）。JIT 的 kernel 完全不读 tiling：
  Host 不再为 dispatch 构建或上传 32B tiling（`native.low_latency_jit` 为真时 Python 跳过该步），
  启动身份只来自缓存键。DeepJIT 的 include 解析器拒绝
  引号形式的 include，而入口以下的头文件继续使用与 AOT 相同的相对引号 include；kernel 与
  SHMEM 头文件树的内容哈希仍折进缓存签名，改任何头文件都会使全部 case 的磁盘缓存失效。
- 形状静态化只作用于 JIT：AOT 的同一入口仍按 tiling 在“输入元素类 × group”的内联实例间选择，
  tiling、ABI 与通信协议不变。JIT 下 tiling 的每个字段都是编译期常量（`expertTokenNumsType`
  取协议常量 `kLlExpertTokenNumsType`，每 rank 容量只通过 dedup 位体现），world 与 active core
  数也一并烘进 kernel（`common/ll_jit_spec.h` 的 `DEEPEP_LL_LAUNCH_WORLD/CORES`）。仍为运行期
  值的只有 `CommArgs` 里的 rank、peer 映射、ffts 配置与 RDMA 能力。
- 编译缓存与 toolkit 兼容目录共用同一个根：默认 `$HOME/.dj`，
  用 `DEEPEP_JIT_CACHE_DIR` 改到共享目录（`:` 分隔，第一项可写、其余只读回退），
  多 rank 首次调用只需一次真实编译。
- 其它可调项：`DEEPEP_LL_JIT_SIGNATURE`（追加进缓存键）、`DEEPEP_LL_JIT_DEFINES`
  （`;` 分隔的额外 `-D`）、`DEEPEP_LL_JIT_INCLUDE_DIRS`（`:` 分隔的额外 `-I`）、
  `DEEPEP_LL_JIT_SHMEM_ROOT`（SHMEM 头文件根，见上）。
- 兼容性：CANN 把 `bisheng` 放在 `tools/bisheng_compiler/bin` 时，运行期会在
  `DEEPEP_JIT_CACHE_DIR` 的 `toolkit/` 下建立 `bin/bisheng`、`bin/ld.lld` 与
  `asc/include/adv_api` 符号链接。DeepJIT 固定探测 `aarch64-linux/asc/include/adv_api`
  这个名字，兼容根同时提供宿主架构（如 `x86_64-linux`）与 `aarch64-linux` 两套目录，
  因此 x86_64 主机也能运行期编译，其余头文件路径按 `uname -m` 选择。

两种模式的 kernel 名同为 `dispatch_server_dedup_kernel`，profiler 与既有诊断脚本无需区分。
JIT 的冷/热编译、多 rank 并发首编与精度/性能对比需在真机验证，CPU 测试不覆盖该路径。

## 自动化构建、隔离安装与验证

在已准备好匹配 torch/torch_npu、pytest 与构建依赖的环境中运行：

```bash
python scripts/validate_package.py --output /path/to/new-validation-directory \
  --system-site-packages
```

该命令创建新 venv，在其中自动安装固定的构建/测试依赖，运行源码测试，构建 wheel/sdist、从 sdist 再构建 wheel、检查内容与元数据，安装后在源码目录外运行测试。
无网络环境使用 `--wheelhouse /path/to/wheels`，只从预下载目录安装；缺包明确失败。
`--system-site-packages` **明确复用已有框架依赖**；不是宣称完全隔离所有 Python 依赖。
省略此选项创建不继承系统包的环境，需要另行准备框架/测试依赖，否则验证失败。
不会修改调用方 Python 环境，也不会默认占用设备。输出目录必须全新；失败日志/产物保留用于定位。

设备资源已由作业调度方分配后，可追加真实运行时验证：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export DEEPEP_SHMEM_ENDPOINT=tcp://127.0.0.1:19091
python scripts/validate_package.py --output /path/to/another-new-directory \
  --system-site-packages --runtime-ranks 2
```

设备编号和端口为示例，必须替换为本作业已预留资源。不要复用已有任务的通信端口。
支持显式选择 EP2/4/8 的资源测试；结果仅覆盖实际执行的规模。
每个 rank 的成功结果、JUnit XML 和 `validation.json` 一并保留；失败不会写成功摘要。

`--runtime-ranks` 只验证资源、事件与生命周期，不是 dispatch/combine 正确性或带宽。安装后分别运行 HT dispatch、HT combine 和 LL 系统测试，命令集中在[测试入口](../tests/README.md)。普通 pytest 和默认构建安装检查不自动执行设备 ST。

`--ep-roundtrip npu-stub` 可在构建安装后追加双卡 NPU Tensor/HCCL 参考闭环；使用同一组两张已分配设备。该入口不是 DeepEP 原生通信验收。`--ep-roundtrip native` 要求 V1 Buffer 通信方法，不能用于当前 V2 算子，详细边界见[参考闭环说明](../tests/README.md#cpu-与-npu-参考闭环)。

## CI 与产物

当前仓库未配置自动 CI 流水线。请手动运行 pre-commit，并按变更范围运行 CPU API、参考、运行时和构建预检。CPU fault-injection 使用 Linux/g++，不等于 CANN ABI 或 NPU 测试。检查命令见[开发指南](development.md#提交前检查与-ci)。

在具备 CANN、SHMEM、框架与已分配 NPU 的环境执行本页验证命令。保留 `source-tests.xml`、`installed-tests.xml`、`validation.json` 和逐 rank 输出，wheel/sdist 作为独立产物归档。安装成功、CPU 通过和设备算子通过分别记录，不互相替代。

SDK 身份包含 CANN 版本/时间戳、目标、源码修改指纹及可选的已知 RDMA 能力。未登记或身份不完整的 SDK 应由 `scripts/build_shmem.py` 构建和记录；仅提供 revision 无法证明这些信息，登记器会在写入前拒绝。显式许可并记录的其他本地源码修改仍受支持。
