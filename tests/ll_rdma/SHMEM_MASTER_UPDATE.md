# Metadata 分支 SHMEM 版本对照（2026-11-02）

本次对照使用同一份 R 核接管和有效槽扫描实现。A 锁定 `4c671684`，B 锁定
`8e3fc610`。B 的父提交即 A，两棵源码树只在 SHMEM gitlink 和依赖锁的 revision 上有差异。
各节点按当前 checkout 的依赖锁校验 SDK 源码、六个公开头指纹、安装库、wheel 和 HNS1825 PI 身份。
此实验可能同时测到该版本范围内多项上游改动的影响。

在每个节点将本地交付的增量 bundle 放在实验目录之外，使用一个全新 clone。两节点分别设置
NODE_RANK=0、1，MASTER_ADDR 填首机地址。以下示例沿用 M4、normal、seed0 和默认 scalar
计数；原部署使用 histogram 时，两轮都把 COUNT_METHOD 设为 histogram。

```bash
git clone --single-branch --branch kernel/ll-ndr-metadata-cores \
  https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-shmem-ab && cd DeepEP-LL-shmem-ab
```

```bash
BUNDLE=/absolute/path/metadata-shmem-ab.bundle
BRANCH=kernel/ll-ndr-metadata-cores
export NODE_RANK=0 MASTER_ADDR=192.0.2.1 COUNT_METHOD=scalar TAG=shmem-ab-s0-001

sdk_run() {
    git switch -C "$BRANCH" "$1" &&
    python3 tests/ll_rdma/metadata_opt.py "$COUNT_METHOD" "$NODE_RANK" "$MASTER_ADDR" "$2" \
      --mode perf --num-nodes 2 --tokens 1 8 32 256 --capacity 256 --hidden 3584 \
      --topk 8 --experts 384 --dtype fp8 --route normal --seed 0 --rebuild-shmem
}

if git bundle verify "$BUNDLE" &&
   git fetch "$BUNDLE" "refs/heads/$BRANCH:refs/remotes/sdk-ab/rollback" &&
   B=$(git rev-parse refs/remotes/sdk-ab/rollback) &&
   A=$(git rev-parse "$B^") &&
   sdk_run "$A" "${TAG}-4c671684" &&
   mkdir ".cache/${TAG}-first-sdk" &&
   mv -- third-party/shmem/build third-party/shmem/install ".cache/${TAG}-first-sdk/" &&
   sdk_run "$B" "${TAG}-8e3fc610"; then
    echo "Both full-wheel OFF runs passed"
else
    echo "Stop here and keep this terminal, logs and results"
fi
```

A 完成并退出后在新实验 clone 内保存其生成的 build/install，使 B 从空的 SDK 编译和安装目录开始。
两轮均由 one_click 限定初始化 third-party/shmem，重建全 rank wheel。每轮在目标机核验正确性并采
dispatch 和 BF16 combine。可用另一新 clone、新 TAG 反序复测以检查顺序效应。

在节点 0 的 B 版本汇总，完整采样留在目标机。

```bash
python3 tests/ll_rdma/metadata_report.py --shmem-ab \
  "${TAG}-4c671684-perf" "${TAG}-8e3fc610-perf" \
  --output "results/${TAG}-return_report.md"
```

只有显式的 `--shmem-ab` 模式接受这一对 SDK。该模式核验 Git 树、每版 gitlink 和锁、
采样时源码清洁状态、节点 0 SDK 编译配置（含 CANN 时间戳和 PI）、LL 配置及路由 hash。
普通版本对照仍要求相同 SDK。回传一个不超过 60000 UTF-8 字节的报告。

## SHMEM 官方 master 修复更新历史（2026-10-30）

本次仅更新用户指定的两个分支：`kernel/ll-ndr-metadata-cores`、`kernel/ll-ndr-ab-per-core-qp`。
二者统一从`8e3fc610f91ae9e5e8cf13748cfb74826b272c2a`更新到实时核实的官方master：

`4c671684e1698aec199619850a3120bfdf85c95e`（上游提交时间2026-10-30 10:59:03 +08:00）。

`third-party/shmem` gitlink与`dependencies.lock.json`同步锁定这个提交；构建时不追逐移动master。
六个锁定公开头的LF规范化SHA256已重新核对，本次均未变化。SHMEM工作树不加私有补丁。
其他历史A/B、小B、hns1825和独立LL工作区不更新；旧性能结果也不改写。

## 主要上游变更

- [e658bbce：HNS_1825 WQE ULP修复](https://gitcode.com/cann/shmem/commit/e658bbcec97ec3e653da8bc3d9f0d602a958b690)：任务段`ulp`使用`lkey & 0xffffff00U`，替代旧`lkey & 0xffffU`；低8位为ULP flags。
- [9c7770c3：UDMA队列状态发布修复](https://gitcode.com/cann/shmem/commit/9c7770c34c55da9f88ae524dd77bc1dd37ef9d4e)：doorbell前发布队列生产索引/CQE目标，并调整完成索引刷新及聚合提交顺序。
- 同批包含性能示例FFTS地址获取时序修复、新增SIMT UDMA接口及对应测试/构建功能检测。它们不等同于本项目已经启用新SIMT路径。

项目使用的`aclshmemx_roce_qp_put_nbi/quiet`公开接口及RDMA封装未变化，不需要改LL调用。
kernel、wire、QP分工、采样协议及分支原实现身份保持各自原样；SDK revision和完整库/头指纹单独拒绝旧SDK或混装。
HNS_1825与严格整数PI-on-chip 0/1校验保留，不手改manifest或恢复私有SDK补丁。
新增`tests/runtime/test_shmem_pin.py`核对gitlink、锁定revision、已初始化源码及公开头指纹。

## 目标机更新与重编

本地只做源码/API检查和CPU回归，未执行CANN编译或NPU验证。
此次SDK源码确实变化，不能只重装wheel或继续使用旧SDK。各主机先结束使用该工程SDK的作业，加载已有CANN环境，然后在各自分支工程根执行。

使用现有NDR一键入口（例如2机，每机NODE_RANK分别为0和1、MASTER_ADDR相同；测试名为全新名称）：

```bash
if git pull --ff-only; then
    if bash tests/ll_rdma/one_click.sh "$NODE_RANK" "$MASTER_ADDR" shmem4c67168-perf-001 \
        --num-nodes 2 --mode perf --rebuild-shmem --tokens 8 --dtype bf16 fp8; then
        echo "PASS: rebuilt SDK/wheel and completed OFF dispatch/combine tests"
    else
        echo "FAILED: keep this terminal; inspect .logs/ndr-one-click"
    fi
else
    echo "Git update failed; keep the terminal and inspect the error"
fi
```

metadata分支沿用已选的`DEEPEP_LL_METADATA_CORES`（默认4）；每核QP基线保留原S/R分工。
一键入口显式使用`git submodule update --init --recursive --depth 1 -- third-party/shmem`，只拉取SHMEM及其必要依赖，禁止拉CATLASS。
显式`--rebuild-shmem`更新本工程SDK，随后重建完整wheel，所有rank一起更新。
已有`--skip-build`仅适用于本次新SDK和wheel已经成功构建并通过身份检查的后续测试，不能用于首次升级。
SDK重建失败不登记新身份；不要手动修改旧manifest绕过检查。不删除历史结果、日志或其他工程SDK。
