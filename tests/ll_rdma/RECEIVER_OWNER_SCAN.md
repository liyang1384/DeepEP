# M=4 接管分工与前缀后扫描逐版本实验

限定kernel/ll-ndr-metadata-cores；保留当前normal/hotspot生成器、M=4、S/R核数、SDK、
物理wire槽和q、count/prefix与输出顺序、1.25接管、完整flags判定和清flag依赖。
基线为59d008d；两个源码改动独立提交。本轮仅本地检查/提交，不推送、不SSH/CANN/NPU。

## 第一步：接管owner去除B低位耦合

`region=((source+destination)%EP)*localExperts+expert`，物理槽仍为`region*B+q`。
接管owner改为`(region+q)%R`。每region枚举`q=(owner-region) mod R, +R, ... <B`，
使用每owner的紧凑bitmap索引，顺序为region、q；不修改搬出owner或新增通知/屏障。
Host与Device按相同精确候选数核算位图；新分工可能使某核位图跨32B对齐边界，
因此Host采用最重核预算。没有新增缓冲对象，业务lane仍按原剩余UB规则、最多8个分配。
不能声称所有shape的lane数必然不变；接近UB边界的shape须重新核算。

默认EP32/B32/K8/E384/seed0/Rank0的CPU造数重建：有效接管数从每核0～22条变为7～14条，
原奇数owner共2条。生成器本身不改；均衡性改善不等于目标设备性能收益。
1.25仍由唯一接管owner读取并提交必要转发后发布；搬出可以先完成，清flag仍须完整Taken状态。
bitmap置位及未完成计数只处理本核所有权，零任务核与晚到包仍参与原循环。

第一步18项CPU检查通过：R=1～36的owner/index完整覆盖、部分region和尾q、原生成器分布，
真实count任务表、M配置，以及16/56/72核接收ON/OFF模型中的早搬出晚接管、晚到包、
重复扫描不重复转发、零任务和跨case复用。固定版本命令在两个提交完成后补充。

## 第二步：前缀后有效槽位图

前缀前继续使用第一步的候选枚举，bitmap的1代表已经接管。前缀到达后，在同一块
bitmap中原地构造`valid & ~seen`：每region直接按真实count枚举本owner的有效q，
不逐个访问所有无效q，也不读取它们的GM flags。转化后1代表有效且尚未接管；
接管完成清该bit，forwardLeft只统计剩余有效接管数。提前转发过的包不会重新加入。
此转换每核最多一次，代价为O(E+本核有效任务数+位图word数)，不能假定代价为零。

随后跳过零word并定位非零位，每批最多16个真实待处理槽；绕回不重复收集同一bit，
未ready保留bit继续轮询。用第一步紧凑索引的逆映射恢复原物理槽；无需新UB、任务队列、
lane或跨核通知。空扫描仍不逐轮读clock，reference schema5字段/包日志含义不变。
OutputBatch、Move、Inspect中的1.25发布和完整flags判定保持原样；搬出后未接管的任务
仍保留pending，只有完整Taken状态才能清flag。循环仍同时等待接管与搬出/清理完成。

新增CPU读访问检查：前缀加载后，任何接收flags读取必须属于真实有效槽；零任务不得有
后续flags读取。测试覆盖E384、B32、56核及43场景的晚到包、提前搬出/延迟接管、
前缀前已转发、跨word/扫描绕回和跨case复用。完整验证结果见下方。

第二步最终相关CPU回归419项通过（包含第一步检查，不累计相加），覆盖生产接收器、
owner/index双向映射、Host预算、count/任务表/CQ退休、reference解析和部署/报告入口。
未执行CANN编译、NPU正确性或性能测试；本次无SSH和推送。

## 三版固定源码对照

| 版 | 源码 | 改动 |
|---|---|---|
| base | 59d008dcd7122e400f0074ddf5ab760bccbe81dd | 本次改动前，包括此前meta任务表/offset优化 |
| owner | f8b9f2a19111f80d3dcbedee9bfca7fccd16e98a | 只改接管owner和对应索引/容量 |
| sparse | 本文所在第二提交，亦为交付bundle的分支头 | owner加前缀后有效槽扫描 |

本次没有推送。将交付的receiver-owner-sparse-20261102.bundle复制到每个目标主机，
再在全新独立clone内导入；不要重置现有生产或有未提交修改的工程。
bundle仅包含相对base的两个提交，需先从远端取得已有base。以下动作由用户在目标机执行。

```bash
git clone --single-branch --branch kernel/ll-ndr-metadata-cores \
  https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-receiver-ab-20261102 &&
cd DeepEP-LL-receiver-ab-20261102
```

确认进入新目录后，填写bundle的实际绝对路径：

```bash
BUNDLE=/absolute/path/receiver-owner-sparse-20261102.bundle
git bundle verify "$BUNDLE" &&
git fetch "$BUNDLE" \
  refs/heads/kernel/ll-ndr-metadata-cores:refs/remotes/receiver-local/kernel/ll-ndr-metadata-cores
SPARSE_REV=$(git rev-parse refs/remotes/receiver-local/kernel/ll-ndr-metadata-cores)
BASE_REV=59d008dcd7122e400f0074ddf5ab760bccbe81dd
OWNER_REV=f8b9f2a19111f80d3dcbedee9bfca7fccd16e98a
git log --oneline "$BASE_REV..$SPARSE_REV"
```

每个节点设置自己的NODE_RANK；其他参数、TAG和COUNT_METHOD必须全节点一致。
COUNT_METHOD沿用当前部署：原默认OFF填scalar；原来ON则三版全部填histogram。
metadata_opt入口强制M=4，普通normal造数和seed保持；不允许skip-build。

```bash
export NODE_RANK=0
export MASTER_ADDR=192.0.2.1
export COUNT_METHOD=scalar
export TEST_SEED=0
export TAG=receiver-s0-a1

receiver_case() {
    local rev="$1" name="$2" mode="$3"
    shift 3
    git diff --quiet && git diff --cached --quiet &&
    git switch -C kernel/ll-ndr-metadata-cores "$rev" &&
    python3 tests/ll_rdma/metadata_opt.py "$COUNT_METHOD" "$NODE_RANK" "$MASTER_ADDR" "$name" \
      --num-nodes 2 --mode "$mode" --tokens 1 8 32 256 --capacity 256 \
      --hidden 3584 --topk 8 --experts 384 --dtype fp8 --route normal --seed "$TEST_SEED" "$@"
}

if receiver_case "$BASE_REV" "${TAG}-base" perf --rebuild-shmem &&
   receiver_case "$OWNER_REV" "${TAG}-owner" perf &&
   receiver_case "$SPARSE_REV" "${TAG}-sparse" perf; then
    echo "PASS: three full OFF dispatch/combine runs"
else
    echo "FAILED: stop this sequence; retain terminal, logs and results"
fi
```

仅第一版在此新工程新编SDK，随后同一SDK完整重建wheel。只初始化third-party/shmem，
不拉CATLASS。失败不自动重试、不跨attempt拼结果。再次测试换TAG，反序运行三个版本；
同工程SDK未变时不再加rebuild-shmem。需要多seed时保持每组三版同seed。

阶段诊断独立使用新run名；覆盖默认tokens只选B32，仍保持完整正确性检查：

```bash
if receiver_case "$BASE_REV" "${TAG}-base-diag" reference --tokens 32 --reference-iters 1 &&
   receiver_case "$OWNER_REV" "${TAG}-owner-diag" reference --tokens 32 --reference-iters 1 &&
   receiver_case "$SPARSE_REV" "${TAG}-sparse-diag" reference --tokens 32 --reference-iters 1; then
    echo "PASS: compare per-core receiver loops, polling, first-ready and total durations"
else
    echo "FAILED: retain diagnostic evidence"
fi
```

OFF主指标看dispatch全rank尾50样本汇池均值、最慢rank均值和p99，combine用于回退检查。
reference只诊断dispatch；各核时段/父子累计不相加，不用ON内部时长冒充OFF收益。
source SHA与配置由原安装检查及全节点握手拒混；不要混装不同提交的wheel。

节点0将三版OFF汇为一个<=60000字节文件回传；完整raw/xlsx保留目标机：

```bash
mkdir -p results/receiver-report &&
python3 tests/ll_rdma/metadata_report.py \
  "${TAG}-base-perf" "${TAG}-owner-perf" "${TAG}-sparse-perf" \
  --output "results/receiver-report/${TAG}-return_report.md"
```

该汇总工具在三版均存在，不需要再次切换源码；相同shape/routehash/seed/SDK/采样条件
核对后只整理统计，不删除慢样本，不自动宣布性能改善。
