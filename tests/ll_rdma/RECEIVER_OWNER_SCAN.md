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
