# SparseWorld + Radar M0：首版训练协议

日期：2026-09-18。当前目标是实现、验证并启动 H200 训练；正收益与顶会结论尚未得到实验支持。

## 源码与方法边界

- 主体：SparseWorld-TC，upstream `8ef331da491f0c2c317d3d6cf3e21a9c8a1576be`，https://github.com/MrPicklesGG/SparseWorld-TC 。
- 参考：Sparse4D-Radar (https://arxiv.org/abs/2607.04098)，官方仓库 https://github.com/Aiuan/Sparse4D-Radar 当前只有说明/资源，完整实现未公开。
- M0 是独立的 sparse-query 融合适配：局部 XY radar association、可学习偏移采样、特征编码、残差门控。论文的 BEV PointPillars 融合与当前点级实现不同，不声称官方复现。
- 仅向各 decoder layer 的当前时刻 query 注入 radar，再通过 SparseWorld 原有跨时刻注意力传播。保留相机 backbone、6 层 decoder、600 queries、8 帧历史/当前相机输入，以及 0/1/2/3 s 输出。
- M0 不加入 VCS：occupancy queries 尚无可信物体速度头，直接套用检测 anchor 速度会改变待验证问题。第二阶段 Doppler 约束 future evolution / uncertainty 仍为后续研究，尚未实现或验证。
- nuScenes 为传统 radar，不是 OmniHD 的 imaging radar；这里的“4D occupancy”指三维占用随时间的预测。

## 数据与因果边界

- nuScenes v1.0-trainval 官方 scene split；标签用 Occ3D 200×200×16、0.4 m、17 个 occupied 类 + free=17。
- 仅纳入完整具有 +6 keyframes 的 anchors，禁止复制末帧作为未来目标。全部 target tokens 留在同一场景；train/val scene 不重叠。
- radar：5 路传感器，各最多 5 sweeps，最大 age=0.5 s，只接受 timestamp <= 当前 LIDAR_TOP timestamp。最多4096 points，超限稳定优先保留最近回波。
- xyz、compensated velocity、LOS 都变换到当前 LIDAR_TOP 对应的 ego frame；velocity/LOS 只旋转不平移。
- 10维特征为 xyz、vx_comp、vy_comp、RCS、age、compensated radial velocity、LOS xy。radial 是补偿速度向传感器平面 LOS 的投影，不声称完整二维真实运动。
- 无邻域/空点云输出严格零增量；没有未来图像/radar作为输入。未来 ego pose 是 SparseWorld 的 trajectory-conditioning 输入，属于条件预测协议，不声称无条件未来自车轨迹预测。
- 标签镜像来源和校验必须记录；镜像 hash 校验不能写成与不可访问原始文件完成逐字节一致性验证。

## 公共代码修正（相机对照同样采用）

- 移除 import 时加载固定路径全量 nuScenes；按 data_root 懒加载。
- 统一标签轨迹变换到 LiDAR timestamp ego frame，避免 CAM_FRONT 时间与占用参考系差异。
- 批次按 horizon-major 排列，时序注意力先转为 batch-major，避免不同场景互相注意。
- camera sweep 收集不越过 scene 边界，不用负索引穿到数据末尾。
- PyTorch fallback 正确融合所有 pyramid levels；正式训练编译 H200 sm90 CUDA sampler。
- 输出目录可配置；保存 resolved config；非有限 decoder 训练输出立即失败。

## 初始训练与对照

- H200 / huayiming，GPU 仅在启动前验证空闲后指定 UUID。
- 8 帧 × 6相机，256×704；所有预测尺度保持上游配置。
- 官方 R50 backbone 初始化，SHA256：4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e。
- AdamW，lr=2e-4，weight_decay=.01，batch=1/GPU，累积8步形成 effective batch8；FP16 dynamic loss scale，clip norm35，70epochs，warmup4000 microsteps=500optimizer updates，seed0。此处是计划配置，实际以 resolved config 为准。
- 先用相同分辨率/完整结构和真实数据跑16次迭代、确认2次有效更新和checkpoint，再新启动完整 M0。
- `sw-camera-control.py` 保持训练数据、初始化、优化、预测尺度和修复相同，禁用 radar branch。相机对照首版仅准备配置，后续另行运行；不能从单个M0 loss推断正收益。

## 研究通过条件与停止条件

1. 上线门槛：全量路径/划分检查、传感器时间与坐标测试、空radar等价性、batch隔离、真实数据正反向、雷达梯度、checkpoint以及稳定日志。
2. 第一阶段有效性：matched camera/R+C 至少3 seeds，报告0/1/2/3s mIoU/IoU、动态类别和场景bootstrap区间；不仅看当前帧，未来平均与3s不得系统性退化。额外跑radar-zero/velocity-shuffle定位几何与运动贡献。
3. 不足以证实：单seed、小样本、训练loss下降、静态类主导提升、只验证当前占用。
4. 第二阶段才研究基于Doppler可观测性的未来query演化与不确定性；先比较固定velocity propagation和无motion控制，再增加复杂模型。
5. 检测到未来输入、坐标错位、nonfinite、缺文件、无radar梯度时停止该实验并修复。不得以OpenOccupancy标签、mini数据或缩小任务替代正式M0并报告完成。

## 运行入口

- `tools/prepare_m0_data.py`：从官方 nuScenes JSON 生成元数据；`tools/preflight_m0.py`：全量文件/目标/划分检查。
- `tools/check_sampler_cuda.py`：H200 CUDA sampler 正反向对照 PyTorch；`pytest -q tests`：融合/时序/坐标/因果测试。
- `bash tools/run_h200.sh configs/sw-radar-m0-smoke.py`：16步真实预检。
- `bash tools/run_h200.sh configs/sw-radar-m0.py`：正式训练，工作目录非空时禁止覆盖，须人工或代理检查后显式恢复。
- `python val.py --config configs/sw-radar-m0.py --weights CHECKPOINT --out METRICS_JSON`：固定完整horizon验证，按真实sample.next解析目标token，禁止在已筛选索引中用i+offset找目标。
- 首epoch完成后应评估并根据动态类别与未来指标决定继续；当前70epoch仅为最大训练配置，不代表已完成或已取得正收益。
