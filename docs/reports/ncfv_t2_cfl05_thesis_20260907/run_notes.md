# t=2、统一 CFL 的运行记录

## 固定设置

- NCFV EfficientDifferential；论文式（3-34）分方向半跨度归一化。
- CFL=0.5；全域物理步长；终止时间2；SSPRK3。
- `minimumTimeStep=1e-30`、`maximumTimeStep=1e30`；不再让0.01的上限限制粗网格。
- 限制器、黏性关闭；Roe_M2；四套物理、重构、积分和时间设置一致。
- 仅最后一步为准确到达 t=2 而截短。相同 CFL 不要求相同 dt。
- 每进程 OMP_NUM_THREADS=1；iv10/20/40/80 的 MPI 进程数分别为2/4/8/16。

独立诊断程序直接调用现有 Solver.Initialize 和 Solver.Run，未重写时间推进、通量或重构。
在起始状态读取一次 CFL 物理步长；初始和最终额外快照通过现有重构/点值恢复接口生成。
额外诊断只读物理状态，不替换重构矩阵，不修改求解器库。

## CFL 计算口径

该无黏性算例的边谱半径为

\[
\lambda_e=(|\boldsymbol v_e\cdot\widehat{\boldsymbol n}_e|+a_e)|S_e|,
\qquad
\Delta t_n=\min\left(2-t_n,\min_i\frac{0.5V_i}{\sum_e\lambda_{ie}}\right).
\]

边状态使用两端恢复守恒点值的平均，法向与面积使用已有对偶宏面几何。
周期控制体先合并各份额的体积和谱半径，然后选全域最小时间步。
所有 SSPRK3 阶段共用第一阶段开始前选出的物理步长。
原生产日志的 dt[min,max] 是第三阶段更新后的候选值，不能直接当作逐步实际时间增量累加。
实际累计时间由 Solver.SimulationTime 给出，平均物理步长用2/总步数计算。

## 编译标识

- 诊断源码：`cases/NCFV/diagnostics/transient_accuracy_probe.cpp`。
- 分析源码：`cases/NCFV/diagnostics/analyze_t2_common_cfl.py`。
- 可执行文件：`/tmp/ncfv-t2-cfl05.fmHJPk/transient_accuracy_probe`。
- 可执行文件 SHA-256：`58c640dfe7ad0693c5e015805002a3135ae10a1fb7a74edd32ac0cdf9128ad82`。
- `build/src/NCFV/libncfv.a` SHA-256：`c6a7448d3b503e13f85c3ba628d8f76ba7f6b9e39f5d13b8e0344efd3c29e481`。
- `src/NCFV/NCFVSolver.cpp` SHA-256：`dce36ee25589001f934f881d01f5b73d0b67f7f3d70333e89ff86b4cb6c77062`。
- `src/NCFV/NCFVSpatial.cpp` SHA-256：`dc084b75a54f9658cfef1c689813bb3d24e9281fe37e8c9db3ad798cd680761d`。

## 输入和输出

配置：`cases/NCFV/t2_cfl05_thesis_20260907/iv{10,20,40,80}.json`。
数据：`data/out/NCFV/t2_cfl05_thesis_20260907/iv{10,20,40,80}/`。
日志：上述数据根目录的 `iv{10,20,40,80}.log`。

每套网格仅在 Solver.Run 返回且最终时间通过检查后，才写入 `accuracy.json`。
收敛分析必须同时存在四份完成数据，不接受中间结果或历史结果代替。
逐节点恢复状态保存在 `points_initial.rank*.csv`、`points_final.rank*.csv`，
其中包含完整五个守恒点值及由它们计算的压力。
精度统计不使用压力的等熵强制关系，也不使用控制体均值对应的压力代替节点值。

现有生产 VTK 的量来自控制体均值；本报告精度表及误差图使用额外快照中的恢复点值。
最终 H5 重启和原生产诊断输出均保留，没有覆盖此前计算。
