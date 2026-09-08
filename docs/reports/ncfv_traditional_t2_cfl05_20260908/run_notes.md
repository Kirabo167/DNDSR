# 传统三阶 NCFV t=2：运行设置与复现

本轮实际调用现有 `Solver.Run()`，不修改生产求解器、时间推进、重构矩阵、通量或对偶控制体生成代码。
仅新增独立诊断入口、四份JSON配置及后处理分析。

- 传统模式：`TraditionalQuadrature`，保留全部9个二次重构系数，使用论文式(3-34)的分方向半跨度归一化。
- `quadratureOrder=4`：每个微三角面6点，每个微四面体14点；初始化均值也使用体积分。
- 通量：沿用此前精度测试的 `Roe_M2`（当前库实际为局部 Lax–Friedrichs/Rusanov 型标量耗散）。
- 时间推进：生产SSPRK(3,3)，最终物理时间2，CFL=0.5，关闭局部伪时间步。
- dt下限1e-30、上限1e30，正常步长不受固定上限限制；最后一步截短到t=2。
- 每个物理步的三个RK阶段共用起始状态确定的dt；日志中的dt是第三阶段重新估计的候选值，不是完整实际步长历史。
- 等熵涡：β=5、γ=1.4、初始中心(5,5)，背景速度(1,1,0)，周期盒[10,10,4]；终点解析中心(7,7)。
- 四套网格均从初始场开始，无黏性、无限制器、不读取历史重启。
- MPI进程数：iv10=2、iv20=4、iv40=8、iv80=32；`OMP_NUM_THREADS=1`，`--bind-to none`。

误差使用恢复的格点守恒量；压力由完整守恒点值计算，而非强行令数值压力等于密度的γ次方。
L1/L2以各原始节点的部分对偶体积加权，周期节点不重复计入完整控制体体积。
VTK场来自生产程序的控制体均值，精度表使用`points_final.rank*.csv`中的恢复点值；二者不能混用。

## 构建与运行

从项目根目录构建当前NCFV库，并编译独立入口：

```bash
CCACHE_DISABLE=1 cmake --build build -t NCFV -j6
venv/bin/python cases/NCFV/diagnostics/compile_reconstruction_probe.py \
  --source cases/NCFV/diagnostics/traditional_transient_accuracy_probe.cpp \
  --output /tmp/ncfv-traditional-t2.5LVg8o/traditional_transient_accuracy_probe
```

从build目录运行各套配置，例如iv80：

```bash
OMP_NUM_THREADS=1 mpirun --bind-to none -np 32 \
  /tmp/ncfv-traditional-t2.5LVg8o/traditional_transient_accuracy_probe \
  ../cases/NCFV/traditional_t2_cfl05_20260908/iv80.json
```

诊断入口拒绝覆盖已存在的算例输出目录。复跑需在新配置中修改所有输出前缀。
最终输出包含完整恢复点值、生产节点数据、VTK及H5重启；`accuracy.json`仅在t=2完成并验证最终快照后写入。

汇总入口：`cases/NCFV/diagnostics/analyze_traditional_t2.py`；只有四套都到达t=2后才生成最终四网格报告。

## 完成记录

四套全部完成t=2，实际步数依次为60、124、266、548，最终VTK分片及H5重启均通过文件完整性检查。
独立误差复算、周期副本、网格数组校验与总对偶体积检查均通过。
生产源码及链接库在计算前后的SHA256一致，详见metrics.json中的source_and_library_sha256。
本轮诊断可执行文件SHA256：`6b3703e5d7ef9d42f05b0814e6f9acd8037c386ea82e65f501365e04565fe751`。
