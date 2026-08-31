<!--
文件说明：DNDSR 三维 ACM 求解器族的总体分阶段实施计划。
修改人：Runzhi Ma
修改日期：2026-08-31
-->

# DNDSR 三维 ACM 预处理方法实现计划

## 1. 计划目标

在不改变现有可压缩 `euler3D`、`eulerSA3D` 和 `euler2EQ3D` 数值行为的前提下，为 DNDSR 新增三维人工可压缩性方法（Artificial Compressibility Method，ACM）求解器族，覆盖：

- 常密度、变密度不可压缩流；
- 层流、Spalart–Allmaras（SA）和两方程 RANS；
- 稳态伪时间推进和非稳态双时间推进；
- MPI 域分解、OpenMP 节点内并行以及 MPI + OpenMP 混合并行；
- 显式推进、矩阵自由 GMRES 和可选的并行 LU-SGS/SGS 预条件；
- 物理压力 `p`（方案 A）和固定 β²下的缩放压力 `p/β²`（方案 B）。

本计划的首要验收目标不是一次性覆盖全部组合，而是先建立一个数学正确、并行结果可复现的常密度层流基线，再逐项扩展。

## 2. 推导依据和公式冻结

技术依据为《ACM 预处理方法完整推导报告》（2026-06-30）。实现前必须把下列关系固化为符号测试和单元测试，不能只依赖手工抄写公式。

### 2.1 规范变量

常密度：

\[
\widetilde U_A=(u,v,w,p)^T,
\qquad
\widetilde U_B=(u,v,w,p/\beta^2)^T.
\]

变密度：

\[
\widetilde U_A=(\rho,\rho u,\rho v,\rho w,p)^T,
\qquad
\widetilde U_B=(\rho,\rho u,\rho v,\rho w,p/\beta^2)^T.
\]

湍流输运变量一律追加在压力变量之后：常密度存储 `νTilde` 或 `(k, ω/ε)`，变密度存储 `ρνTilde` 或 `(ρk, ρω/ρε)`。禁止把新增变量插入速度/动量和压力之间。

### 2.2 方案变换

令 `S` 仅将压力分量缩放为 `1/β²`。第 8.3 节给出的精确变换作为实现规范：

\[
\widetilde U_B=S\widetilde U_A,
\quad F_B=F_A,
\quad A_{n,B}=A_{n,A}S^{-1},
\quad \widetilde\Gamma_B=\widetilde\Gamma_A S^{-1},
\]

\[
\widetilde A_{n,B}
=S\widetilde A_{n,A}S^{-1},
\qquad
\widetilde A_n=\widetilde\Gamma^{-1}A_n.
\]

其中物理通量保持不变；通量 Jacobian 和预处理矩阵是右乘 `S⁻¹`，只有预处理 Jacobian 是相似变换。实现和代码审查中不得混用这三种变换。

### 2.3 常密度通量和关键勘误

局部法向坐标中的物理通量为：

\[
F_n=(q_n^2+p/\rho,\ q_nq_{t1},\ q_nq_{t2},\ q_n)^T.
\]

其正确 Jacobian 第一列必须包含：

\[
(2q_n,\ q_{t1},\ q_{t2},\ 1)^T.
\]

必须通过测试防止把守恒形式 Jacobian 错写为非守恒形式系数矩阵，即防止 `2qn → qn`、`qt1/qt2 → 0`。

### 2.4 Turkel 参数约定

代码只暴露 `alpha`，内部统一定义 `turkelGamma = alpha + 1`，不得复用可压缩气体中的比热比 `gamma` 命名。正确的常密度伪声波特征值以报告第 5.3 节为准：

\[
\lambda_{\pm}=\frac{(1-\alpha)q_n
\pm\sqrt{(1-\alpha)^2q_n^2+4\beta^2/\rho}}{2}.
\]

关键极限：

- `alpha = -1`：Euler-like；
- `alpha = 0`：标准 Chorin ACM；
- `alpha = 1`：速度无关的 Turkel 条件数优化形式。

报告不同章节存在 `gamma` 记号切换及部分中间公式不一致。因此首个版本固定 `alpha = 0`；一般 `alpha` 和变密度全预处理只有在 SymPy 参考脚本与有限差分 Jacobian 同时通过后才能启用。

### 2.5 方案选择

- 默认使用方案 A，直接存储物理压力 `p`。
- 方案 B 仅允许固定 β²或伪时间不变的空间分布 β²。
- 若 β²随伪时间变化，配置阶段必须拒绝方案 B，因为变量缩放会产生额外伪源项。
- 方案 A/B 的停止条件使用物理残差下降量，不能用相同 LU-SGS 扫描次数代替收敛等价性。

## 3. 总体软件架构

### 3.1 不直接修改现有 Euler 模型

当前 `EulerEvaluator` 普遍假设：密度位于 0，三维动量位于 1–3，总能量位于 4，湍流变量位于 5 以后。常密度 ACM 的 `[u,v,w,p]` 与该布局不兼容，强行加入大量 `if constexpr` 会使通量、周期变换、正性限制、边界条件和输出难以验证。

因此新增独立 `src/ACM/` 模块和可执行程序：

- `acm3D`：常密度层流；
- `acmSA3D`：常密度 SA；
- `acm2EQ3D`：常密度两方程 RANS；
- `acmVD3D`：变密度层流；
- `acmVDSA3D`：变密度 SA；
- `acmVD2EQ3D`：变密度两方程 RANS。

现有 `euler*` 目标继续作为可压缩求解器回归基线。

### 3.2 建议文件结构

```text
src/ACM/
  ACM.hpp                         模型枚举、traits、变量索引
  ACMSettings.hpp                 β²、alpha、密度模式、压力存储方案
  ACMFlux.hpp                     物理通量、Γ、Jacobian、特征系统、Roe/LLF
  ACMViscous.hpp                  不可压缩黏性通量
  ACMRANS.hpp                     SA/两方程模型的布局无关适配
  ACMBC.hpp                       ACM 边界类型和值解释
  ACMEvaluator.hpp
  ACMEvaluator.hxx                隐式算子和公共实现
  ACMEvaluator_EvaluateDt.hxx     谱半径与局部伪时间步
  ACMEvaluator_EvaluateRHS.hxx    面通量、源项和残差装配
  ACMSolver.hpp
  ACMSolver.hxx
  ACMSolver_Init.hxx
  ACMSolver_PrintData.hxx
  SingleBlockApp.hpp
  _explicit_instantiation/
app/ACM/
  acm3D.cpp
  acmSA3D.cpp
  acm2EQ3D.cpp
  acmVD3D.cpp
  acmVDSA3D.cpp
  acmVD2EQ3D.cpp
test/cpp/ACM/
cases/acm3D/
cases/acmSA3D/
cases/acm2EQ3D/
```

### 3.3 基础设施复用

直接复用：

- `Geom::UnstructuredMesh` 的分区、ghost、周期映射和几何量；
- `CFV::VariationalReconstruction<3>` 的重构、积分和限制器框架；
- DNDS 分布式数组和持久化 halo 通信；
- `Solver::ODE` 的双时间推进；
- `Solver::Linear` 的 GMRES 和局部预条件接口；
- 现有面缓冲后按单元归集的无竞争 OpenMP 模式。

不直接复用：

- `Gas.hpp` 中依赖总能量和理想气体状态方程的通量；
- `EulerEvaluator` 中以密度/动量/能量固定索引实现的旋转、边界和正性修复；
- 直接读取 `I4+1/I4+2` 的 RANS 函数。

在第一个 ACM 模型落地前，把 `ArrayDOFV`、`ArrayRECV`、`JacobianDiagBlock` 等与物理无关的容器提取到共享头文件（建议 `src/CFV/FlowSolution.hpp`），并在 `Euler.hpp` 保留兼容别名。该重构必须先通过现有 Euler 全量测试，避免复制基础设施。

## 4. 数值实现设计

### 4.1 状态布局 traits

`ACMModelTraits` 至少提供：

- `densityMode`：constant / variable；
- `pressureStorage`：physical / scaled；
- `nVarsFixed`；
- `velocityBegin`、`momentumBegin`、`pressureIndex`；
- `hasSA`、`has2EQ`、`nRANSVars`；
- `RotateState()` 和 `RotateGradient()` 所需的向量分量区间。

所有通量、边界、周期变换和输出只能通过 traits 访问分量，禁止散布裸下标 `0/1/4/5`。

### 4.2 物理状态访问器

引入轻量 `ACMStateView`，统一返回 `rho`、`velocity`、`pressure` 和 RANS 原始量：

- 常密度从设置读取 `rho0`，状态前三项直接为速度；
- 变密度从 `U(0)` 读取密度并将动量除以密度；
- 方案 B 输出和边界读取时执行 `p = beta2 * U(pressureIndex)`；
- β²只通过 `Beta2Policy::AtCell/AtFace/AtPseudoTime()` 获取。

这层访问器同时用于 CPU 标量核、批量 Eigen 核和测试参考实现，防止不同路径的变量解释不一致。

### 4.3 无黏数值通量

分两步实现：

1. 首先实现 ACM Local Lax–Friedrichs/Rusanov 通量，用正确最大谱半径打通并行 RHS、边界和时间推进。
2. 再实现报告中的 Roe 特征分解：物理中央通量、波强度、`Γrᵢ` 修正右特征向量和耗散向量。

要求：

- 在局部 `(n,t1,t2)` 坐标计算后旋转回全局坐标；
- 网格速度/ALE 支持在首版标记为不支持并进行运行时拒绝，后续单独推导后开放；
- SA/两方程变量的无黏部分按相应对流速度输运；
- 方案 A/B 必须返回相同物理通量和 Roe 耗散；
- 面谱半径同时供 CFL、Rusanov、隐式对角块使用，避免三套公式漂移。

### 4.4 黏性通量

层流首版实现不可压缩 Newtonian 应力：

\[
\tau=\mu\left(\nabla u+\nabla u^T\right)
-\frac{2}{3}\mu(\nabla\cdot u)I.
\]

压力/连续性方程不加入黏性通量。常密度采用运动黏度或动力黏度之一作为唯一配置表示，并在读配置时完成换算；变密度按动力黏度计算。

RANS 阶段再加入：

- SA 的涡黏度、扩散和生产/耗散源项；
- SST、Wilcox k-ω、Realizable k-ε 的涡黏度、扩散和源项；
- RANS 源项局部 Jacobian；
- 壁面距离及壁面变量处理。

现有 `RANS_ke.hpp` 应重构为“物理量输入 + 输出”的布局无关核，Euler 和 ACM 分别提供 state adapter；不得为 ACM 复制一份公式。

### 4.5 边界条件

第一阶段必须覆盖：

- 周期边界；
- 无滑移壁面：速度 Dirichlet，压力法向 Neumann；
- 滑移/对称边界：法向速度反射，切向速度外推；
- 速度入口：速度给定，压力兼容外推；
- 压力出口：压力给定，速度外推。

RANS 阶段补充 SA、k、ω/ε 的入口值和壁面条件。方案 B 的 JSON 接口仍接收物理压力，缩放只发生在内部，避免用户配置依赖 β²存储方案。

周期变换必须通过 traits 旋转常密度速度 `U[0:3]` 或变密度动量 `U[1:4]`；这是不能直接复用 Euler `Seq123` 的原因之一。

### 4.6 伪时间步和物理时间

稳态伪时间步按每个面的最大绝对特征值与黏性谱半径计算：

\[
\Delta\tau_i=
\mathrm{CFL}\,V_i\bigg/
\sum_{f\in i}\left(\rho_{A,f}+\rho_{V,f}\right)S_f.
\]

先实现局部伪时间步，并用 `MPI_Allreduce(MIN)` 得到诊断用全局最小值。非稳态阶段把真实时间项作为双时间系统中的物理质量矩阵加入；压力人工时间项不进入真实时间守恒项。

### 4.7 隐式算子

并行默认路径采用矩阵自由 GMRES：

- `MatVec` 使用面 Jacobian 作用 `A_n ΔU`，不组装全局矩阵；
- 每次 MatVec 前更新 `ΔU` ghost；
- 单元对角块包含 `Γ/Δτ`、真实时间质量矩阵、面谱半径和源项 Jacobian；
- 分区内 block-Jacobi 或 local ILU/LU 作为第一版预条件器；
- LU-SGS/SGS 仅作为局部或着色预条件器，不能假设跨 MPI rank 的全局扫描顺序。

方案 A/B 的 LU-SGS 对角块必须满足报告关系 `D_B = D_A S⁻¹`，不得按相似变换实现。方案 B 条件数优势通过迭代数和残差曲线验证，而不是通过最终解差异判断。

## 5. 并行计算方案

### 5.1 MPI 域分解

沿用 DNDSR 的 cell ownership 和 ghost 映射：

1. 单元平均值和重构梯度使用分布式数组；
2. 启动 `startPersistentPull()` 更新 ghost；
3. 在通信期间计算不依赖 ghost 的内部面；
4. `waitPersistentPull()` 后计算进程边界面；
5. 面通量只在面缓冲中写一次；
6. 各 owned cell 遍历 `cell2face` 归集残差，避免跨线程和跨 rank 原子写；
7. 仅对残差范数、最小时间步、边界积分等全局量执行 collective。

内部面两侧通量必须逐位相反；跨 rank 面由双方基于相同左右状态和确定的主从方向计算，或由 owner 计算后传输。第一版优先沿用当前“双方确定性重算”方式，并用守恒误差测试验证。

### 5.2 OpenMP

采用三段式并行：

- 面循环：每个线程只写 `faceFlux[iFace]`、`faceLambda[iFace]`；
- 单元归集循环：每个线程只写 owned cell 的 residual；
- 单元源项/时间步循环：每个线程只写当前 cell，最小值和范数使用 reduction。

要求：

- scratch buffer 成为 evaluator 实例成员或线程局部变量，禁止函数内共享 `static std::vector`；
- MPI 调用不得位于 OpenMP 并行区；
- 边界积分采用线程局部累加后合并；
- `schedule(static)` 用于均匀 cell gather，复杂高阶面核可配置 `schedule(guided/runtime)`；
- 所有 Eigen 临时量必须为循环体局部对象，避免线程共享 resize。

### 5.3 混合并行隐式求解

- GMRES 的向量点积/范数使用本地 OpenMP reduction + MPI Allreduce；
- MatVec 将 halo 通信与内部 cell/face 计算重叠；
- block-Jacobi 每个 cell 独立求解小块，可直接 OpenMP；
- local ILU/LU 按 `NLocalParts()` 并行，每个分区只访问自己的块；
- 并行 SGS 如需启用，先对 cell graph 着色，同色 cell 并行，颜色间同步；MPI 边界使用上一轮 halo 值，构成 additive Schwarz 外迭代。

### 5.4 并行正确性约束

- 1/2/4/8 MPI rank 的最终物理残差和监测量在规定容差内一致；
- `OMP_NUM_THREADS=1/2/4/8` 结果一致；
- MPI + OpenMP 不出现死锁、数据竞争或随线程数变化的 NaN；
- 进程分区改变只能影响浮点归约末位，不能改变收敛阶或最终物理解；
- 所有 MPI 测试保留超时设置。

## 6. 配置设计

新增 `acmSettings`：

```json
{
  "densityMode": "Constant",
  "rho0": 1.0,
  "pressureStorage": "PhysicalP",
  "beta2Mode": "Constant",
  "beta2": 1.0,
  "beta2Min": 1e-6,
  "beta2Max": 1e6,
  "alpha": 0.0,
  "inviscidFlux": "Rusanov",
  "viscosity": 1e-3,
  "ransModel": "RANS_None"
}
```

读取后执行以下 release 有效校验：

- `rho0 > 0`、`beta2 > 0`、黏度非负；
- 方案 B 禁止 pseudo-time adaptive β²；
- 第一阶段禁止非零网格速度和旋转坐标；
- model traits 与 `densityMode/ransModel` 匹配；
- 所有初值和边界向量长度与模型一致；
- JSON 中的压力始终为物理 `p`。

## 7. 实施阶段与交付物

### 阶段 0：数学基线冻结（2–4 人日）

- 将报告中的 SymPy 验证脚本整理为仓库脚本；
- 统一 `alpha`、`turkelGamma`、方案 A/B 变换记号；
- 验证常密度和变密度 Jacobian、特征多项式、特征向量；
- 对报告中不一致的中间公式形成一页“实现采用公式”记录；
- 交付：符号验证脚本、公式测试数据和设计评审记录。

退出条件：随机正状态下符号结果与数值特征分解一致；`alpha=-1/0/1` 三个极限全部通过。

### 阶段 1：共享容器和 ACM 骨架（3–5 人日）

- 提取物理无关数组/Jacobian 容器；
- 新增 ACM traits、settings、CMake 目标和 `acm3D` 空骨架；
- 建立固定大小 4 变量分布式 DOF、ghost 和重构；
- 交付：可读取三维网格并完成 MPI 初始化的 `acm3D`。

退出条件：现有 `euler_unit_tests` 无回归；ACM DOF 在 1/2/4 rank halo 测试一致。

### 阶段 2：常密度层流显式基线（7–10 人日）

- 实现方案 A、`alpha=0`、固定 β²；
- 实现物理通量、Rusanov 通量、黏性通量和 CFL；
- 实现周期、壁面、入口、出口；
- 接入高阶重构和显式 SSPRK；
- 增加场输出 `U,V,W,P,divU`；
- 交付：稳态常密度 `acm3D`。

退出条件：制造解达到预期空间阶；周期域内部通量全局守恒；不同 MPI/OMP 布局结果一致。

### 阶段 3：Roe 与并行隐式求解（7–12 人日）

- 实现正确 Jacobian、特征值/特征向量、Roe 耗散；
- 实现矩阵自由 GMRES 和 block-Jacobi/local LU；
- 实现通信-计算重叠；
- 可选实现着色 local SGS；
- 交付：可扩展稳态隐式求解路径。

退出条件：Rusanov/Roe 收敛到同一稳态解；解析 Jacobian-vector 与有限差分误差符合一阶差分预期；MPI 强缩放达到验收门槛。

### 阶段 4：方案 B 与一般 Turkel 参数（4–7 人日）

- 启用固定 β²方案 B；
- 启用 `alpha=-1...1`；
- 加入 A/B 通量、更新和最终物理解等价测试；
- 验证 `D_B=D_A S⁻¹` 及条件数/迭代数变化；
- 对时间变化 β² + 方案 B 做配置拒绝测试。

退出条件：显式 A/B 物理通量和单步物理更新一致；隐式最终物理解一致，收敛路径允许不同。

### 阶段 5：变密度 ACM（7–12 人日）

- 新增 `[ρ,ρu,ρv,ρw,p]` 状态；
- 实现变密度通量、Jacobian、谱半径和密度正性保护；
- 实现密度入口/界面输运边界；
- 先固定 `alpha=0`，再开放经过符号验证的一般 alpha；
- 交付：`acmVD3D`。

退出条件：均匀密度极限退化到常密度结果；密度输运全局守恒；无负密度。

### 阶段 6：SA 和两方程 RANS（10–15 人日）

- 重构布局无关 RANS 核；
- 接入壁面距离、涡黏度、扩散、源项和源项 Jacobian；
- 新增 `acmSA3D/acm2EQ3D` 及变密度对应目标；
- 增加 RANS 正性/下限处理和输出；
- 交付：六个 ACM 三维目标。

退出条件：层流极限回到 `acm3D/acmVD3D`；平板/通道基准趋势正确；1/2/4 rank 湍流残差曲线一致。

### 阶段 7：非稳态、性能和文档（5–8 人日）

- 接入双时间 BDF/SDIRK；
- 增加真实时间质量矩阵和内迭代停止条件；
- 完成混合并行 profiling、通信重叠和批量 Eigen 核优化；
- 编写用户配置、构建、算例和限制说明；
- 交付：可发布的 ACM 功能。

估算总工作量：一名熟悉 DNDSR 的开发者约 8–12 周；阶段 0–3 可形成约 3–5 周的常密度层流 MVP。估算不包含 CUDA 版本。

## 8. 测试计划

### 8.1 数学单元测试

- 物理通量有限差分 Jacobian；
- `Γ Γ⁻¹ = I`；
- `Ã rᵢ = λᵢ rᵢ` 和 `L R = I`；
- 特征多项式和 `alpha=-1/0/1` 极限；
- 常密度 Jacobian 第一列 `2qn,qt1,qt2,1`；
- A/B 的 `F_B=F_A`、`A_B=A_A S⁻¹`、`Γ_B=Γ_A S⁻¹`、`Ã_B=S Ã_A S⁻¹`；
- A/B Roe 耗散和物理通量相同；
- `D_B=D_A S⁻¹`；
- 解析 Jacobian-vector 与中心有限差分对比。

### 8.2 组件测试

- 周期状态/梯度旋转往返；
- 壁面、入口、出口 ghost 状态；
- 面两侧残差严格反号；
- 黏性通量在均匀场为零；
- 均匀流保持；
- β²配置策略和非法组合拒绝；
- RANS 涡黏度、源项、黏性通量有限且满足下限。

### 8.3 MPI/OpenMP 测试矩阵

对核心 pipeline 注册：

```text
MPI ranks:       1, 2, 4, 8
OMP threads:     1, 2, 4
pressure mode:   A, B（B 仅固定 β²）
flux:            Rusanov, Roe
density:         constant, variable
physics:         laminar, SA, 2EQ
```

CI 只运行代表性子集，夜间或 HPC 测试运行笛卡尔组合。比较全局质量/动量、压力均值、`||div u||`、残差下降和监测点，不直接逐位比较所有浮点数据。

### 8.4 验证算例

- 三维周期均匀流：保持性和 MPI 守恒；
- 三维不可压 Taylor–Green vortex：非稳态精度和能量衰减；
- Poiseuille/周期通道：黏性项和压降；
- 三维 lid-driven cavity：稳态压力-速度耦合；
- 制造解：空间收敛阶；
- 变密度涡/密度团平移：密度守恒；
- SA 平板边界层；
- SST/Wilcox 湍流通道或平板。

## 9. 性能验收

在固定编译器、MPI、网格和硬件下保存基线。建议首轮门槛：

- 百万单元级网格从 1 到 8 MPI rank 强缩放效率不低于 65%；
- 单节点 1 到 8 OpenMP 线程加速比保持单调增长，且无明显 false sharing；
- 混合并行下 halo 通信能够与内部面计算重叠；
- 稳态隐式方案相对显式方案显著减少达到同一残差下降所需的 wall time；
- A/B 性能比较使用相同物理残差阈值；
- profiling 输出至少分解 reconstruction、halo、face flux、cell gather、source、MatVec、preconditioner 和 collective 时间。

若目标机器拓扑不适合上述固定阈值，应在阶段 1 记录机器基线并调整，但不能取消强/弱缩放验收。

## 10. 风险与控制措施

| 风险 | 影响 | 控制措施 |
|---|---|---|
| 报告内 `gamma/alpha` 记号不一致 | 特征值或 Jacobian 实现错误 | 阶段 0 符号冻结；首版 `alpha=0` |
| 混用物理通量与变量缩放 | A/B 得到不同物理解 | 将第 8.3 节变换写成单元测试 |
| 直接复用 Euler 裸下标 | 周期、边界或 RANS 访问错误 | traits + state view；禁止裸布局假设 |
| 面循环同时写两个 cell | OpenMP 数据竞争 | 面缓冲 + cell gather |
| 全局 LU-SGS 顺序跨 rank 不成立 | 并行收敛恶化或结果依赖分区 | GMRES + additive Schwarz 为默认；SGS 仅局部/着色 |
| 方案 B 配合时间变化 β² | 非物理压力伪源项 | release 配置校验直接拒绝 |
| RANS 公式复制 | Euler/ACM 模型漂移 | 重构布局无关 RANS 物理核 |
| 变密度负密度 | NaN 和发散 | 均值正性、更新限幅、密度通量测试 |
| 大规模显式模板实例化 | 编译时间和二进制膨胀 | 模型列表显式实例化；按阶段添加目标 |

## 11. 完成定义

功能只有同时满足以下条件才视为完成：

- 数学公式、SymPy oracle 和 C++ 单元测试一致；
- 现有 Euler 测试全部通过，现有可执行程序输出无回归；
- ACM 在至少一个三维层流和一个三维湍流基准上收敛；
- 1/2/4 MPI rank 与 1/2/4 OpenMP 线程结果在容差内一致；
- 显式、GMRES 和可选 SGS 路径收敛到同一物理解；
- A/B 在允许范围内满足通量和最终物理解等价性；
- 非法 β²/压力存储组合在 release 构建中被拒绝；
- 构建、配置、变量定义、边界条件和已知限制已有用户文档；
- 性能报告包含强缩放、混合并行和主要 kernel 时间占比。

## 12. 推荐首个开发切片

首个可评审 PR 只包含：

1. 数学符号验证脚本和 C++ ACM flux/Jacobian 单元测试；
2. `ACMModelTraits<ConstantDensity, NoRANS>`；
3. 方案 A、`alpha=0`、固定 β²的常密度物理通量和 Rusanov 通量；
4. 不涉及网格和 MPI 的纯函数实现。

第二个 PR 再接入分布式 DOF、CFV 重构和 MPI/OpenMP RHS。这样可在并行复杂性进入前先冻结最容易出错的数学核心。
