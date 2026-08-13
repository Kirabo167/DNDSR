# DNDSR 项目文件说明 —— 以求解器形式整理

> 生成时间：2026-06-02

本文档按**求解器/可执行程序**组织，说明每个程序入口涉及的核心文件、类与函数，同时列出支撑库的关键接口。

---

## 目录

1. [Euler 可压缩流求解器族](#1-euler-可压缩流求解器族)
2. [CFV 测试与基准程序](#2-cfv-测试与基准程序)
3. [Geom 网格测试程序](#3-geom-网格测试程序)
4. [DNDS 核心测试程序](#4-dnds-核心测试程序)
5. [Solver 线性/ODE 测试](#5-solver-线性ode-测试)
6. [外部依赖测试程序](#6-外部依赖测试程序)
7. [支撑库：DNDS 核心模块](#7-支撑库dnds-核心模块)
8. [支撑库：Geom 几何网格模块](#8-支撑库geom-几何网格模块)
9. [支撑库：CFV 高阶有限体积模块](#9-支撑库cfv-高阶有限体积模块)
10. [支撑库：Euler 求解器库](#10-支撑库euler-求解器库)
11. [支撑库：EulerP GPU 求值器](#11-支撑库eulerp-gpu-求值器)
12. [支撑库：Solver ODE/线性求解器](#12-支撑库solver-ode线性求解器)
13. [Python 绑定模块](#13-python-绑定模块)

---

## 1. Euler 可压缩流求解器族

所有 Euler 求解器共享同一套模板代码，通过编译期 `EulerModel` 区分物理模型与维度。

### 1.1 求解器模型对照表

| 可执行文件 | 入口文件 | Model | 维度 | gDim | nVars | 物理模型 |
|-----------|---------|-------|------|------|-------|---------|
| `euler` | `app/Euler/euler.cpp` | `NS` | 3 | 2 | 5 | 2D Euler/NS (3速度分量) |
| `euler2D` | `app/Euler/euler2D.cpp` | `NS_2D` | 2 | 2 | 4 | 纯 2D Euler/NS |
| `euler3D` | `app/Euler/euler3D.cpp` | `NS_3D` | 3 | 3 | 5 | 3D Euler/NS |
| `eulerSA` | `app/Euler/eulerSA.cpp` | `NS_SA` | 3 | 2 | 6 | 2D Spalart-Allmaras |
| `eulerSA3D` | `app/Euler/eulerSA3D.cpp` | `NS_SA_3D` | 3 | 3 | 6 | 3D SA RANS |
| `euler2EQ` | `app/Euler/euler2EQ.cpp` | `NS_2EQ` | 3 | 2 | 7 | 2D k-w / k-e / SST |
| `euler2EQ3D` | `app/Euler/euler2EQ3D.cpp` | `NS_2EQ_3D` | 3 | 3 | 7 | 3D 2方程 RANS |
| `eulerEX` | `app/Euler/eulerEX.cpp` | `NS_EX` | 3 | 2 | 动态 | 2D Euler + 额外标量 |
| `eulerEX3D` | `app/Euler/eulerEX3D.cpp` | `NS_EX_3D` | 3 | 3 | 动态 | 3D Euler + 额外标量 |

### 1.2 入口与主循环

**入口函数**：`RunSingleBlockConsoleApp<model>(argc, argv)` （定义于 `src/Euler/SingleBlockApp.hpp`）

执行流程：
1. CLI 解析（`argparse`）—— 读取 JSON 配置、键值覆盖、`--emit-schema`
2. 构造 `EulerSolver<model>`
3. `ConfigureFromJson()` → `ReadMeshAndInitialize()` → `RunImplicitEuler()`

### 1.3 核心类与文件

#### `src/Euler/EulerSolver.hpp` / `.hxx`
**类**：`template <EulerModel model> class EulerSolver`

| 方法 | 说明 |
|------|------|
| `ConfigureFromJson(jsonName, read, merge, overwrites...)` | 读写配置 |
| `ReadMeshAndInitialize()` | CGNS 读取、分区、提升、二分、构建 VFV |
| `PrintConfig(updateCommit)` | 写出配置 JSON + 时间戳 |
| `ReadRestart(fname)` / `ReadRestartOtherSolver(...)` / `PrintRestart(fname)` | 重启 I/O |
| `PrintData(...)` / `WriteSerializer(...)` | VTK/Tecplot/HDF5 输出 |
| `RunImplicitEuler()` | **主驱动循环** |
| `InitializeRunningEnvironment(env)` | 初始化 ODE、GMRES、PCG |
| `solveLinear(...)` | 线性系统求解包装器 |
| `doPrecondition(...)` | 预处理应用 |
| `functor_fstop(...)` / `functor_fmainloop(...)` | 停机判据与主循环体 |

**嵌套配置结构**（`Configuration`）：
- `TimeMarchControl` —— dt, CFL, ODE 代码, 重启, PP
- `ImplicitReconstructionControl` —— 显式/隐式重构, GMRES/PCG/SOR
- `OutputControl` —— 控制台, VTK, Tecplot, HDF5, 重启间隔
- `ImplicitCFLControl` —— CFL 爬升, 局部 dt, RANS 松弛
- `ConvergenceControl` —— 内迭代, 残差范数, CL 驱动
- `DataIOControl` —— 网格文件, 提升, 二分, 输出格式
- `BoundaryDefinition` —— 周期性平移/旋转
- `LimiterControl` —— WBAP/CWBAP, PP 限制器
- `LinearSolverControl` —— Jacobi/GS/ILU, GMRES, 多级网格
- `RestartState`, `TimeAverageControl`, `Others`

**`RunningEnvironment`** —— 持有全部瞬态求解器状态（残差、计时器、步数计数器等）

---

#### `src/Euler/EulerEvaluator.hpp` / `.hxx`
**类**：`template <EulerModel model> class EulerEvaluator`

| 方法 | 说明 |
|------|------|
| `InitializeFV(mesh, vfv, pBCHandler)` | 设置周期性变换, 构建度量、基/权重、重构系数 |
| `GetWallDist()` | 计算壁面距离场 |
| `EvaluateDt(dt, u, uRec, CFL, dtMin, MaxDt, UseLocaldt, t, flags)` | 局部/稳定时间步 |
| `EvaluateRHS(rhs, JSource, u, uRecUnlim, uRec, uRecBeta, cellRHSAlpha, onlyOnHalfAlpha, t, flags)` | 右端项残差 |
| `LUSGSMatrixInit(JDiag, JSource, dTau, dt, alphaDiag, u, uRec, jacobianCode, t)` | 初始化 LUSGS/ILU 预处理 |
| `LUSGSMatrixVec(alphaDiag, t, u, uInc, JDiag, AuInc)` | 预处理矩阵-向量乘 |
| `LUSGSMatrixToJacobianLU(...)` | 转换 LUSGS 矩阵到局部 LU |
| `UpdateLUSGSForward(...)` / `UpdateLUSGSBackward(...)` | LUSGS 前/后扫 |
| `UpdateSGS(...)` / `UpdateSGSWithRec(...)` | 通用 SGS 扫 |
| `LUSGSMatrixSolveJacobianLU(...)` | 用局部 Jacobian LU 求解 |
| `FixUMaxFilter(u)` | 截断最大值 |
| `TimeAverageAddition(w, wAveraged, dt, tCur)` | 时间平均 |
| `MeanValueCons2Prim(u, w)` / `MeanValuePrim2Cons(w, u)` | 单元均值 C↔P |
| `EvaluateNorm(res, rhs, P, volWise, average)` | 残差范数 |
| `EvaluateRecNorm(...)` | 重构误差范数 |
| `LimiterUGrad(u, uGrad, uGradNew, flags)` | 梯度限制 |
| `EvaluateURecBeta(u, uRec, uRecBeta, nLim, betaMin, flag)` | 保正 beta 限制器 |
| `AssertMeanValuePP(u, panic)` | 断言均值保正 |
| `EvaluateCellRHSAlpha(...)` / `EvaluateCellRHSAlphaExpansion(...)` | RHS alpha 限制器 |
| `MinSmoothDTau(dTau, dTauNew)` | 平滑局部 dt |
| `muEff(U, T)` | 有效粘性（Sutherland/常数） |
| `getMuTur(...)` | 湍流粘性（SA / 2EQ） |
| `visFluxTurVariable(...)` | 湍流粘性通量 |
| `fluxFace(...)` | 面通量（无粘+粘性） |
| `source(...)` | 源项（轴对称、科氏力、RANS） |
| `fluxJacobian0_Right(...)` | 无粘通量雅可比 |
| `fluxJacobian0_Right_Times_du(...)` | 雅可比×du |
| `fluxJacobian0_Right_Times_du_AsMatrix(...)` | 组装雅可比矩阵 |
| `fluxJacobianC_Right_Times_du(...)` | 仅对流部分 |
| `GetFaceVGrid(...)` | 面网格速度 |
| `TransformVelocityRotatingFrame(...)` / `TransformURotatingFrame(...)` | 旋转坐标系变换 |
| `updateBCAnchors(u, uRec)` / `updateBCProfiles(...)` | BC 锚点/剖面更新 |
| `generateBoundaryValue(...)` | 生成幽灵/边界状态 |
| `PrintBCProfiles(...)` | 写出剖面 CSV |
| `ConsoleOutputBndIntegrations()` / `BndIntegrationLogWriteLine(...)` | 边界积分日志 |
| `CLDriverGetIntegrationUpdate(iter)` | CL 驱动力/AoA 更新 |
| `CompressRecPart(...)` / `CompressInc(...)` | 保正压缩 |
| `FixIncrement(...)` / `AddFixedIncrement(...)` | 安全增量应用 |
| `CentralSmoothResidual(...)` | 残差中心光滑 |
| `InitializeUDOF(u)` | 从设置初始化 DOF |
| `InitializeOutputPicker(...)` | 注册输出场 |

---

#### `src/Euler/EulerBC.hpp`
**枚举**：`EulerBCType` —— `BCFar`, `BCWall`, `BCWallInvis`, `BCWallIsothermal`, `BCOut`, `BCOutP`, `BCIn`, `BCInPsTs`, `BCSym`, `BCSpecial`

**类**：`template <EulerModel model> class BoundaryHandler`
- JSON 可序列化（`from_json` / `to_json`）
- 存储 `BCValues`, `BCTypes`, `BCFlags`, `BCValuesExtra`, 名称↔ID 映射
- `GetIDFromName`, `GetNameFormID`, `GetTypeFromID`, `GetValueFromID`, `GetFlagFromIDSoft`

**辅助结构**：
- `IntegrationRecorder` —— MPI 归约的边界面积分
- `AnchorPointRecorder<nVarsFixed>` —— 最近锚点记录
- `OutputPicker` —— 单元标量输出注册表
- `OneDimProfile<nVarsFixed>` —— 1D 剖面提取（均匀/tanh 网格、归约、CSV 输出）

---

#### `src/Euler/EulerJacobian.hpp`
**类**：`template <int nVarsFixed> class JacobianDiagBlock`
- 纯对角或块对角存储，`SetModeAndInit`, `getBlock`/`getDiag`, `GetInvert`, `MatVecLeft`/`MatVecLeftInvert`

**结构**：`template <int nVarsFixed> struct JacobianLocalLU : Direct::LocalLUBase<...>`
- LDU 稀疏块矩阵存储，`GetDiag`, `GetLower`, `GetUpper`, `InvertDiag`

**结构**：`template <int nVarsFixed> struct JacobianLocalLDLT`
- LDL^T 对称预处理变体

---

#### `src/Euler/Gas.hpp`
**命名空间**：`DNDS::Euler::Gas`

**枚举**：`RiemannSolverType` —— `Roe`, `HLLC`, `HLLEP`, `HLLEP_V1`, `Roe_M1`…`Roe_M9`

**关键自由函数**：
| 函数 | 说明 |
|------|------|
| `EulerGasRightEigenVector<dim>(velo, Vsqr, H, a, ReV)` | 右特征向量矩阵 |
| `EulerGasLeftEigenVector<dim>(velo, Vsqr, H, a, gamma, LeV)` | 左特征向量矩阵 |
| `IdealGasThermal(E, rho, vSqr, gamma, p, asqr, H)` | 状态方程 |
| `IdealGasThermalConservative2Primitive<dim>(U, prim, gamma)` | C→P |
| `IdealGasThermalPrimitive2Conservative<dim>(prim, U, gamma)` | P→C |
| `GasInviscidFlux<dim>(U, velo, vg, p, F)` | 1D 无粘通量 |
| `GasInviscidFlux_XY<dim>(U, velo, vg, n, p, F)` | 法向无粘通量 |
| `HLLEPFlux_IdealGas<dim,type>(...)` | HLLEP 黎曼求解器 |
| `HLLCFlux_IdealGas_HartenYee<dim>(...)` | HLLC 求解器 |
| `Roe_EntropyFixer<eigScheme>(...)` | 熵修正（H2, cLLF, LD, ID 等） |
| `RoeFlux_IdealGas_HartenYee<dim,eigScheme>(...)` | Roe 求解器（标量） |
| `RoeFlux_IdealGas_HartenYee_Batch<dim,eigScheme>(...)` | Roe 求解器（批量） |
| `InviscidFlux_IdealGas_Dispatcher(...)` | 分发给选定黎曼求解器 |

---

#### `src/Euler/CLDriver.hpp`
**结构**：`CLDriverSettings` —— AoA 初值、轴、refArea、targetCL、松弛、收敛窗口
**类**：`CLDriver`
- `GetAOA()`, `Update(iter, CL, mpi)`, `ConvergedAtTarget()`
- `GetAOARotation()`, `GetCL0Direction()`, `GetCD0Direction()`, `GetForce2CoeffRatio()`

---

#### `src/Euler/RANS_ke.hpp`
**命名空间**：`DNDS::Euler::RANS`

| 函数 | 模型 |
|------|------|
| `GetMut_RealizableKe` / `GetVisFlux_RealizableKe` / `GetSource_RealizableKe` / `GetSourceJacobianDiag_RealizableKe` | Realizable k-e |
| `GetMut_SST` / `GetVisFlux_SST` / `GetSource_SST` | k-w SST |
| `GetMut_KOWilcox` / `GetVisFlux_KOWilcox` / `GetSource_KOWilcox` | k-w Wilcox |
| `GetSource_SA` | Spalart-Allmaras（含 DDES/IDDES, QCR, 旋转修正） |

---

#### `src/Euler/SpecialFields.hpp`
**命名空间**：`DNDS::Euler::SpecialFields`
- `IsentropicVortex10<model>(eval, x, t, nVars, chi)` —— 周期涡（10×10 域）
- `IsentropicVortex30<model>(eval, x, t, nVars)` —— 30×30 域
- `IsentropicVortexCent<model>(eval, x, t, nVars)` —— 原点中心

---

### 1.4 显式实例化体系

模板代码拆分进 `.hxx` 头文件，由 `src/Euler/_explicit_instantiation/` 下的微型 `.cpp` 文件显式实例化，避免重复编译。

| 模板头文件 | 实例化宏 | 包含的方法 |
|-----------|---------|-----------|
| `EulerEvaluator_EvaluateDt.hxx` | `DNDS_EulerEvaluator_EvaluateDt_INS_EXTERN` | `GetWallDist`, `EvaluateDt`, `fluxFace`, `source`, `generateBoundaryValue`, `InitializeOutputPicker` |
| `EulerEvaluator_EvaluateRHS.hxx` | `DNDS_EulerEvaluator_EvaluateRHS_INS_EXTERN` | `EvaluateRHS` |
| `EulerEvaluator.hxx` | `DNDS_EulerEvaluator_INS_EXTERN` | `LUSGSMatrixInit`, `LUSGSMatrixVec`, `LUSGSMatrixToJacobianLU`, `UpdateLUSGSForward/Backward`, `UpdateSGS`, `LUSGSMatrixSolveJacobianLU`, `InitializeUDOF`, `FixUMaxFilter`, `TimeAverageAddition`, `MeanValueCons2Prim/Prim2Cons`, `EvaluateNorm/RecNorm`, `LimiterUGrad`, `EvaluateURecBeta`, `AssertMeanValuePP`, `EvaluateCellRHSAlpha`, `MinSmoothDTau`, `updateBCProfiles` |
| `EulerSolver_Init.hxx` | `DNDS_EULERSOLVER_INIT_INS_EXTERN` | `ReadMeshAndInitialize`, `functor_fstop`, `functor_fmainloop` |
| `EulerSolver.hxx` | `DNDS_EULERSOLVER_INS_EXTERN` | `RunImplicitEuler`, `InitializeRunningEnvironment` |
| `EulerSolver_PrintData.hxx` | `DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN` | `PrintData`, `PrintRestart`, `ReadRestart`, `ReadRestartOtherSolver` |

**CMake 构建**：每个模型生成两个库
- `euler_library_fast_<model>`（`SHARED`, `FAST` 编译标志）：`EvaluateDt`, `EvaluateRHS`, `EulerEvaluator`, `PrintData`, `Init`
- `euler_library_<model>`：`EulerSolver`（链接 fast 库）

---

## 2. CFV 测试与基准程序

### 2.1 可执行程序

| 程序 | 入口文件 | 说明 |
|------|---------|------|
| `diffTensors_Test` | `app/CFV/diffTensors_Test.cpp` | 独立单元测试 `Geom::Base::DxDxi2DxiDx`。测试 diff-tensor 转换（3D↔3D），含硬编码与随机矩阵，打印往返误差。无 MPI 网格。 |
| `vrBasic_Test` | `app/CFV/vrBasic_Test.cpp` | 读取 CGNS 网格 (`Uniform32_Periodic.cgns`)，分区，构建幽灵单元，构造 `VariationalReconstruction<2>`（`ConstructMetrics`, `ConstructBaseAndWeight`, `ConstructRecCoeff`）。测试网格序列化往返（写出再读取 JSON 分区文件）。 |
| `vrStatic_Test` | `app/CFV/vrStatic_Test.cpp` | 读取均匀方形网格 (`UniformSquare_80.cgns`)，构建最高 3 阶 VR，对制造光滑函数 (`cos(kpix)cos(kpiy)`) 做静态重构收敛测试。计算单元均值，运行 `DoReconstructionIter` 定点迭代，可选 GMRES 求解 Jacobian (`DoReconstructionIterDiff`)，计算光滑指示器，应用 `DoLimiterWBAP_C`，打印 L1/L2/L∞ 误差。 |

### 2.2 涉及核心类

#### `src/CFV/FiniteVolume.hpp` / `.cpp`
**类**：`FiniteVolume`（基类，继承 `DeviceTransferable<FiniteVolume>`）

| 方法 | 说明 |
|------|------|
| `FiniteVolume(MPIInfo, ssp<Geom::UnstructuredMesh>)` | 构造函数 |
| `getDim()` | 返回空间维度 |
| `getSettings()` / `parseSettings(json&)` | 访问/解析 `FiniteVolumeSettings` |
| `MakePairDefaultOnCell<T>(aPair, name, ...)` | 在单元上初始化 ArrayPair |
| `MakePairDefaultOnFace<T>(aPair, name, ...)` | 在面上初始化 ArrayPair |
| `SetCellAtrBasic()` | 设置单元 `RecAtr` |
| `ConstructCellVolume()` | 数值积分计算单元体积 |
| `ConstructCellBary()` / `ConstructCellCent()` | 单元重心/形心 |
| `ConstructCellIntJacobiDet()` / `ConstructCellIntPPhysics()` | 单元积分点 Jacobi/物理坐标 |
| `ConstructCellAlignedHBox()` / `ConstructCellMajorHBoxCoordInertia()` | 轴对齐包围盒/惯性张量 |
| `SetFaceAtrBasic()` | 设置面 `RecAtr` |
| `ConstructFaceArea()` / `ConstructFaceCent()` | 面面积/形心 |
| `ConstructFaceIntJacobiDet()` / `ConstructFaceIntPPhysics()` | 面积分点 Jacobi/物理坐标 |
| `ConstructFaceUnitNorm()` / `ConstructFaceMeanNorm()` | 面单位法向/平均法向 |
| `ConstructCellSmoothScale()` | 单元光滑长度尺度 |
| `BuildUDof<u>(u, nVars, ...)` / `BuildUGradD<u>(u, nVars, ...)` | 构建 DOF/梯度数组对 |
| `GetCellAtr(iCell)` / `GetFaceAtr(iFace)` / `GetCellOrder(iCell)` | 访问属性 |
| `GetCellVol(iCell)` / `GetFaceArea(iFace)` / `GetGlobalVol()` | 体积/面积 |
| `GetCellJacobiDet(iCell, iG)` / `GetFaceJacobiDet(iFace, iG)` | 积分点 Jacobi |
| `GetFaceQuad(iFace)` / `GetCellQuad(iCell)` / `GetFaceQuadO1(iFace)` / `GetCellQuadO1(iCell)` | 积分对象 |
| `GetFaceNorm(iFace, iG)` / `GetFaceNormFromCell(...)` | 法向（含周期性变换） |
| `GetFaceQuadraturePPhys(...)` / `GetFaceQuadraturePPhysFromCell(...)` | 面积分点（含周期性） |
| `GetCellQuadraturePPhys(iCell, iG)` | 单元积分点 |
| `deviceView<B>()` | 返回 `FiniteVolumeDeviceView<B>` |

#### `src/CFV/VariationalReconstruction.hpp` / `.cpp` / `_Reconstruction.hxx` / `_LimiterProcedure.hxx`
**类**：`template <int dim> class VariationalReconstruction`（继承 `FiniteVolume`）

| 方法 | 说明 |
|------|------|
| `ConstructMetrics()` | 调用全部 FV 度量构造 |
| `ConstructBaseAndWeight(f)` | 计算单元基矩、缓存 diff-base、计算面权重与尺度 |
| `ConstructRecCoeff()` | 构建重构矩阵：`A`, `A^{-1}B`, `A^{-1}b`、次级矩阵、Green-Gauss 项 |
| `FDiffBaseValue(DiBj, pPhy, iCell, iFace, iG, flag)` | 物理点处多项式基（及导数）求值 |
| `GetIntPointDiffBaseValue(iCell, iFace, if2c, iG, diffList, maxDiff)` | 积分点处缓存或即时 diff-base |
| `GetMatrixSecondary(iCell, iFace, if2c)` | 面的次级重构矩阵切片 |
| `FFaceFunctional(DiffI, DiffJ, iFace, iCellL, iCellR)` | 面泛函内积矩阵（含各向异性选项） |
| `GetGreenGauss1WeightOnCell(iCell)` | Green-Gauss 权重 |
| `GetCellAR(iCell)` | 主轴包围盒纵横比 |
| `MatrixAMult(uRec, uRec1)` | `A * uRec` |
| `DownCastURecOrder(curOrder, iCell, uRec, downCastMethod)` | 高阶系数置零（可选正交化） |
| `BuildURec(u, nVars, ...)` / `BuildUGrad(u, nVars, ...)` / `BuildScalar(u, ...)` / `BuildUDofNode(u, ...)` | 分配初始化重构数据 |
| `DoReconstruction2ndGrad(uRec, u, FBoundary, method)` | 2 阶 Gauss-Green 或最小二乘梯度重构 |
| `DoReconstruction2nd(uRec, u, FBoundary, method, mask)` | 2 阶重构生成 `uRec` 系数 |
| `DoReconstructionIter(uRec, uRecNew, u, FBoundary, putIntoNew, recordInc, uRecIsZero)` | 变分重构定点求解的一次 SOR/Jacobi 迭代 |
| `DoReconstructionIterDiff(uRec, uRecDiff, uRecNew, u, FBoundaryDiff)` | 重构算子的 Jacobian-向量积 |
| `DoReconstructionIterSOR(...)` | 增量 RHS 的 SOR 扫 |
| `DoCalculateSmoothIndicator(si, uRec, u, varsSee)` | 计算单元光滑指示器 |
| `DoCalculateSmoothIndicatorV1(si, uRec, u, varsSee, FPost)` | 带后处理回调的变体 |
| `DoLimiterWBAP_C(u, uRec, uRecNew, uRecBuf, si, ifAll, FM, FMI, putIntoNew)` | 紧致模板 WBAP 限制器 |
| `DoLimiterWBAP_3(u, uRec, uRecNew, uRecBuf, si, ifAll, FM, FMI, putIntoNew)` | 3 遍 WBAP 限制器 |
| `WriteSerializeRecMatrix(serializerP)` | 序列化 `matrixAAInvB` |

#### `src/CFV/Limiters.hpp`
**自由函数模板**：
- `PolynomialSquaredNorm<dim>(theta)` / `PolynomialDotProduct<dim>(theta1, theta2)` —— 多项式系数加权 L2 范数/点积
- `FWBAP_L2_Multiway_Polynomial2D` / `FMEMM_Multiway_Polynomial2D` / `FWBAP_L2_Multiway_PolynomialOrth` / `FWBAP_L2_Multiway` —— 多路限制器
- `FWBAP_L2_Biway` / `FWBAP_L2_Cut_Biway` / `FMINMOD_Biway` / `FVanLeer_Biway` —— 双路限制器
- `FWBAP_L2_Biway_PolynomialNorm<dim,nVarsFixed>` / `FMEMM_Biway_PolynomialNorm<dim,nVarsFixed>` / `FWBAP_L2_Biway_PolynomialOrth` —— 带多项式范数的双路限制器

#### `src/CFV/ModelEvaluator.hpp` / `.cpp`
**类**：`ModelEvaluator` —— 具体 2D 对流-扩散测试物理
- `EvaluateRHS(rhs, u, uRec, t, options)` —— 计算完整 RHS：面通量积分（对流+扩散）
- `DoReconstructionIter(...)` —— 委托给 `vfv->DoReconstructionIter`

---

## 3. Geom 网格测试程序

| 程序 | 入口文件 | 说明 |
|------|---------|------|
| `elements_Test` | `app/Geom/elements_Test.cpp` | 测试所有单元类型：形函数 delta 性质、参考单元体积积分、面法向一致性、`GetElemNodeMajorSpan` 旋转不变性。 |
| `meshSerial_Test` | `app/Geom/meshSerial_Test.cpp` | 端到端网格管线：串行读取 CGNS，去重周期性节点，构建 `cell2cell`，ParMetis 分区，构建幽灵层，插值面，构建边界网格，全局/局部索引往返，坐标变换，输出 VTK/Plt。 |
| `ofReader_Test` | `app/Geom/ofReader_Test.cpp` | 读取 OpenFOAM `points`, `faces`, `owner`, `neighbour`, `boundary`，运行 `OpenFOAMConverter` 构建 `cell2face` 和 `cell2node`。 |
| `partitionMeshSerial` | `app/Geom/partitionMeshSerial.cpp` | 独立网格分区器：读取 CGNS，去重周期节点，构建拓扑，ParMetis 分区，构建幽灵主层，写出每秩 JSON 分区文件。 |

### 涉及核心类

#### `src/Geom/Mesh.hpp` / `.cpp`
**类**：`UnstructuredMesh`（继承 `DeviceTransferable<UnstructuredMesh>`）

| 方法 | 说明 |
|------|------|
| `NumNode()` / `NumCell()` / `NumFace()` / `NumBnd()` | 本地大小 |
| `NumNodeGhost()` / `NumCellGhost()` / `NumFaceGhost()` / `NumBndGhost()` | 幽灵大小 |
| `NumNodeProc()` / `NumCellProc()` / `NumFaceProc()` / `NumBndProc()` | 本地+幽灵 |
| `NumNodeGlobal()` / `NumCellGlobal()` / `NumFaceGlobal()` / `NumBndGlobal()` | 集体全局大小 |
| `GetCellElement()` / `GetFaceElement()` / `GetBndElement()` | 获取单元类型 |
| `GetCellZone()` / `GetFaceZone()` / `GetBndZone()` | 获取区域标记 |
| `NodeIndexGlobal2Local` / `NodeIndexLocal2Global` / `CellIndexGlobal2Local` 等 | 索引转换 |
| `ConvertAdjEntries()` / `PermuteRows()` | 邻接转换/行置换 |
| `GetCoordsOnCell()` / `GetCoordsOnFace()` / `GetCoordNodeOnCell()` / `GetCoordNodeOnFace()` | 坐标获取 |
| `CellIsFaceBack()` / `CellFaceOther()` | 拓扑辅助 |
| `RecoverNode2CellAndNode2Bnd()` / `RecoverCell2CellAndBnd2Cell()` | 拓扑重建 |
| `BuildGhostPrimary()` / `BuildGhostN2CB()` | 幽灵层构建 |
| `InterpolateFace()` / `BuildCell2CellFace()` | 面插值/构建 |
| `ConstructBndMesh()` | 构建边界网格 |
| `AdjGlobal2LocalPrimary()` / `AdjLocal2GlobalPrimary()` / `AdjGlobal2LocalFacial()` | 邻接全局/局部转换 |
| `BuildO2FromO1Elevation()` / `ElevatedNodesGetBoundarySmooth()` / `ElevatedNodesSolveInternalSmooth*()` | O1→O2 阶提升 |
| `BuildBisectO1FormO2()` / `IsO1()` / `IsO2()` | h 细化/阶判断 |
| `RecreatePeriodicNodes()` | 重建周期性节点 |
| `BuildVTKConnectivity()` | VTK 连通性 |
| `TransformCoords()` | 坐标变换 |
| `BuildNodeWallDist()` | 节点壁面距离 |
| `WriteSerialize()` / `ReadSerialize()` / `ReadSerializeAndDistribute()` | 序列化/反序列化 |
| `PrintParallelVTKHDFDataArray()` / `PrintMeshCGNS()` | VTK/CGNS 输出 |
| `deviceView<B>()` / `to_device()` / `to_host()` / `getArrayBytes()` | 设备管理 |

**类**：`UnstructuredMeshSerialRW`
- `ReadFromCGNSSerial()` / `ReadFromOpenFOAMAndConvertSerial()` —— 串行读取
- `Deduplicate1to1Periodic()` —— 1-1 周期性去重
- `BuildCell2Cell()` / `BuildNode2Node()` —— 拓扑构建
- `MeshPartitionCell2Cell()` / `PartitionReorderToMeshCell2Cell()` —— 分区与重排序
- `PrintSerialPartPltBinaryDataArray()` / `PrintSerialPartVTKDataArray()` —— 串行输出

**结构**：`PartitionOptions` —— ParMetis 分区选项

---

#### `src/Geom/Elements.hpp` / `ElementTraits.hpp`
**类**：`Element`

| 方法 | 说明 |
|------|------|
| `GetParamSpace()` / `GetDim()` / `GetOrder()` / `GetNumVertices()` / `GetNumNodes()` / `GetNumFaces()` | 元数据查询 |
| `GetNumElev_O1O2()` / `GetO2NumBisect()` | 提升/二分信息 |
| `ObtainFace(iFace)` / `ObtainElevNodeSpan(iNodeElev)` / `ObtainElevatedElem()` / `ObtainO1Elem()` / `ObtainO2BisectElem(iSubElem)` | 获取子实体 |
| `ExtractFaceNodes()` / `ExtractElevNodeSpanNodes()` / `ExtractO2BisectElemNodes()` | 提取节点 |
| `GetNj()` / `GetD1Nj()` / `GetD01Nj()` / `GetDiNj()` | 形函数求值 |

**自由函数**：
- `ElemType_to_ParamSpace()` / `GetFaceType()` / `ParamSpaceVol()`
- `ShapeFunc_DiNj<diffOrder>()` —— 分派到 `ShapeFuncImpl<ElemType>`
- `ShapeJacobianCoordD01Nj()` / `PPhysicsCoordD01Nj()`
- `JacobiDetFace()` / `CellJacobianDet()` / `FaceJacobianDet()`
- `cellsAreFaceConnected()`
- `ToVTKVertsAndData()`
- `DispatchElementType(ElemType t, Func&& func)` —— 编译期 switch 覆盖所有单元类型

---

#### `src/Geom/PeriodicInfo.hpp`
**类**：`NodePeriodicBits` —— 紧凑 3-bit 周期性标志（P1/P2/P3），MPI 可通信
**类**：`NodeIndexPBI` —— `{index i, NodePeriodicBits pbi}`
**类**：`NodePeriodicBitsRow` —— `NodePeriodicBits*` 的原始指针行视图
**类**：`ArrayNodePeriodicBits` —— `ParArray<NodePeriodicBits>`，`operator[]` 返回 `NodePeriodicBitsRow`
**结构**：`Periodicity`
- `TransCoord()` / `TransCoordBack()` —— 坐标变换
- `TransVector()` / `TransVectorBack()` —— 向量变换
- `TransMat()` / `TransMatBack()` —— 矩阵变换
- `GetCoordByBits()` / `GetCoordBackByBits()` / `GetVectorByBits()` / `GetVectorBackByBits()` —— 按位应用周期性

---

#### `src/Geom/Quadrature.hpp`
**类**：`Quadrature`
- `Integration(acc, f)` —— 通用积分 `f(acc, iG, pParam, D01Nj)`
- `IntegrationSimple(acc, f)` —— 无 shape function 的简单积分
- `GetQuadraturePointInfo(iG)` → `(pParam, w)`
- `GetWeight(iG)` / `GetNumPoints()`

---

#### `src/Geom/OpenFOAMMesh.hpp`
**结构**：`OpenFOAMBoundaryCondition` —— `{type, nFaces, startFace}`
**类**：`OpenFOAMReader` —— `ReadPoints()`, `ReadFaces()`, `ReadOwner()`, `ReadNeighbour()`, `ReadBoundary()`
**类**：`OpenFOAMConverter` —— `BuildFaceElemInfo()`, `BuildCell2Face()`, `BuildCell2Node()`

---

## 4. DNDS 核心测试程序

| 程序 | 入口文件 | 说明 |
|------|---------|------|
| `array_Test` | `app/DNDS/array_Test.cpp` | 测试全部 5 种 `Array` 布局（StaticFixed, Fixed, StaticMax, Max, CSR）。测试 resize、元素访问、压缩/解压往返、JSON 序列化读写。 |
| `arrayTrans_test` | `app/DNDS/arrayTrans_test.cpp` | MPI 测试 `ArrayTransformer` / `ParArray`。CSR 与 TABLE_Fixed 幽灵拉取（`pullOnce()`）、大规模随机幽灵拉取、复合类型（`std::array`, `Eigen::Matrix`）。 |
| `arrayDOF_test` | `app/DNDS/arrayDOF_test.cpp` | 测试 `ArrayDof<3, DynamicSize>`：resize、`setConstant`、`norm2`，以及 CUDA 设备分派（Host + Device 后端）。 |
| `arrayDerived_test` | `app/DNDS/arrayDerived_test.cpp` | 测试派生数组：`ArrayAdjacency`、`ArrayEigenVector`、`ArrayEigenMatrix`、`ArrayEigenUniMatrixBatch`、`ArrayEigenMatrixBatch`。每项创建数据、压缩、JSON 序列化往返。 |
| `objectPool_test` | `app/DNDS/objectPool_test.cpp` | 测试 `ObjectPool<std::string>`：初始化、获取、自动分配、跨池安全。 |
| `serializerH5_Test` | `app/DNDS/serializerH5_Test.cpp` | MPI 测试 `SerializerH5`：写/读 real/index/rowsize 向量、共享索引向量、uint8 数组（`Parts` 与显式分布式偏移），验证正确性。 |
| `serializerJSON_Test` | `app/DNDS/serializerJSON_Test.cpp` | 串行测试 `SerializerJSON`（base64 编码）：写/读 real/rowsize 向量、共享索引向量、uint8 数组。 |
| `stdPowerTest` | `app/DNDS/stdPowerTest.cpp` | MPI 微基准：测量 `std::pow` 在各值区间的性能，使用 `PerformanceTimer` 与 `IntervalRecorder`。 |

### 涉及核心类

#### `src/DNDS/Array.hpp`
**类**：`Array<T, _row_size, _row_max, _align>`
- `Size()`, `RowSize()`, `RowSize(index)`, `RowSizeMax()`, `DataSize()`, `DataSizeBytes()`
- `Resize(index nSize)`, `Resize(index nSize, rowsize nRowSize)`, `ResizeRow(index iRow, rowsize nRowSize)`, `ReserveRow()`
- `Compress()` / `Decompress()` / `IfCompressed()` / `CSRCompress()` / `CSRDecompress()`
- `operator()(index iRow, rowsize iCol)`, `operator[](index iRow)`, `at()`
- `clone()`, `CopyData()`, `CopyRowFrom()`, `SwapData()`
- `WriteSerializer()`, `ReadSerializer()`, `ReadSerializerMeta()`
- `to_device()`, `to_host()`, `clear_device()`, `deviceView<B>()`
- `begin<B>()`, `end<B>()`

#### `src/DNDS/ArrayTransformer.hpp`
**类**：`ParArray<T, _row_size, _row_max, _align>`（继承 `Array`）
- `setMPI()`, `getMPI()`, `createGlobalMapping()`, `globalSize()`
- `WriteSerializer()`, `ReadSerializer()`
- `setDataType()`, `getDataType()`

**类**：`ArrayTransformer<T, _row_size, _row_max, _align>`
- `setFatherSon()`, `createFatherGlobalMapping()`, `createGhostMapping()`（pull/push）
- `createMPITypes()`
- `initPersistentPull()` / `startPersistentPull()` / `waitPersistentPull()` / `clearPersistentPull()`（及 Push 等价物）
- `pullOnce()`, `pushOnce()`

#### `src/DNDS/ArrayPair.hpp`
**类**：`ArrayPair<TArray>`
- `InitPair(name, args...)`, `TransAttach()`
- `BorrowAndPull(primary)` / `BorrowSetup(primary)`
- `operator[]`, `operator()`, `RowSize()`, `ResizeRow()`, `Size()`
- `deviceView<B>()`, `to_device()`, `to_host()`
- `WriteSerialize()`, `ReadSerialize()`, `ReadSerializeRedistributed()`

#### `src/DNDS/ArrayDOF.hpp`
**类**：`ArrayDof<n_m, n_n>`（继承 `ArrayEigenMatrixPair`）
- `setConstant(real)` / `setConstant(Eigen::Matrix)`
- `operator+=`, `operator-=`, `operator*=`, `operator/=`
- `addTo(other, scale)`
- `norm2()`, `dot(other)`, `min()`, `max()`, `sum()` —— 均为 MPI 集体
- `componentWiseNorm1()`
- `deviceView<B>()`, `to_device()`, `to_host()`

#### `src/DNDS/IndexMapping.hpp`
**类**：`GlobalOffsetsMapping`
- `setMPIAlignBcast(mpi, myLength)` —— 通过 MPI_Bcast 构建秩长度/偏移
- `operator()(rank, val)` —— local → global
- `search(globalQuery, rank, val)` —— global → local

**类**：`OffsetAscendIndexMapping`
- 构造函数：pull-based / push-based
- `sort()`, `ghostAt(rank, ighost)`
- `searchInMain()`, `searchInGhost()`, `searchInAllGhost()`
- `search()` → `(bool, rank, val)`
- `operator()(rank, val)` —— reverse mapping

#### `src/DNDS/MPI.hpp`
**结构**：`MPIInfo` —— `{comm, rank, size}`，`setWorld()`
**类**：`MPITypePairHolder` —— RAII MPI 数据类型包装
**类**：`MPIReqHolder` —— RAII MPI 请求包装
**类**：`MPIBufferHandler` —— 单例 MPI 缓冲发送缓冲区
**类**：`MPI::CommStrategy` —— 单例，控制数组通信策略（`HIndexed` / `InSituPack`）
**类**：`MPI::ResourceRecycler` —— 单例，延迟 MPI 资源清理

**自由函数**：
- `BasicType_To_MPIIntType<T>()` → `pair<MPI_Datatype, MPI_int>`
- `MPIWorldSize()`, `MPIWorldRank()`
- `MPI::Init_thread()`, `MPI::Finalize()`
- `MPI::Bcast()`, `Alltoall()`, `Alltoallv()`, `Allreduce()`, `Scan()`, `Allgather()`, `Barrier()`, `WaitallAuto()`
- `AllreduceOneReal()`, `AllreduceOneIndex()`
- `InsertCheck()` —— 带位置信息的 debug barrier
- `MPISerialDo(mpi, f)` —— 跨秩串行化执行

#### `src/DNDS/SerializerBase.hpp`
**类**：`ArrayGlobalOffset` —— `{size, offset}`，`isDist()`, `size()`, `offset()`
**类**：`SerializerBase`（抽象）
- `OpenFile()`, `CloseFile()`, `CreatePath()`, `GoToPath()`, `IsPerRank()`
- `WriteInt()` / `WriteIndex()` / `WriteReal()` / `WriteString()`
- `WriteIndexVector()` / `WriteRowsizeVector()` / `WriteRealVector()` / `WriteSharedIndexVector()` / `WriteUint8Array()`
- `ReadInt()` / `ReadIndex()` / `ReadReal()` / `ReadString()`
- `ReadIndexVector()` / `ReadRowsizeVector()` / `ReadRealVector()` / `ReadSharedIndexVector()` / `ReadUint8Array()`

#### `src/DNDS/DeviceStorage.hpp`
**类**：`DeviceStorageBase` —— `raw_ptr()`, `copy_host_to_device()`, `copy_device_to_host()`, `bytes()`, `backend()`
**类**：`DeviceStorage<B>` —— 按后端（Host/CUDA）特化
**类**：`device_storage_factory<B>` —— `device_storage_create_unique()` / `device_storage_create_shared()`

#### `src/DNDS/ConfigRegistry.hpp`
**枚举**：`ConfigTypeTag` —— `Bool`, `Int`, `Real`, `String`, `Enum`, `Array`, `Object`, `ArrayOfObjects`, `MapOfObjects`, `Json`
**结构**：`FieldMeta` —— 单配置字段描述符：`name`, `description`, `typeTag`, `readField()`, `writeField()`, `schemaEntry()`
**结构**：`ConfigContext` —— 运行时上下文：`nVars`, `dim`, `gDim`, `modelCode`
**结构**：`CheckResult` —— `{passed, message}`
**类**：`ConfigRegistry<T>`（单例）
- `registerField()`, `registerCheck()`, `registerContextualCheck()`, `registerPostReadHook()`
- `readFromJson(j, obj)`, `writeToJson(j, obj)`
- `emitSchema(description)` → JSON Schema (draft-07)
- `validate(obj)`, `validateWithContext(obj, ctx)`, `validateKeys(userJson)`

---

## 5. Solver 线性/ODE 测试

| 程序 | 入口文件 | 说明 |
|------|---------|------|
| `krylovTest` | `app/Solver/krylovTest.cpp` | 独立 C++ 测试 `Linear.hpp` 求解器（GMRES 与 PCG）。定义 `WrappedVXD`（`Eigen::VectorXd` 薄包装），`testPCG()` 构造随机 SPD 矩阵 `A=B*B^T`，对角预处理 PCG 求解；`testGMRES()` 相同矩阵对角左预处理 GMRES 求解。打印迭代与残差。 |

### 涉及核心类

#### `src/Solver/ODE.hpp`
**抽象基类**：`template <class TDATA, class TDTAU> class ImplicitDualTimeStep`
- `Step(x, xinc, frhs, fdt, fsolve, maxIter, fstop, fincrement, dt)` —— 纯虚
- `getLatestRHS()`, `getRHS(i)`, `getRES(i)`
- `SetExtraParams(json)`

**具体积分器**：
| 类 | 说明 |
|----|------|
| `ImplicitEulerDualTimeStep` | 1 阶后向 Euler 伪时间步 |
| `ImplicitSDIRK4DualTimeStep` | SDIRK 方案（schemeCode 0=SDIRK4, 1=ESDIRK4, 2=ESDIRK3, 3=Trapezoid, 4=ESDIRK2），多阶段 Butcher 表 |
| `ImplicitBDFDualTimeStep` | BDF1–BDF4 固定阶，`StepPP()` 物理基础变体 |
| `ImplicitVBDFDualTimeStep` | 变步长 BDF（k≤2），`VBDFFrontMatters()` 变系数，`LimitDt_StepPPV2()` 自适应 dt 限制 |
| `ImplicitHermite3SimpleJacobianDualStep` | Hermite-3 双时间步（中点积分），可选 pMG 光滑，`SetCoefs(hR1)` 设置插值/积分权重 |
| `ExplicitSSPRK3TimeStepAsImplicitDualTimeStep` | 显式 SSPRK3 包装为隐式双时间接口 |

#### `src/Solver/Linear.hpp`
**类**：`template <class TDATA> class GMRES_LeftPreconditioned`
- `solve(FA, FML, fDot, b, x, nRestart, FStop)` —— 左预处理重启 GMRES，Gram-Schmidt Arnoldi，最小二乘求解，早停

**类**：`template <class TDATA, class TScalar> class PCG_PreconditionedRes`
- `reset()`, `getPHistorySize()`
- `solve(FA, FM, FResPrec, fDot, x, niter, FStop)` —— 残差预处理的灵活 PCG，支持 warm-start

#### `src/Solver/Direct.hpp`
**结构**：`DirectPrecControl` —— `useDirectPrec`, `iluCode`, `orderingCode`
**结构**：`SerialSymLUStructure`
- `ObtainSymmetricSymbolicFactorization(cell2cellFaceV, localPartStarts, iluCode)` —— 构建 ILU(k) 或完整 LU 符号模式

**结构**：`template <class Derived, class tComponent, class tVec> struct LocalLUBase`
- `InPlaceDecompose()` / `InPlaceDecomposeV1()` / `InPlaceDecomposeV2()` —— 左看 LU，OpenMP 分区并行
- `MatMul(x, result)` —— 分解前矩阵-向量乘
- `Solve(b, result)` —— 前/后向求解

**结构**：`template <class Derived, class tComponent, class tVec> struct LocalLDLTBase`
- `InPlaceDecompose()` —— 对称 LDL^T 分解
- `MatMul(x, result)` / `Solve(b, result)`

#### `src/Solver/Scalar.hpp`
**自由函数**：`template <class TF> real BisectSolveLower(TF&& F, real v0, real v1, real fTarget, int maxIter)` —— 二分求根，返回满足 `F(v0) <= fTarget` 的下界

---

## 6. 外部依赖测试程序

| 程序 | 入口文件 | 说明 |
|------|---------|------|
| `cgns_APITest` | `app/external/cgns_APITest.cpp` | CGNS API 基础测试 |
| `eigen_Test` | `app/external/eigen_Test.cpp` | Eigen 库基础测试 |
| `STL_Test` | `app/external/STL_Test.cpp` | C++ STL 测试 |
| `json_Test` | `app/external/json_Test.cpp` | nlohmann_json 测试 |
| `cgal_AABBTest` | `app/external/cgal_AABBTest.cpp` | CGAL AABB 树测试 |
| `mpi_test` | `app/external/mpi_test.cpp` | MPI 基础测试 |
| `cuda_test` | `app/external/cuda_test.cu` | CUDA 基础测试（仅在 CUDA 启用时构建） |

---

## 7. 支撑库：DNDS 核心模块

| 文件 | 核心内容 |
|------|---------|
| `Defines.hpp` | 全局类型别名：`real`/`index`/`rowsize`/`ssp<T>`；常量 `DynamicSize`/`NonUniformSize`/`NoAlign`；哨兵值；`Empty`/`EmptyNoDefault`/`ObjectNaming` |
| `MPI.hpp` / `MPI.cpp` | `MPIInfo`；`MPITypePairHolder`/`MPIReqHolder`；`MPIBufferHandler`/`ResourceRecycler`/`CommStrategy`；MPI 包装函数（`Allreduce`, `Alltoallv`, `Bcast`, `Barrier` 等） |
| `ArrayBasic.hpp` | `DataLayout` 枚举；`ArrayLayout`；`ArrayView`；`ArrayIteratorBase` |
| `Array.hpp` | `Array<T,rs,rm,al>` —— 核心 2D 变长容器，5 种布局，序列化，设备传输 |
| `ArrayTransformer.hpp` | `ParArray`（MPI 感知 Array）；`ArrayTransformer`（幽灵通信管理） |
| `ArrayPair.hpp` | `ArrayPair<TArray>`（父子数组对）；设备视图 |
| `ArrayDerived/*.hpp` | `ArrayAdjacency`, `ArrayIndex`, `ArrayEigenVector`, `ArrayEigenMatrix`, `ArrayEigenMatrixBatch`, `ArrayEigenUniMatrixBatch`, `AdjacencyRow` |
| `ArrayDOF.hpp` | `ArrayDof<n_m,n_n>`（MPI 向量空间操作）；`ArrayDofOp`（Host/CUDA 分发） |
| `IndexMapping.hpp` | `GlobalOffsetsMapping`（全局偏移）；`OffsetAscendIndexMapping`（幽灵映射，pull/push） |
| `ArrayRedistributor.hpp` | `BuildRedistributionPullingIndex()`；`RedistributeArrayWithTransformer()` |
| `SerializerBase.hpp` / `.cpp` | `SerializerBase`（抽象读写接口）；`ArrayGlobalOffset` |
| `SerializerJSON.hpp` / `.cpp` | `SerializerJSON` —— 每秩 JSON 序列化 |
| `SerializerH5.hpp` / `.cpp` | `SerializerH5` —— 集体 HDF5 + MPI-IO 序列化 |
| `DeviceStorage.hpp` / `.cpp` / `.cu` | `DeviceBackend`；`DeviceStorageBase`；`DeviceStorage<B>`；`device_storage_factory`；`host_device_vector` |
| `Vector.hpp` / `.cpp` | `vector_DeviceView`；`ArrayDeviceView` |
| `ConfigRegistry.hpp` / `ConfigParam.hpp` | `ConfigTypeTag`；`FieldMeta`；`ConfigContext`；`CheckResult`；`ConfigRegistry<T>`（单例注册表）；`ConfigSectionBuilder<T>` |
| `ObjectPool.hpp` | `ObjectPool<T>`（预分配对象池） |
| `Profiling.hpp` / `.cpp` | `PerformanceTimer`（单例挂钟计时器）；`ScalarStatistics`（Welford 算法） |
| `Errors.hpp` | `DNDS_assert` / `DNDS_assert_info` / `DNDS_check_throw` / `DNDS_check_throw_info` |
| `EigenUtil.hpp` / `HardEigen.hpp` | Eigen 辅助与高性能最小二乘求解 |
| `JsonUtil.hpp` | JSON 辅助函数 |
| `ExprtkWrapper.hpp` / `.cpp` | exprtk 表达式包装器 |
| `CsvLog.hpp` | CSV 日志辅助 |

---

## 8. 支撑库：Geom 几何网格模块

| 文件 | 核心内容 |
|------|---------|
| `Geometric.hpp` | 几何基础类型：`tPoint`/`tJacobi`/`tGPoint`/`tPointPortable`/`tGPointPortable`/`tSmallCoords` |
| `ElemEnum.hpp` | `ElemType` 枚举（Line2/3, Tri3/6, Quad4/9, Tet4/10, Hex8/27, Prism6/18, Pyramid5/14）；`ParamSpace` 枚举 |
| `ElementTraitsBase.hpp` / `ElementTraits.hpp` / `Elements.hpp` / `Elements/*.hpp` | `ElementTraits<ElemType>`（编译时元数据）；`ShapeFuncImpl<ElemType>`（形函数）；`Element`（运行时包装）；`DispatchElementType()` |
| `Mesh.hpp` / `.cpp` / `.cu` | `UnstructuredMesh`（中心分布式网格）；`UnstructuredMeshSerialRW`（串行读写）；`PartitionOptions` |
| `Mesh_DeviceView.hpp` | `UnstructuredMeshDeviceView<B>`（设备端网格视图）；数组别名（`tAdjPair`, `tCoordPair`, `tElemInfoArrayPair` 等） |
| `Mesh_Elevation.cpp` / `Mesh_Elevation_SmoothSolver.cpp` | `BuildO2FromO1Elevation()` 与 RBF 光滑求解 |
| `Mesh_PartitionHelpers.hpp` / `Mesh_Serial_*.cpp` | 分区、拓扑构建、重排序辅助 |
| `Mesh_Plts.cpp` / `Mesh_ReadSerializeDistributed.cpp` | VTK/Plt/CGNS 输出；分布式序列化读取 |
| `Mesh_WallDist.cpp` | `BuildNodeWallDist()` |
| `PeriodicInfo.hpp` / `.cpp` / `.cu` | `NodePeriodicBits`；`NodeIndexPBI`；`Periodicity`（旋转/平移/变换） |
| `BoundaryCondition.hpp` | BC ID 常量；`AutoAppendName2ID`；面分类辅助函数 |
| `Quadrature.hpp` / `QuadratureHub.hpp` / `Quadratures/*.hpp` | `Quadrature`（数值积分 API）；`GetQuadratureScheme()`；`GetQuadraturePoint()`；全局 `__TNBufferAtQuadrature` 缓存 |
| `PointCloud.hpp` | `PointCloudKDTree` / `PointCloudFunctional`（nanoflann 适配器） |
| `Octree.hpp` | `Octree`（占位符/未完成的八叉树） |
| `RadialBasisFunction.hpp` | `RBFKernelType`；RBF 插值辅助（`RBFCPC2`, `RBFInterpolateSolveCoefs`） |
| `OpenFOAMMesh.hpp` | `OpenFOAMBoundaryCondition`；`OpenFOAMReader`；`OpenFOAMConverter` |
| `CGNS.hpp` | CGNS↔DNDS 单元类型转换函数 |
| `BaseFunction.hpp` / `DiffTensors.hpp` / `EigenTensor.hpp` | 多项式基求值；导数张量坐标变换；`CFVPeriodicity`；`ETensorR3`（3 阶张量） |
| `SerialAdjReordering.hpp` / `CorrectRCM.hpp` / `Metis.hpp` | Metis 分区/重排序；Boost MMD/RCM；自定义 CorrectRCM；`OffsetIterator`/`OffsetRange` |
| `Grid.hpp` | `GetTanhDistributionBilateral()`（双端聚集 tanh 节点分布） |

---

## 9. 支撑库：CFV 高阶有限体积模块

| 文件 | 核心内容 |
|------|---------|
| `FiniteVolume.hpp` / `.cpp` | `FiniteVolume`（FV 基类：几何度量、单元/面属性、DOF 构建辅助） |
| `FiniteVolumeSettings.hpp` | `FiniteVolumeSettings`（POD 配置：`maxOrder`, `intOrder` 等） |
| `FiniteVolume_DeviceView.hpp` | `FiniteVolumeDeviceView<B>`（设备端 FV 视图） |
| `VariationalReconstruction.hpp` / `.cpp` / `_Reconstruction.hxx` / `_LimiterProcedure.hxx` | `VariationalReconstruction<dim>`（VR 高阶重构：基/权重、重构矩阵、SOR/Jacobi、限制器）；`VRBaseWeight` / `VRCoefficients` |
| `VRDefines.hpp` / `.cpp` / `.cu` | `RecAtr`；`GetRecDOFRange<dim>()`；大量 Array-pair 别名（`tURec`, `tUDof`, `tUGrad`, `tScalarPair` 等） |
| `VRSettings.hpp` | `VRSettings`（继承 `FiniteVolumeSettings`）；嵌套 `BaseSettings` / `FunctionalSettings`；枚举 `ScaleType`/`DirWeightScheme`/`GeomWeightScheme`/`AnisotropicType` |
| `Limiters.hpp` | 限制器核：WBAP 多路/双路、MINMOD、Van Leer、MEMM；`PolynomialSquaredNorm`/`PolynomialDotProduct` |
| `ModelEvaluator.hpp` / `.cpp` | `ModelEvaluator`（2D 对流-扩散测试模型） |
| `DOFFactory.hpp` | `BuildUDofOnMesh<>()` / `BuildUGradDOnMesh<>()`（DOF 数组工厂） |
| `BenchmarkFiniteVolume.hpp` / `.cpp` / `.cu` | 设备可调用 FV 梯度算子测试核（动态/固定/SOA 变体） |

---

## 10. 支撑库：Euler 求解器库

| 文件 | 核心内容 |
|------|---------|
| `Euler.hpp` | `EulerModel` / `RANSModel` 枚举；`ArrayDOFV` / `ArrayRECV` / `ArrayGRADV`；`JacobianValue` / `JacobianValue::Type` |
| `EulerEvaluator.hpp` / `.hxx` | `EulerEvaluator<model>`（核心残差/雅可比求值器：通量、源项、边界、LUSGS、限制器） |
| `EulerEvaluator_EvaluateDt.hxx` / `EvaluateRHS.hxx` | `EulerEvaluator` 的 `EvaluateDt` / `EvaluateRHS` 模板实现 |
| `EulerSolver.hpp` / `.hxx` / `_Init.hxx` / `_PrintData.hxx` | `EulerSolver<model>`（顶层驱动：网格 I/O、时间推进、隐式重构、输出、重启）；`Configuration` 及全部子配置结构 |
| `SingleBlockApp.hpp` | `RunSingleBlockConsoleApp<model>()` —— 所有 Euler 可执行文件的统一入口 |
| `EulerEvaluatorSettings.hpp` | `EulerEvaluatorSettings<model>`；`IdealGasProperty`；`FrameConstRotation`；`BoxInitializer` / `PlaneInitializer` / `ExprtkInitializer` |
| `EulerBC.hpp` | `EulerBCType`；`BoundaryHandler<model>`；`IntegrationRecorder`；`AnchorPointRecorder`；`OutputPicker`；`OneDimProfile` |
| `EulerJacobian.hpp` | `JacobianDiagBlock<nVarsFixed>`；`JacobianLocalLU<nVarsFixed>`；`JacobianLocalLDLT<nVarsFixed>` |
| `Gas.hpp` | `RiemannSolverType`；理想气体 EOS；`Roe`/`HLLC`/`HLLEP` 求解器；特征向量；通量函数族 |
| `CLDriver.hpp` | `CLDriverSettings`；`CLDriver`（自适应攻角驱动） |
| `SpecialFields.hpp` | `IsentropicVortex10` / `Vortex30` / `VortexCent`（解析初始场） |
| `RANS_ke.hpp` | SA / k-ω Wilcox / SST / Realizable k-ε 的 `GetMut` / `GetVisFlux` / `GetSource` 模板函数 |
| `_explicit_instantiation/*.cpp` | 9 个模型 × 6 组模板 = 54 个显式实例化 `.cpp` 文件 |

---

## 11. 支撑库：EulerP GPU 求值器

| 文件 | 核心内容 |
|------|---------|
| `EulerP.hpp` | 核心类型别名：`TU`/`TDiffU`/`TUDof`/`TUGrad` 等；`EvaluatorArgBase<TDerived>`（CRTP 参数束基类，`WaitAllPull`） |
| `EulerP_Evaluator.hpp` / `.cpp` / `_impl.hpp` / `_impl.cpp` / `.cu` | `Evaluator`（Host 端 GPU 求值器：面缓冲区、设备视图、核编排）；`EvaluatorConfig`；6 组 `*_Arg` 参数束 + 主机分派方法；`Evaluator_impl` 设备端包装 + 静态核入口 |
| `EulerP_Physics.hpp` | `PhysicsParams`；`PhysicsDeviceView<B>`（`Cons2Prim`, `Prim2Cons`, `Prim2Pressure`, `Prim2GammaAcousticSpeed`）；`Physics`（Host 端物理对象） |
| `EulerP_BC.hpp` / `.cpp` / `.cu` | `BCType`；`BCFunc_Impl<B,T>`（设备端 BC 特化）；`BC_DeviceView<B>`；`BC`（Host）；`BCHandlerDeviceView<B>`；`BCHandler`（Host）；`BCInput` |
| `EulerP_ARS.hpp` | `RoeEigenValueFixer`；`RoeAverageNS`；`RoeFluxFlow`（设备可调用 Roe 通量） |

---

## 12. 支撑库：Solver ODE/线性求解器

| 文件 | 核心内容 |
|------|---------|
| `ODE.hpp` | `ImplicitDualTimeStep<TDATA,TDTAU>`（抽象基类）；`ImplicitEulerDualTimeStep`；`ImplicitSDIRK4DualTimeStep`（SDIRK4/ESDIRK3/ESDIRK2/Trapezoid）；`ImplicitBDFDualTimeStep`（BDF1–4）；`ImplicitVBDFDualTimeStep`（变步长 BDF）；`ImplicitHermite3SimpleJacobianDualStep`（HM3 + pMG）；`ExplicitSSPRK3TimeStepAsImplicitDualTimeStep` |
| `Linear.hpp` | `GMRES_LeftPreconditioned<TDATA>`（重启左预 GMRES）；`PCG_PreconditionedRes<TDATA,TScalar>`（残差预处理灵活 PCG） |
| `Direct.hpp` | `DirectPrecControl`；`SerialSymLUStructure`（符号分解）；`LocalLUBase<Derived,tComponent,tVec>`（CRTP 局部 LU，V1/V2 算法，OpenMP）；`LocalLDLTBase<...>`（对称 LDL^T） |
| `Scalar.hpp` | `Scalar::BisectSolveLower<TF>()`（标量二分求根） |

---

## 13. Python 绑定模块

| 模块 | 入口文件 | 绑定内容 |
|------|---------|---------|
| `DNDS` (`dnds_pybind11`) | `src/DNDS/dnds_pybind11.cpp` | `MPIInfo`；`Array` 各布局；`ArrayPair`；`ArrayDof`；`SerializerBase`/`SerializerJSON`/`SerializerH5`；`ConfigRegistry`；`DeviceStorage` |
| `Geom` (`geom_pybind11`) | `src/Geom/geom_pybind11.cpp` | `UnstructuredMesh`；`Element`；`ElemType`；`Periodicity`；`NodePeriodicBits`；`Quadrature`；`PointCloudKDTree`；网格读取工具 |
| `CFV` (`cfv_pybind11`) | `src/CFV/cfv_pybind11.cpp` | `FiniteVolume`；`VariationalReconstruction_<dim>`；`RecAtr`；`tURec`/`tUDof`/`tUGrad` 别名；`ModelEvaluator`；`VRSettings` |
| `EulerP` (`eulerP_pybind11`) | `src/EulerP/eulerP_pybind11.cpp` | `Evaluator`；`Physics`；`BC`/`BCHandler`；`EvaluatorConfig`；设备端视图工厂 |

每个 pybind11 模块通过 `*_bind.hpp`/`*_bind.cpp` 文件暴露 C++ API。构建后通过 `cmake --install build --component py` 将 `.so` 安装到 `python/DNDSR/` 目录。
