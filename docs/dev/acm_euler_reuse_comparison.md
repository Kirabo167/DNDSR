<!--
File description: comparison and implementation recommendation for reusing Euler WBAP/CWBAP,
LU-SGS, and GMRES in the constant-density ACM solver.
Modifier: Runzhi Ma
Last modified: 2026-09-01
-->

# ACM 接入 WBAP/CWBAP、LU-SGS/GMRES 对比与实现报告

> 实现状态（2026-09-01）：本文推荐的混合方案已经落地。`src/Euler` 未修改；WBAP/CWBAP
> 复用 CFV 核，GMRES 复用通用 Krylov 类，LU-SGS 和全部 4×4 物理线性化位于 ACM 模块。

## 1. 结论

不建议把 `EulerSolver` 或 `EulerEvaluator` 的限制器与隐式求解函数直接连接到 ACM。
推荐采用“基础设施复用、物理算子重写”的混合方案：

- WBAP/CWBAP：复用 `CFV::VariationalReconstruction` 的通用限制器核，新增 ACM 的
  4 变量平滑指示器和特征空间正/逆变换；不复制限制器算法。
- GMRES：直接复用 `Solver::Linear::GMRES_LeftPreconditioned<TDof>`；ACM 提供矩阵向量积、
  左预条件器、全局内积和停止准则回调。
- LU-SGS：在 `src/ACM` 内重写 4×4 面 Jacobian、对角块和前后扫，不调用 Euler 的
  `LUSGSMatrix*`/`UpdateLUSGS*`。网格分区、邻接、MPI ghost 交换和扫描顺序可以沿用。

这样不会修改 `src/Euler`，也不会把密度、总能量、温度或湍流变量的假设泄漏到
ACM 状态 `[u,v,w,p]` 中。

## 2. 状态与算子差异

| 项目 | Euler/Navier–Stokes | 常密度 ACM | 对复用的影响 |
|---|---|---|---|
| 基本状态 | `[rho,rho*u,rho*v,(rho*w),rho*E,...]` | `[u,v,w,p]` | 变量下标和变量含义均不同 |
| 密度 | 方程未知量，必须保持正值 | 常数 `rho0` | Euler 正性限制不可直接使用 |
| 压力 | 由总能量和状态方程恢复 | 独立压力变量 | 平滑指示器和特征变换必须重写 |
| 温度/声速 | 理想气体热力学计算 | 无温度；波速由 `beta2/rho0` 决定 | Euler 特征矩阵不可复用 |
| 伪时间质量矩阵 | Euler/低马赫预处理定义 | `Gamma(u,p,beta2,alpha)` | 隐式时间对角块必须包含 ACM `Gamma` |
| 黏性项 | 动量、能量、可能含湍流 | 仅常密度动量扩散 | 黏性 Jacobian 维数与结构不同 |
| 边界线性化 | 对可压缩守恒量求导 | 对 `[u,v,w,p]` 的 ghost 映射求导 | Euler 边界 Jacobian 不适用 |

## 3. WBAP/CWBAP

### 3.1 直接接入的可行部分

`CFV::VariationalReconstruction<dim>::DoLimiterWBAP_C<nVarsFixed>` 和
`DoLimiterWBAP_3<nVarsFixed>` 是以变量数为模板参数的通用算法。它们接收：

- 单元均值 `tUDof<nVarsFixed>`；
- 重构系数 `tURec<nVarsFixed>`；
- 平滑指示器；
- `FM`/`FMI` 两个“物理空间与特征空间”转换回调。

因此，WBAP/CWBAP 的多邻居加权、逐阶限制、MPI 重构系数同步等主体逻辑可以直接复用。
二维 4 变量实例原本已经存在；本次增加了
`VariationalReconstruction_LimiterProcedure_3_4.cpp` 及对应 extern 声明，三维 ACM 现在使用
固定四变量实例，不会误用二维 Euler 的四变量语义。

### 3.2 不能直接复用的 Euler 部分

Euler 在 `EulerSolver.hxx` 中构造的 `fML/fMR`、平滑指示器后处理和正性控制依赖：

- 守恒量到原始量转换；
- 理想气体压力/声速；
- 密度、压力正性；
- Euler/RANS 的特征左右矩阵和额外方程。

这些回调即使变量数恰好相同，也不能用于 ACM。特别是二维 Euler 的 4 个守恒量与 ACM 的
4 个变量只是“维数相同”，物理意义完全不同。

### 3.3 直接接入与重写对比

| 方案 | 工作量 | 正确性风险 | 维护性 | 结论 |
|---|---:|---:|---:|---|
| 直接调用 Euler 限制器驱动 | 低 | 极高 | 差 | 不可行 |
| 完整复制 WBAP/CWBAP 到 ACM | 中高 | 中 | 差，算法会分叉 | 不推荐 |
| 复用 CFV 核，重写 ACM 回调 | 中 | 低 | 最好 | 推荐 |

### 3.4 已实现函数

`ACMEvaluator` 和 ACM 通量层已经实现：

```cpp
Matrix4 LeftEigenvectorsGlobal(...);
Matrix4 RightEigenvectorsGlobal(...);
bool TryCharacteristicMatricesGlobal(...);
void ACMEvaluator::ApplyCharacteristicLimiter(TDof &u);
```

特征矩阵由 ACM 的 `Gamma^{-1} A_n` 构造。当前已实现一般 `alpha` 下声学模态的切向
耦合，并由 Roe、WBAP/CWBAP 和远场边界共同调用。配置中的 `alpha` 不再限制为零。
当 `alpha*q_n^2=beta2/rho0` 导致声学波与切向波碰撞、完整特征基不存在时，Roe 和远场
边界使用谱簇公式，WBAP/CWBAP 则局部退回分量空间。
压力建议使用无量纲尺度 `p/(rho0*beta2)` 参与平滑指示器，避免速度和压力数值量级差异
导致限制器偏置。

## 4. LU-SGS

### 4.1 为什么 Euler 实现不能直接接入

Euler 的以下函数都直接调用可压缩通量 Jacobian、热力学状态、Euler 边界生成器和
Euler 状态索引：

- `EulerEvaluator::LUSGSMatrixInit`；
- `EulerEvaluator::LUSGSMatrixVec`；
- `EulerEvaluator::UpdateLUSGSForward/Backward`；
- `EulerEvaluator::LUSGSMatrixToJacobianLU`；
- `EulerEvaluator::LUSGSMatrixSolveJacobianLU`。

把 `nVarsFixed` 改成 4 不能解决问题，因为二维 Euler 的 4 变量是
`[rho,rho*u,rho*v,rho*E]`，而 ACM 是 `[u,v,w,p]`。此外 ACM 后向 Euler 方程的时间项是
`Gamma_i/dTau_i`，不是简单标量 `I/dTau_i`；若忽略这一点，当前 Jacobi 实现中已经验证过的
压力–速度耦合会在 LU-SGS 中丢失。

### 4.2 可沿用的结构

以下结构与具体方程无关，可以沿用或提取为公共帮助函数：

- `cell2face/face2cell` 邻接和本地单元扫描顺序；
- 周期邻居状态/增量旋转；
- MPI ghost increment pull；
- 前扫处理低序邻居、后扫处理高序邻居的 SGS 流程；
- DNDS 分布式数组和 4×4 小矩阵求解。

### 4.3 ACM 需要重写的算子

对残差 `R(U)` 和后向 Euler 伪时间离散，建议线性系统为：

```text
[ Gamma_i / dTau_i - dR_i/dU_i ] deltaU_i = defect_i
```

已经新增：

```cpp
void AssembleImplicitLinearization(...);
void ApplyImplicitLinearization(...);
void ApplyBlockJacobi(...);
void SolveLUSGS(...);
```

当前实现使用冻结的一阶数值差分面 Jacobian，保证边界链式求导和符号一致；后续可将内部面
的 4×4 ACM Roe/Rusanov Jacobian 替换为解析形式。边界面的外状态依赖
内部状态，因此必须包含 `dUghost/dUinside`，不能把 ghost state 当常量。

### 4.4 直接接入与重写对比

| 方案 | 代码改动 | 数学一致性 | 并行可用性 | 结论 |
|---|---:|---:|---:|---|
| 直接调用 Euler LU-SGS | 少 | 不成立 | 接口也不匹配 | 不可行 |
| 复制 Euler 文件后替换下标 | 大 | 容易残留热力学假设 | 可实现但风险高 | 不推荐 |
| ACM 重写 4×4 算子，沿用扫描/MPI 结构 | 中高 | 可验证 | 与现有分区兼容 | 推荐 |

## 5. GMRES

`Solver::Linear::GMRES_LeftPreconditioned<TDATA>` 只要求数据容器支持初始化、缩放、
`addTo`，并由调用方提供矩阵向量积、左预条件和内积。因此 GMRES 算法本身与 Euler 状态
无关，可以直接实例化为 ACM 的 `TDof`。

不能直接复用的是 Euler 提供给 GMRES 的两个回调：其 `FA` 调用 Euler
`LUSGSMatrixVec`，其 `FML` 调用 Euler LU/SGS 预条件器。ACM 应提供：

```cpp
FA(x, Ax)  -> ACMEvaluator::ApplyImplicitOperator(...)
FML(r, z)  -> ACM block-Jacobi 或 ACM LU-SGS sweep
fDot(x,y)  -> 本地 owned-cell 点积 + MPI_Allreduce(SUM)
```

当前已经支持“GMRES + 4×4 block-Jacobi”和“GMRES + ACM LU-SGS”两种左预条件配置。
GMRES 的矩阵向量积、预条件器和 MPI 全局点积均由 ACM 回调提供。

## 6. 已完成顺序与后续验收

1. 已增加三维 4 变量 CFV 限制器显式实例化。
2. 已实现并单测 ACM 特征正/逆变换。
3. 已接入可配置 WBAP/CWBAP，并保留局部极值限制器。
4. 已实现分布式一阶冻结隐式矩阵与边界链式线性化。
5. 已直接接入通用 GMRES 和 block-Jacobi 左预条件器。
6. 已实现 ACM LU-SGS 前后扫和跨 MPI rank 的 lagged ghost 耦合。
7. 已将 ACM LU-SGS 接为 GMRES 可选左预条件器。

最低验收要求：

- 二维、三维特征变换误差小于 `1e-11`；
- 隐式矩阵向量积与对应的一阶冻结残差有限差分方向导数相对误差小于 `1e-5`；
- 1/2/4/8 MPI rank 下残差历史在浮点容差内一致；
- WBAP/CWBAP 不产生非有限压力或速度；
- GMRES 与 LU-SGS 最终稳态残差、质量守恒误差和边界积分一致。
