<!--
文件说明：DNDSR 常密度三维 ACM 独立模块的详细实现、接口、测试与验收计划。
修改人：Runzhi Ma
修改日期：2026-08-31
-->

# DNDSR 常密度三维 ACM 独立模块详细实现计划

## 0. 文档定位

本文是针对 DNDSR 当前代码结构的实施计划，不是对附件内容的逐字转录。附件《ACM 预处理方法完整推导报告》只作为数学依据；本文中的目录、接口、阶段划分、测试和验收要求由当前工程约束决定。

本计划只覆盖第一阶段：

- 三维、常密度、单相不可压缩流；
- 新状态向量和 ACM 无黏通量/黎曼求解器；
- 层流黏性项作为可运行算例所需的配套能力；
- 稳态伪时间推进；
- 沿用现有网格、CFV 重构、MPI/OpenMP、ODE/线性求解和 JSON 配置机制；
- 新建独立 `ACM` 模块，不把 ACM 分支加入任何现有 `EulerModel`，不改变现有 Euler 算例的变量布局和数值行为。

本阶段明确不做：变密度、SA、两方程 RANS、双时间真实非定常、移动网格/ALE、CUDA 版本、压力存储方案 B 的生产启用。这些能力只预留接口，不进入首轮验收。

## 1. 目标和验收边界

### 1.1 最终交付物

实现完成后，工程应新增一个可执行程序 `acm3D`，能用与现有 `euler3D` 相同的方式启动：

```bash
mpirun -np 4 ./build/app/ACM/acm3D cases/acm3D/cavity.json
```

并继续支持：

```bash
./build/app/ACM/acm3D --emit-schema
./build/app/ACM/acm3D cases/acm3D/cavity.json \
  -k /implicitCFLControl/CFL -v 20 \
  -k /acmSettings/beta2 -v 4.0
```

首轮功能验收要求：

1. 状态固定为 `U = [u, v, w, p]^T`，常密度 `rho0` 从配置读取，不进入状态向量。
2. 物理通量、ACM 谱半径、Rusanov 通量和 Roe 通量均有独立单元测试。
3. 串行、2/4/8 MPI rank 的残差与最终场在容差内一致。
4. OpenMP 1/2/4 线程结果一致，不在面循环中直接竞争写单元残差。
5. 能运行均匀流保持、周期流、三维 lid-driven cavity 或同等级常密度层流算例。
6. 原 `euler3D`、`eulerSA3D`、`euler2EQ3D` 的目标、配置字段和测试结果不变。

### 1.2 “不修改 Euler 模型”的具体含义

本工作不得：

- 向 `EulerModel` 枚举添加 ACM 类型；
- 在 `EulerEvaluator`、`Gas.hpp` 或 Euler 边界函数中加入 ACM 的 `if constexpr`；
- 改变 Euler 的 `[rho, rho*u, rho*v, rho*w, E, ...]` 布局；
- 让现有 Euler 可执行程序链接 ACM 库；
- 用 ACM 配置字段替换或重解释 `eulerSettings`。

允许修改的公共工程接线仅限新增项，例如根 CMake 中加入 `src/ACM`、应用列表中加入 `app/ACM`、测试列表中加入 `acm_unit_tests`。这些修改不能改变已有目标的编译选项和链接顺序。

## 2. 数学规范冻结

### 2.1 状态和索引

首版只采用附件中的方案 A：

\[
U=(u,v,w,p)^T.
\]

代码中的唯一合法布局为：

```cpp
struct ACM3DLayout
{
    static constexpr int dim = 3;
    static constexpr int gDim = 3;
    static constexpr int nVars = 4;
    static constexpr int iU = 0;
    static constexpr int iV = 1;
    static constexpr int iW = 2;
    static constexpr int iP = 3;
};
```

这里的 `U` 是 ACM 伪时间系统的求解变量，也可称“伪守恒变量”。其中速度和压力并不构成可压缩 Euler 意义下的质量、动量、总能量守恒量，代码注释中应避免把第 4 项称为能量。

### 2.2 局部坐标物理通量

建立局部正交基 `B = [n, t1, t2]`，定义：

\[
q_n=u\cdot n,\qquad q_{t1}=u\cdot t_1,\qquad q_{t2}=u\cdot t_2.
\]

局部法向物理通量固定为：

\[
F_n(U)=
\begin{bmatrix}
q_n^2+p/\rho_0\\
q_nq_{t1}\\
q_nq_{t2}\\
q_n
\end{bmatrix}.
\]

前三项是局部速度方程通量，最后一项是连续性通量。返回全局坐标前，仅旋转前三项，压力/连续性通量分量保持标量。

对 `U_local=[qn,qt1,qt2,p]` 的正确通量 Jacobian 为：

\[
A_n=
\begin{bmatrix}
2q_n&0&0&1/\rho_0\\
q_{t1}&q_n&0&0\\
q_{t2}&0&q_n&0\\
1&0&0&0
\end{bmatrix}.
\]

单元测试必须显式锁定 `2*qn`、`qt1`、`qt2` 三项，防止把非守恒形式的系数矩阵误当成通量 Jacobian。

### 2.3 首版预处理矩阵和特征系统

附件不同章节存在 `alpha/gamma` 记号切换和个别中间式不一致。因此生产首版只启用：

- `pressureStorage = PhysicalP`；
- `alpha = 0`；
- `beta2 > 0`，全域常数；
- `rho0 > 0`，全域常数。

为了避免从互相矛盾的中间式推回实现，代码直接冻结以下相互可验证的对象。

令 `gammaT = 1 + alpha`。局部坐标中：

\[
\Gamma_A=
\begin{bmatrix}
1&0&0&\gamma_T q_n/\beta^2\\
0&1&0&\gamma_T q_{t1}/\beta^2\\
0&0&1&\gamma_T q_{t2}/\beta^2\\
0&0&0&1/\beta^2
\end{bmatrix},
\]

\[
\Gamma_A^{-1}=
\begin{bmatrix}
1&0&0&-\gamma_T q_n\\
0&1&0&-\gamma_T q_{t1}\\
0&0&1&-\gamma_T q_{t2}\\
0&0&0&\beta^2
\end{bmatrix},
\]

从而：

\[
\widetilde A_n=\Gamma_A^{-1}A_n=
\begin{bmatrix}
(1-\alpha)q_n&0&0&1/\rho_0\\
-\alpha q_{t1}&q_n&0&0\\
-\alpha q_{t2}&0&q_n&0\\
\beta^2&0&0&0
\end{bmatrix}.
\]

伪声波特征值为：

\[
\lambda_{\pm}=\frac{(1-\alpha)q_n\pm
\sqrt{(1-\alpha)^2q_n^2+4\beta^2/\rho_0}}{2},
\]

另有两个切向对流特征值 `lambda_t1=lambda_t2=qn`。首版 `alpha=0` 时：

\[
\lambda_{\pm}=\frac{q_n\pm\sqrt{q_n^2+4\beta^2/\rho_0}}{2}.
\]

所有 CFL、Rusanov 和 Roe 代码必须调用同一个 `ComputeEigenvalues()`，不得各自复制公式。

### 2.4 Roe 线性化约定

常密度通量是状态的二次/线性组合，首版采用算术 Roe 平均：

\[
\bar U=\frac{U_L+U_R}{2}.
\]

它对 `qn^2` 和 `qn*qt` 满足精确跳跃关系。首版 `alpha=0` 的局部右特征向量可取：

\[
r_{t1}=(0,1,0,0)^T,\quad
r_{t2}=(0,0,1,0)^T,\quad
r_{\pm}=(\lambda_{\pm}/\beta^2,0,0,1)^T.
\]

声学波强度由 `DeltaU_local = [dqn,dqt1,dqt2,dp]` 求得：

\[
a_- = \frac{\beta^2\Delta q_n-\lambda_+\Delta p}
{\lambda_- - \lambda_+},\qquad
a_+ = \frac{\beta^2\Delta q_n-\lambda_-\Delta p}
{\lambda_+ - \lambda_-}.
\]

切向波强度为 `a_t1=dqt1`、`a_t2=dqt2`。Roe 耗散必须按附件结论使用修正右特征向量：

\[
D=\sum_i |\lambda_i|a_i(\Gamma r_i),
\]

而不是直接使用 `r_i`。数值通量为：

\[
\widehat F_n=\frac{F_n(U_L)+F_n(U_R)}{2}-\frac{1}{2}D.
\]

首版熵修正采用统一函数：

```text
absFixed(lambda, delta) =
    abs(lambda),                         abs(lambda) >= delta
    (lambda^2 + delta^2) / (2*delta),   abs(lambda) < delta
```

其中 `delta = entropyFixRatio * max(abs(lambdaMinus), abs(lambdaPlus), abs(qn), beta/sqrt(rho0))`。默认 `entropyFixRatio=0.05`，设为 0 时关闭。

### 2.5 Rusanov 作为打通链路的第一求解器

先实现 Rusanov，再实现 Roe。对于预处理系统，耗散项使用：

\[
D_{LLF}=s_{max}\,\bar\Gamma\,(U_R-U_L),
\]

其中：

\[
s_{max}=\max(|q_n|,|\lambda_-|,|\lambda_+|).
\]

必须通过一个参考矩阵测试确认 `ApplyGamma(Ubar, deltaU)` 与显式构造 `Gamma(Ubar)*deltaU` 一致。若后续数学复核决定采用另一种 LLF 定义，只允许修改该函数和对应参考测试，不能在面装配中散布变体。

### 2.6 方案 B 只保留变换测试

附件给出的严格变换关系为：

\[
U_B=SU_A,\quad F_B=F_A,\quad A_{n,B}=A_{n,A}S^{-1},
\]

\[
\Gamma_B=\Gamma_A S^{-1},\quad
\widetilde A_{n,B}=S\widetilde A_{n,A}S^{-1},
\]

其中 `S=diag(1,1,1,1/beta2)`。首版不允许配置启用方案 B，但应写一个纯数学单元测试验证这些关系，作为以后开放 `p/beta2` 存储的保护网。

## 3. 当前代码复用边界

### 3.1 原样复用

| 能力 | 当前位置 | ACM 使用方式 |
|---|---|---|
| MPI 初始化、通信策略、归约 | `src/DNDS/` | 直接使用 |
| 分布式数组 father/son 与 persistent pull | `src/DNDS/`、`src/CFV/` | 使用相同 ghost 通信顺序 |
| 非结构网格、分区、周期映射 | `src/Geom/` | 直接使用 |
| 面/体积分和变分重构 | `src/CFV/` | `BuildUDof`、`BuildURec`、重构与积分接口不变 |
| ODE、GMRES、时间推进框架 | `src/Solver/` | 保持 evaluator 入口兼容 |
| JSON 注册、默认值、schema | `src/DNDS/Config/` | 使用 `DNDS_DECLARE_CONFIG`/`DNDS_FIELD` |
| JSON 默认配置 + merge patch + CLI JSON pointer 覆盖 | 现 `SingleBlockApp`/`ConfigureFromJson` 行为 | 在 ACM 驱动中保持同样顺序和参数 |
| JSON/HDF5 serializer | `src/DNDS/Serializer/` | 状态仍写入数据集 `u`，另写 ACM 元数据 |

### 3.2 只复用控制流，不复用物理实现

下列 Euler 文件只能作为调用顺序参考，ACM 不包含或调用其物理函数：

- `EulerEvaluator_EvaluateRHS.hxx`：复用“先 faceFluxBuf、后 cell gather”的并行模式；
- `EulerEvaluator_EvaluateDt.hxx`：复用“面谱半径、单元累加、MPI_MIN”的流程；
- `EulerSolver_Init.hxx`：复用网格/VFV/数组构建顺序；
- `EulerSolver_PrintData.hxx`：复用序列化和按原始 cell index 重分布流程；
- `SingleBlockApp.hpp`：复用 CLI 和两阶段配置读取行为。

不得复用：

- `Euler/Gas.hpp` 中任何依赖理想气体、总能量或声速的 Riemann 实现；
- `EulerBC.hpp` 对总压、总温、密度和能量的解释；
- Euler 的正性保持、Mach 数、温度、总焓转换；
- 固定索引 `0=rho`、`1..3=rho*u`、`4=E` 的周期旋转和输出代码。

### 3.3 开发时应对照的现有函数

这些位置用于理解框架协议，不表示 ACM 要继承 Euler 物理类：

| 现有位置 | 需要沿用的协议 | ACM 对应实现 |
|---|---|---|
| `src/Euler/SingleBlockApp.hpp`：`RunSingleBlockConsoleApp` | CLI、默认配置、patch、`-k/-v`、schema、启动顺序 | `src/ACM/SingleBlockApp.hpp` |
| `src/Euler/EulerSolver.hpp`：`Configuration::DNDS_DECLARE_CONFIG` | 顶层 section 名称和 schema 注册方式 | `ACMConfiguration::DNDS_DECLARE_CONFIG` |
| `src/Euler/EulerSolver.hpp`：`ConfigureFromJson` | 注释 JSON、merge patch、JSON pointer 覆盖、rank 0 写文件 | `ACMSolver::ConfigureFromJson` |
| `src/Euler/EulerSolver_Init.hxx`：`ReadMeshAndInitialize` | mesh/VFV/DOF/ghost/restart 初始化顺序 | `ACMSolver::ReadMeshAndInitialize` |
| `src/Euler/EulerEvaluator_EvaluateRHS.hxx`：`EvaluateRHS` | 重构 pull、面并行、`faceFluxBuf`、cell gather | `ACMEvaluator::EvaluateRHS` |
| `src/Euler/EulerEvaluator_EvaluateDt.hxx`：`EvaluateDt` | 面谱半径、cell CFL、`MPI_Allreduce(MPI_MIN)` | `ACMEvaluator::EvaluateDt` |
| `src/Euler/EulerEvaluator_EvaluateDt.hxx`：`fluxFace` | 批量积分点输入输出形状 | `ACMEvaluator::FluxFace` |
| `src/Euler/EulerEvaluator.hpp`：`UFromCell2Face` 等 | 周期面状态/梯度变换协议 | ACM 独立旋转实现 |
| `src/Euler/EulerSolver_PrintData.hxx`：restart 函数 | JSON/H5、origIndex、跨分区读取流程 | ACM 独立 restart 元数据与状态检查 |
| `src/Euler/Gas.hpp`：Riemann dispatcher | 只参考 dispatcher 的调用形状 | 不调用；改用 `ACMFlux.hpp` |

实现人员的推荐阅读顺序是：`SingleBlockApp` → `ConfigureFromJson` → `ReadMeshAndInitialize` → `EvaluateRHS` → `fluxFace` → `EvaluateDt` → restart。这样先掌握数据生命周期，再替换物理核。

## 4. 目标目录和文件职责

```text
src/ACM/
  ACM.hpp
  ACMState.hpp
  ACMSettings.hpp
  ACMFlux.hpp
  ACMFlux.hxx
  ACMViscousFlux.hpp
  ACMBC.hpp
  ACMConfiguration.hpp
  ACMEvaluator.hpp
  ACMEvaluator.hxx
  ACMEvaluator_EvaluateRHS.hxx
  ACMEvaluator_EvaluateDt.hxx
  ACMSolver.hpp
  ACMSolver.hxx
  ACMSolver_Init.hxx
  ACMSolver_PrintData.hxx
  SingleBlockApp.hpp
  CMakeLists.txt
  _explicit_instantiation/
    ACMEvaluator.cpp
    ACMEvaluator_EvaluateRHS.cpp
    ACMEvaluator_EvaluateDt.cpp
    ACMSolver.cpp
    ACMSolver_Init.cpp
    ACMSolver_PrintData.cpp

src/CFV/
  FlowSolution.hpp              新增；ACM 使用的通用 DOF/Rec 数组包装

app/ACM/
  acm3D.cpp

cases/
  acm3D_schema.json
  acm3D/
    config_base.json
    cavity.json
    uniform_periodic.json

test/cpp/ACM/
  test_ACMState.cpp
  test_ACMFlux.cpp
  test_ACMRiemann.cpp
  test_ACMBC.cpp
  test_ACMEvaluatorPipeline.cpp
  test_ACMParallel.cpp
```

`src/CFV/FlowSolution.hpp` 只新增 ACM 所需的通用数组包装，不要求本阶段迁移或编辑 `src/Euler/Euler.hpp`。这样 ACM 不必反向依赖 Euler，现有 Euler 又完全不受影响。后续若要统一重复代码，应单独提交一次仅做别名迁移的基础设施重构。

## 5. 类型、状态和设置接口

### 5.1 `ACM.hpp`

建议接口：

```cpp
namespace DNDS::ACM
{
enum class ACMModel
{
    ConstantDensity3D
};

enum class PressureStorage
{
    PhysicalP,
    ScaledPOverBeta2
};

enum class ACMRiemannSolverType
{
    Rusanov,
    Roe
};

template <ACMModel model>
struct ACMModelTraits;

template <>
struct ACMModelTraits<ACMModel::ConstantDensity3D>
{
    static constexpr int dim = 3;
    static constexpr int gDim = 3;
    static constexpr int nVarsFixed = 4;
    static constexpr int velocityBegin = 0;
    static constexpr int pressureIndex = 3;
    static constexpr int nExtraVars = 0;
};
}
```

不为未来湍流提前增加占位分量；未来通过新 model trait 追加变量，保证基础四变量索引不变。

### 5.2 `ACMState.hpp`

必须提供单一状态解释入口：

```cpp
template <class TU>
Eigen::Vector3d Velocity(const TU &U);

template <class TU>
real StoredPressure(const TU &U);

template <class TU>
real PhysicalPressure(const TU &U, const ACMSettings &settings);

template <class TU>
void SetPhysicalPressure(TU &U, real p, const ACMSettings &settings);

template <class TU>
Eigen::Vector4d ToLocalState(const TU &U, const Eigen::Matrix3d &normBase);

template <class TU>
TU FromLocalFlux(const Eigen::Vector4d &FLocal, const Eigen::Matrix3d &normBase);

template <class TU>
void RotateStateInPlace(TU &U, const Eigen::Matrix3d &rotation);

template <class TGrad>
void RotateGradientInPlace(TGrad &gradU,
                           const Eigen::Matrix3d &coordRotation,
                           const Eigen::Matrix3d &vectorRotation);

bool IsFiniteState(const Eigen::Vector4d &U);
```

约束：

- 只有前三个分量参与向量旋转；
- 压力只做存储标度转换，不做旋转；
- 常密度不做 `rho` 除法；
- 不调用 `IdealGasThermalConservative2Primitive`；
- 面周期变换与输出都调用这里，禁止各写一份裸下标逻辑。

### 5.3 `ACMSettings.hpp`

首版配置结构：

```cpp
struct ACMSettings
{
    real rho0 = 1.0;
    real beta2 = 1.0;
    real alpha = 0.0;
    real dynamicViscosity = 0.0;
    ACMRiemannSolverType riemannSolverType = ACMRiemannSolverType::Roe;
    PressureStorage pressureStorage = PressureStorage::PhysicalP;
    real entropyFixRatio = 0.05;
    real pressureReference = 0.0;
    Eigen::Vector4d farFieldValue = Eigen::Vector4d::Zero();
    bool enableViscousFlux = true;
    bool enableMovingMesh = false;

    DNDS_DECLARE_CONFIG(ACMSettings)
    {
        DNDS_FIELD(rho0, "Constant density", DNDS::Config::range(0.0));
        DNDS_FIELD(beta2, "Artificial compressibility beta squared", DNDS::Config::range(0.0));
        DNDS_FIELD(alpha, "Turkel alpha; production MVP requires zero");
        DNDS_FIELD(dynamicViscosity, "Dynamic viscosity", DNDS::Config::range(0.0));
        config.field_alias(&T::riemannSolverType, "riemannSolverType", "ACM Riemann solver");
        DNDS_FIELD(pressureStorage, "Pressure storage policy");
        DNDS_FIELD(entropyFixRatio, "Entropy fix ratio", DNDS::Config::range(0.0));
        DNDS_FIELD(pressureReference, "Pressure gauge reference");
        DNDS_FIELD(farFieldValue, "Far-field [u,v,w,p]");
        DNDS_FIELD(enableViscousFlux, "Enable laminar viscous flux");
        DNDS_FIELD(enableMovingMesh, "Moving mesh switch; unsupported in MVP");
    }

    void Validate() const;
};
```

`Validate()` 首版必须拒绝：

```text
rho0 <= 0
beta2 <= 0
alpha != 0
pressureStorage != PhysicalP
enableMovingMesh == true
farFieldValue.size != 4
任意非有限数
```

拒绝未完成能力比静默采用错误公式更安全。以后开放 `alpha` 或方案 B 时，只放宽验证和增加对应测试，不改变调用接口。

## 6. ACM 通量与黎曼求解器的具体函数

### 6.1 数据结构

```cpp
struct ACMEigenvalues
{
    real lambdaMinus;
    real lambdaTangential;
    real lambdaPlus;

    real SpectralRadius() const;
};

struct ACMFluxResult
{
    Eigen::Vector4d fluxGlobal;
    ACMEigenvalues eigenvalues;
};
```

保留三类特征值而不是只返回一个最大值，是为了与当前 evaluator 的 `lambdaFace0/lambdaFace123/lambdaFace4` 数据流兼容。映射定义为：

```text
lambdaFace0   <- abs(lambdaMinus)
lambdaFace123 <- abs(qn)
lambdaFace4   <- abs(lambdaPlus)
```

变量名可暂时兼容现有隐式框架，但 ACM 代码注释必须说明这里只是“负声学/切向/正声学”三组波，不代表 Euler 的 0/123/4 方程索引。

### 6.2 纯数学函数签名

`ACMFlux.hpp` 提供：

```cpp
Eigen::Vector4d PhysicalFluxLocal(
    const Eigen::Vector4d &ULocal,
    real rho0);

Eigen::Matrix4d PhysicalFluxJacobianLocal(
    const Eigen::Vector4d &UBarLocal,
    real rho0);

Eigen::Matrix4d GammaLocal(
    const Eigen::Vector4d &UBarLocal,
    real beta2,
    real alpha);

Eigen::Matrix4d GammaInvLocal(
    const Eigen::Vector4d &UBarLocal,
    real beta2,
    real alpha);

Eigen::Vector4d ApplyGammaLocal(
    const Eigen::Vector4d &UBarLocal,
    const Eigen::Vector4d &dU,
    real beta2,
    real alpha);

Eigen::Matrix4d PreconditionedJacobianLocal(
    const Eigen::Vector4d &UBarLocal,
    real rho0,
    real beta2,
    real alpha);

ACMEigenvalues ComputeEigenvalues(
    real qn,
    real rho0,
    real beta2,
    real alpha);

real EntropyFixedAbs(real lambda, real delta);

Eigen::Vector4d RoeDissipationLocalAlpha0(
    const Eigen::Vector4d &ULocal,
    const Eigen::Vector4d &URLocal,
    const ACMSettings &settings,
    ACMEigenvalues &eigenvalues);

Eigen::Vector4d RusanovDissipationLocal(
    const Eigen::Vector4d &ULocal,
    const Eigen::Vector4d &URLocal,
    const ACMSettings &settings,
    ACMEigenvalues &eigenvalues);

ACMFluxResult InviscidFluxACM(
    ACMRiemannSolverType type,
    const Eigen::Vector4d &UL,
    const Eigen::Vector4d &UR,
    const Eigen::Vector3d &unitNormal,
    const ACMSettings &settings);
```

这些函数不访问 mesh、不通信、不写全局数组，因而可被串行单测、面批处理和未来 CUDA 核共同使用。

### 6.3 物理通量伪代码

```text
function PhysicalFluxLocal(U=[qn, qt1, qt2, p], rho0):
    require rho0 > 0
    F[0] = qn * qn + p / rho0
    F[1] = qn * qt1
    F[2] = qn * qt2
    F[3] = qn
    return F
```

```text
function InviscidFluxACM(type, UL_global, UR_global, n, settings):
    require abs(norm(n) - 1) < geometryTolerance
    B = NormBuildLocalBaseV<3>(n)
    UL = [B^T * velocity(UL_global), pressure(UL_global)]
    UR = [B^T * velocity(UR_global), pressure(UR_global)]

    FL = PhysicalFluxLocal(UL, settings.rho0)
    FR = PhysicalFluxLocal(UR, settings.rho0)

    if type == Rusanov:
        D = RusanovDissipationLocal(UL, UR, settings, eig)
    else if type == Roe:
        D = RoeDissipationLocalAlpha0(UL, UR, settings, eig)
    else:
        throw invalid Riemann solver

    F_local = 0.5 * (FL + FR - D)
    F_global[0:3] = B * F_local[0:3]
    F_global[3] = F_local[3]
    require all finite(F_global)
    return {F_global, eig}
```

### 6.4 Roe 耗散伪代码

```text
function RoeDissipationLocalAlpha0(UL, UR, settings, eigOut):
    assert settings.alpha == 0
    Ubar = 0.5 * (UL + UR)
    dU = UR - UL
    qn = Ubar[0]

    eig = ComputeEigenvalues(qn, rho0, beta2, 0)
    deltaEntropy = entropyFixRatio *
        max(abs(eig.minus), abs(qn), abs(eig.plus), sqrt(beta2 / rho0))

    lm = EntropyFixedAbs(eig.minus, deltaEntropy)
    lt = EntropyFixedAbs(qn, deltaEntropy)
    lp = EntropyFixedAbs(eig.plus, deltaEntropy)

    denom = eig.plus - eig.minus
    require denom > smallReal

    aMinus = (eig.plus * dU[p] - beta2 * dU[qn]) / denom
    aPlus  = (beta2 * dU[qn] - eig.minus * dU[p]) / denom
    aT1 = dU[qt1]
    aT2 = dU[qt2]

    rMinus = [eig.minus / beta2, 0, 0, 1]
    rPlus  = [eig.plus  / beta2, 0, 0, 1]
    rT1    = [0, 1, 0, 0]
    rT2    = [0, 0, 1, 0]

    D = lm * aMinus * ApplyGammaLocal(Ubar, rMinus)
      + lp * aPlus  * ApplyGammaLocal(Ubar, rPlus)
      + lt * aT1    * ApplyGammaLocal(Ubar, rT1)
      + lt * aT2    * ApplyGammaLocal(Ubar, rT2)

    eigOut = eig
    return D
```

实现时要用“重构 `dU`”测试检查 `sum(a_i*r_i) == dU`，并用矩阵绝对值参考实现检查耗散向量。若这两项不成立，不得通过调整算例 CFL 掩盖问题。

### 6.5 Rusanov 伪代码

```text
function RusanovDissipationLocal(UL, UR, settings, eigOut):
    Ubar = 0.5 * (UL + UR)
    eig = ComputeEigenvalues(Ubar[qn], rho0, beta2, alpha)
    s = max(abs(eig.minus), abs(eig.tangential), abs(eig.plus))
    D = s * ApplyGammaLocal(Ubar, UR - UL, beta2, alpha)
    eigOut = eig
    return D
```

### 6.6 批量接口

为保持当前每个面的多积分点处理方式，再提供薄封装：

```cpp
void InviscidFluxACMBatch(
    ACMRiemannSolverType type,
    const TU_Batch &UL,
    const TU_Batch &UR,
    const TVec_Batch &unitNormals,
    TU_Batch &flux,
    TReal_Batch &lambdaMinus,
    TReal_Batch &lambdaTangential,
    TReal_Batch &lambdaPlus,
    const ACMSettings &settings);
```

首版内部按积分点调用标量核，不另写一套公式。性能数据证明这里成为热点后，再做 Eigen 向量化；向量化版本仍必须与标量核逐点对比。

## 7. 层流黏性通量

若目标算例包含 no-slip wall，仅有无黏通量无法完成 cavity 等验收，因此新增最小层流黏性核：

\[
\tau=\mu\left(\nabla u+\nabla u^T\right)
-\frac{2}{3}\mu(\nabla\cdot u)I.
\]

```cpp
Eigen::Vector4d ViscousFluxACM(
    const Eigen::Matrix<real, 3, 4> &gradU,
    const Eigen::Vector3d &unitNormal,
    real rho0,
    real dynamicViscosity);
```

伪代码：

```text
gradV = gradU[:, velocityColumns]
divV = trace(gradV)
tau = mu * (gradV + gradV^T) - (2/3) * mu * divV * I
Fvis[velocity] = tau * n / rho0
Fvis[pressure] = 0
return Fvis
```

总面通量沿用当前符号约定：`F_total = F_inviscid - F_viscous`。若当前 RHS 方向定义不同，以均匀剪切解析测试确定符号，不能凭注释猜测。

## 8. 边界条件独立实现

### 8.1 类型和 JSON

`ACMBC.hpp` 定义：

```cpp
enum class ACMBCType
{
    Unknown,
    FarField,
    VelocityInlet,
    PressureOutlet,
    NoSlipWall,
    SlipWall,
    Symmetry
};

class ACMBoundaryHandler
{
public:
    explicit ACMBoundaryHandler(int nVars = 4);
    ACMBCType GetTypeFromID(Geom::t_index id) const;
    const Eigen::Vector4d &GetValueFromID(Geom::t_index id) const;
    uint32_t GetFlagFromIDSoft(Geom::t_index id, const std::string &key) const;
    void RenewID2name();

    friend void from_json(const ordered_json &, ACMBoundaryHandler &);
    friend void to_json(ordered_json &, const ACMBoundaryHandler &);
};
```

JSON 条目继续使用现有风格：

```json
{
  "type": "NoSlipWall",
  "name": "wall",
  "value": [0.0, 0.0, 0.0, 0.0],
  "integrationOption": 1
}
```

保留 `name`、`value`、`integrationOption` 和几何 zone 名称映射机制，但 `value` 始终按 `[u,v,w,p]` 解释。

### 8.2 ghost 状态规则

统一入口：

```cpp
TU GenerateBoundaryValue(
    const TU &UL,
    const TU &UMean,
    index iCell,
    index iFace,
    int iG,
    const Eigen::Vector3d &unitNormal,
    const Eigen::Matrix3d &normBase,
    const Geom::tPoint &point,
    real time,
    Geom::t_index boundaryID,
    bool forReconstruction,
    int stage) const;
```

首版规则：

| 类型 | ghost 速度 | ghost 压力 |
|---|---|---|
| `FarField` | `2*uSpecified-uL` | `2*pSpecified-pL` |
| `VelocityInlet` | `2*uSpecified-uL` | `pL` |
| `PressureOutlet` | `uL` | `2*pSpecified-pL` |
| `NoSlipWall` | `2*uWall-uL` | `pL` |
| `SlipWall` | `uL-2*(uL-uWall)·n*n` | `pL` |
| `Symmetry` | `uL-2*(uL·n)*n` | `pL` |

这些是有限体积 ghost-state 基线规则。高阶重构边界约束必须通过同一入口，避免均值通量与重构通量使用不同边界定义。

压力只有梯度有物理意义。封闭域算例需要额外的 gauge 处理：每次完整非线性迭代后减去全局体积加权平均压力，再加 `pressureReference`。

```cpp
void RemovePressureNullSpace(ArrayDOFV<4> &u, real pressureReference);
```

```text
localPV = sum_i(p_i * volume_i)
localV  = sum_i(volume_i)
[globalPV, globalV] = MPI_Allreduce(sum)
pMean = globalPV / globalV
parallel for owned cells:
    U[i][p] += pressureReference - pMean
pull ghost values
```

只在没有 `PressureOutlet` 或其他压力锚定边界时启用；配置初始化阶段自动检测并记录日志。

## 9. Evaluator 接口和并行结构

### 9.1 `ACMEvaluator` 的兼容入口

```cpp
template <ACMModel model>
class ACMEvaluator
{
public:
    static constexpr int nVarsFixed = ACMModelTraits<model>::nVarsFixed;
    using TU = Eigen::Vector<real, nVarsFixed>;
    using TDof = CFV::FlowSolutionDOF<nVarsFixed>;
    using TRec = CFV::FlowSolutionRec<nVarsFixed>;

    ACMEvaluatorSettings settings;

    void InitializeUDOF(TDof &u);

    void EvaluateRHS(
        TDof &rhs,
        JacobianDiagBlock<nVarsFixed> &JSource,
        TDof &u,
        TRec &uRec,
        real time,
        uint64_t flags = RHS_No_Flags);

    void EvaluateDt(
        CFV::FlowSolutionDOF<1> &dt,
        TDof &u,
        TRec &uRec,
        real CFL,
        real &dtMinAll,
        real maxDt,
        bool useLocalDt,
        real time,
        uint64_t flags = DT_No_Flags);

    void FluxFace(
        const TU_Batch &UL,
        const TU_Batch &UR,
        const TDiffU_Batch &gradU,
        const TVec_Batch &unitNormal,
        TU_Batch &flux,
        TReal_Batch &lambdaMinus,
        TReal_Batch &lambdaTangential,
        TReal_Batch &lambdaPlus,
        Geom::t_index boundaryID,
        ACMRiemannSolverType rsType,
        index iFace,
        bool ignoreViscous);

    TU CompressRecPart(const TU &uMean, const TU &uRecInc, bool &compressed) const;
    void UFromCell2Face(TU &u, index iFace, index iCell, rowsize side) const;
    void UFromFace2Cell(TU &u, index iFace, index iCell, rowsize side) const;
    void DiffUFromCell2Face(TDiffU &grad, index iFace, index iCell,
                            rowsize side, bool reverse = false) const;
};
```

`CompressRecPart` 首版只做有限性检查和可配置的速度/压力增量上限，不做 Euler 密度/压力正性限制。默认不压缩，`compressed=false`。

### 9.2 `EvaluateRHS` 并行流程

必须保持当前两段式无竞争装配：

```text
function EvaluateRHS(rhs, JSource, u, uRec, time, flags):
    rhs = 0

    # 与当前方案相同：重构需要的 ghost 数据先通信
    ReconstructGradients(u, boundaryCallback)
    gradient.trans.startPersistentPull()
    gradient.trans.waitPersistentPull()

    ensure faceFluxBuf.size >= mesh.NumFaceProc()

    omp parallel for over iFace in [0, NumFaceProc):
        f2c = mesh.face2cell[iFace]
        build face quadrature
        reconstruct UL at every quadrature point

        if interior/partition face:
            reconstruct UR from neighbor/ghost cell
        else:
            UR = GenerateBoundaryValue(UL, ...)

        transform periodic states and gradients from cell frame to face frame
        FluxFace(UL_batch, UR_batch, grad_batch, normals, ...)
        integrate quadrature flux into one vector
        faceFluxBuf[iFace] = integratedFlux

    # 不在 face loop 内写 rhs[cell]
    omp parallel for over local partitions:
        for owned iCell in partition:
            for iFace in mesh.cell2face[iCell]:
                sign = orientation of cell against face
                fluxCell = sign * faceFluxBuf[iFace]
                rotate periodic flux from face frame to cell frame
                rhs[iCell] += fluxCell / cellVolume

    JSource = 0  # 常密度首版无体源项
    MPI consistency check
```

必须保留：

- `mesh->NumFaceProc()` 的处理范围；
- `mesh->NLocalParts()` / `LocalPartStart()` / `LocalPartEnd()` 的 cell gather；
- `faceFluxBuf[iFace]` 每面唯一写入；
- 周期面 `UFromCell2Face/UFromFace2Cell` 的方向处理；
- 当前 persistent pull 和 MPI 检查点。

不得在本工作中重新设计域分解、face ownership、halo 层数或 OpenMP 调度策略。

### 9.3 `FluxFace` 伪代码

```text
function FluxFace(ULBatch, URBatch, gradBatch, normalBatch, ...):
    resize outputs
    for each quadrature point k:
        inv = InviscidFluxACM(rsType, UL[k], UR[k], normal[k], settings)

        if ignoreViscous or not settings.enableViscousFlux:
            vis = 0
        else:
            vis = ViscousFluxACM(gradBatch[k], normal[k], rho0, dynamicViscosity)

        flux[k] = inv.fluxGlobal - vis
        lambdaMinus[k] = abs(inv.eigenvalues.lambdaMinus)
        lambdaTangential[k] = abs(inv.eigenvalues.lambdaTangential)
        lambdaPlus[k] = abs(inv.eigenvalues.lambdaPlus)
```

### 9.4 `EvaluateDt` 伪代码

```text
function EvaluateDt(dt, u, uRec, CFL, dtMinAll, maxDt, useLocalDt, time):
    reconstruct/pull gradients exactly as EvaluateRHS requires

    omp parallel for iFace in [0, NumFaceProc):
        build left/right mean state, applying periodic and BC transforms
        Ubar = 0.5 * (UL + UR)
        qn = dot(velocity(Ubar), unitNormal)
        eig = ComputeEigenvalues(qn, rho0, beta2, alpha)
        lambdaConv = eig.SpectralRadius()

        nu = dynamicViscosity / rho0
        volL = cellVolume(left)
        volR = right exists ? cellVolume(right) : volL
        lambdaVis = viscousSpectralFactor * nu * faceArea * (1/volL + 1/volR)

        lambdaFace[iFace] = lambdaConv + lambdaVis
        lambdaFace0[iFace] = abs(eig.minus)
        lambdaFace123[iFace] = abs(eig.tangential)
        lambdaFace4[iFace] = abs(eig.plus)

    localDtMin = +inf
    omp parallel for reduction(min: localDtMin) over owned cells:
        lambdaCell = sum_face(lambdaFace[face] * faceArea[face])
        dt[cell] = min(CFL * volume[cell] * smoothScale[cell]
                       / (lambdaCell + tiny), maxDt)
        localDtMin = min(localDtMin, dt[cell])

    MPI_Allreduce(localDtMin, dtMinAll, MPI_MIN)
    if not useLocalDt:
        dt.setConstant(dtMinAll)
```

`viscousSpectralFactor` 首版设为 2，并通过一维扩散稳定性测试校准；不得复用 Euler 中含 `gamma/prGas` 的热扩散表达式。

## 10. Solver、初始化、配置和 I/O

### 10.1 `ACMConfiguration`

顶层 JSON 字段尽量与现有算例保持同名：

```cpp
struct ACMConfiguration
{
    TimeMarchControl timeMarchControl;
    ImplicitReconstructionControl implicitReconstructionControl;
    OutputControl outputControl;
    ImplicitCFLControl implicitCFLControl;
    ConvergenceControl convergenceControl;
    DataIOControl dataIOControl;
    BoundaryDefinition boundaryDefinition;
    LimiterControl limiterControl;
    LinearSolverControl linearSolverControl;
    RestartState restartState;
    CFV::VRSettings vfvSettings;
    ACMSettings acmSettings;
    ordered_json bcSettings;
    std::map<std::string, std::string> bcNameMapping;

    DNDS_DECLARE_CONFIG(ACMConfiguration);
};
```

注册的顶层键：

```text
timeMarchControl
implicitReconstructionControl
outputControl
implicitCFLControl
convergenceControl
dataIOControl
boundaryDefinition
limiterControl
linearSolverControl
restartState
vfvSettings
acmSettings
bcSettings
bcNameMapping
```

其中公共 section 的字段名、默认值语义和 JSON 形状与现有 Euler 配置一致；物理 section 只使用 `acmSettings`。不得让 ACM 同时接受 `eulerSettings`，否则用户可能误以为理想气体参数有效。

首版为了严格不编辑 Euler，可在 `ACMConfiguration.hpp` 中定义 ACM 自己的公共控制结构，字段注册与现有配置保持一致。后续若要消除定义重复，再单独把公共结构提取到 `src/Solver/FlowSolverConfig.hpp`，并以无行为变化的提交迁移两个模块。

### 10.2 配置读取顺序必须保持一致

`ACMSolver::ConfigureFromJson` 签名：

```cpp
void ConfigureFromJson(
    const std::string &jsonName,
    bool read = false,
    const std::string &jsonMergeName = "",
    const std::vector<std::string> &overwriteKeys = {},
    const std::vector<std::string> &overwriteValues = {});
```

读取伪代码：

```text
if read == false:
    json = object
    config.ReadWriteJson(json, nVars=4, read=false)
    json["bcSettings"] = boundaryHandler defaults, if present
    rank 0 writes file
    MPI_Barrier
    return

base = parse(jsonName, allowComments=true)
if jsonMergeName not empty:
    patch = parse(jsonMergeName, allowComments=true)
    base.merge_patch(patch)

require overwriteKeys.size == overwriteValues.size
for each (key, valueText):
    key is a JSON pointer
    parse valueText as JSON value; if parsing fails, treat it as string
    base[key] = parsedValue

config.ReadWriteJson(base, 4, read=true)
config.acmSettings.Validate()
boundaryHandler = make_shared<ACMBoundaryHandler>(4)
from_json(config.bcSettings, *boundaryHandler)
ValidateBoundaryCompleteness()
base["bcSettings"] = *boundaryHandler
print resolved configuration on rank 0
```

合并优先级保持：

```text
编译内默认值 < config_base.json < 用户 case patch < CLI -k/-v
```

一个用户 case patch 不需要复制完整默认文件。例如 `cases/acm3D/uniform_periodic.json` 可采用：

```json
{
  "$schema": "../acm3D_schema.json",
  "timeMarchControl": {
    "steadyQuit": true,
    "useRestart": false
  },
  "convergenceControl": {
    "nTimeStepInternal": 5000,
    "rhsThresholdInternal": 1e-10
  },
  "implicitCFLControl": {
    "CFL": 5.0,
    "useLocalDt": true
  },
  "dataIOControl": {
    "meshFile": "../data/mesh/periodic_box.cgns",
    "outPltName": "../data/out/acm3D/uniform_periodic"
  },
  "acmSettings": {
    "rho0": 1.0,
    "beta2": 1.0,
    "alpha": 0.0,
    "dynamicViscosity": 0.0,
    "riemannSolverType": "Roe",
    "pressureStorage": "PhysicalP",
    "farFieldValue": [1.0, 0.0, 0.0, 0.0]
  },
  "bcSettings": [
    {
      "type": "FarField",
      "name": "far",
      "value": [1.0, 0.0, 0.0, 0.0]
    }
  ]
}
```

这里没有 `eulerSettings`。`vfvSettings`、`limiterControl`、`linearSolverControl`、输出格式和网格变换等未出现的公共部分全部从默认配置继承。

### 10.3 `SingleBlockApp`

```cpp
int RunSingleBlockConsoleApp(int argc, char *argv[]);
```

保持现有 CLI：

- 位置参数 `config`；
- 可重复的 `-k/--overwrite_key`；
- 可重复的 `-v/--overwrite_value`；
- `--debug`；
- `--emit-schema`。

固定应用名 `acm3D`，默认路径：

```text
../cases/acm3D/config_base.json
```

若用户传入 `cases/acm3D/cavity.json`，驱动会在该文件同目录查找可跟踪的
`config_base.json`，再将用户文件作为 merge patch。该名称不会匹配 `cases/.gitignore`
中的 `*default_config.json`，因此干净克隆也具备完整基线配置。

启动调用链：

```text
main
  -> MPI::Init_thread
  -> ACM::RunSingleBlockConsoleApp
  -> ACMSolver(mpi, 4)
  -> ConfigureFromJson(default, false)
  -> ConfigureFromJson(default, true, casePatch, CLI keys, CLI values)
  -> ReadMeshAndInitialize
  -> RunImplicitACM
  -> MPI_Finalize
```

### 10.4 初始化

`ReadMeshAndInitialize()` 沿用现有顺序：

```text
1. 按 dataIOControl 读取 CGNS/OpenFOAM/H5 网格
2. 分区、建立 connectivity、ghost 和周期映射
3. PrepareMesh，生成几何量
4. 构建 CFV::VariationalReconstruction<3>
5. vfv.BuildUDof(u, 4)
6. 构建 rhs、uInc、dt、重构系数、梯度和 lambda 数组
7. 构造 ACMEvaluator(mesh, vfv, boundaryHandler, settings)
8. 若 useRestart，读取 ACM restart
9. 否则 InitializeUDOF(u)
10. pull ghost solution
11. 生成初始输出
```

`InitializeUDOF`：

```text
u[cell] = acmSettings.farFieldValue
if initializer expression configured:
    expose x,y,z,t and U/UPrim (两者都按 [u,v,w,p])
    evaluate expression per owned cell
validate finite values
remove pressure null space if closed domain
pull ghost cells
```

首版不复用 Euler 的特殊激波、DMR、Noh 等 initializer code。

### 10.5 重启

复用 serializer 和数据集名 `u`，但新增元数据：

```text
solverFamily = "ACM"
model = "ConstantDensity3D"
stateLayout = ["U", "V", "W", "Pressure"]
stateVersion = 1
rho0
beta2
pressureStorage = "PhysicalP"
```

读取前必须检查：

- `solverFamily == ACM`；
- `stateVersion == 1`；
- 向量宽度为 4；
- `pressureStorage` 与当前设置一致；
- `rho0` 一致，或用户显式允许转换。

首版禁止直接把 Euler restart 的前四项映射成 ACM；`rho,rhou,rhov,rhow` 绝不是 `u,v,w,p`。以后若需要跨求解器初始化，必须新增有名字的转换工具，读取 Euler 的五变量并显式计算速度和压力。

### 10.6 输出

默认 cell fields：

```text
VelocityX
VelocityY
VelocityZ
Pressure
VelocityMagnitude
Divergence
VorticityMagnitude
ResidualU
ResidualV
ResidualW
ResidualContinuity
```

不得输出 Euler 的 Density、Temperature、Mach、TotalPressure、TotalTemperature。压力输出为物理压力，并可选同时输出 `PressureGauge = p-pressureReference`。

## 11. 隐式推进的接入策略

用户要求沿用并行计算方案，因此不重新设计 LU-SGS/GMRES。实施分两层：

### 11.1 第一可运行层

- 使用已有 ODE/伪时间控制；
- 使用标量谱半径 Jacobian（`useScalarJacobian=true`）；
- `EvaluateRHS`、`EvaluateDt`、数组运算和 GMRES 的 MPI/OpenMP 结构保持一致；
- 源项 Jacobian 为零；
- 不实现 ACM Roe block Jacobian。

这一层足以验证变量、通量、边界、并行残差和稳态收敛。

### 11.2 第二增强层

在第一层全部通过后，增加：

```cpp
Eigen::Vector4d FluxJacobianRightTimesDU(
    const Eigen::Vector4d &UBar,
    const Eigen::Vector4d &dU,
    const Eigen::Vector3d &normal,
    real lambdaScale) const;

void LUSGSMatrixInit(...);
void LUSGSMatrixVec(...);
void UpdateSGS(...);
```

其局部作用先用有限差分验证：

```text
Jv_FD = [F(U + eps*v) - F(U - eps*v)] / (2*eps)
Jv_impl = PhysicalFluxJacobianLocal(U) * v
relativeError(Jv_impl, Jv_FD) < 1e-7
```

首版只有方案 A，因此不需要实现方案 B 的 `D_B = D_A*S^-1`。以后开放方案 B 时必须单独验证该右乘关系，不能把 LU-SGS 对角块误作相似变换。

## 12. CMake 和显式实例化

### 12.1 根 CMake

新增：

```cmake
set(DNDS_ACM_MODELS_LIST
    ConstantDensity3D=3D
    CACHE INTERNAL "ACM model explicit instantiations")

add_subdirectory(${CMAKE_SOURCE_DIR}/src/ACM)
```

顺序放在 `CFV` 之后、应用和测试之前。`src/ACM/CMakeLists.txt` 生成：

```text
acm_library_fast_ConstantDensity3D
acm_library_ConstantDensity3D
```

链接只包含：

```text
cfv;geom;dnds;solver/common dependencies;DNDS_EXTERNAL_LIBS
```

不得链接 `euler_library_*`。

### 12.2 应用

`app/ACM/acm3D.cpp`：

```cpp
#include "ACM/SingleBlockApp.hpp"

int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    int errc = DNDS::ACM::RunSingleBlockConsoleApp(argc, argv);
    if (errc)
        MPI_Abort(MPI_COMM_WORLD, errc);
    MPI_Finalize();
    return errc;
}
```

在 `cmake/DndsApps.cmake` 中单独加入 `DNDS_APPS_ACM`，不把它塞进 `DNDS_APPS_Euler_Models`。

### 12.3 测试目标

新增：

```text
acm_test_state
acm_test_flux
acm_test_riemann
acm_test_bc
acm_test_evaluator_pipeline
acm_test_parallel
acm_unit_tests
```

CTest 名称使用 `acm_*` 前缀，MPI 测试按现有规则生成 `_np1/_np2/_np4/_np8`。

## 13. 测试矩阵

### 13.1 状态与旋转

1. 任意正交矩阵 `R` 下，`RotateState` 只改变前三项。
2. `R^-1(R(U)) == U`。
3. 局部状态转全局通量再转回局部保持一致。
4. 方案 A 压力读写恒等。
5. 非有限状态被拒绝。

### 13.2 Jacobian

对随机但有界的 `qn,qt1,qt2,p`：

```text
A_analytic = PhysicalFluxJacobianLocal(U)
A_FD[:,j] = [F(U+eps*e_j)-F(U-eps*e_j)]/(2*eps)
maxRelativeError < 1e-8
```

另加固定断言：

```text
A(0,0) == 2*qn
A(1,0) == qt1
A(2,0) == qt2
A(3,0) == 1
```

### 13.3 预处理矩阵与特征值

1. `GammaInv*Gamma == I`。
2. `PreconditionedJacobian == GammaInv*PhysicalJacobian`。
3. Eigen 数值特征值与 `ComputeEigenvalues` 一致。
4. `alpha=0, qn=0` 时 `lambda±=±sqrt(beta2/rho0)`。
5. 即使生产配置暂不开放，也测试 `alpha=-1,0,1` 的公式极限。
6. 方案 A/B 相似变换测试。

### 13.4 Riemann 通量

1. 一致性：`Flux(U,U,n) == PhysicalFlux(U,n)`。
2. 反对称：`Flux(UL,UR,n) == -Flux(UR,UL,-n)`。
3. 常量保持：均匀状态所有单元 RHS 接近机器零。
4. Roe 波强度重构 `sum(a_i*r_i)==dU`。
5. Roe 耗散与直接矩阵参考 `Gamma*R*abs(Lambda)*L*dU` 一致。
6. `UL≈UR` 时无 NaN，耗散随 jump 一阶趋零。
7. Rusanov 的谱半径不小于任一波绝对值。
8. `beta2/rho0` 缩放测试。

### 13.5 边界

1. no-slip wall 的面中值速度等于 wall velocity。
2. slip/symmetry 的面中值法向速度为零，切向速度不变。
3. pressure outlet 的面中值压力等于指定压力。
4. velocity inlet 的面中值速度等于指定速度。
5. 封闭域 pressure null-space 去除后全局体积平均压力等于参考值。

### 13.6 Evaluator 和并行

使用一个含内部面、物理边界、周期面和跨 rank 分区面的微型三维网格：

1. `np=1` 与 `np=2/4/8` 的全局 L2/Linf RHS 一致。
2. 相同 cell original index 上的 RHS 分量一致。
3. OpenMP 1/2/4 线程一致。
4. local dt 的全局最小值一致。
5. 每个 MPI rank 处理的 face/cell 计数归约后正确。
6. AddressSanitizer/ThreadSanitizer 可用构建下无越界和明显竞态。

容差建议：纯数学核 `1e-12~1e-9`，积分/并行归约 `1e-10~1e-8`，最终稳态场按离散误差设置。

### 13.7 算例验收

按顺序运行：

1. `uniform_periodic`：三维周期盒均匀速度/压力，RHS 接近零。
2. `pressure_pulse`：仅用于检查伪声波传播、Roe/Rusanov 稳定性和谱半径。
3. `lid_driven_cavity_3d`：低 Reynolds 数，检查无滑移、压力 gauge 和稳态收敛。
4. 可选 `channel_3d`：速度入口 + 压力出口，检查压降和质量守恒。

每个算例记录：全局连续性残差、三个速度残差、入口/出口体积流量、压力均值、迭代数和 wall time。

## 14. 分阶段实施清单

### 阶段 A：数学核，不接网格

新增 `ACM.hpp`、`ACMState.hpp`、`ACMSettings.hpp`、`ACMFlux.*`。

完成条件：

- 状态布局测试通过；
- 通量 Jacobian 有限差分测试通过；
- 预处理矩阵/特征值测试通过；
- Rusanov 和 Roe 的一致性、反对称和矩阵参考测试通过。

### 阶段 B：边界和层流黏性核

新增 `ACMBC.hpp`、`ACMViscousFlux.hpp`。

完成条件：所有 ghost-state 解析测试和黏性通量解析测试通过。

### 阶段 C：Evaluator 与现有并行结构接线

新增 `ACMEvaluator*`，只使用 scalar Jacobian 路径。

完成条件：

- 微型网格均匀流 RHS 为零；
- `np=1/2/4/8` 一致；
- OMP 线程数变化不改变结果；
- `EvaluateDt` 与手工谱半径一致。

### 阶段 D：配置、Solver、可执行程序

新增 `ACMConfiguration`、`ACMSolver*`、`SingleBlockApp`、CMake 和案例默认配置。

完成条件：

- `--emit-schema` 成功；
- 默认 + case patch + CLI override 的最终 JSON 正确；
- 可读取现有网格并完成初始化和稳态推进；
- 错误的 `alpha`、方案 B、移动网格在配置阶段给出明确错误。

### 阶段 E：I/O 与算例

新增 ACM 输出 picker、restart 元数据和 cavity/periodic 案例。

完成条件：

- HDF5 restart 可跨不同 MPI rank 数读取；
- JSON per-rank restart 在同一分区下可恢复；
- 输出字段不包含 Euler 热力学量；
- cavity 和 periodic 回归基线建立。

### 阶段 F：隐式增强

在基线全部稳定后，加入 ACM block Jacobian/LU-SGS 接口。

完成条件：矩阵-向量作用有限差分测试通过，且相对 scalar Jacobian 明确减少迭代时间；否则保留 scalar 路径为默认。

## 15. 建议的提交拆分

1. `ACM math kernel and unit tests`
2. `ACM boundary and viscous flux`
3. `ACM evaluator with existing MPI/OpenMP assembly`
4. `ACM configuration and single-block app`
5. `ACM restart/output and regression cases`
6. `ACM implicit block operator`（可选、后置）

每个提交都应能独立编译，并运行已存在的 Euler 单元测试。不要把数学核、配置迁移、并行接线和算例调参压在一个提交中。

## 16. 构建与验证命令

实现后建议：

```bash
cmake --preset release-test
cmake --build build -t acm3D acm_unit_tests euler_unit_tests -j 8
ctest --test-dir build -R "^acm_" --output-on-failure
ctest --test-dir build -R "^euler_" --output-on-failure
mpirun -np 1 ./build/app/ACM/acm3D cases/acm3D/uniform_periodic.json
mpirun -np 4 ./build/app/ACM/acm3D cases/acm3D/uniform_periodic.json
mpirun -np 4 ./build/app/ACM/acm3D cases/acm3D/cavity.json
```

若新增 Python 测试或改动任何 pybind C++ 源，必须按项目规则先构建并安装四个 pybind 目标，再运行 pytest。本计划本身不要求新增 Python 绑定。

## 17. 关键风险和停止条件

### 17.1 数学记号风险

附件中 `gamma=alpha+1`、`1-alpha` 以及 Chorin 解释存在局部不一致。代码评审应以本计划冻结的 `A_n`、`GammaInv`、`Atilde` 三者乘法关系和有限差分/数值特征值测试为准。若三者不能同时成立，停止接入 evaluator，先修正数学核。

### 17.2 压力零空间

封闭不可压缩域没有绝对压力锚点。若不去除均值，残差可以下降而压力整体漂移。该功能属于基线正确性，不应延后到“算例调参”。

### 17.3 变量误解释

最大的软件风险是把 ACM 的 `U[0]` 当密度、`U[3]` 当动量或把 `U[3]` 送入 Euler 能量转换。通过独立命名空间、独立 BC、独立输出、restart 元数据和禁止链接 Euler physics 来隔离。

### 17.4 并行竞态

不得为了快速打通而在 OpenMP face loop 中直接更新两个 cell 的 RHS。必须保留 face buffer + cell gather；否则串行测试通过也不能进入算例验收。

### 17.5 过早扩展

以下任一条件未满足前，不进入 SA/两方程 RANS、变密度或方案 B：

- Roe 数学核参考测试全部通过；
- uniform/cavity 的 MPI/OMP 一致性通过；
- restart 和配置 schema 稳定；
- Euler 回归无变化。

## 18. 首轮开发的函数完成清单

按依赖顺序，首轮至少实现并测试以下函数：

```text
ACMSettings::Validate
Velocity
PhysicalPressure
SetPhysicalPressure
ToLocalState
FromLocalFlux
RotateStateInPlace
RotateGradientInPlace
PhysicalFluxLocal
PhysicalFluxJacobianLocal
GammaLocal
GammaInvLocal
ApplyGammaLocal
PreconditionedJacobianLocal
ComputeEigenvalues
EntropyFixedAbs
RusanovDissipationLocal
RoeDissipationLocalAlpha0
InviscidFluxACM
InviscidFluxACMBatch
ViscousFluxACM
ACMBoundaryHandler::from_json
ACMBoundaryHandler::to_json
ACMEvaluator::GenerateBoundaryValue
ACMEvaluator::InitializeUDOF
ACMEvaluator::UFromCell2Face
ACMEvaluator::UFromFace2Cell
ACMEvaluator::DiffUFromCell2Face
ACMEvaluator::CompressRecPart
ACMEvaluator::FluxFace
ACMEvaluator::EvaluateRHS
ACMEvaluator::EvaluateDt
ACMSolver::ConfigureFromJson
ACMSolver::ReadMeshAndInitialize
ACMSolver::RunImplicitACM
ACMSolver::PrintData
ACMSolver::PrintRestart
ACMSolver::ReadRestart
ACMSolver::RemovePressureNullSpace
ACM::RunSingleBlockConsoleApp
```

这份清单完成后，常密度 ACM 已形成一个可单独编译、可并行运行、可配置、可重启、可继续扩展的模块。后续新增湍流模型时，应只追加状态尾部变量及其 `ExtraEquationFlux/Source/BC/Output` 策略，不再改动基础四变量的索引、ACM 声学特征系统或 MPI/OpenMP 装配骨架。
