---
title: "DNDSR 不可压缩计算模块：计算流程、离散方法与数学推导"
author: "基于 DNDSR 当前源码整理"
date: "2026-09-04"
lang: zh-CN
toc: true
toc-depth: 3
numbersections: true
geometry: margin=2.2cm
fontsize: 10.5pt
documentclass: ctexart
classoption:
  - UTF8
  - a4paper
header-includes:
  - |
    \usepackage{amsmath,amssymb,booktabs,longtable,array,mathtools}
  - |
    \usepackage{xcolor}
  - |
    \definecolor{codegray}{RGB}{245,247,249}
  - |
    \setlength{\LTpre}{0.5em}\setlength{\LTpost}{0.5em}
---

# 文档定位与范围

本文说明 DNDSR 中与不可压缩计算直接相关的完整计算链。这里的“不可压缩”特指基于人工可压缩法（Artificial Compressibility Method, ACM）的两套独立模块：

- `src/ACM`：常密度二维/三维 ACM，状态为 $U=(u,v,w,p)^T$；
- `src/ACMVariable`：变密度二维/三维 ACM，状态为 $U=(\rho,m_x,m_y,m_z,p)^T$，其中 $\boldsymbol{m}=\rho\boldsymbol{u}$。

两者共同使用：

- `src/Geom`：CGNS 网格读取、周期匹配、分区、幽灵层、面构造、重排和 VTK-HDF 输出；
- `src/CFV`：有限体积几何、高阶变分重构（VR）、Green--Gauss 梯度、光滑度/间断探测和 WBAP/CWBAP 限制器；
- `src/Solver`：通用 GMRES 及共享 ODE 基础设施；
- `src/DNDS`：MPI 分布式数组、持久通信、配置注册、序列化和基础类型。

本文以 2026-09-04 工作区源码为准。它是一份“实现说明 + 数学推导”，不是泛化的不可压缩 CFD 教材。凡是代码采用近似 Jacobian、冻结重构或分离耦合，均明确写出，不把它描述成完整 Newton 法。

> **重要约定**：代码里的残差使用“更新方向”号，即 $dU/dt=R(U)$。若读者习惯写成 $M\,dU/dt+\mathcal R(U)=0$，则 $R=-\mathcal R$。本文在每个公式组内保持同一符号，避免把通量散度的正负号与时间更新号混淆。

# 总体架构与一次计算的完整链路

## 模块责任表

| 层次 | 主要文件 | 责任 |
|---|---|---|
| 应用入口 | `app/ACM/*.cpp`, `app/ACMVariable/*.cpp` | 初始化 MPI，选择 2D/3D 模型并进入单块应用 |
| 配置/装配 | `SingleBlockApp.hpp`, `ACMConfig.hpp`, `ACMSolver.hpp/.hxx` | 读取单一 JSON、命令行覆盖、网格和场初始化、外层步循环、输出 |
| 物理核 | `ACM.cpp`, `ACMVariable/ACM.cpp`, `ACMFlux.hpp` | 状态变换、物理通量、$A_n$、$\Gamma$、特征系统、Roe/Rusanov、黏性通量、边界状态 |
| 空间离散 | `ACMEvaluator.hpp/.hxx` | 重构、探测、限制、面积分、残差、CFL、面 Jacobian、矩阵乘和 LU--SGS |
| 高阶公共核 | `CFV/VariationalReconstruction*`, `CFV/Limiters.hpp` | 基函数/度量、VR 系数、Green--Gauss、光滑指示器、WBAP/CWBAP |
| 时间推进 | `ACMTime.hpp/.cpp`, `ACMBDF2.hpp/.cpp`, `ACMVariable/ACMPhysicalTime.hxx` | SSPRK3、隐式 BE、BDF1/BDF2 双时间、共享 ODE/DAE 适配 |
| 湍流 | `ACMTurbulence*`, `ACMTurbulenceTransport*` | Laminar/SA/$k$--$\omega$/SST/realizable $k$--$\epsilon$，分离式输运与涡黏性 |
| 网格/并行 | `Geom/Mesh/Mesh_Helpers.hpp`, `ACMParallel.hpp` | CGNS、METIS/ParMETIS 路径、ghost、周期旋转、MPI 归约与一致性检查 |

## 调用顺序

一次标准运行遵循下面的顺序：

1. `RunSingleBlockConsoleApp` 初始化配置注册，读取 JSON，并应用 `-k/-v` JSON Pointer 覆盖。
2. `ACMSolver::ReadMeshAndInitialize` 创建分布式网格和 CGNS reader，写入三组周期平移向量，建立边界名字到边界 ID 的映射。
3. `Geom::ReadMeshFromCGNS` 串行读取 CGNS，周期节点去重，建立 cell-to-cell 图，分区并迁移到各 MPI rank，构造第一层主 ghost；可选 O1→O2 几何升阶及 0--4 次直接二分。
4. `Geom::PrepareMesh` 可选重排单元，插值/生成面，检查面拓扑，建立 node-to-cell/boundary 的第二类 ghost 映射。
5. 重建周期节点；若启用输出则建立 VTK connectivity。
6. 构造 `CFV::VariationalReconstruction`，解析 `vfvSettings`，预计算体积、面积、面法向、积分点、局部基、重构矩阵与通信计划。
7. 分配单元均值 $U$、残差、线性右端和线性增量；用 `initialState` 初始化。
8. 构造 `ACMEvaluator`；若启用 RANS，再构造 `ACMTurbulenceTransport`，向流场 evaluator 注入“准备涡黏性”和“查询面积分点涡黏性”回调。
9. 外层时间步中，每次先进行 halo 拉取和空间重构，再执行间断探测/限制、面高斯积分、残差装配和全局残差归约。
10. 根据积分器选择 SSPRK3、隐式 BE、LU--SGS/GMRES 或物理时间 BDF/共享 ODE；必要时推进分离式湍流方程。
11. 按间隔输出并行 VTK-HDF。物理时间模式写入真实时间，稳态伪时间模式写入步号。

# 控制方程与人工可压缩思想

## 从不可压约束到伪时间方程

常密度不可压 Navier--Stokes 方程为

$$
\nabla\cdot\boldsymbol{u}=0,
\qquad
\frac{\partial\boldsymbol{u}}{\partial t}
+\nabla\cdot(\boldsymbol{u}\otimes\boldsymbol{u})
+\frac{1}{\rho_0}\nabla p
-\frac{1}{\rho_0}\nabla\cdot\boldsymbol{\tau}=0.
$$

压力没有独立演化方程，而是用于施加散度约束。ACM 引入伪时间 $\tau$ 与人工声速尺度 $\beta$，把稳态问题扩展成双曲型伪时间系统：

$$
\Gamma(U)\frac{\partial U}{\partial\tau}
+\nabla\cdot F(U)-\nabla\cdot F^v(U,\nabla U)=0.
$$

当伪时间收敛时 $\partial U/\partial\tau\to0$，最后一行仍恢复 $\nabla\cdot\boldsymbol{u}=0$。因此 $\beta^2$ 只控制伪时间特征速度和收敛性，不是流体真实声速；$\Gamma$ 是伪时间预处理/质量矩阵，不是物理时间质量矩阵。

对非定常不可压问题，代码采用

$$
\Gamma(U)\frac{\partial U}{\partial\tau}
+M\frac{\partial U}{\partial t}
=R(U,t),
$$

其中 $M$ 在压力分量为零。每个物理时刻通过伪时间内迭代使代数压力约束收敛，这就是双时间推进。

## 常密度状态、通量和雅可比

代码只支持物理压力存储：

$$
U=(u,v,w,p)^T,qquad \rho_0>0,quad b\equiv\beta^2>0,quad \gamma_T=1+\alpha.
$$

在单位面法向 $\boldsymbol{n}$ 上构造右手正交基 $T=[\boldsymbol{n},\boldsymbol{t}_1,\boldsymbol{t}_2]$。令局部速度为 $(q,s_1,s_2)$，其中 $q=\boldsymbol{u}\cdot\boldsymbol{n}$，则法向物理通量为

$$
F_n(U)=
\begin{bmatrix}
q^2+p/\rho_0\\
qs_1\\
qs_2\\
q
\end{bmatrix}.
$$

前三行对应动量方程除以常密度后的速度形式，最后一行是散度通量。直接对同一状态 $U=(q,s_1,s_2,p)^T$ 求导：

$$
A_n=\frac{\partial F_n}{\partial U}=
\begin{bmatrix}
2q&0&0&1/\rho_0\\
s_1&q&0&0\\
s_2&0&q&0\\
1&0&0&0
\end{bmatrix}.
$$

左上角必须是 $2q$：$\partial(q^2)/\partial q=2q$。把非守恒对流形式的系数矩阵直接当作通量 Jacobian 会漏掉一个 $q$。

Scheme-A 伪时间矩阵及其逆为

$$
\Gamma=
\begin{bmatrix}
1&0&0&\gamma_Tq/b\\
0&1&0&\gamma_Ts_1/b\\
0&0&1&\gamma_Ts_2/b\\
0&0&0&1/b
\end{bmatrix},
\quad
\Gamma^{-1}=
\begin{bmatrix}
1&0&0&-\gamma_Tq\\
0&1&0&-\gamma_Ts_1\\
0&0&1&-\gamma_Ts_2\\
0&0&0&b
\end{bmatrix}.
$$

故预处理法向 Jacobian 为

$$
B_n\equiv\Gamma^{-1}A_n=
\begin{bmatrix}
(1-\alpha)q&0&0&1/\rho_0\\
-\alpha s_1&q&0&0\\
-\alpha s_2&0&q&0\\
b&0&0&0
\end{bmatrix}.
$$

## 常密度特征系统

特征多项式为

$$
\det(\lambda I-B_n)=(\lambda-q)^2
\left[\lambda^2-(1-\alpha)q\lambda-\frac{b}{\rho_0}\right].
$$

因此两支切向波和两支人工声学波为

$$
\lambda_{t,1}=\lambda_{t,2}=q,
\qquad
\lambda_\pm=\frac{(1-\alpha)q\pm
\sqrt{(1-\alpha)^2q^2+4b/\rho_0}}{2}.
$$

代码用 `hypot` 计算根号，降低极大参数下的溢出风险。令声学右特征向量的压力分量为 1，可取

$$
r_\lambda=
\begin{bmatrix}
\lambda/b\\
\alpha s_1\lambda/[b(q-\lambda)]\\
\alpha s_2\lambda/[b(q-\lambda)]\\
1
\end{bmatrix},\qquad \lambda\in\{\lambda_-,\lambda_+\}.
$$

两支纯切向向量可取 $e_2,e_3$。代码组装 $R$ 后数值求 $L=R^{-1}$，从而避免手写一般 $\alpha$ 左特征向量时的符号错误。

## 变密度守恒状态、通量和雅可比

变密度模块采用

$$
U=(\rho,m_n,m_1,m_2,p)^T,qquad
q=m_n/\rho,\quad a=m_1/\rho,\quad c=m_2/\rho.
$$

其局部物理通量是

$$
F_n(U)=
\begin{bmatrix}
m_n\\
m_n^2/\rho+p\\
m_nm_1/\rho\\
m_nm_2/\rho\\
m_n/\rho
\end{bmatrix}.
$$

第五项必须是 $u_n=m_n/\rho$；它提供不可压散度方程。直接对守恒变量求导得到

$$
A_n=
\begin{bmatrix}
0&1&0&0&0\\
-q^2&2q&0&0&1\\
-qa&a&q&0&0\\
-qc&c&0&q&0\\
-q/\rho&1/\rho&0&0&0
\end{bmatrix}.
$$

Scheme-A 守恒变量伪时间矩阵采用 $\gamma_1=\alpha+1$、$\gamma_2=2\alpha+1$：

$$
\Gamma=
\begin{bmatrix}
1&0&0&0&\gamma_1\rho/b\\
0&1&0&0&\gamma_2m_n/b\\
0&0&1&0&\gamma_2m_1/b\\
0&0&0&1&\gamma_2m_2/b\\
0&0&0&0&1/b
\end{bmatrix},
$$

$$
\Gamma^{-1}=
\begin{bmatrix}
1&0&0&0&-\gamma_1\rho\\
0&1&0&0&-\gamma_2m_n\\
0&0&1&0&-\gamma_2m_1\\
0&0&0&1&-\gamma_2m_2\\
0&0&0&0&b
\end{bmatrix}.
$$

用原始变量 $W=(\rho,q,a,c,p)^T$ 和 $J=\partial U/\partial W$ 作相似变换：

$$
B_W=J^{-1}\Gamma^{-1}A_nJ=
\begin{bmatrix}
q&-\alpha\rho&0&0&0\\
0&(1-\alpha)q&0&0&1/\rho\\
0&-\alpha a&q&0&0\\
0&-\alpha c&0&q&0\\
0&b&0&0&0
\end{bmatrix}.
$$

于是

$$
\det(\lambda I-B)=(\lambda-q)^3
\left[\lambda^2-(1-\alpha)q\lambda-b/\rho\right],
$$

$$
\lambda_\pm=\frac{(1-\alpha)q\pm
\sqrt{(1-\alpha)^2q^2+4b/\rho}}{2},
\qquad \lambda_c=q\quad(\text{三重}).
$$

三重 $q$ 模态分别是密度接触模态和两支切向剪切模态。密度模态在原始变量中是 $(1,0,0,0,0)^T$，映射回守恒变量后为 $(1,q,a,c,0)^T$，不能误写成只改变密度而保持动量不变。

平方根密度 Roe 平均为

$$
\widetilde\rho=\sqrt{\rho_L\rho_R},\qquad
\widetilde{\boldsymbol{u}}=
\frac{\sqrt{\rho_L}\boldsymbol{u}_L+\sqrt{\rho_R}\boldsymbol{u}_R}
{\sqrt{\rho_L}+\sqrt{\rho_R}},\qquad
\widetilde p=\frac{p_L+p_R}{2}.
$$

它满足包括第五个散度通量在内的 Roe 性质
$A(\widetilde U)(U_R-U_L)=F(U_R)-F(U_L)$。

# 数值通量与特征碰撞处理

## Roe 与 Rusanov

在面局部坐标中，Roe 型数值通量统一写成

$$
\widehat F_n=\frac{F_n(U_L)+F_n(U_R)}{2}
-\frac12\Gamma(\widetilde U)\,f(B_n(\widetilde U))\,(U_R-U_L),
$$

其中 $f(\lambda)=|\lambda|_\delta$ 是带 Harten 熵修正的绝对值：

$$
|\lambda|_\delta=
\begin{cases}
|\lambda|,&|\lambda|\ge\delta,\\
(\lambda^2+\delta^2)/(2\delta),&|\lambda|<\delta.
\end{cases}
$$

在特征基良态时 $f(B)=R|\Lambda|_\delta L$。常密度以算术平均为 Roe 状态；变密度采用平方根密度平均。

Rusanov 通量为

$$
\widehat F_n^{Rus}=\frac{F_L+F_R}{2}
-\frac12s_{max}\Gamma(\widetilde U)(U_R-U_L).
$$

常密度的 $s_{max}$ 由平均态谱半径给出；变密度实现同时检查左态、Roe 态和右态的谱半径，避免低密度一侧的人工声速 $\sqrt{b/\rho}$ 被平均态低估。

## 特征碰撞与合流 Hermite 矩阵函数

当

$$
\lambda_\pm=q
\quad\Longleftrightarrow\quad
\alpha\rho q^2=b
$$

时，声学根与重复切向/接触根碰撞。若存在非零切向速度，$B-qI$ 可能形成 Jordan 块，直接反演 $R$ 会病态或失败。当前实现不是简单把碰撞簇乘上 $|q|$，而是对最小多项式使用合流 Hermite 插值计算 $f(B)$。重复节点保留 $f'(q)$，需要时还使用二阶合流差商，因此与 Jordan 极限一致。

这一处理同时服务于：

- Roe 耗散矩阵 $\Gamma f(B)\Delta U$；
- 特征边界在碰撞附近的入射/出射谱投影；
- WBAP/CWBAP 的特征变换判定。若完整特征基不可用，限制器回退到物理/守恒分量空间，而不是反演奇异矩阵。

# 非结构有限体积离散

对单元 $\Omega_i$，代码采用单元均值

$$
\bar U_i=\frac1{V_i}\int_{\Omega_i}U\,dV
$$

和面高斯积分。按代码的更新号约定，半离散残差为

$$
R_i(U)=-\frac1{V_i}\sum_{f\subset\partial\Omega_i}
\int_f\left(\widehat F_f-\widehat F_f^v\right)dS.
$$

内部面只计算一次。若面法向从左单元指向右单元，则同一个积分通量对左单元加 $-H_f/V_L$，对右单元加 $+H_f/V_R$。因此

$$
V_LR_L^{(f)}+V_RR_R^{(f)}=0,
$$

内部面严格守恒。跨 MPI 分区的面仍遵循这一逻辑：面由拥有者计算，所需邻侧状态和重构系数通过 ghost/持久通信获得。

在每个面积分点，实际次序为：

1. 从左右单元均值与重构系数得到 $U_L^g,U_R^g$；
2. 对周期面把右状态、梯度和位移旋转/平移到左面坐标框架；
3. 对物理边界生成 ghost 状态；
4. 检查变密度候选状态并按单元比例回退，确保 $\rho>\rho_{floor}$；
5. 计算 Roe/Rusanov 无黏通量；
6. 若启用黏性，构造修正面梯度并计算层流黏性 + 面涡黏性通量；
7. 乘面积分权重和面 Jacobian，累加到面缓冲；
8. 面缓冲散射到相邻单元残差。

# 网格读取、分区与几何准备

## CGNS 读取和分区

ACM 求解器当前从 `meshSettings.meshFile` 读取 CGNS。`Geom::ReadMeshFromCGNS` 的计算步骤不是单纯的文件解析，而是一条固定的分布式建网流水线：

1. rank 0 通过 `UnstructuredMeshSerialRW::ReadFromCGNSSerial` 读取节点、单元、边界区和 1-to-1 周期信息；边界区名字经 `BoundaryHandler::GetIDFromName` 映射为数值 ID。
2. `Deduplicate1to1Periodic(tol)` 在给定几何容差内识别周期等价节点，避免周期接口重复拓扑。
3. `BuildCell2Cell()` 从 cell-to-node 关系构建分区图；图的顶点为体单元，公共面对应图边。
4. `MeshPartitionCell2Cell(partitionOptions)` 调用项目的 METIS/ParMETIS 分区路径，使各 rank 单元数和图割尽量均衡。
5. `PartitionReorderToMeshCell2Cell()` 把串行数组迁移为每 rank 的 owned 数组并建立全局/局部编号映射。
6. `BuildGhostPrimary()` 找出分区边界所需的远程单元/节点，创建主 ghost 与通信 transformer。

若 `meshElevation=1`，先把 O1 几何提升成 O2，然后重新构造 ghost。若 `meshDirectBisect=k`，每轮先临时升为 O2，恢复 node-to-cell 和 cell-to-cell 关系，再由 O2 几何构造二分后的 O1 网格；每轮都会重新分布 ghost。这个过程是几何/拓扑细化，不等于解的自适应误差控制。

## 求解器就绪的网格

`Geom::PrepareMesh` 继续完成：

- 可选 `ReorderLocalCells(reorderParts)`，改善局部性并为 OpenMP 分块；
- `InterpolateFace()` 从单元拓扑生成/插值面；
- `AssertOnFaces()` 检查面左右单元、节点和方向一致性；
- 把 N2CB 邻接临时转成全局编号，建立 node-to-cell/boundary ghost，再转回局部编号；
- 检查 owned 节点依赖的单元邻居均已解析为局部非负索引。

随后 ACM 求解器调用 `RecreatePeriodicNodes()`。若启用输出，额外调用 `BuildVTKConnectivity()`。`PrepareMesh` 本身不做壁面距离；湍流模块需要时再基于真实边界几何构造壁面距离。

## 周期边界

JSON 提供三组平移向量 `periodicTranslation1/2/3` 和匹配容差。网格层存储周期节点对应关系；面积分时还必须把向量量旋转到同一物理框架：

- 常密度：旋转速度三分量，压力不变；
- 变密度：密度、压力不变，旋转动量三分量；
- 梯度同时变换空间方向和变量向量方向；
- 面中心位移包含周期平移，否则非正交黏性修正会使用错误的中心连线。

# 重构方法

## 一阶重构

`FirstOrder` 令所有非均值自由度为零：

$$
U_i(x)=\bar U_i.
$$

它最耗散，但重构无历史依赖，适合初始化、调试和隐式 Jacobian 的冻结一阶近似。

## Green--Gauss 二阶梯度

由高斯定理

$$
\int_{\Omega_i}\nabla U\,dV
=\int_{\partial\Omega_i}U\boldsymbol{n}\,dS
$$

得到单元梯度近似

$$
(\nabla U)_i\approx\frac1{V_i}
\sum_{f\subset\partial\Omega_i}U_f\boldsymbol{n}_fS_f.
$$

面值由相邻均值/边界状态插值。面积分点状态为

$$
U_i(x_g)=\bar U_i+(\nabla U)_i\cdot(x_g-x_i).
$$

ACM 调用 CFV 的 `DoReconstruction2ndGrad`。其优点是便宜；在强非正交网格上精度取决于面值插值和几何质量，因此黏性通量还需独立的中心线一致性修正。

## CFV 变分重构

每个单元使用零均值多项式基

$$
U_i(x)=\bar U_i+\sum_{\ell=1}^{N_p}c_i^\ell\phi_i^\ell(x),
\qquad
\int_{\Omega_i}\phi_i^\ell dV=0.
$$

零均值保证不论如何修改高阶系数，单元守恒均值保持不变。对共享面 $f=(i,j)$，CFV 定义值和若干阶导数跳跃的加权泛函

$$
I_f(c_i,c_j)=
\sum_{|\boldsymbol{k}|\le r}w_{f,\boldsymbol{k}}^2
\int_f
\left\|D^{\boldsymbol{k}}U_i-D^{\boldsymbol{k}}U_j\right\|_2^2dS.
$$

对单元周围所有面求和并令对 $c_i$ 的导数为零，得到局部线性耦合系统

$$
A_ic_i=\sum_{j\in N(i)}
\left(B_{ij}c_j+b_{ij}(\bar U_j-\bar U_i)\right).
$$

$A_i,B_{ij},b_{ij}$ 只依赖网格、基函数、积分阶数和权重，初始化阶段即可预计算。每次残差求值时通过固定次数 Jacobi 或 SOR 迭代更新：

$$
c_i^{(k+1)}=(1-\omega)c_i^{(k)}+omega A_i^{-1}
\sum_j\left(B_{ij}c_j^{(k)}+b_{ij}\Delta\bar U_{ij}\right).
$$

`variationalIterations` 控制迭代次数，`vfvSettings.jacobiRelax` 控制 $\omega$，`SORInstead` 选择更新风格。`resetVariationalCoefficients=false` 时以上一步系数热启动，有利于稳态收敛；但此时有限次数 VR 是带历史的近似空间算子。若要严格评价物理时间精度，应提高迭代收敛度或验证热启动误差远小于时间离散误差。

`vfvSettings.maxOrder` 决定最高多项式阶数；`intOrderVR`、`intOrderVRBC` 决定内部面和边界面泛函积分阶数；各向异性长度、方向权重和几何权重用于改善拉伸网格上的条件数。

# 间断探测器与限制器

## 光滑度/间断指示器

WBAP/CWBAP 之前调用 `DoCalculateSmoothIndicatorV1`。对单元 $i$ 的每个面高斯点，先构造左右重构值 $U_L,U_R$，定义平均和半跳跃

$$
U^{avg}=\frac{U_L+U_R}{2},
\qquad
U^{jmp}=\frac{U_L-U_R}{2}.
$$

通过与 VR 相同的面泛函度量分别累计跳跃能量 $I_{J,m}$ 和解能量 $I_{S,m}$：

$$
I_{J,m}=\sum_{f\in\partial\Omega_i}S_f
\mathcal F_f(U_m^{jmp},U_m^{jmp}),
$$

$$
I_{S,m}=\sum_{f\in\partial\Omega_i}S_f
\mathcal F_f(U_m^{avg},U_m^{avg}).
$$

分量指标为 $I_{J,m}/(I_{S,m}+\epsilon)$，单元指标取指定分量的最大值并作缩放：

$$
SI_i=p_{max}^2
\sqrt{\max_m\left|w_m\frac{I_{J,m}}{I_{S,m}+\epsilon}\right|}.
$$

当 $SI_i<\texttt{smoothThreshold}$ 时，CWBAP/WBAP 可以跳过该单元的限制。`FPost` 在计算指标前将 ACM 状态无量纲化/后处理；其目的在于避免速度、压力、密度量纲差异让某一分量无条件支配探测器。不可压压力存在规范自由度，稳健指标应围绕 `pressureReference` 或局部压力差构造；使用绝对压力时必须验证 $p\mapsto p+C$ 不改变限制决策。

## LocalExtrema（Barth--Jespersen 类）

对每个单元和变量，从相邻单元均值与边界 ghost 值形成 $U_{i,m}^{min},U_{i,m}^{max}$。令积分点未限制增量

$$
\delta U_{i,f,g,m}=U_{i,f,g,m}^{rec}-\bar U_{i,m},
$$

则

$$
\theta_{i,f,g,m}=
\begin{cases}
\min\left(1,\dfrac{U_{i,m}^{max}-\bar U_{i,m}}{\delta U_{i,f,g,m}}\right),&\delta U>0,\\[0.8em]
\min\left(1,\dfrac{U_{i,m}^{min}-\bar U_{i,m}}{\delta U_{i,f,g,m}}\right),&\delta U<0,\\
1,&|\delta U|\le\epsilon.
\end{cases}
$$

流场 evaluator 取

$$
\theta_i=\min_{f,g,m}\theta_{i,f,g,m}\in[0,1]
$$

并统一缩放单元高阶增量。优点是不会在已知邻域均值范围外创造新极值；缺点是任一变量触发都会使所有变量同步降阶，可能过度耗散。

变密度模块另有密度正性回退。若任一候选面积分点满足 $\rho^{rec}<\rho_{floor}$，对高阶增量求最大允许比例

$$
\theta_\rho\le
\frac{\bar\rho-\rho_{floor}}
{\bar\rho-\rho^{rec}},
$$

再用 $\min(\theta_i,\theta_\rho)$ 缩放，从而不改变单元均值而保证面状态密度正。

## WBAP 与 CWBAP

WBAP 对中心和邻接方向的同阶多项式系数进行非线性加权。以标量系数为例，设中心值 $c_0$ 和已映射到中心基的邻接候选 $c_j$，可写成

$$
\vartheta_j=\frac{c_0}{|c_j|+\epsilon^{1/4}}\operatorname{sgn}(c_j),
\qquad p=4,
$$

$$
c^{lim}=c_0\,\chi\,
\frac{n+\sum_{j>0}\vartheta_j^{p-1}}
{n+\sum_{j>0}\vartheta_j^p}.
$$

$\chi$ 在候选异号或退化时抑制输出，`WBAP_nStd` 对应参数 $n$。`normWBAP=true` 时不是逐系数独立衡量，而是用同一多项式阶次块的 $L_2$/多项式范数构造权重；二维与三维通过模板维数选择不同的系数块和范数。

CWBAP 按多项式阶次从高到低处理，并在光滑单元保留原系数；普通 WBAP 更直接地对每阶中心/邻域候选做多向组合。两者都先用单元间的 secondary matrix 把邻居多项式映射到当前单元基，不能直接比较不同几何单元中的原始系数。

## 特征空间限制

对每个面平均态构造 $L,R$，将一组多项式系数从物理空间变为特征空间：

$$
C^{char}=LC^{phys},
\qquad
C_{lim}^{phys}=R\,\mathcal L(C^{char}).
$$

这样限制器分别作用于人工声学、切向剪切和（变密度时）密度接触模态，减少变量耦合造成的伪振荡。若特征值碰撞或 SVD 检测到 $R$ 病态，代码回退到物理/守恒分量限制；这是有意的稳健路径。

# 黏性通量与非正交修正

## 常密度 Newton 流体

速度梯度 $G=\nabla\boldsymbol{u}$ 给出

$$
\boldsymbol{\tau}=\mu_{eff}
\left(G+G^T-\frac23(\nabla\cdot\boldsymbol{u})I\right),
\qquad
\mu_{eff}=\mu+\mu_t.
$$

常密度速度方程中的面黏性通量为 $\boldsymbol{\tau}\boldsymbol{n}/\rho_0$，压力/散度方程黏性通量为零。

## 变密度梯度链式法则

重构变量是 $(\rho,\boldsymbol{m},p)$，但应力需要 $\nabla\boldsymbol{u}$。由 $\boldsymbol{u}=\boldsymbol{m}/\rho$：

$$
\nabla\boldsymbol{u}=\frac{\nabla\boldsymbol{m}-(\nabla\rho)\otimes\boldsymbol{u}}{\rho}.
$$

代码先对守恒变量做面梯度修正，再通过 `PrimitiveGradient` 执行这一步链式变换。变密度动量方程的黏性通量为 $\boldsymbol{\tau}\boldsymbol{n}$，质量和散度分量为零。

## 线性精确面梯度

令左右单元重构梯度平均为 $\bar G=(G_L+G_R)/2$，中心位移 $\boldsymbol{d}=\boldsymbol{x}_R-\boldsymbol{x}_L$，单位法向为 $\boldsymbol{n}$。当前代码采用 over-relaxed 非正交修正

$$
G_f=\bar G+\boldsymbol{n}\otimes
\frac{(U_R-U_L)-\boldsymbol{d}^T\bar G}{\boldsymbol{d}\cdot\boldsymbol{n}}.
$$

验证其线性精确性：若 $U(x)=U_0+G_*^Tx$，则 $U_R-U_L=\boldsymbol{d}^TG_*$，且理想重构 $\bar G=G_*$，修正项为零，所以 $G_f=G_*$。即便 $\bar G$ 的法向投影有误，修正也强制

$$
\boldsymbol{d}^TG_f=U_R-U_L.
$$

这比用 $2V_L/S_f$ 近似中心距更适合非均匀、非正交内部面。周期面必须先把 $U_R,G_R,\boldsymbol{d}$ 映射到左框架再使用该式。

# 边界条件

边界接口沿用 Euler 风格的名字，但 ACM 不含总能量、温度或理想气体状态方程。名字的实际数学语义如下。

| JSON/枚举 | ACM 语义 | ghost 构造要点 |
|---|---|---|
| `BCFar` / `FarField` | 特征远场 | 以边界内态/给定远场态构造一般 $\alpha$ 特征投影；只从远场注入入射模态 |
| `BCWall` / `NoSlipWall` | 无滑移壁 | 镜像全部速度，使面值达到给定壁速；压力通常外推 |
| `BCWallIsothermal` | ACM 中仍按壁面速度处理 | 没有温度方程，“Isothermal”只是兼容名字 |
| `BCWallInvis` / `SlipWall` | 滑移壁 | 反射法向速度，切向速度保持 |
| `BCSym` / `Symmetry` | 对称面 | 与滑移壁相同的法向反射语义 |
| `BCOut` | 外推出口 | ghost 取内部状态，零法向梯度近似 |
| `BCOutP` / `PressureOutlet` | 指定压力出口 | 速度/密度外推，压力关于指定面值镜像 |
| `BCIn` | 给定状态入口 | 常密度给定速度/压力；变密度给定 $\rho,\rho\boldsymbol{u},p$ |
| `BCInPsTs` / `VelocityInlet` | 密度+速度入口 | 无 EOS，`PsTs` 不表示真实静压/静温换算 |
| `BCSpecial` | 验证专用 | 常状态或代码定义的制造/周期验证型边界 |

## 滑移壁推导

把内部速度分解为

$$
\boldsymbol{u}_I=u_n\boldsymbol{n}+\boldsymbol{u}_t.
$$

静止滑移壁 ghost 取

$$
\boldsymbol{u}_G=-u_n\boldsymbol{n}+\boldsymbol{u}_t
=\boldsymbol{u}_I-2(\boldsymbol{u}_I\cdot\boldsymbol{n})\boldsymbol{n}.
$$

中心面值为 $(\boldsymbol{u}_I+\boldsymbol{u}_G)/2=\boldsymbol{u}_t$，故法向速度为零。变密度状态保持 $\rho_G=\rho_I$，再设 $\boldsymbol{m}_G=\rho_G\boldsymbol{u}_G$。

## 无滑移壁推导

若给定壁速 $\boldsymbol{u}_w$，ghost 取

$$
\boldsymbol{u}_G=2\boldsymbol{u}_w-\boldsymbol{u}_I,
$$

使中心插值得到 $\boldsymbol{u}_w$。湍流壁面值由湍流模块另行处理：SA 取 $\widetilde\nu=0$；$k$ 在壁面趋零；$\omega$ 使用与 $\mu/(\rho d^2)$ 同量级的壁面条件。

## 压力出口

要使面中心压力等于 $p_b$，ghost 压力取

$$
p_G=2p_b-p_I.
$$

速度和常密度外推。不可压压力只确定到常数；封闭域需要通过 `pressureReference`、一个压力锚点或每步去除压力零空间来固定规范，否则线性系统含常数压力零模态。

## 特征远场

令 $B_n=R\Lambda L$，内态与远场态之差为 $\Delta U=U_\infty-U_I$。只注入 $\lambda_k<0$（相对于外法向进入计算域）的模态：

$$
U_B=U_I+R\,\mathrm{diag}(\chi_{\lambda_k<0})L\Delta U.
$$

实际实现还处理零速附近、特征碰撞和病态基。碰撞时用矩阵多项式谱投影，而不是强行反演奇异 $R$。

# 伪时间与物理时间推进

## 局部 CFL 步长

对单元 $i$，无黏稳定尺度由各面最大特征速度累计：

$$
\Delta\tau_i^{conv}
\sim \mathrm{CFL}\,
\frac{V_i}{\sum_f S_f\rho(B_{n,f})}.
$$

实现还乘以 CFV 的 `GetCellSmoothScaleRatio`，用于不同阶次/网格尺度的稳定修正。启用黏性时加入类似

$$
\sigma_i^{vis}\sim
\sum_f S_f\frac{\nu_{eff}}{h_{i,f}}
$$

的扩散谱估计，最终以对流与黏性尺度共同限制步长。各 rank 对最小步长和非有限状态做 MPI 归约；`maximumPseudoTimeStep` 提供上限。

## 显式 SSPRK(3,3)

伪时间系统为

$$
\Gamma(U)\frac{dU}{d\tau}=R(U),
\qquad L(U)=\Gamma(U)^{-1}R(U).
$$

代码使用 Shu--Osher SSPRK3：

$$
U^{(1)}=U^n+\Delta\tau L(U^n),
$$

$$
U^{(2)}=\frac34U^n+\frac14
\left[U^{(1)}+\Delta\tau L(U^{(1)})\right],
$$

$$
U^{n+1}=\frac13U^n+\frac23
\left[U^{(2)}+\Delta\tau L(U^{(2)})\right].
$$

每个 stage 都重新进行 halo、重构、限制和残差计算。变密度每个 stage 后检查密度正性和全局有限性。若 VR 只做有限次数且热启动，严格的三阶结论应通过网格/时间收敛试验确认。

## 隐式后向 Euler 伪时间

常密度在一步内近似求解

$$
\frac{\Gamma(U^*)}{\Delta\tau}(U^*-U^n)-R(U^*)=0.
$$

线性化缺陷得到

$$
\left[\frac{\Gamma}{\Delta\tau}-\frac{\partial R}{\partial U}\right]\delta U
=D(U^*),
\qquad U^*\leftarrow U^*+\omega\delta U.
$$

常密度 $\Gamma$ 依赖速度，但当前稳态隐式算子把部分非线性冻结，属于近似 Newton/Picard 缺陷校正。

变密度 $\Gamma(U)$ 显式依赖 $\rho,\boldsymbol{m}$。对乘积必须使用乘积求导：

$$
\frac{\partial}{\partial U}
\left[\Gamma(U)(U-U^0)/\Delta\tau\right]
=\frac1{\Delta\tau}\left[
\Gamma(U)+
\mathrm{diag}\left(
\frac{(\alpha+1)\Delta p}{b},
\frac{(2\alpha+1)\Delta p}{b}I_3,0
\right)\right].
$$

`PseudoTimeProductJacobian` 实现这一项，避免仅用 $\Gamma/\Delta\tau$ 时漏掉状态依赖。

## BDF1/BDF2 双时间

常密度物理质量矩阵为

$$
M_c=\mathrm{diag}(1,1,1,0),
$$

变密度为

$$
M_v=\mathrm{diag}(1,1,1,1,0).
$$

压力没有物理时间导数。第一个物理步用 BDF1：

$$
\left.\frac{dQ}{dt}\right|^{n+1}
\approx\frac{M(U^{n+1}-U^n)}{\Delta t}.
$$

从第二步起用等步长 BDF2：

$$
\left.\frac{dQ}{dt}\right|^{n+1}
\approx\frac{M(3U^{n+1}-4U^n+U^{n-1})}{2\Delta t}.
$$

统一系数 $(a_0,a_1,a_2)$ 为第一步 $(1,-1,0)$，后续 $(3/2,-2,1/2)$。目标物理时刻的非线性缺陷为

$$
D(U^*)=R(U^*,t^{n+1})-
M\frac{a_0U^*+a_1U^n+a_2U^{n-1}}{\Delta t}.
$$

每次伪时间修正近似解

$$
\left[
J_\tau+\frac{a_0}{\Delta t}M-\frac{\partial R}{\partial U}
\right]\delta U=D(U^*),
$$

其中 $J_\tau$ 是常密度近似 $\Gamma/\Delta\tau$ 或变密度乘积 Jacobian。$U^n,U^{n-1}$ 在整个内迭代中冻结，只有当前物理步收敛/完成后才移动历史。

常密度 BDF2 当前限制为 Laminar；分离式常密度湍流输运没有完整的两层物理历史。变密度湍流输运包含守恒 $\rho\phi$ 的物理缺陷/历史接口，但仍应按具体积分器与算例验证耦合收敛。

## 变密度共享 ODE/DAE 入口

`ACMVariable/ACMPhysicalTime.hxx` 把物理状态和压力代数变量分离，允许若干共享 ODE 引擎：BDF、SDIRK/ESDIRK、梯形和 Hermite 类方案。核心原则是：

- 物理微分变量为 $(\rho,\boldsymbol{m})$ 以及启用时的守恒湍流变量 $\rho\phi$；
- 压力属于 DAE 代数变量，不进入物理时间差分历史；
- stiffly accurate 方案的末级可作为步末状态；
- 非 stiffly accurate 的显式终值若没有单独压力投影，会破坏散度约束，因此配置阶段明确拒绝 `physicalODECode` 2/101。

# 隐式离散、LU--SGS 与 GMRES

## 冻结一阶面 Jacobian

完整高阶残差 Jacobian 包括：重构系数对所有邻域均值的导数、限制器非光滑导数、Roe 平均、矩阵绝对值、边界 ghost 链式法则、黏性非正交修正和涡黏性耦合。当前实现在线性求解期间冻结高阶重构、限制器和 $\mu_t$，并对每个面的一阶数值通量用中心有限差分形成左右块

$$
J_f^L=\frac{\partial\widehat F_f}{\partial U_L},
\qquad
J_f^R=\frac{\partial\widehat F_f}{\partial U_R}.
$$

所以它是与当前一阶面离散一致的近似 Jacobian，不是高阶残差的完整解析 Jacobian。外层非线性迭代会重新构造重构和面块，从而以 defect-correction 方式收敛高阶解。

## 线性系统

对单元 $i$，矩阵可写成块形式

$$
D_i\delta U_i+\sum_{j\in N(i)}A_{ij}\delta U_j=b_i.
$$

$D_i$ 包含伪时间质量块、物理 BDF 质量块和所有邻面关于本单元状态的导数；$A_{ij}$ 来自邻侧面 Jacobian。常密度块为 $4\times4$，变密度块为 $5\times5$。

## 块 Jacobi

忽略非对角块：

$$
\delta U_i^{(k+1)}=D_i^{-1}
\left(b_i-\sum_{j\ne i}A_{ij}\delta U_j^{(k)}\right).
$$

它并行性最好但传播慢，可作为 GMRES 左预条件器。

## LU--SGS

按本 rank 的局部单元顺序把矩阵分为 $A=L+D+U$。一次对称 Gauss--Seidel 近似为

$$
(D+L)\delta U^{*}=b-U\delta U^{old},
$$

$$
(D+U)\delta U^{new}=b-L\delta U^{*}.
$$

实现使用真实 $4\times4$/$5\times5$ 对角块求解，不复用 Euler 的能量行。跨 rank 邻居无法进入同一严格顺序扫描，使用 halo 交换后的滞后增量，因此是分布式块 SGS 预条件/平滑器，而不是全局串行精确 SGS。

## 左预条件 GMRES

通用 `Solver/Linear.hpp` 求解

$$
P^{-1}A\delta U=P^{-1}b,
$$

其中 $P^{-1}$ 可选块 Jacobi 或若干次 LU--SGS。Arnoldi 正交生成 Krylov 子空间

$$
\mathcal K_m(P^{-1}A,P^{-1}b)
=\mathrm{span}\{r_0,(P^{-1}A)r_0,\ldots,(P^{-1}A)^{m-1}r_0\}.
$$

`gmresSubspace` 控制 restart 长度，`gmresRestarts` 控制重启次数，`gmresRelativeTolerance` 控制线性相对残差。每次矩阵向量积前都需交换输入向量 halo，否则跨分区面块使用旧数据。

# 湍流模块

## 耦合方式

湍流不是把额外变量直接拼进 ACM 的 $4/5$ 变量 Roe 系统，而是分离式求解：

1. 由当前流场重构速度梯度，计算单元/面积分点壁面距离和湍流变量梯度；
2. 计算 $\mu_t$ 并缓存每个面高斯点的涡黏性；
3. 流场黏性通量使用 $\mu+\mu_t$；
4. 独立组装湍流对流、扩散和源项残差；
5. 以若干 SSPRK3 子步推进湍流变量，并在每 stage 施加上下界。

`transportSubsteps` 是每个流场步的湍流子步数，`transportTimeScale` 给出相对于流场伪时间步的尺度。该算法是松耦合/分离耦合；强耦合效应需依靠外层迭代收敛。

## 通用输运形式

常密度模型按原始湍流变量 $\phi$ 可概括为

$$
\frac{\partial\phi}{\partial t}+\nabla\cdot(\boldsymbol{u}\phi)
=\nabla\cdot(D_\phi\nabla\phi)+S_\phi.
$$

变密度模块使用 Favre/守恒形式

$$
\frac{\partial(\rho\phi)}{\partial t}+\nabla\cdot(\rho\boldsymbol{u}\phi)
=\nabla\cdot(D_{\rho\phi}\nabla\phi)+S_{\rho\phi}.
$$

对流数值通量复用流场 Riemann 质量通量 $\widehat{\rho u_n}$，再按上风方向选择 $\phi$。这样湍流标量与离散质量输运一致，而不是另造一个与流场不一致的速度通量。

## 重构、限制与正性

湍流输运使用一阶或受限 Green--Gauss 二阶重构。`EvaluateLimiter` 同时考虑邻域极值、`minimumValue`/`maximumValue` 和流场共同缩放因子 `_flowLimiter`。每个 SSPRK stage 后再次 clamp。这个截断是数值保护，不是连续方程的一部分；频繁触发意味着时间步、边界值或模型分辨率需要检查。

## Spalart--Allmaras

输运变量为修正运动黏度 $\widetilde\nu$。典型闭合关系为

$$
\chi=\frac{\widetilde\nu}{\nu},\qquad
f_{v1}=\frac{\chi^3}{\chi^3+c_{v1}^3},\qquad
\mu_t=\rho\widetilde\nu f_{v1}.
$$

源项由产生、壁面破坏和梯度项组成：

$$
S_{SA}=c_{b1}\widetilde S\widetilde\nu
-c_{w1}f_w\left(\frac{\widetilde\nu}{d}\right)^2
+\frac{c_{b2}}{\sigma}|\nabla\widetilde\nu|^2.
$$

$d$ 是到无滑移壁的最近几何距离。变密度模块将方程乘以 $\rho$ 并以 $\rho\widetilde\nu$ 作物理历史；把扩散写入散度后还需要密度梯度修正

$$
S_{\rho,SA}^{corr}=-\frac{\nu+\widetilde\nu}{\sigma}
\nabla\rho\cdot\nabla\widetilde\nu,
$$

当前实现包含该项。壁面取 $\widetilde\nu=0$，入口/远场使用 JSON 正值，出口外推。

## Wilcox $k$--$\omega$

变量为 $k,\omega$，基本结构为

$$
\frac{Dk}{Dt}=P_k-\beta^*k\omega
+\nabla\cdot[(\nu+\sigma_k\nu_t)\nabla k],
$$

$$
\frac{D\omega}{Dt}=\alpha\frac{\omega}{k}P_k-\beta\omega^2
+\nabla\cdot[(\nu+\sigma_\omega\nu_t)\nabla\omega].
$$

$P_k$ 由应变率和 $\mu_t$ 计算并做非负/上限保护。涡黏性近似 $\mu_t=\rho k/\omega$，再受 `maximumEddyViscosityRatio` 限制。壁面 $k\to0$，$\omega_w=C\nu/d^2$，其中 $C$ 为 `wallOmegaCoefficient`。

## Menter SST

SST 用 $F_1$ 在近壁 $k$--$\omega$ 与远场 $k$--$\epsilon$ 等效系数间混合，并用 $F_2$ 限制涡黏性：

$$
\mu_t=\frac{\rho a_1k}{\max(a_1\omega,SF_2)}.
$$

$F_1,F_2$ 依赖 $k,\omega,d$、运动黏度以及 $\nabla k\cdot\nabla\omega$ 的交叉扩散尺度。源项含产生限制和交叉扩散；代码对分母、指数/双曲函数参数和极小壁距做保护。

## Realizable $k$--$\epsilon$

变量为 $k,\epsilon$。基本结构为

$$
\frac{Dk}{Dt}=P_k-\epsilon
+\nabla\cdot[(\nu+\nu_t/\sigma_k)\nabla k],
$$

$$
\frac{D\epsilon}{Dt}=C_{\epsilon1}\frac{\epsilon}{k}P_k
-C_{\epsilon2}\frac{\epsilon^2}{k}+S_{extra}
+\nabla\cdot[(\nu+\nu_t/\sigma_\epsilon)\nabla\epsilon].
$$

实现属于与项目既有 Euler 湍流形式兼容的工程 realizable 变体，含产生限制和额外耗散保护，不应未经逐项核对就称为某一论文版本的完全复现。

## 壁面距离

壁距由 Geom 对真实壁面几何求最近距离：`wallDistanceMethod=0` 为直接搜索，`1` 为 AABB tree；`wallDistanceExecution` 控制 MPI 执行方式，`wallDistanceSubdivide` 控制曲面/四边形细分。最终用 `minimumWallDistance` 截断，避免 $1/d^2$ 奇异。它不是沿网格线累计，因此适用于一般非结构网格。

# 二维与三维的实际变量布局

常密度 2D 和 3D 都使用四变量 $(u,v,w,p)$；变密度 2D 和 3D 都使用五变量 $(\rho,m_x,m_y,m_z,p)$。二维模型只是几何维数 `gDim=2`，仍保留跨平面 $w$ 或 $m_z$：

- 面法向和空间梯度只含 $x,y$；
- $w$ 作为被面内速度 $\boldsymbol{u}_{xy}$ 对流的标量，并可有 $x,y$ 黏性扩散；
- 如果初始/边界 $w=0$ 且无源，它保持零；
- 保留固定变量数便于共享 2D/3D 通量、限制器、输出和线性代数模板。

因此文档或后处理不能把 2D 常密度状态误标成 $(u,v,p)$，也不能把变密度 2D 误标成四变量。

# 并行实现

## 分布式数组和 halo

单元均值、重构系数、梯度、限制因子和线性增量均使用 DNDS 分布式数组。典型通信是

```text
field.trans.startPersistentPull()
field.trans.waitPersistentPull()
```

计算 owned 单元前，必须确保其读取的 ghost 已更新。关键同步点包括重构迭代、跨 rank 面通量、WBAP/CWBAP 邻域系数、隐式矩阵向量积和 LU--SGS 扫描。

## 面所有权与守恒

`NumFaceProc()` 包含本 rank 负责计算的面。跨 rank 面不能由两边独立用稍有差异的几何/状态重复计算，否则会破坏逐位守恒；evaluator 使用统一面所有权和 face buffer，再把贡献散射到 owned/ghost 关系。全局残差范数、最小时间步和合法性通过 MPI collective 归约。

## OpenMP

单元局部步骤（重构、限制、对角块、源项）通常可直接并行。面循环向两个单元写残差存在冲突，需要面着色、原子操作或先写 face buffer 后做 cell gather。当前代码大量采用 face buffer/cell gather 结构；具体循环是否启用 OpenMP 取决于编译宏和该段实现，性能评估应分别检查 MPI-only 与 MPI+OpenMP。

# 配置字段与选择逻辑

## 主要 JSON section

| section | 重要字段 | 作用 |
|---|---|---|
| `acmSettings` | `rho0`, `densityFloor`（仅变密度）, `beta2`, `alpha`, `dynamicViscosity`, `riemannSolverType`, `entropyFixRatio`, `pressureReference`, `enableViscousFlux` | 物理与通量核 |
| `meshSettings` | `meshFile`, `meshElevation`, `meshDirectBisect`, `meshReorderCells`, 周期平移/容差, `partitionOptions` | 网格读取与并行分区 |
| `reconstructionSettings` | `type`, `variationalIterations`, `resetVariationalCoefficients`, `enableLimiter`, `limiterType` | 重构和限制器选择 |
| `vfvSettings` | `maxOrder`, 积分阶数、VR 权重、`jacobiRelax`, `SORInstead`, `smoothThreshold`, `WBAP_nStd`, `normWBAP` | CFV 数学/几何细节 |
| `timeMarchSettings` | `integrator`, `nSteps`, `physicalTimeStep`, `pseudoTimeStep`, CFL、隐式容差、LU--SGS/GMRES 参数 | 时间推进与线性解 |
| `turbulenceSettings` | `model`, 初/远场值、上下界、$\mu_t/\mu$ 上限、子步、壁距参数 | RANS 闭合 |
| `boundaryConditions` | zone 名/ID、类型、值及特殊参数 | 每个 CGNS 边界区的物理语义 |
| `outputSettings` | `interval`, `directory`, `prefix`, `writeInitial` | VTK-HDF 输出 |

## 积分器选择

常规枚举包括：

- `ExplicitSSPRK3`：稳态伪时间显式；
- `ImplicitEulerBlockJacobi`：局部块 Jacobi 隐式；
- `ImplicitEulerLUSGS`：分布式 ACM 块 LU--SGS；
- `ImplicitEulerGMRES`：GMRES，预条件器可选块 Jacobi/LU--SGS；
- `BDF2DualTimeLUSGS`、`BDF2DualTimeGMRES`：首步 BDF1、后续 BDF2 的物理双时间。

变密度还可由 `pseudoODECode`/`physicalODECode` 进入共享 ODE/DAE 引擎。使用前应以运行时校验结果为准，不能只根据共享类名猜测算法；例如一个类名含 SSPRK4 的共享入口，实际配置映射可能执行 SSPRK3。

# 输出、重启与后处理

流场输出调用 `PrintParallelVTKHDFDataArray`：

- 常密度写标量 `Pressure` 和向量 `Velocity`；
- 变密度输出应同时核对 `Density`、速度/动量和压力字段的当前命名；
- 二维向量仍以三分量写出，便于 VTK 工具处理；
- 文件名按外层步编号补零，series time 在 BDF/物理 ODE 模式为真实物理时间。

当前求解器主链直接实现的是流场 VTK-HDF 输出。BDF 两层历史、湍流历史和完整 restart 是否序列化必须按具体分支再次检查；若只保存瞬时 $U$ 而不保存 $U^n,U^{n-1}$，重启后的第一个物理步只能重新以 BDF1 启动，不能无损接续 BDF2。

# 完整算法伪代码

## 稳态/伪时间

```text
read JSON and validate
read CGNS -> periodic dedup -> partition -> primary ghost
optional elevate/bisect
prepare faces and N2CB ghosts; recreate periodic nodes
build CFV metrics, bases, quadrature and reconstruction matrices
allocate U, residual and linear work arrays
initialize turbulence and wall distance if enabled

for outer step = 1 ... nSteps:
    choose local/global pseudo dt from fixed dt or CFL

    residual(U, time):
        pull U halo
        reconstruct: first-order / Green-Gauss / iterative VR
        if WBAP/CWBAP:
            compute smooth indicator
            transform coefficients to characteristic space when regular
            limit by order and transform back
        else if LocalExtrema:
            compute common cell theta
        enforce density-positive face candidates (variable density)
        prepare turbulence gradients and face eddy viscosity
        for each owned face and each quadrature point:
            reconstruct left/right or boundary ghost
            apply periodic transforms
            evaluate Roe/Rusanov inviscid flux
            evaluate corrected viscous flux when enabled
            integrate to face buffer
        gather face buffer to cells with opposite signs
        add turbulence source/transport residual separately
        reduce residual norms

    if SSPRK3:
        evaluate three stages with Gamma^{-1} residual
    else:
        repeat nonlinear corrections:
            assemble pseudo-time diagonal and finite-difference face blocks
            solve with block-Jacobi / LU-SGS / preconditioned GMRES
            relax update; check global RMS defect

    advance segregated turbulence substeps if configured
    write VTK-HDF at interval
```

## BDF2 双时间

```text
history.previous = initial U
history.previous_previous = initial U

for physical step n+1:
    coefficients = BDF1 if first step else BDF2
    target time = (n+1) * physicalTimeStep
    Ustar = U^n

    for pseudo iteration:
        R = spatialResidual(Ustar, target time)
        defect = R - M*(a0*Ustar + a1*U^n + a2*U^(n-1))/dt
        A = pseudoTimeJacobian + a0*M/dt - frozenSpatialJacobian
        solve A*dU = defect by ACM-LUSGS or GMRES
        Ustar += relaxation*dU
        stop when MPI-global physical defect is small

    commit history only after the physical step
    output with target physical time
```

# 数值验证建议

一套完整验收不应只看“算例能跑”，而应逐层隔离误差。

| 层次 | 建议测试 | 判据 |
|---|---|---|
| 状态/旋转 | 任意正交基往返、周期旋转 | 状态与通量往返至舍入误差 |
| 通量 Jacobian | 对 `PhysicalFluxLocal` 中心差分 | $A_n\delta U$ 与差分一致 |
| $\Gamma$ | $\Gamma^{-1}\Gamma$ | 接近单位阵 |
| 特征系统 | $BR-R\Lambda$, $LR-I$ | 一般点误差接近舍入量级 |
| 碰撞 | 从两侧逼近 $\alpha\rho q^2=b$ | Hermite 矩阵函数连续，含 Jordan 导数极限 |
| Roe 性质 | $A(\widetilde U)\Delta U-\Delta F$ | 常/变密度均接近零 |
| 黏性梯度 | 任意非正交网格上的线性场 | `CorrectedFaceGradient` 精确恢复常梯度 |
| 守恒 | 周期域内部面求和 | $\sum_iV_iR_i=0$（无源） |
| 重构阶数 | 制造光滑解网格加密 | 一阶/GG/VR 达到相应预期阶 |
| 探测器 | 光滑、跳跃、压力常数平移 | 光滑区低指标，跳跃区高；规范平移不改变决策 |
| 限制器 | 阶跃/涡量平流 | 无新极值，密度不低于 floor |
| 时间 | 光滑周期制造解减小 $\Delta t$ | SSPRK3/BDF2 达到时间阶，内迭代误差更小 |
| MPI | np=1/2/4/8 相同网格 | 守恒量、残差和解在容差内一致 |
| 湍流 | NASA TMR 类平板/翼型/后向台阶 | 壁面量、$\mu_t$ 和剖面与参考解一致 |

# 实现边界与使用注意事项

1. ACM 的人工特征波只服务于数值收敛，不是可压缩流的真实声学；不能由 $\beta$ 推导真实马赫数。
2. 常密度状态第 4 项、变密度状态第 5 项是压力，不是总能量；绝不能把 Euler restart 的前 4/5 项直接映射成 ACM。
3. 二维仍保存三分量速度/动量，是 2.5D 状态布局。
4. 高阶隐式算子冻结重构、限制器和涡黏性，是近似 Newton defect correction。
5. VR 固定迭代次数和热启动会给空间算子带来历史依赖；高精度非定常计算要做迭代敏感性分析。
6. 不可压压力具有常数零空间；封闭域必须明确压力规范处理。
7. RANS 是工程闭合，尤其 realizable $k$--$\epsilon$ 应按代码公式而非模型名字认定版本。
8. 若 BDF 历史未完整写入 restart，重启不能无损保持二阶时间历史。
9. `nSteps` 是固定外层上限；是否达到稳态或物理步内迭代收敛，应读取日志中的全局 RMS 缺陷和 `converged`，不能只凭程序正常退出。

# 源码索引

| 主题 | 源码位置 |
|---|---|
| 常密度状态/通量/$\Gamma$/特征/边界/黏性 | `src/ACM/ACM.cpp`, `ACMState.hpp`, `ACMFlux.hpp`, `ACMBC.hpp` |
| 变密度对应实现 | `src/ACMVariable/ACM.cpp`, `ACMState.hpp`, `ACMFlux.hpp`, `ACMBC.hpp` |
| 重构/限制/残差/CFL/隐式块 | 两模块的 `ACMEvaluator.hpp/.hxx` |
| 配置与校验 | 两模块的 `ACMConfig.hpp`, `ACMSettings.hpp`, `ACMTime.hpp` |
| 网格装配/输出/外层循环 | 两模块的 `ACMSolver.hpp/.hxx` |
| BDF2 | 两模块的 `ACMBDF2.hpp/.cpp` |
| 共享物理 ODE/DAE | `src/ACMVariable/ACMPhysicalTime.hxx` |
| 湍流闭合 | 两模块的 `ACMTurbulence.hpp/.cpp` |
| 湍流空间/时间输运 | 两模块的 `ACMTurbulenceTransport.hpp/.hxx` |
| VR 与间断探测 | `src/CFV/VariationalReconstruction*.hxx`, `VRSettings.hpp` |
| WBAP/CWBAP 数学核 | `src/CFV/Limiters.hpp`, `VariationalReconstruction_LimiterProcedure.hxx` |
| CGNS/分区/ghost/准备 | `src/Geom/Mesh/Mesh_Helpers.hpp` 及 `src/Geom/Mesh/` |
| GMRES | `src/Solver/Linear.hpp` |
| 入口 | `app/ACM/`, `app/ACMVariable/`, 两模块 `SingleBlockApp.hpp` |

# 结论

DNDSR 的不可压缩计算不是“在 Euler 求解器中关闭密度变化”，而是两套独立的 ACM 方程、边界、特征系统和隐式块算子。其核心是：用 $\Gamma$ 在伪时间引入人工双曲性，用 CFV 在一般非结构网格上构造高阶面状态，用光滑度指标与 LocalExtrema/WBAP/CWBAP 控制非光滑区，用 Roe/Rusanov 和合流 Hermite 矩阵函数处理一般 $\alpha$ 及特征碰撞，再通过 SSPRK3、隐式 LU--SGS/GMRES 或 BDF2 双时间把空间残差推进到稳态/物理时刻。网格、周期、ghost、壁距和分布式线性代数不是外围细节，而是这一离散链保持守恒、稳定和并行一致性的组成部分。

