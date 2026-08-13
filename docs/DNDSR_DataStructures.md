# DNDSR 项目数据结构总览

> 生成时间：2026-06-02
> 项目：DNDSR —— C++17 / Python CFD 研究代码（Compact Finite Volume，MPI 并行，可选 CUDA GPU 支持）

---

## 目录

1. [核心类型别名与常量 (DNDS/Defines.hpp)](#1-核心类型别名与常量)
2. [MPI 封装 (DNDS/MPI.hpp)](#2-mpi-封装)
3. [数组核心 (DNDS/Array.hpp / ArrayBasic.hpp)](#3-数组核心)
4. [MPI 感知数组 (DNDS/ArrayTransformer.hpp)](#4-mpi-感知数组)
5. [父子数组对 (DNDS/ArrayPair.hpp)](#5-父子数组对)
6. [派生数组类型 (DNDS/ArrayDerived/)](#6-派生数组类型)
7. [自由度数组 (DNDS/ArrayDOF.hpp)](#7-自由度数组)
8. [索引映射 (DNDS/IndexMapping.hpp)](#8-索引映射)
9. [数组重分布 (DNDS/ArrayRedistributor.hpp)](#9-数组重分布)
10. [序列化系统 (DNDS/Serializer*.hpp)](#10-序列化系统)
11. [设备 / CUDA 抽象 (DNDS/DeviceStorage.hpp / Vector.hpp)](#11-设备-cuda-抽象)
12. [配置系统 (DNDS/ConfigRegistry.hpp / ConfigParam.hpp)](#12-配置系统)
13. [工具类 (DNDS/ObjectPool.hpp / Profiling.hpp 等)](#13-工具类)
14. [几何基础 (Geom/Geometric.hpp)](#14-几何基础)
15. [单元系统 (Geom/ElemEnum.hpp / ElementTraits.hpp / Elements.hpp)](#15-单元系统)
16. [网格核心 (Geom/Mesh.hpp)](#16-网格核心)
17. [边界条件 (Geom/BoundaryCondition.hpp)](#17-边界条件)
18. [周期性 (Geom/PeriodicInfo.hpp)](#18-周期性)
19. [点云与空间搜索 (Geom/PointCloud.hpp / Octree.hpp)](#19-点云与空间搜索)
20. [数值积分 (Geom/Quadrature.hpp)](#20-数值积分)
21. [径向基函数 (Geom/RadialBasisFunction.hpp)](#21-径向基函数)
22. [OpenFOAM 网格 I/O (Geom/OpenFOAMMesh.hpp)](#22-openfoam-网格-io)
23. [串行重排序与分区 (Geom/SerialAdjReordering.hpp / Metis.hpp)](#23-串行重排序与分区)
24. [基函数与微分算子 (Geom/BaseFunction.hpp / DiffTensors.hpp)](#24-基函数与微分算子)
25. [有限体积基础 (CFV/FiniteVolume.hpp)](#25-有限体积基础)
26. [变分重构 (CFV/VariationalReconstruction.hpp)](#26-变分重构)
27. [CFV 设置与定义 (CFV/VRSettings.hpp / VRDefines.hpp)](#27-cfv-设置与定义)
28. [限制器 (CFV/Limiters.hpp)](#28-限制器)
29. [模型求值器 (CFV/ModelEvaluator.hpp)](#29-模型求值器)
30. [Euler 求解器核心 (Euler/Euler.hpp / EulerEvaluator.hpp)](#30-euler-求解器核心)
31. [Euler 求解器设置 (Euler/EulerEvaluatorSettings.hpp / EulerSolver.hpp)](#31-euler-求解器设置)
32. [Euler 边界条件与特殊场 (Euler/EulerBC.hpp / Gas.hpp / CLDriver.hpp)](#32-euler-边界条件与特殊场)
33. [Jacobian 与线性求解 (Euler/EulerJacobian.hpp / Solver/)](#33-jacobian-与线性求解)
34. [EulerP GPU 求值器 (EulerP/)](#34-eulerp-gpu-求值器)
35. [ODE 与线性求解器 (Solver/ODE.hpp / Linear.hpp / Direct.hpp)](#35-ode-与线性求解器)

---

## 1. 核心类型别名与常量

**文件**: `src/DNDS/Defines.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `real` | `double` | 项目全局浮点标量类型 |
| `index` | `int64_t` | 全局/局部索引 |
| `rowsize` | `int32_t` | 行/列大小与步长 |
| `ssp<T>` | `std::shared_ptr<T>` | 项目全局共享指针别名 |
| `t_IndexVec` | `std::vector<index>` | 索引向量 |
| `t_RowsizeVec` | `std::vector<rowsize>` | 行大小向量 |
| `tDiFj` | `Eigen::Matrix<real,-1,-1,Eigen::RowMajor>` | 行优先动态矩阵 |
| `DynamicSize` | 常量 `= -1` | 运行时固定行大小编译标记 |
| `NonUniformSize` | 常量 `= -2` | 每行可变大小（CSR）编译标记 |
| `NoAlign` | 常量 `= -1024` | 无填充对齐标记 |
| `UnInitReal` / `UnInitIndex` / `UnInitRowsize` | 常量 | 未初始化哨兵值（`NAN`, `INT64_MIN`, `INT32_MIN`） |
| `Empty` | struct | 可平凡复制占位符，接受任意赋值 |
| `EmptyNoDefault` | struct | 同 `Empty`，但无默认构造 |
| `ObjectNaming` | class | 混入基类，提供运行时实例名称（用于 Array 及子类） |
| `ObjName` | struct | `make_ssp()` 命名对象的标签类型 |

---

## 2. MPI 封装

**文件**: `src/DNDS/MPI.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `MPIInfo` | struct | `{MPI_Comm comm, int rank, int size}` —— MPI 操作核心句柄 |
| `MPITypePairHolder` | struct | `std::vector<std::pair<MPI_int, MPI_Datatype>>` 的 RAII 包装，自动释放已提交的 MPI 数据类型 |
| `MPIReqHolder` | struct | `std::vector<MPI_Request>` 的 RAII 包装，自动释放未完成的请求 |
| `MPIBufferHandler` | class | 线程安全单例，管理 MPI 缓冲发送缓冲区 |
| `ResourceRecycler` | class | 线程安全单例，注册清理 lambda（被 MPITypePairHolder/MPIReqHolder 使用） |
| `CommStrategy` | class | 单例，选择数组通信策略：`HIndexed`（MPI 自定义类型）或 `InSituPack`（手动打包） |

---

## 3. 数组核心

**文件**: `src/DNDS/ArrayBasic.hpp`, `src/DNDS/Array.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `DataLayout` | enum | `ErrorLayout`, `TABLE_StaticFixed`, `TABLE_Fixed`, `TABLE_Max`, `TABLE_StaticMax`, `CSR` |
| `ArrayLayout<T,rs,rm,al>` | class | 编译时布局描述符，将模板参数映射到 `DataLayout` |
| `ArrayView<T,rs,rm,al>` | class | Array 数据的非拥有只读视图，可在设备上调用。含嵌套 `RowView` |
| `ArrayIteratorBase<Derived>` | class | CRTP 随机访问迭代器基类，用于逐行遍历 Array |
| `Array<T,rs,rm,al>` | class | 核心二维变长数据容器。支持五种布局：静态固定、动态固定、静态最大、动态最大（TABLE）、CSR。拥有 `host_device_vector<value_type>` 数据，可选 `_pRowStart`（CSR）与 `_pRowSizes`（TABLE_Max）。可序列化到 JSON/HDF5 |

---

## 4. MPI 感知数组

**文件**: `src/DNDS/ArrayTransformer.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `ParArray<T,rs,rm,al>` | class | `Array` + MPI 上下文（`MPIInfo`）、全局偏移映射（`pLGlobalMapping`）与 MPI 数据类型信息。支持集体序列化 |
| `ArrayTransformer<T,rs,rm,al>` | class | 管理父子 `ParArray` 的幽灵/光晕通信。构建 push/pull MPI 类型，支持持久通信、基于 pull 与 push 的幽灵映射 |
| `ArrayTransformerType<TArray>::Type` | type alias | 元函数，提取给定数组类型的变换器类型 |

---

## 5. 父子数组对

**文件**: `src/DNDS/ArrayPair.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `ArrayPair<TArray>` | struct | 持有 `father`（拥有的本地数据）+ `son`（幽灵/光晕数据）`ssp<TArray>` 及 `ArrayTransformer`。提供统一的 `operator[]`/`operator()` 访问父子数据，支持幽灵通信与重分布序列化 |
| `ArrayPairDeviceView<B,TArray>` | struct | 父子对的可变设备端视图 |
| `ArrayPairDeviceViewConst<B,TArray>` | struct | 父子对的常量设备端视图 |

---

## 6. 派生数组类型

**文件**: `src/DNDS/ArrayDerived/`

| 名称 | 类型 | 说明 |
|------|------|------|
| `ArrayAdjacency<rs,rm,al>` | class | CSR 风格索引数组，用于网格连通性（如 `cell2node`）。继承 `ParArray<index,…>`，每行返回 `AdjacencyRow<index>` |
| `ArrayIndex` | class | 单列索引数组（`ArrayAdjacency<1>` 子类） |
| `ArrayEigenVector<vs,rm,al>` | class | 每行为 Eigen 向量（`Eigen::Map<Vector>`） |
| `ArrayEigenMatrix<ni,nj,ni_max,nj_max,al>` | class | 每行为 Eigen 矩阵（`Eigen::Map<Matrix>`）。支持静态、动态、非统一每行矩阵维度 |
| `ArrayEigenMatrixBatch` | class | 动态批大小的 Eigen 矩阵批次 |
| `ArrayEigenUniMatrixBatch<n_row,n_col>` | class | 固定大小 Eigen 矩阵的均匀批次 |
| `AdjacencyRow<T>` | struct | 单个邻接行的非拥有视图（指针 + 大小），设备可调用 |
| `ArrayEigenMatrixDeviceView<…>` | class | `ArrayEigenMatrix` 的设备端视图 |

---

## 7. 自由度数组

**文件**: `src/DNDS/ArrayDOF.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `ArrayDof<n_m,n_n>` | class | 继承 `ArrayEigenMatrixPair`。MPI 集体向量空间操作：`+=`、`-=`、`*=`、`norm2`、`dot`、`min`、`max`、`sum`、`componentWiseNorm1`、`addTo`。通过 `ArrayDofOp` 分发到 Host 或 CUDA 后端 |
| `ArrayDofOp<Backend,n_m,n_n>` | class | 静态分发类，实现 DOF 向量空间操作（Host 或 CUDA 特化） |
| `ArrayDofDeviceView<B,n_m,n_n>` | class | `ArrayDof` 的可变设备视图 |
| `ArrayDofDeviceViewConst<B,n_m,n_n>` | class | `ArrayDof` 的常量设备视图 |

---

## 8. 索引映射

**文件**: `src/DNDS/IndexMapping.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `GlobalOffsetsMapping` | class | 基于每秩前缀和偏移的局部<->全局索引映射。支持 `operator()(rank,local)` → global 与 `search(global)` → `(rank,local)` |
| `OffsetAscendIndexMapping` | class | 基于 pull 或 push 的光晕模式的幽灵索引映射。存储排序后的 `ghostIndex`、每秩 `ghostSizes`/`ghostStart` 及反向 `pushingIndexGlobal`。支持主数据/幽灵数据的二分查找 |

---

## 9. 数组重分布

**文件**: `src/DNDS/ArrayRedistributor.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `BuildRedistributionPullingIndex()` | function | 三轮 MPI 会合（Alltoallv），构建跨不同 MPI 分区的重分布拉取索引映射 |
| `RedistributeArrayWithTransformer<TArray>()` | function | 核心重分布例程：给定读取数组 + 原始索引，通过临时 `ArrayTransformer` 将所需行拉取到输出数组 |

---

## 10. 序列化系统

**文件**: `src/DNDS/SerializerBase.hpp`, `SerializerJSON.hpp`, `SerializerH5.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `SerializerBase` | class (abstract) | 标量、向量、字节数组的读写接口。支持每秩（JSON）与集体（HDF5）模式 |
| `SerializerBaseSSP` | type alias | `ssp<SerializerBase>` |
| `ArrayGlobalOffset` | class | 描述一秩的分布式数组片段：`{size, offset}`。特殊值：`Unknown`、`Parts`、`EvenSplit`、`One` |
| `SerializerJSON` | class | 每秩 JSON 序列化器（nlohmann::json 后端） |
| `SerializerH5` | class | 基于 MPI-IO 的集体 HDF5 序列化器 |

---

## 11. 设备 / CUDA 抽象

**文件**: `src/DNDS/DeviceStorage.hpp`, `Vector.hpp`, `DeviceView.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `DeviceBackend` | enum class | `Unknown`, `Host`, `CUDA`, `Custom1` |
| `DeviceStorageBase` | class (abstract) | 设备内存接口：raw_ptr、H<->D 拷贝、大小查询 |
| `DeviceStorage<B>` | class template | 具体设备存储，按 `DeviceBackend` 特化 |
| `device_storage_factory<B>` | struct | 工厂，创建 `unique_ptr`/`shared_ptr` `DeviceStorageBase` 实例 |
| `host_device_vector<T>` | class template | 主 Host-Device 向量（默认 `host_device_vector_r1`）。管理独立 Host 与可选 Device 分配，支持 `to_device()` / `to_host()` / `deviceView()` |
| `vector_DeviceView<B,T,TSize>` | class | 连续类型数组的非拥有设备端视图 |
| `ArrayDeviceView<B,T,rs,rm,al>` | class | `Array` 的设备端视图，针对 `Host` 与 `CUDA` 特化。继承 `ArrayView` |

---

## 12. 配置系统

**文件**: `src/DNDS/ConfigRegistry.hpp`, `ConfigParam.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `ConfigTypeTag` | enum class | JSON Schema 类型标签：`Bool`、`Int`、`Real`、`String`、`Enum`、`Array`、`Object`、`ArrayOfObjects`、`MapOfObjects`、`Json` |
| `CheckResult` | struct | `{bool passed; std::string message;}` —— 验证检查结果 |
| `ConfigContext` | struct | 上下文感知验证的运行时上下文：`nVars`、`dim`、`gDim`、`modelCode` |
| `FieldMeta` | struct | 单个配置字段的描述符：名称、描述、类型标签、读写 lambda、Schema 条目 lambda、范围约束、枚举值、辅助信息 |
| `CrossFieldCheck` | type alias | `std::function<CheckResult(const void*)>` |
| `ContextualCheck` | type alias | `std::function<CheckResult(const void*, const ConfigContext&)>` |
| `ConfigRegistry<T>` | class template | 每类型单例，存储 `FieldMeta` 列表与验证检查。提供 `readFromJson`、`writeToJson`、`emitSchema`、`validate` |
| `ConfigSectionBuilder<T>` | class | pybind11 风格注册器，传递给 `DNDS_DECLARE_CONFIG`。方法：`field()`、`field_section()`、`field_array_of()`、`field_map_of()`、`field_json()`、`check()`、`check_ctx()`、`post_read()` |
| `ConfigTypeTagOf<T>` | struct | 类型特征，将 C++ 类型映射到 `ConfigTypeTag` |
| `Config::RangeTag` / `InfoTag` / `EnumValuesTag` | struct | 字段注册的标签参数（范围约束、辅助信息、枚举值） |

---

## 13. 工具类

**文件**: `src/DNDS/ObjectPool.hpp`, `Profiling.hpp` 等

| 名称 | 类型 | 说明 |
|------|------|------|
| `ObjectPool<T>` | class template | 预分配的可复用对象池，带 RAII 检出（`ObjectPoolAllocated`） |
| `PerformanceTimer` | class | 单例挂钟计时器，带命名分类（RHS、Comm、LinSolve 等） |
| `ScalarStatistics` | class | 运行均值与标准差累加器（Welford 算法） |
| `vector_hash<T>` | struct | `std::vector<T>` 的哈希函数（通过 XOR 组合元素哈希） |
| `array_hash<T,s>` | struct | `std::array<T,s>` 的哈希函数 |
| `is_fixed_data_real_eigen_matrix<T>` | struct | 检测固定大小 `Eigen::Matrix<real,…>` 类型的类型特征 |

---

## 14. 几何基础

**文件**: `src/Geom/Geometric.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `t_index` | type alias | `int32_t` —— Geom 模块紧凑索引类型 |
| `t_real` | type alias | `double` |
| `tPoint` | type alias | `Eigen::Vector3d` —— 3D 坐标 |
| `tJacobi` | type alias | `Eigen::Matrix3d` —— 雅可比矩阵 |
| `tGPoint` | type alias | `Eigen::Matrix3d` —— 旋转/线性变换矩阵 |
| `tPointPortable` | struct | 设备可移植的 `std::array<real,3>` 包装器，带 `map()` → `Eigen::Map` |
| `tGPointPortable` | struct | 设备可移植的 `std::array<real,9>` 包装器，用于 3×3 矩阵 |
| `tSmallCoords` | type alias | `Eigen::Matrix<real, 3, Dynamic>` —— 单元的堆叠节点坐标 |
| `SmallCoordsAsVector` | struct | `tSmallCoords` 的薄包装器，提供每列 `operator[]` |
| `tLocalMatStruct` | type alias | `std::vector<std::vector<index>>` —— 局部 CSR 邻接 |

---

## 15. 单元系统

**文件**: `src/Geom/ElemEnum.hpp`, `ElementTraitsBase.hpp`, `ElementTraits.hpp`, `Elements.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `ElemType` | enum | 所有支持的 CGNS 兼容单元：`Line2`、`Line3`、`Tri3`、`Tri6`、`Quad4`、`Quad9`、`Tet4`、`Tet10`、`Hex8`、`Hex27`、`Prism6`、`Prism18`、`Pyramid5`、`Pyramid14` |
| `ParamSpace` | enum | 参考参数空间：`LineSpace`、`TriSpace`、`QuadSpace`、`TetSpace`、`HexSpace`、`PrismSpace`、`PyramidSpace` |
| `ElementTraits<ElemType>` | template struct | 主模板（未定义）；按单元完全特化，存放静态常量：`dim`、`order`、`numVertices`、`numNodes`、`numFaces`、`paramSpace`、`standardCoords`、`faceNodes`、`elevatedType`、`elevSpans`、`vtkNodeOrder`、`vtkCellType` 等 |
| `ShapeFuncImpl<ElemType>` | template struct | 主模板（未定义）；自动生成特化，提供 `Diff0`…`Diff3` 形函数求值器 |
| `Element` | struct | 运行时薄包装器，围绕 `ElemType` 值。分派到 `ElementTraits` / `ShapeFuncImpl` 进行查询（`GetNumNodes`、`GetNj`、`GetD1Nj`、`GetDiNj`、`ObtainFace`、`ExtractFaceNodes` 等） |
| `tElevSpan` | type alias | `std::array<t_index, 8>` —— 阶提升的父节点索引 |
| `tBisectSub` | type alias | `std::array<t_index, 8>` —— h 细化的子单元节点映射 |
| `tVTKNodeOrder` | type alias | `std::array<int, 27>` —— DNDS→VTK 节点排列 |

---

## 16. 网格核心

**文件**: `src/Geom/Mesh.hpp`, `Mesh_DeviceView.hpp`

### 枚举

| 名称 | 类型 | 说明 |
|------|------|------|
| `MeshLoc` | enum | 实体位置：`Unknown`、`Node`、`Face`、`Cell` |
| `MeshAdjState` | enum | 邻接索引解释：`Adj_Unknown`、`Adj_PointToLocal`、`Adj_PointToGlobal` |
| `MeshElevationState` | enum | `Elevation_Untouched`、`Elevation_O1O2` |

### 关键结构 / 类

| 名称 | 类型 | 说明 |
|------|------|------|
| `ElemInfo` | struct | 2×`int32_t` 打包元数据：`type`（ElemType）+ `zone`（边界 ID / 内部标记）。MPI 可通信 |
| `UnstructuredMesh` | struct | **中心分布式网格对象**。继承 `DeviceTransferable`。持有 `ArrayPair`：坐标、cell2node、bnd2node、cell2cell、face2node、face2cell、cell2face、node2cell、node2bnd，以及周期性数组（`tPbiPair`）、`ElemInfo` 数组、VTK 输出缓冲区、父子映射、提升信息、壁面距离数组等。提供幽灵构建、索引转换、重排序、提升、二分、周期性重建、VTK/CGNS/HDF5 输出 |
| `UnstructuredMeshDeviceView<DeviceBackend>` | template struct | `UnstructuredMesh` 的轻量 CUDA/设备副本，仅含数组对的设备视图句柄，用于 GPU 核函数 |
| `PartitionOptions` | struct | JSON 可序列化的 ParMetis/Metis 选项：`metisType`、`metisUfactor`、`metisSeed`、`edgeWeightMethod`、`metisNcuts` |
| `UnstructuredMeshSerialRW` | struct | `UnstructuredMesh` 的串行读写伴侣。持有串行专用数组（`coordSerial`、`cell2nodeSerial`…）、分区向量、串行输出变换器。方法：`ReadFromCGNSSerial`、`ReadFromOpenFOAMAndConvertSerial`、`Deduplicate1to1Periodic`、`BuildCell2Cell`、`MeshPartitionCell2Cell`、`PartitionReorderToMeshCell2Cell` |
| `HDF5OutSetting` | struct | HDF5 输出的块大小、deflate 级别、集体 I/O 标志 |
| `ElevationInfo` | struct | 基于 RBF 的 O1→O2 提升参数：半径、最大角度、迭代次数、核类型等 |
| `WallDistOptions` | struct | 壁面距离计算的 JSON 可序列化选项 |

### 数组类型别名（Mesh_DeviceView.hpp）

| 名称 | 类型 | 说明 |
|------|------|------|
| `tAdjPair` / `tAdj` | type alias | 变长邻接 `ArrayPair`（CSR 风格） |
| `tAdj1Pair` / `tAdj1` | type alias | 固定-1 邻接 `ArrayPair` |
| `tAdj2Pair` / `tAdj2` | type alias | 固定-2 邻接 `ArrayPair` |
| `tCoordPair` / `tCoord` | type alias | 3D 坐标 `ArrayPair`（`ArrayEigenVector<3>`） |
| `tElemInfoArrayPair` / `tElemInfoArray` | type alias | `ElemInfo` 的 `ArrayPair` |
| `tPbiPair` / `tPbi` | type alias | 变长 `NodePeriodicBits` 的 `ArrayPair` |
| `tIndPair` / `tInd` | type alias | 索引 `ArrayPair` |
| `tFGetName` / `tFGetData` / `tFGetVecData` | type alias | VTK 场输出回调签名 |

---

## 17. 边界条件

**文件**: `src/Geom/BoundaryCondition.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `BC_ID_INTERNAL` … `BC_ID_DEFAULT_MAX` | 常量 | 硬编码的边界条件 ID 值（内部、周期性 1-3 + 供体、壁面、远场等） |
| `BCName_2_ID` | class | 将边界名称映射到整数 ID（自动分配新 ID） |
| `t_FBCName_2_ID` / `t_FBCID_2_Name` | type alias | 名称<->ID 翻译函数类型 |
| `AutoAppendName2ID` | struct | 从默认名称映射开始、对未识别名称自动递增 ID 的函子 |
| `FaceIDIsExternalBC` / `FaceIDIsPeriodic` / `FaceIDIsPeriodicMain` / `FaceIDIsPeriodicDonor` | functions | 面区 ID 的内联谓词辅助函数 |

---

## 18. 周期性

**文件**: `src/Geom/PeriodicInfo.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `NodePeriodicBits` | struct | 位掩码（uint8_t），编码 P1/P2/P3 周期性偏移。可设备复制；有 MPI 通信类型 |
| `NodeIndexPBI` | struct | `(index i, NodePeriodicBits pbi)` 对 —— 唯一标识周期性节点镜像 |
| `NodePeriodicBitsRow` | class | 原始 `NodePeriodicBits*` 的行视图（迭代器接口，`bitandReduce`） |
| `ArrayNodePeriodicBits<...>` | template class | `ParArray` 子类，`operator[]` 返回 `NodePeriodicBitsRow` |
| `Periodicity` | struct | 存储 3 组 `rotation`、`rotationCenter`、`translation`（`tGPointPortable` / `tPointPortable`）。提供 `TransCoord`、`TransCoordBack`、`TransVector`、`TransMat`、`GetCoordByBits`、`GetVectorByBits` 等。JSON/HDF5 可序列化 |

---

## 19. 点云与空间搜索

**文件**: `src/Geom/PointCloud.hpp`, `Octree.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `PointCloudKDTree` | struct | nanoflann 兼容适配器，基于 `std::vector<tPoint>` 进行 k-d 树查询 |
| `PointCloudFunctional` | struct | nanoflann 兼容适配器，使用 `std::function<tPoint(size_t)>` 代替存储点 |
| `Octree` | class | **占位符/未完成** 的 3D 八叉树。含内嵌 `Node` 结构，带子节点指针、范围与索引 |

---

## 20. 数值积分

**文件**: `src/Geom/Quadrature.hpp`, `QuadratureHub.hpp`, `Quadratures/*.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `INT_SCHEME_*` | 常量 | 所有参数空间的方案标识符（也等于点数）：Line (1,2,3,4)、Quad (1,4,9,16)、Tri (1,3,6,7,12)、Tet (1,4,8,14,24)、Hex (1,8,27,64)、Prism (1,6,18,21,48)、Pyramid (1,8,27,64) |
| `INT_ORDER_MAX` | 常量 | `6` —— 支持的最大积分阶数 |
| `SummationNoOp` | class | 积分模板的空操作累加器类型 |
| `__TNBufferAtQuadrature` | static struct | 全局缓存，预计算所有单元类型与阶数（至 `INT_ORDER_MAX`）在每个积分点处的 `tD01Nj` 形函数缓冲区 |
| `Quadrature` | struct | 主积分 API。由 `Element` + `int_order` 构造。提供 `Integration`、`IntegrationSimple`、`GetQuadraturePointInfo`、`GetWeight`、`GetNumPoints` |
| `GetQuadratureScheme` | function | 将 `(ParamSpace, int_order)` 映射到方案常量 |
| `GetQuadraturePoint` | function | 按 `(ParamSpace, scheme, iG)` 检索 `(pParam, w)` |

---

## 21. 径向基函数

**文件**: `src/Geom/RadialBasisFunction.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `RBFKernelType` | enum | `Distance`、`DistanceA1`、`InversedDistanceA1`、`InversedDistanceA1Compact`、`Gaussian`、`CPC2`、`CPC0` |
| `FRBFBasis` | function template | 从距离矩阵求值 RBF 核矩阵 |
| `RBFCPC2` / `RBFInterpolateSolveCoefs` / `RBFInterpolateSolveCoefsNoPoly` | function templates | 网格提升使用的 RBF 插值辅助函数 |

---

## 22. OpenFOAM 网格 I/O

**文件**: `src/Geom/OpenFOAMMesh.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `OpenFOAMBoundaryCondition` | struct | `{ type, nFaces, startFace }` |
| `OpenFOAMReader` | struct | 解析 OpenFOAM polyMesh 文件：`points`、`faces`、`owner`、`neighbour`、`boundaryConditions` |
| `OpenFOAMConverter` | struct | 将 OpenFOAM polyMesh 转换为 DNDS 的 `cell2node` + `ElemInfo` 数组。当前支持 Hex8 重建 |

---

## 23. 串行重排序与分区

**文件**: `src/Geom/SerialAdjReordering.hpp`, `CorrectRCM.hpp`, `Metis.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `PartitionSerialAdj_Metis` | function | Metis K-way / 递归分区，作用于局部 CSR 图 |
| `ReorderSerialAdj_Metis` | function | Metis 节点嵌套剖分填充减少排序 |
| `ReorderSerialAdj_BoostMMD` | function | Boost 最小度排序 |
| `ReorderSerialAdj_BoostRCM` | function | Boost Cuthill-McKee 排序（带宽缩减） |
| `ReorderSerialAdj_CorrectRCM` | function | 自定义正确 RCM 排序（处理不连通图） |
| `ReorderSerialAdj_PartitionMetisC` | function | 两级分区 + 可选内部 RCM 重排序 |
| `OffsetIterator<T>` / `OffsetRange<T>` | class templates | 迭代器/范围包装器，对每次解引用值加常数偏移（用于子图分区） |
| `CorrectRCM::UndirectedGraphProxy` | class template | 在通用邻接函子上操作的 BFS / Cuthill-McKee 排序引擎 |
| `CorrectRCM::hash_pair` | struct | 图边检查使用的 `std::pair` 哈希 |

---

## 24. 基函数与微分算子

**文件**: `src/Geom/BaseFunction.hpp`, `DiffTensors.hpp`, `EigenTensor.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `diffOperatorOrderList` / `diffOperatorDimList` | static arrays | 将扁平导数索引映射到 `(dx, dy, dz)` 指数与微分维度 |
| `diffOperatorIJK2I` / `diffOperatorIJK2I2D` | static structs | 将多重索引导数元组转换为扁平索引的查找表 |
| `t_diffOpIJK2I` | type alias | 持有 1/2/3 阶导数索引查找表的元组 |
| `FPolynomialFill2D` / `FPolynomialFill3D` | functions | 手优化与回退多项式基 + 导数求值 |
| `NormSymDiffOrderTensorV` | function template | 导数张量的对称内积 |
| `TransSymDiffOrderTensorV` | function template | 导数张量的坐标变换（0–3 阶，2D/3D） |
| `ConvertDiffsLinMap` / `ConvertDiffsFullMap` | function templates | 在参数坐标与物理坐标之间转换形函数导数 |
| `DxDxi2DxiDx` | function template | 倒置紧凑导数数组（参数 <-> 物理） |
| `CFVPeriodicity` | class | 扩展 `Periodicity`，增加 `TransDiValueInplace` / `TransDiValueBackInplace`，用于周期性面间的导数变换 |
| `ETensorR3<T,d0,d1,d2>` | class template | 紧凑 3 阶张量，带 `MatTransform0/1/2` 与维度变换。用于 3 阶导数张量坐标变换 |

---

## 25. 有限体积基础

**文件**: `src/CFV/FiniteVolume.hpp`, `FiniteVolume_DeviceView.hpp`, `FiniteVolumeSettings.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `FiniteVolume` | class | 非结构网格上的有限体积机制基类。持有几何度量（体积、面面积、重心、法向、积分数据、惯性张量）、单元/面属性（`RecAtr`）及构建 DOF 数组的辅助方法。支持通过 `DeviceTransferable<FiniteVolume>` 进行设备传输 |
| `FiniteVolumeDeviceView<B>` | class template | `FiniteVolume` 的 GPU 设备镜像（按 `DeviceBackend` 模板化）。以设备可调用的方法暴露相同几何查询（单元体积、面法向、积分点、周期性变换） |
| `FiniteVolumeSettings` | struct | `FiniteVolume` 的 POD 设置结构。持有 `maxOrder`、`intOrder`、`ignoreMeshGeometryDeficiency`、`nIterCellSmoothScale`。使用 `DNDS_DECLARE_CONFIG` 进行 JSON 序列化 |

---

## 26. 变分重构

**文件**: `src/CFV/VariationalReconstruction.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `VariationalReconstruction<dim>` | class template | 高阶**变分重构**类，按空间维度（2 或 3）模板化。继承 `FiniteVolume`。管理基函数权重、重构系数矩阵、迭代求解器（SOR/Jacobi）、Green-Gauss 回退重构、光滑性指示器与 WBAP 限制器。包含两个私有实现结构：`VRBaseWeight` 与 `VRCoefficients` |
| `VariationalReconstruction::VRBaseWeight` | struct (private) | 分组成员结构。持有面 aligned 尺度、面主坐标、单元基矩、面积分权重、缓存的单元/面积分点与重心处的 diff-base 值、边界 VR 点缓存（`BndVRPointCache`） |
| `VariationalReconstruction::VRBaseWeight::BndVRPointCache` | struct (private) | 边界积分点缓存项：`norm`、`PPhy`、`JDet`、`D0Bj`（diff-base 行向量） |
| `VariationalReconstruction::VRCoefficients` | struct (private) | 分组成员结构。持有重构的线性系统数据：`matrixAB`、`vectorB`、`matrixAAInvB`（主重构矩阵）、`vectorAInvB`、`matrixSecondary`（次级重构矩阵）、`matrixAHalf_GG`（Green-Gauss 半矩阵）、可选原始 `matrixA`、可选 Cholesky 因子 |

---

## 27. CFV 设置与定义

**文件**: `src/CFV/VRSettings.hpp`, `VRDefines.hpp`

### 结构

| 名称 | 类型 | 说明 |
|------|------|------|
| `VRSettings` | struct | 继承 `FiniteVolumeSettings`。增加 VR 专用参数：`intOrderVR`、`intOrderVRBC`、`cacheDiffBase`、`jacobiRelax`、`SORInstead`、光滑/限制器参数（`smoothThreshold`、`WBAP_nStd`、`normWBAP`、`limiterBiwayAlter`）、`subs2ndOrder`、`svdTolerance`、`bcWeight`，以及两个嵌套设置结构 |
| `VRSettings::BaseSettings` | struct (nested) | 控制基函数行为：`localOrientation`（使用单元主坐标）、`anisotropicLengths` |
| `VRSettings::FunctionalSettings` | struct (nested) | 控制变分泛函：尺度类型、方向/几何权重方案、各向异性泛函标志、Green-Gauss 权重、手动权重数组 |
| `RecAtr` | struct | 每单元/面的重构属性：`relax`、`Order`、`NDOF`、`NDIFF`、`intOrder`。用作 `cellAtr` / `faceAtr` 数组的元素类型 |
| `ModelSettings` | struct | 测试模型的简单 POD：`ax`、`ay`、`sigma` |
| `ModelEvaluator::EvaluateRHSOptions` | struct (nested) | RHS 求值选项标志：`direct2ndRec`、`direct2ndRec1stConv` |

### 枚举

| 名称 | 所属 | 说明 |
|------|------|------|
| `ScaleType` | `VRSettings::FunctionalSettings` | `UnknownScale=-1`、`MeanAACBB=0`、`BaryDiff=1`、`CellMax=2` —— 面长度尺度计算方式 |
| `DirWeightScheme` | `VRSettings::FunctionalSettings` | `UnknownDirWeight=-1`、`Factorial=0`、`HQM_OPT=1`、`ManualDirWeight=999`、`TEST_OPT=1000` —— 方向加权方案 |
| `GeomWeightScheme` | `VRSettings::FunctionalSettings` | `UnknownGeomWeight=-1`、`GWNone=0`、`HQM_SD=1`、`SD_Power=2` —— 几何加权方案 |
| `AnisotropicType` | `VRSettings::FunctionalSettings` | `UnknownAnisotropic=-1`、`InertiaCoord=0`、`InertiaCoordBB=1`、`Norm=2`、`CentDiff=3`、`WallDist=4`、`InertiaCoordBBNorm=5`、`InertiaCoordBBSym=6` —— 各向异性泛函类型 |

### 类型别名

| 别名 | 定义 | 说明 |
|------|------|------|
| `tCoeffPair` / `tCoeff` | `ArrayPair<ArrayEigenVector<NonUniformSize>>` | 每单元/面的非统一大小实向量（如积分点处的雅可比行列式） |
| `t3VecsPair` / `t3Vecs` | `ArrayPair<ArrayEigenUniMatrixBatch<3,1>>` | 每实体的 3 向量批次（如积分点坐标、单位法向） |
| `t3VecPair` / `t3Vec` | `Geom::tCoordPair` / `Geom::tCoord` | 每单元/面的单 3D 点（重心、形心） |
| `t3MatPair` / `t3Mat` | `ArrayPair<ArrayEigenMatrix<3,3>>` | 每单元/面的 3×3 矩阵（主坐标架、惯性张量） |
| `tVVecPair` / `tVVec` | `ArrayPair<ArrayEigenVector<DynamicSize>>` | 每实体的动态大小向量 |
| `tMatsPair` / `tMats` | `ArrayPair<ArrayEigenUniMatrixBatch<DynamicSize,DynamicSize>>` | 每实体的动态大小矩阵批次 |
| `tVecsPair` / `tVecs` | `ArrayPair<ArrayEigenUniMatrixBatch<DynamicSize,1>>` | 每实体的动态列向量批次 |
| `tVMatPair` / `tVMat` | `ArrayPair<ArrayEigenMatrix<DynamicSize,DynamicSize>>` | 每实体的动态 Eigen 矩阵 |
| `tRecAtrPair` / `tRecAtr` | `ArrayPair<ParArray<RecAtr,1>>` | 每单元/面的 `RecAtr` 数组 |
| `tURec<nVarsFixed>` | `ArrayDof<DynamicSize, nVarsFixed>` | 重构 DOF：每单元持有 `(NDOF-1) × nVars` 的多项式系数矩阵 |
| `tUDof<nVarsFixed>` | `ArrayDof<nVarsFixed, 1>` | 均值/守恒 DOF：每单元持有 `nVars × 1` 向量 |
| `tUGrad<nVarsFixed, gDim>` | `ArrayDof<gDim, nVarsFixed>` | 梯度 DOF：每单元持有 `gDim × nVars` 矩阵 |
| `tScalarPair` / `tScalar` | `ArrayPair<ArrayEigenVector<1>>` | 每单元/面的单标量值（如光滑性指示器） |

### 其他重要类型别名

| 别名 | 定义 | 说明 |
|------|------|------|
| `TFTrans` | `std::function<void(Eigen::Ref<MatrixXR>, Geom::t_index)>` | 周期性变换函子类型 |
| `tFGetBoundaryWeight` | `std::function<real(Geom::t_index, int)>` | 按区 ID 与阶数查找边界权重 |
| `TFBoundary<nVarsFixed>` | `std::function<Eigen::Vector<real,nVarsFixed>(...)>` | 重构的边界条件函子 |
| `TFBoundaryDiff<nVarsFixed>` | `std::function<Eigen::Vector<real,nVarsFixed>(...)>` | 用于雅可比应用的微分 BC 函子 |
| `TFPost<nVarsFixed>` | `std::function<void(Eigen::Matrix<real,1,nVarsFixed>&)>` | 重构值的后处理函子（用于光滑性指示器 V1） |
| `tFMEig<nVarsFixed>` | `std::function<tLimitBatch<nVarsFixed>(...)>` | 限制器的特征系统变换函子（左特征向量投影） |
| `tLimitBatch<nVarsFixed>` | `Eigen::Matrix<real, -1, nVarsFixed, 0, maxRecDOFBatch>` | 限制器过程内部使用的固定最大大小批次矩阵 |

---

## 28. 限制器

**文件**: `src/CFV/Limiters.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `PolynomialSquaredNorm<dim>` / `PolynomialDotProduct<dim>` | function templates | 多项式系数向量的加权 L2 范数/点积（WBAP/MEMM 限制器使用） |
| `FWBAP_L2_Multiway` / `FWBAP_L2_Biway` / `FMINMOD_Biway` / `FVanLeer_Biway` / `FMEMM_Multiway_Polynomial2D` 等 | functions | 限制器核：WBAP（多路/双路）、MINMOD、Van Leer、MEMM，含多项式范数与正交变体 |
| `DispatchBiwayLimiter<dim,nVarsFixed>` | function template | 根据 `limiterBiwayAlter` 设置分派双路限制器 |

---

## 29. 模型求值器

**文件**: `src/CFV/ModelEvaluator.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `ModelEvaluator` | class | 具体的对流-扩散测试模型（为绑定方便保留在 CFV 中）。包装 `VariationalReconstruction<2>`，使用重构梯度与用户提供的边界函子求值 PDE 右端项 |

---

## 30. Euler 求解器核心

**文件**: `src/Euler/Euler.hpp`, `EulerEvaluator.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `ArrayDOFV<nVarsFixed>` | class | 单元平均守恒变量的分布式 DOF 数组；继承 `CFV::tUDof`，增加 MPI 归约、分量最小值、点积、向量-标量操作 |
| `ArrayRECV<nVarsFixed>` | class | 分布式重构系数数组；继承 `CFV::tURec`，含类似 MPI 感知操作符 |
| `ArrayGRADV<nVarsFixed, gDim>` | class | 分布式梯度数组；继承 `CFV::tUGrad` |
| `JacobianValue<nVarsFixed>` | class | 雅可比存储容器（对角、对角块或完整），含类型化子块 |
| `JacobianValue::Type` | enum | `Diagonal`、`DiagonalBlock`、`Full` |
| `EulerModel` | enum | 流模型选择器：`NS`、`NS_SA`、`NS_2D`、`NS_3D`、`NS_SA_3D`、`NS_2EQ`、`NS_2EQ_3D`、`NS_EX`、`NS_EX_3D` |
| `RANSModel` | enum | 湍流模型选择器：`RANS_None`、`RANS_SA`、`RANS_KOWilcox`、`RANS_KOSST`、`RANS_RKE` |
| `EulerEvaluator<model>` | class template | 给定 `EulerModel` 的核心残差/雅可比求值器。管理通量求值、源项、边界生成、LUSGS/雅可比操作、限制器与 CL 驱动集成 |
| `EulerEvaluator::OutputOverlapDataRefs` | struct | 引用束（`u`、`uRec`、`betaPP`、`alphaPP`），传递给输出选取器 |

---

## 31. Euler 求解器设置

**文件**: `src/Euler/EulerEvaluatorSettings.hpp`, `EulerSolver.hpp`

### 求值器设置

| 名称 | 类型 | 说明 |
|------|------|------|
| `EulerEvaluatorSettings<model>` | struct | `EulerEvaluator` 的配置结构。持有黎曼求解器选择、气体属性、壁面距离设置、RANS 选项、旋转参考系、盒/平面/exprtk 初始化器与 CL 驱动 BC 名称 |
| `FrameConstRotation` | struct | 恒定旋转参考系设置（轴、中心、rpm） |
| `BoxInitializer` | struct | 带边界与值向量的空间盒区域初始化器 |
| `PlaneInitializer` | struct | 半空间平面初始化器（法向/偏移与值向量） |
| `ExprtkInitializer` | struct | 基于 exprtk 字符串的表达式初始化器 |
| `IdealGasProperty` | struct | 热力学属性：`gamma`、`Rgas`、`muGas`、`Pr`、`CpGas`、`TRef`、`CSutherland`、`muModel` |

### 求解器设置

| 名称 | 类型 | 说明 |
|------|------|------|
| `EulerSolver<model>` | class template | 顶层求解器驱动：网格 I/O、时间推进、隐式重构、输出、重启与收敛控制 |
| `Configuration` | struct | 主配置容器，聚合所有求解器子配置 |
| `TimeMarchControl` | struct | 时间步进参数：`dtImplicit`、`nTimeStep`、`odeCode`、`useRestart`、`tEnd` 等 |
| `ImplicitReconstructionControl` | struct | 重构求解器设置：显式/隐式、GMRES/PCG、阈值、零梯度选项 |
| `OutputControl` | struct | 控制台/日志/输出间隔、文件名、精度、VTK/Tecplot/HDF 标志 |
| `ImplicitCFLControl` | struct | CFL 数、局部 dt、光滑、RANS 松弛 |
| `ConvergenceControl` | struct | 内迭代限制、残差阈值、范数阶、体积加权、CL 驱动启用 |
| `DataIOControl` | struct | 网格文件、提升、二分、壁面距离、分区选项、序列化器设置 |
| `BoundaryDefinition` | struct | 周期性平移/旋转与边界配对的容差 |
| `LimiterControl` | struct | 斜率限制器设置：`useLimiter`、`usePPRecLimiter`、过程、部分限制 |
| `LinearSolverControl` | struct | 线性求解器设置：Jacobi/GS/ILU、SGS/GMRES 迭代、多级网格、缩放 |
| `CoarseGridLinearSolverControl` | struct | 多级网格每级粗网格求解器设置 |
| `RestartState` | struct | 重启簿记：步索引、先前 ODE 代码、重启文件路径 |
| `TimeAverageControl` | struct | 启用/禁用时间平均场输出 |
| `Others` | struct | 杂项：冻结被动标量、轴对称模式、重构矩阵写入器 |
| `PrintDataMode` | enum | `PrintDataLatest`、`PrintDataTimeAverage` |
| `RunningEnvironment` | struct | 可变运行时状态束：ODE 积分器、GMRES/PCG 实例、计时器、计数器、残差、dt 历史 |

---

## 32. Euler 边界条件与特殊场

**文件**: `src/Euler/EulerBC.hpp`, `Gas.hpp`, `CLDriver.hpp`, `SpecialFields.hpp`, `RANS_ke.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `EulerBCType` | enum | 边界条件类型：`BCFar`、`BCWall`、`BCWallInvis`、`BCWallIsothermal`、`BCOut`、`BCOutP`、`BCIn`、`BCInPsTs`、`BCSym`、`BCSpecial` |
| `BoundaryHandler<model>` | class template | 存储所有面区的 BC 值、类型、标志与名称<->ID 映射；JSON 可序列化 |
| `IntegrationRecorder` | struct | 面积分边界通量累加器，含 MPI 归约 |
| `AnchorPointRecorder<nVarsFixed>` | struct | 记录边界上最近锚点值（用于锚定 BC） |
| `OutputPicker` | class | 命名单元标量输出函数的注册表；按名称生成子集 |
| `OneDimProfile<nVarsFixed>` | struct | 1D 剖面记录器，支持均匀或 tanh 节点分布、区间平均与 CSV 输出 |
| `RiemannSolverType` | enum | 无粘通量黎曼求解器：`Roe`、`HLLC`、`HLLEP`、`HLLEP_V1`、`Roe_M1`…`Roe_M9` |
| `CLDriverSettings` | struct | CL 驱动配置：目标 CL、AoA 轴/方向、参考面积/动压、收敛窗口 |
| `CLDriver` | class | 自适应攻角驱动器，更新 AoA 以匹配目标升力系数 |
| `SpecialFields` (namespace) | — | 解析场生成器：`IsentropicVortex10`、`IsentropicVortex30`、`IsentropicVortexCent` |
| `RANS` (namespace) | — | 湍流模型模板函数：`GetMut_*`、`GetVisFlux_*`、`GetSource_*`，用于 Realizable k-epsilon、SST、k-omega Wilcox |

---

## 33. Jacobian 与线性求解

**文件**: `src/Euler/EulerJacobian.hpp`, `src/Solver/Direct.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `JacobianDiagBlock<nVarsFixed>` | class template | 每单元对角或块对角雅可比块，支持求逆 |
| `JacobianLocalLU<nVarsFixed>` | struct | 局部稀疏 LU 分解存储（LDU 格式），用于雅可比求解；继承 `LocalLUBase` |
| `JacobianLocalLDLT<nVarsFixed>` | struct | 局部稀疏 LDL^T 分解存储；继承 `LocalLDLTBase` |
| `DirectPrecControl` | struct | 直接预处理配置：`useDirectPrec`、`iluCode`、`orderingCode` |
| `SerialSymLUStructure` | struct | 串行对称 LU/LDL^T 的符号分解结构：下/上稀疏性、重排序映射、分区起点 |
| `LocalLUBase<Derived, tComponent, tVec>` | struct | 局部 LU 分解与三角求解的 CRTP 基（V1 与 V2 算法，OpenMP 并行） |
| `LocalLDLTBase<Derived, tComponent, tVec>` | struct | 局部 LDL^T 分解与对称求解的 CRTP 基 |

---

## 34. EulerP GPU 求值器

**文件**: `src/EulerP/EulerP.hpp`, `EulerP_Evaluator.hpp`, `EulerP_Physics.hpp`, `EulerP_BC.hpp`, `EulerP_ARS.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `EvaluatorArgBase<TDerived>` | class | CRTP 基，用于参数束，提供 `WaitAllPull` 进行设备数组同步 |
| `EvaluatorDeviceView<B>` | class | 求值器的轻量设备端视图（FV、BC 处理程序、物理） |
| `EvaluatorConfig` | class | 基于 JSON 的配置持有者，支持合并/补丁验证 |
| `Evaluator` | class | Host 端求值器，管理面缓冲区、设备视图与核函数：`RecGradient`、`Cons2PrimMu`、`Cons2Prim`、`EstEigenDt`、`RecFace2nd`、`Flux2nd` |
| `RecGradient_Arg` | struct | `RecGradient` 参数束：`u`、`uGrad`、标量数组/梯度 |
| `Cons2PrimMu_Arg` | struct | `Cons2PrimMu` 参数束：守恒 + 原始 DOF、压力、温度、声速、gamma、粘性 |
| `Cons2Prim_Arg` | struct | 简化版 `Cons2Prim` 参数束（无梯度/粘性） |
| `EstEigenDt_Arg` | struct | 基于特征值的 dt 估计参数束：`u`、`muCell`、`aCell`、面/单元 lambda 估计、`dt` |
| `RecFace2nd_Arg` | struct | 2 阶面重构参数束：单元/面 DOF 与梯度 |
| `Flux2nd_Arg` | struct | 2 阶通量求值参数束：左/右面状态、原始变量、通量输出、RHS 输出 |
| `PhysicsParams` | struct | 简单气体属性：`gamma`、`mu0`、`cp`、`Rg`、`TRef`、`muModel` |
| `PhysicsDeviceView<B>` | struct | 设备端物理视图，含 `Cons2Prim`、`Prim2Cons`、`Prim2Pressure`、`Prim2GammaAcousticSpeed`、`getMuTot` |
| `Physics` | struct | Host 端物理对象，带 `host_device_vector` 参考值与 JSON 序列化 |
| `BCType` | enum class | 边界类型：`Far`、`Wall`、`WallInvis`、`WallIsothermal`、`Out`、`OutP`、`In`、`InPsTs`、`Sym`、`Special` |
| `BCFunc_Impl<B, T>` | struct | 设备端 BC 应用的模板特化（CRTP 式每类型分派） |
| `BC_DeviceView<B>` | class | 单个 BC 的设备端视图：id、类型、值与 `apply()` 方法 |
| `BC` | class | Host 端 BC 对象，带 `host_device_vector` 值与设备视图工厂 |
| `BCHandlerDeviceView<B>` | class | 完整 BC 处理程序的设备端视图（`BC_DeviceView` 数组） |
| `BCInput` | struct | BC JSON 输入的反序列化辅助 |
| `BCHandler` | class | Host 端 BC 处理程序：从 `BCInput` 列表构建 `BC` 数组并提供设备视图 |
| `RoeEigenValueFixer` / `RoeAverageNS` / `RoeFluxFlow` | functions | 设备可调用的 Roe 通量例程 |

---

## 35. ODE 与线性求解器

**文件**: `src/Solver/ODE.hpp`, `Linear.hpp`, `Direct.hpp`, `Scalar.hpp`

| 名称 | 类型 | 说明 |
|------|------|------|
| `ImplicitDualTimeStep<TDATA, TDTAU>` | class | 隐式双时间步进的抽象基类：定义 `Step`、`getRHS`、`getRES`、`getLatestRHS` |
| `ImplicitEulerDualTimeStep<TDATA, TDTAU>` | class | 后向 Euler 双时间步进器 |
| `ImplicitSDIRK4DualTimeStep<TDATA, TDTAU>` | class | SDIRK4 / ESDIRK3 / ESDIRK2 / Trapezoid 多级隐式时间步进器，带 Butcher 表 |
| `ImplicitBDFDualTimeStep<TDATA, TDTAU>` | class | BDF1–BDF4 隐式时间步进器，固定阶系数与历史缓冲区 |
| `ImplicitVBDFDualTimeStep<TDATA, TDTAU>` | class | 变步长 BDF（VBDF）时间步进器，自适应 dt 限制 |
| `ImplicitHermite3SimpleJacobianDualStep<TDATA, TDTAU>` | class | Hermite-3（HM3）隐式双时间步进器，带中点与多级网格阻尼选项 |
| `ExplicitSSPRK3TimeStepAsImplicitDualTimeStep<TDATA, TDTAU>` | class | 以隐式双时间接口包装的显式 SSPRK3 |
| `GMRES_LeftPreconditioned<TDATA>` | class | 左预处理 GMRES，带重启、Arnoldi 正交化与最小二乘求解 |
| `PCG_PreconditionedRes<TDATA, TScalar>` | class | 预处理共轭梯度，带残差预处理与可选历史重置 |
| `Scalar::BisectSolveLower<TF>` | function | 标量二分求根 |

---

## 附录：核心继承与组合关系

### DNDS 数组继承体系

```
ObjectNaming
      -- Array<T,rs,rm,al>
              -- ParArray<T,rs,rm,al>
                      -- ArrayAdjacency<rs,rm,al>
                               -- ArrayIndex
                      -- ArrayEigenVector<vs,rm,al>
                      -- ArrayEigenMatrix<ni,nj,…>
                      -- (其他派生 ParArray 类型)

ArrayPair<TArray>
      -- ArrayAdjacencyPair
      -- ArrayEigenVectorPair
      -- ArrayEigenMatrixPair
      -- ArrayDof<n_m,n_n>   (增加向量空间操作 + MPI 归约)

ArrayView<T,rs,rm,al>
      -- ArrayDeviceView<Host,…>
      -- ArrayDeviceView<CUDA,…>
      -- (被上述设备视图使用)

IndexMapping:
    GlobalOffsetsMapping          (秩级 local<->global)
    OffsetAscendIndexMapping      (幽灵级 local<->global, pull/push)
```

### CFV 继承体系

```
FiniteVolume
      -- VariationalReconstruction<dim>
              -- VRBaseWeight      (基函数矩、权重、缓存)
              -- VRCoefficients    (A^-¹B 重构矩阵)
              -- 使用 Limiters.hpp (WBAP, MEMM, MINMOD 等)
```

### 网格核心关系

```
UnstructuredMesh              (中心分布式网格，MPI + 设备感知)
      -- UnstructuredMeshSerialRW   (串行伴侣：读取 CGNS/OpenFOAM 并分区)

Element  (运行时包装)
      -- ElementTraits<ElemType>    (编译时元数据)
      -- ShapeFuncImpl<ElemType>    (形函数求值)
```
