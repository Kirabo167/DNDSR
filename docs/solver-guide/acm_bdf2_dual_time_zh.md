# ACM BDF2 双时间推进说明

## 1. 实现范围

ACM 求解器现在支持定常伪时间推进和非定常 BDF2 双时间推进。BDF2 模块参考了 Euler
求解器的时间历史管理和隐式线性求解组织方式，并采用所给 FORTRAN 程序中的 ACM
物理时间方程。它没有直接复用 Euler 的通用 BDF 状态方程，因为 ACM 压力只参与伪时间
人工压缩过程，不应具有物理时间导数。

可选格式为：

- `BDF2DualTimeLUSGS`：内迭代直接使用分布式 ACM LU-SGS；
- `BDF2DualTimeGMRES`：内迭代使用左预条件 GMRES，预条件器仍由
  `gmresPreconditioner` 选择。

## 2. 控制方程

ACM 状态和物理状态分别定义为

```text
U = [u, v, w, p]^T
Q = M U,   M = diag(1, 1, 1, 0)
```

双时间方程写成

```text
Gamma(U) * dU/dtau + dQ/dt = R(U, t)
```

其中 `tau` 是用于收敛每个物理步的伪时间，`t` 是真实物理时间，`R` 是现有空间离散
残差。矩阵 `M` 的最后一个对角元为零，因此速度采用 BDF 物理时间离散，压力没有物理
时间导数。

对新物理时刻的当前内迭代值 `U*`，非线性缺陷为

```text
D(U*) = R(U*, t[n+1])
        - M * (a0*U* + a1*U[n] + a2*U[n-1]) / dt
```

第一物理步使用后向欧拉启动：

```text
(a0, a1, a2) = (1, -1, 0)
```

从第二物理步开始使用等时间步 BDF2：

```text
(a0, a1, a2) = (3/2, -2, 1/2)
```

## 3. 隐式内迭代

每次伪时间修正近似求解

```text
[Gamma(U*)/dtau + a0*M/dt - dR/dU] * deltaU = D(U*)
U* <- U* + implicitRelaxation * deltaU
```

现有 `ACMEvaluator::AssembleImplicitLinearization` 提供
`Gamma/dtau - dR/dU` 的冻结重构、一阶面通量线性化；BDF2 模块只向三个速度分量的单元
对角块加入 `a0/dt`。同一矩阵既可交给 LU-SGS，也可交给 GMRES。边界条件、残差和
CFL 估计均接收目标物理时刻 `t[n+1]`。

`U[n]` 和 `U[n-1]` 在整个内迭代期间保持不变。只有一个物理步结束后才移动历史，避免
伪时间修正污染已经完成的物理时间层。

## 4. 配置语义

BDF2 通过 `timeMarchSettings` 选择，当前算例 JSON 文件未被本次实现修改。以后需要启用
时可设置：

```json
{
  "integrator": "BDF2DualTimeLUSGS",
  "nSteps": 4000,
  "physicalTimeStep": 0.01,
  "maxImplicitIterations": 20,
  "implicitTolerance": 1e-10
}
```

相关字段含义如下：

- `nSteps`：BDF2 模式下为物理时间步数；
- `physicalTimeStep`：固定物理时间步 `dt`；
- `pseudoTimeStep`：固定内迭代伪时间步 `dtau`；
- `useCFLTimeStep`、`cfl`、`maximumPseudoTimeStep`：只控制 `dtau`；
- `maxImplicitIterations`：每个物理步允许的最大内迭代次数；
- `implicitTolerance`：全局 RMS 物理缺陷收敛阈值；
- `implicitRelaxation`：每次隐式修正的松弛因子。

为了保持旧算例可读，加载器在内存中为缺少 `physicalTimeStep` 的旧 JSON 补入默认值
`0.01`。这不会回写或改动磁盘上的 JSON 文件。动态 `--emit-schema` 输出会列出新字段和
两种 BDF2 枚举值。

## 5. 输出和日志

BDF2 模式的日志使用 `ACM physical step`，并显示物理时间、当前 BDF 阶数、初末缺陷、
最小伪时间步和内迭代次数。第一个物理步显示 `BDF1`，第二步起显示 `BDF2`。流场输出
文件仍按外层步编号命名，但 VTK 时间序列值使用真实物理时间。

## 6. 当前限制

- 只支持固定 `physicalTimeStep`，尚未实现变步长 BDF2 系数；
- 当前只允许 `Laminar`，因为现有分离式湍流输运还没有两个物理时间层；
- 物理步在达到最大内迭代次数后仍继续推进，使用者应根据日志中的 `converged` 和缺陷
  判断内迭代是否充分收敛；
- 重启文件和 BDF2 两层历史尚未序列化，重启接续非定常计算前需先补齐该功能。

## 7. 代码位置

- `src/ACM/ACMBDF2.hpp/.cpp`：BDF 系数、物理质量矩阵、缺陷、对角项和历史管理；
- `src/ACM/ACMSolver.hxx`：物理步循环、目标时间传递、LU-SGS/GMRES 内迭代；
- `src/ACM/ACMTime.hpp`：JSON 可选的积分格式和 `physicalTimeStep`；
- `src/ACM/ACM.cpp`：旧 JSON 默认值兼容和 Laminar 限制检查；
- `test/cpp/ACM/test_ACMTime.cpp`：启动阶数、压力屏蔽、缺陷/矩阵一致性和枚举测试。
