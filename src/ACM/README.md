<!--
File description: implementation status and integration boundary of the initial ACM module.
Modifier: Runzhi Ma
Last modified: 2026-09-01
-->

# ACM high-order initial solver

This directory contains an independent constant-density ACM solver for two- and
three-dimensional unstructured meshes. It reuses `Geom`, `CFV`, and the existing
MPI/OpenMP array infrastructure, but it does not modify or inherit the Euler equation model.

Implemented now:

- state layout `[u,v,w,p]`;
- physical flux and its conservative Jacobian;
- Scheme-A preconditioning matrices and eigenvalues;
- general-Turkel-`alpha` Rusanov and Roe fluxes;
- laminar viscous flux kernel;
- all Euler-named boundary families, implemented with ACM `[u,v,w,p]` semantics;
- per-CGNS-zone boundary configuration and reserved Euler zone-name mapping;
- OpenMP face-buffer evaluation and MPI checksum reduction;
- explicit three-stage SSPRK3 pseudo-time integration with `Gamma^{-1} R`;
- implicit backward-Euler pseudo-time integration with nonlinear 4x4 block-Jacobi corrections;
- CGNS mesh reading, METIS partitioning, ghost construction, and periodic translation reuse;
- direct second-order Green-Gauss reconstruction;
- arbitrary-order CFV variational reconstruction selected by `vfvSettings.maxOrder`;
- selectable local-extrema, WBAP, and CWBAP limiting;
- ACM-specific general-`alpha` 4x4 characteristic transforms for WBAP/CWBAP and a dedicated
  three-dimensional four-variable CFV template instantiation;
- face-quadrature evaluation of reconstructed left/right states and viscous gradients;
- frozen-reconstruction face-flux diagonal Jacobians for implicit block-Jacobi updates;
- ACM 4x4 first-order face Jacobians including viscous jump terms and boundary chain rules;
- parallel block LU-SGS forward/backward sweeps with lagged off-rank ghost coupling;
- direct reuse of generic left-preconditioned GMRES with block-Jacobi or ACM LU-SGS preconditioning;
- Euler-style local/global CFL pseudo-time steps with convective and viscous spectral radii;
- DNDS configuration registration, self-contained single-case JSON loading, CLI overrides, and schema output.

Application organization follows the compact Euler entry-point style:

- `app/ACM/ACM.cpp`: short default 3-D launcher;
- `app/ACM/acm2D.cpp` and `app/ACM/acm3D.cpp`: short dimension-specific launchers;
- `SingleBlockApp.hpp`: shared CLI and configuration workflow;
- `ACMSolver.*`: mesh/reconstruction/time-loop assembly;
- `ACMEvaluator.*`: high-order spatial residual and frozen-reconstruction Jacobian;
- `acm2D.cpp` and `acm3D.cpp`: explicit template instantiations for four ACM variables.

Case configuration uses one file per case. `cases/acm2D/acm2D.json` and
`cases/acm3D/acm3D.json` each contain the complete physical, numerical, mesh, reconstruction,
boundary, and initial-state configuration. The application does not search for or merge an
adjacent base file. A user-supplied positional JSON path completely selects the case; `-k/-v`
overrides remain available for short parameter studies.

Boundary-model notes:

- `BCFar` uses incoming characteristics of the general-`alpha` ACM preconditioned system;
- at the isolated condition `alpha*q_n^2=beta2/rho0`, the normal operator can be defective; Roe
  and `BCFar` use a finite spectral-cluster formula, while WBAP/CWBAP falls back to component space;
- `BCWallIsothermal` is equivalent to `BCWall`, because constant-density ACM has no temperature
  or energy variable;
- `BCInPsTs` interprets the configured velocity as face data and extrapolates pressure, because
  total pressure/temperature cannot be reconstructed from `[u,v,w,p]`;
- `BCSpecial` currently supports `specialOption=0`, meaning a prescribed complete ACM state;
- legacy names (`FarField`, `NoSlipWall`, `SlipWall`, `PressureOutlet`, `VelocityInlet`,
  `Symmetry`) remain accepted by the JSON reader.

One named outlet can be configured without changing the mesh reader or solver driver:

```json
"defaultBoundaryType": "BCFar",
"boundaryConditions": [
  {
    "type": "BCOutP",
    "name": "OUTLET",
    "value": [0.0, 0.0, 0.0, 1.0],
    "frameOption": 0,
    "anchorOption": 0,
    "integrationOption": 0,
    "specialOption": 0,
    "rectifyOption": 0,
    "valueExtra": []
  }
]
```

The `name` must match the CGNS boundary-zone name. Set `useCFLTimeStep=true` in
`timeMarchSettings` to enable local spectral-radius stepping; `useLocalTimeStep=false` replaces
all local values with the MPI-global minimum.

Current initial-version limits:

- periodic translations are configurable, while rotational periodic setup is not exposed yet;
- WBAP/CWBAP requires `Variational` reconstruction;
- LU-SGS/GMRES uses a first-order frozen face linearization as the implicit operator while the
  nonlinear residual retains the selected high-order reconstruction;
- solution/restart/VTK output is not connected yet.

Relevant selections are:

```json
"acmSettings": {
  "rho0": 1.0,
  "beta2": 4.0,
  "alpha": 0.5,
  "riemannSolverType": "Roe"
},
"reconstructionSettings": {
  "type": "Variational",
  "enableLimiter": true,
  "limiterType": "CWBAP"
},
"timeMarchSettings": {
  "integrator": "ImplicitEulerGMRES",
  "lusgsSweeps": 2,
  "gmresSubspace": 10,
  "gmresRestarts": 3,
  "gmresRelativeTolerance": 1e-6,
  "gmresPreconditioner": "LUSGS"
}
```

The source-level comparison and implemented integration route for WBAP/CWBAP and LU-SGS/GMRES
is documented in `docs/dev/acm_euler_reuse_comparison.md`.
