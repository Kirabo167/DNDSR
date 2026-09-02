<!--
File description: implementation status and integration boundary of the initial ACM module.
Modifier: Runzhi Ma
Last modified: 2026-09-02
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
- laminar viscous flux kernel with a linearly exact, center-connection face-gradient correction;
- all Euler-named boundary families, implemented with ACM `[u,v,w,p]` semantics;
- per-CGNS-zone boundary configuration and reserved Euler zone-name mapping;
- OpenMP face-buffer evaluation and MPI checksum reduction;
- explicit three-stage SSPRK3 pseudo-time integration with `Gamma^{-1} R`;
- implicit backward-Euler pseudo-time integration with nonlinear 4x4 block-Jacobi corrections;
- CGNS mesh reading, METIS partitioning, ghost construction, and periodic translation reuse;
- direct second-order Green-Gauss reconstruction;
- arbitrary-order CFV variational reconstruction selected by `vfvSettings.maxOrder`;
- selectable local-extrema, WBAP, and CWBAP limiting;
- ACM-specific general-`alpha` 4x4 characteristic transforms for WBAP/CWBAP, dimension-aware
  two-/three-dimensional polynomial norms, and a dedicated 3-D four-variable CFV instantiation;
- face-quadrature evaluation of reconstructed left/right states and viscous gradients;
- frozen-reconstruction face-flux diagonal Jacobians for implicit block-Jacobi updates;
- ACM 4x4 first-order face Jacobians including viscous jump terms and boundary chain rules;
- parallel block LU-SGS forward/backward sweeps with lagged off-rank ghost coupling;
- direct reuse of generic left-preconditioned GMRES with block-Jacobi or ACM LU-SGS preconditioning;
- Euler-style local/global CFL pseudo-time steps with convective and viscous spectral radii;
- runtime-selectable laminar, SA, Wilcox k-omega, SST k-omega, and Realizable k-epsilon modes;
- segregated limited second-order turbulence transport with wall distance, MPI ghost exchange,
  positivity-bounded SSPRK3 substeps, and frozen eddy-viscosity coupling to the flow equations;
- DNDS configuration registration, self-contained single-case JSON loading, CLI overrides, and schema output.

Application organization follows the compact Euler entry-point style:

- `app/ACM/ACM.cpp`: short default 3-D launcher;
- `app/ACM/acm2D.cpp` and `app/ACM/acm3D.cpp`: short dimension-specific launchers;
- `SingleBlockApp.hpp`: shared CLI and configuration workflow;
- `ACMSolver.*`: mesh/reconstruction/time-loop assembly;
- `ACMEvaluator.*`: high-order spatial residual and frozen-reconstruction Jacobian;
- `ACMTurbulence.*`: model-local viscosity, diffusion, source, and boundary kernels;
- `ACMTurbulenceTransport.*`: dimension-generic finite-volume transport and flow coupling;
- `acm2D.cpp` and `acm3D.cpp`: explicit template instantiations for four ACM variables.

Case configuration uses one file per case. `cases/acm2D/acm2D.json` and
`cases/acm3D/acm3D.json` each contain the complete physical, numerical, mesh, reconstruction,
boundary, and initial-state configuration. The application does not search for or merge an
adjacent base file. A user-supplied positional JSON path completely selects the case; `-k/-v`
overrides remain available for short parameter studies.

Turbulence is modular and does not enlarge the ACM flow state. The flow solver always stores
`[u,v,w,p]`; a separate distributed two-entry field stores only the active turbulence variables:

- `Laminar`: no turbulence equation and `mu_t=0`;
- `SpalartAllmaras`: `[nuTilde, unused]`;
- `KOmegaWilcox` and `KOmegaSST`: `[k, omega]`;
- `RealizableKEpsilon`: `[k, epsilon]`.

The same implementation is instantiated for both two and three dimensions. It reuses Geom wall
distance/mesh metrics, CFV face quadrature, and the existing MPI arrays, but it neither includes,
links, nor modifies the Euler module. At each flow residual evaluation, the transport module freezes
face `mu_t`; the flow viscous stress, viscous CFL radius, and frozen 4x4 implicit operator then use
`mu+mu_t`. After each completed flow pseudo-time step, the turbulence field advances segregatedly
with positivity-bounded SSPRK3 substeps. Therefore LU-SGS/GMRES remains the four-variable ACM flow
solve; turbulence source Jacobians are not inserted into that implicit system.

A non-laminar model requires `acmSettings.enableViscousFlux=true` and positive
`acmSettings.dynamicViscosity`. A typical three-dimensional SST selection is:

```json
"turbulenceSettings": {
  "model": "KOmegaSST",
  "initialValue": [0.001, 10.0],
  "farFieldValue": [0.001, 10.0],
  "minimumValue": [1e-12, 1e-10],
  "maximumValue": [1000000.0, 1000000000000.0],
  "maximumEddyViscosityRatio": 100000.0,
  "wallOmegaCoefficient": 800.0,
  "enableSourceTerms": true,
  "secondOrderReconstruction": true,
  "transportSubsteps": 4,
  "transportTimeScale": 0.25,
  "wallDistanceMethod": 1,
  "wallDistanceExecution": 0,
  "wallDistanceSubdivide": 0,
  "minimumWallDistance": 1e-10,
  "wallDistanceVerbose": 0
}
```

`initialValue` and `farFieldValue` use the model-specific layouts above. `maximumValue` bounds the
explicit stages independently of `maximumEddyViscosityRatio`, which caps only `mu_t/mu`. Increasing
`transportSubsteps` or decreasing `transportTimeScale` is the first stability adjustment for stiff
wall-omega cases. For high-order RANS calculations, enabling the main ACM reconstruction limiter is
recommended as well. Far-field turbulence values are imposed only where `BCFar` is locally inflow;
outflow uses zero normal gradient. Wall states impose zero `nuTilde`/`k`, with
`omega=wallOmegaCoefficient*nu/d^2` for the two k-omega models and
`epsilon=2*nu*k/d^2` for Realizable k-epsilon.

Boundary-model notes:

- `BCFar` uses incoming characteristics of the general-`alpha` ACM preconditioned system;
- at the isolated condition `alpha*q_n^2=beta2/rho0`, the normal operator can be defective; Roe
  evaluates the entropy-fixed matrix absolute value by a confluent-Hermite polynomial and retains
  the exact Jordan derivative term; `BCFar` uses a finite spectral cluster, while WBAP/CWBAP falls
  back to component space because no complete characteristic basis exists there;
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
- turbulence transport is segregated and explicit even when the four-variable flow integrator is
  implicit; no coupled turbulence Jacobian or implicit turbulence source linearization is present;
- the supplied SA model is baseline RANS; DES, transition, and rotation/curvature corrections are
  not enabled;
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
