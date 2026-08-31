<!--
File description: implementation status and integration boundary of the initial ACM module.
Modifier: Runzhi Ma
Last modified: 2026-08-31
-->

# ACM initial module

This directory contains the first constant-density three-dimensional ACM implementation.

Implemented now:

- state layout `[u,v,w,p]`;
- physical flux and its conservative Jacobian;
- Scheme-A preconditioning matrices and eigenvalues;
- Rusanov and `alpha=0` Roe fluxes;
- laminar viscous flux kernel;
- basic ghost-state boundary rules;
- OpenMP face-buffer evaluation and MPI checksum reduction;
- explicit three-stage SSPRK3 pseudo-time integration with `Gamma^{-1} R`;
- implicit backward-Euler pseudo-time integration with nonlinear 4x4 block-Jacobi corrections;
- DNDS configuration registration, default + merge-patch loading, CLI overrides, and schema output.

`acm3D` is currently a kernel/configuration preview driver. It advances a conservative rank-local periodic line so both time integrators can be exercised, but it does not yet read a production mesh. The next integration step is to connect the residual and diagonal-Jacobian callbacks to a new `ACMEvaluator` while preserving the existing Euler face-buffer/cell-gather parallel structure.
