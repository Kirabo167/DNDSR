# C++ API Reference

This section is auto-generated from the Doxygen XML output via
[Breathe](https://breathe.readthedocs.io/). Dedicated pages currently cover
DNDS, Geom, CFV, Euler, EulerP, and Solver, mirroring their Doxygen
class/namespace hierarchy. ACM, ACMVariable, and NCFV remain available through
the full Doxygen API rather than separate Breathe pages.

```{note}
The C++ API pages require a prior Doxygen XML build. When using CMake,
the `sphinx` target automatically runs `doxygen` first.

For the full Doxygen HTML documentation (with interactive call graphs
and SVG class diagrams), run `cmake --build build -t serve-doxygen`.
```

```{toctree}
:maxdepth: 2
:caption: Modules

api_cpp_dnds
api_cpp_geom
api_cpp_cfv
api_cpp_euler
api_cpp_eulerp
api_cpp_solver
```
