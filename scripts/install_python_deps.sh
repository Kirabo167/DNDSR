#!/usr/bin/env bash
# install_python_deps.sh — Install the core DNDSR contributor environment.
#
# This script must be run AFTER building cfd_externals (HDF5, zlib, etc.).
# The Cantera build deps (scons, packaging, ruamel.yaml, patchelf) should
# already be installed from external/cfd_externals/requirements.txt before
# running cfd_externals_build.py.
#
# This is the supported entry point for the core Python environment. One-off
# documentation/debugging utilities may declare additional local dependencies.
# It installs:
#   1. Binary-wheel packages from requirements.txt  (numpy, scipy, pytest, …)
#   2. mpi4py compiled from source against the selected MPI
#   3. h5py   compiled from source against the project's HDF5 (cfd_externals)
#   4. An import/ABI smoke test for NumPy, mpi4py, and parallel h5py
#
# h5py and mpi4py MUST be compiled from source for the supported developer
# environment. Ordinary h5py wheels use a serial/bundled HDF5, while an
# arbitrary mpi4py wheel may not match the selected MPI implementation or
# features. Both must match the libraries used by the DNDSR native modules.
# --no-build-isolation avoids pip creating temporary venvs that would
# reinstall build deps (including mpi4py when building h5py).
#
# Usage:
#   ./scripts/install_python_deps.sh          # uses venv/bin/pip, auto-detect -j
#   JOBS=8 ./scripts/install_python_deps.sh   # limit parallel compilation
#   PIP=path/to/pip ./scripts/install_python_deps.sh
#
# Environment variables:
#   PIP          — pip executable          (default: venv/bin/pip)
#   PYTHON       — Python executable       (default: next to PIP)
#   JOBS         — parallel make jobs      (default: $(nproc))
#   HDF5_DIR     — HDF5 install prefix     (default: external/cfd_externals/install)
#   MPI_CC       — MPI C compiler wrapper  (default: mpicc)
#   MPI4PY_SPEC  — mpi4py requirement      (default: mpi4py>=4,<5)
#   H5PY_SPEC    — h5py requirement        (default: h5py>=3.16,<4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PIP="${PIP:-$PROJECT_ROOT/venv/bin/pip}"
PYTHON="${PYTHON:-$(dirname "$PIP")/python}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
HDF5_DIR="${HDF5_DIR:-$PROJECT_ROOT/external/cfd_externals/install}"
MPI_CC="${MPI_CC:-mpicc}"
MPI4PY_SPEC="${MPI4PY_SPEC:-mpi4py>=4,<5}"
H5PY_SPEC="${H5PY_SPEC:-h5py>=3.16,<4}"

if [ ! -x "$PIP" ]; then
    echo "Error: pip not found at $PIP" >&2
    echo "Create a venv first:  python3 -m venv venv && venv/bin/pip install --upgrade pip" >&2
    exit 1
fi

if [ ! -x "$PYTHON" ]; then
    echo "Error: Python not found at $PYTHON" >&2
    exit 1
fi

if [ "$("$PIP" --version)" != "$("$PYTHON" -m pip --version)" ]; then
    echo "Error: PIP and PYTHON refer to different environments" >&2
    echo "  PIP:    $("$PIP" --version)" >&2
    echo "  PYTHON: $("$PYTHON" -m pip --version)" >&2
    exit 1
fi

if ! command -v "$MPI_CC" >/dev/null 2>&1; then
    echo "Error: MPI compiler wrapper not found: $MPI_CC" >&2
    exit 1
fi

if [ ! -d "$HDF5_DIR/include" ]; then
    echo "Error: HDF5_DIR=$HDF5_DIR does not contain include/" >&2
    echo "Build external dependencies first:  cd external/cfd_externals && python cfd_externals_build.py" >&2
    exit 1
fi

echo "=== install_python_deps.sh ==="
echo "  PIP       = $PIP"
echo "  PYTHON    = $PYTHON"
echo "  JOBS      = $JOBS"
echo "  HDF5_DIR  = $HDF5_DIR"
echo "  MPI_CC    = $MPI_CC"
echo "  MPI4PY    = $MPI4PY_SPEC"
echo "  H5PY      = $H5PY_SPEC"
echo ""

# ---- 1. Binary-wheel packages from requirements.txt ----------------------
REQ_FILE="$PROJECT_ROOT/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "--- Installing packages from requirements.txt ---"
    "$PIP" install -r "$REQ_FILE"
else
    echo "Warning: $REQ_FILE not found, skipping" >&2
fi

# ---- 2. mpi4py — compiled against the selected MPI -----------------------
echo ""
echo "--- Installing mpi4py (from source, CC=$MPI_CC, -j$JOBS) ---"
CC="$MPI_CC" \
    MPICC="$MPI_CC" \
    MPI4PY_BUILD_MPICC="$MPI_CC" \
    MAKEFLAGS="-j$JOBS" \
    CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" \
    "$PIP" install --no-binary mpi4py --no-build-isolation --no-deps \
    --no-cache-dir \
    "$MPI4PY_SPEC" --force-reinstall \
    --verbose

# ---- 3. h5py — compiled against the project's HDF5 (MPI-enabled) ---------
echo ""
echo "--- Installing h5py (from source, HDF5_DIR=$HDF5_DIR, HDF5_MPI=ON, -j$JOBS) ---"
CC="$MPI_CC" \
    HDF5_DIR="$HDF5_DIR" \
    HDF5_MPI="ON" \
    MAKEFLAGS="-j$JOBS" \
    CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" \
    "$PIP" install --no-binary h5py --no-build-isolation --no-deps \
    --no-cache-dir \
    "$H5PY_SPEC" --force-reinstall \
    --verbose

echo ""
echo "--- Verifying Python MPI/HDF5 ABI features ---"
"$PYTHON" -c 'import h5py, mpi4py.MPI as MPI, numpy as np; assert h5py.get_config().mpi, "h5py is not MPI-enabled"; print(f"numpy={np.__version__} mpi={MPI.Get_library_version().splitlines()[0]} h5py={h5py.__version__} hdf5={h5py.version.hdf5_version}")'
"$PYTHON" -m pip check

echo ""
echo "=== Done ==="
"$PIP" list | grep -iE "mpi4py|h5py|numpy|scipy|pytest"
