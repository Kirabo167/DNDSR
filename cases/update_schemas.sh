#!/usr/bin/env bash
# Regenerate JSON Schema files for all solver variants.
#
# Usage:
#   cd <project_root>
#   bash cases/update_schemas.sh [build_dir]
#
# The build directory defaults to "build-reactive". All 9 solver executables
# must already be compiled with DNDS_USE_CANTERA=ON so the committed EulerEX
# schemas describe the complete supported feature set.

set -euo pipefail

BUILD_DIR="${1:-build-reactive}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

VARIANTS=(euler euler2D euler3D eulerSA eulerSA3D euler2EQ euler2EQ3D eulerEX eulerEX3D)

cache="${BUILD_DIR}/CMakeCache.txt"
if [[ ! -f "$cache" ]] || ! grep -q '^DNDS_USE_CANTERA:BOOL=ON$' "$cache"; then
    echo "Error: schema regeneration requires a configured DNDS_USE_CANTERA=ON build." >&2
    echo "Run: cmake --preset reactive-test && cmake --build --preset schemas -j8" >&2
    exit 1
fi

# Refuse a partial refresh: leaving old schemas in place while reporting
# success makes configuration compatibility failures very difficult to spot.
missing=()
for v in "${VARIANTS[@]}"; do
    exe="${BUILD_DIR}/app/${v}.exe"
    if [[ ! -x "$exe" ]]; then
        missing+=("$exe")
    fi
done
if (( ${#missing[@]} > 0 )); then
    echo "Error: all schema-producing solver executables must be built first:" >&2
    printf '  missing %s\n' "${missing[@]}" >&2
    exit 1
fi

# Generate and validate every schema before replacing any committed output.
# Staging beside the destination keeps the final rename on the same filesystem.
staging_dir="$(mktemp -d "${SCRIPT_DIR}/.schema-update.XXXXXX")"
cleanup()
{
    if [[ -d "$staging_dir" ]]; then
        rm -rf -- "$staging_dir"
    fi
}
trap cleanup EXIT

for v in "${VARIANTS[@]}"; do
    exe="${BUILD_DIR}/app/${v}.exe"
    staged="${staging_dir}/${v}_schema.json"
    stderr_log="${staging_dir}/${v}.stderr.log"
    if mpirun -np 1 "$exe" --emit-schema 2>"$stderr_log" \
        | grep -v '^JSON:' > "$staged"; then
        if ! python3 -m json.tool "$staged" >/dev/null; then
            echo "FAIL $v: generated output is not valid JSON" >&2
            exit 1
        fi
        echo "  generated $v"
    else
        if [[ -s "$stderr_log" ]]; then
            cat "$stderr_log" >&2
        fi
        echo "FAIL $v: mpirun pipeline failed" >&2
        exit 1
    fi
done

# Each rename is atomic, and this phase is reached only after all nine staged
# schemas have passed generation and JSON parsing.
for v in "${VARIANTS[@]}"; do
    out="${SCRIPT_DIR}/${v}_schema.json"
    mv -- "${staging_dir}/${v}_schema.json" "$out"
    echo "  $v -> $(basename "$out")"
done

echo "Done."
