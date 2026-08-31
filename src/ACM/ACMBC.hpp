/**
 * @file ACMBC.hpp
 * @brief Ghost-state construction interface for constant-density ACM boundary conditions.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMState.hpp"

namespace DNDS::ACM
{
    /**
     * @brief Construct a ghost state that imposes the requested boundary condition at the face.
     * @param type Boundary-condition family to apply.
     * @param interiorState Reconstructed state on the interior side of the boundary face.
     * @param boundaryValue Prescribed `[u,v,w,p]`; individual entries are used according to `type`.
     * @param unitNormal Outward face-normal direction, normalized internally.
     * @return Ghost state paired with `interiorState` by the numerical flux.
     * @throws std::runtime_error If an input is invalid or `type` is unsupported.
     */
    State GenerateBoundaryState(
        BoundaryType type,
        const State &interiorState,
        const State &boundaryValue,
        const Vector3 &unitNormal);
}
