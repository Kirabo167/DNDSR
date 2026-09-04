/**
 * @file ACMState.hpp
 * @brief State accessors and face-local coordinate transformations for the ACM module.
 *
 * @author Runzhi Ma
 * @date 2026-09-03
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACM.hpp"

namespace DNDS::ACMVariable
{
    /**
     * @brief Return a square-root-density Roe state satisfying Abar*dU=dF.
     * @param left Positive-density conservative left state.
     * @param right Positive-density conservative right state.
     * @return Conservative Roe mean, with geometric density and weighted velocity.
     * @author Runzhi Ma
     */
    State RoeAverage(const State &left, const State &right);

    /**
     * @brief Convert conservative gradient columns to primitive [rho,u,v,w,p].
     * @param state Positive-density conservative state at the gradient location.
     * @param gradient Spatial rows, conservative-variable columns.
     * @return Gradient using du=(dm-u*drho)/rho, including density chain rule.
     * @author Runzhi Ma
     */
    Eigen::Matrix<real, 3, 5> PrimitiveGradient(
        const State &state, const Eigen::Matrix<real, 3, 5> &gradient);

    /**
     * @brief Convert conservative data to primitive variables without pressure scaling.
     * @param state Conservative [rho,mx,my,mz,p].
     * @return Primitive [rho,u,v,w,p].
     * @author Runzhi Ma
     */
    State PrimitiveState(const State &state);
    /**
     * @brief Build a right-handed orthonormal face basis `[normal, tangent1, tangent2]`.
     * @param unitNormal Face normal direction; it is normalized internally and therefore need not
     * have unit length.
     * @return Rotation matrix whose columns are the local basis vectors in global coordinates.
     * @throws std::runtime_error If `unitNormal` is non-finite or has zero length.
     */
    Matrix3 BuildLocalBasis(const Vector3 &unitNormal);

    /**
     * @brief Extract the three velocity components from a five-component ACM state.
     * @param state Input state `[rho,mx,my,mz,p]`.
     * @return Velocity vector `[u,v,w]`.
     */
    Vector3 Velocity(const State &state);

    /**
     * @brief Read physical pressure from an ACM state.
     * @param state Input state `[rho,mx,my,mz,p]`.
     * @return Physical pressure stored in zero-based component 4.
     */
    real PhysicalPressure(const State &state);

    /**
     * @brief Replace the physical-pressure component of an ACM state.
     * @param state State modified in place.
     * @param pressure New physical pressure.
     */
    void SetPhysicalPressure(State &state, real pressure);

    /**
     * @brief Rotate global momenta into a face-local basis, preserving density and pressure.
     * @param state Global state `[rho,mx,my,mz,p]`.
     * @param localBasis Orthonormal basis returned by BuildLocalBasis().
     * @return Face-local state `[rho,m_n,m_t1,m_t2,p]`.
     */
    State ToLocalState(const State &state, const Matrix3 &localBasis);

    /**
     * @brief Rotate a face-local flux back to global coordinates while preserving its scalar entry.
     * @param localFlux Face-local flux `[F_rho,F_mn,F_mt1,F_mt2,F_div]`.
     * @param localBasis Orthonormal basis returned by BuildLocalBasis().
     * @return Flux represented in global coordinates.
     */
    State FromLocalFlux(const State &localFlux, const Matrix3 &localBasis);

    /**
     * @brief Rotate the momentum block without changing density or pressure.
     * @param state State whose entries 1 through 3 (zero-based) are modified.
     * @param rotation Three-dimensional rotation matrix applied to momentum.
     */
    void RotateStateInPlace(State &state, const Matrix3 &rotation);

    /**
     * @brief Check whether every state component is finite.
     * @param state State to inspect.
     * @return `true` when all five entries are finite and density is positive.
     */
    bool IsFiniteState(const State &state);
}
