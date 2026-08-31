/**
 * @file ACM.hpp
 * @brief Common model identifiers, enumerations, traits, and fixed-size types for the ACM module.
 *
 * @details This header defines the public vocabulary shared by the constant-density three-dimensional
 * artificial-compressibility kernels without introducing dependencies on the existing Euler model.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "DNDS/Config/ConfigEnum.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/EigenUtil.hpp"

namespace DNDS::ACM
{
    /// Available artificial-compressibility equation sets.
    enum class ACMModel
    {
        ConstantDensity3D,
    };

    /// Policy used to store the pressure-like component in an ACM state.
    enum class PressureStorage
    {
        PhysicalP,
        ScaledPOverBeta2,
    };

    DNDS_DEFINE_ENUM_JSON(
        PressureStorage,
        {
            {PressureStorage::PhysicalP, "PhysicalP"},
            {PressureStorage::ScaledPOverBeta2, "ScaledPOverBeta2"},
        })

    /// Numerical flux families supported by the initial ACM implementation.
    enum class RiemannSolverType
    {
        Rusanov,
        Roe,
    };

    DNDS_DEFINE_ENUM_JSON(
        RiemannSolverType,
        {
            {RiemannSolverType::Rusanov, "Rusanov"},
            {RiemannSolverType::Roe, "Roe"},
        })

    /// Boundary conditions supported by the ACM ghost-state generator.
    enum class BoundaryType
    {
        FarField,
        VelocityInlet,
        PressureOutlet,
        NoSlipWall,
        SlipWall,
        Symmetry,
    };

    DNDS_DEFINE_ENUM_JSON(
        BoundaryType,
        {
            {BoundaryType::FarField, "FarField"},
            {BoundaryType::VelocityInlet, "VelocityInlet"},
            {BoundaryType::PressureOutlet, "PressureOutlet"},
            {BoundaryType::NoSlipWall, "NoSlipWall"},
            {BoundaryType::SlipWall, "SlipWall"},
            {BoundaryType::Symmetry, "Symmetry"},
        })

    /**
     * @brief Compile-time properties of an ACM equation set.
     * @tparam model ACM equation-set identifier.
     */
    template <ACMModel model>
    struct ModelTraits;

    /// Compile-time layout of the constant-density three-dimensional state `[u, v, w, p]`.
    template <>
    struct ModelTraits<ACMModel::ConstantDensity3D>
    {
        static constexpr int dim = 3;
        static constexpr int gDim = 3;
        static constexpr int nVarsFixed = 4;
        static constexpr int velocityBegin = 0;
        static constexpr int pressureIndex = 3;
        static constexpr int nExtraVars = 0;
    };

    using State = Eigen::Vector<real, 4>;      ///< ACM state or flux vector `[u, v, w, p]`.
    using Vector3 = Eigen::Vector<real, 3>;    ///< Three-dimensional geometry or velocity vector.
    using Matrix3 = Eigen::Matrix<real, 3, 3>; ///< Three-dimensional rotation or tensor matrix.
    using Matrix4 = Eigen::Matrix<real, 4, 4>; ///< Four-variable ACM operator matrix.
}
