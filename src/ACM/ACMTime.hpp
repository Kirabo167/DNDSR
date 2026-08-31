/**
 * @file ACMTime.hpp
 * @brief Explicit and implicit pseudo-time integration interfaces for constant-density ACM states.
 *
 * @details The time integrators operate on rank-local state fields and obtain spatial residuals
 * through callbacks. A callback may perform mesh reconstruction and MPI ghost communication, so
 * the algorithms can later be connected to ACMEvaluator without changing their update formulas.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMFlux.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/MPI.hpp"

#include <functional>
#include <vector>

namespace DNDS::ACM
{
    /// Pseudo-time algorithms implemented by the initial ACM time-marching module.
    enum class TimeIntegratorType
    {
        ExplicitSSPRK3,
        ImplicitEulerBlockJacobi,
    };

    DNDS_DEFINE_ENUM_JSON(
        TimeIntegratorType,
        {
            {TimeIntegratorType::ExplicitSSPRK3, "ExplicitSSPRK3"},
            {TimeIntegratorType::ImplicitEulerBlockJacobi, "ImplicitEulerBlockJacobi"},
        })

    /// Runtime controls shared by explicit SSPRK3 and implicit backward-Euler stepping.
    struct TimeMarchSettings
    {
        TimeIntegratorType integrator = TimeIntegratorType::ExplicitSSPRK3; ///< Selected algorithm.
        int nSteps = 0;                                                     ///< Number of preview steps.
        real pseudoTimeStep = 0.01;                                         ///< Positive fixed pseudo-time step.
        int maxImplicitIterations = 20;                                     ///< Block-Jacobi iterations per implicit step.
        real implicitTolerance = 1e-10;                                     ///< Global RMS backward-Euler defect tolerance.
        real implicitRelaxation = 1.0;                                      ///< Damping applied to every implicit correction.

        DNDS_DECLARE_CONFIG(TimeMarchSettings)
        {
            DNDS_FIELD(
                integrator,
                "ACM pseudo-time integrator",
                DNDS::Config::enum_values(DNDS_ENUM_ALLOWED_VALUES(TimeIntegratorType)));
            DNDS_FIELD(nSteps, "Number of pseudo-time preview steps", DNDS::Config::range(0));
            DNDS_FIELD(pseudoTimeStep, "Fixed pseudo-time step", DNDS::Config::range(0.0));
            DNDS_FIELD(maxImplicitIterations, "Maximum implicit block-Jacobi iterations", DNDS::Config::range(1));
            DNDS_FIELD(implicitTolerance, "Implicit global RMS defect tolerance", DNDS::Config::range(0.0));
            DNDS_FIELD(implicitRelaxation, "Implicit correction relaxation", DNDS::Config::range(0.0, 1.0));
            config.post_read([](T &settings)
                             { settings.Validate(); });
        }

        /**
         * @brief Validate pseudo-time integration controls.
         * @throws std::runtime_error If a count, time step, tolerance, or relaxation is invalid.
         */
        void Validate() const;
    };

    using StateField = std::vector<State>;    ///< Rank-local owned ACM states.
    using MatrixField = std::vector<Matrix4>; ///< Rank-local block-diagonal matrices.
    using ScalarField = std::vector<real>;    ///< One scalar value per rank-local state.

    /**
     * @brief Callback that evaluates the raw spatial residual `R(U)` for all local states.
     * @param states Current rank-local states.
     * @param residual Output field resized or filled to `states.size()`.
     */
    using ResidualEvaluator = std::function<void(const StateField &states, StateField &residual)>;

    /**
     * @brief Callback that evaluates the block diagonal `dR_i/dU_i` used by the implicit solver.
     * @param states Current rank-local states.
     * @param diagonalJacobian Output matrix field resized or filled to `states.size()`.
     */
    using DiagonalJacobianEvaluator =
        std::function<void(const StateField &states, MatrixField &diagonalJacobian)>;

    /// Diagnostics returned after one explicit or implicit pseudo-time step.
    struct TimeStepReport
    {
        int iterations = 0;         ///< Explicit stages or implicit nonlinear iterations.
        real initialDefectNorm = 0; ///< Global RMS derivative/defect before updating.
        real finalDefectNorm = 0;   ///< Global RMS derivative/defect after updating.
        bool converged = false;     ///< True if the algorithm completed its acceptance criterion.
    };

    /**
     * @brief Apply the inverse ACM pseudo-time mass matrix to a raw spatial residual field.
     * @param states States defining the nonlinear preconditioning matrices.
     * @param residual Raw spatial residual field `R(U)`.
     * @param pseudoDerivative Output `Gamma(U)^{-1} R(U)` field.
     * @param settings ACM physical and preconditioning settings.
     */
    void ApplyGammaInverseToResidual(
        const StateField &states,
        const StateField &residual,
        StateField &pseudoDerivative,
        const Settings &settings);

    /**
     * @brief Advance one pseudo-time step with the three-stage third-order SSP Runge-Kutta method.
     * @param states Rank-local states updated in place.
     * @param pseudoTimeStep Positive local pseudo-time step for every state.
     * @param settings ACM physical and preconditioning settings.
     * @param residualEvaluator Callback evaluating the raw spatial residual.
     * @param mpi Optional communicator metadata used to report global RMS norms.
     * @return Stage count and global RMS derivative norms before and after the step.
     */
    TimeStepReport AdvanceExplicitSSPRK3(
        StateField &states,
        const ScalarField &pseudoTimeStep,
        const Settings &settings,
        const ResidualEvaluator &residualEvaluator,
        const MPIInfo *mpi = nullptr);

    /**
     * @brief Advance one backward-Euler pseudo-time step by nonlinear block-Jacobi iterations.
     * @param states Rank-local states updated in place.
     * @param pseudoTimeStep Positive local pseudo-time step for every state.
     * @param settings ACM physical and preconditioning settings.
     * @param timeSettings Implicit iteration count, tolerance, and damping controls.
     * @param residualEvaluator Callback evaluating the raw spatial residual `R(U)`.
     * @param diagonalJacobianEvaluator Callback evaluating block diagonals `dR_i/dU_i`.
     * @param mpi Optional communicator metadata used for global convergence norms.
     * @return Nonlinear iteration count, initial/final defect norms, and convergence status.
     */
    TimeStepReport AdvanceImplicitEulerBlockJacobi(
        StateField &states,
        const ScalarField &pseudoTimeStep,
        const Settings &settings,
        const TimeMarchSettings &timeSettings,
        const ResidualEvaluator &residualEvaluator,
        const DiagonalJacobianEvaluator &diagonalJacobianEvaluator,
        const MPIInfo *mpi = nullptr);

    /**
     * @brief Dispatch one pseudo-time step to the configured explicit or implicit algorithm.
     * @param states Rank-local states updated in place.
     * @param pseudoTimeStep Positive local pseudo-time steps.
     * @param settings ACM physical and preconditioning settings.
     * @param timeSettings Time-integration selection and nonlinear controls.
     * @param residualEvaluator Raw spatial-residual callback.
     * @param diagonalJacobianEvaluator Implicit block-diagonal callback; ignored by SSPRK3.
     * @param mpi Optional communicator metadata for global diagnostics.
     * @return Report produced by the selected algorithm.
     */
    TimeStepReport AdvancePseudoTimeStep(
        StateField &states,
        const ScalarField &pseudoTimeStep,
        const Settings &settings,
        const TimeMarchSettings &timeSettings,
        const ResidualEvaluator &residualEvaluator,
        const DiagonalJacobianEvaluator &diagonalJacobianEvaluator = {},
        const MPIInfo *mpi = nullptr);
}
