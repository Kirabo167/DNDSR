/**
 * @file ACMBDF2.cpp
 * @brief Implementation of physical-time BDF2 kernels for ACM dual-time marching.
 * @author Runzhi Ma
 * @date 2026-09-03
 * @note Modifier: Runzhi Ma.
 */
#include "ACMBDF2.hpp"

#include "DNDS/Errors.hpp"
#include "DNDS/OMP.hpp"

#include <cmath>

namespace DNDS::ACMVariable
{
    namespace
    {
        void ValidateCoefficients(const BDF2Coefficients &coefficients)
        {
            DNDS_check_throw_info(
                std::isfinite(coefficients.current) &&
                    std::isfinite(coefficients.previous) &&
                    std::isfinite(coefficients.previousPrevious),
                "ACM BDF coefficients must be finite");
            DNDS_check_throw_info(
                coefficients.order == 1 || coefficients.order == 2,
                "ACM BDF order must be one or two");
        }

        void ValidatePhysicalTimeStep(real physicalTimeStep)
        {
            DNDS_check_throw_info(
                std::isfinite(physicalTimeStep) && physicalTimeStep > 0,
                "ACM BDF2 physical time step must be finite and positive");
        }

        void ValidateStateField(const StateField &field, const char *message)
        {
            DNDS_check_throw_info(!field.empty(), message);
            for (const State &state : field)
                DNDS_check_throw_info(state.allFinite(), message);
        }
    }

    /** @copydoc GetBDF2Coefficients */
    BDF2Coefficients GetBDF2Coefficients(std::size_t completedPhysicalSteps)
    {
        if (completedPhysicalSteps == 0)
            return {1, -1, 0, 1};
        return {1.5, -2, 0.5, 2};
    }

    /** @copydoc PhysicalTimeMassMatrix */
    Matrix5 PhysicalTimeMassMatrix()
    {
        Matrix5 mass = Matrix5::Identity();
        mass(4, 4) = 0;
        return mass;
    }

    /** @copydoc EvaluateBDF2PhysicalDerivative */
    State EvaluateBDF2PhysicalDerivative(
        const State &current,
        const State &previous,
        const State &previousPrevious,
        const BDF2Coefficients &coefficients,
        real physicalTimeStep)
    {
        DNDS_check_throw_info(
            current.allFinite() && previous.allFinite() && previousPrevious.allFinite(),
            "ACM BDF2 history contains a non-finite state");
        ValidateCoefficients(coefficients);
        ValidatePhysicalTimeStep(physicalTimeStep);
        return PhysicalTimeMassMatrix() *
               (coefficients.current * current +
                coefficients.previous * previous +
                coefficients.previousPrevious * previousPrevious) /
               physicalTimeStep;
    }

    /** @copydoc FormBDF2PhysicalDefect */
    void FormBDF2PhysicalDefect(
        const StateField &current,
        const StateField &previous,
        const StateField &previousPrevious,
        const StateField &spatialResidual,
        const BDF2Coefficients &coefficients,
        real physicalTimeStep,
        StateField &defect)
    {
        DNDS_check_throw_info(
            !current.empty() && current.size() == previous.size() &&
                current.size() == previousPrevious.size() &&
                current.size() == spatialResidual.size(),
            "ACM BDF2 fields have incompatible sizes");
        ValidateCoefficients(coefficients);
        ValidatePhysicalTimeStep(physicalTimeStep);
        defect.resize(current.size());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index i = 0; i < static_cast<index>(current.size()); i++)
        {
            const std::size_t ii = static_cast<std::size_t>(i);
            DNDS_check_throw_info(
                current[ii].allFinite() && previous[ii].allFinite() &&
                    previousPrevious[ii].allFinite() && spatialResidual[ii].allFinite(),
                "ACM BDF2 field contains a non-finite value");
            defect[ii] = spatialResidual[ii] - EvaluateBDF2PhysicalDerivative(
                                                   current[ii],
                                                   previous[ii],
                                                   previousPrevious[ii],
                                                   coefficients,
                                                   physicalTimeStep);
        }
    }

    /** @copydoc AddBDF2PhysicalDiagonal */
    void AddBDF2PhysicalDiagonal(
        MatrixField &diagonal,
        const BDF2Coefficients &coefficients,
        real physicalTimeStep)
    {
        DNDS_check_throw_info(!diagonal.empty(), "ACM BDF2 diagonal field is empty");
        ValidateCoefficients(coefficients);
        ValidatePhysicalTimeStep(physicalTimeStep);
        const Matrix5 physicalBlock =
            coefficients.current * PhysicalTimeMassMatrix() / physicalTimeStep;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index i = 0; i < static_cast<index>(diagonal.size()); i++)
        {
            Matrix5 &block = diagonal[static_cast<std::size_t>(i)];
            DNDS_check_throw_info(block.allFinite(), "ACM BDF2 diagonal contains a non-finite block");
            block += physicalBlock;
        }
    }

    /** @copydoc IsBDF2DualTimeIntegrator */
    bool IsBDF2DualTimeIntegrator(TimeIntegratorType integrator)
    {
        return integrator == TimeIntegratorType::BDF2DualTimeLUSGS ||
               integrator == TimeIntegratorType::BDF2DualTimeGMRES;
    }

    /** @copydoc BDF2UsesLUSGS */
    bool BDF2UsesLUSGS(TimeIntegratorType integrator)
    {
        DNDS_check_throw_info(
            IsBDF2DualTimeIntegrator(integrator),
            "ACM BDF2 linear-solver query requires a BDF2 integrator");
        return integrator == TimeIntegratorType::BDF2DualTimeLUSGS;
    }

    /** @copydoc BDF2History::Initialize */
    void BDF2History::Initialize(const StateField &initialState)
    {
        ValidateStateField(initialState, "ACM BDF2 initial history is empty or non-finite");
        _previous = initialState;
        _previousPrevious = initialState;
        _completedPhysicalSteps = 0;
        _initialized = true;
    }

    /** @copydoc BDF2History::Coefficients */
    BDF2Coefficients BDF2History::Coefficients() const
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 history is not initialized");
        return GetBDF2Coefficients(_completedPhysicalSteps);
    }

    /** @copydoc BDF2History::Previous */
    const StateField &BDF2History::Previous() const
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 history is not initialized");
        return _previous;
    }

    /** @copydoc BDF2History::PreviousPrevious */
    const StateField &BDF2History::PreviousPrevious() const
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 history is not initialized");
        return _previousPrevious;
    }

    /** @copydoc BDF2History::Commit */
    void BDF2History::Commit(const StateField &completedState)
    {
        DNDS_check_throw_info(_initialized, "ACM BDF2 history is not initialized");
        DNDS_check_throw_info(
            completedState.size() == _previous.size(),
            "ACM BDF2 completed state has an incompatible size");
        ValidateStateField(completedState, "ACM BDF2 completed state is empty or non-finite");
        _previousPrevious = _previous;
        _previous = completedState;
        _completedPhysicalSteps++;
    }
}
