/**
 * @file test_ACMTime.cpp
 * @brief Unit tests for explicit SSPRK3 and implicit backward-Euler ACM pseudo-time stepping.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "ACM/ACMTime.hpp"

using namespace DNDS;
using namespace DNDS::ACM;

namespace
{
    /**
     * @brief Construct the raw residual `R=-rate*U` used by analytic time-integration tests.
     * @param rate Positive linear decay rate.
     * @return Residual callback operating independently on every rank-local state.
     */
    ResidualEvaluator LinearDecayResidual(real rate)
    {
        return [rate](const StateField &states, StateField &residual)
        {
            residual.resize(states.size());
            for (std::size_t i = 0; i < states.size(); i++)
                residual[i] = -rate * states[i];
        };
    }

    /**
     * @brief Construct exact diagonal blocks `dR/dU=-rate*I` for the linear decay residual.
     * @param rate Positive linear decay rate.
     * @return Block-diagonal Jacobian callback.
     */
    DiagonalJacobianEvaluator LinearDecayJacobian(real rate)
    {
        return [rate](const StateField &states, MatrixField &jacobian)
        {
            jacobian.assign(states.size(), -rate * Matrix4::Identity());
        };
    }
}

/// @test Verify inverse-Gamma application independently of a time-integration scheme.
TEST_CASE("ACM inverse Gamma converts raw residual to pseudo-time derivative")
{
    Settings settings;
    settings.beta2 = 2.5;
    StateField states(1);
    states[0] << 0.7, -0.2, 0.4, 1.1;
    StateField residual(1);
    residual[0] << 0.3, -0.8, 0.1, 0.6;
    StateField derivative;

    ApplyGammaInverseToResidual(states, residual, derivative, settings);

    REQUIRE(derivative.size() == 1);
    const State reconstructed = GammaLocal(states[0], settings.beta2, settings.alpha) * derivative[0];
    CHECK((reconstructed - residual[0]).norm() < 1e-13);
}

/// @test Check the exact SSPRK3 amplification polynomial for a linear decay equation.
TEST_CASE("ACM explicit SSPRK3 has the expected linear amplification")
{
    Settings settings;
    settings.beta2 = 1.0;
    StateField states(1, State::Zero());
    states[0](3) = 2.0;
    constexpr real rate = 1.7;
    constexpr real dt = 0.08;
    const ScalarField pseudoTimeStep(1, dt);

    const TimeStepReport report = AdvanceExplicitSSPRK3(
        states,
        pseudoTimeStep,
        settings,
        LinearDecayResidual(rate));

    const real z = rate * dt;
    const real amplification = 1 - z + 0.5 * z * z - z * z * z / 6;
    CHECK(states[0](3) == doctest::Approx(2.0 * amplification).epsilon(1e-13));
    CHECK(states[0].head<3>().norm() == doctest::Approx(0.0));
    CHECK(report.iterations == 3);
    CHECK(report.converged);
}

/// @test Check backward Euler against its exact result for a block-diagonal linear residual.
TEST_CASE("ACM implicit backward Euler block Jacobi matches linear reference")
{
    Settings settings;
    settings.beta2 = 1.0;
    TimeMarchSettings timeSettings;
    timeSettings.integrator = TimeIntegratorType::ImplicitEulerBlockJacobi;
    timeSettings.maxImplicitIterations = 4;
    timeSettings.implicitTolerance = 1e-13;
    StateField states(1, State::Zero());
    states[0](3) = 2.0;
    constexpr real rate = 1.7;
    constexpr real dt = 0.08;
    const ScalarField pseudoTimeStep(1, dt);

    const TimeStepReport report = AdvanceImplicitEulerBlockJacobi(
        states,
        pseudoTimeStep,
        settings,
        timeSettings,
        LinearDecayResidual(rate),
        LinearDecayJacobian(rate));

    CHECK(states[0](3) == doctest::Approx(2.0 / (1 + rate * dt)).epsilon(1e-13));
    CHECK(states[0].head<3>().norm() == doctest::Approx(0.0));
    CHECK(report.converged);
    CHECK(report.iterations <= 2);
    CHECK(report.finalDefectNorm <= timeSettings.implicitTolerance);
}

/// @test Verify the nonlinear Gamma mass matrix is present on both sides of backward Euler.
TEST_CASE("ACM implicit backward Euler converges with a nonlinear Gamma mass matrix")
{
    Settings settings;
    settings.beta2 = 2.5;
    TimeMarchSettings timeSettings;
    timeSettings.integrator = TimeIntegratorType::ImplicitEulerBlockJacobi;
    timeSettings.maxImplicitIterations = 20;
    timeSettings.implicitTolerance = 1e-12;
    StateField states(1);
    states[0] << 0.8, -0.4, 0.5, 1.2;
    const State initial = states[0];
    State target;
    target << 0.2, -0.1, 0.3, 0.4;
    constexpr real rate = 1.3;
    constexpr real dt = 0.06;
    const ScalarField pseudoTimeStep(1, dt);
    const ResidualEvaluator residualEvaluator = [&](const StateField &input, StateField &residual)
    {
        residual.resize(input.size());
        for (std::size_t i = 0; i < input.size(); i++)
            residual[i] = GammaLocal(input[i], settings.beta2, settings.alpha) *
                          (-rate * (input[i] - target));
    };
    const DiagonalJacobianEvaluator jacobianEvaluator = [&](const StateField &input, MatrixField &jacobian)
    {
        jacobian.resize(input.size());
        constexpr real epsilon = 1e-7;
        for (std::size_t i = 0; i < input.size(); i++)
        {
            for (int iVariable = 0; iVariable < 4; iVariable++)
            {
                State plus = input[i];
                State minus = input[i];
                plus(iVariable) += epsilon;
                minus(iVariable) -= epsilon;
                const State residualPlus = GammaLocal(plus, settings.beta2, settings.alpha) *
                                           (-rate * (plus - target));
                const State residualMinus = GammaLocal(minus, settings.beta2, settings.alpha) *
                                            (-rate * (minus - target));
                jacobian[i].col(iVariable) = (residualPlus - residualMinus) / (2 * epsilon);
            }
        }
    };

    const TimeStepReport report = AdvanceImplicitEulerBlockJacobi(
        states,
        pseudoTimeStep,
        settings,
        timeSettings,
        residualEvaluator,
        jacobianEvaluator);

    const State expected = (initial + rate * dt * target) / (1 + rate * dt);
    CHECK((states[0] - expected).norm() < 1e-11);
    CHECK(report.converged);
    CHECK(report.finalDefectNorm <= timeSettings.implicitTolerance);
}

/// @test Ensure invalid local time steps are rejected before a state can be modified.
TEST_CASE("ACM time integrators reject non-positive pseudo-time steps")
{
    Settings settings;
    StateField states(1, State::Zero());
    const ScalarField pseudoTimeStep(1, 0.0);
    CHECK_THROWS(AdvanceExplicitSSPRK3(
        states,
        pseudoTimeStep,
        settings,
        LinearDecayResidual(1.0)));
}
