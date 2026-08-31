/**
 * @file test_ACMCore.cpp
 * @brief Serial unit tests for constant-density ACM state, flux, boundary, and configuration kernels.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "ACM/ACMBC.hpp"
#include "ACM/ACMConfig.hpp"
#include "ACM/ACMFlux.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace DNDS;
using namespace DNDS::ACM;

namespace
{
    /**
     * @brief Compare two four-component ACM vectors with doctest diagnostics.
     * @param actual Vector produced by the kernel under test.
     * @param expected Reference vector.
     * @param tolerance Relative epsilon used by `doctest::Approx` for each component.
     */
    void CheckVectorNear(const State &actual, const State &expected, real tolerance = 1e-11)
    {
        for (int i = 0; i < 4; i++)
        {
            CAPTURE(i);
            CAPTURE(actual(i));
            CAPTURE(expected(i));
            CHECK(actual(i) == doctest::Approx(expected(i)).epsilon(tolerance));
        }
    }
}

/// @test Verify orthonormal local-basis construction and lossless state rotation.
TEST_CASE("ACM local basis and state rotation preserve pressure")
{
    const Vector3 normal = Vector3(1, 2, -3).normalized();
    const Matrix3 basis = BuildLocalBasis(normal);
    CHECK((basis.transpose() * basis - Matrix3::Identity()).norm() < 1e-13);
    CHECK((basis.col(0) - normal).norm() < 1e-13);

    State state;
    state << 2.0, -1.0, 0.5, 7.0;
    const State local = ToLocalState(state, basis);
    const State roundTrip = FromLocalFlux(local, basis);
    CheckVectorNear(roundTrip, state);
    CHECK(local(3) == doctest::Approx(7.0));
}

/// @test Compare the analytic physical-flux Jacobian against centered finite differences.
TEST_CASE("ACM physical flux Jacobian matches finite differences")
{
    State state;
    state << 1.3, -0.4, 0.7, 2.2;
    constexpr real rho0 = 1.7;
    const Matrix4 analytic = PhysicalFluxJacobianLocal(state, rho0);
    Matrix4 finiteDifference;
    const real epsilon = 1e-7;
    for (int column = 0; column < 4; column++)
    {
        State perturbation = State::Zero();
        perturbation(column) = epsilon;
        finiteDifference.col(column) =
            (PhysicalFluxLocal(state + perturbation, rho0) -
             PhysicalFluxLocal(state - perturbation, rho0)) /
            (2 * epsilon);
    }

    CHECK((analytic - finiteDifference).norm() < 1e-8);
    CHECK(analytic(0, 0) == doctest::Approx(2 * state(0)));
    CHECK(analytic(1, 0) == doctest::Approx(state(1)));
    CHECK(analytic(2, 0) == doctest::Approx(state(2)));
    CHECK(analytic(3, 0) == doctest::Approx(1.0));
}

/// @test Check that Gamma inversion and the preconditioned characteristic speeds are consistent.
TEST_CASE("ACM Gamma and preconditioned eigensystem are consistent")
{
    State mean;
    mean << 0.8, -0.3, 0.2, 1.0;
    constexpr real rho0 = 1.25;
    constexpr real beta2 = 3.4;
    constexpr real alpha = 0.0;
    const Matrix4 gamma = GammaLocal(mean, beta2, alpha);
    const Matrix4 gammaInv = GammaInvLocal(mean, beta2, alpha);
    CHECK((gammaInv * gamma - Matrix4::Identity()).norm() < 1e-13);

    const Matrix4 preconditioned = PreconditionedJacobianLocal(mean, rho0, beta2, alpha);
    CHECK((preconditioned - gammaInv * PhysicalFluxJacobianLocal(mean, rho0)).norm() < 1e-13);

    Eigen::EigenSolver<Matrix4> solver(preconditioned, false);
    std::vector<real> numeric;
    for (int i = 0; i < 4; i++)
    {
        CHECK(std::abs(solver.eigenvalues()(i).imag()) < 1e-12);
        numeric.push_back(solver.eigenvalues()(i).real());
    }
    std::sort(numeric.begin(), numeric.end());
    const auto expectedValues = ComputeEigenvalues(mean(0), rho0, beta2, alpha);
    std::array<real, 4> expected{
        expectedValues.lambdaMinus,
        expectedValues.lambdaTangential,
        expectedValues.lambdaTangential,
        expectedValues.lambdaPlus};
    std::sort(expected.begin(), expected.end());
    for (int i = 0; i < 4; i++)
        CHECK(numeric[static_cast<std::size_t>(i)] == doctest::Approx(expected[static_cast<std::size_t>(i)]).epsilon(1e-11));
}

/// @test Check equal-state consistency and normal-reversal symmetry for both Riemann solvers.
TEST_CASE("ACM Riemann solvers are consistent and normal-symmetric")
{
    Settings settings;
    settings.rho0 = 1.3;
    settings.beta2 = 2.1;
    settings.entropyFixRatio = 0.03;

    State left;
    left << 0.8, -0.2, 0.4, 1.5;
    State right;
    right << -0.1, 0.5, -0.3, 0.7;
    const Vector3 normal = Vector3(0.3, -0.4, 0.5).normalized();
    const Matrix3 basis = BuildLocalBasis(normal);
    const State exact = FromLocalFlux(PhysicalFluxLocal(ToLocalState(left, basis), settings.rho0), basis);

    for (const auto type : {RiemannSolverType::Rusanov, RiemannSolverType::Roe})
    {
        const auto consistent = InviscidFlux(type, left, left, normal, settings);
        CheckVectorNear(consistent.flux, exact, 1e-10);

        const auto forward = InviscidFlux(type, left, right, normal, settings);
        const auto reverse = InviscidFlux(type, right, left, -normal, settings);
        CheckVectorNear(forward.flux, -reverse.flux, 1e-10);
        CHECK(forward.flux.allFinite());
        CHECK(forward.eigenvalues.SpectralRadius() > 0);
    }
}

/// @test Compare the specialized alpha-zero Roe dissipation against an explicit matrix reference.
TEST_CASE("ACM Roe dissipation matches matrix reference")
{
    Settings settings;
    settings.rho0 = 1.1;
    settings.beta2 = 2.7;
    settings.entropyFixRatio = 0.0;
    State left;
    left << 0.9, -0.4, 0.3, 1.2;
    State right;
    right << -0.2, 0.5, -0.1, 0.6;

    Eigenvalues eigenvalues;
    const State actual = RoeDissipationLocalAlpha0(left, right, settings, eigenvalues);
    const State mean = 0.5 * (left + right);
    Matrix4 rightEigenvectors = Matrix4::Zero();
    rightEigenvectors.col(0) << eigenvalues.lambdaMinus / settings.beta2, 0, 0, 1;
    rightEigenvectors(1, 1) = 1;
    rightEigenvectors(2, 2) = 1;
    rightEigenvectors.col(3) << eigenvalues.lambdaPlus / settings.beta2, 0, 0, 1;
    Matrix4 lambdaAbs = Matrix4::Zero();
    lambdaAbs.diagonal() << std::abs(eigenvalues.lambdaMinus),
        std::abs(eigenvalues.lambdaTangential),
        std::abs(eigenvalues.lambdaTangential),
        std::abs(eigenvalues.lambdaPlus);
    const State expected = GammaLocal(mean, settings.beta2, 0) *
                           rightEigenvectors * lambdaAbs * rightEigenvectors.inverse() *
                           (right - left);
    CheckVectorNear(actual, expected, 1e-10);
}

/// @test Check that ghost states impose no-slip, slip, and pressure-outlet face values.
TEST_CASE("ACM boundary ghost states impose face values")
{
    State interior;
    interior << 2.0, -1.0, 0.5, 3.0;
    State prescribed;
    prescribed << 0.2, 0.3, -0.4, 1.1;
    const Vector3 normal = Vector3::UnitX();

    const State noSlip = GenerateBoundaryState(BoundaryType::NoSlipWall, interior, prescribed, normal);
    CheckVectorNear(0.5 * (interior + noSlip).head<4>(),
                    (State() << prescribed(0), prescribed(1), prescribed(2), interior(3)).finished());

    const State slip = GenerateBoundaryState(BoundaryType::SlipWall, interior, State::Zero(), normal);
    CHECK((0.5 * (interior.head<3>() + slip.head<3>())).dot(normal) == doctest::Approx(0.0));
    CHECK(slip(1) == doctest::Approx(interior(1)));
    CHECK(slip(2) == doctest::Approx(interior(2)));

    const State outlet = GenerateBoundaryState(BoundaryType::PressureOutlet, interior, prescribed, normal);
    CHECK(0.5 * (interior(3) + outlet(3)) == doctest::Approx(prescribed(3)));
}

/// @test Verify round-trip serialization through the existing DNDS JSON configuration registry.
TEST_CASE("ACM settings use DNDS JSON registration")
{
    Settings settings;
    settings.rho0 = 2.0;
    settings.beta2 = 4.0;
    settings.riemannSolverType = RiemannSolverType::Rusanov;
    nlohmann::ordered_json json = settings;
    const Settings restored = json.get<Settings>();
    CHECK(restored.rho0 == doctest::Approx(2.0));
    CHECK(restored.beta2 == doctest::Approx(4.0));
    CHECK(restored.riemannSolverType == RiemannSolverType::Rusanov);
    CHECK(json["pressureStorage"] == "PhysicalP");
}
