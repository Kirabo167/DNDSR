/**
 * @file test_ACMParallel.cpp
 * @brief MPI unit tests for ACM face-buffer evaluation and global checksum reduction.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "ACM/ACMParallel.hpp"
#include "ACM/ACMTime.hpp"

#include <vector>

/**
 * @brief Initialize MPI around the doctest runner used by the parallel ACM test executable.
 * @param argc Number of command-line arguments.
 * @param argv Command-line arguments forwarded to doctest.
 * @return Doctest process result after all registered tests have run.
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}

using namespace DNDS;
using namespace DNDS::ACM;

/// @test Verify face-index independence and rank-count scaling of the MPI all-reduced checksum.
TEST_CASE("ACM face buffer and MPI reduction are rank-count invariant")
{
    MPIInfo mpi;
    mpi.setWorld();
    Settings settings;
    settings.riemannSolverType = RiemannSolverType::Roe;
    settings.rho0 = 1.0;
    settings.beta2 = 2.0;

    FaceInput face;
    face.left << 1.0, 0.2, -0.1, 0.4;
    face.right << 0.3, -0.2, 0.5, 0.9;
    face.unitNormal = Vector3(1.0, 2.0, -1.0).normalized();
    constexpr int nFaces = 32;
    std::vector<FaceInput> faces(nFaces, face);
    std::vector<FluxResult> faceFluxBuffer;
    EvaluateFaceFluxes(faces, faceFluxBuffer, settings);

    REQUIRE(faceFluxBuffer.size() == faces.size());
    const FluxResult reference = InviscidFlux(
        settings.riemannSolverType, face.left, face.right, face.unitNormal, settings);
    for (const auto &result : faceFluxBuffer)
        CHECK((result.flux - reference.flux).norm() < 1e-12);

    const State localSum = LocalFluxSum(faceFluxBuffer);
    const State globalSum = GlobalFluxSum(localSum, mpi);
    const State expected = reference.flux * static_cast<real>(nFaces * mpi.size);
    for (int i = 0; i < 4; i++)
        CHECK(globalSum(i) == doctest::Approx(expected(i)).epsilon(1e-11));
}

/// @test Exercise global time-step diagnostics and zero-residual preservation on every MPI rank.
TEST_CASE("ACM explicit and implicit time stepping preserve a distributed steady state")
{
    MPIInfo mpi;
    mpi.setWorld();
    Settings settings;
    StateField states(8);
    for (std::size_t i = 0; i < states.size(); i++)
        states[i] << 0.2 * static_cast<real>(mpi.rank + 1), -0.1, 0.3, 1.0;
    const StateField reference = states;
    const ScalarField pseudoTimeStep(states.size(), 0.05);
    const ResidualEvaluator zeroResidual = [](const StateField &input, StateField &residual)
    {
        residual.assign(input.size(), State::Zero());
    };
    const DiagonalJacobianEvaluator zeroJacobian = [](const StateField &input, MatrixField &jacobian)
    {
        jacobian.assign(input.size(), Matrix4::Zero());
    };

    const auto explicitReport = AdvanceExplicitSSPRK3(
        states,
        pseudoTimeStep,
        settings,
        zeroResidual,
        &mpi);
    CHECK(explicitReport.converged);
    CHECK(explicitReport.initialDefectNorm == doctest::Approx(0.0));

    TimeMarchSettings timeSettings;
    timeSettings.integrator = TimeIntegratorType::ImplicitEulerBlockJacobi;
    const auto implicitReport = AdvanceImplicitEulerBlockJacobi(
        states,
        pseudoTimeStep,
        settings,
        timeSettings,
        zeroResidual,
        zeroJacobian,
        &mpi);
    CHECK(implicitReport.converged);
    CHECK(implicitReport.iterations == 1);
    for (std::size_t i = 0; i < states.size(); i++)
        CHECK((states[i] - reference[i]).norm() < 1e-14);
}
