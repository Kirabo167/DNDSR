/**
 * @file test_mpi.cpp
 * @brief MPI integration regression for 2-D/3-D variable-density ACM reconstruction.
 * @details Reads and partitions real CGNS meshes, runs CFV/CWBAP and evaluates a
 * stationary nonuniform-density contact. Its exact inviscid residual is zero.
 * @author Runzhi Ma
 * @date 2026-09-04
 */
#include "ACMVariable/ACMSolver.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>

using namespace DNDS;
using namespace DNDS::ACMVariable;

/**
 * @brief Verify that two variable-density LU-SGS sweeps equal one sweep plus a residual correction.
 * @tparam model Variable-density ACM dimensional specialization.
 * @param solver Initialized solver supplying the distributed implicit operator.
 */
template <ACMModel model>
static void VerifyLUSGSResidualCorrection(ACMSolver<model> &solver)
{
    using TSolver = ACMSolver<model>;
    using TDof = typename TSolver::TDof;
    const auto &vfv = solver.GetReconstruction();
    const auto &evaluator = solver.GetEvaluator();
    auto &state = solver.GetState();
    DNDS_check_throw_info(vfv != nullptr && evaluator != nullptr,
                          "Variable ACM LU-SGS verification requires an initialized solver");

    TDof rhs;
    TDof oneSweep;
    TDof twoSweeps;
    TDof firstResidual;
    TDof operatorProduct;
    TDof correction;
    TDof expected;
    TDof difference;
    for (TDof *field : {&rhs, &oneSweep, &twoSweeps, &firstResidual,
                        &operatorProduct, &correction, &expected, &difference})
        vfv->BuildUDof(*field, 5);

    for (DNDS::index iCell = 0; iCell < solver.GetMesh()->NumCell(); iCell++)
    {
        const auto barycenter = vfv->GetCellBary(iCell);
        rhs[iCell] << 0.8 + 0.1 * barycenter(0),
            -0.3 + 0.1 * barycenter(1),
            0.2 + 0.05 * barycenter(0),
            -0.1,
            0.4 - 0.1 * barycenter(1);
    }

    const ScalarField pseudoTimeStep(
        static_cast<std::size_t>(solver.GetMesh()->NumCell()), 0.01);
    MatrixField diagonal;
    typename TSolver::TEvaluator::FaceJacobianField faceJacobians;
    evaluator->AssembleImplicitLinearization(
        state, pseudoTimeStep, diagonal, faceJacobians, 0);

    evaluator->SolveLUSGS(rhs, diagonal, faceJacobians, oneSweep, 1);
    evaluator->ApplyImplicitLinearization(
        oneSweep, diagonal, faceJacobians, operatorProduct);
    firstResidual = rhs;
    firstResidual.addTo(operatorProduct, -1);
    evaluator->SolveLUSGS(
        firstResidual, diagonal, faceJacobians, correction, 1);
    expected = oneSweep;
    expected.addTo(correction, 1);

    evaluator->SolveLUSGS(rhs, diagonal, faceJacobians, twoSweeps, 2);
    difference = twoSweeps;
    difference.addTo(expected, -1);
    const real expectedNorm = expected.norm2();
    DNDS_check_throw_info(
        difference.norm2() < 1e-11 * std::max(real(1), expectedNorm),
        "Variable ACM LU-SGS second sweep is not a b-A*x residual correction");
    DNDS_check_throw_info(
        correction.norm2() > 1e-12 * std::max(real(1), oneSweep.norm2()),
        "Variable ACM LU-SGS residual-correction test is degenerate");
}

/**
 * @brief Execute one dimensional specialization of the MPI spatial pipeline.
 * @tparam model VariableDensity2D or VariableDensity3D.
 * @param mpi Initialized world communicator information.
 * @param meshFile Absolute CGNS verification mesh path.
 * @return Global maximum absolute residual of the stationary contact.
 * @author Runzhi Ma
 */
template <ACMModel model>
static real RunStationaryContact(const MPIInfo &mpi, const std::string &meshFile)
{
    KernelConfiguration configuration;
    configuration.meshSettings.meshFile = meshFile;
    configuration.initialState = {1, 0, 0, 0, 0};
    configuration.boundaryValue = {1, 0, 0, 0, 0};
    configuration.acmSettings.farFieldValue = {1, 0, 0, 0, 0};
    configuration.initialDensityAmplitude = 0.2;
    configuration.initialDensityWaveNumber = {6.283185307179586, 0, 0};
    configuration.acmSettings.entropyFixRatio = 0;
    configuration.reconstructionSettings.type = ReconstructionType::Variational;
    configuration.reconstructionSettings.limiterType = LimiterType::CWBAP;
    configuration.reconstructionSettings.enableLimiter = true;
    configuration.vfvSettings.maxOrder = 2;
    configuration.vfvSettings.intOrder = 4;
    configuration.Validate();

    ACMSolver<model> solver(mpi, configuration);
    solver.ReadMeshAndInitialize();
    auto &state = solver.GetState();
    typename ACMSolver<model>::TDof rhs;
    solver.GetReconstruction()->BuildUDof(rhs, 5);
    rhs.setConstant(0.0);
    solver.GetEvaluator()->EvaluateRHS(rhs, state, 0);
    real localMaximum = 0;
    for (DNDS::index cell = 0; cell < solver.GetMesh()->NumCell(); ++cell)
        localMaximum = std::max(localMaximum, rhs[cell].cwiseAbs().maxCoeff());
    real globalMaximum = 0;
    MPI_Allreduce(&localMaximum, &globalMaximum, 1, DNDS_MPI_REAL, MPI_MAX, mpi.comm);
    VerifyLUSGSResidualCorrection(solver);
    return globalMaximum;
}

/**
 * @brief Initialize MPI and verify both dimensional modules.
 * @param argc Standard argument count passed to MPI.
 * @param argv Standard argument vector passed to MPI.
 * @return Zero when both global residuals are below roundoff; nonzero otherwise.
 * @author Runzhi Ma
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPIInfo mpi;
    mpi.setWorld();
    int failed = 0;
    try
    {
        const std::filesystem::path root = std::filesystem::path(__FILE__).parent_path()
                                                   .parent_path().parent_path().parent_path();
        const real residual2D = RunStationaryContact<ACMModel::VariableDensity2D>(
            mpi, (root / "data/mesh/ACMVariable_verify2D.cgns").string());
        const real residual3D = RunStationaryContact<ACMModel::VariableDensity3D>(
            mpi, (root / "data/mesh/ACMVariable_verify3D.cgns").string());
        failed = !(residual2D < 1e-12 && residual3D < 1e-12);
        if (mpi.rank == 0)
            std::cout << "ACMVariable MPI residuals: 2D=" << residual2D
                      << " 3D=" << residual3D << std::endl;
    }
    catch (const std::exception &exception)
    {
        if (mpi.rank == 0) std::cerr << exception.what() << std::endl;
        failed = 1;
    }
    int globalFailed = 0;
    MPI_Allreduce(&failed, &globalFailed, 1, MPI_INT, MPI_MAX, mpi.comm);
    MPI_Finalize();
    return globalFailed;
}
