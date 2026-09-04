/**
 * @file test_mpi.cpp
 * @brief MPI integration regression for 2-D/3-D variable-density ACM reconstruction.
 * @details Reads and partitions real CGNS meshes, runs CFV/CWBAP and evaluates a
 * stationary nonuniform-density contact. Its exact inviscid residual is zero.
 * @author Runzhi Ma
 * @date 2026-09-04
 */
#include "ACMVariable/ACMSolver.hpp"

#include <filesystem>
#include <iostream>

using namespace DNDS;
using namespace DNDS::ACMVariable;

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
