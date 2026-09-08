/**
 * @file test_ACMSelfPeriodic.cpp
 * @brief Regression tests for one-cell periodic ACM implicit operators.
 */
#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include "ACM/ACMSolver.hpp"
#include "ACMVariable/ACMSolver.hpp"

#include <cgnslib.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

using namespace DNDS;

namespace
{
    void CheckCGNS(int status)
    {
        if (status != CG_OK)
            throw std::runtime_error(cg_get_error());
    }

    std::filesystem::path MakeTempMeshPath()
    {
        static int uniquenessAnchor;
        auto nonce = static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        nonce ^= static_cast<unsigned long long>(
            reinterpret_cast<std::uintptr_t>(&uniquenessAnchor));
        return std::filesystem::temp_directory_path() /
               ("dndsr_acm_self_periodic_" + std::to_string(nonce) + ".cgns");
    }

    struct TempMeshFile
    {
        std::filesystem::path path = MakeTempMeshPath();

        ~TempMeshFile()
        {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    };

    void WriteSingleCellPeriodicQuad(const std::filesystem::path &path)
    {
        int file = -1;
        CheckCGNS(cg_open(path.c_str(), CG_MODE_WRITE, &file));
        try
        {
            int base = 0;
            int zone = 0;
            int coordinate = 0;
            int section = 0;
            int boundary = 0;
            CheckCGNS(cg_base_write(file, "Base", 2, 2, &base));

            cgsize_t zoneSize[3] = {4, 1, 0};
            CheckCGNS(cg_zone_write(
                file, base, "Zone", zoneSize, Unstructured, &zone));

            const std::array<double, 4> x{0, 1, 0, 1};
            const std::array<double, 4> y{0, 0, 1, 1};
            CheckCGNS(cg_coord_write(
                file, base, zone, RealDouble, "CoordinateX", x.data(), &coordinate));
            CheckCGNS(cg_coord_write(
                file, base, zone, RealDouble, "CoordinateY", y.data(), &coordinate));

            const std::array<cgsize_t, 4> cell{1, 2, 4, 3};
            CheckCGNS(cg_section_write(
                file, base, zone, "Cells", QUAD_4, 1, 1, 0, cell.data(), &section));

            const std::array<const char *, 4> names{
                "PERIODIC_1", "PERIODIC_1_DONOR",
                "PERIODIC_2", "PERIODIC_2_DONOR"};
            const std::array<std::array<cgsize_t, 2>, 4> edges{{
                {{1, 3}}, {{2, 4}}, {{1, 2}}, {{3, 4}}
            }};
            for (std::size_t i = 0; i < names.size(); i++)
            {
                const cgsize_t element = static_cast<cgsize_t>(2 + i);
                CheckCGNS(cg_section_write(
                    file, base, zone, names[i], BAR_2,
                    element, element, 0, edges[i].data(), &section));
                cgsize_t range[2] = {element, element};
                CheckCGNS(cg_boco_write(
                    file, base, zone, names[i], BCTypeNull,
                    PointRange, 2, range, &boundary));
                CheckCGNS(cg_boco_gridlocation_write(
                    file, base, zone, boundary, EdgeCenter));
            }

            CheckCGNS(cg_close(file));
            file = -1;
        }
        catch (...)
        {
            if (file >= 0)
                cg_close(file);
            throw;
        }
    }

    template <int nVars, class TSolver>
    void VerifySelfPeriodicImplicitOperator(TSolver &solver)
    {
        using TDof = typename TSolver::TDof;
        using TEvaluator = typename TSolver::TEvaluator;
        using Matrix = Eigen::Matrix<real, nVars, nVars>;

        solver.ReadMeshAndInitialize();
        const auto &mesh = solver.GetMesh();
        const auto &vfv = solver.GetReconstruction();
        const auto &evaluator = solver.GetEvaluator();
        REQUIRE(mesh->NumCell() == 1);

        int selfIncidences = 0;
        const auto cellFaces = mesh->cell2face[0];
        for (rowsize ic2f = 0; ic2f < cellFaces.size(); ic2f++)
            if (mesh->CellFaceOther(0, cellFaces[ic2f], ic2f) == 0)
                selfIncidences++;
        REQUIRE(selfIncidences == 4);

        TDof rhs;
        TDof blockSolution;
        TDof lusgsSolution;
        TDof operatorProduct;
        for (TDof *field : {&rhs, &blockSolution, &lusgsSolution, &operatorProduct})
            vfv->BuildUDof(*field, nVars);

        for (int i = 0; i < nVars; i++)
            rhs[0](i) = 0.25 + 0.1 * static_cast<real>(i);

        std::vector<Matrix> residualDiagonal;
        evaluator->EvaluateDiagonalJacobian(
            solver.GetState(), residualDiagonal, 0);
        REQUIRE(residualDiagonal.size() == 1);
        CHECK(residualDiagonal[0].norm() < 1e-10);

        const ACM::ScalarField pseudoTimeStep(1, 0.02);
        std::vector<Matrix> diagonal;
        typename TEvaluator::FaceJacobianField faceJacobians;
        evaluator->AssembleImplicitLinearization(
            solver.GetState(), pseudoTimeStep, diagonal, faceJacobians, 0);

        evaluator->ApplyBlockJacobi(rhs, diagonal, blockSolution);
        evaluator->ApplyImplicitLinearization(
            blockSolution, diagonal, faceJacobians, operatorProduct);
        CHECK((operatorProduct[0] - rhs[0]).norm() < 1e-10);

        evaluator->SolveLUSGS(
            rhs, diagonal, faceJacobians, lusgsSolution, 1);
        evaluator->ApplyImplicitLinearization(
            lusgsSolution, diagonal, faceJacobians, operatorProduct);
        CHECK((operatorProduct[0] - rhs[0]).norm() < 1e-10);
    }
}

TEST_CASE("ACM implicit solvers fold self-periodic coupling into the diagonal")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);

    TempMeshFile meshFile;
    WriteSingleCellPeriodicQuad(meshFile.path);

    ACM::KernelConfiguration configuration;
    configuration.meshSettings.meshFile = meshFile.path.string();
    configuration.meshSettings.periodicTranslation1 = {1, 0, 0};
    configuration.meshSettings.periodicTranslation2 = {0, 1, 0};
    configuration.initialState = {0.4, -0.2, 0.1, 0.3};
    configuration.boundaryValue = configuration.initialState;
    configuration.acmSettings.farFieldValue = configuration.initialState;
    configuration.reconstructionSettings.type = ACM::ReconstructionType::FirstOrder;
    configuration.reconstructionSettings.enableLimiter = false;
    configuration.Validate();

    ACM::ACMSolver<ACM::ACMModel::ConstantDensity2D> solver(mpi, configuration);
    VerifySelfPeriodicImplicitOperator<4>(solver);
}

TEST_CASE("ACMVariable implicit solvers fold self-periodic coupling into the diagonal")
{
    MPIInfo mpi;
    mpi.setWorld();
    REQUIRE(mpi.size == 1);

    TempMeshFile meshFile;
    WriteSingleCellPeriodicQuad(meshFile.path);

    ACMVariable::KernelConfiguration configuration;
    configuration.meshSettings.meshFile = meshFile.path.string();
    configuration.meshSettings.periodicTranslation1 = {1, 0, 0};
    configuration.meshSettings.periodicTranslation2 = {0, 1, 0};
    configuration.initialState = {1.1, 0.2, -0.1, 0.05, 0.3};
    configuration.boundaryValue = configuration.initialState;
    configuration.acmSettings.farFieldValue = configuration.initialState;
    configuration.reconstructionSettings.type =
        ACMVariable::ReconstructionType::FirstOrder;
    configuration.reconstructionSettings.enableLimiter = false;
    configuration.Validate();

    ACMVariable::ACMSolver<ACMVariable::ACMModel::VariableDensity2D> solver(
        mpi, configuration);
    VerifySelfPeriodicImplicitOperator<5>(solver);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    const int result = context.run();
    MPI_Finalize();
    return result;
}
