/**
 * @file acm3D.cpp
 * @brief MPI-enabled command-line preview driver for the constant-density three-dimensional ACM module.
 *
 * @details The driver exercises existing DNDS configuration loading, local face-buffer evaluation,
 * and MPI reduction. Mesh reconstruction and residual time marching are intentionally not connected
 * in this initial modular version.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#include "ACM/ACMConfig.hpp"
#include "ACM/ACMParallel.hpp"
#include "ACM/ACMTime.hpp"

#include <argparse.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    /**
     * @brief Evaluate one unit-area periodic-line face flux, including optional laminar diffusion.
     * @param left State on the negative side of the oriented face.
     * @param right State on the positive side of the oriented face.
     * @param unitNormal Constant line direction and face-normal orientation.
     * @param settings ACM physical and numerical settings.
     * @return Total numerical flux `F_inviscid-F_viscous` for unit spacing and area.
     */
    DNDS::ACM::State PeriodicLineFaceFlux(
        const DNDS::ACM::State &left,
        const DNDS::ACM::State &right,
        const DNDS::ACM::Vector3 &unitNormal,
        const DNDS::ACM::Settings &settings)
    {
        DNDS::ACM::State flux = DNDS::ACM::InviscidFlux(
                                    settings.riemannSolverType,
                                    left,
                                    right,
                                    unitNormal,
                                    settings)
                                    .flux;
        if (settings.enableViscousFlux)
        {
            const Eigen::Matrix<DNDS::real, 3, 4> gradient =
                unitNormal * (right - left).transpose();
            flux -= DNDS::ACM::ViscousFlux(
                gradient,
                unitNormal,
                settings.rho0,
                settings.dynamicViscosity);
        }
        return flux;
    }

    /**
     * @brief Assemble a conservative first-order residual on a rank-local periodic line.
     * @param states Periodic cell states with unit cell volume and spacing.
     * @param residual Output residual `F_{i-1/2}-F_{i+1/2}` for every cell.
     * @param unitNormal Constant positive face-normal direction.
     * @param settings ACM physical and numerical settings.
     */
    void EvaluatePeriodicLineResidual(
        const DNDS::ACM::StateField &states,
        DNDS::ACM::StateField &residual,
        const DNDS::ACM::Vector3 &unitNormal,
        const DNDS::ACM::Settings &settings)
    {
        std::vector<DNDS::ACM::State> faceFlux(states.size(), DNDS::ACM::State::Zero());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (DNDS::index i = 0; i < static_cast<DNDS::index>(states.size()); i++)
        {
            const std::size_t ii = static_cast<std::size_t>(i);
            const std::size_t iRight = (ii + 1) % states.size();
            faceFlux[ii] = PeriodicLineFaceFlux(states[ii], states[iRight], unitNormal, settings);
        }

        residual.resize(states.size());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (DNDS::index i = 0; i < static_cast<DNDS::index>(states.size()); i++)
        {
            const std::size_t ii = static_cast<std::size_t>(i);
            const std::size_t iLeftFace = (ii + states.size() - 1) % states.size();
            residual[ii] = faceFlux[iLeftFace] - faceFlux[ii];
        }
    }

    /**
     * @brief Build finite-difference diagonal residual Jacobians for the periodic preview system.
     * @param states Current periodic-line states.
     * @param diagonalJacobian Output blocks `dR_i/dU_i`.
     * @param unitNormal Constant positive face-normal direction.
     * @param settings ACM physical and numerical settings.
     */
    void EvaluatePeriodicLineDiagonalJacobian(
        const DNDS::ACM::StateField &states,
        DNDS::ACM::MatrixField &diagonalJacobian,
        const DNDS::ACM::Vector3 &unitNormal,
        const DNDS::ACM::Settings &settings)
    {
        diagonalJacobian.resize(states.size());
        constexpr DNDS::real epsilon = 1e-7;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (DNDS::index i = 0; i < static_cast<DNDS::index>(states.size()); i++)
        {
            const std::size_t ii = static_cast<std::size_t>(i);
            const std::size_t iLeft = (ii + states.size() - 1) % states.size();
            const std::size_t iRight = (ii + 1) % states.size();
            DNDS::ACM::Matrix4 jacobian;
            for (int iVariable = 0; iVariable < 4; iVariable++)
            {
                DNDS::ACM::State plus = states[ii];
                DNDS::ACM::State minus = states[ii];
                plus(iVariable) += epsilon;
                minus(iVariable) -= epsilon;
                const DNDS::ACM::State residualPlus =
                    PeriodicLineFaceFlux(states[iLeft], plus, unitNormal, settings) -
                    PeriodicLineFaceFlux(plus, states[iRight], unitNormal, settings);
                const DNDS::ACM::State residualMinus =
                    PeriodicLineFaceFlux(states[iLeft], minus, unitNormal, settings) -
                    PeriodicLineFaceFlux(minus, states[iRight], unitNormal, settings);
                jacobian.col(iVariable) = (residualPlus - residualMinus) / (2 * epsilon);
            }
            diagonalJacobian[ii] = jacobian;
        }
    }
}

/**
 * @brief Initialize MPI, resolve an ACM case configuration, evaluate preview faces, and print diagnostics.
 * @param argc Number of command-line arguments supplied by the process launcher.
 * @param argv Command-line argument array. The optional positional argument is a case merge-patch;
 * `-k/-v` add JSON-pointer overrides and `--emit-schema` prints the generated schema.
 * @return Zero on success; non-zero when configuration or kernel evaluation throws an exception.
 */
int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    DNDS::MPIInfo mpi;
    mpi.setWorld();

    int returnCode = 0;
    try
    {
        argparse::ArgumentParser parser("acm3D", DNDS_VERSION_STRING);
        parser.add_argument("config").default_value("");
        parser.add_argument("-k", "--overwrite_key")
            .append()
            .default_value<std::vector<std::string>>({});
        parser.add_argument("-v", "--overwrite_value")
            .append()
            .default_value<std::vector<std::string>>({});
        parser.add_argument("--emit-schema").flag().default_value(false);
        parser.parse_args(argc, argv);

        if (parser.get<bool>("--emit-schema"))
        {
            if (mpi.rank == 0)
            {
                auto schema = DNDS::ACM::KernelConfiguration::schema(
                    "DNDSR constant-density ACM kernel preview configuration");
                schema["$schema"] = "http://json-schema.org/draft-07/schema#";
                std::cout << schema.dump(4) << std::endl;
            }
            MPI_Barrier(mpi.comm);
            MPI_Finalize();
            return 0;
        }

        std::string defaultConfiguration = "../cases/acm3D/config_base.json";
        const std::string caseConfiguration = parser.get<std::string>("config");
        if (!caseConfiguration.empty())
        {
            const std::filesystem::path casePath(caseConfiguration);
            const std::filesystem::path siblingDefault =
                casePath.parent_path() / "config_base.json";
            if (std::filesystem::exists(siblingDefault))
                defaultConfiguration = siblingDefault.string();
        }

        const auto loaded = DNDS::ACM::LoadConfiguration(
            defaultConfiguration,
            caseConfiguration,
            parser.get<std::vector<std::string>>("--overwrite_key"),
            parser.get<std::vector<std::string>>("--overwrite_value"));
        const auto &configuration = loaded.configuration;

        std::vector<DNDS::ACM::FaceInput> faces(
            static_cast<std::size_t>(configuration.nFacesPerRank));
        for (auto &face : faces)
        {
            face.left = configuration.LeftState();
            face.right = configuration.RightState();
            face.unitNormal = configuration.UnitNormal();
        }

        std::vector<DNDS::ACM::FluxResult> faceFluxBuffer;
        DNDS::ACM::EvaluateFaceFluxes(faces, faceFluxBuffer, configuration.acmSettings);
        const DNDS::ACM::State localSum = DNDS::ACM::LocalFluxSum(faceFluxBuffer);
        const DNDS::ACM::State globalSum = DNDS::ACM::GlobalFluxSum(localSum, mpi);

        DNDS::ACM::StateField timeStates(
            static_cast<std::size_t>(configuration.nFacesPerRank),
            configuration.LeftState());
        for (std::size_t i = timeStates.size() / 2; i < timeStates.size(); i++)
            timeStates[i] = configuration.RightState();
        const DNDS::ACM::ScalarField pseudoTimeStep(
            timeStates.size(),
            configuration.timeMarchSettings.pseudoTimeStep);
        const auto residualEvaluator = [&](const DNDS::ACM::StateField &states, DNDS::ACM::StateField &residual)
        {
            EvaluatePeriodicLineResidual(
                states,
                residual,
                configuration.UnitNormal().normalized(),
                configuration.acmSettings);
        };
        const auto diagonalJacobianEvaluator =
            [&](const DNDS::ACM::StateField &states, DNDS::ACM::MatrixField &diagonalJacobian)
        {
            EvaluatePeriodicLineDiagonalJacobian(
                states,
                diagonalJacobian,
                configuration.UnitNormal().normalized(),
                configuration.acmSettings);
        };

        DNDS::ACM::TimeStepReport timeReport;
        for (int iStep = 0; iStep < configuration.timeMarchSettings.nSteps; iStep++)
            timeReport = DNDS::ACM::AdvancePseudoTimeStep(
                timeStates,
                pseudoTimeStep,
                configuration.acmSettings,
                configuration.timeMarchSettings,
                residualEvaluator,
                diagonalJacobianEvaluator,
                &mpi);

        if (mpi.rank == 0)
        {
            std::cout << "ACM constant-density kernel preview\n"
                      << "NOTE: pseudo-time marching uses a rank-local periodic-line preview; production mesh reconstruction is not connected yet.\n"
                      << "Resolved configuration:\n"
                      << std::setw(4) << loaded.resolvedJson << "\n"
                      << "First face flux [Fu,Fv,Fw,Fcontinuity]: "
                      << faceFluxBuffer.front().flux.transpose() << "\n"
                      << "First face eigenvalues [minus,tangential,plus]: "
                      << faceFluxBuffer.front().eigenvalues.lambdaMinus << " "
                      << faceFluxBuffer.front().eigenvalues.lambdaTangential << " "
                      << faceFluxBuffer.front().eigenvalues.lambdaPlus << "\n"
                      << "MPI-global flux checksum: " << globalSum.transpose() << "\n"
                      << "Periodic-line time steps: " << configuration.timeMarchSettings.nSteps << "\n";
            if (configuration.timeMarchSettings.nSteps > 0)
                std::cout << "Time integrator: "
                          << nlohmann::ordered_json(configuration.timeMarchSettings.integrator).get<std::string>() << "\n"
                          << "Last step iterations: " << timeReport.iterations << "\n"
                          << "Last step initial/final defect: "
                          << timeReport.initialDefectNorm << " " << timeReport.finalDefectNorm << "\n"
                          << "Last step converged: " << std::boolalpha << timeReport.converged << "\n"
                          << "First evolved periodic cell [u,v,w,p]: "
                          << timeStates.front().transpose() << "\n";
            std::cout << std::flush;
        }
    }
    catch (const std::exception &exception)
    {
        std::cerr << "ACM initial driver error on rank " << mpi.rank << ": "
                  << exception.what() << std::endl;
        returnCode = 1;
    }

    if (returnCode)
        MPI_Abort(mpi.comm, returnCode);
    MPI_Finalize();
    return returnCode;
}
