/** Run unmodified traditional NCFV to t=2; export recovered states at both endpoints. */
#include "NCFV/NCFVSolver.hpp"
#include "NCFV/NCFVAnalytic.hpp"

#include <array>
#include <fstream>
#include <iomanip>

namespace
{
    using namespace DNDS::NCFV;
    using State = Eigen::Matrix<DNDS::real, 5, 1>;

    nlohmann::ordered_json Snapshot(
        const DNDS::MPIInfo &mpi, const Solver<3> &solver,
        const Configuration &cfg, const std::filesystem::path &directory,
        const std::string &label)
    {
        const auto mesh = solver.Mesh();
        const auto &geometry = solver.Geometry();
        const auto &reconstruction = solver.ReconstructionData();
        const auto &means = solver.StateField();
        NodeMatrixPair gradients, coefficients;
        gradients.InitPair("NCFV.transientAudit.gradients", mpi);
        gradients.father->Resize(mesh->NumNode(), 3, 5);
        gradients.son->Resize(mesh->NumNodeGhost(), 3, 5);
        gradients.BorrowSetup(mesh->coords);
        gradients.trans.initPersistentPull();
        coefficients.InitPair("NCFV.traditionalTransientAudit.coefficients", mpi);
        coefficients.father->Resize(mesh->NumNode(), 9, 5);
        coefficients.son->Resize(mesh->NumNodeGhost(), 9, 5);
        coefficients.BorrowSetup(mesh->coords);
        coefficients.trans.initPersistentPull();
        NodeStatePair points;
        DNDS::CFV::BuildUDofOnMesh(points, "NCFV.transientAudit.points", mpi,
                                   mesh, 5, true, true, DNDS::Geom::MeshLoc::Node);
        reconstruction.ComputeCoefficients(means, gradients, coefficients);
        coefficients.trans.startPersistentPull();
        coefficients.trans.waitPersistentPull();
        reconstruction.RecoverPointValues(means, gradients, coefficients, points);

        const auto pressure = [&](const State &u)
        {
            return (cfg.physics.gamma - 1) * (u(4) - 0.5 * u.segment<3>(1).squaredNorm() / u(0));
        };
        std::ofstream out(directory / fmt::format("points_{}.rank{:04d}.csv", label, mpi.rank));
        DNDS_check_throw_info(out.good(), "Cannot open transient audit output");
        out << "original_node,x,y,z,partial_volume,mean_rho,mean_rhou,mean_rhov,mean_rhow,mean_rhoE,"
               "point_rho,point_rhou,point_rhov,point_rhow,point_rhoE,rho_exact,p_point,p_exact\n"
            << std::setprecision(17);
        DNDS::real sums[5]{}, maxima[3]{};
        DNDS::index gauss = 0;
        for (DNDS::index i = 0; i < mesh->NumNode(); i++)
        {
            const State point = points[i];
            const State exact = IsentropicVortex<3>(cfg, mesh->coords[i], solver.SimulationTime()).first;
            const DNDS::real p = pressure(point), pExact = pressure(exact);
            DNDS_check_throw_info(point.allFinite() && std::isfinite(p) && point(0) > 0 && p > 0,
                                  "Nonphysical recovered state in transient audit");
            const DNDS::real volume = geometry.NodeVolume(i).moments.measure;
            sums[0] += volume;
            for (int v = 0; v < 2; v++)
            {
                const DNDS::real error = v == 0 ? point(0) - exact(0) : p - pExact;
                sums[1 + 2 * v] += volume * std::abs(error);
                sums[2 + 2 * v] += volume * error * error;
                maxima[v] = std::max(maxima[v], std::abs(error));
            }
            DNDS_check_throw_info(reconstruction.Operator(i).inverseRows.rows() == 9,
                                  "Traditional reconstruction must retain all nine basis rows");
            maxima[2] = std::max(maxima[2], reconstruction.Operator(i).conditionNumber);
            gauss += geometry.NodeVolume(i).volumeQuadrature.size();
            for (const auto &piece : geometry.NodeVolume(i).boundaryPieces)
                gauss += piece.quadrature.size();
            out << mesh->node2nodeOrig(i, 0);
            for (int d = 0; d < 3; d++)
                out << ',' << mesh->coords[i](d);
            out << ',' << volume;
            for (int v = 0; v < 5; v++)
                out << ',' << means[i](v);
            for (int v = 0; v < 5; v++)
                out << ',' << point(v);
            out << ',' << exact(0) << ',' << p << ',' << pExact << '\n';
        }
        out.close();
        DNDS_check_throw_info(out.good(), "Transient audit output failed");
        for (const auto &surface : geometry.EdgeSurfaces())
            gauss += surface.quadrature.size();
        DNDS::real globalSums[5]{}, globalMaxima[3]{};
        DNDS::index globalGauss = 0;
        MPI_Allreduce(sums, globalSums, 5, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
        MPI_Allreduce(maxima, globalMaxima, 3, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
        MPI_Allreduce(&gauss, &globalGauss, 1, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
        DNDS_check_throw_info(globalGauss > 0, "Traditional solver has no quadrature storage");
        nlohmann::ordered_json result = {{"time", solver.SimulationTime()},
                                         {"iteration", solver.CurrentIteration()},
                                         {"volume", globalSums[0]},
                                         {"condition_max", globalMaxima[2]},
                                         {"gauss_points_stored", globalGauss}};
        for (int v = 0; v < 2; v++)
            result[v == 0 ? "rho_point" : "pressure_point"] = {
                {"L1", globalSums[1 + 2 * v] / globalSums[0]},
                {"L2", std::sqrt(globalSums[2 + 2 * v] / globalSums[0])},
                {"Linf", globalMaxima[v]}};
        return result;
    }
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        try
        {
            DNDS_check_throw_info(argc == 2, "Usage: traditional_transient_accuracy_probe config.json");
            const auto cfg = DNDS::NCFV::LoadConfiguration(argv[1], {}, {}).configuration;
            DNDS_check_throw_info(cfg.dimension == 3 && cfg.initialField.isentropicVortex &&
                                      cfg.algorithm.mode == DNDS::NCFV::IntegrationMode::TraditionalQuadrature &&
                                      !cfg.reconstruction.enableLimiter && cfg.io.restartInput.empty() &&
                                      !cfg.physics.viscous.enabled && cfg.algorithm.quadratureOrder == 4 &&
                                      cfg.physics.riemannSolver == DNDS::Euler::Gas::Roe_M2,
                                  "Expected fresh unlimited traditional 3-D vortex");
            DNDS_check_throw_info(cfg.time.useCFLTimeStep && !cfg.time.useLocalTimeStep &&
                                      cfg.time.cfl == 0.5 && cfg.time.endTime == 2 &&
                                      cfg.time.maximumTimeStep == 1e30 && cfg.time.minimumTimeStep == 1e-30,
                                  "Expected the common CFL=0.5, t=2, effectively unclamped settings");
            const auto directory = std::filesystem::path(cfg.io.outputPrefix).parent_path();
            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(directory), "Refusing to overwrite a run directory");
                std::filesystem::create_directories(directory);
            }
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            DNDS::NCFV::Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const auto initial = Snapshot(mpi, solver, cfg, directory, "initial");
            solver.EvaluateResidual(); // Initial unrestricted CFL step, before end-time clipping.
            DNDS::real localMin = DNDS::veryLargeReal, localMax = 0, minStep = 0, maxStep = 0;
            for (DNDS::index i = 0; i < solver.Mesh()->NumNode(); i++)
            {
                localMin = std::min(localMin, solver.LocalTimeStep(i));
                localMax = std::max(localMax, solver.LocalTimeStep(i));
            }
            MPI_Allreduce(&localMin, &minStep, 1, DNDS::DNDS_MPI_REAL, MPI_MIN, mpi.comm);
            MPI_Allreduce(&localMax, &maxStep, 1, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(minStep > cfg.time.minimumTimeStep && maxStep < cfg.time.maximumTimeStep &&
                                      std::abs(maxStep - minStep) < 1e-13,
                                  "Initial physical step is clamped or nonuniform");
            if (mpi.rank == 0)
                DNDS::log() << "Transient audit: CFL=" << cfg.time.cfl << ", initial dt="
                            << std::setprecision(17) << minStep << ", target t=" << cfg.time.endTime << std::endl;
            solver.Run(); // Existing production SSPRK3, reconstruction and flux path unchanged.
            DNDS_check_throw_info(std::abs(solver.SimulationTime() - cfg.time.endTime) < 1e-12,
                                  "Solver stopped before requested physical time");
            const auto final = Snapshot(mpi, solver, cfg, directory, "final");
            const double localSeconds = MPI_Wtime() - start;
            double seconds = 0;
            MPI_Allreduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
            if (mpi.rank == 0)
            {
                nlohmann::ordered_json result = {{"configuration", cfg}, {"mpi_ranks", mpi.size}, {"normalization", "dual-bounds-half-span (thesis 3-34)"}, {"initial_cfl_step", minStep}, {"wall_seconds", seconds}, {"initial", initial}, {"final", final}};
                std::ofstream out(directory / "accuracy.json");
                out << result.dump(2) << '\n';
                out.close();
                DNDS_check_throw_info(out.good(), "Cannot finish transient accuracy metadata");
                DNDS::log() << "Transient audit complete: " << final.dump() << ", wall_seconds=" << seconds << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "NCFV transient audit: " << e.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
