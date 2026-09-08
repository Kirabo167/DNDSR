/** Traditional NCFV first reconstruction: Fnum only, no Solver::Run or time update. */
#include "NCFV/NCFVSolver.hpp"
#include "Geom/Quadrature.hpp"

#include <array>
#include <fstream>
#include <iomanip>

using DNDS::real;
using Index = DNDS::index;
using namespace DNDS::NCFV;
using State = Eigen::Matrix<real, 5, 1>;
using Values = Eigen::Matrix<real, 15, 1>;

namespace
{
    real Pressure(const State &u, real gamma)
    {
        return (gamma - 1) * (u(4) - 0.5 * u.segment<3>(1).squaredNorm() / u(0));
    }

    State PhysicalFlux(const State &u, const Vector3 &n, real gamma)
    {
        const real un = u.segment<3>(1).dot(n) / u(0), p = Pressure(u, gamma);
        State f;
        f << u(0) * un, u.segment<3>(1) * un + p * n, (u(4) + p) * un;
        return f;
    }

    State Flux(const State &l, const State &r, const Vector3 &n, real gamma,
               DNDS::Euler::Gas::RiemannSolverType scheme)
    {
        State f = State::Zero();
        const Vector3 grid = Vector3::Zero();
        real lm = 0, lc = 0, lp = 0;
        DNDS::Euler::Gas::InviscidFlux_IdealGas_Dispatcher<3>(
            scheme, l, r, l, r, grid, n, gamma, f, 0.0, 1.0, 1.0,
            []() {}, lm, lc, lp);
        return f;
    }

    struct Moments
    {
        Values absolute = Values::Zero(), square = Values::Zero(), maximum = Values::Zero();
        real weight = 0;

        void Add(const Values &x, real w)
        {
            DNDS_check_throw_info(x.allFinite() && w > 0, "Invalid quadrature sample");
            absolute += w * x.cwiseAbs();
            square += w * x.cwiseProduct(x);
            maximum = maximum.cwiseMax(x.cwiseAbs());
            weight += w;
        }

        void Merge(const Moments &other)
        {
            absolute += other.absolute;
            square += other.square;
            maximum = maximum.cwiseMax(other.maximum);
            weight += other.weight;
        }

        nlohmann::ordered_json Reduce(const DNDS::MPIInfo &mpi,
                                      const std::array<std::string, 3> &groups) const
        {
            Moments global;
            MPI_Allreduce(absolute.data(), global.absolute.data(), 15, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(square.data(), global.square.data(), 15, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            MPI_Allreduce(maximum.data(), global.maximum.data(), 15, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            MPI_Allreduce(&weight, &global.weight, 1, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            const std::array<std::string, 5> fields{"rho", "rhou", "rhov", "rhow", "rhoE"};
            nlohmann::ordered_json result;
            for (int k = 0; k < 15; k++)
                result[groups[k / 5] + "." + fields[k % 5]] = {
                    {"L1", global.absolute(k) / global.weight},
                    {"L2", std::sqrt(global.square(k) / global.weight)},
                    {"Linf", global.maximum(k)},
                    {"weight", global.weight}};
            return result;
        }
    };
}

int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    {
        DNDS::MPIInfo mpi;
        mpi.setWorld();
        try
        {
            DNDS_check_throw_info(argc == 3 || argc == 4, "Usage: traditional_dissipation_probe config output [quadratureOrder]");
            auto cfg = LoadConfiguration(argv[1], {}, {}).configuration;
            DNDS_check_throw_info(cfg.dimension == 3 && cfg.initialField.isentropicVortex &&
                                      !cfg.reconstruction.enableLimiter && !cfg.physics.viscous.enabled &&
                                      cfg.io.restartInput.empty() && cfg.physics.riemannSolver == DNDS::Euler::Gas::Roe_M2,
                                  "Requires the fresh unlimited inviscid Roe_M2 vortex configuration");
            cfg.algorithm.mode = IntegrationMode::TraditionalQuadrature;
            cfg.algorithm.retainMicroGeometry = false;
            if (argc == 4)
                cfg.algorithm.quadratureOrder = std::stoi(argv[3]);
            DNDS_check_throw_info(cfg.algorithm.quadratureOrder >= 4 && cfg.algorithm.quadratureOrder <= 6,
                                  "Use order 4--6 for this diagnostic");
            cfg.time.iterations = 0;
            cfg.time.endTime = 0;
            cfg.io.writeVTK = cfg.io.writeInitial = cfg.io.writeFinal = false;
            cfg.io.writeFinalRestart = cfg.io.writeResolvedConfiguration = false;
            cfg.io.restartInterval = 0;
            const std::filesystem::path output = argv[2];
            if (mpi.rank == 0)
            {
                DNDS_check_throw_info(!std::filesystem::exists(output), "Refusing to overwrite diagnostic data");
                std::filesystem::create_directories(output);
            }
            MPI_Barrier(mpi.comm);
            const double start = MPI_Wtime();
            Solver<3> solver(mpi, cfg);
            solver.Initialize();
            const auto mesh = solver.Mesh();
            const auto &geometry = solver.Geometry();
            const auto &topology = solver.EdgeTopology();
            const auto &reconstruction = solver.ReconstructionData();
            PeriodicNodes periodic(mpi, mesh, geometry, cfg.mesh);
            periodic.Build(topology, cfg.mesh.periodicTolerance);
            NodeStatePair means, rhs, nodeParts;
            auto allocate = [&](NodeStatePair &field, const std::string &name, int rows)
            {
                DNDS::CFV::BuildUDofOnMesh(field, name, mpi, mesh, rows, true, true, DNDS::Geom::MeshLoc::Node);
            };
            allocate(means, "traditionalDiss.means", 5);
            allocate(rhs, "traditionalDiss.rhs", 5);
            allocate(nodeParts, "traditionalDiss.nodeParts", 15);
            for (Index i = 0; i < mesh->NumNode(); i++)
                means[i] = solver.StateField()[i];
            SpatialOperator<3> spatial(mpi, mesh, topology, geometry, reconstruction, solver.Boundaries(),
                                       cfg.algorithm.mode, cfg.reconstruction, cfg.physics, cfg.time, &periodic);
            spatial.Initialize();
            spatial.EvaluateRHS(means, rhs); // Exactly one reconstruction; no state/time update.
            const auto &points = spatial.PointValues();
            const auto &coefficients = spatial.Coefficients();
            DNDS_check_throw_info(coefficients.father->MatRowSize() == 9,
                                  "Traditional mode must retain all nine quadratic coefficients");
            std::vector<Vector3> lengths(mesh->NumNodeProc());
            for (Index i = 0; i < mesh->NumNodeProc(); i++)
                lengths[i] = reconstruction.ReferenceLengths(i);

            NodeStatePair edgeParts;
            edgeParts.InitPair("traditionalDiss.edgeParts", mpi);
            edgeParts.father->Resize(topology.NumEdge(), 15, 1);
            edgeParts.son->Resize(topology.NumEdgeGhost(), 15, 1);
            edgeParts.BorrowSetup(const_cast<DNDS::Geom::tAdjPair &>(topology.Edge2Node()));
            edgeParts.trans.initPersistentPull();
            Moments quadratureMoments, faceMeanMoments;
            real oracle = 0, areaMismatch = 0, normalMismatch = 0, minRho = DNDS::veryLargeReal;
            real minPressure = DNDS::veryLargeReal, alphaMin = DNDS::veryLargeReal, alphaMax = 0;
            Index surfaceQuadrature = 0, volumeQuadrature = 0, boundaryQuadrature = 0;
            // Per-edge sums retain pointwise norms without a many-million-row point dump.
            const std::array<int, 5> stored{0, 5, 10, 9, 14};
            const std::array<std::string, 5> storedNames{
                "jump_rho", "M2_Fnum_rho", "Roe_Fnum_rho", "M2_Fnum_rhoE", "Roe_Fnum_rhoE"};
            std::ofstream edgeFile(output / fmt::format("edges.rank{:04d}.csv", mpi.rank));
            DNDS_check_throw_info(edgeFile.good(), "Cannot write quadrature diagnostics");
            edgeFile << "node0,node1,area,nq";
            for (const auto &name : storedNames)
                for (const auto &suffix : {"sum_w_abs", "sum_w_square", "max_abs"})
                    edgeFile << ',' << name << '_' << suffix;
            edgeFile << ",mean_M2_Fnum_rho,mean_Roe_Fnum_rho,mean_M2_Fnum_rhoE,mean_Roe_Fnum_rhoE\n"
                     << std::setprecision(17);
            for (Index e = 0; e < topology.NumEdge(); e++)
            {
                const auto &surface = geometry.EdgeSurface(e);
                DNDS_check_throw_info(!surface.quadrature.empty(), "Traditional face has no quadrature");
                surfaceQuadrature += surface.quadrature.size();
                Values integrals = Values::Zero(); // M2 Fnum, Roe Fnum, central flux.
                State jumpIntegral = State::Zero();
                Moments edgeMoments;
                for (const auto &q : surface.quadrature)
                {
                    const Vector3 normal = q.vectorWeight / q.weight;
                    normalMismatch = std::max(normalMismatch, std::abs(normal.norm() - 1));
                    auto evaluate = [&](Index anchor) -> State
                    {
                        const State u = State(points[anchor]) + coefficients[anchor].transpose() *
                                                                    Reconstruction::EvaluateBasis(q.coordinate - mesh->coords[anchor], lengths[anchor], 3);
                        DNDS_check_throw_info(u.allFinite() && u(0) > 1e-12 && Pressure(u, cfg.physics.gamma) > 1e-12,
                                              "Positivity fallback would be required; not an unlimited accuracy sample");
                        minRho = std::min(minRho, u(0));
                        minPressure = std::min(minPressure, Pressure(u, cfg.physics.gamma));
                        return u;
                    };
                    const State left = evaluate(surface.nodes[0]), right = evaluate(surface.nodes[1]);
                    const State jump = right - left;
                    const State center = 0.5 * (PhysicalFlux(left, normal, cfg.physics.gamma) +
                                                PhysicalFlux(right, normal, cfg.physics.gamma));
                    const State m2 = Flux(left, right, normal, cfg.physics.gamma, DNDS::Euler::Gas::Roe_M2) - center;
                    const State roe = Flux(left, right, normal, cfg.physics.gamma, DNDS::Euler::Gas::Roe) - center;
                    auto signal = [&](const State &u)
                    {
                        return std::abs(u.segment<3>(1).dot(normal) / u(0)) +
                               std::sqrt(cfg.physics.gamma * Pressure(u, cfg.physics.gamma) / u(0));
                    };
                    const real alpha = std::max(signal(left), signal(right));
                    alphaMin = std::min(alphaMin, alpha);
                    alphaMax = std::max(alphaMax, alpha);
                    oracle = std::max(oracle, (m2 + 0.5 * alpha * jump).cwiseAbs().maxCoeff());
                    Values samples;
                    samples << jump, m2, roe;
                    edgeMoments.Add(samples, q.weight);
                    integrals.segment<5>(0) += q.weight * m2;
                    integrals.segment<5>(5) += q.weight * roe;
                    integrals.segment<5>(10) += q.weight * center;
                    jumpIntegral += q.weight * jump;
                }
                edgeParts[e] = integrals;
                quadratureMoments.Merge(edgeMoments);
                Values faceMeans;
                faceMeans << jumpIntegral / surface.measure, integrals.segment<5>(0) / surface.measure,
                    integrals.segment<5>(5) / surface.measure;
                faceMeanMoments.Add(faceMeans, surface.measure);
                areaMismatch = std::max(areaMismatch, std::abs(edgeMoments.weight / surface.measure - 1));
                edgeFile << mesh->node2nodeOrig(surface.nodes[0], 0) << ','
                         << mesh->node2nodeOrig(surface.nodes[1], 0) << ',' << surface.measure << ',' << surface.quadrature.size();
                for (int k : stored)
                    edgeFile << ',' << edgeMoments.absolute(k) << ',' << edgeMoments.square(k) << ',' << edgeMoments.maximum(k);
                for (int k : {5, 10, 9, 14})
                    edgeFile << ',' << faceMeans(k);
                edgeFile << '\n';
            }
            edgeFile.close();
            DNDS_check_throw_info(edgeFile.good(), "Incomplete edge output");
            edgeParts.trans.startPersistentPull();
            edgeParts.trans.waitPersistentPull();
            real volume = 0, unknowns = 0, stateChange = 0;
            int minimumRows = 9, maximumRows = 9;
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                const auto &v = geometry.NodeVolume(i);
                volume += v.moments.measure;
                unknowns += v.moments.measure / periodic.Volume(i);
                volumeQuadrature += v.volumeQuadrature.size();
                DNDS_check_throw_info(!v.volumeQuadrature.empty() && v.pointRecoveryWeights.empty(),
                                      "Wrong traditional initialization path");
                for (const auto &piece : v.boundaryPieces)
                {
                    boundaryQuadrature += piece.quadrature.size();
                    DNDS_check_throw_info(solver.Boundaries().Get(piece.zone).mode == BoundaryMode::Periodic,
                                          "Diagnostic assumes periodic boundaries only");
                }
                nodeParts[i].setZero();
                for (const auto &incidence : topology.Node2Edge(i))
                    nodeParts[i] -= incidence.outwardSign * edgeParts[incidence.edge] / v.moments.measure;
                stateChange = std::max(stateChange, (State(means[i]) - State(solver.StateField()[i])).cwiseAbs().maxCoeff());
                const int rows = reconstruction.Operator(i).inverseRows.rows();
                minimumRows = std::min(minimumRows, rows);
                maximumRows = std::max(maximumRows, rows);
            }
            periodic.Average(nodeParts);
            real rhsMatch = 0;
            Values conservation = Values::Zero(), conservationGlobal;
            for (Index i = 0; i < mesh->NumNode(); i++)
            {
                const State mismatch = nodeParts[i].segment<5>(0) + nodeParts[i].segment<5>(10) - State(rhs[i]);
                rhsMatch = std::max(rhsMatch, mismatch.cwiseAbs().maxCoeff());
                conservation += geometry.NodeVolume(i).moments.measure * nodeParts[i];
            }
            MPI_Allreduce(conservation.data(), conservationGlobal.data(), 15, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            const auto qNorms = quadratureMoments.Reduce(mpi, {"jump", "M2.Fnum", "Roe.Fnum"});
            const auto meanNorms = faceMeanMoments.Reduce(mpi, {"jump", "M2.Fnum", "Roe.Fnum"});
            real sumsLocal[]{volume, unknowns}, sums[2]{};
            MPI_Allreduce(sumsLocal, sums, 2, DNDS::DNDS_MPI_REAL, MPI_SUM, mpi.comm);
            Index countsLocal[]{volumeQuadrature, surfaceQuadrature, boundaryQuadrature, topology.NumEdge()}, counts[4]{};
            MPI_Allreduce(countsLocal, counts, 4, DNDS::DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            real maximaLocal[]{oracle, areaMismatch, normalMismatch, rhsMatch, stateChange, alphaMax, MPI_Wtime() - start}, maxima[7]{};
            real minimaLocal[]{minRho, minPressure, alphaMin}, minima[3]{};
            MPI_Allreduce(maximaLocal, maxima, 7, DNDS::DNDS_MPI_REAL, MPI_MAX, mpi.comm);
            MPI_Allreduce(minimaLocal, minima, 3, DNDS::DNDS_MPI_REAL, MPI_MIN, mpi.comm);
            int minRows = 0, maxRows = 0;
            MPI_Allreduce(&minimumRows, &minRows, 1, MPI_INT, MPI_MIN, mpi.comm);
            MPI_Allreduce(&maximumRows, &maxRows, 1, MPI_INT, MPI_MAX, mpi.comm);
            DNDS_check_throw_info(maxima[0] < 1e-12 && maxima[1] < 1e-11 && maxima[2] < 1e-12 &&
                                      maxima[3] < 1e-10 && maxima[4] == 0 && minRows == 9 && maxRows == 9 &&
                                      solver.CurrentIteration() == 0 && solver.SimulationTime() == 0,
                                  "Traditional static diagnostic verification failed");
            if (mpi.rank == 0)
            {
                const Index periodicCount = std::llround(sums[1]);
                const DNDS::Geom::Elem::Quadrature tri({DNDS::Geom::Elem::Tri3}, cfg.algorithm.quadratureOrder);
                const DNDS::Geom::Elem::Quadrature tet({DNDS::Geom::Elem::Tet4}, cfg.algorithm.quadratureOrder);
                nlohmann::ordered_json result = {
                    {"configuration", cfg}, {"source_configuration", argv[1]}, {"time", 0}, {"time_steps", 0}, {"reconstruction_calls", 1}, {"rhs_evaluations", 1}, {"mpi_ranks", mpi.size}, {"nodes", mesh->NumNodeGlobal()}, {"cells", mesh->NumCellGlobal()}, {"edges", counts[3]}, {"volume", sums[0]}, {"periodic_unknowns", periodicCount}, {"h_3d", std::cbrt(sums[0] / periodicCount)}, {"coefficient_rows_min", minRows}, {"coefficient_rows_max", maxRows}, {"volume_quadrature_points", counts[0]}, {"surface_quadrature_points", counts[1]}, {"boundary_quadrature_points", counts[2]}, {"points_per_micro_triangle", tri.GetNumPoints()}, {"points_per_micro_tetrahedron", tet.GetNumPoints()}, {"M2_formula_max_mismatch", maxima[0]}, {"face_area_relative_mismatch", maxima[1]}, {"unit_normal_max_mismatch", maxima[2]}, {"rhs_decomposition_max_mismatch", maxima[3]}, {"mean_state_max_change", maxima[4]}, {"surface_rho_min", minima[0]}, {"surface_pressure_min", minima[1]}, {"alpha_min", minima[2]}, {"alpha_max", maxima[5]}, {"global_rhs_parts_integral", std::vector<real>(conservationGlobal.data(), conservationGlobal.data() + 15)}, {"wall_seconds", maxima[6]}, {"quadrature_norms", qNorms}, {"face_mean_norms", meanNorms}};
                std::ofstream file(output / "metrics.json");
                file << result.dump(2) << '\n';
                DNDS_check_throw_info(file.good(), "Cannot write traditional metrics");
                DNDS::log() << "TRADITIONAL STATIC COMPLETE: time=0, reconstruction=1, M2 rho Fnum L2="
                            << qNorms["M2.Fnum.rho"]["L2"] << ", Roe rho Fnum L2=" << qNorms["Roe.Fnum.rho"]["L2"]
                            << std::endl;
            }
        }
        catch (const std::exception &exception)
        {
            std::cerr << "Traditional dissipation probe: " << exception.what() << std::endl;
            MPI_Abort(mpi.comm, 1);
        }
    }
    MPI_Finalize();
    return 0;
}
