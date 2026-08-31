/**
 * @file ACM.cpp
 * @brief Implementation of constant-density ACM state, flux, boundary, parallel, and configuration kernels.
 *
 * @details This translation unit intentionally remains independent of the Euler evaluator. It provides
 * modular kernels that can later be assembled into a dedicated mesh-based ACM evaluator while reusing
 * DNDS configuration, OpenMP face loops, and MPI collectives.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#include "ACMBC.hpp"
#include "ACMConfig.hpp"
#include "ACMFlux.hpp"
#include "ACMParallel.hpp"

#include "DNDS/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

namespace DNDS::ACM
{
    namespace
    {
        constexpr real normalTolerance = 1e-14;

        /**
         * @brief Validate and normalize a face-normal vector used internally by ACM kernels.
         * @param normal Finite non-zero face-normal direction.
         * @return Unit-length vector parallel to `normal`.
         * @throws std::runtime_error If the vector is non-finite or numerically zero.
         */
        Vector3 NormalizedNormal(const Vector3 &normal)
        {
            DNDS_check_throw_info(normal.allFinite(), "ACM face normal contains a non-finite value");
            const real normalNorm = normal.norm();
            DNDS_check_throw_info(normalNorm > normalTolerance, "ACM face normal has zero length");
            return normal / normalNorm;
        }
    }

    /** @copydoc BuildLocalBasis */
    Matrix3 BuildLocalBasis(const Vector3 &unitNormal)
    {
        const Vector3 normal = NormalizedNormal(unitNormal);
        Vector3 reference;
        const Vector3 normalAbs = normal.cwiseAbs();
        if (normalAbs(0) <= normalAbs(1) && normalAbs(0) <= normalAbs(2))
            reference = Vector3::UnitX();
        else if (normalAbs(1) <= normalAbs(2))
            reference = Vector3::UnitY();
        else
            reference = Vector3::UnitZ();

        Vector3 tangent1 = reference - normal * normal.dot(reference);
        tangent1.normalize();
        Vector3 tangent2 = normal.cross(tangent1);
        tangent2.normalize();

        Matrix3 basis;
        basis.col(0) = normal;
        basis.col(1) = tangent1;
        basis.col(2) = tangent2;
        return basis;
    }

    /** @copydoc Velocity */
    Vector3 Velocity(const State &state)
    {
        return state.head<3>();
    }

    /** @copydoc PhysicalPressure */
    real PhysicalPressure(const State &state)
    {
        return state(3);
    }

    /** @copydoc SetPhysicalPressure */
    void SetPhysicalPressure(State &state, real pressure)
    {
        state(3) = pressure;
    }

    /** @copydoc ToLocalState */
    State ToLocalState(const State &state, const Matrix3 &localBasis)
    {
        DNDS_check_throw_info(state.allFinite(), "ACM state contains a non-finite value");
        DNDS_check_throw_info(localBasis.allFinite(), "ACM local basis contains a non-finite value");
        State localState;
        localState.head<3>() = localBasis.transpose() * state.head<3>();
        localState(3) = state(3);
        return localState;
    }

    /** @copydoc FromLocalFlux */
    State FromLocalFlux(const State &localFlux, const Matrix3 &localBasis)
    {
        DNDS_check_throw_info(localFlux.allFinite(), "ACM local flux contains a non-finite value");
        State globalFlux;
        globalFlux.head<3>() = localBasis * localFlux.head<3>();
        globalFlux(3) = localFlux(3);
        return globalFlux;
    }

    /** @copydoc RotateStateInPlace */
    void RotateStateInPlace(State &state, const Matrix3 &rotation)
    {
        state.head<3>() = rotation * state.head<3>();
    }

    /** @copydoc IsFiniteState */
    bool IsFiniteState(const State &state)
    {
        return state.allFinite();
    }

    /** @copydoc Eigenvalues::SpectralRadius */
    real Eigenvalues::SpectralRadius() const
    {
        return std::max({std::abs(lambdaMinus), std::abs(lambdaTangential), std::abs(lambdaPlus)});
    }

    /** @copydoc PhysicalFluxLocal */
    State PhysicalFluxLocal(const State &localState, real rho0)
    {
        DNDS_check_throw_info(localState.allFinite(), "ACM local state contains a non-finite value");
        DNDS_check_throw_info(std::isfinite(rho0) && rho0 > 0, "ACM rho0 must be finite and positive");
        const real qn = localState(0);
        State flux;
        flux(0) = qn * qn + localState(3) / rho0;
        flux(1) = qn * localState(1);
        flux(2) = qn * localState(2);
        flux(3) = qn;
        return flux;
    }

    /** @copydoc PhysicalFluxJacobianLocal */
    Matrix4 PhysicalFluxJacobianLocal(const State &localState, real rho0)
    {
        DNDS_check_throw_info(localState.allFinite(), "ACM local state contains a non-finite value");
        DNDS_check_throw_info(std::isfinite(rho0) && rho0 > 0, "ACM rho0 must be finite and positive");
        Matrix4 jacobian = Matrix4::Zero();
        jacobian(0, 0) = 2 * localState(0);
        jacobian(0, 3) = 1 / rho0;
        jacobian(1, 0) = localState(1);
        jacobian(1, 1) = localState(0);
        jacobian(2, 0) = localState(2);
        jacobian(2, 2) = localState(0);
        jacobian(3, 0) = 1;
        return jacobian;
    }

    /** @copydoc GammaLocal */
    Matrix4 GammaLocal(const State &meanLocalState, real beta2, real alpha)
    {
        DNDS_check_throw_info(meanLocalState.allFinite(), "ACM mean state contains a non-finite value");
        DNDS_check_throw_info(std::isfinite(beta2) && beta2 > 0, "ACM beta2 must be finite and positive");
        const real gammaT = 1 + alpha;
        Matrix4 gamma = Matrix4::Identity();
        gamma.block<3, 1>(0, 3) = gammaT * meanLocalState.head<3>() / beta2;
        gamma(3, 3) = 1 / beta2;
        return gamma;
    }

    /** @copydoc GammaInvLocal */
    Matrix4 GammaInvLocal(const State &meanLocalState, real beta2, real alpha)
    {
        DNDS_check_throw_info(meanLocalState.allFinite(), "ACM mean state contains a non-finite value");
        DNDS_check_throw_info(std::isfinite(beta2) && beta2 > 0, "ACM beta2 must be finite and positive");
        const real gammaT = 1 + alpha;
        Matrix4 gammaInv = Matrix4::Identity();
        gammaInv.block<3, 1>(0, 3) = -gammaT * meanLocalState.head<3>();
        gammaInv(3, 3) = beta2;
        return gammaInv;
    }

    /** @copydoc ApplyGammaLocal */
    State ApplyGammaLocal(
        const State &meanLocalState,
        const State &increment,
        real beta2,
        real alpha)
    {
        return GammaLocal(meanLocalState, beta2, alpha) * increment;
    }

    /** @copydoc PreconditionedJacobianLocal */
    Matrix4 PreconditionedJacobianLocal(
        const State &meanLocalState,
        real rho0,
        real beta2,
        real alpha)
    {
        return GammaInvLocal(meanLocalState, beta2, alpha) *
               PhysicalFluxJacobianLocal(meanLocalState, rho0);
    }

    /** @copydoc ComputeEigenvalues */
    Eigenvalues ComputeEigenvalues(real qn, real rho0, real beta2, real alpha)
    {
        DNDS_check_throw_info(std::isfinite(qn), "ACM normal velocity must be finite");
        DNDS_check_throw_info(std::isfinite(rho0) && rho0 > 0, "ACM rho0 must be finite and positive");
        DNDS_check_throw_info(std::isfinite(beta2) && beta2 > 0, "ACM beta2 must be finite and positive");
        DNDS_check_throw_info(std::isfinite(alpha), "ACM alpha must be finite");
        const real centeredVelocity = (1 - alpha) * qn;
        const real discriminant = centeredVelocity * centeredVelocity + 4 * beta2 / rho0;
        const real root = std::sqrt(discriminant);
        return {(centeredVelocity - root) * 0.5, qn, (centeredVelocity + root) * 0.5};
    }

    /** @copydoc EntropyFixedAbs */
    real EntropyFixedAbs(real lambda, real delta)
    {
        const real lambdaAbs = std::abs(lambda);
        if (!(delta > 0) || lambdaAbs >= delta)
            return lambdaAbs;
        return (lambda * lambda + delta * delta) / (2 * delta);
    }

    /** @copydoc RusanovDissipationLocal */
    State RusanovDissipationLocal(
        const State &leftLocal,
        const State &rightLocal,
        const Settings &settings,
        Eigenvalues &eigenvalues)
    {
        settings.Validate();
        const State meanState = 0.5 * (leftLocal + rightLocal);
        eigenvalues = ComputeEigenvalues(meanState(0), settings.rho0, settings.beta2, settings.alpha);
        return eigenvalues.SpectralRadius() *
               ApplyGammaLocal(meanState, rightLocal - leftLocal, settings.beta2, settings.alpha);
    }

    /** @copydoc RoeDissipationLocalAlpha0 */
    State RoeDissipationLocalAlpha0(
        const State &leftLocal,
        const State &rightLocal,
        const Settings &settings,
        Eigenvalues &eigenvalues)
    {
        settings.Validate();
        DNDS_check_throw_info(std::abs(settings.alpha) <= 1e-14,
                              "RoeDissipationLocalAlpha0 requires alpha = 0");

        const State meanState = 0.5 * (leftLocal + rightLocal);
        const State increment = rightLocal - leftLocal;
        eigenvalues = ComputeEigenvalues(meanState(0), settings.rho0, settings.beta2, 0);

        const real entropyDelta = settings.entropyFixRatio *
                                  std::max({eigenvalues.SpectralRadius(),
                                            std::sqrt(settings.beta2 / settings.rho0)});
        const real lambdaMinusAbs = EntropyFixedAbs(eigenvalues.lambdaMinus, entropyDelta);
        const real lambdaTangentialAbs = EntropyFixedAbs(eigenvalues.lambdaTangential, entropyDelta);
        const real lambdaPlusAbs = EntropyFixedAbs(eigenvalues.lambdaPlus, entropyDelta);
        const real denominator = eigenvalues.lambdaPlus - eigenvalues.lambdaMinus;
        DNDS_check_throw_info(denominator > std::numeric_limits<real>::epsilon(),
                              "ACM Roe eigenvalues are degenerate");

        const real amplitudeMinus =
            (eigenvalues.lambdaPlus * increment(3) - settings.beta2 * increment(0)) / denominator;
        const real amplitudePlus =
            (settings.beta2 * increment(0) - eigenvalues.lambdaMinus * increment(3)) / denominator;

        State rightMinus;
        rightMinus << eigenvalues.lambdaMinus / settings.beta2, 0, 0, 1;
        State rightPlus;
        rightPlus << eigenvalues.lambdaPlus / settings.beta2, 0, 0, 1;
        State rightTangential1 = State::Zero();
        rightTangential1(1) = 1;
        State rightTangential2 = State::Zero();
        rightTangential2(2) = 1;

        State dissipation =
            lambdaMinusAbs * amplitudeMinus *
                ApplyGammaLocal(meanState, rightMinus, settings.beta2, 0) +
            lambdaPlusAbs * amplitudePlus *
                ApplyGammaLocal(meanState, rightPlus, settings.beta2, 0) +
            lambdaTangentialAbs * increment(1) *
                ApplyGammaLocal(meanState, rightTangential1, settings.beta2, 0) +
            lambdaTangentialAbs * increment(2) *
                ApplyGammaLocal(meanState, rightTangential2, settings.beta2, 0);
        DNDS_check_throw_info(dissipation.allFinite(), "ACM Roe dissipation is non-finite");
        return dissipation;
    }

    /** @copydoc InviscidFlux */
    FluxResult InviscidFlux(
        RiemannSolverType type,
        const State &left,
        const State &right,
        const Vector3 &unitNormal,
        const Settings &settings)
    {
        settings.Validate();
        DNDS_check_throw_info(left.allFinite() && right.allFinite(), "ACM interface state is non-finite");
        const Matrix3 localBasis = BuildLocalBasis(unitNormal);
        const State leftLocal = ToLocalState(left, localBasis);
        const State rightLocal = ToLocalState(right, localBasis);
        const State leftFlux = PhysicalFluxLocal(leftLocal, settings.rho0);
        const State rightFlux = PhysicalFluxLocal(rightLocal, settings.rho0);

        Eigenvalues eigenvalues;
        State dissipation;
        if (type == RiemannSolverType::Rusanov)
            dissipation = RusanovDissipationLocal(leftLocal, rightLocal, settings, eigenvalues);
        else if (type == RiemannSolverType::Roe)
            dissipation = RoeDissipationLocalAlpha0(leftLocal, rightLocal, settings, eigenvalues);
        else
            DNDS_check_throw_info(false, "unknown ACM Riemann solver");

        FluxResult result;
        result.flux = FromLocalFlux(0.5 * (leftFlux + rightFlux - dissipation), localBasis);
        result.eigenvalues = eigenvalues;
        return result;
    }

    /** @copydoc ViscousFlux */
    State ViscousFlux(
        const Eigen::Matrix<real, 3, 4> &stateGradient,
        const Vector3 &unitNormal,
        real rho0,
        real dynamicViscosity)
    {
        DNDS_check_throw_info(stateGradient.allFinite(), "ACM state gradient contains a non-finite value");
        DNDS_check_throw_info(std::isfinite(rho0) && rho0 > 0, "ACM rho0 must be finite and positive");
        DNDS_check_throw_info(std::isfinite(dynamicViscosity) && dynamicViscosity >= 0,
                              "ACM dynamic viscosity must be finite and non-negative");
        const Vector3 normal = NormalizedNormal(unitNormal);

        // stateGradient(row, column) = d(state[column]) / d(x[row]).
        const Matrix3 velocityGradient = stateGradient.block<3, 3>(0, 0).transpose();
        const real divergence = velocityGradient.trace();
        const Matrix3 stress = dynamicViscosity *
                               (velocityGradient + velocityGradient.transpose() -
                                (2.0 / 3.0) * divergence * Matrix3::Identity());

        State flux = State::Zero();
        flux.head<3>() = stress * normal / rho0;
        return flux;
    }

    /** @copydoc GenerateBoundaryState */
    State GenerateBoundaryState(
        BoundaryType type,
        const State &interiorState,
        const State &boundaryValue,
        const Vector3 &unitNormal)
    {
        DNDS_check_throw_info(interiorState.allFinite() && boundaryValue.allFinite(),
                              "ACM boundary state contains a non-finite value");
        const Vector3 normal = NormalizedNormal(unitNormal);
        State ghost = interiorState;
        if (type == BoundaryType::FarField)
            ghost = 2 * boundaryValue - interiorState;
        else if (type == BoundaryType::VelocityInlet)
            ghost.head<3>() = 2 * boundaryValue.head<3>() - interiorState.head<3>();
        else if (type == BoundaryType::PressureOutlet)
            ghost(3) = 2 * boundaryValue(3) - interiorState(3);
        else if (type == BoundaryType::NoSlipWall)
            ghost.head<3>() = 2 * boundaryValue.head<3>() - interiorState.head<3>();
        else if (type == BoundaryType::SlipWall)
        {
            const Vector3 relativeVelocity = interiorState.head<3>() - boundaryValue.head<3>();
            ghost.head<3>() = interiorState.head<3>() - 2 * relativeVelocity.dot(normal) * normal;
        }
        else if (type == BoundaryType::Symmetry)
            ghost.head<3>() = interiorState.head<3>() -
                              2 * interiorState.head<3>().dot(normal) * normal;
        else
            DNDS_check_throw_info(false, "unknown ACM boundary type");
        return ghost;
    }

    /** @copydoc EvaluateFaceFluxes */
    void EvaluateFaceFluxes(
        const std::vector<FaceInput> &faces,
        std::vector<FluxResult> &faceFluxBuffer,
        const Settings &settings)
    {
        settings.Validate();
        for (const auto &face : faces)
        {
            DNDS_check_throw_info(face.left.allFinite() && face.right.allFinite(),
                                  "ACM face state contains a non-finite value");
            static_cast<void>(NormalizedNormal(face.unitNormal));
        }
        faceFluxBuffer.resize(faces.size());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < static_cast<index>(faces.size()); iFace++)
        {
            faceFluxBuffer[static_cast<std::size_t>(iFace)] = InviscidFlux(
                settings.riemannSolverType,
                faces[static_cast<std::size_t>(iFace)].left,
                faces[static_cast<std::size_t>(iFace)].right,
                faces[static_cast<std::size_t>(iFace)].unitNormal,
                settings);
        }
    }

    /** @copydoc LocalFluxSum */
    State LocalFluxSum(const std::vector<FluxResult> &faceFluxBuffer)
    {
        State sum = State::Zero();
        for (const auto &faceFlux : faceFluxBuffer)
            sum += faceFlux.flux;
        return sum;
    }

    /** @copydoc GlobalFluxSum */
    State GlobalFluxSum(const State &localFluxSum, const MPIInfo &mpi)
    {
        State globalFluxSum = State::Zero();
        MPI::Allreduce(
            localFluxSum.data(),
            globalFluxSum.data(),
            4,
            DNDS_MPI_REAL,
            MPI_SUM,
            mpi.comm);
        return globalFluxSum;
    }

    /** @copydoc KernelConfiguration::Validate */
    void KernelConfiguration::Validate() const
    {
        acmSettings.Validate();
        timeMarchSettings.Validate();
        const State left = LeftState();
        const State right = RightState();
        const Vector3 normal = UnitNormal();
        DNDS_check_throw_info(left.allFinite() && right.allFinite(),
                              "ACM preview state contains a non-finite value");
        DNDS_check_throw_info(normal.allFinite() && normal.norm() > normalTolerance,
                              "ACM preview unitNormal is invalid");
        DNDS_check_throw_info(nFacesPerRank > 0, "ACM nFacesPerRank must be positive");
    }

    /** @copydoc KernelConfiguration::LeftState */
    State KernelConfiguration::LeftState() const
    {
        return Eigen::Map<const State>(leftState.data());
    }

    /** @copydoc KernelConfiguration::RightState */
    State KernelConfiguration::RightState() const
    {
        return Eigen::Map<const State>(rightState.data());
    }

    /** @copydoc KernelConfiguration::UnitNormal */
    Vector3 KernelConfiguration::UnitNormal() const
    {
        return Eigen::Map<const Vector3>(unitNormal.data());
    }

    /** @copydoc LoadConfiguration */
    LoadedConfiguration LoadConfiguration(
        const std::string &defaultJsonName,
        const std::string &jsonMergeName,
        const std::vector<std::string> &overwriteKeys,
        const std::vector<std::string> &overwriteValues)
    {
        DNDS_check_throw_info(overwriteKeys.size() == overwriteValues.size(),
                              "ACM overwrite keys and values have different lengths");

        std::ifstream defaultInput(defaultJsonName);
        DNDS_check_throw_info(defaultInput.good(), "ACM default configuration file does not exist: " + defaultJsonName);
        nlohmann::ordered_json resolved = nlohmann::ordered_json::parse(defaultInput, nullptr, true, true);

        if (!jsonMergeName.empty())
        {
            std::ifstream mergeInput(jsonMergeName);
            DNDS_check_throw_info(mergeInput.good(), "ACM configuration patch does not exist: " + jsonMergeName);
            const auto patch = nlohmann::ordered_json::parse(mergeInput, nullptr, true, true);
            resolved.merge_patch(patch);
        }

        for (std::size_t i = 0; i < overwriteKeys.size(); i++)
        {
            const nlohmann::ordered_json::json_pointer key(overwriteKeys[i]);
            try
            {
                resolved[key] = nlohmann::ordered_json::parse(overwriteValues[i]);
            }
            catch (const nlohmann::ordered_json::parse_error &)
            {
                resolved[key] = overwriteValues[i];
            }
        }

        KernelConfiguration configuration = resolved.get<KernelConfiguration>();
        configuration.Validate();
        nlohmann::ordered_json normalized = configuration;
        return {configuration, normalized};
    }
}
