/**
 * @file ACMEvaluator.hxx
 * @brief Template implementation of the ACM high-order mesh evaluator.
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMEvaluator.hpp"

#include "DNDS/Errors.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace DNDS::ACM
{
    template <int gDim>
    ACMEvaluator<gDim>::ACMEvaluator(
        ssp<Geom::UnstructuredMesh> mesh,
        TpVFV vfv,
        const Settings &settings,
        const ReconstructionSettings &reconstructionSettings,
        ssp<BoundaryHandler> boundaryHandler)
        : _mesh(std::move(mesh)),
          _vfv(std::move(vfv)),
          _settings(settings),
          _reconstructionSettings(reconstructionSettings),
          _boundaryHandler(std::move(boundaryHandler))
    {
        DNDS_check_throw_info(_mesh != nullptr && _vfv != nullptr, "ACM evaluator requires mesh and VFV objects");
        DNDS_check_throw_info(_mesh->getDim() == gDim, "ACM evaluator dimension does not match the mesh");
        _settings.Validate();
        DNDS_check_throw_info(_boundaryHandler != nullptr, "ACM evaluator requires a boundary handler");

        _vfv->BuildURec(_uRec, nVarsFixed);
        _vfv->BuildURec(_uRecWork, nVarsFixed);
        _vfv->BuildURec(_uRecLimited, nVarsFixed);
        _vfv->BuildUGrad(_uGrad, nVarsFixed);
        _vfv->BuildUDof(_limiter, 1);
        _vfv->BuildScalar(_smoothIndicator);
        _uRec.setConstant(0.0);
        _uRecWork.setConstant(0.0);
        _uRecLimited.setConstant(0.0);
        _uGrad.setConstant(0.0);
        _limiter.setConstant(1.0);
    }

    template <int gDim>
    void ACMEvaluator<gDim>::InitializeFV(
        const ssp<Geom::UnstructuredMesh> &mesh,
        const TpVFV &vfv,
        const ssp<BoundaryHandler> &boundaryHandler)
    {
        DNDS_check_throw_info(mesh != nullptr && vfv != nullptr, "ACM InitializeFV requires mesh and VFV objects");
        DNDS_check_throw_info(boundaryHandler != nullptr, "ACM InitializeFV requires a boundary handler");
        if constexpr (gDim == 2)
            vfv->SetPeriodicTransformations(std::array<int, 2>{0, 1});
        else
            vfv->SetPeriodicTransformations(std::array<int, 3>{0, 1, 2});

        vfv->ConstructMetrics();
        vfv->ConstructBaseAndWeight(
            [boundaryHandler](Geom::t_index faceID, int derivativeOrder) -> real
            {
                if (Geom::FaceIDIsPeriodic(faceID))
                    return 1.0;
                (void)boundaryHandler->GetTypeFromID(faceID);
                return derivativeOrder == 0 ? 1.0 : 0.0;
            });
        vfv->ConstructRecCoeff();
    }

    template <int gDim>
    typename ACMEvaluator<gDim>::TBoundaryFunction
    ACMEvaluator<gDim>::GetBoundaryFunction(real time) const
    {
        return [this, time](
                   const State &interior,
                   const State &cellMean,
                   index iCell,
                   index iFace,
                   int iG,
                   const Geom::tPoint &normal,
                   const Geom::tPoint &point,
                   Geom::t_index faceType) -> State
        {
            (void)cellMean;
            (void)iCell;
            (void)iG;
            const Geom::t_index faceZone = Geom::FaceIDIsExternalBC(faceType)
                                                   ? faceType
                                                   : _mesh->GetFaceZone(iFace);
            return GenerateBoundaryForFace(
                faceZone,
                interior,
                ToVector3(normal),
                ToVector3(point),
                time);
        };
    }

    template <int gDim>
    State ACMEvaluator<gDim>::GenerateBoundaryForFace(
        Geom::t_index faceZone,
        const State &interior,
        const Vector3 &normal,
        const Vector3 &point,
        real time) const
    {
        return GenerateBoundaryState(
            _boundaryHandler->GetConditionFromID(faceZone),
            interior,
            normal,
            _settings,
            point,
            time);
    }

    template <int gDim>
    Vector3 ACMEvaluator<gDim>::ToVector3(const Geom::tPoint &normal)
    {
        Vector3 result = Vector3::Zero();
        result.template head<gDim>() = normal.template head<gDim>();
        return result;
    }

    template <int gDim>
    void ACMEvaluator<gDim>::Reconstruct(TDof &u, real time)
    {
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();

        const auto boundary = GetBoundaryFunction(time);
        if (_reconstructionSettings.type == ReconstructionType::FirstOrder)
        {
            _uRec.setConstant(0.0);
            _uGrad.setConstant(0.0);
        }
        else if (_reconstructionSettings.type == ReconstructionType::GreenGauss)
        {
            _vfv->template DoReconstruction2ndGrad<nVarsFixed>(_uGrad, u, boundary, 1);
            _uGrad.trans.startPersistentPull();
            _uGrad.trans.waitPersistentPull();
        }
        else
        {
            if (_reconstructionSettings.resetVariationalCoefficients)
                _uRec.setConstant(0.0);
            for (int iteration = 0; iteration < _reconstructionSettings.variationalIterations; iteration++)
            {
                _vfv->template DoReconstructionIter<nVarsFixed>(
                    _uRec,
                    _uRecWork,
                    u,
                    boundary,
                    false);
                _uRec.trans.startPersistentPull();
                _uRec.trans.waitPersistentPull();
            }
        }
        if (_reconstructionSettings.enableLimiter &&
            _reconstructionSettings.type == ReconstructionType::Variational &&
            _reconstructionSettings.limiterType != LimiterType::LocalExtrema)
            ApplyCharacteristicLimiter(u);
        else
            EvaluateLimiter(u, time);
    }

    template <int gDim>
    State ACMEvaluator<gDim>::ReconstructFaceState(
        const TDof &u,
        index iCell,
        index iFace,
        rowsize if2c,
        int iG) const
    {
        State state = u[iCell];
        if (_reconstructionSettings.type == ReconstructionType::FirstOrder)
            return state;

        State increment = State::Zero();
        if (_reconstructionSettings.type == ReconstructionType::GreenGauss)
        {
            const auto displacement =
                (_vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, iG) -
                 _vfv->GetCellQuadraturePPhys(iCell, -1))
                    .template head<gDim>();
            increment = _uGrad[iCell].transpose() * displacement;
        }
        else
        {
            increment =
                (_vfv->GetIntPointDiffBaseValue(
                     iCell,
                     iFace,
                     if2c,
                     iG,
                     std::array<int, 1>{0},
                     1) *
                 _uRec[iCell])
                    .transpose();
        }
        state += _limiter[iCell](0) * increment;
        return state;
    }

    template <int gDim>
    Eigen::Matrix<real, 3, 4> ACMEvaluator<gDim>::ReconstructFaceGradient(
        index iCell,
        index iFace,
        rowsize if2c,
        int iG) const
    {
        Eigen::Matrix<real, 3, 4> gradient = Eigen::Matrix<real, 3, 4>::Zero();
        if (_reconstructionSettings.type == ReconstructionType::GreenGauss)
            gradient.template topRows<gDim>() = _uGrad[iCell];
        else if (_reconstructionSettings.type == ReconstructionType::Variational)
        {
            const auto derivativeIndices = Eigen::seq(Eigen::fix<1>, Eigen::fix<gDim>);
            gradient.template topRows<gDim>() =
                _vfv->GetIntPointDiffBaseValue(
                    iCell,
                    iFace,
                    if2c,
                    iG,
                    derivativeIndices,
                    gDim + 1) *
                _uRec[iCell];
        }
        gradient *= _limiter[iCell](0);
        return gradient;
    }

    template <int gDim>
    void ACMEvaluator<gDim>::EvaluateLimiter(TDof &u, real time)
    {
        _limiter.setConstant(1.0);
        if (!_reconstructionSettings.enableLimiter ||
            _reconstructionSettings.type == ReconstructionType::FirstOrder)
        {
            _limiter.trans.startPersistentPull();
            _limiter.trans.waitPersistentPull();
            return;
        }

        constexpr real tolerance = 1e-14;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            const State mean = u[iCell];
            State minimum = mean;
            State maximum = mean;
            const auto cellFaces = _mesh->cell2face[iCell];

            for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
            {
                const index iFace = cellFaces[iCellFace];
                const index otherCell = _mesh->CellFaceOther(iCell, iFace);
                const rowsize if2c = _mesh->CellIsFaceBack(iCell, iFace) ? 0 : 1;
                State neighbour;
                if (otherCell != UnInitIndex)
                {
                    neighbour = u[otherCell];
                    Eigen::RowVector<real, 4> neighbourRow = neighbour.transpose();
                    _vfv->ApplyPeriodicTransform(if2c, _mesh->GetFaceZone(iFace), neighbourRow);
                    neighbour = neighbourRow.transpose();
                }
                else
                {
                    const Vector3 outward = ToVector3(_vfv->GetFaceNormFromCell(iFace, iCell, -1, -1));
                    neighbour = GenerateBoundaryForFace(
                        _mesh->GetFaceZone(iFace),
                        mean,
                        outward,
                        ToVector3(_vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, -1)),
                        time);
                }
                minimum = minimum.cwiseMin(neighbour);
                maximum = maximum.cwiseMax(neighbour);
            }

            real theta = 1.0;
            for (rowsize iCellFace = 0; iCellFace < cellFaces.size(); iCellFace++)
            {
                const index iFace = cellFaces[iCellFace];
                const rowsize if2c = _mesh->CellIsFaceBack(iCell, iFace) ? 0 : 1;
                const auto quadrature = _vfv->GetFaceQuad(iFace);
                for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
                {
                    State unlimited = mean;
                    if (_reconstructionSettings.type == ReconstructionType::GreenGauss)
                    {
                        const auto displacement =
                            (_vfv->GetFaceQuadraturePPhysFromCell(iFace, iCell, if2c, iG) -
                             _vfv->GetCellQuadraturePPhys(iCell, -1))
                                .template head<gDim>();
                        unlimited += _uGrad[iCell].transpose() * displacement;
                    }
                    else
                    {
                        unlimited +=
                            (_vfv->GetIntPointDiffBaseValue(
                                 iCell, iFace, if2c, iG,
                                 std::array<int, 1>{0}, 1) *
                             _uRec[iCell])
                                .transpose();
                    }

                    for (int variable = 0; variable < nVarsFixed; variable++)
                    {
                        const real increment = unlimited(variable) - mean(variable);
                        if (increment > tolerance)
                            theta = std::min(theta, (maximum(variable) - mean(variable)) / increment);
                        else if (increment < -tolerance)
                            theta = std::min(theta, (minimum(variable) - mean(variable)) / increment);
                    }
                }
            }
            _limiter[iCell](0) = std::clamp(theta, 0.0, 1.0);
        }
        _limiter.trans.startPersistentPull();
        _limiter.trans.waitPersistentPull();
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::ApplyCharacteristicLimiter */
    void ACMEvaluator<gDim>::ApplyCharacteristicLimiter(TDof &u)
    {
        using TLimitBatch = typename TVFV::template tLimitBatch<nVarsFixed>;
        const real velocityScale = std::sqrt(_settings.beta2 / _settings.rho0);
        const real pressureScale = _settings.beta2;

        _vfv->template DoCalculateSmoothIndicatorV1<nVarsFixed>(
            _smoothIndicator,
            _uRec,
            u,
            State::Ones(),
            [velocityScale, pressureScale](Eigen::Matrix<real, 1, nVarsFixed> &value)
            {
                const State state = value.transpose();
                const real scaledVelocitySquared =
                    state.head<3>().squaredNorm() / (velocityScale * velocityScale);
                const real scaledPressure = state(3) / pressureScale;
                value.setConstant(std::sqrt(scaledVelocitySquared + scaledPressure * scaledPressure));
            });

        const auto toCharacteristic = [this](
                                                const State &left,
                                                const State &right,
                                                const Geom::tPoint &normal,
                                                const Eigen::Ref<TLimitBatch> &coefficients) -> TLimitBatch
        {
            Matrix4 leftTransform;
            Matrix4 rightTransform;
            if (!TryCharacteristicMatricesGlobal(
                    0.5 * (left + right),
                    ToVector3(normal),
                    _settings,
                    leftTransform,
                    rightTransform))
                return coefficients;
            return (leftTransform * coefficients.transpose()).transpose();
        };
        const auto fromCharacteristic = [this](
                                                  const State &left,
                                                  const State &right,
                                                  const Geom::tPoint &normal,
                                                  const Eigen::Ref<TLimitBatch> &coefficients) -> TLimitBatch
        {
            Matrix4 leftTransform;
            Matrix4 rightTransform;
            if (!TryCharacteristicMatricesGlobal(
                    0.5 * (left + right),
                    ToVector3(normal),
                    _settings,
                    leftTransform,
                    rightTransform))
                return coefficients;
            return (rightTransform * coefficients.transpose()).transpose();
        };

        if (_reconstructionSettings.limiterType == LimiterType::CWBAP)
            _vfv->template DoLimiterWBAP_C<nVarsFixed>(
                u,
                _uRec,
                _uRecLimited,
                _uRecWork,
                _smoothIndicator,
                false,
                toCharacteristic,
                fromCharacteristic,
                false);
        else
            _vfv->template DoLimiterWBAP_3<nVarsFixed>(
                u,
                _uRec,
                _uRecLimited,
                _uRecWork,
                _smoothIndicator,
                false,
                toCharacteristic,
                fromCharacteristic,
                false);

        // WBAP/CWBAP modifies polynomial coefficients directly, so the scalar face limiter must
        // remain one to avoid applying the local-extrema limiter a second time.
        _limiter.setConstant(1.0);
        _limiter.trans.startPersistentPull();
        _limiter.trans.waitPersistentPull();
    }

    template <int gDim>
    void ACMEvaluator<gDim>::EvaluateRHS(TDof &rhs, TDof &u, real time)
    {
        Reconstruct(u, time);
        rhs.setConstant(0.0);

        for (index iFace = 0; iFace < _mesh->NumFaceProc(); iFace++)
        {
            const auto faceToCell = _mesh->face2cell[iFace];
            auto quadrature = _vfv->GetFaceQuad(iFace);
            State integratedFlux = State::Zero();
            quadrature.IntegrationSimple(
                integratedFlux,
                [&](State &contribution, int iG, real weight)
                {
                    (void)weight;
                    const Vector3 unitNormal = ToVector3(_vfv->GetFaceNorm(iFace, iG));
                    State left = ReconstructFaceState(u, faceToCell[0], iFace, 0, iG);
                    State right;
                    Eigen::Matrix<real, 3, 4> gradientLeft =
                        ReconstructFaceGradient(faceToCell[0], iFace, 0, iG);
                    Eigen::Matrix<real, 3, 4> gradientRight = gradientLeft;

                    if (faceToCell[1] != UnInitIndex)
                    {
                        right = ReconstructFaceState(u, faceToCell[1], iFace, 1, iG);
                        gradientRight = ReconstructFaceGradient(faceToCell[1], iFace, 1, iG);
                        Eigen::RowVector<real, 4> rightRow = right.transpose();
                        _vfv->ApplyPeriodicTransform(1, _mesh->GetFaceZone(iFace), rightRow);
                        right = rightRow.transpose();
                    }
                    else
                    {
                        right = GenerateBoundaryForFace(
                            _mesh->GetFaceZone(iFace),
                            left,
                            unitNormal,
                            ToVector3(_vfv->GetFaceQuadraturePPhys(iFace, iG)),
                            time);
                    }

                    State totalFlux = InviscidFlux(
                                          _settings.riemannSolverType,
                                          left,
                                          right,
                                          unitNormal,
                                          _settings)
                                          .flux;
                    if (_settings.enableViscousFlux)
                    {
                        Eigen::Matrix<real, 3, 4> faceGradient = 0.5 * (gradientLeft + gradientRight);
                        const real distance = std::max(
                            2.0 * _vfv->GetCellVol(faceToCell[0]) / _vfv->GetFaceArea(iFace),
                            verySmallReal);
                        faceGradient += unitNormal * (right - left).transpose() / distance;
                        totalFlux -= ViscousFlux(
                            faceGradient,
                            unitNormal,
                            _settings.rho0,
                            _settings.dynamicViscosity);
                    }
                    contribution = -totalFlux * _vfv->GetFaceJacobiDet(iFace, iG);
                });

            rhs[faceToCell[0]] += integratedFlux / _vfv->GetCellVol(faceToCell[0]);
            if (faceToCell[1] != UnInitIndex)
                rhs[faceToCell[1]] -= integratedFlux / _vfv->GetCellVol(faceToCell[1]);
        }
    }

    template <int gDim>
    void ACMEvaluator<gDim>::EvaluateDiagonalJacobian(
        TDof &u,
        std::vector<Matrix4> &diagonal,
        real time)
    {
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();
        diagonal.assign(static_cast<std::size_t>(_mesh->NumCell()), Matrix4::Zero());
        constexpr real relativeEpsilon = 1e-7;

        for (index iFace = 0; iFace < _mesh->NumFaceProc(); iFace++)
        {
            const auto faceToCell = _mesh->face2cell[iFace];
            const auto quadrature = _vfv->GetFaceQuad(iFace);
            for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
            {
                const real weight = quadrature.GetWeight(iG);
                {
                    const Vector3 unitNormal = ToVector3(_vfv->GetFaceNorm(iFace, iG));
                    const State left = u[faceToCell[0]];
                    const Vector3 point = ToVector3(_vfv->GetFaceQuadraturePPhys(iFace, iG));
                    State right = faceToCell[1] == UnInitIndex
                                      ? GenerateBoundaryForFace(
                                            _mesh->GetFaceZone(iFace), left, unitNormal, point, time)
                                      : State(u[faceToCell[1]]);
                    const real jacobianDeterminant = _vfv->GetFaceJacobiDet(iFace, iG);

                    Matrix4 leftFluxJacobian = Matrix4::Zero();
                    Matrix4 rightFluxJacobian = Matrix4::Zero();
                    for (int variable = 0; variable < nVarsFixed; variable++)
                    {
                        const real epsilonLeft = relativeEpsilon * std::max(1.0, std::abs(left(variable)));
                        State leftPlus = left;
                        State leftMinus = left;
                        leftPlus(variable) += epsilonLeft;
                        leftMinus(variable) -= epsilonLeft;
                        State rightForPlus = right;
                        State rightForMinus = right;
                        if (faceToCell[1] == UnInitIndex)
                        {
                            rightForPlus = GenerateBoundaryForFace(
                                _mesh->GetFaceZone(iFace), leftPlus, unitNormal, point, time);
                            rightForMinus = GenerateBoundaryForFace(
                                _mesh->GetFaceZone(iFace), leftMinus, unitNormal, point, time);
                        }
                        leftFluxJacobian.col(variable) =
                            (InviscidFlux(_settings.riemannSolverType, leftPlus, rightForPlus, unitNormal, _settings).flux -
                             InviscidFlux(_settings.riemannSolverType, leftMinus, rightForMinus, unitNormal, _settings).flux) /
                            (2.0 * epsilonLeft);

                        if (faceToCell[1] != UnInitIndex)
                        {
                            const real epsilonRight = relativeEpsilon * std::max(1.0, std::abs(right(variable)));
                            State rightPlus = right;
                            State rightMinus = right;
                            rightPlus(variable) += epsilonRight;
                            rightMinus(variable) -= epsilonRight;
                            rightFluxJacobian.col(variable) =
                                (InviscidFlux(_settings.riemannSolverType, left, rightPlus, unitNormal, _settings).flux -
                                 InviscidFlux(_settings.riemannSolverType, left, rightMinus, unitNormal, _settings).flux) /
                                (2.0 * epsilonRight);
                        }
                    }

                    if (faceToCell[0] < _mesh->NumCell())
                        diagonal[static_cast<std::size_t>(faceToCell[0])] -=
                            leftFluxJacobian * (weight * jacobianDeterminant) / _vfv->GetCellVol(faceToCell[0]);
                    if (faceToCell[1] != UnInitIndex && faceToCell[1] < _mesh->NumCell())
                        diagonal[static_cast<std::size_t>(faceToCell[1])] +=
                            rightFluxJacobian * (weight * jacobianDeterminant) / _vfv->GetCellVol(faceToCell[1]);
                }
            }
        }
    }

    template <int gDim>
    real ACMEvaluator<gDim>::EvaluateTimeStep(
        ScalarField &pseudoTimeStep,
        TDof &u,
        real cfl,
        real maximumTimeStep,
        bool useLocalTimeStep,
        real time)
    {
        DNDS_check_throw_info(std::isfinite(cfl) && cfl > 0, "ACM CFL must be finite and positive");
        DNDS_check_throw_info(std::isfinite(maximumTimeStep) && maximumTimeStep > 0,
                              "ACM maximum pseudo-time step must be finite and positive");
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();

        std::vector<real> faceSpectralRadius(static_cast<std::size_t>(_mesh->NumFaceProc()), 0.0);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < _mesh->NumFaceProc(); iFace++)
        {
            const auto faceToCell = _mesh->face2cell[iFace];
            const Vector3 unitNormal = ToVector3(_vfv->GetFaceNorm(iFace, -1));
            const Vector3 point = ToVector3(_vfv->GetFaceQuadraturePPhys(iFace, -1));
            State left = u[faceToCell[0]];
            State right;
            if (faceToCell[1] != UnInitIndex)
            {
                right = u[faceToCell[1]];
                Eigen::RowVector<real, 4> rightRow = right.transpose();
                _vfv->ApplyPeriodicTransform(1, _mesh->GetFaceZone(iFace), rightRow);
                right = rightRow.transpose();
            }
            else
                right = GenerateBoundaryForFace(
                    _mesh->GetFaceZone(iFace), left, unitNormal, point, time);

            const real normalVelocityLeft = left.head<3>().dot(unitNormal);
            const real normalVelocityRight = right.head<3>().dot(unitNormal);
            const real convectiveRadius = std::max(
                ComputeEigenvalues(normalVelocityLeft, _settings.rho0, _settings.beta2, _settings.alpha)
                    .SpectralRadius(),
                ComputeEigenvalues(normalVelocityRight, _settings.rho0, _settings.beta2, _settings.alpha)
                    .SpectralRadius());

            real viscousRadius = 0;
            if (_settings.enableViscousFlux && _settings.dynamicViscosity > 0)
            {
                const real kinematicViscosity = _settings.dynamicViscosity / _settings.rho0;
                const real area = _vfv->GetFaceArea(iFace);
                const real leftVolume = _vfv->GetCellVol(faceToCell[0]);
                const real rightVolume = faceToCell[1] == UnInitIndex
                                             ? leftVolume
                                             : _vfv->GetCellVol(faceToCell[1]);
                viscousRadius = (4.0 / 3.0) * kinematicViscosity * area *
                                (1.0 / leftVolume + 1.0 / rightVolume);
            }
            faceSpectralRadius[static_cast<std::size_t>(iFace)] = convectiveRadius + viscousRadius;
        }

        pseudoTimeStep.resize(static_cast<std::size_t>(_mesh->NumCell()));
        real localMinimum = maximumTimeStep;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime) reduction(min : localMinimum)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            real cellSpectralSum = 0;
            for (const index iFace : _mesh->cell2face[iCell])
                cellSpectralSum += faceSpectralRadius[static_cast<std::size_t>(iFace)] *
                                   _vfv->GetFaceArea(iFace);
            const real dt = std::min(
                cfl * _vfv->GetCellVol(iCell) * _vfv->GetCellSmoothScaleRatio(iCell) /
                    (cellSpectralSum + 1e-100),
                maximumTimeStep);
            pseudoTimeStep[static_cast<std::size_t>(iCell)] = dt;
            localMinimum = std::min(localMinimum, dt);
        }

        real globalMinimum = localMinimum;
        MPI::Allreduce(
            &localMinimum,
            &globalMinimum,
            1,
            DNDS_MPI_REAL,
            MPI_MIN,
            _mesh->getMPI().comm);
        if (!useLocalTimeStep)
            std::fill(pseudoTimeStep.begin(), pseudoTimeStep.end(), globalMinimum);
        return globalMinimum;
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::AssembleImplicitLinearization */
    void ACMEvaluator<gDim>::AssembleImplicitLinearization(
        TDof &u,
        const ScalarField &pseudoTimeStep,
        MatrixField &diagonal,
        FaceJacobianField &faceJacobians,
        real time)
    {
        DNDS_check_throw_info(
            pseudoTimeStep.size() == static_cast<std::size_t>(_mesh->NumCell()),
            "ACM implicit time-step field does not match owned cells");
        u.trans.startPersistentPull();
        u.trans.waitPersistentPull();

        faceJacobians.assign(
            static_cast<std::size_t>(_mesh->NumFaceProc()),
            FaceJacobianBlocks{});
        constexpr real relativeEpsilon = 1e-7;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iFace = 0; iFace < _mesh->NumFaceProc(); iFace++)
        {
            const auto faceToCell = _mesh->face2cell[iFace];
            const State leftBase = u[faceToCell[0]];
            const State rightNativeBase = faceToCell[1] == UnInitIndex
                                              ? State::Zero()
                                              : State(u[faceToCell[1]]);
            const auto quadrature = _vfv->GetFaceQuad(iFace);
            FaceJacobianBlocks integrated;

            for (int iG = 0; iG < quadrature.GetNumPoints(); iG++)
            {
                const Vector3 unitNormal = ToVector3(_vfv->GetFaceNorm(iFace, iG));
                const Vector3 point = ToVector3(_vfv->GetFaceQuadraturePPhys(iFace, iG));
                const real distance = std::max(
                    2.0 * _vfv->GetCellVol(faceToCell[0]) / _vfv->GetFaceArea(iFace),
                    verySmallReal);

                const auto transformRight = [&](const State &rightNative)
                {
                    State transformed = rightNative;
                    Eigen::RowVector<real, 4> row = transformed.transpose();
                    _vfv->ApplyPeriodicTransform(1, _mesh->GetFaceZone(iFace), row);
                    return State(row.transpose());
                };
                const auto evaluateFlux = [&](const State &left, const State &right)
                {
                    State flux = InviscidFlux(
                                     _settings.riemannSolverType,
                                     left,
                                     right,
                                     unitNormal,
                                     _settings)
                                     .flux;
                    if (_settings.enableViscousFlux)
                    {
                        const Eigen::Matrix<real, 3, 4> jumpGradient =
                            unitNormal * (right - left).transpose() / distance;
                        flux -= ViscousFlux(
                            jumpGradient,
                            unitNormal,
                            _settings.rho0,
                            _settings.dynamicViscosity);
                    }
                    return flux;
                };

                Matrix4 leftJacobian = Matrix4::Zero();
                Matrix4 rightJacobian = Matrix4::Zero();
                for (int variable = 0; variable < nVarsFixed; variable++)
                {
                    const real epsilonLeft =
                        relativeEpsilon * std::max(1.0, std::abs(leftBase(variable)));
                    State leftPlus = leftBase;
                    State leftMinus = leftBase;
                    leftPlus(variable) += epsilonLeft;
                    leftMinus(variable) -= epsilonLeft;

                    if (faceToCell[1] == UnInitIndex)
                    {
                        const State ghostPlus = GenerateBoundaryForFace(
                            _mesh->GetFaceZone(iFace), leftPlus, unitNormal, point, time);
                        const State ghostMinus = GenerateBoundaryForFace(
                            _mesh->GetFaceZone(iFace), leftMinus, unitNormal, point, time);
                        leftJacobian.col(variable) =
                            (evaluateFlux(leftPlus, ghostPlus) -
                             evaluateFlux(leftMinus, ghostMinus)) /
                            (2.0 * epsilonLeft);
                    }
                    else
                    {
                        const State rightBase = transformRight(rightNativeBase);
                        leftJacobian.col(variable) =
                            (evaluateFlux(leftPlus, rightBase) -
                             evaluateFlux(leftMinus, rightBase)) /
                            (2.0 * epsilonLeft);

                        const real epsilonRight =
                            relativeEpsilon * std::max(1.0, std::abs(rightNativeBase(variable)));
                        State rightPlus = rightNativeBase;
                        State rightMinus = rightNativeBase;
                        rightPlus(variable) += epsilonRight;
                        rightMinus(variable) -= epsilonRight;
                        rightJacobian.col(variable) =
                            (evaluateFlux(leftBase, transformRight(rightPlus)) -
                             evaluateFlux(leftBase, transformRight(rightMinus))) /
                            (2.0 * epsilonRight);
                    }
                }

                const real integrationFactor =
                    quadrature.GetWeight(iG) * _vfv->GetFaceJacobiDet(iFace, iG);
                integrated.left += integrationFactor * leftJacobian;
                integrated.right += integrationFactor * rightJacobian;
            }
            faceJacobians[static_cast<std::size_t>(iFace)] = integrated;
        }

        diagonal.resize(static_cast<std::size_t>(_mesh->NumCell()));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            DNDS_check_throw_info(
                std::isfinite(pseudoTimeStep[static_cast<std::size_t>(iCell)]) &&
                    pseudoTimeStep[static_cast<std::size_t>(iCell)] > 0,
                "ACM implicit pseudo-time step must be finite and positive");
            Matrix4 block = GammaLocal(
                                u[iCell],
                                _settings.beta2,
                                _settings.alpha) /
                            pseudoTimeStep[static_cast<std::size_t>(iCell)];
            const real inverseVolume = 1.0 / _vfv->GetCellVol(iCell);
            for (const index iFace : _mesh->cell2face[iCell])
            {
                const auto faceToCell = _mesh->face2cell[iFace];
                const auto &faceBlock = faceJacobians[static_cast<std::size_t>(iFace)];
                if (faceToCell[0] == iCell)
                    block += inverseVolume * faceBlock.left;
                else
                    block -= inverseVolume * faceBlock.right;
            }
            DNDS_check_throw_info(block.allFinite(), "ACM implicit diagonal block is non-finite");
            diagonal[static_cast<std::size_t>(iCell)] = block;
        }
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::ApplyImplicitLinearization */
    void ACMEvaluator<gDim>::ApplyImplicitLinearization(
        TDof &increment,
        const MatrixField &diagonal,
        const FaceJacobianField &faceJacobians,
        TDof &result) const
    {
        DNDS_check_throw_info(
            diagonal.size() == static_cast<std::size_t>(_mesh->NumCell()) &&
                faceJacobians.size() == static_cast<std::size_t>(_mesh->NumFaceProc()),
            "ACM implicit operator storage has an invalid size");
        increment.trans.startPersistentPull();
        increment.trans.waitPersistentPull();
        result.setConstant(0.0);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            State value = diagonal[static_cast<std::size_t>(iCell)] * increment[iCell];
            const real inverseVolume = 1.0 / _vfv->GetCellVol(iCell);
            for (const index iFace : _mesh->cell2face[iCell])
            {
                const auto faceToCell = _mesh->face2cell[iFace];
                const index otherCell = _mesh->CellFaceOther(iCell, iFace);
                if (otherCell == UnInitIndex)
                    continue;
                const auto &faceBlock = faceJacobians[static_cast<std::size_t>(iFace)];
                const Matrix4 coupling = faceToCell[0] == iCell
                                             ? inverseVolume * faceBlock.right
                                             : -inverseVolume * faceBlock.left;
                value += coupling * increment[otherCell];
            }
            result[iCell] = value;
        }
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::ApplyBlockJacobi */
    void ACMEvaluator<gDim>::ApplyBlockJacobi(
        const TDof &rhs,
        const MatrixField &diagonal,
        TDof &result) const
    {
        DNDS_check_throw_info(
            diagonal.size() == static_cast<std::size_t>(_mesh->NumCell()),
            "ACM block-Jacobi diagonal has an invalid size");
        result.setConstant(0.0);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
        {
            const auto decomposition =
                diagonal[static_cast<std::size_t>(iCell)].partialPivLu();
            result[iCell] = decomposition.solve(State(rhs[iCell]));
            DNDS_check_throw_info(result[iCell].allFinite(),
                                  "ACM block-Jacobi solve produced a non-finite value");
        }
    }

    template <int gDim>
    /** @copydoc ACMEvaluator::SolveLUSGS */
    void ACMEvaluator<gDim>::SolveLUSGS(
        const TDof &rhs,
        const MatrixField &diagonal,
        const FaceJacobianField &faceJacobians,
        TDof &result,
        int nSweeps) const
    {
        DNDS_check_throw_info(nSweeps > 0, "ACM LU-SGS requires at least one sweep");
        result.setConstant(0.0);
        const index nOwned = _mesh->NumCell();

        for (int sweep = 0; sweep < nSweeps; sweep++)
        {
            result.trans.startPersistentPull();
            result.trans.waitPersistentPull();
            for (index iCell = 0; iCell < nOwned; iCell++)
            {
                State value = rhs[iCell];
                const real inverseVolume = 1.0 / _vfv->GetCellVol(iCell);
                for (const index iFace : _mesh->cell2face[iCell])
                {
                    const auto faceToCell = _mesh->face2cell[iFace];
                    const index otherCell = _mesh->CellFaceOther(iCell, iFace);
                    if (otherCell == UnInitIndex)
                        continue;
                    const bool offRank = otherCell >= nOwned;
                    if (!offRank && otherCell >= iCell)
                        continue;
                    const auto &faceBlock = faceJacobians[static_cast<std::size_t>(iFace)];
                    const Matrix4 coupling = faceToCell[0] == iCell
                                                 ? inverseVolume * faceBlock.right
                                                 : -inverseVolume * faceBlock.left;
                    value -= coupling * result[otherCell];
                }
                result[iCell] = diagonal[static_cast<std::size_t>(iCell)].partialPivLu().solve(value);
            }

            for (index iScan = nOwned; iScan > 0; iScan--)
            {
                const index iCell = iScan - 1;
                State correction = State::Zero();
                const real inverseVolume = 1.0 / _vfv->GetCellVol(iCell);
                for (const index iFace : _mesh->cell2face[iCell])
                {
                    const auto faceToCell = _mesh->face2cell[iFace];
                    const index otherCell = _mesh->CellFaceOther(iCell, iFace);
                    if (otherCell == UnInitIndex || otherCell >= nOwned || otherCell <= iCell)
                        continue;
                    const auto &faceBlock = faceJacobians[static_cast<std::size_t>(iFace)];
                    const Matrix4 coupling = faceToCell[0] == iCell
                                                 ? inverseVolume * faceBlock.right
                                                 : -inverseVolume * faceBlock.left;
                    correction -= coupling * result[otherCell];
                }
                result[iCell] += diagonal[static_cast<std::size_t>(iCell)]
                                     .partialPivLu()
                                     .solve(correction);
            }
        }
        result.trans.startPersistentPull();
        result.trans.waitPersistentPull();
    }
}
