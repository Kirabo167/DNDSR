/**
 * @file ACMSolver.hxx
 * @brief Template implementation of the standalone ACM solver assembly.
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMSolver.hpp"

#include "DNDS/Errors.hpp"
#include "DNDS/OMP.hpp"

#include <iomanip>

namespace DNDS::ACM
{
    template <ACMModel model>
    ACMSolver<model>::ACMSolver(
        const MPIInfo &mpi,
        const KernelConfiguration &configuration)
        : _mpi(mpi), _configuration(configuration)
    {
        _configuration.Validate();
    }

    template <ACMModel model>
    void ACMSolver<model>::ReadMeshAndInitialize()
    {
        DNDS_check_throw_info(
            !_configuration.meshSettings.meshFile.empty(),
            "ACM meshSettings.meshFile must name a CGNS mesh");

        DNDS_MAKE_SSP(_mesh, _mpi, gDim);
        DNDS_MAKE_SSP(_reader, _mesh, 0);
        _mesh->periodicInfo.translation[1].map() = Eigen::Map<const Vector3>(
            _configuration.meshSettings.periodicTranslation1.data());
        _mesh->periodicInfo.translation[2].map() = Eigen::Map<const Vector3>(
            _configuration.meshSettings.periodicTranslation2.data());
        _mesh->periodicInfo.translation[3].map() = Eigen::Map<const Vector3>(
            _configuration.meshSettings.periodicTranslation3.data());
        DNDS_MAKE_SSP(
            _boundaryHandler,
            _configuration.defaultBoundaryType,
            _configuration.BoundaryValue(),
            _configuration.boundaryConditions);
        Geom::ReadMeshFromCGNS(
            _mesh,
            _reader,
            _configuration.meshSettings.meshFile,
            _configuration.meshSettings.partitionOptions,
            _configuration.meshSettings.periodicTolerance,
            _configuration.meshSettings.meshElevation,
            _configuration.meshSettings.meshDirectBisect,
            [this](const std::string &name) -> Geom::t_index
            { return _boundaryHandler->GetIDFromName(name); });

        Geom::PrepareMeshOptions preparationOptions;
        preparationOptions.reorderCells = _configuration.meshSettings.meshReorderCells;
#ifdef DNDS_USE_OMP
        preparationOptions.reorderParts = std::max(get_env_DNDS_DIST_OMP_NUM_THREADS(), 1);
#else
        preparationOptions.reorderParts = 1;
#endif
        preparationOptions.buildSerialOut = false;
        Geom::PrepareMesh(*_mesh, *_reader, preparationOptions);
        _mesh->RecreatePeriodicNodes();

        DNDS_MAKE_SSP(_vfv, _mpi, _mesh);
        _vfv->parseSettings(_configuration.vfvSettings);
        TEvaluator::InitializeFV(
            _mesh,
            _vfv,
            _boundaryHandler);

        _vfv->BuildUDof(_u, Traits::nVarsFixed);
        _vfv->BuildUDof(_rhs, Traits::nVarsFixed);
        _vfv->BuildUDof(_linearRhs, Traits::nVarsFixed);
        _vfv->BuildUDof(_linearIncrement, Traits::nVarsFixed);
        _u.setConstant(_configuration.InitialState());
        _rhs.setConstant(0.0);
        _linearRhs.setConstant(0.0);
        _linearIncrement.setConstant(0.0);

        DNDS_MAKE_SSP(
            _evaluator,
            _mesh,
            _vfv,
            _configuration.acmSettings,
            _configuration.reconstructionSettings,
            _boundaryHandler);

        if (_mpi.rank == 0)
            log() << "ACM mesh/reconstruction initialized: dim=" << gDim
                  << ", global cells=" << _mesh->NumCellGlobal()
                  << ", reconstruction order=" << _configuration.vfvSettings.maxOrder
                  << std::endl;
    }

    template <ACMModel model>
    void ACMSolver<model>::CopyOwnedToStateField(
        const TDof &source,
        StateField &destination) const
    {
        destination.resize(static_cast<std::size_t>(_mesh->NumCell()));
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            destination[static_cast<std::size_t>(iCell)] = source[iCell];
    }

    template <ACMModel model>
    void ACMSolver<model>::CopyStateFieldToOwned(
        const StateField &source,
        TDof &destination) const
    {
        DNDS_check_throw_info(
            source.size() == static_cast<std::size_t>(_mesh->NumCell()),
            "ACM rank-local state field does not match the number of owned mesh cells");
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
        for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            destination[iCell] = source[static_cast<std::size_t>(iCell)];
    }

    template <ACMModel model>
    /** @copydoc ACMSolver::AdvanceImplicitDistributed */
    TimeStepReport ACMSolver<model>::AdvanceImplicitDistributed(
        StateField &states,
        const ScalarField &pseudoTimeStep)
    {
        DNDS_check_throw_info(
            states.size() == pseudoTimeStep.size() &&
                states.size() == static_cast<std::size_t>(_mesh->NumCell()),
            "ACM distributed implicit fields have incompatible sizes");
        const StateField statesOld = states;
        MatrixField diagonal;
        typename TEvaluator::FaceJacobianField faceJacobians;
        TimeStepReport report;

        const auto evaluateDefect = [&]()
        {
            CopyStateFieldToOwned(states, _u);
            _evaluator->EvaluateRHS(_rhs, _u);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
            {
                const std::size_t ii = static_cast<std::size_t>(iCell);
                _linearRhs[iCell] = _rhs[iCell] -
                                    GammaLocal(
                                        _u[iCell],
                                        _configuration.acmSettings.beta2,
                                        _configuration.acmSettings.alpha) *
                                        (states[ii] - statesOld[ii]) /
                                        pseudoTimeStep[ii];
            }
            return _linearRhs.norm2() /
                   std::sqrt(std::max<real>(
                       1.0,
                       static_cast<real>(_mesh->NumCellGlobal()) * Traits::nVarsFixed));
        };

        report.initialDefectNorm = evaluateDefect();
        report.finalDefectNorm = report.initialDefectNorm;
        for (int iteration = 0;
             iteration < _configuration.timeMarchSettings.maxImplicitIterations;
             iteration++)
        {
            if (report.finalDefectNorm <= _configuration.timeMarchSettings.implicitTolerance)
            {
                report.converged = true;
                break;
            }

            _evaluator->AssembleImplicitLinearization(
                _u,
                pseudoTimeStep,
                diagonal,
                faceJacobians);
            _linearIncrement.setConstant(0.0);

            if (_configuration.timeMarchSettings.integrator ==
                TimeIntegratorType::ImplicitEulerLUSGS)
            {
                _evaluator->SolveLUSGS(
                    _linearRhs,
                    diagonal,
                    faceJacobians,
                    _linearIncrement,
                    _configuration.timeMarchSettings.lusgsSweeps);
            }
            else
            {
                Linear::GMRES_LeftPreconditioned<TDof> gmres(
                    static_cast<uint32_t>(_configuration.timeMarchSettings.gmresSubspace),
                    [this](TDof &field)
                    {
                        _vfv->BuildUDof(field, Traits::nVarsFixed);
                        field.setConstant(0.0);
                    });
                gmres.solve(
                    [&](TDof &input, TDof &output)
                    {
                        _evaluator->ApplyImplicitLinearization(
                            input,
                            diagonal,
                            faceJacobians,
                            output);
                    },
                    [&](TDof &input, TDof &output)
                    {
                        if (_configuration.timeMarchSettings.gmresPreconditioner ==
                            GMRESPreconditionerType::LUSGS)
                            _evaluator->SolveLUSGS(
                                input,
                                diagonal,
                                faceJacobians,
                                output,
                                _configuration.timeMarchSettings.lusgsSweeps);
                        else
                            _evaluator->ApplyBlockJacobi(input, diagonal, output);
                    },
                    [](TDof &left, TDof &right)
                    { return left.dot(right); },
                    _linearRhs,
                    _linearIncrement,
                    static_cast<uint32_t>(_configuration.timeMarchSettings.gmresRestarts),
                    [&](uint32_t restart, real residual, real initialResidual)
                    {
                        if (_mpi.rank == 0 && restart > 0)
                            log() << "ACM GMRES restart=" << restart
                                  << " residual=" << residual
                                  << " initial=" << initialResidual << std::endl;
                        return residual <=
                               _configuration.timeMarchSettings.gmresRelativeTolerance *
                                   std::max(initialResidual, verySmallReal);
                    });
            }

#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(runtime)
#endif
            for (index iCell = 0; iCell < _mesh->NumCell(); iCell++)
                states[static_cast<std::size_t>(iCell)] +=
                    _configuration.timeMarchSettings.implicitRelaxation *
                    State(_linearIncrement[iCell]);

            report.iterations = iteration + 1;
            report.finalDefectNorm = evaluateDefect();
        }
        report.converged =
            report.finalDefectNorm <= _configuration.timeMarchSettings.implicitTolerance;
        return report;
    }

    template <ACMModel model>
    void ACMSolver<model>::Run()
    {
        DNDS_check_throw_info(_evaluator != nullptr, "ACM Run requires ReadMeshAndInitialize first");

        StateField states;
        CopyOwnedToStateField(_u, states);
        ScalarField pseudoTimeStep(
            states.size(),
            _configuration.timeMarchSettings.pseudoTimeStep);

        const ResidualEvaluator residualEvaluator =
            [this](const StateField &input, StateField &output)
        {
            CopyStateFieldToOwned(input, _u);
            _evaluator->EvaluateRHS(_rhs, _u);
            CopyOwnedToStateField(_rhs, output);
        };
        const DiagonalJacobianEvaluator diagonalJacobianEvaluator =
            [this](const StateField &input, MatrixField &output)
        {
            CopyStateFieldToOwned(input, _u);
            _evaluator->EvaluateDiagonalJacobian(_u, output);
        };

        for (int iStep = 1; iStep <= _configuration.timeMarchSettings.nSteps; iStep++)
        {
            real minimumTimeStep = _configuration.timeMarchSettings.pseudoTimeStep;
            if (_configuration.timeMarchSettings.useCFLTimeStep)
            {
                CopyStateFieldToOwned(states, _u);
                minimumTimeStep = _evaluator->EvaluateTimeStep(
                    pseudoTimeStep,
                    _u,
                    _configuration.timeMarchSettings.cfl,
                    _configuration.timeMarchSettings.maximumPseudoTimeStep,
                    _configuration.timeMarchSettings.useLocalTimeStep);
            }
            const bool useDistributedImplicit =
                _configuration.timeMarchSettings.integrator ==
                    TimeIntegratorType::ImplicitEulerLUSGS ||
                _configuration.timeMarchSettings.integrator ==
                    TimeIntegratorType::ImplicitEulerGMRES;
            const TimeStepReport report = useDistributedImplicit
                                              ? AdvanceImplicitDistributed(states, pseudoTimeStep)
                                              : AdvancePseudoTimeStep(
                                                    states,
                                                    pseudoTimeStep,
                                                    _configuration.acmSettings,
                                                    _configuration.timeMarchSettings,
                                                    residualEvaluator,
                                                    diagonalJacobianEvaluator,
                                                    &_mpi);
            if (_mpi.rank == 0)
                log() << std::scientific
                      << "ACM step " << std::setw(8) << iStep
                      << " residual " << report.initialDefectNorm
                      << " -> " << report.finalDefectNorm
                      << " dtMin=" << minimumTimeStep
                      << " inner=" << report.iterations
                      << " converged=" << report.converged
                      << std::endl;
        }
        CopyStateFieldToOwned(states, _u);
    }
}
