#include "NCFVSpatial.hpp"

#include "CFV/DOFFactory.hpp"
#include "DNDS/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace DNDS::NCFV
{
    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::PrimitiveToConservative(
        const std::vector<real> &primitive) const
    {
        DNDS_check_throw_info(primitive.size() == static_cast<std::size_t>(dimension + 2),
                              "NCFV primitive-state size mismatch");
        State state = State::Zero();
        const real density = primitive[0];
        SpatialVector velocity;
        for (int i = 0; i < dimension; i++)
            velocity(i) = primitive[static_cast<std::size_t>(i + 1)];
        const real pressure = primitive[static_cast<std::size_t>(dimension + 1)];
        state(0) = density;
        state.template segment<dimension>(1) = density * velocity;
        state(dimension + 1) = pressure / (_physics.gamma - 1.0) +
                               0.5 * density * velocity.squaredNorm();
        return state;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::ConservativeToPrimitive(const State &state) const
    {
        State primitive = State::Zero();
        primitive(0) = state(0);
        primitive.template segment<dimension>(1) =
            state.template segment<dimension>(1) / state(0);
        primitive(dimension + 1) = Pressure(state);
        return primitive;
    }

    template <int dimension>
    real SpatialOperator<dimension>::Pressure(const State &state) const
    {
        return (_physics.gamma - 1.0) *
               (state(dimension + 1) -
                0.5 * state.template segment<dimension>(1).squaredNorm() / state(0));
    }

    template <int dimension>
    real SpatialOperator<dimension>::Temperature(const State &state) const
    {
        return Pressure(state) /
               (state(0) * _physics.viscous.gasConstant);
    }

    template <int dimension>
    real SpatialOperator<dimension>::MolecularViscosity(const State &state) const
    {
        const auto &settings = _physics.viscous;
        switch (settings.model)
        {
        case ViscosityModel::Constant:
            return settings.dynamicViscosity;
        case ViscosityModel::Sutherland:
        {
            const real temperature = Temperature(state);
            DNDS_check_throw_info(temperature > 0,
                                  "NCFV Sutherland law received a non-positive temperature");
            const real ratio = temperature / settings.referenceTemperature;
            const real denominator = temperature + settings.sutherlandConstant;
            DNDS_check_throw_info(std::abs(denominator) > verySmallReal,
                                  "NCFV Sutherland denominator is zero");
            return settings.dynamicViscosity * ratio * std::sqrt(ratio) *
                   (settings.referenceTemperature + settings.sutherlandConstant) /
                   denominator;
        }
        case ViscosityModel::DensityProportional:
            return settings.dynamicViscosity * state(0);
        }
        DNDS_check_throw_info(false, "NCFV viscosity model is invalid");
        return 0;
    }

    template <int dimension>
    bool SpatialOperator<dimension>::IsPhysical(const State &state) const
    {
        return state.allFinite() && state(0) > 1e-12 && Pressure(state) > 1e-12;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::PreservePhysical(
        const State &candidate,
        const State &anchor) const
    {
        if (IsPhysical(candidate))
            return candidate;
        DNDS_check_throw_info(
            IsPhysical(anchor),
            fmt::format("NCFV encountered a nonphysical anchor: rho={}, E={}, p={}; candidate rho={}, E={}, p={}",
                        anchor(0), anchor(dimension + 1),
                        anchor(0) != 0 ? Pressure(anchor) : -veryLargeReal,
                        candidate(0), candidate(dimension + 1),
                        candidate(0) != 0 ? Pressure(candidate) : -veryLargeReal));
        real lower = 0;
        real upper = 1;
        for (int iteration = 0; iteration < 40; iteration++)
        {
            const real middle = 0.5 * (lower + upper);
            if (IsPhysical(anchor + middle * (candidate - anchor)))
                lower = middle;
            else
                upper = middle;
        }
        return anchor + (0.95 * lower) * (candidate - anchor);
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::PhysicalFlux(
        const State &state,
        const SpatialVector &normal) const
    {
        State flux = State::Zero();
        const SpatialVector momentum = state.template segment<dimension>(1);
        const SpatialVector velocity = momentum / state(0);
        const real pressure = Pressure(state);
        const real normalVelocity = velocity.dot(normal);
        flux(0) = state(0) * normalVelocity;
        flux.template segment<dimension>(1) = momentum * normalVelocity + pressure * normal;
        flux(dimension + 1) = (state(dimension + 1) + pressure) * normalVelocity;
        return flux;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::PhysicalFluxDerivative(
        const State &state,
        const State &stateDerivative,
        int fluxDirection) const
    {
        const SpatialVector momentum = state.template segment<dimension>(1);
        const SpatialVector momentumDerivative =
            stateDerivative.template segment<dimension>(1);
        const SpatialVector velocity = momentum / state(0);
        const SpatialVector velocityDerivative =
            (momentumDerivative - velocity * stateDerivative(0)) / state(0);
        const real pressure = Pressure(state);
        const real pressureDerivative = (_physics.gamma - 1.0) *
                                        (stateDerivative(dimension + 1) -
                                         0.5 * (momentumDerivative.dot(velocity) +
                                                momentum.dot(velocityDerivative)));

        State derivative = State::Zero();
        derivative(0) = momentumDerivative(fluxDirection);
        derivative.template segment<dimension>(1) =
            momentumDerivative * velocity(fluxDirection) +
            momentum * velocityDerivative(fluxDirection);
        derivative(1 + fluxDirection) += pressureDerivative;
        derivative(dimension + 1) =
            (stateDerivative(dimension + 1) + pressureDerivative) *
                velocity(fluxDirection) +
            (state(dimension + 1) + pressure) * velocityDerivative(fluxDirection);
        return derivative;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::NumericalFlux(
        const State &left,
        const State &right,
        const SpatialVector &unitNormal) const
    {
        State flux = State::Zero();
        SpatialVector gridVelocity = SpatialVector::Zero();
        real acousticMinus = 0;
        real contact = 0;
        real acousticPlus = 0;
        Euler::Gas::InviscidFlux_IdealGas_Dispatcher<dimension>(
            _physics.riemannSolver,
            left, right, left, right,
            gridVelocity, unitNormal, _physics.gamma, _physics.gamma, flux,
            0.0, 1.0, 1.0,
            []() {}, acousticMinus, contact, acousticPlus);
        return flux;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::BoundaryExterior(
        const State &inside,
        const SpatialVector &unitNormal,
        const BoundaryZoneSettings &boundary) const
    {
        switch (boundary.mode)
        {
        case BoundaryMode::FarField:
        case BoundaryMode::SupersonicInlet:
            return PrimitiveToConservative(boundary.primitive);
        case BoundaryMode::SupersonicOutlet:
        case BoundaryMode::Periodic:
            return inside;
        case BoundaryMode::PressureOutlet:
        {
            State primitive = ConservativeToPrimitive(inside);
            primitive(dimension + 1) = boundary.staticPressure;
            return PrimitiveToConservative(
                std::vector<real>(primitive.data(), primitive.data() + primitive.size()));
        }
        case BoundaryMode::SlipWall:
        case BoundaryMode::Symmetry:
        {
            State outside = inside;
            SpatialVector momentum = inside.template segment<dimension>(1);
            momentum -= 2.0 * momentum.dot(unitNormal) * unitNormal;
            outside.template segment<dimension>(1) = momentum;
            return outside;
        }
        case BoundaryMode::NoSlipAdiabaticWall:
        case BoundaryMode::NoSlipIsothermalWall:
        {
            State primitive = ConservativeToPrimitive(inside);
            for (int i = 0; i < dimension; i++)
                primitive(1 + i) =
                    2.0 * boundary.wallVelocity[static_cast<std::size_t>(i)] -
                    primitive(1 + i);
            return PrimitiveToConservative(
                std::vector<real>(primitive.data(), primitive.data() + primitive.size()));
        }
        }
        DNDS_check_throw_info(false, "NCFV boundary mode is invalid");
        return inside;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::BoundaryNumericalFlux(
        const State &inside,
        const SpatialVector &unitNormal,
        const BoundaryZoneSettings &boundary) const
    {
        return NumericalFlux(
            inside, BoundaryExterior(inside, unitNormal, boundary), unitNormal);
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EvaluateTraditionalState(
        index anchorNode,
        const Vector3 &point) const
    {
        const real volume = _geometry.NodeMoments()(anchorNode, 0);
        DNDS_check_throw_info(volume > verySmallReal,
                              "NCFV traditional reconstruction has a zero-volume anchor");
        const Vector3 referenceLengths = _reconstruction.ReferenceLengths(anchorNode);
        State state = _pointValues[anchorNode];
        state += _coefficients[anchorNode].transpose() *
                 Reconstruction::EvaluateBasis(
                     point - _mesh->coords[anchorNode],
                     referenceLengths, dimension);
        return PreservePhysical(state, State(_pointValues[anchorNode]));
    }

    template <int dimension>
    typename SpatialOperator<dimension>::Gradient
    SpatialOperator<dimension>::EvaluateTraditionalGradient(
        index anchorNode,
        const Vector3 &point) const
    {
        const real volume = _geometry.NodeMoments()(anchorNode, 0);
        DNDS_check_throw_info(volume > verySmallReal,
                              "NCFV traditional gradient has a zero-volume anchor");
        const Vector3 referenceLengths = _reconstruction.ReferenceLengths(anchorNode);
        return Reconstruction::EvaluateBasisGradient(
                   point - _mesh->coords[anchorNode], referenceLengths, dimension) *
               _coefficients[anchorNode];
    }

    template <int dimension>
    typename SpatialOperator<dimension>::Gradient
    SpatialOperator<dimension>::EfficientSurfaceGradient(
        const std::vector<SparseScalarWeight> &weights,
        real measure) const
    {
        DNDS_check_throw_info(measure > verySmallReal,
                              "NCFV efficient surface has zero measure");
        Gradient gradient = Gradient::Zero();
        for (const auto &weight : weights)
            gradient += weight.value * Gradient(_gradients[weight.node]);
        return gradient / measure;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EfficientIntegratedPhysicalFlux(
        index anchorNode,
        const Vector3 &vectorMeasure,
        const std::vector<SparseMatrixWeight> &weights) const
    {
        State integral = PhysicalFlux(
            State(_pointValues[anchorNode]), vectorMeasure.template head<dimension>());
        for (const auto &weight : weights)
        {
            const State state = _pointValues[weight.node];
            for (int derivativeDirection = 0; derivativeDirection < dimension;
                 derivativeDirection++)
            {
                const State stateDerivative =
                    _gradients[weight.node].row(derivativeDirection).transpose();
                for (int fluxDirection = 0; fluxDirection < dimension; fluxDirection++)
                    integral += weight.value(derivativeDirection, fluxDirection) *
                                PhysicalFluxDerivative(state, stateDerivative, fluxDirection);
            }
        }
        return integral;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EfficientSurfaceMean(
        index anchorNode,
        real measure,
        const std::vector<SparseVectorWeight> &weights) const
    {
        State integral = measure * State(_pointValues[anchorNode]);
        for (const auto &weight : weights)
            integral += _gradients[weight.node].transpose() *
                        weight.value.template head<dimension>();
        const State mean = integral / measure;
        return PreservePhysical(mean, State(_pointValues[anchorNode]));
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::InternalViscousFlux(
        const State &left,
        const State &right,
        const Gradient &leftGradient,
        const Gradient &rightGradient,
        const SpatialVector &unitNormal,
        real normalDistance) const
    {
        State flux = State::Zero();
        if (!_physics.viscous.enabled)
            return flux;

        Gradient leftPrimitiveGradient;
        Gradient rightPrimitiveGradient;
        Euler::Gas::GradientCons2Prim_IdealGas<dimension>(
            left, leftGradient, leftPrimitiveGradient, _physics.gamma,
            Eigen::Vector<real, 0>{});
        Euler::Gas::GradientCons2Prim_IdealGas<dimension>(
            right, rightGradient, rightPrimitiveGradient, _physics.gamma,
            Eigen::Vector<real, 0>{});
        Gradient primitiveGradient =
            0.5 * (leftPrimitiveGradient + rightPrimitiveGradient);

        const State leftPrimitive = ConservativeToPrimitive(left);
        const State rightPrimitive = ConservativeToPrimitive(right);
        const real distance = std::max(normalDistance, verySmallReal);
        const Eigen::RowVector<real, dimension + 2> desiredNormalDerivative =
            (rightPrimitive - leftPrimitive).transpose() / distance;
        // The two traces are reconstructed at the same surface point. Their
        // jump is a penalty, not an estimate of the physical normal gradient.
        primitiveGradient += unitNormal * desiredNormalDerivative;

        const State faceState = PreservePhysical(
            0.5 * (left + right), IsPhysical(left) ? left : right);
        const real viscosity = MolecularViscosity(faceState);
        const real cp = _physics.viscous.gasConstant * _physics.gamma /
                        (_physics.gamma - 1.0);
        const real conductivity =
            viscosity * cp / _physics.viscous.prandtlNumber;
        Euler::Gas::ViscousFlux_IdealGas<dimension>(
            faceState, primitiveGradient, unitNormal, false,
            _physics.gamma, _physics.gamma, viscosity, 0.0, false,
            conductivity, cp, flux);
        return flux;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::BoundaryViscousFlux(
        const State &inside,
        const Gradient &insideGradient,
        const SpatialVector &unitNormal,
        const BoundaryZoneSettings &boundary,
        real lengthScale) const
    {
        State flux = State::Zero();
        if (!_physics.viscous.enabled ||
            boundary.mode == BoundaryMode::SlipWall ||
            boundary.mode == BoundaryMode::Symmetry)
            return flux;

        Gradient primitiveGradient;
        Euler::Gas::GradientCons2Prim_IdealGas<dimension>(
            inside, insideGradient, primitiveGradient, _physics.gamma,
            Eigen::Vector<real, 0>{});
        State fluxState = inside;
        bool adiabatic = false;

        if (boundary.mode == BoundaryMode::NoSlipAdiabaticWall ||
            boundary.mode == BoundaryMode::NoSlipIsothermalWall)
        {
            State insidePrimitive = ConservativeToPrimitive(inside);
            State wallPrimitive = insidePrimitive;
            for (int i = 0; i < dimension; i++)
                wallPrimitive(1 + i) =
                    boundary.wallVelocity[static_cast<std::size_t>(i)];
            if (boundary.mode == BoundaryMode::NoSlipIsothermalWall)
                wallPrimitive(dimension + 1) =
                    wallPrimitive(0) * _physics.viscous.gasConstant *
                    boundary.wallTemperature;
            adiabatic = boundary.mode == BoundaryMode::NoSlipAdiabaticWall;

            const real wallDistance = std::max(0.5 * lengthScale, verySmallReal);
            const Eigen::RowVector<real, dimension + 2> desiredNormalDerivative =
                (wallPrimitive - insidePrimitive).transpose() / wallDistance;
            primitiveGradient += unitNormal * desiredNormalDerivative;
            fluxState = PrimitiveToConservative(
                std::vector<real>(wallPrimitive.data(),
                                  wallPrimitive.data() + wallPrimitive.size()));
        }

        const real viscosity = MolecularViscosity(fluxState);
        const real cp = _physics.viscous.gasConstant * _physics.gamma /
                        (_physics.gamma - 1.0);
        const real conductivity =
            viscosity * cp / _physics.viscous.prandtlNumber;
        Euler::Gas::ViscousFlux_IdealGas<dimension>(
            fluxState, primitiveGradient, unitNormal, adiabatic,
            _physics.gamma, _physics.gamma, viscosity, 0.0, false,
            conductivity, cp, flux);
        return flux;
    }

    template <int dimension>
    real SpatialOperator<dimension>::SurfaceSpectralRadius(
        const State &state,
        const SpatialVector &unitNormal,
        real measure,
        real lengthScale) const
    {
        const SpatialVector velocity =
            state.template segment<dimension>(1) / state(0);
        const real acousticSpeed =
            std::sqrt(_physics.gamma * Pressure(state) / state(0));
        real spectralRadius =
            (std::abs(velocity.dot(unitNormal)) + acousticSpeed) * measure;
        if (_physics.viscous.enabled)
        {
            const real kinematicViscosity = MolecularViscosity(state) / state(0);
            const real largestDiffusivity = std::max(
                (4.0 / 3.0) * kinematicViscosity,
                kinematicViscosity / _physics.viscous.prandtlNumber);
            spectralRadius += _physics.viscous.spectralRadiusFactor *
                              largestDiffusivity * measure /
                              std::max(lengthScale, verySmallReal);
        }
        return spectralRadius;
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EvaluateOwnedEdgeFlux(index iEdge) const
    {
        const auto &surface = _geometry.EdgeSurface(iEdge);
        const index node0 = surface.nodes[0];
        const index node1 = surface.nodes[1];
        const SpatialVector unitNormal =
            surface.vectorMeasure.template head<dimension>().normalized();
        const Vector3 edgeVector = _mesh->coords[node1] - _mesh->coords[node0];
        real normalDistance =
            std::abs(edgeVector.template head<dimension>().dot(unitNormal));
        if (normalDistance <= verySmallReal)
            normalDistance = edgeVector.template head<dimension>().norm();

        if (_mode == IntegrationMode::TraditionalQuadrature)
        {
            State integral = State::Zero();
            for (const auto &point : surface.quadrature)
            {
                const SpatialVector pointNormal =
                    point.vectorWeight.template head<dimension>() / point.weight;
                const State left = EvaluateTraditionalState(node0, point.coordinate);
                const State right = EvaluateTraditionalState(node1, point.coordinate);
                State flux = NumericalFlux(left, right, pointNormal);
                if (_physics.viscous.enabled)
                    flux -= InternalViscousFlux(
                        left, right,
                        EvaluateTraditionalGradient(node0, point.coordinate),
                        EvaluateTraditionalGradient(node1, point.coordinate),
                        pointNormal, normalDistance);
                integral += point.weight * flux;
            }
            return integral;
        }

        const State leftIntegral = EfficientIntegratedPhysicalFlux(
            node0, surface.vectorMeasure, surface.leftFluxWeights);
        const State rightIntegral = EfficientIntegratedPhysicalFlux(
            node1, surface.vectorMeasure, surface.rightFluxWeights);
        const State leftMean = EfficientSurfaceMean(
            node0, surface.measure, surface.leftStateWeights);
        const State rightMean = EfficientSurfaceMean(
            node1, surface.measure, surface.rightStateWeights);
        const State numerical = NumericalFlux(leftMean, rightMean, unitNormal);
        const State centralMean = 0.5 *
                                  (PhysicalFlux(leftMean, unitNormal) +
                                   PhysicalFlux(rightMean, unitNormal));
        State integral = 0.5 * (leftIntegral + rightIntegral) +
                         surface.measure * (numerical - centralMean);
        if (_physics.viscous.enabled)
        {
            const Gradient surfaceGradient = EfficientSurfaceGradient(
                surface.gradientIntegralWeights, surface.measure);
            integral -= surface.vectorMeasure.template head<dimension>().norm() *
                        InternalViscousFlux(
                            leftMean, rightMean, surfaceGradient, surfaceGradient,
                            unitNormal, normalDistance);
        }
        return integral;
    }

    template <int dimension>
    real SpatialOperator<dimension>::EvaluateOwnedEdgeSpectralRadius(index iEdge) const
    {
        const auto &surface = _geometry.EdgeSurface(iEdge);
        const State mean = PreservePhysical(
            0.5 * (State(_pointValues[surface.nodes[0]]) +
                   State(_pointValues[surface.nodes[1]])),
            State(_pointValues[surface.nodes[0]]));
        const SpatialVector normal =
            surface.vectorMeasure.template head<dimension>().normalized();
        const real length0 = std::pow(
            _geometry.NodeMoments()(surface.nodes[0], 0),
            1.0 / static_cast<real>(dimension));
        const real length1 = std::pow(
            _geometry.NodeMoments()(surface.nodes[1], 0),
            1.0 / static_cast<real>(dimension));
        return SurfaceSpectralRadius(
            mean, normal, surface.measure, std::min(length0, length1));
    }

    template <int dimension>
    typename SpatialOperator<dimension>::State
    SpatialOperator<dimension>::EvaluateOwnedBoundaryFlux(index iNode) const
    {
        State total = State::Zero();
        const auto &controlVolume = _geometry.NodeVolume(iNode);
        if (_mode == IntegrationMode::TraditionalQuadrature)
        {
            for (const auto &piece : controlVolume.boundaryPieces)
            {
                const auto &boundary = _boundaries.Get(piece.zone);
                if (boundary.mode == BoundaryMode::Periodic)
                    continue;
                for (const auto &point : piece.quadrature)
                {
                    const SpatialVector unitNormal =
                        point.vectorWeight.template head<dimension>() / point.weight;
                    const State inside = EvaluateTraditionalState(iNode, point.coordinate);
                    State flux = BoundaryNumericalFlux(inside, unitNormal, boundary);
                    if (_physics.viscous.enabled)
                        flux -= BoundaryViscousFlux(
                            inside,
                            EvaluateTraditionalGradient(iNode, point.coordinate),
                            unitNormal, boundary, controlVolume.lengthScale);
                    total += point.weight * flux;
                }
            }
            return total;
        }

        // No Gaussian points: inviscid boundary fluxes use construction
        // vertices and viscous gradients use their exact affine surface mean.
        for (const auto &piece : controlVolume.boundaryPieces)
        {
            const auto &boundary = _boundaries.Get(piece.zone);
            if (boundary.mode == BoundaryMode::Periodic)
                continue;
            const SpatialVector unitNormal =
                piece.vectorMeasure.template head<dimension>() / piece.measure;
            State vertexFluxSum = State::Zero();
            State insideMean = State::Zero();
            for (int iPoint = 0; iPoint < piece.nPoints; iPoint++)
            {
                State affineState = State::Zero();
                for (const auto &coefficient :
                     piece.points[static_cast<std::size_t>(iPoint)].support)
                    affineState += coefficient.coefficient *
                                   State(_pointValues[coefficient.node]);
                affineState = PreservePhysical(affineState, State(_pointValues[iNode]));
                insideMean += affineState;
                vertexFluxSum += BoundaryNumericalFlux(
                    affineState, unitNormal, boundary);
            }
            insideMean /= static_cast<real>(piece.nPoints);
            total += piece.measure / static_cast<real>(piece.nPoints) * vertexFluxSum;
            if (_physics.viscous.enabled)
                total -= piece.measure * BoundaryViscousFlux(
                                             insideMean,
                                             EfficientSurfaceGradient(
                                                 piece.gradientIntegralWeights,
                                                 piece.measure),
                                             unitNormal, boundary,
                                             controlVolume.lengthScale);
        }
        return total;
    }

    template <int dimension>
    real SpatialOperator<dimension>::EvaluateOwnedBoundarySpectralRadius(
        index iNode) const
    {
        real total = 0;
        const auto &controlVolume = _geometry.NodeVolume(iNode);
        const State state = _pointValues[iNode];
        for (const auto &piece : controlVolume.boundaryPieces)
        {
            if (_boundaries.Get(piece.zone).mode == BoundaryMode::Periodic)
                continue;
            const SpatialVector normal =
                piece.vectorMeasure.template head<dimension>() / piece.measure;
            total += SurfaceSpectralRadius(
                state, normal, piece.measure, controlVolume.lengthScale);
        }
        return total;
    }

    template <int dimension>
    void SpatialOperator<dimension>::AllocateNodeMatrix(
        NodeMatrixPair &field,
        const std::string &name,
        int rows)
    {
        field.InitPair(name, _mpi);
        field.father->Resize(_mesh->NumNode(), rows, dimension + 2);
        field.son->Resize(_mesh->NumNodeGhost(), rows, dimension + 2);
        field.BorrowSetup(_mesh->coords);
        field.trans.initPersistentPull();
        for (index iNode = 0; iNode < field.Size(); iNode++)
            field[iNode].setZero();
    }

    template <int dimension>
    void SpatialOperator<dimension>::AllocateEdgeField(
        NodeStatePair &field,
        const std::string &name,
        int rows)
    {
        field.InitPair(name, _mpi);
        field.father->Resize(_topology.NumEdge(), rows, 1);
        field.son->Resize(_topology.NumEdgeGhost(), rows, 1);
        field.BorrowSetup(const_cast<Geom::tAdjPair &>(_topology.Edge2Node()));
        field.trans.initPersistentPull();
        for (index iEdge = 0; iEdge < field.Size(); iEdge++)
            field[iEdge].setZero();
    }

    template <int dimension>
    void SpatialOperator<dimension>::Initialize()
    {
        AllocateNodeMatrix(_gradients, "NCFV.gradients", dimension);
        if (_mode == IntegrationMode::TraditionalQuadrature)
            AllocateNodeMatrix(_coefficients, "NCFV.coefficients",
                               Reconstruction::QuadraticBasisSize(dimension));
        CFV::BuildUDofOnMesh(
            _pointValues, "NCFV.pointValues", _mpi, _mesh,
            dimension + 2, true, true, Geom::MeshLoc::Node);
        CFV::BuildUDofOnMesh(
            _localTimeSteps, "NCFV.localTimeSteps", _mpi, _mesh,
            1, true, true, Geom::MeshLoc::Node);

        AllocateEdgeField(_edgeFlux, "NCFV.edgeFlux", dimension + 2);
        AllocateEdgeField(_edgeSpectralRadius, "NCFV.edgeSpectralRadius", 1);

        _farField = PrimitiveToConservative(_physics.farFieldPrimitive);
        for (index iNode = 0; iNode < _localTimeSteps.Size(); iNode++)
            _localTimeSteps[iNode](0, 0) = _time.timeStep;
    }

    template <int dimension>
    void SpatialOperator<dimension>::Reconstruct(NodeStatePair &means)
    {
        means.trans.startPersistentPull();
        means.trans.waitPersistentPull();

        _reconstruction.ComputeCoefficients(means, _gradients, _coefficients);
        if (_mode == IntegrationMode::EfficientDifferential)
        {
            _gradients.trans.startPersistentPull();
            _gradients.trans.waitPersistentPull();
        }
        else
        {
            _coefficients.trans.startPersistentPull();
            _coefficients.trans.waitPersistentPull();
        }

        _reconstruction.RecoverPointValues(
            means, _gradients, _coefficients, _pointValues);
        if (_reconstructionSettings.enableLimiter)
        {
            const auto factors = _reconstruction.ComputeLimiterFactors(
                means, _pointValues, _gradients, _coefficients);
            _reconstruction.ApplyLimiter(factors, _gradients, _coefficients);
            if (_mode == IntegrationMode::EfficientDifferential)
            {
                _gradients.trans.startPersistentPull();
                _gradients.trans.waitPersistentPull();
            }
            else
            {
                _coefficients.trans.startPersistentPull();
                _coefficients.trans.waitPersistentPull();
            }
            _reconstruction.RecoverPointValues(
                means, _gradients, _coefficients, _pointValues);
        }
        _pointValues.trans.startPersistentPull();
        _pointValues.trans.waitPersistentPull();
    }

    template <int dimension>
    void SpatialOperator<dimension>::UpdateLocalTimeSteps()
    {
        std::vector<real> candidate(static_cast<std::size_t>(_mesh->NumNode()),
                                    _time.timeStep);
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            if (!_time.useCFLTimeStep)
                continue;
            real spectralRadius = EvaluateOwnedBoundarySpectralRadius(iNode);
            for (const auto &incidence : _topology.Node2Edge(iNode))
                spectralRadius += _edgeSpectralRadius[incidence.edge](0, 0);
            const real volume = _geometry.NodeVolume(iNode).moments.measure;
            candidate[static_cast<std::size_t>(iNode)] = std::clamp(
                _time.cfl * volume / std::max(spectralRadius, verySmallReal),
                _time.minimumTimeStep, _time.maximumTimeStep);
        }

        // Combine inverse steps on the quotient: sum spectra / sum volumes.
        if (_periodic && _time.useCFLTimeStep)
        {
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                _localTimeSteps[iNode](0, 0) = 1.0 / candidate[iNode];
            _periodic->Average(_localTimeSteps);
            for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
                candidate[iNode] = 1.0 / _localTimeSteps[iNode](0, 0);
        }
        for (real &step : candidate)
            step = std::min(step, _maximumStep);

        if (_time.useCFLTimeStep && !_time.useLocalTimeStep)
        {
            real localMinimum = candidate.empty()
                                    ? std::numeric_limits<real>::max()
                                    : *std::min_element(candidate.begin(), candidate.end());
            real globalMinimum = 0;
            MPI_Allreduce(&localMinimum, &globalMinimum, 1,
                          DNDS_MPI_REAL, MPI_MIN, _mpi.comm);
            std::fill(candidate.begin(), candidate.end(), globalMinimum);
        }

        real localMinimum = std::numeric_limits<real>::max();
        real localMaximum = 0;
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const real value = candidate[static_cast<std::size_t>(iNode)];
            _localTimeSteps[iNode](0, 0) = value;
            localMinimum = std::min(localMinimum, value);
            localMaximum = std::max(localMaximum, value);
        }
        MPI_Allreduce(&localMinimum, &_lastMinimumTimeStep, 1,
                      DNDS_MPI_REAL, MPI_MIN, _mpi.comm);
        MPI_Allreduce(&localMaximum, &_lastMaximumTimeStep, 1,
                      DNDS_MPI_REAL, MPI_MAX, _mpi.comm);
        _localTimeSteps.trans.startPersistentPull();
        _localTimeSteps.trans.waitPersistentPull();
    }

    template <int dimension>
    void SpatialOperator<dimension>::ApplyStrongBoundaryConditions(
        NodeStatePair &state) const
    {
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const BoundaryZoneSettings *selected = nullptr;
            int selectedPriority = -1;
            Geom::t_index selectedZone = std::numeric_limits<Geom::t_index>::max();
            for (const auto &piece : _geometry.NodeVolume(iNode).boundaryPieces)
            {
                const auto &candidate = _boundaries.Get(piece.zone);
                if (!candidate.strongState ||
                    !BoundaryRegistry::IsStrongType(candidate.mode))
                    continue;
                const int priority = candidate.mode == BoundaryMode::SupersonicInlet
                                         ? 2
                                         : 1;
                if (priority > selectedPriority ||
                    (priority == selectedPriority && piece.zone < selectedZone))
                {
                    selected = &candidate;
                    selectedPriority = priority;
                    selectedZone = piece.zone;
                }
            }
            if (!selected)
                continue;

            if (selected->mode == BoundaryMode::SupersonicInlet)
            {
                state[iNode] = PrimitiveToConservative(selected->primitive);
                continue;
            }

            State primitive = ConservativeToPrimitive(State(state[iNode]));
            for (int i = 0; i < dimension; i++)
                primitive(1 + i) =
                    selected->wallVelocity[static_cast<std::size_t>(i)];
            if (selected->mode == BoundaryMode::NoSlipIsothermalWall)
                primitive(dimension + 1) =
                    primitive(0) * _physics.viscous.gasConstant *
                    selected->wallTemperature;
            state[iNode] = PrimitiveToConservative(
                std::vector<real>(primitive.data(),
                                  primitive.data() + primitive.size()));
        }
    }

    template <int dimension>
    real SpatialOperator<dimension>::EvaluateRHS(
        NodeStatePair &means,
        NodeStatePair &rhs)
    {
        Reconstruct(means);

        for (index iEdge = 0; iEdge < _topology.NumEdge(); iEdge++)
        {
            _edgeFlux[iEdge] = EvaluateOwnedEdgeFlux(iEdge);
            _edgeSpectralRadius[iEdge](0, 0) =
                EvaluateOwnedEdgeSpectralRadius(iEdge);
        }
        _edgeFlux.trans.startPersistentPull();
        _edgeSpectralRadius.trans.startPersistentPull();
        _edgeFlux.trans.waitPersistentPull();
        _edgeSpectralRadius.trans.waitPersistentPull();
        UpdateLocalTimeSteps();

        real localSquared = 0;
        real localVolume = 0;
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            State residual = EvaluateOwnedBoundaryFlux(iNode);
            for (const auto &incidence : _topology.Node2Edge(iNode))
                residual += incidence.outwardSign * State(_edgeFlux[incidence.edge]);
            const real volume = _geometry.NodeVolume(iNode).moments.measure;
            rhs[iNode] = -residual / volume;
        }
        if (_periodic)
            _periodic->Average(rhs);
        for (index iNode = 0; iNode < _mesh->NumNode(); iNode++)
        {
            const real volume = _geometry.NodeVolume(iNode).moments.measure;
            localSquared += volume * rhs[iNode].squaredNorm();
            localVolume += volume;
        }

        real globalSquared = 0;
        real globalVolume = 0;
        MPI_Allreduce(&localSquared, &globalSquared, 1,
                      DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
        MPI_Allreduce(&localVolume, &globalVolume, 1,
                      DNDS_MPI_REAL, MPI_SUM, _mpi.comm);
        return std::sqrt(globalSquared /
                         std::max(globalVolume, verySmallReal));
    }

    template class SpatialOperator<2>;
    template class SpatialOperator<3>;
}
