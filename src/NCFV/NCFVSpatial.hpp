/**
 * @file NCFVSpatial.hpp
 * @brief Euler/Navier--Stokes residuals for both third-order NCFV modes.
 */
#pragma once

#include "NCFVBoundary.hpp"
#include "NCFVReconstruction.hpp"

#include "Euler/Gas.hpp"

namespace DNDS::NCFV
{
    template <int dimension>
    class SpatialOperator
    {
        static_assert(dimension == 2 || dimension == 3);

        using State = Eigen::Vector<real, dimension + 2>;
        using Gradient = Eigen::Matrix<real, dimension, dimension + 2>;
        using SpatialVector = Eigen::Vector<real, dimension>;

        const MPIInfo &_mpi;
        ssp<Geom::UnstructuredMesh> _mesh;
        const Topology &_topology;
        const DualGeometry &_geometry;
        const Reconstruction &_reconstruction;
        const BoundaryRegistry &_boundaries;
        IntegrationMode _mode;
        ReconstructionSettings _reconstructionSettings;
        PhysicsSettings _physics;
        TimeSettings _time;
        const PeriodicNodes *_periodic = nullptr;
        real _maximumStep = veryLargeReal;

        NodeMatrixPair _gradients;
        NodeMatrixPair _coefficients;
        NodeStatePair _pointValues;
        NodeStatePair _edgeFlux;
        NodeStatePair _edgeSpectralRadius;
        NodeStatePair _localTimeSteps;
        State _farField = State::Zero();
        real _lastMinimumTimeStep = 0;
        real _lastMaximumTimeStep = 0;

        State PrimitiveToConservative(const std::vector<real> &primitive) const;
        State ConservativeToPrimitive(const State &state) const;
        real Pressure(const State &state) const;
        real Temperature(const State &state) const;
        real MolecularViscosity(const State &state) const;
        bool IsPhysical(const State &state) const;
        State PreservePhysical(const State &candidate, const State &anchor) const;

        State PhysicalFlux(const State &state, const SpatialVector &normal) const;
        State PhysicalFluxDerivative(
            const State &state,
            const State &stateDerivative,
            int fluxDirection) const;
        State NumericalFlux(
            const State &left,
            const State &right,
            const SpatialVector &unitNormal) const;
        State BoundaryExterior(
            const State &inside,
            const SpatialVector &unitNormal,
            const BoundaryZoneSettings &boundary) const;
        State BoundaryNumericalFlux(
            const State &inside,
            const SpatialVector &unitNormal,
            const BoundaryZoneSettings &boundary) const;

        State EvaluateTraditionalState(index anchorNode, const Vector3 &point) const;
        Gradient EvaluateTraditionalGradient(index anchorNode, const Vector3 &point) const;
        Gradient EfficientSurfaceGradient(
            const std::vector<SparseScalarWeight> &weights,
            real measure) const;
        State EfficientIntegratedPhysicalFlux(
            index anchorNode,
            const Vector3 &vectorMeasure,
            const std::vector<SparseMatrixWeight> &weights) const;
        State EfficientSurfaceMean(
            index anchorNode,
            real measure,
            const std::vector<SparseVectorWeight> &weights) const;
        State InternalViscousFlux(
            const State &left,
            const State &right,
            const Gradient &leftGradient,
            const Gradient &rightGradient,
            const SpatialVector &unitNormal,
            real normalDistance) const;
        State BoundaryViscousFlux(
            const State &inside,
            const Gradient &insideGradient,
            const SpatialVector &unitNormal,
            const BoundaryZoneSettings &boundary,
            real lengthScale) const;
        real SurfaceSpectralRadius(
            const State &state,
            const SpatialVector &unitNormal,
            real measure,
            real lengthScale) const;
        State EvaluateOwnedEdgeFlux(index iEdge) const;
        real EvaluateOwnedEdgeSpectralRadius(index iEdge) const;
        State EvaluateOwnedBoundaryFlux(index iNode) const;
        real EvaluateOwnedBoundarySpectralRadius(index iNode) const;

        void AllocateNodeMatrix(NodeMatrixPair &field, const std::string &name, int rows);
        void AllocateEdgeField(NodeStatePair &field, const std::string &name, int rows);
        void Reconstruct(NodeStatePair &means);
        void UpdateLocalTimeSteps();

    public:
        SpatialOperator(
            const MPIInfo &mpi,
            const ssp<Geom::UnstructuredMesh> &mesh,
            const Topology &topology,
            const DualGeometry &geometry,
            const Reconstruction &reconstruction,
            const BoundaryRegistry &boundaries,
            IntegrationMode mode,
            const ReconstructionSettings &reconstructionSettings,
            const PhysicsSettings &physics,
            const TimeSettings &time,
            const PeriodicNodes *periodic = nullptr)
            : _mpi(mpi), _mesh(mesh), _topology(topology), _geometry(geometry),
              _reconstruction(reconstruction), _boundaries(boundaries), _mode(mode),
              _reconstructionSettings(reconstructionSettings), _physics(physics),
              _time(time), _periodic(periodic)
        {
        }

        void Initialize();
        void SetMaximumStep(real step) { _maximumStep = step; }

        /**
         * @brief Evaluate d(dual mean)/dt; performs all node and edge halo pulls.
         * @return Volume-weighted global RMS residual.
         */
        real EvaluateRHS(NodeStatePair &means, NodeStatePair &rhs);

        /** @brief Apply configured strong inlet/no-slip conditions to owned nodes. */
        void ApplyStrongBoundaryConditions(NodeStatePair &state) const;

        [[nodiscard]] real LocalTimeStep(index iNode) const
        {
            return _localTimeSteps[iNode](0, 0);
        }
        [[nodiscard]] real LastMinimumTimeStep() const { return _lastMinimumTimeStep; }
        [[nodiscard]] real LastMaximumTimeStep() const { return _lastMaximumTimeStep; }

        [[nodiscard]] const NodeMatrixPair &Gradients() const { return _gradients; }
        [[nodiscard]] const NodeMatrixPair &Coefficients() const { return _coefficients; }
        [[nodiscard]] const NodeStatePair &PointValues() const { return _pointValues; }
        [[nodiscard]] const NodeStatePair &LocalTimeSteps() const { return _localTimeSteps; }
    };

    extern template class SpatialOperator<2>;
    extern template class SpatialOperator<3>;
}
