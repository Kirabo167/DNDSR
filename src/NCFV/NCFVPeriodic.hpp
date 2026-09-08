#pragma once

#include "NCFVDualGeometry.hpp"
#include "DNDS/ArrayDOF.hpp"

namespace DNDS::NCFV
{
    /** Translational periodic quotient of the original nodal dual-volume pieces.
     * Geometry is kept in each original Euclidean frame. Only state, centered
     * moments and residuals are identified. The first implementation replicates
     * the quotient directory and uses Allgatherv; edge flux work remains distributed.
     */
    class PeriodicNodes
    {
        MPIInfo _mpi;
        ssp<Geom::UnstructuredMesh> _mesh;
        const DualGeometry &_geometry;
        Vector3 _lengths;
        std::vector<int> _counts, _offsets;
        std::vector<index> _representative;
        std::vector<Vector3> _coordinates;
        std::vector<RawMoments> _centeredMoments;
        std::vector<Vector3> _referenceLengths;
        std::vector<real> _partialVolumes;
        std::vector<std::vector<index>> _graph;

    public:
        PeriodicNodes(const MPIInfo &mpi, const ssp<Geom::UnstructuredMesh> &mesh,
                      const DualGeometry &geometry, const MeshSettings &settings);
        void Build(const Topology &topology, real tolerance);
        [[nodiscard]] index Representative(index local) const
        {
            return _representative.at(_mesh->NodeIndexLocal2Global(local));
        }
        [[nodiscard]] const auto &Graph() const { return _graph; }
        [[nodiscard]] Vector3 Displacement(index from, index to) const;
        [[nodiscard]] RawMoments Moments(index representative, const Vector3 &shift) const;
        [[nodiscard]] real Volume(index local) const
        {
            return _centeredMoments.at(Representative(local)).measure;
        }
        [[nodiscard]] const Vector3 &ReferenceLengths(index local) const
        {
            return _referenceLengths.at(Representative(local));
        }
        [[nodiscard]] Eigen::MatrixXd Gather(const ArrayDof<DynamicSize, 1> &field) const;
        void Average(ArrayDof<DynamicSize, 1> &field) const;
        [[nodiscard]] const Vector3 &Lengths() const { return _lengths; }
    };
}
