#include "NCFVPeriodic.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace DNDS::NCFV
{
    PeriodicNodes::PeriodicNodes(
        const MPIInfo &mpi, const ssp<Geom::UnstructuredMesh> &mesh,
        const DualGeometry &geometry, const MeshSettings &settings)
        : _mpi(mpi), _mesh(mesh), _geometry(geometry)
    {
        for (int d = 0; d < 3; d++)
            _lengths(d) = settings.periodicLengths.at(d);
        _counts.resize(mpi.size);
        _offsets.resize(mpi.size + 1, 0);
        const int count = static_cast<int>(mesh->NumNode());
        MPI_Allgather(&count, 1, MPI_INT, _counts.data(), 1, MPI_INT, mpi.comm);
        for (int r = 0; r < mpi.size; r++)
            _offsets[r + 1] = _offsets[r] + _counts[r];
    }

    Eigen::MatrixXd PeriodicNodes::Gather(const ArrayDof<DynamicSize, 1> &field) const
    {
        const int nVars = field.father->MatRowSize();
        Eigen::MatrixXd local(nVars, _mesh->NumNode());
        for (index i = 0; i < _mesh->NumNode(); i++)
            local.col(i) = field[i];
        Eigen::MatrixXd global(nVars, _offsets.back());
        std::vector<int> counts(_mpi.size), offsets(_mpi.size);
        for (int r = 0; r < _mpi.size; r++)
        {
            counts[r] = _counts[r] * nVars;
            offsets[r] = _offsets[r] * nVars;
        }
        MPI_Allgatherv(local.data(), static_cast<int>(local.size()), DNDS_MPI_REAL,
                       global.data(), counts.data(), offsets.data(), DNDS_MPI_REAL, _mpi.comm);
        return global;
    }

    void PeriodicNodes::Build(const Topology &topology, real tolerance)
    {
        const index nGlobal = _offsets.back();
        constexpr int geometryFields = 19;
        Eigen::MatrixXd local(geometryFields, _mesh->NumNode()), global(geometryFields, nGlobal);
        for (index i = 0; i < _mesh->NumNode(); i++)
        {
            local.col(i).head<3>() = _mesh->coords[i];
            const auto &m = _geometry.NodeVolume(i).moments;
            const Vector3 x = _mesh->coords[i];
            const Vector3 first = m.first - m.measure * x;
            const Matrix3 second = m.second - x * m.first.transpose() -
                                   m.first * x.transpose() + m.measure * x * x.transpose();
            local(3, i) = m.measure;
            local.col(i).segment<3>(4) = first;
            local.col(i).segment<6>(7) << second(0, 0), second(1, 1), second(2, 2),
                second(0, 1), second(1, 2), second(2, 0);
            // Node-relative bounds already share the same frame after translation.
            // Do not union absolute coordinates across opposite periodic faces.
            local.col(i).segment<3>(13) = _geometry.NodeVolume(i).lowerOffset;
            local.col(i).segment<3>(16) = _geometry.NodeVolume(i).upperOffset;
        }
        std::vector<int> counts(_mpi.size), offsets(_mpi.size);
        for (int r = 0; r < _mpi.size; r++)
        {
            counts[r] = geometryFields * _counts[r];
            offsets[r] = geometryFields * _offsets[r];
        }
        MPI_Allgatherv(local.data(), static_cast<int>(local.size()), DNDS_MPI_REAL,
                       global.data(), counts.data(), offsets.data(), DNDS_MPI_REAL, _mpi.comm);
        Vector3 lower, upper;
        for (int d = 0; d < 3; d++)
        {
            lower(d) = global.row(d).minCoeff();
            upper(d) = global.row(d).maxCoeff();
            DNDS_check_throw_info(std::abs(upper(d) - lower(d) - _lengths(d)) < tolerance * 10,
                                  "NCFV periodic box length differs from mesh bounds");
        }
        _representative.resize(nGlobal);
        _coordinates.resize(nGlobal);
        _centeredMoments.assign(nGlobal, RawMoments{});
        _referenceLengths.assign(nGlobal, Vector3::Ones());
        std::vector<Vector3> lowerOffsets(nGlobal, Vector3::Zero());
        std::vector<Vector3> upperOffsets(nGlobal, Vector3::Zero());
        _partialVolumes.resize(nGlobal);
        std::map<std::array<long long, 3>, index> keys;
        std::vector<int> multiplicity(nGlobal, 0);
        for (index i = 0; i < nGlobal; i++)
        {
            Vector3 x = global.col(i).head<3>();
            std::array<long long, 3> key{};
            for (int d = 0; d < 3; d++)
            {
                if (std::abs(x(d) - upper(d)) < tolerance * 10)
                    x(d) = lower(d);
                key[d] = std::llround((x(d) - lower(d)) / tolerance);
            }
            auto [found, inserted] = keys.emplace(key, i);
            const index rep = found->second;
            _representative[i] = rep;
            _coordinates[i] = x;
            multiplicity[rep]++;
            auto &m = _centeredMoments[rep];
            _partialVolumes[i] = global(3, i);
            m.measure += global(3, i);
            m.first += global.col(i).segment<3>(4);
            Matrix3 second;
            second << global(7, i), global(10, i), global(12, i),
                global(10, i), global(8, i), global(11, i),
                global(12, i), global(11, i), global(9, i);
            m.second += second;
            lowerOffsets[rep] = lowerOffsets[rep].cwiseMin(global.col(i).segment<3>(13));
            upperOffsets[rep] = upperOffsets[rep].cwiseMax(global.col(i).segment<3>(16));
        }
        for (index i = 0; i < nGlobal; i++)
            if (_representative[i] == i)
            {
                _referenceLengths[i] = DualGeometry::ReferenceLengthsFromBounds(
                    lowerOffsets[i], upperOffsets[i], _mesh->getDim());
                int expected = 1;
                for (int d = 0; d < 3; d++)
                    if (std::abs(_coordinates[i](d) - lower(d)) < tolerance * 10)
                        expected *= 2;
                DNDS_check_throw_info(multiplicity[i] == expected,
                                      "NCFV periodic faces do not have matching translated nodes");
            }

        std::vector<index> localEdges;
        for (index e = 0; e < topology.NumEdge(); e++)
            for (int d = 0; d < 2; d++)
                localEdges.push_back(Representative(topology.Edge2Node()(e, d)));
        const int localCount = static_cast<int>(localEdges.size());
        MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, _mpi.comm);
        offsets[0] = 0;
        for (int r = 1; r < _mpi.size; r++)
            offsets[r] = offsets[r - 1] + counts[r - 1];
        std::vector<index> globalEdges(offsets.back() + counts.back());
        MPI_Allgatherv(localEdges.data(), localCount, DNDS_MPI_INDEX,
                       globalEdges.data(), counts.data(), offsets.data(), DNDS_MPI_INDEX, _mpi.comm);
        std::vector<std::set<index>> graph(nGlobal);
        for (std::size_t i = 0; i < globalEdges.size(); i += 2)
        {
            const index a = globalEdges[i], b = globalEdges[i + 1];
            DNDS_check_throw_info(a != b, "NCFV periodic direction needs more than one mesh layer");
            graph[a].insert(b);
            graph[b].insert(a);
        }
        _graph.resize(nGlobal);
        for (index i = 0; i < nGlobal; i++)
            _graph[i].assign(graph[i].begin(), graph[i].end());
        if (_mpi.rank == 0)
            log() << "NCFV periodic quotient: " << nGlobal << " mesh nodes -> "
                  << keys.size() << " dual unknowns, lengths=" << _lengths.transpose()
                  << "; replicated directory/Allgatherv communication" << std::endl;
    }

    Vector3 PeriodicNodes::Displacement(index from, index to) const
    {
        Vector3 delta = _coordinates[to] - _coordinates[from];
        for (int d = 0; d < 3; d++)
            delta(d) -= _lengths(d) * std::round(delta(d) / _lengths(d));
        return delta;
    }

    RawMoments PeriodicNodes::Moments(index representative, const Vector3 &shift) const
    {
        RawMoments m = _centeredMoments.at(representative);
        m.second += shift * m.first.transpose() + m.first * shift.transpose() +
                    m.measure * shift * shift.transpose();
        m.first += m.measure * shift;
        return m;
    }

    void PeriodicNodes::Average(ArrayDof<DynamicSize, 1> &field) const
    {
        const Eigen::MatrixXd global = Gather(field);
        Eigen::MatrixXd sums = Eigen::MatrixXd::Zero(global.rows(), global.cols());
        for (index i = 0; i < global.cols(); i++)
            sums.col(_representative[i]) += _partialVolumes[i] * global.col(i);
        for (index i = 0; i < _mesh->NumNode(); i++)
            field[i] = sums.col(Representative(i)) / Volume(i);
    }
}
