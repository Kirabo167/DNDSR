/**
 * @file NCFVBoundary.hpp
 * @brief CGNS-name to boundary-condition registry private to NCFV.
 */
#pragma once

#include "NCFVConfig.hpp"
#include "NCFVTopology.hpp"

#include <unordered_map>

namespace DNDS::NCFV
{
    /**
     * @brief Resolves user boundary names after CGNS assigns integer zone IDs.
     *
     * The registry intentionally does not alter Geom's boundary tables.  It
     * owns a normalized copy of every NCFV setting and provides a default
     * for unlisted zones when strict coverage is disabled.
     */
    class BoundaryRegistry
    {
        MPIInfo _mpi;
        int _dimension = 0;
        bool _requireCoverage = false;
        BoundaryZoneSettings _default;
        std::unordered_map<Geom::t_index, BoundaryZoneSettings> _settings;
        std::unordered_map<Geom::t_index, std::string> _idToName;

        BoundaryZoneSettings Normalize(BoundaryZoneSettings settings) const;

    public:
        BoundaryRegistry(
            const MPIInfo &mpi,
            int dimension,
            const PhysicsSettings &physics);

        void Build(
            const std::unordered_map<std::string, Geom::t_index> &nameToID,
            const Topology &topology,
            const PhysicsSettings &physics);

        [[nodiscard]] const BoundaryZoneSettings &Get(Geom::t_index zone) const;
        [[nodiscard]] std::string Name(Geom::t_index zone) const;
        [[nodiscard]] bool HasExplicit(Geom::t_index zone) const
        {
            return _settings.count(zone) != 0;
        }

        [[nodiscard]] static bool IsWall(BoundaryMode mode);
        [[nodiscard]] static bool IsStrongType(BoundaryMode mode);
    };
}
