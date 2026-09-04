/**
 * @file ACMPeriodic.hpp
 * @brief Separate spatial-vector rotation from conservative momentum-column rotation.
 * @author Runzhi Ma
 * @date 2026-09-03
 */
#pragma once
#include "Geom/Mesh/Mesh.hpp"

namespace DNDS::ACMVariable
{
    /**
     * @brief Map rows of spatial vectors into the opposite periodic face frame.
     * @tparam dim Spatial dimension, 2 or 3.
     * @tparam Matrix Dense matrix type with dim columns.
     * @param mesh Mesh owning the periodic rotation map.
     * @param side Face side (0 owner, 1 neighbor).
     * @param zone Face zone identifying periodic direction.
     * @param vectors In/out matrix of spatial row vectors, NOT conservative variables.
     * @note Momentum indices [1,2,3] must never index a dim-column spatial tensor.
     */
    template <int dim, class Matrix>
    void TransformSpatialGradient(const ssp<Geom::UnstructuredMesh> &mesh,
                                  int side, Geom::t_index zone, Matrix &vectors)
    {
        if (!mesh->isPeriodic) return;
        if ((side == 1 && Geom::FaceIDIsPeriodicMain(zone)) ||
            (side == 0 && Geom::FaceIDIsPeriodicDonor(zone)))
            vectors = mesh->periodicInfo.template TransVector<dim, Eigen::Dynamic>(
                vectors.transpose(), zone).transpose().eval();
        if ((side == 1 && Geom::FaceIDIsPeriodicDonor(zone)) ||
            (side == 0 && Geom::FaceIDIsPeriodicMain(zone)))
            vectors = mesh->periodicInfo.template TransVectorBack<dim, Eigen::Dynamic>(
                vectors.transpose(), zone).transpose().eval();
    }
}
