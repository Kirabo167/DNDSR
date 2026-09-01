/**
 * @file acm2D.cpp
 * @brief Explicit template instantiation for the two-dimensional high-order ACM solver.
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#include "CFV/VariationalReconstruction_Reconstruction.hxx"
#include "CFV/VariationalReconstruction_LimiterProcedure.hxx"

#include "ACMEvaluator.hxx"
#include "ACMSolver.hxx"

namespace DNDS::ACM
{
    template class ACMEvaluator<2>;
    template class ACMSolver<ACMModel::ConstantDensity2D>;
}
