/**
 * @file acm2D.cpp
 * @brief Explicit template instantiation for the two-dimensional high-order ACM solver.
 * @author Runzhi Ma
 * @date 2026-09-03
 * @note Modifier: Runzhi Ma.
 */
#include "CFV/VariationalReconstruction_Reconstruction.hxx"
#include "CFV/VariationalReconstruction_LimiterProcedure.hxx"

#include "ACMEvaluator.hxx"
#include "ACMTurbulenceTransport.hxx"
#include "ACMSolver.hxx"
#include "ACMPhysicalTime.hxx"

namespace DNDS::ACMVariable
{
    template class ACMEvaluator<2>;
    template class ACMTurbulenceTransport<2>;
    template class ACMSolver<ACMModel::VariableDensity2D>;
}
