/**
 * @file VariationalReconstruction_LimiterProcedure_3_4.cpp
 * @brief Explicitly instantiate three-dimensional CFV WBAP/CWBAP kernels for four variables.
 * @details The instantiation is equation-independent and enables the ACM `[u,v,w,p]` evaluator
 * to reuse the existing CFV limiter implementation without adding Euler dependencies.
 * @author Runzhi Ma
 * @date 2026-09-01
 * @note Modifier: Runzhi Ma.
 */
#include "../VariationalReconstruction_LimiterProcedure.hxx"

DNDS_VARIATIONALRECONSTRUCTION_LIMITERPROCEDURE_INS_EXTERN(3, 4, )
