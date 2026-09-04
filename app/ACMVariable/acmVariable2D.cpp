/**
 * @file acm2D.cpp
 * @brief Minimal two-dimensional ACM executable entry point.
 * @author Runzhi Ma
 * @date 2026-09-03
 * @note Modifier: Runzhi Ma.
 */
#include "ACMVariable/SingleBlockApp.hpp"

/**
 * @brief Initialize MPI and dispatch the independent two-dimensional ACM solver.
 * @param argc Command-line argument count.
 * @param argv Command-line argument array.
 * @return Process exit code; MPI is aborted collectively on solver failure.
 */
int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const int errorCode = DNDS::ACMVariable::RunSingleBlockConsoleApp<
        DNDS::ACMVariable::ACMModel::VariableDensity2D>(argc, argv);
    if (errorCode != 0)
        MPI_Abort(MPI_COMM_WORLD, errorCode);
    MPI_Finalize();
    return errorCode;
}
