/**
 * @file ACM.cpp
 * @brief Compact default ACM driver; the default executable selects the 3-D model.
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#include "ACM/SingleBlockApp.hpp"

/**
 * @brief Initialize MPI and launch the default three-dimensional ACM application.
 * @param argc Command-line argument count.
 * @param argv Command-line argument array.
 * @return Process exit code.
 */
int main(int argc, char *argv[])
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const int errorCode = DNDS::ACM::RunSingleBlockConsoleApp<
        DNDS::ACM::ACMModel::ConstantDensity3D>(argc, argv);
    if (errorCode != 0)
        MPI_Abort(MPI_COMM_WORLD, errorCode);
    MPI_Finalize();
    return errorCode;
}
