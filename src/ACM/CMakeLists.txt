# @file CMakeLists.txt
# @brief Build definition for the standalone constant-density ACM library.
# @note Modifier: Runzhi Ma.

set(LIBNAME acm)

set(CPPS
    ACM.cpp
    ACMTime.cpp
    ACMTurbulence.cpp
    acm2D.cpp
    acm3D.cpp
)

set(LINKS
    cfv
    geom
    dnds
    ${DNDS_EXTERNAL_LIBS}
)

dnds_add_lib(${LIBNAME} "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}"
    OFF ON ON)
