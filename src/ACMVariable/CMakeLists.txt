# Independent variable-density ACM assembly. Author: Runzhi Ma.
set(LIBNAME acmVariable)
set(CPPS ACM.cpp ACMBDF2.cpp ACMTime.cpp ACMTurbulence.cpp acm2D.cpp acm3D.cpp)
set(LINKS cfv geom dnds ${DNDS_EXTERNAL_LIBS})
dnds_add_lib(${LIBNAME} "${CPPS}" "${LINKS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
