/**
 * @file ACMSolver.hpp
 * @brief Solver-level assembly of mesh input, CFV reconstruction, ACM residuals, and pseudo-time marching.
 *
 * @details The class mirrors the high-level organization of EulerSolver while remaining a separate
 * ACM module. Existing Geom/CFV infrastructure is reused; no Euler source file is changed.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMEvaluator.hpp"
#include "ACMTime.hpp"
#include "ACMTurbulenceTransport.hpp"

#include "Geom/Mesh/Mesh_Helpers.hpp"
#include "Solver/Linear.hpp"

namespace DNDS::ACM
{
    /**
     * @brief Constant-density ACM solver for a compile-time equation model.
     * @tparam model Two- or three-dimensional constant-density ACM model.
     * @note Modifier: Runzhi Ma.
     */
    template <ACMModel model>
    class ACMSolver
    {
    public:
        using Traits = ModelTraits<model>;        ///< Compile-time ACM model traits.
        static constexpr int gDim = Traits::gDim; ///< Mesh dimension.
        using TEvaluator = ACMEvaluator<gDim>;    ///< Spatial evaluator type.
        using TDof = typename TEvaluator::TDof;   ///< Distributed state array.
        using TVFV = typename TEvaluator::TVFV;   ///< CFV reconstruction type.
        using TTurbulence = ACMTurbulenceTransport<gDim>; ///< Independent runtime RANS transport.

        /**
         * @brief Construct a solver around an MPI communicator and validated configuration.
         * @param mpi MPI communicator metadata copied into the solver.
         * @param configuration Complete ACM case configuration.
         */
        ACMSolver(const MPIInfo &mpi, const KernelConfiguration &configuration);

        /**
         * @brief Read/partition the mesh and initialize CFV reconstruction and state arrays.
         */
        void ReadMeshAndInitialize();

        /**
         * @brief Run configured SSPRK3, block-Jacobi, ACM LU-SGS, or generic-GMRES steps.
         */
        void Run();

        /**
         * @brief Access the initialized mesh for diagnostics and tests.
         * @return Shared mesh pointer.
         */
        const ssp<Geom::UnstructuredMesh> &GetMesh() const { return _mesh; }

        /**
         * @brief Access the distributed cell-mean state.
         * @return Mutable state array.
         */
        TDof &GetState() { return _u; }

        /**
         * @brief Access the optional segregated turbulence state.
         * @return Pointer to the two-entry turbulence field, or `nullptr` in Laminar mode.
         */
        typename TTurbulence::TTurbulenceDof *GetTurbulenceState()
        {
            return _turbulence ? &_turbulence->GetState() : nullptr;
        }

    private:
        MPIInfo _mpi;                                ///< MPI communicator metadata.
        KernelConfiguration _configuration;          ///< Validated runtime configuration.
        ssp<Geom::UnstructuredMesh> _mesh;           ///< Distributed volume mesh.
        ssp<Geom::UnstructuredMeshSerialRW> _reader; ///< Existing DNDS CGNS reader.
        ssp<BoundaryHandler> _boundaryHandler;       ///< Per-zone ACM boundary mapping.
        ssp<TVFV> _vfv;                              ///< Existing CFV reconstruction engine.
        ssp<TEvaluator> _evaluator;                  ///< ACM high-order spatial evaluator.
        ssp<TTurbulence> _turbulence;                ///< Optional segregated RANS transport module.
        TDof _u;                                     ///< Distributed cell-mean ACM state.
        TDof _rhs;                                   ///< Distributed raw spatial residual.
        TDof _linearRhs;                             ///< Distributed implicit defect/right-hand side.
        TDof _linearIncrement;                       ///< Distributed LU-SGS/GMRES correction.

        /**
         * @brief Copy owned distributed cell means into the rank-local time-integrator field.
         * @param source Distributed DOF array.
         * @param destination Rank-local contiguous state field.
         */
        void CopyOwnedToStateField(const TDof &source, StateField &destination) const;

        /**
         * @brief Copy a rank-local time-integrator field into owned distributed cell means.
         * @param source Rank-local contiguous state field.
         * @param destination Distributed DOF array receiving owned values.
         */
        void CopyStateFieldToOwned(const StateField &source, TDof &destination) const;

        /**
         * @brief Advance one nonlinear backward-Euler step with ACM LU-SGS or generic GMRES.
         * @param states Rank-local owned states updated in place.
         * @param pseudoTimeStep Positive local pseudo-time steps.
         * @return Nonlinear iteration and defect report.
         * @note Modifier: Runzhi Ma.
         */
        TimeStepReport AdvanceImplicitDistributed(
            StateField &states,
            const ScalarField &pseudoTimeStep);
    };

    extern template class ACMSolver<ACMModel::ConstantDensity2D>;
    extern template class ACMSolver<ACMModel::ConstantDensity3D>;
}
