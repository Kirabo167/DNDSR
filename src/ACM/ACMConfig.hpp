/**
 * @file ACMConfig.hpp
 * @brief DNDS-compatible JSON configuration model and loader for the ACM preview driver.
 *
 * @details The loader follows the existing project convention: read a default JSON document,
 * apply an optional RFC 7396 merge patch, apply command-line JSON-pointer overrides, then validate.
 *
 * @author Runzhi Ma
 * @date 2026-08-31
 * @note Modifier: Runzhi Ma.
 */
#pragma once

#include "ACMTime.hpp"

#include <array>
#include <string>
#include <vector>

namespace DNDS::ACM
{
    /// Complete kernel-preview configuration read by the `acm3D` application.
    struct KernelConfiguration
    {
        Settings acmSettings;
        TimeMarchSettings timeMarchSettings;
        std::array<real, 4> leftState{1, 0, 0, 0};
        std::array<real, 4> rightState{0, 0, 0, 0};
        std::array<real, 3> unitNormal{1, 0, 0};
        int nFacesPerRank = 1;

        DNDS_DECLARE_CONFIG(KernelConfiguration)
        {
            config.field_section(&T::acmSettings, "acmSettings", "Constant-density ACM settings");
            config.field_section(&T::timeMarchSettings, "timeMarchSettings", "ACM pseudo-time integration settings");
            DNDS_FIELD(leftState, "Preview left state [u,v,w,p]");
            DNDS_FIELD(rightState, "Preview right state [u,v,w,p]");
            DNDS_FIELD(unitNormal, "Preview unit normal");
            DNDS_FIELD(nFacesPerRank, "Preview faces evaluated on each MPI rank", DNDS::Config::range(1));
            config.post_read([](T &configuration)
                             { configuration.Validate(); });
        }

        /**
         * @brief Validate nested ACM settings, preview states, normal, and face count.
         * @throws std::runtime_error If any configured value is invalid or unsupported.
         */
        void Validate() const;

        /**
         * @brief Convert the JSON-compatible left-state array to the fixed-size Eigen state type.
         * @return Left preview state `[u,v,w,p]`.
         */
        State LeftState() const;

        /**
         * @brief Convert the JSON-compatible right-state array to the fixed-size Eigen state type.
         * @return Right preview state `[u,v,w,p]`.
         */
        State RightState() const;

        /**
         * @brief Convert the JSON-compatible normal array to the ACM geometry-vector type.
         * @return Configured face-normal vector; validation ensures it has non-zero length.
         */
        Vector3 UnitNormal() const;
    };

    /// Validated typed configuration paired with its normalized JSON representation.
    struct LoadedConfiguration
    {
        KernelConfiguration configuration;
        nlohmann::ordered_json resolvedJson;
    };

    /**
     * @brief Load, merge, override, deserialize, and validate an ACM configuration.
     * @param defaultJsonName Path to the required baseline JSON configuration.
     * @param jsonMergeName Optional path to an RFC 7396 merge-patch document.
     * @param overwriteKeys JSON-pointer paths supplied by command-line `-k` options.
     * @param overwriteValues Values paired with `overwriteKeys`; valid JSON text is parsed, while
     * non-JSON text is stored as a string.
     * @return Validated typed configuration and normalized resolved JSON.
     * @throws std::runtime_error If files cannot be read, override counts differ, or validation fails.
     */
    LoadedConfiguration LoadConfiguration(
        const std::string &defaultJsonName,
        const std::string &jsonMergeName = "",
        const std::vector<std::string> &overwriteKeys = {},
        const std::vector<std::string> &overwriteValues = {});
}
