/** @file NCFVApp.hpp @brief Command-line entry shared by runtime dimensions. */
#pragma once

#include "NCFVSolver.hpp"

#include <argparse.hpp>

#include <filesystem>
#include <iostream>

namespace DNDS::NCFV
{
    inline int RunConsoleApp(int argc, char *argv[])
    {
        MPIInfo mpi;
        mpi.setWorld();
        argparse::ArgumentParser parser("NCFV", DNDS_VERSION_STRING);
        parser.add_description(std::string("NCFV: ") + MethodName);
        parser.add_argument("config").default_value("");
        parser.add_argument("-k", "--overwrite_key")
            .append()
            .default_value<std::vector<std::string>>({});
        parser.add_argument("-v", "--overwrite_value")
            .append()
            .default_value<std::vector<std::string>>({});
        parser.add_argument("--emit-schema").flag().default_value(false);

        try
        {
            parser.parse_args(argc, argv);
            if (parser.get<bool>("--emit-schema"))
            {
                if (mpi.rank == 0)
                {
                    auto schema = Configuration::schema(
                        std::string("DNDSR NCFV (") + MethodName + ") configuration");
                    schema["$schema"] = "http://json-schema.org/draft-07/schema#";
                    std::cout << schema.dump(4) << std::endl;
                }
                return 0;
            }

            std::filesystem::path configurationPath =
                std::filesystem::path("../cases/NCFV/NCFV.json");
            const std::string requested = parser.get<std::string>("config");
            if (!requested.empty())
                configurationPath = requested;
            const auto loaded = LoadConfiguration(
                configurationPath.string(),
                parser.get<std::vector<std::string>>("--overwrite_key"),
                parser.get<std::vector<std::string>>("--overwrite_value"));

            if (loaded.configuration.dimension == 2)
            {
                Solver<2> solver(mpi, loaded.configuration);
                solver.Initialize();
                solver.Run();
            }
            else
            {
                Solver<3> solver(mpi, loaded.configuration);
                solver.Initialize();
                solver.Run();
            }
        }
        catch (const std::exception &exception)
        {
            if (mpi.rank == 0)
                std::cerr << "DNDS NCFV top-level exception: "
                          << exception.what() << std::endl;
            return 1;
        }
        return 0;
    }
}
