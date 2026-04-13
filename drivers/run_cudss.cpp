/**
 * cuDSS Driver
 *
 * Author(s): Ganesh Arivoli
 * Email(s): arivoli@wisc.edu
 *
 * Usage: ./build/run_cudss --precision <float|double> --rigs <num_rigs> [--iterations <count>]
 *
 * This driver parses the fixed-order CLI for the cuDSS backend,
 * validates the selected multi-rig case, and dispatches either a
 * single solve or repeated solve timing via the iterations flag.
 */

#include <iostream>
#include <string>

#include "cudss_solver.h"
#include "utils.h"

namespace {

struct DriverArgs
{
    Precision precision = Precision::Float64;
    int num_rigs = 4;
    int iterations = 1;
};

void printUsage(const char *program_name)
{
    std::cout << "Usage: " << program_name
              << " --precision <float|double> --rigs <num_rigs> [--iterations <count>]" << std::endl;
    std::cout << "Supported num_rigs values: " << supportedMultiRigValues() << std::endl;
}

bool isHelpRequest(int argc, char *argv[])
{
    return argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
}

bool parseArgs(int argc, char *argv[], DriverArgs &args)
{
    if ((argc != 5 && argc != 7) || std::string(argv[1]) != "--precision" || std::string(argv[3]) != "--rigs")
    {
        return false;
    }

    if (!tryParsePrecision(argv[2], args.precision))
    {
        std::cerr << "Error: --precision expects 'float' or 'double'" << std::endl;
        return false;
    }

    if (!tryParsePositiveInt(argv[4], args.num_rigs))
    {
        std::cerr << "Error: --rigs expects a positive integer" << std::endl;
        return false;
    }

    if (argc == 7)
    {
        if (std::string(argv[5]) != "--iterations")
        {
            return false;
        }

        if (!tryParsePositiveInt(argv[6], args.iterations))
        {
            std::cerr << "Error: --iterations expects a positive integer" << std::endl;
            return false;
        }
    }

    return true;
}

}  // namespace

int main(int argc, char *argv[])
{
    try
    {
        if (isHelpRequest(argc, argv))
        {
            printUsage(argv[0]);
            return 0;
        }

        DriverArgs args;
        if (!parseArgs(argc, argv, args))
        {
            printUsage(argv[0]);
            return 1;
        }

        if (!isSupportedMultiRigCount(args.num_rigs))
        {
            std::cerr << "Error: Unsupported num_rigs value: " << args.num_rigs << ". Supported values are "
                      << supportedMultiRigValues() << "." << std::endl;
            return 1;
        }

        const bool loop_mode = args.iterations > 1;
        const ProblemFiles files =
            getMultiRigCaseFiles(args.num_rigs, loop_mode ? "cudss_loop" : "cudss", args.precision);
        const CudssRunOptions options{
            args.num_rigs,
            args.iterations,
            !loop_mode,
            loop_mode ? "output/logs/cudss_loop_timing.csv" : "output/logs/cudss_timing.csv",
        };

        if (args.precision == Precision::Float64)
        {
            return runCudss<double>(files, options);
        }

        return runCudss<float>(files, options);
    }
    catch (const std::exception &error_message)
    {
        std::cerr << "Error: " << error_message.what() << std::endl;
        return 1;
    }
}
