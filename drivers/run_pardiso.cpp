/**
 * PARDISO Driver
 *
 * Author(s): Ganesh Arivoli
 * Email(s): arivoli@wisc.edu
 *
 * Usage: ./build/run_pardiso --threads <num_threads> --precision <float|double> --rigs <num_rigs>
 *
 * This driver parses the fixed-order CLI for the MKL PARDISO backend,
 * validates the selected multi-rig case, and dispatches the run in
 * single or double precision.
 */
#include <iostream>
#include <string>

#include "pardiso_solver.h"
#include "utils.h"

namespace {

struct DriverArgs
{
    int num_threads = 1;
    Precision precision = Precision::Float64;
    int num_rigs = 4;
    bool refine = false;  // interpret --rigs as a spoke count from the refine1 set
    bool shell = false;   // interpret --rigs as the grid size of the ANCF shell set
};

void printUsage(const char *program_name)
{
    std::cerr << "Usage: " << program_name
              << " --threads <num_threads> --precision <float|double> --rigs <num_rigs>" << std::endl;
    std::cerr << "Supported num_rigs values: " << supportedMultiRigValues() << std::endl;
}

bool isHelpRequest(int argc, char *argv[])
{
    return argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
}

bool parseArgs(int argc, char *argv[], DriverArgs &args)
{
    // Optional trailing --refine flag selects the single-tire mesh-refinement cases.
    int effective_argc = argc;
    if (effective_argc > 1 && std::string(argv[effective_argc - 1]) == "--refine")
    {
        args.refine = true;
        --effective_argc;
    }
    else if (effective_argc > 1 && std::string(argv[effective_argc - 1]) == "--shell")
    {
        args.shell = true;
        --effective_argc;
    }

    if (effective_argc != 7 || std::string(argv[1]) != "--threads" || std::string(argv[3]) != "--precision" ||
        std::string(argv[5]) != "--rigs")
    {
        return false;
    }

    if (!tryParsePositiveInt(argv[2], args.num_threads))
    {
        std::cerr << "Error: --threads expects a positive integer" << std::endl;
        return false;
    }

    if (!tryParsePrecision(argv[4], args.precision))
    {
        std::cerr << "Error: --precision expects 'float' or 'double'" << std::endl;
        return false;
    }

    if (!tryParsePositiveInt(argv[6], args.num_rigs))
    {
        std::cerr << "Error: --rigs expects a positive integer" << std::endl;
        return false;
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

        const bool supported = args.shell    ? isSupportedShellGrid(args.num_rigs)
                                : args.refine ? isSupportedRefineCount(args.num_rigs)
                                              : isSupportedMultiRigCount(args.num_rigs);
        if (!supported)
        {
            std::cerr << "Error: Unsupported value: " << args.num_rigs << ". Supported values are "
                      << (args.shell ? "10, 20, 30, 40, 50, 70, 100, 140, 200" : args.refine ? "16, 80" : supportedMultiRigValues()) << "."
                      << std::endl;
            return 1;
        }

        const ProblemFiles files = args.shell    ? getShellCaseFiles(args.num_rigs, "pardiso", args.precision)
                                   : args.refine ? getRefineCaseFiles(args.num_rigs, "pardiso", args.precision)
                                                 : getMultiRigCaseFiles(args.num_rigs, "pardiso", args.precision);
        const PardisoRunOptions options{args.num_threads, args.num_rigs, "output/logs/pardiso_timing.csv"};

        if (args.precision == Precision::Float32)
        {
            return runPardiso<float>(files, options);
        }

        return runPardiso<double>(files, options);
    }
    catch (const std::exception &error_message)
    {
        std::cerr << "Error: " << error_message.what() << std::endl;
        return 1;
    }
}
