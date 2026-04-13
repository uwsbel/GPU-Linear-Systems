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
    if (argc != 7 || std::string(argv[1]) != "--threads" || std::string(argv[3]) != "--precision" ||
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

        if (!isSupportedMultiRigCount(args.num_rigs))
        {
            std::cerr << "Error: Unsupported num_rigs value: " << args.num_rigs << ". Supported values are "
                      << supportedMultiRigValues() << "." << std::endl;
            return 1;
        }

        const ProblemFiles files = getMultiRigCaseFiles(args.num_rigs, "pardiso", args.precision);
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
