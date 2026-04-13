/**
 * Eigen PARDISO Driver
 *
 * Author(s): Ganesh Arivoli
 * Email(s): arivoli@wisc.edu
 *
 * Usage: ./build/run_eigen_pardiso --threads <num_threads> --rigs <num_rigs>
 *
 * This driver parses the fixed-order CLI for the Eigen PARDISO backend,
 * validates the selected multi-rig case, and runs the solve path for
 * the double-precision reference comparison workflow.
 */
#define EIGEN_USE_MKL_ALL

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/PardisoSupport>
#include <Eigen/Sparse>
#include <mkl.h>

#include "utils.h"

namespace {

struct DriverArgs
{
    int num_threads = 1;
    int num_rigs = 4;
};

Eigen::SparseMatrix<double> readMatrix(const std::string &filename)
{
    const CsrMatrix<double> matrix_data = readMatrixCSR<double>(filename);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(matrix_data.values.size());

    for (int row = 0; row < matrix_data.n; ++row)
    {
        for (int j = matrix_data.row_offsets[row]; j < matrix_data.row_offsets[row + 1]; ++j)
        {
            triplets.emplace_back(row, matrix_data.columns[j], matrix_data.values[j]);
        }
    }

    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix(matrix_data.n, matrix_data.n);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

void writeEigenVectorToFile(const Eigen::VectorXd &vector, const std::string &filename)
{
    std::vector<double> vec(vector.data(), vector.data() + vector.size());
    writeVectorToFile<double>(vec, filename);
}

void printUsage(const char *program_name)
{
    std::cerr << "Usage: " << program_name << " --threads <num_threads> --rigs <num_rigs>" << std::endl;
    std::cerr << "Supported num_rigs values: " << supportedMultiRigValues() << std::endl;
}

bool isHelpRequest(int argc, char *argv[])
{
    return argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
}

bool parseArgs(int argc, char *argv[], DriverArgs &args)
{
    if (argc != 5 || std::string(argv[1]) != "--threads" || std::string(argv[3]) != "--rigs")
    {
        return false;
    }

    if (!tryParsePositiveInt(argv[2], args.num_threads))
    {
        std::cerr << "Error: --threads expects a positive integer" << std::endl;
        return false;
    }

    if (!tryParsePositiveInt(argv[4], args.num_rigs))
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

        mkl_set_num_threads(args.num_threads);

        const ProblemFiles files = getMultiRigCaseFiles(args.num_rigs, "eigen_pardiso", Precision::Float64);
        Eigen::SparseMatrix<double> A = readMatrix(files.matrix);
        const std::vector<double> b_values = readVector<double>(files.rhs);
        const std::vector<double> known_solution_values = readKnownSolution<double>(files.dv, files.dl);
        Eigen::VectorXd b = Eigen::Map<const Eigen::VectorXd>(b_values.data(), b_values.size());
        Eigen::VectorXd known_solution =
            Eigen::Map<const Eigen::VectorXd>(known_solution_values.data(), known_solution_values.size());

        std::cout << "Matrix A dimensions: " << A.rows() << " x " << A.cols() << std::endl;
        std::cout << "Vector b size: " << b.size() << std::endl;
        std::cout << "Known solution size: " << known_solution.size() << std::endl;

        const int n = A.rows();
        if (A.cols() != n || b.size() != n || known_solution.size() != n)
        {
            std::cerr << "Error: Matrix and vector dimensions are inconsistent" << std::endl;
            return 1;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        Eigen::PardisoLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        Eigen::VectorXd x = solver.solve(b);
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();

        const double relative_error = (x - known_solution).norm() / known_solution.norm();
        std::cout << "First element: " << x(0) << std::endl;
        std::cout << "Last element: " << x(n - 1) << std::endl;
        std::cout << "Relative Error: " << relative_error << std::endl;
        std::cout << "Time (ms): " << elapsed_ms << std::endl;

        writeEigenVectorToFile(x, files.output);
        return 0;
    }
    catch (const std::exception &error_message)
    {
        std::cerr << "Error: " << error_message.what() << std::endl;
        return 1;
    }
}
