#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <Eigen/Sparse>
#include <Eigen/PardisoSupport>
#include <mkl.h>
#include "utils.h"

/**
 * Reads a matrix from file in COO format and converts it to Eigen sparse format.
 * Uses the utility functions from utils.h
 */
Eigen::SparseMatrix<double> readMatrix(const std::string& filename) {
    // Read in CSR format first using our utility
    std::vector<double> values;
    std::vector<int> rowIndex;
    std::vector<int> columns;
    int n;
    readMatrixCSR(filename, values, rowIndex, columns, n);

    // Convert to Eigen format using triplets
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(values.size());

    // Loop through CSR format and create triplets
    for (int row = 0; row < n; row++) {
        for (int j = rowIndex[row]; j < rowIndex[row + 1]; j++) {
            int col = columns[j];
            double val = values[j];
            tripletList.emplace_back(row, col, val);
        }
    }

    // Initialize and fill the sparse matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix(n, n);
    matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    matrix.makeCompressed();

    return matrix;
}

/**
 * Converts a std::vector to Eigen::VectorXd
 */
Eigen::VectorXd vectorToEigen(const std::vector<double>& vec) {
    return Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size());
}

/**
 * Reads a vector from file and returns it as Eigen::VectorXd.
 * Uses the utility function from utils.h
 */
Eigen::VectorXd readVectorEigen(const std::string& filename) {
    std::vector<double> vec = readVector(filename);
    return vectorToEigen(vec);
}

/**
 * Reads the known solution using utils.h and converts to Eigen vector
 */
Eigen::VectorXd readKnownSolutionEigen(const std::string& dvFilename, const std::string& dlFilename) {
    std::vector<double> sol = readKnownSolution(dvFilename, dlFilename);
    return vectorToEigen(sol);
}

/**
 * Writes an Eigen vector to a file
 */
void writeVectorToFile(const Eigen::VectorXd& vector, const std::string& filename) {
    std::vector<double> vec(vector.data(), vector.data() + vector.size());
    writeVectorToFile(vec, filename);
}

int main(int argc, char* argv[]) {
    // Check command line arguments for num_threads (required)
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> [num_spokes]" << std::endl;
        std::cerr << "  num_threads: Number of threads for MKL (required)" << std::endl;
        std::cerr << "  num_spokes: Number of spokes for geometry (optional, default: 16)" << std::endl;
        return 1;
    }

    // Parse num_threads (first argument, required)
    int num_threads = std::stoi(argv[1]);
    if (num_threads <= 0) {
        std::cerr << "Error: num_threads must be a positive integer" << std::endl;
        return 1;
    }

    // Parse num_spokes (second argument, optional with default value of 16)
    int num_spokes = 16;  // Default value
    if (argc > 2) {
        num_spokes = std::stoi(argv[2]);
        if (num_spokes <= 0) {
            std::cerr << "Error: num_spokes must be a positive integer" << std::endl;
            return 1;
        }
    } else {
        std::cout << "No num_spokes provided. Using default value = " << num_spokes << std::endl;
    }

    // Set the number of threads for MKL
    mkl_set_num_threads(num_threads);

    // Data file paths
    std::string matrixFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Z.dat";
    std::string rhsFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_rhs.dat";
    std::string dvFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Dv.dat";
    std::string dlFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Dl.dat";
    std::string solnFile = "soln_eigen_pardiso_" + std::to_string(num_spokes) + ".dat";

    // Read matrix and vectors
    Eigen::SparseMatrix<double> A = readMatrix(matrixFile);
    Eigen::VectorXd b = readVectorEigen(rhsFile);
    Eigen::VectorXd knownSolution = readKnownSolutionEigen(dvFile, dlFile);

    // Print sizes for debugging
    std::cout << "Matrix A dimensions: " << A.rows() << " x " << A.cols() << std::endl;
    std::cout << "Vector b size: " << b.size() << std::endl;
    std::cout << "Known solution size: " << knownSolution.size() << std::endl;

    int n = A.rows();

    // Check dimensions for consistency
    if (A.cols() != n || b.size() != n || knownSolution.size() != n) {
        std::cerr << "Error: Matrix and vector dimensions are inconsistent" << std::endl;
        return 1;
    }

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Solve using PardisoLU
    Eigen::PardisoLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    Eigen::VectorXd x = solver.solve(b);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Calculate error compared to known solution
    double error = (x - knownSolution).norm() / knownSolution.norm();

    // Output first and last elements for verification, plus error
    std::cout << "First element: " << x(0) << std::endl;
    std::cout << "Last element: " << x(n - 1) << std::endl;
    std::cout << "Relative Error: " << error << std::endl;
    std::cout << "Time (ms): " << duration.count() << std::endl;

    // Write solution to file
    writeVectorToFile(x, solnFile);

    return 0;
}