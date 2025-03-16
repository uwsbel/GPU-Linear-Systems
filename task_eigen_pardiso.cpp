#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <Eigen/Sparse>
#include <Eigen/PardisoSupport>
#include <mkl.h>

// Function to read matrix from file in COO format
Eigen::SparseMatrix<double> readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    // Read all triplets first to determine matrix size
    std::vector<Eigen::Triplet<double>> tripletList;
    int row, col;
    double value;
    int max_row = 0, max_col = 0;
    
    // Read all entries
    while (file >> row >> col >> value) {
        // Convert from 1-based to 0-based indexing
        row--;
        col--;
        
        // Keep track of matrix dimensions
        max_row = std::max(max_row, row);
        max_col = std::max(max_col, col);
        
        tripletList.emplace_back(row, col, value);
    }
    
    // Matrix dimensions are max indices + 1 (since we converted to 0-based)
    int rows = max_row + 1;
    int cols = max_col + 1;
    
    // Initialize and fill the sparse matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix(rows, cols);
    matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    matrix.makeCompressed();
    
    file.close();
    return matrix;
}

// Function to read vector from file
Eigen::VectorXd readVector(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::vector<double> values;
    double value;
    
    // Read all values from the file
    while (file >> value) {
        values.push_back(value);
    }
    
    // Check if we read anything
    if (values.empty()) {
        std::cerr << "Warning: No data read from " << filename << std::endl;
    }
    
    // Convert to Eigen vector
    Eigen::VectorXd vector = Eigen::Map<Eigen::VectorXd>(values.data(), values.size());
    file.close();
    return vector;
}

// Function to read the known solution (combining Dl and Dv)
Eigen::VectorXd readKnownSolution(const std::string& dvFilename, const std::string& dlFilename) {
    Eigen::VectorXd dvPart = readVector(dvFilename);
    Eigen::VectorXd dlPart = readVector(dlFilename);
    
    // Negate dlPart before combining
    dlPart = -dlPart;
    
    // Create combined vector
    Eigen::VectorXd solution(dvPart.size() + dlPart.size());
    solution << dvPart, dlPart;
    
    return solution;
}

// Function to write vector to file
void writeVectorToFile(const Eigen::VectorXd& vector, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        exit(1);
    }
    
    // Set precision for output
    file.precision(16);
    file << std::scientific;
    
    // Write each element on a new line
    for (int i = 0; i < vector.size(); i++) {
        file << vector(i) << std::endl;
    }
    
    file.close();
    std::cout << "Solution written to " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <num_threads>" << std::endl;
        return 1;
    }
    
    int num_threads = std::stoi(argv[1]);
    
    // Set the number of threads for MKL
    mkl_set_num_threads(num_threads);
    
    // Data file paths
    std::string matrixFile = "data/ancf/16/solve_2002_0_Z.dat";
    std::string rhsFile = "data/ancf/16/solve_2002_0_rhs.dat";
    std::string dvFile = "data/ancf/16/solve_2002_0_Dv.dat";
    std::string dlFile = "data/ancf/16/solve_2002_0_Dl.dat";
    std::string solnFile = "soln_eigen_pardiso_16.dat";
    
    // Read matrix and vectors
    Eigen::SparseMatrix<double> A = readMatrix(matrixFile);
    Eigen::VectorXd b = readVector(rhsFile);
    Eigen::VectorXd knownSolution = readKnownSolution(dvFile, dlFile);
    
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
    std::cout << "Last element: " << x(n-1) << std::endl;
    std::cout << "Relative Error: " << error << std::endl;
    std::cout << "Time (ms): " << duration.count() << std::endl;
    
    // Write solution to file
    writeVectorToFile(x, solnFile);
    
    return 0;
} 