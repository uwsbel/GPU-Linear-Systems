#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include "linear_sys.h"
#include <omp.h>

// Function to read matrix from file in COO format
std::vector<std::vector<double>> readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    // First line contains dimensions and nnz (non-zero entries)
    int rows, cols, nnz;
    file >> rows >> cols >> nnz;

    // Initialize matrix with zeros
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));
    
    // Read triplets (row, column, value)
    int row, col;
    double value;
    for (int i = 0; i < nnz; i++) {
        if (!(file >> row >> col >> value)) {
            std::cerr << "Error reading COO triplet at line " << i+2 << std::endl;
            exit(1);
        }
        // Check if indices are 0-based or 1-based
        // Assuming 0-based; adjust if 1-based
        matrix[row][col] = value;
    }
    
    file.close();
    return matrix;
}

// Function to read vector from file
std::vector<double> readVector(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::vector<double> vector;
    double value;
    
    // Read all values from the file
    while (file >> value) {
        vector.push_back(value);
    }
    
    // Check if we read anything
    if (vector.empty()) {
        std::cerr << "Warning: No data read from " << filename << std::endl;
    }
    
    file.close();
    return vector;
}

// Function to read the known solution (combining Dl and Dv)
std::vector<double> readKnownSolution(const std::string& dlFilename, const std::string& dvFilename) {
    std::vector<double> dlPart = readVector(dlFilename);
    std::vector<double> dvPart = readVector(dvFilename);
    
    // Combine the two vectors
    std::vector<double> solution = dlPart;
    solution.insert(solution.end(), dvPart.begin(), dvPart.end());
    
    return solution;
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <num_threads>" << std::endl;
        return 1;
    }
    
    int num_threads = std::stoi(argv[1]);
    
    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);
    
    // Data file paths
    std::string matrixFile = "data/ancf/solve_2002_0_Z.dat";
    std::string rhsFile = "data/ancf/solve_2002_0_rhs.dat";
    std::string dlFile = "data/ancf/solve_2002_0_Dl.dat";
    std::string dvFile = "data/ancf/solve_2002_0_Dv.dat";
    
    // Read matrix and vectors
    std::vector<std::vector<double>> A = readMatrix(matrixFile);
    std::vector<double> B = readVector(rhsFile);
    std::vector<double> knownSolution = readKnownSolution(dlFile, dvFile);
    
    // Print sizes for debugging
    std::cout << "Matrix A dimensions: " << A.size() << " x " << (A.empty() ? 0 : A[0].size()) << std::endl;
    std::cout << "Vector B size: " << B.size() << std::endl;
    std::cout << "Known solution size: " << knownSolution.size() << std::endl;
    
    int n = A.size();
    
    // Check dimensions for consistency
    if (A[0].size() != n || B.size() != n || knownSolution.size() != n) {
        std::cerr << "Error: Matrix and vector dimensions are inconsistent" << std::endl;
        return 1;
    }
    
    // Perform LU decomposition and solve
    std::vector<std::vector<double>> L(n, std::vector<double>(n));
    std::vector<std::vector<double>> U(n, std::vector<double>(n));
    
    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Call LU decomposition
    luDecomposition(A, L, U);
    
    // Solve Ax = B using LU decomposition
    std::vector<double> X = solveLU(L, U, B);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    // Calculate error compared to known solution
    double error = 0.0;
    for (int i = 0; i < n; i++) {
        error += std::abs(X[i] - knownSolution[i]);
    }
    
    // Output first and last elements for verification, plus error
    std::cout << X[0] << std::endl;
    std::cout << X[n-1] << std::endl;
    std::cout << "Error: " << error / n << std::endl;
    std::cout << duration.count() << std::endl;
    
    return 0;
} 