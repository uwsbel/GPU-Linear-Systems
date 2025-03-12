#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "linear_sys.h"
#include <omp.h>

// Generate a symmetric positive definite matrix
void generateSPDMatrix(std::vector<std::vector<double>>& A, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 1.0);
    
    // Start with a random matrix
    std::vector<std::vector<double>> temp(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp[i][j] = dis(gen);
        }
    }
    
    // Create a symmetric matrix by multiplying A * A^T
    // This guarantees symmetry
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                A[i][j] += temp[i][k] * temp[j][k]; // Matrix multiplication A * A^T
            }
        }
    }
    
    // Add a small value to the diagonal to ensure positive definiteness
    // This prevents the matrix from being singular
    for (int i = 0; i < n; i++) {
        A[i][i] += n;
    }
}

// Generate a vector b = A*x where x is a known vector
// This ensures the system has a known solution
void generateConsistentVector(const std::vector<std::vector<double>>& A, 
                             std::vector<double>& B, 
                             const std::vector<double>& knownSolution, 
                             int n) {
    for (int i = 0; i < n; i++) {
        B[i] = 0.0;
        for (int j = 0; j < n; j++) {
            B[i] += A[i][j] * knownSolution[j];
        }
    }
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>" << std::endl;
        return 1;
    }
    
    int n = std::stoi(argv[1]);
    int num_threads = std::stoi(argv[2]);
    
    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);
    
    // Generate a symmetric positive definite matrix
    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    generateSPDMatrix(A, n);
    
    // Create a known solution vector (all 1's for simplicity)
    std::vector<double> knownSolution(n, 1.0);
    
    // Generate right-hand side vector consistent with the known solution
    std::vector<double> B(n);
    generateConsistentVector(A, B, knownSolution, n);
    
    // Perform LU decomposition and solve
    std::vector<std::vector<double>> L(n, std::vector<double>(n));
    std::vector<std::vector<double>> U(n, std::vector<double>(n));
    
    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Call LU decomposition (implementation varies: naive or MKL)
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