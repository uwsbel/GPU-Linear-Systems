// Define MKL support before including Eigen
#define EIGEN_USE_MKL_ALL

#include <vector>
#include <iostream>
// Update include path for Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/PardisoSupport>  // For PardisoLU
// Include MKL core
#include <mkl.h>
#include <omp.h>

// Define the linear_sys.h interface
void luDecomposition(const std::vector<std::vector<double>>& A, 
                    std::vector<std::vector<double>>& L, 
                    std::vector<std::vector<double>>& U) {
    
    int n = A.size();
    
    // Convert std::vector matrix to Eigen sparse matrix
    Eigen::SparseMatrix<double> eigenA(n, n);
    std::vector<Eigen::Triplet<double>> tripletList;
    
    // Reserve space for non-zero elements
    tripletList.reserve(n*n); // Can be optimized if sparsity pattern is known
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (A[i][j] != 0) {
                tripletList.push_back(Eigen::Triplet<double>(i, j, A[i][j]));
            }
        }
    }
    
    eigenA.setFromTriplets(tripletList.begin(), tripletList.end());
    eigenA.makeCompressed();
    
    // Use PardisoLU for better performance with MKL
    Eigen::PardisoLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(eigenA);
    solver.factorize(eigenA);
    
    // Unfortunately, PardisoLU also doesn't expose L and U directly in a convenient way
    // We need to extract them manually
    
    // Extract L and U as dense matrices for now
    Eigen::MatrixXd eigenL = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd eigenU = Eigen::MatrixXd::Zero(n, n);
    
    // Reconstruct L and U through testing with unit vectors
    for (int j = 0; j < n; j++) {
        Eigen::VectorXd ej = Eigen::VectorXd::Zero(n);
        ej(j) = 1.0;
        
        Eigen::VectorXd x = solver.solve(ej);
        
        for (int i = 0; i < n; i++) {
            if (i > j) {
                // L is lower triangular with diagonal = 1
                eigenL(i, j) = -x(i);
            } else if (i <= j) {
                // U is upper triangular
                eigenU(i, j) = x(i);
            }
        }
    }
    
    // Convert back to std::vector
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = eigenL(i, j);
            U[i][j] = eigenU(i, j);
        }
    }
}

std::vector<double> solveLU(const std::vector<std::vector<double>>& L, 
                           const std::vector<std::vector<double>>& U, 
                           const std::vector<double>& B) {
    
    int n = L.size();
    
    // Convert std::vector to Eigen sparse format
    Eigen::SparseMatrix<double> eigenL(n, n);
    Eigen::SparseMatrix<double> eigenU(n, n);
    Eigen::VectorXd eigenB(n);
    
    std::vector<Eigen::Triplet<double>> tripletsL, tripletsU;
    tripletsL.reserve(n*n); // Can be optimized based on sparsity
    tripletsU.reserve(n*n);
    
    for (int i = 0; i < n; i++) {
        eigenB(i) = B[i];
        for (int j = 0; j < n; j++) {
            if (L[i][j] != 0) {
                tripletsL.push_back(Eigen::Triplet<double>(i, j, L[i][j]));
            }
            if (U[i][j] != 0) {
                tripletsU.push_back(Eigen::Triplet<double>(i, j, U[i][j]));
            }
        }
    }
    
    eigenL.setFromTriplets(tripletsL.begin(), tripletsL.end());
    eigenU.setFromTriplets(tripletsU.begin(), tripletsU.end());
    eigenL.makeCompressed();
    eigenU.makeCompressed();
    
    // Forward substitution to solve Ly = B using PardisoLU
    Eigen::PardisoLU<Eigen::SparseMatrix<double>> solverL;
    solverL.compute(eigenL);
    Eigen::VectorXd y = solverL.solve(eigenB);
    
    // Back substitution to solve Ux = y using PardisoLU
    Eigen::PardisoLU<Eigen::SparseMatrix<double>> solverU;
    solverU.compute(eigenU);
    Eigen::VectorXd x = solverU.solve(y);
    
    // Convert back to std::vector
    std::vector<double> X(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        X[i] = x(i);
    }
    
    return X;
} 