#include "../linear_sys.h"
#include <iostream>
#include <stdexcept>
#include <mkl.h>

void luDecomposition(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U) {
    int n = A.size();
    
    // Create a flat copy of A for MKL
    std::vector<double> flatA(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            flatA[i * n + j] = A[i][j];
        }
    }
    
    // Arrays for LAPACK
    std::vector<int> ipiv(n);
    int info;
    
    // Perform LU decomposition using MKL's LAPACK
    dgetrf(&n, &n, flatA.data(), &n, ipiv.data(), &info);
    
    if (info != 0) {
        throw std::runtime_error("LU decomposition failed in MKL");
    }
    
    // Extract L and U from the result
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) {
                L[i][j] = flatA[i * n + j];
                U[i][j] = 0.0;
            } else if (i == j) {
                L[i][j] = 1.0;
                U[i][j] = flatA[i * n + j];
            } else {
                L[i][j] = 0.0;
                U[i][j] = flatA[i * n + j];
            }
        }
    }
}

std::vector<double> solveLU(const std::vector<std::vector<double>>& L, const std::vector<std::vector<double>>& U, const std::vector<double>& B) {
    int n = L.size();
    
    // Create a combined LU matrix for MKL
    std::vector<double> flatLU(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i > j) {
                flatLU[i * n + j] = L[i][j];
            } else {
                flatLU[i * n + j] = U[i][j];
            }
        }
    }
    
    // Copy B to X (will be overwritten with the solution)
    std::vector<double> X = B;
    
    // Arrays for LAPACK
    std::vector<int> ipiv(n);
    char trans = 'N';
    int nrhs = 1;
    int info;
    
    // Generate the pivot information
    dgetrf(&n, &n, flatLU.data(), &n, ipiv.data(), &info);
    
    if (info != 0) {
        throw std::runtime_error("LU decomposition failed in MKL solve");
    }
    
    // Solve the system using the LU factorization
    dgetrs(&trans, &n, &nrhs, flatLU.data(), &n, ipiv.data(), X.data(), &n, &info);
    
    if (info != 0) {
        throw std::runtime_error("Solving system failed in MKL");
    }
    
    return X;
} 