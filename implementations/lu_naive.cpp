#include "../linear_sys.h"
#include <iostream>
#include <stdexcept>
#include <omp.h>

void luDecomposition(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U) {
    int n = A.size();
    
    // Initialize L and U matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[i][j] = 0.0;
            U[i][j] = 0.0;
        }
        L[i][i] = 1.0; // Diagonal of L is 1
    }
    
    for (int i = 0; i < n; i++) {
        // Upper Triangular
        #pragma omp parallel for
        for (int j = i; j < n; j++) {
            U[i][j] = A[i][j];
            for (int k = 0; k < i; k++) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }

        // Lower Triangular - this part must be sequential due to dependencies
        for (int j = i + 1; j < n; j++) {
            L[j][i] = A[j][i];
            for (int k = 0; k < i; k++) {
                L[j][i] -= L[j][k] * U[k][i];
            }
            L[j][i] /= U[i][i];
        }
    }
}

std::vector<double> solveLU(const std::vector<std::vector<double>>& L, const std::vector<std::vector<double>>& U, const std::vector<double>& B) {
    int n = L.size();
    std::vector<double> Y(n, 0.0);
    std::vector<double> X(n, 0.0);

    // Forward substitution to solve Ly = B
    for (int i = 0; i < n; i++) {
        Y[i] = B[i];
        for (int j = 0; j < i; j++) {
            Y[i] -= L[i][j] * Y[j];
        }
        // No need to divide by L[i][i] since it's 1
    }

    // Back substitution to solve Ux = y
    for (int i = n - 1; i >= 0; i--) {
        X[i] = Y[i];
        // Parallelize the loop without reduction
        #pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            X[i] -= U[i][j] * X[j];
        }
        X[i] /= U[i][i];
    }

    return X;
} 