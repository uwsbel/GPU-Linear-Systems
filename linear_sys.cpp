#include "linear_sys.h"
#include <iostream>
#include <stdexcept>
#include <mkl.h> // Include MKL header for MKL functions

void luDecomposition(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U) {
    // Implementation of LU decomposition
    // ...
}

std::vector<double> solveLU(const std::vector<std::vector<double>>& L, const std::vector<std::vector<double>>& U, const std::vector<double>& B) {
    // Implementation of solving Ax = B using L and U
    // ...
}

void luDecompositionMKL(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U) {
    // MKL implementation of LU decomposition
    // ...
}

std::vector<double> solveLUMKL(const std::vector<std::vector<double>>& L, const std::vector<std::vector<double>>& U, const std::vector<double>& B) {
    // MKL implementation of solving Ax = B using L and U
    // ...
} 