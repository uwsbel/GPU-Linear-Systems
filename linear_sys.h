#ifndef LINEAR_SYS_H
#define LINEAR_SYS_H

#include <vector>
#include <omp.h>

// LU decomposition function
void luDecomposition(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U);

// Solve Ax = B using the LU decomposition
std::vector<double> solveLU(const std::vector<std::vector<double>>& L, const std::vector<std::vector<double>>& U, const std::vector<double>& B);

#endif // LINEAR_SYS_H 