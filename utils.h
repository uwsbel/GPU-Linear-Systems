#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <tuple>
#include <algorithm>
#include <cmath>

// Add CUDA compatibility
#ifdef __CUDACC__
    #define UTILS_HOST_DEVICE __host__ __device__
    #define UTILS_HOST __host__
    #define UTILS_DEVICE __device__
#else
    #define UTILS_HOST_DEVICE
    #define UTILS_HOST
    #define UTILS_DEVICE
#endif

/**
 * Reads a matrix in COO format from a file and converts it to CSR format.
 *
 * @param filename Path to the input file containing matrix data in COO format
 * @param values Output vector to store the non-zero values
 * @param rowIndex Output vector to store the row pointers
 * @param columns Output vector to store the column indices
 * @param n Output parameter to store the matrix dimension
 */
void readMatrixCSR(const std::string& filename,
                   std::vector<double>& values,
                   std::vector<int>& rowIndex,
                   std::vector<int>& columns,
                   int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    // Read all triplets first to determine matrix size
    std::vector<std::tuple<int, int, double>> triplets;
    int row, col;
    double value;
    int max_row = 0, max_col = 0;

    // Read all entries
    while (file >> row >> col >> value) {
        // Convert from 1-based to 0-based indexing if needed
        row--;
        col--;

        // Keep track of matrix dimensions
        max_row = std::max(max_row, row);
        max_col = std::max(max_col, col);

        triplets.emplace_back(row, col, value);
    }

    // Matrix dimensions are max indices + 1 (since we converted to 0-based)
    n = max_row + 1;

    // Check if matrix is square
    if (max_row != max_col) {
        std::cerr << "Error: Matrix is not square. Rows: " << max_row + 1 << ", Cols: " << max_col + 1 << std::endl;
        exit(1);
    }

    // Sort triplets by row, then by column for CSR format
    std::sort(triplets.begin(), triplets.end());

    // Initialize CSR arrays
    values.resize(triplets.size());
    columns.resize(triplets.size());
    rowIndex.resize(n + 1, 0);

    // Fill in the CSR arrays
    int current_row = -1;
    for (size_t i = 0; i < triplets.size(); i++) {
        int row = std::get<0>(triplets[i]);
        int col = std::get<1>(triplets[i]);
        double val = std::get<2>(triplets[i]);

        // Update row index array
        while (current_row < row) {
            current_row++;
            rowIndex[current_row] = i;
        }

        // Store column index and value
        columns[i] = col;
        values[i] = val;
    }

    // Set the last element of rowIndex
    rowIndex[n] = triplets.size();

    file.close();
}

/**
 * Reads a vector from a file.
 *
 * @param filename Path to the input file containing vector data
 * @return Vector of double values
 */
std::vector<double> readVector(const std::string& filename) {
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

    file.close();
    return values;
}

/**
 * Reads the known solution by combining and transforming Dv and Dl files.
 *
 * @param dvFilename Path to the Dv file
 * @param dlFilename Path to the Dl file
 * @return Combined solution vector
 */
std::vector<double> readKnownSolution(const std::string& dvFilename, const std::string& dlFilename) {
    std::vector<double> dvPart = readVector(dvFilename);
    std::vector<double> dlPart = readVector(dlFilename);

    // Negate dlPart before combining
    for (auto& val : dlPart) {
        val = -val;
    }

    // Create combined vector
    std::vector<double> solution;
    solution.reserve(dvPart.size() + dlPart.size());
    solution.insert(solution.end(), dvPart.begin(), dvPart.end());
    solution.insert(solution.end(), dlPart.begin(), dlPart.end());

    return solution;
}

/**
 * Writes a vector to a file.
 *
 * @param vector Vector of double values to write
 * @param filename Path to the output file
 */
void writeVectorToFile(const std::vector<double>& vector, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        exit(1);
    }

    // Set precision for output
    file.precision(16);
    file << std::scientific;

    // Write each element on a new line
    for (size_t i = 0; i < vector.size(); i++) {
        file << vector[i] << std::endl;
    }

    file.close();
    std::cout << "Solution written to " << filename << std::endl;
}

/**
 * Calculates the relative error between two vectors.
 * Host-only version that can use std::vector and print error messages.
 *
 * @param computed Computed solution vector
 * @param reference Reference solution vector
 * @return Relative error as a double
 */
UTILS_HOST
double calculateRelativeError(const std::vector<double>& computed, const std::vector<double>& reference) {
    if (computed.size() != reference.size()) {
        std::cerr << "Error: Vector sizes don't match for error calculation" << std::endl;
        return -1.0;
    }

    double norm_diff = 0.0;
    double norm_ref = 0.0;

    for (size_t i = 0; i < computed.size(); i++) {
        double diff = computed[i] - reference[i];
        norm_diff += diff * diff;
        norm_ref += reference[i] * reference[i];
    }

    return std::sqrt(norm_diff) / std::sqrt(norm_ref);
}

/**
 * Calculates the relative error between two arrays.
 * Device-compatible version with raw pointers and size parameters.
 *
 * @param computed Computed solution array
 * @param reference Reference solution array
 * @param size Size of both arrays
 * @return Relative error as a double
 */
UTILS_HOST_DEVICE
double calculateRelativeErrorRaw(const double* computed, const double* reference, int size) {
    double norm_diff = 0.0;
    double norm_ref = 0.0;

    for (int i = 0; i < size; i++) {
        double diff = computed[i] - reference[i];
        norm_diff += diff * diff;
        norm_ref += reference[i] * reference[i];
    }

    return std::sqrt(norm_diff) / std::sqrt(norm_ref);
}