#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <cuda_runtime.h>
#include <cudssSolver.h>  // Include cuDSS header
#include <cudssSupport.h>
#include <cusparse.h>

// Error checking macro for CUDA API calls
#define CHECK_CUDA(func)                                               \
{                                                                      \
    cudaError_t status = (func);                                       \
    if (status != cudaSuccess) {                                       \
        printf("CUDA API failed at %s line %d with error: %s (%d)\n",  \
               __FILE__, __LINE__, cudaGetErrorString(status), status);\
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

// Error checking macro for cuDSS API calls
#define CHECK_CUDSS(func)                                              \
{                                                                      \
    cudssStatus_t status = (func);                                     \
    if (status != CUDSS_STATUS_SUCCESS) {                              \
        printf("cuDSS API failed at %s line %d with error: %d\n",      \
               __FILE__, __LINE__, status);                            \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

// Error checking macro for cuSPARSE API calls
#define CHECK_CUSPARSE(func)                                           \
{                                                                      \
    cusparseStatus_t status = (func);                                  \
    if (status != CUSPARSE_STATUS_SUCCESS) {                           \
        printf("cuSPARSE API failed at %s line %d with error: %d\n",   \
               __FILE__, __LINE__, status);                            \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

// Function to read matrix from file in COO format and convert to CSR format
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
        std::cerr << "Error: Matrix is not square. Rows: " << max_row + 1 
                  << ", Cols: " << max_col + 1 << std::endl;
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

// Function to read vector from file
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

// Function to read the known solution (combining Dl and Dv)
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

// Function to write vector to file
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

// Calculate relative error between two vectors
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

int main(int argc, char* argv[]) {
    // Set CUDA device to 0 (first GPU)
    int deviceId = 0;
    CHECK_CUDA(cudaSetDevice(deviceId));
    
    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << "Using GPU device: " << prop.name << std::endl;
    
    // File paths
    std::string matrixFile = "data/ancf/80/solve_2002_0_Z.dat";
    std::string rhsFile = "data/ancf/80/solve_2002_0_rhs.dat";
    std::string dvFile = "data/ancf/80/solve_2002_0_Dv.dat";
    std::string dlFile = "data/ancf/80/solve_2002_0_Dl.dat";
    std::string outputFile = "soln_cudss_80.dat";
    
    // Read matrix in CSR format
    std::vector<double> csrValues;
    std::vector<int> csrRowPtr;
    std::vector<int> csrColInd;
    int n;
    
    std::cout << "Reading matrix from " << matrixFile << std::endl;
    readMatrixCSR(matrixFile, csrValues, csrRowPtr, csrColInd, n);
    std::cout << "Matrix size: " << n << "x" << n << " with " << csrValues.size() << " non-zero elements" << std::endl;
    
    // Read right-hand side
    std::cout << "Reading RHS from " << rhsFile << std::endl;
    std::vector<double> rhs = readVector(rhsFile);
    if (rhs.size() != static_cast<size_t>(n)) {
        std::cerr << "Error: RHS vector size (" << rhs.size() 
                  << ") does not match matrix size (" << n << ")" << std::endl;
        exit(1);
    }
    
    // Read known solution (for error calculation)
    std::cout << "Reading known solution" << std::endl;
    std::vector<double> knownSolution = readKnownSolution(dvFile, dlFile);
    
    // Prepare solution vector
    std::vector<double> solution(n, 0.0);
    
    // Create cuDSS handle
    cudssHandle_t cudssHandle = nullptr;
    CHECK_CUDSS(cudssCreate(&cudssHandle));
    
    // Create cuSPARSE handle for matrix descriptor
    cusparseHandle_t cusparseHandle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    
    // Create matrix descriptor
    cusparseMatDescr_t matDescr = nullptr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&matDescr));
    CHECK_CUSPARSE(cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO));
    
    // Allocate device memory
    double *d_csrValues = nullptr;
    int *d_csrRowPtr = nullptr;
    int *d_csrColInd = nullptr;
    double *d_rhs = nullptr;
    double *d_solution = nullptr;
    
    CHECK_CUDA(cudaMalloc((void**)&d_csrValues, csrValues.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_csrColInd, csrColInd.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_rhs, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_solution, n * sizeof(double)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_csrValues, csrValues.data(), csrValues.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrRowPtr, csrRowPtr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrColInd, csrColInd.data(), csrColInd.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rhs, rhs.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Solve linear system using cuDSS
    std::cout << "Solving linear system using cuDSS..." << std::endl;
    
    // Start timing
    CHECK_CUDA(cudaEventRecord(start));
    
    // Create the cuDSS configuration
    cudssMatrixDescriptor_t cudssMatDescr = nullptr;
    CHECK_CUDSS(cudssCreateMatrixDescriptor(&cudssMatDescr));
    
    // Set up the matrix descriptor for a general sparse matrix in CSR format
    CHECK_CUDSS(cudssSetMatrixFormat(cudssMatDescr, CUDSS_MFORMAT_CSR, CUDSS_MTYPE_GENERAL, CUDSS_FILL_MODE_FULL, CUDSS_DIAG_TYPE_NON_UNIT));
    
    // Create the cuDSS options
    cudssLUOptions_t luOptions = nullptr;
    CHECK_CUDSS(cudssCreateLUOptions(&luOptions));
    
    // Set solver options (using default values here, can be customized)
    CHECK_CUDSS(cudssSetLUAlgorithm(luOptions, CUDSS_ALG_AUTO));
    CHECK_CUDSS(cudssSetLUPivotType(luOptions, CUDSS_PIVOT_COMPLETE));
    CHECK_CUDSS(cudssSetLUPrecision(luOptions, CUDSS_PRECISION_AUTO));
    
    // Create the solver plan
    cudssLUSolver_t luSolver = nullptr;
    CHECK_CUDSS(cudssCreateLUSolver(cudssHandle, &luSolver));
    
    // Analysis phase
    CHECK_CUDSS(cudssLUAnalyze(
        cudssHandle, luSolver, n, 
        d_csrRowPtr, d_csrColInd, cudssMatDescr, luOptions));
    
    // Factorization phase
    CHECK_CUDSS(cudssLUFactorize(
        cudssHandle, luSolver, d_csrValues));
    
    // Solve phase
    CHECK_CUDSS(cudssLUSolve(
        cudssHandle, luSolver, d_rhs, d_solution));
    
    // Stop timing
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy solution back to host
    CHECK_CUDA(cudaMemcpy(solution.data(), d_solution, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Calculate relative error
    double relError = calculateRelativeError(solution, knownSolution);
    
    // Output results
    std::cout << "Time to solve: " << milliseconds << " ms" << std::endl;
    std::cout << "Relative error: " << relError << std::endl;
    
    // Write solution to file
    writeVectorToFile(solution, outputFile);
    
    // Clean up CUDA events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Clean up cuDSS resources
    CHECK_CUDSS(cudssDestroyLUOptions(luOptions));
    CHECK_CUDSS(cudssDestroyLUSolver(luSolver));
    CHECK_CUDSS(cudssDestroyMatrixDescriptor(cudssMatDescr));
    CHECK_CUDSS(cudssDestroy(cudssHandle));
    
    // Clean up cuSPARSE resources
    CHECK_CUSPARSE(cusparseDestroyMatDescr(matDescr));
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
    
    // Free device memory
    CHECK_CUDA(cudaFree(d_csrValues));
    CHECK_CUDA(cudaFree(d_csrRowPtr));
    CHECK_CUDA(cudaFree(d_csrColInd));
    CHECK_CUDA(cudaFree(d_rhs));
    CHECK_CUDA(cudaFree(d_solution));
    
    return 0;
} 