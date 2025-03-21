#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include "utils.h"

// Error checking macro for CUDA API calls
#define CHECK_CUDA(func)                                                    \
    {                                                                       \
        cudaError_t status = (func);                                        \
        if (status != cudaSuccess)                                          \
        {                                                                   \
            printf("CUDA API failed at %s line %d with error: %s (%d)\n",   \
                   __FILE__, __LINE__, cudaGetErrorString(status), status); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

// Error checking macro for cuSOLVER API calls
#define CHECK_CUSOLVER(func)                                             \
    {                                                                    \
        cusolverStatus_t status = (func);                                \
        if (status != CUSOLVER_STATUS_SUCCESS)                           \
        {                                                                \
            printf("cuSOLVER API failed at %s line %d with error: %d\n", \
                   __FILE__, __LINE__, status);                          \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

// Error checking macro for cuSPARSE API calls
#define CHECK_CUSPARSE(func)                                             \
    {                                                                    \
        cusparseStatus_t status = (func);                                \
        if (status != CUSPARSE_STATUS_SUCCESS)                           \
        {                                                                \
            printf("cuSPARSE API failed at %s line %d with error: %d\n", \
                   __FILE__, __LINE__, status);                          \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

// Function to read matrix from file in COO format and convert to CSR format is now in utils.h
// Function to read vector from file is now in utils.h
// Function to read the known solution (combining Dl and Dv) is now in utils.h
// Function to write vector to file is now in utils.h
// Calculate relative error between two vectors is now in utils.h

int main(int argc, char *argv[])
{
    // Check command line arguments for num_spokes (optional with default value of 16)
    int num_spokes = 16; // Default value
    if (argc > 1)
    {
        num_spokes = std::stoi(argv[1]);
        if (num_spokes <= 0)
        {
            std::cerr << "Error: num_spokes must be a positive integer" << std::endl;
            return 1;
        }
    }
    else
    {
        std::cout << "No num_spokes provided. Using default value = " << num_spokes << std::endl;
    }

    // Set CUDA device to 0 (first GPU)
    int deviceId = 0;
    CHECK_CUDA(cudaSetDevice(deviceId));

    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << "Using GPU device: " << prop.name << std::endl;

    // File paths
    std::string matrixFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Z.dat";
    std::string rhsFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_rhs.dat";
    std::string dvFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Dv.dat";
    std::string dlFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Dl.dat";
    std::string outputFile = "soln_cusolver_" + std::to_string(num_spokes) + ".dat";

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
    if (rhs.size() != static_cast<size_t>(n))
    {
        std::cerr << "Error: RHS vector size (" << rhs.size()
                  << ") does not match matrix size (" << n << ")" << std::endl;
        exit(1);
    }

    // Read known solution (for error calculation)
    std::cout << "Reading known solution" << std::endl;
    std::vector<double> knownSolution = readKnownSolution(dvFile, dlFile);

    // Prepare solution vector
    std::vector<double> solution(n, 0.0);

    // Create cuSOLVER and cuSPARSE handles
    cusolverSpHandle_t cusolverHandle = nullptr;
    cusparseHandle_t cusparseHandle = nullptr;
    CHECK_CUSOLVER(cusolverSpCreate(&cusolverHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // Allocate device memory
    double *d_csrValues = nullptr;
    int *d_csrRowPtr = nullptr;
    int *d_csrColInd = nullptr;
    double *d_rhs = nullptr;
    double *d_solution = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_csrValues, csrValues.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&d_csrRowPtr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_csrColInd, csrColInd.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_rhs, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void **)&d_solution, n * sizeof(double)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_csrValues, csrValues.data(), csrValues.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrRowPtr, csrRowPtr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrColInd, csrColInd.data(), csrColInd.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rhs, rhs.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Create matrix descriptor
    cusparseMatDescr_t matDescr = nullptr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&matDescr));
    CHECK_CUSPARSE(cusparseSetMatType(matDescr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Solve linear system using cuSOLVER
    std::cout << "Solving linear system using cuSOLVER..." << std::endl;

    // Start timing
    CHECK_CUDA(cudaEventRecord(start));

    // Setup for the solver
    int singularity = 0;

    // Create parameter structure for the solver
    // Using LU factorization as requested

    // LU factorization with partial pivoting (host version)
    // Needs to be updated
    CHECK_CUSOLVER(cusolverSpDcsrlsvqr(
        cusolverHandle, n, csrValues.size(),
        matDescr, d_csrValues, d_csrRowPtr, d_csrColInd, // DEVICE pointers
        d_rhs, 1e-12,                                    // tolerance
        1,                                               // reorder = 1 means use symrcm reordering
        d_solution, &singularity));                      // DEVICE solution vector

    // Stop timing
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy solution back to host
    CHECK_CUDA(cudaMemcpy(solution.data(), d_solution, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Check for singularity
    if (singularity >= 0)
    {
        std::cout << "WARNING: The matrix is singular at row " << singularity << std::endl;
    }

    // Calculate relative error
    double relError = calculateRelativeErrorRaw(solution.data(), knownSolution.data(), n);

    // Output results
    std::cout << "Time to solve: " << milliseconds << " ms" << std::endl;
    std::cout << "Relative error: " << relError << std::endl;

    // Write solution to file
    writeVectorToFile(solution, outputFile);
    std::cout << "Solution written to " << outputFile << std::endl;

    // Clean up CUDA events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Clean up
    CHECK_CUSPARSE(cusparseDestroyMatDescr(matDescr));
    CHECK_CUSOLVER(cusolverSpDestroy(cusolverHandle));
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));

    CHECK_CUDA(cudaFree(d_csrValues));
    CHECK_CUDA(cudaFree(d_csrRowPtr));
    CHECK_CUDA(cudaFree(d_csrColInd));
    CHECK_CUDA(cudaFree(d_rhs));
    CHECK_CUDA(cudaFree(d_solution));

    return 0;
}
