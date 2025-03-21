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

// Function to select the appropriate cuSolver function based on data type
template <typename T>
cusolverStatus_t cusolverSpTcsrlsvqr(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const T *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const T *b,
    T tol,
    int reorder,
    T *x,
    int *singularity);

// Template specialization for double
template <>
cusolverStatus_t cusolverSpTcsrlsvqr<double>(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *b,
    double tol,
    int reorder,
    double *x,
    int *singularity)
{
    return cusolverSpDcsrlsvqr(
        handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
        b, tol, reorder, x, singularity);
}

// Template specialization for float
template <>
cusolverStatus_t cusolverSpTcsrlsvqr<float>(
    cusolverSpHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *b,
    float tol,
    int reorder,
    float *x,
    int *singularity)
{
    return cusolverSpScsrlsvqr(
        handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA,
        b, tol, reorder, x, singularity);
}

template <typename T>
int solveLinearSystem(int num_spokes, bool use_double = true)
{
    // Set CUDA device to 0 (first GPU)
    int deviceId = 0;
    CHECK_CUDA(cudaSetDevice(deviceId));

    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << "Using GPU device: " << prop.name << std::endl;
    std::cout << "Using " << (use_double ? "double" : "float") << " precision" << std::endl;

    // File paths
    std::string matrixFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Z.dat";
    std::string rhsFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_rhs.dat";
    std::string dvFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Dv.dat";
    std::string dlFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Dl.dat";
    std::string precision = std::is_same<T, float>::value ? "float" : "double";
    std::string outputFile = "soln_cusolver_" + precision + "_" + std::to_string(num_spokes) + ".dat";

    // Read matrix in CSR format
    std::vector<T> csrValues;
    std::vector<int> csrRowPtr;
    std::vector<int> csrColInd;
    int n;

    std::cout << "Reading matrix from " << matrixFile << std::endl;
    readMatrixCSR<T>(matrixFile, csrValues, csrRowPtr, csrColInd, n);
    std::cout << "Matrix size: " << n << "x" << n << " with " << csrValues.size() << " non-zero elements" << std::endl;

    // Read right-hand side
    std::cout << "Reading RHS from " << rhsFile << std::endl;
    std::vector<T> rhs = readVector<T>(rhsFile);
    if (rhs.size() != static_cast<size_t>(n))
    {
        std::cerr << "Error: RHS vector size (" << rhs.size()
                  << ") does not match matrix size (" << n << ")" << std::endl;
        exit(1);
    }

    // Read known solution (for error calculation)
    std::cout << "Reading known solution" << std::endl;
    std::vector<T> knownSolution = readKnownSolution<T>(dvFile, dlFile);

    // Prepare solution vector
    std::vector<T> solution(n, 0.0);

    // Create cuSOLVER and cuSPARSE handles
    cusolverSpHandle_t cusolverHandle = nullptr;
    cusparseHandle_t cusparseHandle = nullptr;
    CHECK_CUSOLVER(cusolverSpCreate(&cusolverHandle));
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

    // Allocate device memory
    T *d_csrValues = nullptr;
    int *d_csrRowPtr = nullptr;
    int *d_csrColInd = nullptr;
    T *d_rhs = nullptr;
    T *d_solution = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_csrValues, csrValues.size() * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_csrRowPtr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_csrColInd, csrColInd.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_rhs, n * sizeof(T)));
    CHECK_CUDA(cudaMalloc((void **)&d_solution, n * sizeof(T)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_csrValues, csrValues.data(), csrValues.size() * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrRowPtr, csrRowPtr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrColInd, csrColInd.data(), csrColInd.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rhs, rhs.data(), n * sizeof(T), cudaMemcpyHostToDevice));

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

    // Use the template function to call the appropriate solver based on type T
    CHECK_CUSOLVER(cusolverSpTcsrlsvqr<T>(
        cusolverHandle, n, csrValues.size(),
        matDescr, d_csrValues, d_csrRowPtr, d_csrColInd, // DEVICE pointers
        d_rhs, static_cast<T>(1e-12),                    // tolerance
        1,                                               // reorder = 1 means use symrcm reordering
        d_solution, &singularity));                      // DEVICE solution vector

    // Stop timing
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaMemcpy(solution.data(), d_solution, n * sizeof(T), cudaMemcpyDeviceToHost));

    // Check for singularity
    if (singularity >= 0)
    {
        std::cout << "WARNING: The matrix is singular at row " << singularity << std::endl;
    }

    // Calculate relative error
    T relError = calculateRelativeErrorRaw<T>(solution.data(), knownSolution.data(), n);

    // Output results
    std::cout << "Time to solve: " << milliseconds << " ms" << std::endl;
    std::cout << "Relative error: " << relError << std::endl;

    // Write solution to file
    writeVectorToFile<T>(solution, outputFile);
    std::cout << "Solution written to " << outputFile << std::endl;

    // Clean up CUDA events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Clean up
    CHECK_CUSPARSE(cusparseDestroyMatDescr(matDescr));
    CHECK_CUSOLVER(cusolverSpDestroy(cusolverHandle));
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));

    // Free device memory
    CHECK_CUDA(cudaFree(d_csrValues));
    CHECK_CUDA(cudaFree(d_csrRowPtr));
    CHECK_CUDA(cudaFree(d_csrColInd));
    CHECK_CUDA(cudaFree(d_rhs));
    CHECK_CUDA(cudaFree(d_solution));

    return 0;
}

void printUsage(const char *programName)
{
    std::cerr << "Usage: " << programName << " [num_spokes] [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -f, --float    Use single precision (float)" << std::endl;
    std::cerr << "  -d, --double   Use double precision (default)" << std::endl;
    std::cerr << "Example: " << programName << " 32 --float" << std::endl;
}

int main(int argc, char *argv[])
{
    // Check command line arguments for num_spokes (optional with default value of 16)
    int num_spokes = 16;    // Default value
    bool use_double = true; // Default to double precision
    bool custom_spokes = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--float" || arg == "-f")
        {
            use_double = false;
        }
        else if (arg == "--double" || arg == "-d")
        {
            use_double = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            // Assume this is the num_spokes value
            try
            {
                num_spokes = std::stoi(arg);
                if (num_spokes <= 0)
                {
                    std::cerr << "Error: num_spokes must be a positive integer" << std::endl;
                    printUsage(argv[0]);
                    return 1;
                }
                custom_spokes = true;
            }
            catch (...)
            {
                std::cerr << "Error: Invalid argument: " << arg << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        }
    }

    if (!custom_spokes)
    {
        std::cout << "No num_spokes provided. Using default value = " << num_spokes << std::endl;
    }

    // Call the appropriate solver based on precision flag
    if (use_double)
    {
        return solveLinearSystem<double>(num_spokes, true);
    }
    else
    {
        return solveLinearSystem<float>(num_spokes, false);
    }
}
