/*
 * Copyright 2023-2025 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "cudss.h"
#include "utils.h"

// Added includes for STL containers
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <chrono>
#include <iomanip>
#include <unistd.h>

/*
    This example demonstrates repeated usage of cuDSS APIs for solving
    a system of linear algebraic equations with a sparse matrix:
                                Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).
    
    The script allocates memory once and then loops through the
    analysis, factorization, and solve phases 6 times to measure
    performance characteristics.
*/

#define CUDSS_EXAMPLE_FREE       \
    do {                         \
        cudaFree(csr_offsets_d); \
        cudaFree(csr_columns_d); \
        cudaFree(csr_values_d);  \
        cudaFree(x_values_d);    \
        cudaFree(b_values_d);    \
    } while (0);

#define CUDA_CALL_AND_CHECK(call, msg)                                                               \
    do {                                                                                             \
        cudaError_t cuda_error = call;                                                               \
        if (cuda_error != cudaSuccess) {                                                             \
            printf("Example FAILED: CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
            CUDSS_EXAMPLE_FREE;                                                                      \
            return -1;                                                                               \
        }                                                                                            \
    } while (0);

#define CUDSS_CALL_AND_CHECK(call, status, msg)                                                                      \
    do {                                                                                                             \
        status = call;                                                                                               \
        if (status != CUDSS_STATUS_SUCCESS) {                                                                        \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            CUDSS_EXAMPLE_FREE;                                                                                      \
            return -2;                                                                                               \
        }                                                                                                            \
    } while (0);

// Function to create directory if it doesn't exist
void createDirectoryIfNotExists(const std::string& dir) {
    struct stat st = {0};
    if (stat(dir.c_str(), &st) == -1) {
        mkdir(dir.c_str(), 0755);
    }
}

// Function to write timing log for loops
template <typename T>
void writeLoopTimingLog(int num_spokes, bool use_double, 
                       const std::vector<float>& analysis_times, 
                       const std::vector<float>& factorization_times, 
                       const std::vector<float>& solve_times,
                       const std::vector<T>& backward_errors) {
    createDirectoryIfNotExists("./logs");
    
    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::string precision = use_double ? "double" : "float";
    std::string logFile = "../logs/loop_timing_log.csv";
    
    // Check if file exists to determine if we need to write header
    bool fileExists = (access(logFile.c_str(), F_OK) == 0);
    
    std::ofstream log(logFile, std::ios::app);
    if (!log.is_open()) {
        printf("Warning: Could not open log file for writing\n");
        return;
    }
    
    // Write header if file is new
    if (!fileExists) {
        log << "timestamp,num_spokes,precision,iteration,analysis_time_ms,factorization_time_ms,solve_time_ms,total_time_ms,backward_error\n";
    }
    
    // Write timing data for each iteration
    for (size_t i = 0; i < analysis_times.size(); ++i) {
        log << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << ","
            << num_spokes << ","
            << precision << ","
            << (i + 1) << ","
            << std::fixed << std::setprecision(6)
            << analysis_times[i] << ","
            << factorization_times[i] << ","
            << solve_times[i] << ","
            << (analysis_times[i] + factorization_times[i] + solve_times[i]) << ","
            << std::scientific << std::setprecision(6)
            << backward_errors[i] << "\n";
    }
    
    log.close();
    printf("Loop timing data logged to %s\n", logFile.c_str());
}

// Function to print usage information
void printUsage(const char* programName) {
    printf("Usage: %s [num_spokes] [options]\n", programName);
    printf("Options:\n");
    printf("  -f, --float    Use single precision (float)\n");
    printf("  -d, --double   Use double precision (default)\n");
    printf("Example: %s 32 --float\n", programName);
}

// Template function for solving with different precision
template <typename T>
int solveWithCUDSSLoop(int num_spokes, bool use_double) {
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    
    // Define CUDA data type based on template type
    cudaDataType_t cuda_data_type = std::is_same<T, double>::value ? CUDA_R_64F : CUDA_R_32F;
    
    // Set error tolerance based on precision
    T error_tolerance = std::is_same<T, double>::value ? 1e-7 : 1e-5f;

    const int NUM_ITERATIONS = 3;

    // Print precision mode
    printf("Running with %s precision for %d iterations\n", use_double ? "double" : "single (float)", NUM_ITERATIONS);
    
    // Define file paths for the matrix and RHS based on num_spokes
    std::string baseDir = "data/ancf/";
    std::string refineDir, baseName;
    
    if (num_spokes == 16) {
        refineDir = "refine1";
        baseName = "2002";
    } else if (num_spokes == 80) {
        refineDir = "refine2";
        baseName = "1001";
    } else {
        printf("Error: Unsupported num_spokes value: %d. Supported values are 16 and 80.\n", num_spokes);
        return -1;
    }
    
    std::string matrixFile = baseDir + refineDir + "/" + std::to_string(num_spokes) + "/solve_" + baseName + "_0_Z.dat";
    std::string rhsFile = baseDir + refineDir + "/" + std::to_string(num_spokes) + "/solve_" + baseName + "_0_rhs.dat";

    // Host containers for CSR data and RHS vector
    std::vector<T> csr_values_h;
    std::vector<int> csr_offsets_h;
    std::vector<int> csr_columns_h;

    int n;
    readMatrixCSR<T>(matrixFile, csr_values_h, csr_offsets_h, csr_columns_h, n);
    int nnz = csr_values_h.size();
    printf("Matrix read from file: dimension = %d x %d, nnz = %d\n", n, n, nnz);

    std::vector<T> b_values_h = readVector<T>(rhsFile);
    if (b_values_h.size() != static_cast<size_t>(n)) {
        printf("Error: RHS vector size (%zu) does not match matrix dimension (%d)\n", b_values_h.size(), n);
        return -1;
    }

    // Device pointers - allocated once
    int* csr_offsets_d = NULL;
    int* csr_columns_d = NULL;
    T* csr_values_d = NULL;
    T *x_values_d = NULL, *b_values_d = NULL;

    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)), "cudaMalloc for csr_offsets_d");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)), "cudaMalloc for csr_columns_d");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(T)), "cudaMalloc for csr_values_d");
    CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, n * sizeof(T)), "cudaMalloc for b_values_d");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, n * sizeof(T)), "cudaMalloc for x_values_d");

    // Copy host data to device once
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_offsets_d");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h.data(), nnz * sizeof(int), cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_columns_d");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h.data(), nnz * sizeof(T), cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_values_d");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h.data(), n * sizeof(T), cudaMemcpyHostToDevice),
                        "cudaMemcpy for b_values_d");

    // Create a CUDA stream
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    // Create CUDA events for timing each phase
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CALL_AND_CHECK(cudaEventCreate(&start), "cudaEventCreate for start");
    CUDA_CALL_AND_CHECK(cudaEventCreate(&stop), "cudaEventCreate for stop");

    // Vectors to store timing data for each iteration
    std::vector<float> analysis_times(NUM_ITERATIONS);
    std::vector<float> factorization_times(NUM_ITERATIONS);
    std::vector<float> solve_times(NUM_ITERATIONS);
    std::vector<T> backward_errors(NUM_ITERATIONS);
    std::vector<T> relative_errors(NUM_ITERATIONS);

    // Read known solution for error calculation (done once)
    std::string dvFile = baseDir + refineDir + "/" + std::to_string(num_spokes) + "/solve_" + baseName + "_0_Dv.dat";
    std::string dlFile = baseDir + refineDir + "/" + std::to_string(num_spokes) + "/solve_" + baseName + "_0_Dl.dat";
    std::vector<T> knownSolution = readKnownSolution<T>(dvFile, dlFile);

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices). */
    cudssMatrix_t x, b;

    int nrhs = 1;
    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, cuda_data_type, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, cuda_data_type, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL, csr_columns_d, csr_values_d,
                                              CUDA_R_32I, cuda_data_type, mtype, mview, base),
                         status, "cudssMatrixCreateCsr");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;
    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Set Solver Configuration Parameters */
    
    // Reordering algorithm
    cudssAlgType_t reorderingAlg = CUDSS_ALG_DEFAULT;
    CUDSS_CALL_AND_CHECK(
        cudssConfigSet(solverConfig, CUDSS_CONFIG_REORDERING_ALG, &reorderingAlg, sizeof(reorderingAlg)), status,
        "cudssConfigSet for cudssAlgType_t");

    cudssAlgType_t pivotEpsilonAlg = CUDSS_ALG_DEFAULT;
    CUDSS_CALL_AND_CHECK(
        cudssConfigSet(solverConfig, CUDSS_CONFIG_PIVOT_EPSILON_ALG, &pivotEpsilonAlg, sizeof(pivotEpsilonAlg)), status,
        "cudssConfigSet for cudssAlgType_t");

    // Set pivot epsilon value (controls numerical pivoting tolerance)
    T pivotEpsilon = std::is_same<T, double>::value ? 1e-8 : 1e-4f; //default is 1e-13 for double, 1e-5 for float
    printf("Setting pivot epsilon to: %e\n", (double)pivotEpsilon);
    CUDSS_CALL_AND_CHECK(
        cudssConfigSet(solverConfig, CUDSS_CONFIG_PIVOT_EPSILON, &pivotEpsilon, sizeof(pivotEpsilon)), status,
        "cudssConfigSet for pivot epsilon");

    // int matchingType = 1;  // Switched on for pardiso
    // CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_USE_MATCHING, &matchingType, sizeof(matchingType)),
    //                         status, "cudssConfigSet for int");

    int modificator = 0;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_SOLVE_MODE, &modificator, sizeof(modificator)),
                         status, "cudssConfigSet for int");

    int iterRefinement = 0;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_IR_N_STEPS, &iterRefinement, sizeof(iterRefinement)),
                         status, "cudssConfigSet for int");

    cudssPivotType_t pivotType = CUDSS_PIVOT_COL;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_PIVOT_TYPE, &pivotType, sizeof(pivotType)), status,
                         "cudssConfigSet for cudssPivotType_t");

    T pivotThreshold = 1;
    CUDSS_CALL_AND_CHECK(
        cudssConfigSet(solverConfig, CUDSS_CONFIG_PIVOT_THRESHOLD, &pivotThreshold, sizeof(pivotThreshold)), status,
        "cudssConfigSet for real_t");

    int hybridExecuteMode = 0;
    CUDSS_CALL_AND_CHECK(
        cudssConfigSet(solverConfig, CUDSS_CONFIG_HYBRID_EXECUTE_MODE, &hybridExecuteMode, sizeof(hybridExecuteMode)),
        status, "cudssConfigSet for int");

    /* Symbolic factorization (run once) */
    printf("Running analysis phase...\n");
    CUDA_CALL_AND_CHECK(cudaEventRecord(start), "cudaEventRecord start for analysis");
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b), status,
                         "cudssExecute for analysis");
    CUDA_CALL_AND_CHECK(cudaEventRecord(stop), "cudaEventRecord stop for analysis");
    CUDA_CALL_AND_CHECK(cudaEventSynchronize(stop), "cudaEventSynchronize for analysis");
    float analysis_time;
    CUDA_CALL_AND_CHECK(cudaEventElapsedTime(&analysis_time, start, stop), "cudaEventElapsedTime for analysis");
    printf("Analysis time: %f ms\n", analysis_time);

    /* Factorization (run once before the loop) */
    printf("Running factorization phase...\n");
    CUDA_CALL_AND_CHECK(cudaEventRecord(start), "cudaEventRecord start for factorization");
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, x, b), status,
                         "cudssExecute for factorization");
    CUDA_CALL_AND_CHECK(cudaEventRecord(stop), "cudaEventRecord stop for factorization");
    CUDA_CALL_AND_CHECK(cudaEventSynchronize(stop), "cudaEventSynchronize for factorization");
    float factorization_time;
    CUDA_CALL_AND_CHECK(cudaEventElapsedTime(&factorization_time, start, stop), "cudaEventElapsedTime for factorization");
    printf("Factorization time: %f ms\n", factorization_time);

    printf("\n=== Starting %d solve iterations ===\n", NUM_ITERATIONS);
    printf("Analysis time: %f ms (constant)\n", analysis_time);
    printf("Factorization time: %f ms (constant)\n", factorization_time);

    // Loop for multiple solve iterations only
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        printf("\n--- Solve Iteration %d ---\n", iter + 1);

        // Store analysis and factorization times for logging (same for all iterations)
        analysis_times[iter] = analysis_time;
        factorization_times[iter] = factorization_time;

        /* Solving */
        CUDA_CALL_AND_CHECK(cudaEventRecord(start), "cudaEventRecord start for solve");
        CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b), status,
                             "cudssExecute for solve");
        CUDA_CALL_AND_CHECK(cudaEventRecord(stop), "cudaEventRecord stop for solve");
        CUDA_CALL_AND_CHECK(cudaEventSynchronize(stop), "cudaEventSynchronize for solve");
        CUDA_CALL_AND_CHECK(cudaEventElapsedTime(&solve_times[iter], start, stop), "cudaEventElapsedTime for solve");

        /* Synchronize the stream to ensure completion */
        CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        /* Copy the solution back to host and calculate errors */
        std::vector<T> x_values_h(n, 0.0);
        CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h.data(), x_values_d, nrhs * n * sizeof(T), cudaMemcpyDeviceToHost),
                            "cudaMemcpy for x_values");

        // Calculate relative error
        relative_errors[iter] = calculateRelativeErrorRaw<T>(x_values_h.data(), knownSolution.data(), n);

        // Calculate the backward error (residual-based)
        backward_errors[iter] = calculateBackwardError<T>(csr_values_h, csr_offsets_h, csr_columns_h, x_values_h, b_values_h);

        // Output only the varying solve time and errors for this iteration
        printf("Solve time: %f ms\n", solve_times[iter]);
        printf("Total time: %f ms\n", analysis_time + factorization_time + solve_times[iter]);
        printf("Relative error: %f\n", relative_errors[iter]);
        printf("Backward error: %e\n", backward_errors[iter]);
    }

    /* Clean up cuDSS resources after all iterations */
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssDestroy");

    /* Clean up matrix objects after all iterations */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Free CUDA resources and destroy the stream */
    cudaFree(csr_offsets_d);
    cudaFree(csr_columns_d);
    cudaFree(csr_values_d);
    cudaFree(x_values_d);
    cudaFree(b_values_d);
    cudaStreamDestroy(stream);

    // Write timing log before device reset to ensure it always happens
    printf("Writing timing log...\n");
    writeLoopTimingLog<T>(num_spokes, use_double, analysis_times, factorization_times, solve_times, backward_errors);

    /* Reset CUDA device to ensure clean state */
    CUDA_CALL_AND_CHECK(cudaDeviceReset(), "cudaDeviceReset");

    return 0;
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    int num_spokes = 80;  // Default value
    bool use_double = true;  // Default to double precision
    bool custom_spokes = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--float" || arg == "-f") {
            use_double = false;
        }
        else if (arg == "--double" || arg == "-d") {
            use_double = true;
        }
        else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else {
            // Assume this is the num_spokes value
            try {
                num_spokes = std::stoi(arg);
                if (num_spokes <= 0) {
                    printf("Error: num_spokes must be a positive integer\n");
                    printUsage(argv[0]);
                    return 1;
                }
                custom_spokes = true;
                printf("Using num_spokes = %d\n", num_spokes);
            }
            catch (...) {
                printf("Error: Invalid argument: %s\n", arg.c_str());
                printUsage(argv[0]);
                return 1;
            }
        }
    }

    if (!custom_spokes) {
        printf("No num_spokes provided. Using default value = %d\n", num_spokes);
    }

    // Call the appropriate solver based on precision flag
    if (use_double) {
        return solveWithCUDSSLoop<double>(num_spokes, true);
    }
    else {
        return solveWithCUDSSLoop<float>(num_spokes, false);
    }
} 