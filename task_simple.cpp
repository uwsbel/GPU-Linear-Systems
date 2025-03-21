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

/*
    This example demonstrates basic usage of cuDSS APIs for solving
    a system of linear algebraic equations with a sparse matrix:
                                Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).
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

int main(int argc, char* argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a linear system from file input\n");
    printf("---------------------------------------------------------\n");

    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    // Check command line arguments
    int num_spokes = 16;  // Default value
    if (argc > 1) {
        num_spokes = atoi(argv[1]);
        if (num_spokes <= 0) {
            printf("Error: num_spokes must be a positive integer\n");
            return -1;
        }
    } else {
        printf("No argument provided. Using default num_spokes = %d\n", num_spokes);
    }

    // Define file paths for the matrix and RHS (modify these paths as needed)
    std::string matrixFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Z.dat";
    std::string rhsFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_rhs.dat";

    // Host containers for CSR data and RHS vector
    std::vector<double> csr_values_h;
    std::vector<int> csr_offsets_h;
    std::vector<int> csr_columns_h;
    int n;
    readMatrixCSR<double>(matrixFile, csr_values_h, csr_offsets_h, csr_columns_h, n);
    int nnz = csr_values_h.size();
    printf("Matrix read from file: dimension = %d x %d, nnz = %d\n", n, n, nnz);

    std::vector<double> b_values_h = readVector<double>(rhsFile);
    if (b_values_h.size() != static_cast<size_t>(n)) {
        printf("Error: RHS vector size (%zu) does not match matrix dimension (%d)\n", b_values_h.size(), n);
        return -1;
    }

    // Device pointers
    int* csr_offsets_d = NULL;
    int* csr_columns_d = NULL;
    double* csr_values_d = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;

    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)), "cudaMalloc for csr_offsets_d");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)), "cudaMalloc for csr_columns_d");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(double)), "cudaMalloc for csr_values_d");
    CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, n * sizeof(double)), "cudaMalloc for b_values_d");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, n * sizeof(double)), "cudaMalloc for x_values_d");

    // Copy host data to device
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_offsets_d");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h.data(), nnz * sizeof(int), cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_columns_d");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h.data(), nnz * sizeof(double), cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_values_d");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h.data(), n * sizeof(double), cudaMemcpyHostToDevice),
                        "cudaMemcpy for b_values_d");

    // Create a CUDA stream
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

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

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices). */
    cudssMatrix_t x, b;

    int nrhs = 1;
    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;  // Using general matrix type
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL, csr_columns_d, csr_values_d,
                                              CUDA_R_32I, CUDA_R_64F, mtype, mview, base),
                         status, "cudssMatrixCreateCsr");

    // Create CUDA events for timing
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CALL_AND_CHECK(cudaEventCreate(&start), "cudaEventCreate for start");
    CUDA_CALL_AND_CHECK(cudaEventCreate(&stop), "cudaEventCreate for stop");

    // Start timing
    CUDA_CALL_AND_CHECK(cudaEventRecord(start), "cudaEventRecord start");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b), status,
                         "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, x, b), status,
                         "cudssExecute for factorization");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b), status,
                         "cudssExecute for solve");

    // Stop timing
    CUDA_CALL_AND_CHECK(cudaEventRecord(stop), "cudaEventRecord stop");
    CUDA_CALL_AND_CHECK(cudaEventSynchronize(stop), "cudaEventSynchronize");

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CALL_AND_CHECK(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");

    // Output the time taken
    printf("Time to solve: %f ms\n", milliseconds);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Synchronize the stream to ensure completion */
    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Copy the solution back to host and print the results */
    std::vector<double> x_values_h(n, 0.0);
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h.data(), x_values_d, nrhs * n * sizeof(double), cudaMemcpyDeviceToHost),
                        "cudaMemcpy for x_values");

    /* Clean up cuDSS resources */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssDestroy");

    /* Free CUDA resources and destroy the stream */
    cudaFree(csr_offsets_d);
    cudaFree(csr_columns_d);
    cudaFree(csr_values_d);
    cudaFree(x_values_d);
    cudaFree(b_values_d);
    cudaStreamDestroy(stream);

    // Read known solution for error calculation
    std::string dvFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Dv.dat";
    std::string dlFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Dl.dat";
    std::vector<double> knownSolution = readKnownSolution<double>(dvFile, dlFile);

    // Calculate relative error
    double relError = calculateRelativeErrorRaw<double>(x_values_h.data(), knownSolution.data(), n);
    printf("Relative error: %f\n", relError);

    // Write solution to file
    std::string outputFile = "soln_simple_" + std::to_string(num_spokes) + ".dat";
    writeVectorToFile<double>(x_values_h, outputFile);

    if (relError > 1e-7) {
        printf("Example FAILED: Relative error too large\n");
        return -3;
    }

    printf("Example PASSED\n");
    return 0;
}
