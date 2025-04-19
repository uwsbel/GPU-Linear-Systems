/*
 * Copyright 2023-2025 NVIDIA Corporation.  All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <amgx_c.h>

// Added includes for STL containers
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "utils.h"

// Error checking macro for CUDA calls
#define CUDA_CHECK(call)                                                         \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                              \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Error checking macro for AMGX calls
#define AMGX_CHECK(call)                                                        \
    do {                                                                        \
        AMGX_RC err = call;                                                    \
        if (err != AMGX_RC_OK) {                                              \
            char msg[4096];                                                    \
            AMGX_get_error_string(err, msg, 4096);                            \
            printf("AMGX error: %s\n", msg);                                   \
            printf("In file %s at line %d\n", __FILE__, __LINE__);            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

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
int solveWithAMGX(int num_spokes, bool use_double) {
    // Print precision mode
    printf("Running with %s precision\n", use_double ? "double" : "single (float)");

    // Define file paths for the matrix and RHS
    std::string matrixFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Z.dat";
    std::string rhsFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_rhs.dat";

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

    // Initialize AMGX
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());

    // Create AMGX config
    AMGX_config_handle cfg;
    const char* config_string;
    if (use_double) {
        config_string = "config_version=2, solver=FGMRES, determinism_flag=1, "
                       "matrix_precision=DOUBLE, vector_precision=DOUBLE, "
                       "max_iters=1000, convergence=RELATIVE_INI_CORE, "
                       "tolerance=1e-7, norm=L2, "
                       "preconditioner(amg_solver)=AMG, "
                       "amg_solver:max_levels=100, amg_solver:cycle=V, "
                       "amg_solver:presweeps=1, amg_solver:postsweeps=1, "
                       "amg_solver:matrix_coloring_scheme=PARALLEL";
    } else {
        config_string = "config_version=2, solver=FGMRES, determinism_flag=1, "
                       "matrix_precision=SINGLE, vector_precision=SINGLE, "
                       "max_iters=1000, convergence=RELATIVE_INI_CORE, "
                       "tolerance=1e-5, norm=L2, "
                       "preconditioner(amg_solver)=AMG, "
                       "amg_solver:max_levels=100, amg_solver:cycle=V, "
                       "amg_solver:presweeps=1, amg_solver:postsweeps=1, "
                       "amg_solver:matrix_coloring_scheme=PARALLEL";
    }
    AMGX_SAFE_CALL(AMGX_config_create_from_parameters_and_handle(&cfg, config_string));

    // Create resources, matrix, vector objects
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;

    // Get current CUDA device
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));

    // Create resources
    AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));

    // Create matrix and vectors
    AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, cfg));
    AMGX_SAFE_CALL(AMGX_vector_create(&x, rsrc, cfg));
    AMGX_SAFE_CALL(AMGX_vector_create(&b, rsrc, cfg));

    // Upload matrix in CSR format
    AMGX_SAFE_CALL(AMGX_matrix_upload_all(A, n, nnz, 1, 1,
                                         csr_offsets_h.data(),
                                         csr_columns_h.data(),
                                         csr_values_h.data(),
                                         NULL));  // No diagonal explicitly provided

    // Create solver
    AMGX_solver_handle solver;
    AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, cfg, NULL));

    // Setup the solver (symbolic analysis and numeric factorization)
    AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));

    // Upload vectors
    std::vector<T> x_values_h(n, 0.0);  // Initial guess
    AMGX_SAFE_CALL(AMGX_vector_upload(b, n, 1, b_values_h.data()));
    AMGX_SAFE_CALL(AMGX_vector_upload(x, n, 1, x_values_h.data()));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    // Solve the system
    AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, x));

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time to solve: %f ms\n", milliseconds);

    // Get solver status
    AMGX_SOLVE_STATUS status;
    int iterations;
    double residual;
    AMGX_SAFE_CALL(AMGX_solver_get_status(solver, &status, &iterations, &residual));
    printf("\n=== Solver Statistics ===\n");
    printf("Status: %d\n", status);
    printf("Iterations: %d\n", iterations);
    printf("Final residual: %e\n", residual);
    printf("===========================\n\n");

    // Download solution
    AMGX_SAFE_CALL(AMGX_vector_download(x, x_values_h.data()));

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Clean up AMGX resources
    AMGX_SAFE_CALL(AMGX_solver_destroy(solver));
    AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
    AMGX_SAFE_CALL(AMGX_vector_destroy(x));
    AMGX_SAFE_CALL(AMGX_vector_destroy(b));
    AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

    // Finalize AMGX
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());

    // Read known solution for error calculation
    std::string dvFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Dv.dat";
    std::string dlFile = "data/ancf/" + std::to_string(num_spokes) + "/solve_2002_0_Dl.dat";
    std::vector<T> knownSolution = readKnownSolution<T>(dvFile, dlFile);

    // Calculate relative error
    T error_tolerance = use_double ? 1e-7 : 1e-5;
    T relError = calculateRelativeErrorRaw<T>(x_values_h.data(), knownSolution.data(), n);
    printf("Relative error: %e\n", relError);

    // Write solution to file
    std::string precision = use_double ? "double" : "float";
    std::string outputFile = "soln_amgx_" + precision + "_" + std::to_string(num_spokes) + ".dat";
    writeVectorToFile<T>(x_values_h, outputFile);

    if (relError > error_tolerance) {
        printf("Example FAILED: Relative error too large\n");
        return -1;
    }

    printf("Example PASSED\n");
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
        return solveWithAMGX<double>(num_spokes, true);
    }
    else {
        return solveWithAMGX<float>(num_spokes, false);
    }
} 