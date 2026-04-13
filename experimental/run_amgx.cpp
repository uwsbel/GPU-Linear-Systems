/*
 * Portions of this file derive from NVIDIA AMGX sample code.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * AMGX Driver
 *
 * Author(s): Ganesh Arivoli
 * Email(s): arivoli@wisc.edu
 *
 * This driver parses the fixed-order CLI for the AMGX backend,
 * validates the selected multi-rig case, and dispatches the solve in
 * single or double precision for the experimental path.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <amgx_c.h>

#include <iostream>
#include <string>
#include <vector>

#include "utils.h"

// Error checking macro for CUDA calls
#define CUDA_CHECK(call)                                                                         \
    do                                                                                           \
    {                                                                                            \
        cudaError_t err = call;                                                                  \
        if (err != cudaSuccess)                                                                  \
        {                                                                                        \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while (0)

namespace {

struct DriverArgs
{
    Precision precision = Precision::Float64;
    int num_rigs = 4;
};

void printUsage(const char *program_name)
{
    printf("Usage: %s --precision <float|double> --rigs <num_rigs>\n", program_name);
    printf("Supported num_rigs values: %s\n", supportedMultiRigValues().c_str());
}

bool isHelpRequest(int argc, char *argv[])
{
    return argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
}

bool parseArgs(int argc, char *argv[], DriverArgs &args)
{
    if (argc != 5 || std::string(argv[1]) != "--precision" || std::string(argv[3]) != "--rigs")
    {
        return false;
    }

    if (!tryParsePrecision(argv[2], args.precision))
    {
        printf("Error: --precision expects 'float' or 'double'\n");
        return false;
    }

    if (!tryParsePositiveInt(argv[4], args.num_rigs))
    {
        printf("Error: --rigs expects a positive integer\n");
        return false;
    }

    return true;
}

template <typename T>
int solveWithAMGX(int num_rigs, Precision precision)
{
    printf("Running with %s precision\n", precisionToString(precision));

    const ProblemFiles files = getMultiRigCaseFiles(num_rigs, "amgx", precision);

    const CsrMatrix<T> matrix = readMatrixCSR<T>(files.matrix);
    const int n = matrix.n;
    const int nnz = static_cast<int>(matrix.values.size());
    printf("Matrix read from file: dimension = %d x %d, nnz = %d\n", n, n, nnz);

    std::vector<T> b_values_h = readVector<T>(files.rhs);
    if (b_values_h.size() != static_cast<size_t>(n))
    {
        printf("Error: RHS vector size (%zu) does not match matrix dimension (%d)\n", b_values_h.size(), n);
        return -1;
    }

    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());

    AMGX_config_handle cfg;
    const char *config_string = nullptr;
    if (precision == Precision::Float64)
    {
        config_string = "config_version=2, solver=FGMRES, determinism_flag=1, "
                        "matrix_precision=DOUBLE, vector_precision=DOUBLE, "
                        "max_iters=1000, convergence=RELATIVE_INI_CORE, "
                        "tolerance=1e-7, norm=L2, "
                        "preconditioner(amg_solver)=AMG, "
                        "amg_solver:max_levels=100, amg_solver:cycle=V, "
                        "amg_solver:presweeps=1, amg_solver:postsweeps=1, "
                        "amg_solver:matrix_coloring_scheme=PARALLEL";
    }
    else
    {
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

    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b;
    AMGX_vector_handle x;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));
    AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, cfg));
    AMGX_SAFE_CALL(AMGX_vector_create(&x, rsrc, cfg));
    AMGX_SAFE_CALL(AMGX_vector_create(&b, rsrc, cfg));

    AMGX_SAFE_CALL(AMGX_matrix_upload_all(A,
                                          n,
                                          nnz,
                                          1,
                                          1,
                                          matrix.row_offsets.data(),
                                          matrix.columns.data(),
                                          matrix.values.data(),
                                          nullptr));

    AMGX_solver_handle solver;
    AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, cfg, nullptr));
    AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));

    std::vector<T> x_values_h(n, static_cast<T>(0.0));
    AMGX_SAFE_CALL(AMGX_vector_upload(b, n, 1, b_values_h.data()));
    AMGX_SAFE_CALL(AMGX_vector_upload(x, n, 1, x_values_h.data()));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    AMGX_SAFE_CALL(AMGX_solver_solve(solver, b, x));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time to solve: %f ms\n", milliseconds);

    AMGX_SOLVE_STATUS status;
    int iterations = 0;
    double residual = 0.0;
    AMGX_SAFE_CALL(AMGX_solver_get_status(solver, &status, &iterations, &residual));
    printf("\n=== Solver Statistics ===\n");
    printf("Status: %d\n", status);
    printf("Iterations: %d\n", iterations);
    printf("Final residual: %e\n", residual);
    printf("===========================\n\n");

    AMGX_SAFE_CALL(AMGX_vector_download(x, x_values_h.data()));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    AMGX_SAFE_CALL(AMGX_solver_destroy(solver));
    AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
    AMGX_SAFE_CALL(AMGX_vector_destroy(x));
    AMGX_SAFE_CALL(AMGX_vector_destroy(b));
    AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());

    const std::vector<T> known_solution = readKnownSolution<T>(files.dv, files.dl);
    const T error_tolerance = precision == Precision::Float64 ? static_cast<T>(1e-7) : static_cast<T>(1e-5);
    const T relative_error = calculateRelativeErrorRaw<T>(x_values_h.data(), known_solution.data(), n);
    printf("Relative error: %e\n", relative_error);

    writeVectorToFile<T>(x_values_h, files.output);

    if (relative_error > error_tolerance)
    {
        printf("Example FAILED: Relative error too large\n");
        return -1;
    }

    printf("Example PASSED\n");
    return 0;
}

}  // namespace

int main(int argc, char *argv[])
{
    try
    {
        if (isHelpRequest(argc, argv))
        {
            printUsage(argv[0]);
            return 0;
        }

        DriverArgs args;
        if (!parseArgs(argc, argv, args))
        {
            printUsage(argv[0]);
            return 1;
        }

        if (!isSupportedMultiRigCount(args.num_rigs))
        {
            printf("Error: Unsupported num_rigs value: %d. Supported values are %s.\n",
                   args.num_rigs,
                   supportedMultiRigValues().c_str());
            return 1;
        }

        if (args.precision == Precision::Float64)
        {
            return solveWithAMGX<double>(args.num_rigs, args.precision);
        }

        return solveWithAMGX<float>(args.num_rigs, args.precision);
    }
    catch (const std::exception &error_message)
    {
        std::cerr << "Error: " << error_message.what() << std::endl;
        return 1;
    }
}
