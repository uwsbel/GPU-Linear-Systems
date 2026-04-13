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
 *
 * This software contains source code provided by NVIDIA Corporation.
 */
/**
 * cuDSS Solver Helpers
 *
 * Author(s): Ganesh Arivoli, Huzaifa Unjhawala
 * Email(s): arivoli@wisc.edu, unjhawala@wisc.edu
 *
 * This header contains the shared cuDSS solve path, including device
 * setup, solver configuration, phase execution, optional repeated-solve
 * timing, and output/error reporting for the multi-rig benchmark cases.
 */
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "cudss.h"
#include "utils.h"

#define CHECK_CUDA(call, msg)                  \
    do                                         \
    {                                          \
        cudaError_t error__ = (call);          \
        if (error__ != cudaSuccess)            \
        {                                      \
            return fail_cuda((msg), error__);  \
        }                                      \
    } while (0)

#define CHECK_CUDSS(call, msg)                           \
    do                                                   \
    {                                                    \
        status = (call);                                 \
        if (status != CUDSS_STATUS_SUCCESS)              \
        {                                                \
            return fail_cudss((msg), status);            \
        }                                                \
    } while (0)

struct CudssRunOptions
{
    int num_rigs;
    int iterations;
    bool verbose_logging;
    std::string timing_log_file;
};

template <typename T>
int runCudss(const ProblemFiles &files, const CudssRunOptions &options)
{
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;
    const cudaDataType_t cuda_data_type = std::is_same<T, double>::value ? CUDA_R_64F : CUDA_R_32F;
    const Precision precision = std::is_same<T, double>::value ? Precision::Float64 : Precision::Float32;

    int *csr_offsets_d = nullptr;
    int *csr_columns_d = nullptr;
    T *csr_values_d = nullptr;
    T *x_values_d = nullptr;
    T *b_values_d = nullptr;

    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    cudssHandle_t handle = {};
    cudssConfig_t solver_config = {};
    cudssData_t solver_data = {};
    cudssMatrix_t matrix_a = {};
    cudssMatrix_t matrix_x = {};
    cudssMatrix_t matrix_b = {};

    auto cleanup = [&]() {
        if (matrix_a)
        {
            cudssMatrixDestroy(matrix_a);
        }
        if (matrix_b)
        {
            cudssMatrixDestroy(matrix_b);
        }
        if (matrix_x)
        {
            cudssMatrixDestroy(matrix_x);
        }
        if (solver_data && handle)
        {
            cudssDataDestroy(handle, solver_data);
        }
        if (solver_config)
        {
            cudssConfigDestroy(solver_config);
        }
        if (handle)
        {
            cudssDestroy(handle);
        }
        if (start)
        {
            cudaEventDestroy(start);
        }
        if (stop)
        {
            cudaEventDestroy(stop);
        }
        if (stream)
        {
            cudaStreamDestroy(stream);
        }
        if (csr_offsets_d)
        {
            cudaFree(csr_offsets_d);
        }
        if (csr_columns_d)
        {
            cudaFree(csr_columns_d);
        }
        if (csr_values_d)
        {
            cudaFree(csr_values_d);
        }
        if (x_values_d)
        {
            cudaFree(x_values_d);
        }
        if (b_values_d)
        {
            cudaFree(b_values_d);
        }
    };

    auto finalize = [&](int code) {
        cleanup();
        cudaDeviceReset();
        return code;
    };

    auto fail_cuda = [&](const char *msg, cudaError_t error_code) {
        std::cerr << "Example FAILED: CUDA API returned error = " << error_code << ", details: " << msg << std::endl;
        return finalize(-1);
    };

    auto fail_cudss = [&](const char *msg, cudssStatus_t error_code) {
        std::cerr << "Example FAILED: CUDSS call ended unsuccessfully with status = " << error_code
                  << ", details: " << msg << std::endl;
        return finalize(-2);
    };

    const CsrMatrix<T> matrix = readMatrixCSR<T>(files.matrix);
    const int n = matrix.n;
    const int nnz = static_cast<int>(matrix.values.size());
    std::cout << "Matrix read from file: dimension = " << n << " x " << n << ", nnz = " << nnz << std::endl;

    std::vector<T> b_values_h = readVector<T>(files.rhs);
    if (b_values_h.size() != static_cast<std::size_t>(n))
    {
        std::cerr << "Error: RHS vector size (" << b_values_h.size() << ") does not match matrix dimension (" << n
                  << ")" << std::endl;
        return -1;
    }

    std::vector<T> known_solution = readKnownSolution<T>(files.dv, files.dl);

    CHECK_CUDA(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)), "cudaMalloc for csr_offsets_d");
    CHECK_CUDA(cudaMalloc(&csr_columns_d, nnz * sizeof(int)), "cudaMalloc for csr_columns_d");
    CHECK_CUDA(cudaMalloc(&csr_values_d, nnz * sizeof(T)), "cudaMalloc for csr_values_d");
    CHECK_CUDA(cudaMalloc(&b_values_d, n * sizeof(T)), "cudaMalloc for b_values_d");
    CHECK_CUDA(cudaMalloc(&x_values_d, n * sizeof(T)), "cudaMalloc for x_values_d");

    CHECK_CUDA(cudaMemcpy(csr_offsets_d, matrix.row_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy for csr_offsets_d");
    CHECK_CUDA(cudaMemcpy(csr_columns_d, matrix.columns.data(), nnz * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy for csr_columns_d");
    CHECK_CUDA(cudaMemcpy(csr_values_d, matrix.values.data(), nnz * sizeof(T), cudaMemcpyHostToDevice),
               "cudaMemcpy for csr_values_d");
    CHECK_CUDA(cudaMemcpy(b_values_d, b_values_h.data(), n * sizeof(T), cudaMemcpyHostToDevice),
               "cudaMemcpy for b_values_d");

    CHECK_CUDA(cudaStreamCreate(&stream), "cudaStreamCreate");
    CHECK_CUDA(cudaEventCreate(&start), "cudaEventCreate for start");
    CHECK_CUDA(cudaEventCreate(&stop), "cudaEventCreate for stop");

    if (options.verbose_logging)
    {
        if (setenv("CUDSS_LOG_LEVEL", "5", 1) != 0)
        {
            std::cerr << "Warning: Failed to set CUDSS_LOG_LEVEL environment variable" << std::endl;
        }
        else
        {
            std::cout << "Verbose logging enabled (CUDSS_LOG_LEVEL=5)" << std::endl;
        }
    }

    CHECK_CUDSS(cudssCreate(&handle), "cudssCreate");
    CHECK_CUDSS(cudssSetStream(handle, stream), "cudssSetStream");
    CHECK_CUDSS(cudssConfigCreate(&solver_config), "cudssConfigCreate");
    CHECK_CUDSS(cudssDataCreate(handle, &solver_data), "cudssDataCreate");

    cudssAlgType_t reordering_alg = CUDSS_ALG_DEFAULT;
    CHECK_CUDSS(cudssConfigSet(solver_config, CUDSS_CONFIG_REORDERING_ALG, &reordering_alg, sizeof(reordering_alg)),
                "cudssConfigSet for CUDSS_CONFIG_REORDERING_ALG");

    cudssAlgType_t pivot_epsilon_alg = CUDSS_ALG_DEFAULT;
    CHECK_CUDSS(
        cudssConfigSet(solver_config, CUDSS_CONFIG_PIVOT_EPSILON_ALG, &pivot_epsilon_alg, sizeof(pivot_epsilon_alg)),
        "cudssConfigSet for CUDSS_CONFIG_PIVOT_EPSILON_ALG");

    T pivot_epsilon = std::is_same<T, double>::value ? static_cast<T>(1e-8) : static_cast<T>(1e-4);
    std::cout << "Setting pivot epsilon to: " << static_cast<double>(pivot_epsilon) << std::endl;
    CHECK_CUDSS(cudssConfigSet(solver_config, CUDSS_CONFIG_PIVOT_EPSILON, &pivot_epsilon, sizeof(pivot_epsilon)),
                "cudssConfigSet for CUDSS_CONFIG_PIVOT_EPSILON");

    int solve_mode = 0;
    CHECK_CUDSS(cudssConfigSet(solver_config, CUDSS_CONFIG_SOLVE_MODE, &solve_mode, sizeof(solve_mode)),
                "cudssConfigSet for CUDSS_CONFIG_SOLVE_MODE");

    int iter_refinement = 0;
    CHECK_CUDSS(cudssConfigSet(solver_config, CUDSS_CONFIG_IR_N_STEPS, &iter_refinement, sizeof(iter_refinement)),
                "cudssConfigSet for CUDSS_CONFIG_IR_N_STEPS");

    cudssPivotType_t pivot_type = CUDSS_PIVOT_COL;
    CHECK_CUDSS(cudssConfigSet(solver_config, CUDSS_CONFIG_PIVOT_TYPE, &pivot_type, sizeof(pivot_type)),
                "cudssConfigSet for CUDSS_CONFIG_PIVOT_TYPE");

    T pivot_threshold = static_cast<T>(1.0);
    CHECK_CUDSS(cudssConfigSet(solver_config, CUDSS_CONFIG_PIVOT_THRESHOLD, &pivot_threshold, sizeof(pivot_threshold)),
                "cudssConfigSet for CUDSS_CONFIG_PIVOT_THRESHOLD");

    int hybrid_mode = 0;
    CHECK_CUDSS(cudssConfigSet(solver_config, CUDSS_CONFIG_HYBRID_MODE, &hybrid_mode, sizeof(hybrid_mode)),
                "cudssConfigSet for CUDSS_CONFIG_HYBRID_MODE");

    int hybrid_execute_mode = 0;
    CHECK_CUDSS(cudssConfigSet(
                    solver_config, CUDSS_CONFIG_HYBRID_EXECUTE_MODE, &hybrid_execute_mode, sizeof(hybrid_execute_mode)),
                "cudssConfigSet for CUDSS_CONFIG_HYBRID_EXECUTE_MODE");

    const int nrhs = 1;
    const int64_t nrows = n;
    const int64_t ncols = n;
    const int ldb = n;
    const int ldx = n;

    CHECK_CUDSS(cudssMatrixCreateDn(&matrix_b, ncols, nrhs, ldb, b_values_d, cuda_data_type, CUDSS_LAYOUT_COL_MAJOR),
                "cudssMatrixCreateDn for b");
    CHECK_CUDSS(cudssMatrixCreateDn(&matrix_x, nrows, nrhs, ldx, x_values_d, cuda_data_type, CUDSS_LAYOUT_COL_MAJOR),
                "cudssMatrixCreateDn for x");

    cudssMatrixType_t matrix_type = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t matrix_view = CUDSS_MVIEW_FULL;
    cudssIndexBase_t index_base = CUDSS_BASE_ZERO;
    CHECK_CUDSS(cudssMatrixCreateCsr(&matrix_a,
                                     nrows,
                                     ncols,
                                     nnz,
                                     csr_offsets_d,
                                     nullptr,
                                     csr_columns_d,
                                     csr_values_d,
                                     CUDA_R_32I,
                                     cuda_data_type,
                                     matrix_type,
                                     matrix_view,
                                     index_base),
                "cudssMatrixCreateCsr");

    auto elapsed_ms = [&](float &value) -> int {
        CHECK_CUDA(cudaEventSynchronize(stop), "cudaEventSynchronize");
        CHECK_CUDA(cudaEventElapsedTime(&value, start, stop), "cudaEventElapsedTime");
        return 0;
    };

    auto execute_phase = [&](cudssPhase_t phase_id, const char *label, float &elapsed_value) -> int {
        CHECK_CUDA(cudaEventRecord(start), "cudaEventRecord start");
        CHECK_CUDSS(cudssExecute(handle, phase_id, solver_config, solver_data, matrix_a, matrix_x, matrix_b), label);
        CHECK_CUDA(cudaEventRecord(stop), "cudaEventRecord stop");
        return elapsed_ms(elapsed_value);
    };

    auto copy_solution_to_host = [&](std::vector<T> &x_values_h) -> int {
        CHECK_CUDA(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        CHECK_CUDA(cudaMemcpy(x_values_h.data(), x_values_d, nrhs * n * sizeof(T), cudaMemcpyDeviceToHost),
                   "cudaMemcpy for x_values");
        return 0;
    };

    float analysis_time_ms = 0.0f;
    int phase_result = execute_phase(CUDSS_PHASE_ANALYSIS, "cudssExecute for analysis", analysis_time_ms);
    if (phase_result != 0)
    {
        return phase_result;
    }

    float factorization_time_ms = 0.0f;
    phase_result = execute_phase(CUDSS_PHASE_FACTORIZATION, "cudssExecute for factorization", factorization_time_ms);
    if (phase_result != 0)
    {
        return phase_result;
    }

    std::vector<T> x_values_h(n, static_cast<T>(0.0));

    auto print_solver_statistics = [&]() -> int {
        std::cout << "\n=== Solver Statistics ===" << std::endl;

        int info = 0;
        CHECK_CUDSS(cudssDataGet(handle, solver_data, CUDSS_DATA_INFO, &info, sizeof(info), nullptr),
                    "cudssDataGet for CUDSS_DATA_INFO");
        std::cout << "CUDSS_DATA_INFO: " << info << std::endl;

        int64_t lu_nnz = 0;
        CHECK_CUDSS(cudssDataGet(handle, solver_data, CUDSS_DATA_LU_NNZ, &lu_nnz, sizeof(lu_nnz), nullptr),
                    "cudssDataGet for CUDSS_DATA_LU_NNZ");
        std::cout << "Number of non-zeros in LU factors: " << static_cast<long long>(lu_nnz) << std::endl;

        int num_pivots = 0;
        CHECK_CUDSS(cudssDataGet(handle, solver_data, CUDSS_DATA_NPIVOTS, &num_pivots, sizeof(num_pivots), nullptr),
                    "cudssDataGet for CUDSS_DATA_NPIVOTS");
        std::cout << "Number of pivots: " << num_pivots << std::endl;

        int64_t memory_estimates[16] = {0};
        CHECK_CUDSS(cudssDataGet(handle,
                                 solver_data,
                                 CUDSS_DATA_MEMORY_ESTIMATES,
                                 memory_estimates,
                                 sizeof(memory_estimates),
                                 nullptr),
                    "cudssDataGet for CUDSS_DATA_MEMORY_ESTIMATES");
        std::cout << "Permanent device memory: " << static_cast<double>(memory_estimates[0]) / (1024 * 1024 * 1024)
                  << " GB" << std::endl;
        std::cout << "Peak device memory: " << static_cast<double>(memory_estimates[1]) / (1024 * 1024 * 1024)
                  << " GB" << std::endl;
        std::cout << "Permanent host memory: " << static_cast<double>(memory_estimates[2]) / (1024 * 1024 * 1024)
                  << " GB" << std::endl;
        std::cout << "Peak host memory: " << static_cast<double>(memory_estimates[3]) / (1024 * 1024 * 1024)
                  << " GB" << std::endl;
        std::cout << "Minimum device memory (hybrid mode): "
                  << static_cast<double>(memory_estimates[4]) / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "Maximum host memory (hybrid mode): "
                  << static_cast<double>(memory_estimates[5]) / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "===========================\n" << std::endl;

        return 0;
    };

    auto run_single_solve = [&]() -> int {
        float solve_time_ms = 0.0f;
        int phase_result = execute_phase(CUDSS_PHASE_SOLVE, "cudssExecute for solve", solve_time_ms);
        if (phase_result != 0)
        {
            return phase_result;
        }

        std::cout << "Analysis time: " << analysis_time_ms << " ms" << std::endl;
        std::cout << "Factorization time: " << factorization_time_ms << " ms" << std::endl;
        std::cout << "Solve time: " << solve_time_ms << " ms" << std::endl;
        std::cout << "Total time: " << (analysis_time_ms + factorization_time_ms + solve_time_ms) << " ms"
                  << std::endl;

        phase_result = copy_solution_to_host(x_values_h);
        if (phase_result != 0)
        {
            return phase_result;
        }

        phase_result = print_solver_statistics();
        if (phase_result != 0)
        {
            return phase_result;
        }

        const T relative_error = calculateRelativeErrorRaw<T>(x_values_h.data(), known_solution.data(), n);
        const T backward_error = calculateBackwardError<T>(matrix, x_values_h, b_values_h);
        std::cout << "Relative error: " << relative_error << std::endl;
        std::cout << "Backward error: " << backward_error << std::endl;

        writeVectorToFile<T>(x_values_h, files.output);
        if (!options.timing_log_file.empty())
        {
            appendRunTimingLog(options.timing_log_file,
                               options.num_rigs,
                               precision,
                               analysis_time_ms,
                               factorization_time_ms,
                               solve_time_ms,
                               relative_error,
                               backward_error);
        }

        std::cout << "Example PASSED" << std::endl;
        return finalize(0);
    };

    auto run_repeated_solves = [&]() -> int {
        std::vector<double> analysis_times(options.iterations, analysis_time_ms);
        std::vector<double> factorization_times(options.iterations, factorization_time_ms);
        std::vector<double> solve_times(options.iterations, 0.0);
        std::vector<T> relative_errors(options.iterations);
        std::vector<T> backward_errors(options.iterations);

        std::cout << "Running with " << precisionToString(precision) << " precision for " << options.iterations
                  << " iterations" << std::endl;
        std::cout << "Analysis time: " << analysis_time_ms << " ms (constant)" << std::endl;
        std::cout << "Factorization time: " << factorization_time_ms << " ms (constant)" << std::endl;

        for (int iter = 0; iter < options.iterations; ++iter)
        {
            std::cout << "\n--- Solve Iteration " << (iter + 1) << " ---" << std::endl;

            float solve_time_ms = 0.0f;
            int phase_result = execute_phase(CUDSS_PHASE_SOLVE, "cudssExecute for solve", solve_time_ms);
            if (phase_result != 0)
            {
                return phase_result;
            }
            solve_times[iter] = solve_time_ms;

            phase_result = copy_solution_to_host(x_values_h);
            if (phase_result != 0)
            {
                return phase_result;
            }

            relative_errors[iter] = calculateRelativeErrorRaw<T>(x_values_h.data(), known_solution.data(), n);
            backward_errors[iter] = calculateBackwardError<T>(matrix, x_values_h, b_values_h);

            std::cout << "Solve time: " << solve_times[iter] << " ms" << std::endl;
            std::cout << "Total time: " << (analysis_time_ms + factorization_time_ms + solve_times[iter]) << " ms"
                      << std::endl;
            std::cout << "Relative error: " << relative_errors[iter] << std::endl;
            std::cout << "Backward error: " << backward_errors[iter] << std::endl;
        }

        writeVectorToFile<T>(x_values_h, files.output);
        if (!options.timing_log_file.empty())
        {
            appendLoopTimingLog(options.timing_log_file,
                                options.num_rigs,
                                precision,
                                analysis_times,
                                factorization_times,
                                solve_times,
                                relative_errors,
                                backward_errors);
        }

        return finalize(0);
    };

    if (options.iterations <= 1)
    {
        return run_single_solve();
    }

    return run_repeated_solves();
}

#undef CHECK_CUDA
#undef CHECK_CUDSS
