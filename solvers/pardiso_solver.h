/**
 * PARDISO Solver Helpers
 *
 * Author(s): Ganesh Arivoli, Huzaifa Unjhawala
 * Email(s): arivoli@wisc.edu, unjhawala@wisc.edu
 *
 * This header contains the shared PARDISO solve path, including MKL
 * parameter initialization, phase timing, solver statistics, and
 * output/error reporting for the multi-rig benchmark cases.
 */
#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>

#include <mkl.h>
#include <mkl_pardiso.h>

#include "utils.h"

struct PardisoRunOptions
{
    int num_threads;
    int num_rigs;
    std::string timing_log_file;
};

template <typename T>
void initializePardisoParams(MKL_INT mtype, MKL_INT (&iparm)[64])
{
    const bool symmetric = std::abs(mtype) < 10;

    iparm[0] = 1;
    iparm[1] = 2;
    iparm[2] = 0;
    iparm[3] = 0;
    iparm[4] = 0;
    iparm[5] = 0;
    iparm[6] = 0;
    iparm[7] = 0;
    iparm[8] = 0;
    iparm[9] = 13;
    iparm[10] = symmetric ? 0 : 1;
    iparm[11] = 0;
    iparm[12] = symmetric ? 0 : 1;
    iparm[13] = 0;
    iparm[14] = 0;
    iparm[15] = 0;
    iparm[16] = 0;
    iparm[17] = -1;
    iparm[18] = -1;
    iparm[19] = 0;
    iparm[20] = 0;
    iparm[26] = 0;
    iparm[27] = (sizeof(T) == 4) ? 1 : 0;
    iparm[34] = 1;
    iparm[36] = 0;
    iparm[59] = 0;
}

template <typename T>
int runPardiso(const ProblemFiles &files, const PardisoRunOptions &options)
{
    mkl_set_num_threads(options.num_threads);
    mkl_free_buffers();
    mkl_thread_free_buffers();

    MKL_INT mtype = 11;
    MKL_INT nrhs = 1;
    void *pt[64] = {0};
    MKL_INT iparm[64] = {0};
    MKL_INT maxfct = 1;
    MKL_INT mnum = 1;
    MKL_INT msglvl = 0;
    MKL_INT error = 0;
    MKL_INT phase = 0;
    bool solver_active = false;
    CsrMatrix<T> matrix;
    int n = 0;
    std::vector<T> b;
    std::vector<T> x;

    initializePardisoParams<T>(mtype, iparm);

    auto release_solver = [&]() {
        if (!solver_active)
        {
            return;
        }

        phase = -1;
        pardiso(pt,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &n,
                matrix.values.data(),
                matrix.row_offsets.data(),
                matrix.columns.data(),
                nullptr,
                &nrhs,
                iparm,
                &msglvl,
                b.data(),
                x.data(),
                &error);
        solver_active = false;
        mkl_free_buffers();
        mkl_thread_free_buffers();
    };

    matrix = readMatrixCSR<T>(files.matrix);
    n = matrix.n;
    b = readVector<T>(files.rhs);
    std::vector<T> known_solution = readKnownSolution<T>(files.dv, files.dl);

    std::cout << "Matrix A dimensions: " << n << " x " << n << std::endl;
    std::cout << "Non-zero elements: " << matrix.values.size() << std::endl;
    std::cout << "Vector b size: " << b.size() << std::endl;
    std::cout << "Known solution size: " << known_solution.size() << std::endl;

    if (b.size() != static_cast<std::size_t>(n) || known_solution.size() != static_cast<std::size_t>(n))
    {
        std::cerr << "Error: Matrix and vector dimensions are inconsistent" << std::endl;
        return 1;
    }

    x.assign(n, static_cast<T>(0.0));

    auto run_phase = [&](MKL_INT phase_id, const char *label, double &elapsed_ms) -> bool {
        const auto start = std::chrono::high_resolution_clock::now();
        phase = phase_id;
        pardiso(pt,
                &maxfct,
                &mnum,
                &mtype,
                &phase,
                &n,
                matrix.values.data(),
                matrix.row_offsets.data(),
                matrix.columns.data(),
                nullptr,
                &nrhs,
                iparm,
                &msglvl,
                b.data(),
                x.data(),
                &error);
        elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start)
                         .count();

        if (error != 0)
        {
            std::cerr << "ERROR during " << label << ": " << error << std::endl;
            return false;
        }

        return true;
    };

    double analysis_time_ms = 0.0;
    double factorization_time_ms = 0.0;
    double solve_time_ms = 0.0;

    solver_active = true;
    if (!run_phase(11, "analysis", analysis_time_ms) || !run_phase(22, "factorization", factorization_time_ms) ||
        !run_phase(33, "solve", solve_time_ms))
    {
        return 1;
    }

    std::cout << "Analysis time: " << analysis_time_ms << " ms" << std::endl;
    std::cout << "Factorization time: " << factorization_time_ms << " ms" << std::endl;
    std::cout << "Solve time: " << solve_time_ms << " ms" << std::endl;
    std::cout << "Total time: " << (analysis_time_ms + factorization_time_ms + solve_time_ms) << " ms" << std::endl;
    std::cout << "\n=== Solver Statistics ===" << std::endl;
    std::cout << "Number of nonzeros in LU factors: " << iparm[17] << std::endl;
    std::cout << "Mflops for LU factorization: " << iparm[18] << std::endl;
    std::cout << "Number of perturbed pivots: " << iparm[13] << std::endl;
    std::cout << "===========================" << std::endl << std::endl;

    const T relative_error = calculateRelativeError<T>(x, known_solution);
    const T backward_error = calculateBackwardError<T>(matrix, x, b);

    std::cout << "Precision: " << (sizeof(T) == 4 ? "float" : "double") << std::endl;
    std::cout << "First element: " << x.front() << std::endl;
    std::cout << "Last element: " << x.back() << std::endl;
    std::cout << "Relative Error: " << relative_error << std::endl;
    std::cout << "Backward Error: " << backward_error << std::endl;
    writeVectorToFile<T>(x, files.output);

    if (!options.timing_log_file.empty())
    {
        appendRunTimingLog(options.timing_log_file,
                           options.num_rigs,
                           sizeof(T) == 4 ? Precision::Float32 : Precision::Float64,
                           analysis_time_ms,
                           factorization_time_ms,
                           solve_time_ms,
                           relative_error,
                           backward_error);
    }

    release_solver();
    return 0;
}
