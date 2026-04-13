/**
 * Shared Linear Solver Utilities
 *
 * Author(s): Ganesh Arivoli, Huzaifa Unjhawala
 * Email(s): arivoli@wisc.edu, unjhawala@wisc.edu
 *
 * This header provides shared utilities for matrix and vector I/O,
 * multi-rig dataset path construction, timing log helpers, precision
 * parsing, and error metrics used across the solver drivers.
 */
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <vector>

// Add CUDA compatibility
#ifdef __CUDACC__
#define UTILS_HOST_DEVICE __host__ __device__
#define UTILS_HOST __host__
#else
#define UTILS_HOST_DEVICE
#define UTILS_HOST
#endif

// Core types
enum class Precision
{
    Float32,
    Float64,
};

struct ProblemFiles
{
    std::string matrix;
    std::string rhs;
    std::string dv;
    std::string dl;
    std::string output;
};

template <typename T>
struct CsrMatrix
{
    std::vector<T> values;
    std::vector<int> row_offsets;
    std::vector<int> columns;
    int n = 0;
};

// Parsing helpers
inline const char *precisionToString(Precision precision)
{
    return precision == Precision::Float32 ? "float" : "double";
}

inline bool tryParsePrecision(const std::string &value, Precision &precision)
{
    if (value == "float")
    {
        precision = Precision::Float32;
        return true;
    }

    if (value == "double")
    {
        precision = Precision::Float64;
        return true;
    }

    return false;
}

inline bool tryParsePositiveInt(const std::string &value, int &parsed_value)
{
    try
    {
        parsed_value = std::stoi(value);
    }
    catch (...)
    {
        return false;
    }

    return parsed_value > 0;
}

static const std::array<int, 6> kSupportedMultiRigCounts = {1, 2, 4, 8, 10, 25};

// Dataset and path helpers
inline bool isSupportedMultiRigCount(int num_rigs)
{
    return std::find(kSupportedMultiRigCounts.begin(), kSupportedMultiRigCounts.end(), num_rigs) !=
           kSupportedMultiRigCounts.end();
}

inline std::string supportedMultiRigValues()
{
    std::ostringstream stream;
    for (std::size_t i = 0; i < kSupportedMultiRigCounts.size(); ++i)
    {
        if (i != 0)
        {
            stream << ", ";
        }
        stream << kSupportedMultiRigCounts[i];
    }
    return stream.str();
}

inline void ensureDirectoryExists(const std::string &dir)
{
    if (dir.empty())
    {
        return;
    }

    struct stat st = {0};
    if (stat(dir.c_str(), &st) == -1)
    {
        mkdir(dir.c_str(), 0755);
    }
}

inline std::string currentTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t raw_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time = *std::localtime(&raw_time);

    std::ostringstream stream;
    stream << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");
    return stream.str();
}

inline ProblemFiles getMultiRigCaseFiles(int num_rigs, const std::string &solver_name, Precision precision)
{
    const std::string base_dir = "data/ancf/multi_rig/" + std::to_string(num_rigs) + "_rigs";
    const std::string base_name = "solve_201_0_";
    const std::string output =
        "output/soln_" + solver_name + "_" + precisionToString(precision) + "_" + std::to_string(num_rigs) +
        "_rigs.dat";

    return {
        base_dir + "/" + base_name + "Z.dat",
        base_dir + "/" + base_name + "rhs.dat",
        base_dir + "/" + base_name + "Dv.dat",
        base_dir + "/" + base_name + "Dl.dat",
        output,
    };
}

// Logging helpers
inline std::ofstream openTimingLog(const std::string &log_file, const char *header)
{
    const std::size_t slash = log_file.find_last_of('/');
    if (slash != std::string::npos)
    {
        ensureDirectoryExists(log_file.substr(0, slash));
    }

    const bool file_exists = (access(log_file.c_str(), F_OK) == 0);
    std::ofstream log(log_file, std::ios::app);
    if (!log.is_open())
    {
        std::cerr << "Warning: Could not open log file for writing: " << log_file << std::endl;
        return log;
    }

    if (!file_exists)
    {
        log << header << "\n";
    }

    return log;
}

template <typename T>
void appendRunTimingLog(const std::string &log_file,
                        int num_rigs,
                        Precision precision,
                        double analysis_time_ms,
                        double factorization_time_ms,
                        double solve_time_ms,
                        T relative_error,
                        T backward_error)
{
    std::ofstream log = openTimingLog(
        log_file,
        "timestamp,num_rigs,precision,analysis_time_ms,factorization_time_ms,solve_time_ms,total_time_ms,"
        "relative_error,backward_error");
    if (!log.is_open())
    {
        return;
    }

    log << currentTimestamp() << ","
        << num_rigs << ","
        << precisionToString(precision) << ","
        << std::fixed << std::setprecision(6)
        << analysis_time_ms << ","
        << factorization_time_ms << ","
        << solve_time_ms << ","
        << (analysis_time_ms + factorization_time_ms + solve_time_ms) << ","
        << std::scientific << std::setprecision(6)
        << relative_error << ","
        << backward_error << "\n";
}

template <typename T>
void appendLoopTimingLog(const std::string &log_file,
                         int num_rigs,
                         Precision precision,
                         const std::vector<double> &analysis_times,
                         const std::vector<double> &factorization_times,
                         const std::vector<double> &solve_times,
                         const std::vector<T> &relative_errors,
                         const std::vector<T> &backward_errors)
{
    if (analysis_times.size() != factorization_times.size() ||
        analysis_times.size() != solve_times.size() ||
        analysis_times.size() != relative_errors.size() ||
        analysis_times.size() != backward_errors.size())
    {
        std::cerr << "Warning: Skipping loop timing log due to mismatched vector sizes" << std::endl;
        return;
    }

    std::ofstream log = openTimingLog(
        log_file,
        "timestamp,num_rigs,precision,iteration,analysis_time_ms,factorization_time_ms,solve_time_ms,"
        "total_time_ms,relative_error,backward_error");
    if (!log.is_open())
    {
        return;
    }

    const std::string timestamp = currentTimestamp();
    for (std::size_t i = 0; i < analysis_times.size(); ++i)
    {
        log << timestamp << ","
            << num_rigs << ","
            << precisionToString(precision) << ","
            << (i + 1) << ","
            << std::fixed << std::setprecision(6)
            << analysis_times[i] << ","
            << factorization_times[i] << ","
            << solve_times[i] << ","
            << (analysis_times[i] + factorization_times[i] + solve_times[i]) << ","
            << std::scientific << std::setprecision(6)
            << relative_errors[i] << ","
            << backward_errors[i] << "\n";
    }
}

// Matrix and vector I/O
template <typename T>
CsrMatrix<T> readMatrixCSR(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file " + filename);
    }

    // Read all triplets first to determine matrix size
    std::vector<std::tuple<int, int, double>> triplets;
    int row, col;
    double value;
    int max_row = 0, max_col = 0;

    // Read all entries
    while (file >> row >> col >> value)
    {
        // Convert from 1-based to 0-based indexing if needed
        row--;
        col--;

        // Keep track of matrix dimensions
        max_row = std::max(max_row, row);
        max_col = std::max(max_col, col);

        triplets.emplace_back(row, col, value);
    }

    // Matrix dimensions are max indices + 1 (since we converted to 0-based)
    const int n = max_row + 1;

    // Check if matrix is square
    if (max_row != max_col)
    {
        throw std::runtime_error("Matrix is not square in " + filename);
    }

    // Sort triplets by row, then by column for CSR format
    std::sort(triplets.begin(), triplets.end());

    CsrMatrix<T> matrix;
    matrix.values.resize(triplets.size());
    matrix.columns.resize(triplets.size());
    matrix.row_offsets.resize(n + 1, 0);
    matrix.n = n;

    // Fill in the CSR arrays
    int current_row = -1;
    for (size_t i = 0; i < triplets.size(); i++)
    {
        int row = std::get<0>(triplets[i]);
        int col = std::get<1>(triplets[i]);
        double val = std::get<2>(triplets[i]);

        // Update row index array
        while (current_row < row)
        {
            current_row++;
            matrix.row_offsets[current_row] = i;
        }

        // Store column index and value
        matrix.columns[i] = col;
        matrix.values[i] = static_cast<T>(val);
    }

    matrix.row_offsets[n] = triplets.size();
    return matrix;
}

template <typename T>
std::vector<T> readVector(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::vector<T> values;
    double value;

    // Read all values from the file
    while (file >> value)
    {
        values.push_back(static_cast<T>(value));
    }

    // Check if we read anything
    if (values.empty())
    {
        std::cerr << "Warning: No data read from " << filename << std::endl;
    }

    file.close();
    return values;
}

template <typename T>
std::vector<T> readKnownSolution(const std::string &dvFilename, const std::string &dlFilename)
{
    std::vector<T> dvPart = readVector<T>(dvFilename);
    std::vector<T> dlPart = readVector<T>(dlFilename);

    // Negate dlPart before combining
    for (auto &val : dlPart)
    {
        val = -val;
    }

    // Create combined vector
    std::vector<T> solution;
    solution.reserve(dvPart.size() + dlPart.size());
    solution.insert(solution.end(), dvPart.begin(), dvPart.end());
    solution.insert(solution.end(), dlPart.begin(), dlPart.end());

    return solution;
}

template <typename T>
void writeVectorToFile(const std::vector<T> &vector, const std::string &filename)
{
    const std::size_t slash = filename.find_last_of('/');
    if (slash != std::string::npos)
    {
        ensureDirectoryExists(filename.substr(0, slash));
    }

    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file " + filename + " for writing");
    }

    // Set precision for output
    file.precision(16);
    file << std::scientific;

    // Write each element on a new line
    for (size_t i = 0; i < vector.size(); i++)
    {
        file << vector[i] << std::endl;
    }

    file.close();
    std::cout << "Solution written to " << filename << std::endl;
}

// Error metrics
template <typename T>
UTILS_HOST T calculateRelativeError(const std::vector<T> &computed, const std::vector<T> &reference)
{
    if (computed.size() != reference.size())
    {
        std::cerr << "Error: Vector sizes don't match for error calculation" << std::endl;
        return static_cast<T>(-1.0);
    }

    T norm_diff = static_cast<T>(0.0);
    T norm_ref = static_cast<T>(0.0);

    for (size_t i = 0; i < computed.size(); i++)
    {
        T diff = computed[i] - reference[i];
        norm_diff += diff * diff;
        norm_ref += reference[i] * reference[i];
    }

    return std::sqrt(norm_diff) / std::sqrt(norm_ref);
}

template <typename T>
UTILS_HOST_DEVICE T calculateRelativeErrorRaw(const T *computed, const T *reference, int size)
{
    T norm_diff = static_cast<T>(0.0);
    T norm_ref = static_cast<T>(0.0);

    for (int i = 0; i < size; i++)
    {
        T diff = computed[i] - reference[i];
        norm_diff += diff * diff;
        norm_ref += reference[i] * reference[i];
    }

    return std::sqrt(norm_diff) / std::sqrt(norm_ref);
}

template <typename T>
UTILS_HOST T calculateBackwardError(
    const std::vector<T> &values,
    const std::vector<int> &rowIndex,
    const std::vector<int> &columns,
    const std::vector<T> &x,
    const std::vector<T> &b);

template <typename T>
UTILS_HOST T calculateBackwardError(
    const CsrMatrix<T> &matrix,
    const std::vector<T> &x,
    const std::vector<T> &b)
{
    return calculateBackwardError(matrix.values, matrix.row_offsets, matrix.columns, x, b);
}

template <typename T>
UTILS_HOST T calculateBackwardError(
    const std::vector<T> &values,
    const std::vector<int> &rowIndex,
    const std::vector<int> &columns,
    const std::vector<T> &x,
    const std::vector<T> &b)
{
    int n = b.size();
    if (rowIndex.size() != n + 1 || x.size() != static_cast<std::size_t>(n))
    {
        std::cerr << "Error: Dimensions don't match for backward error calculation" << std::endl;
        return static_cast<T>(-1.0);
    }

    // Calculate residual r = b - Ax
    std::vector<T> r = b;
    for (int i = 0; i < n; i++)
    {
        for (int j = rowIndex[i]; j < rowIndex[i + 1]; j++)
        {
            int col = columns[j];
            r[i] -= values[j] * x[col];
        }
    }

    // Calculate ||r||_2
    T r_norm = static_cast<T>(0.0);
    for (int i = 0; i < n; i++)
    {
        r_norm += r[i] * r[i];
    }
    r_norm = std::sqrt(r_norm);

    // Calculate ||A||_F (Frobenius norm)
    T A_norm = static_cast<T>(0.0);
    for (size_t i = 0; i < values.size(); i++)
    {
        A_norm += values[i] * values[i];
    }
    A_norm = std::sqrt(A_norm);

    // Calculate ||x||_2
    T x_norm = static_cast<T>(0.0);
    for (int i = 0; i < n; i++)
    {
        x_norm += x[i] * x[i];
    }
    x_norm = std::sqrt(x_norm);

    // Calculate ||b||_2
    T b_norm = static_cast<T>(0.0);
    for (int i = 0; i < n; i++)
    {
        b_norm += b[i] * b[i];
    }
    b_norm = std::sqrt(b_norm);

    // Calculate backward error: ||r||_2 / (||A||_F * ||x||_2 + ||b||_2)
    T denominator = A_norm * x_norm + b_norm;
    return r_norm / denominator;
}
