#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>  // For std::sort
#include <mkl.h>
#include <mkl_pardiso.h>
#include "utils.h"

// Template function to solve the linear system using PARDISO
template<typename T>
int solveWithPardiso(const std::string& matrixFile, const std::string& rhsFile, 
                     const std::string& dvFile, const std::string& dlFile, 
                     const std::string& solnFile, int num_threads, int n_expected = -1) {
    
    // Set the number of threads for MKL
    mkl_set_num_threads(num_threads);
    
    // Clear MKL memory pool and caches
    mkl_free_buffers();
    mkl_thread_free_buffers();

    // Read matrix in CSR format
    std::vector<T> values;       // Non-zero values
    std::vector<int> rowIndex;   // Row pointers
    std::vector<int> columns;    // Column indices
    int n;                       // Matrix dimension

    readMatrixCSR<T>(matrixFile, values, rowIndex, columns, n);

    // Read RHS vector
    std::vector<T> b = readVector<T>(rhsFile);

    // Read known solution for comparison
    std::vector<T> knownSolution = readKnownSolution<T>(dvFile, dlFile);

    // Print sizes for debugging
    std::cout << "Matrix A dimensions: " << n << " x " << n << std::endl;
    std::cout << "Non-zero elements: " << values.size() << std::endl;
    std::cout << "Vector b size: " << b.size() << std::endl;
    std::cout << "Known solution size: " << knownSolution.size() << std::endl;

    // Check dimensions for consistency
    if (b.size() != n || knownSolution.size() != n) {
        std::cerr << "Error: Matrix and vector dimensions are inconsistent" << std::endl;
        return 1;
    }

    // Prepare solution vector
    std::vector<T> x(n, 0.0);

    // PARDISO parameters
    MKL_INT mtype = 11;       // Real unsymmetric matrix
    MKL_INT nrhs = 1;         // Number of right hand sides
    void* pt[64] = {0};       // Internal solver memory pointer
    MKL_INT iparm[64] = {0};  // PARDISO control parameters
    MKL_INT maxfct = 1;       // Maximum number of numerical factorizations
    MKL_INT mnum = 1;         // Which factorization to use
    MKL_INT msglvl = 0;       // Print statistical information
    MKL_INT error = 0;        // Error indicator
    MKL_INT phase;            // Phase of calculation

    bool symmetric = std::abs(mtype) < 10;
    iparm[0] = 1;                   // No solver default
    iparm[1] = 2;                   // use Metis for the ordering
    iparm[2] = 0;                   // Reserved. Set to zero.
    iparm[3] = 0;                   // No iterative-direct algorithm
    iparm[4] = 0;                   // No user fill-in reducing permutation
    iparm[5] = 0;                   // Write solution into x, b is left unchanged
    iparm[6] = 0;                   // Not in use
    iparm[7] = 2;                   // Max numbers of iterative refinement steps
    iparm[8] = 0;                   // Not in use
    iparm[9] = 13;                  // Perturb the pivot elements with 1E-13
    iparm[10] = symmetric ? 0 : 1;  // Use nonsymmetric permutation and scaling MPS
    iparm[11] = 0;                  // Not in use
    iparm[12] = symmetric ? 0 : 1;  // Maximum weighted matching algorithm is switched-off (default for symmetric).
    iparm[13] = 0;                  // Output: Number of perturbed pivots
    iparm[14] = 0;                  // Not in use
    iparm[15] = 0;                  // Not in use
    iparm[16] = 0;                  // Not in use
    iparm[17] = -1;                 // Output: Number of nonzeros in the factor LU
    iparm[18] = -1;                 // Output: Mflops for LU factorization
    iparm[19] = 0;                  // Output: Numbers of CG Iterations
    iparm[20] = 0;                  // 1x1 pivoting
    iparm[26] = 0;                  // No matrix checker
    iparm[27] = (sizeof(T) == 4) ? 1 : 0;  // Use float or double precision
    iparm[34] = 1;  // C indexing
    iparm[36] = 0;  // CSR
    iparm[59] = 0;  // 0 - In-Core ; 1 - Automatic switch between In-Core and Out-of-Core modes ; 2 - Out-of-Core

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Analysis, numerical factorization, and solution
    phase = 13;  // Analysis + numerical factorization + solve

    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, values.data(), rowIndex.data(), columns.data(), NULL, &nrhs, iparm,
            &msglvl, b.data(), x.data(), &error);

    if (error != 0) {
        std::cerr << "ERROR during solution: " << error << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Calculate error compared to known solution
    T error_norm = calculateRelativeError<T>(x, knownSolution);

    // Calculate backward error (residual-based)
    T backward_error = calculateBackwardError<T>(values, rowIndex, columns, x, b);

    // Output first and last elements for verification, plus error
    std::cout << "Precision: " << (sizeof(T) == 4 ? "float" : "double") << std::endl;
    std::cout << "First element: " << x[0] << std::endl;
    std::cout << "Last element: " << x[n - 1] << std::endl;
    std::cout << "Relative Error: " << error_norm << std::endl;
    std::cout << "Backward Error: " << backward_error << std::endl;
    std::cout << "Time (ms): " << duration.count() << std::endl;

    // Write solution to file
    writeVectorToFile<T>(x, solnFile);

    // Release memory
    phase = -1;  // Release internal memory
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, values.data(), rowIndex.data(), columns.data(), NULL, &nrhs, iparm,
            &msglvl, b.data(), x.data(), &error);
    
    // Clear MKL memory pool and caches after computation
    mkl_free_buffers();
    mkl_thread_free_buffers();

    return 0;
}

int main(int argc, char* argv[]) {
    // Check command line arguments for num_threads and precision (required)
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <precision> [num_spokes]" << std::endl;
        std::cerr << "  num_threads: Number of threads for MKL (required)" << std::endl;
        std::cerr << "  precision: 'float' or 'double' (required)" << std::endl;
        std::cerr << "  num_spokes: Number of spokes for geometry (optional, default: 16)" << std::endl;
        return 1;
    }

    // Parse num_threads (first argument, required)
    int num_threads = std::stoi(argv[1]);
    if (num_threads <= 0) {
        std::cerr << "Error: num_threads must be a positive integer" << std::endl;
        return 1;
    }

    // Parse precision (second argument, required)
    std::string precision = argv[2];
    if (precision != "float" && precision != "double") {
        std::cerr << "Error: precision must be 'float' or 'double'" << std::endl;
        return 1;
    }

    // Parse num_spokes (third argument, optional with default value of 16)
    int num_spokes = 16;  // Default value
    if (argc > 3) {
        num_spokes = std::stoi(argv[3]);
        if (num_spokes <= 0) {
            std::cerr << "Error: num_spokes must be a positive integer" << std::endl;
            return 1;
        }
    } else {
        std::cout << "No num_spokes provided. Using default value = " << num_spokes << std::endl;
    }

    // Data file paths based on number of spokes
    std::string baseDir, baseName;
    if (num_spokes == 16) {
        baseDir = "data/ancf/refine1/16/";
        baseName = "2002";
    } else if (num_spokes == 80) {
        baseDir = "data/ancf/refine2/80/";
        baseName = "1001";
    } else {
        std::cerr << "Error: Unsupported number of spokes. Only 16 and 80 are supported." << std::endl;
        return 1;
    }
    
    std::string matrixFile = baseDir + "solve_" + baseName + "_0_Z.dat";
    std::string rhsFile = baseDir + "solve_" + baseName + "_0_rhs.dat";
    std::string dvFile = baseDir + "solve_" + baseName + "_0_Dv.dat";
    std::string dlFile = baseDir + "solve_" + baseName + "_0_Dl.dat";
    std::string solnFile = "soln_pardiso_" + precision + "_" + std::to_string(num_spokes) + ".dat";

    // Call the appropriate template instantiation based on precision
    if (precision == "float") {
        return solveWithPardiso<float>(matrixFile, rhsFile, dvFile, dlFile, solnFile, num_threads);
    } else {
        return solveWithPardiso<double>(matrixFile, rhsFile, dvFile, dlFile, solnFile, num_threads);
    }
}