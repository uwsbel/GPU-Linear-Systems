#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>  // For std::sort
#include <mkl.h>
#include <mkl_pardiso.h>

// Function to read matrix from file in COO format and convert to CSR format
void readMatrixCSR(const std::string& filename, 
                  std::vector<double>& values, 
                  std::vector<int>& rowIndex, 
                  std::vector<int>& columns,
                  int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    // Read all triplets first to determine matrix size
    std::vector<std::tuple<int, int, double>> triplets;
    int row, col;
    double value;
    int max_row = 0, max_col = 0;
    
    // Read all entries
    while (file >> row >> col >> value) {
        // Convert from 1-based to 0-based indexing if needed
        row--;
        col--;
        
        // Keep track of matrix dimensions
        max_row = std::max(max_row, row);
        max_col = std::max(max_col, col);
        
        triplets.emplace_back(row, col, value);
    }
    
    // Matrix dimensions are max indices + 1 (since we converted to 0-based)
    n = max_row + 1;
    
    // Check if matrix is square
    if (max_row != max_col) {
        std::cerr << "Error: Matrix is not square. Rows: " << max_row + 1 
                  << ", Cols: " << max_col + 1 << std::endl;
        exit(1);
    }
    
    // Sort triplets by row, then by column for CSR format
    std::sort(triplets.begin(), triplets.end());
    
    // Initialize CSR arrays
    values.resize(triplets.size());
    columns.resize(triplets.size());
    rowIndex.resize(n + 1, 0);
    
    // Fill in the CSR arrays
    int current_row = -1;
    for (size_t i = 0; i < triplets.size(); i++) {
        int row = std::get<0>(triplets[i]);
        int col = std::get<1>(triplets[i]);
        double val = std::get<2>(triplets[i]);
        
        // Update row index array
        while (current_row < row) {
            current_row++;
            rowIndex[current_row] = i;
        }
        
        // Store column index and value
        columns[i] = col;
        values[i] = val;
    }
    
    // Set the last element of rowIndex
    rowIndex[n] = triplets.size();
    
    file.close();
}

// Function to read vector from file
std::vector<double> readVector(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::vector<double> values;
    double value;
    
    // Read all values from the file
    while (file >> value) {
        values.push_back(value);
    }
    
    // Check if we read anything
    if (values.empty()) {
        std::cerr << "Warning: No data read from " << filename << std::endl;
    }
    
    file.close();
    return values;
}

// Function to read the known solution (combining Dl and Dv)
std::vector<double> readKnownSolution(const std::string& dvFilename, const std::string& dlFilename) {
    std::vector<double> dvPart = readVector(dvFilename);
    std::vector<double> dlPart = readVector(dlFilename);
    
    // Negate dlPart before combining
    for (auto& val : dlPart) {
        val = -val;
    }
    
    // Create combined vector
    std::vector<double> solution;
    solution.reserve(dvPart.size() + dlPart.size());
    solution.insert(solution.end(), dvPart.begin(), dvPart.end());
    solution.insert(solution.end(), dlPart.begin(), dlPart.end());
    
    return solution;
}

// Function to write vector to file
void writeVectorToFile(const std::vector<double>& vector, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        exit(1);
    }
    
    // Set precision for output
    file.precision(16);
    file << std::scientific;
    
    // Write each element on a new line
    for (size_t i = 0; i < vector.size(); i++) {
        file << vector[i] << std::endl;
    }
    
    file.close();
    std::cout << "Solution written to " << filename << std::endl;
}

// Calculate relative error between two vectors
double calculateRelativeError(const std::vector<double>& computed, const std::vector<double>& reference) {
    if (computed.size() != reference.size()) {
        std::cerr << "Error: Vector sizes don't match for error calculation" << std::endl;
        return -1.0;
    }
    
    double norm_diff = 0.0;
    double norm_ref = 0.0;
    
    for (size_t i = 0; i < computed.size(); i++) {
        double diff = computed[i] - reference[i];
        norm_diff += diff * diff;
        norm_ref += reference[i] * reference[i];
    }
    
    return std::sqrt(norm_diff) / std::sqrt(norm_ref);
}

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <num_threads>" << std::endl;
        return 1;
    }
    
    int num_threads = std::stoi(argv[1]);
    
    // Set the number of threads for MKL
    mkl_set_num_threads(num_threads);
    
    // Data file paths
    std::string matrixFile = "data/ancf/80/solve_2002_0_Z.dat";
    std::string rhsFile = "data/ancf/80/solve_2002_0_rhs.dat";
    std::string dvFile = "data/ancf/80/solve_2002_0_Dv.dat";
    std::string dlFile = "data/ancf/80/solve_2002_0_Dl.dat";
    std::string solnFile = "soln_pardiso_80.dat";
    
    // Read matrix in CSR format
    std::vector<double> values;     // Non-zero values
    std::vector<int> rowIndex;      // Row pointers
    std::vector<int> columns;       // Column indices
    int n;                          // Matrix dimension
    
    readMatrixCSR(matrixFile, values, rowIndex, columns, n);
    
    // Read RHS vector
    std::vector<double> b = readVector(rhsFile);
    
    // Read known solution for comparison
    std::vector<double> knownSolution = readKnownSolution(dvFile, dlFile);
    
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
    std::vector<double> x(n, 0.0);
    
    // PARDISO parameters
    MKL_INT mtype = 11;       // Real unsymmetric matrix
    MKL_INT nrhs = 1;         // Number of right hand sides
    void *pt[64] = {0};       // Internal solver memory pointer
    MKL_INT iparm[64] = {0};  // PARDISO control parameters
    MKL_INT maxfct = 1;       // Maximum number of numerical factorizations
    MKL_INT mnum = 1;         // Which factorization to use
    MKL_INT msglvl = 0;       // Print statistical information
    MKL_INT error = 0;        // Error indicator
    MKL_INT phase;            // Phase of calculation
    
    bool symmetric = std::abs(mtype) < 10;
    iparm[0] = 1;   // No solver default
    iparm[1] = 2;   // use Metis for the ordering
    iparm[2] = 0;   // Reserved. Set to zero.
    iparm[3] = 0;   // No iterative-direct algorithm
    iparm[4] = 0;   // No user fill-in reducing permutation
    iparm[5] = 0;   // Write solution into x, b is left unchanged
    iparm[6] = 0;   // Not in use
    iparm[7] = 2;   // Max numbers of iterative refinement steps
    iparm[8] = 0;   // Not in use
    iparm[9] = 13;  // Perturb the pivot elements with 1E-13
    iparm[10] = symmetric ? 0 : 1; // Use nonsymmetric permutation and scaling MPS
    iparm[11] = 0;  // Not in use
    iparm[12] = symmetric ? 0 : 1;  // Maximum weighted matching algorithm is switched-off (default for symmetric).
    iparm[13] = 0;  // Output: Number of perturbed pivots
    iparm[14] = 0;  // Not in use
    iparm[15] = 0;  // Not in use
    iparm[16] = 0;  // Not in use
    iparm[17] = -1; // Output: Number of nonzeros in the factor LU
    iparm[18] = -1; // Output: Mflops for LU factorization
    iparm[19] = 0;  // Output: Numbers of CG Iterations
    iparm[20] = 0;  // 1x1 pivoting
    iparm[26] = 0;  // No matrix checker
    iparm[27] = (sizeof(double) == 4) ? 1 : 0;
    iparm[34] = 1;  // C indexing
    iparm[36] = 0;  // CSR
    iparm[59] = 0;  // 0 - In-Core ; 1 - Automatic switch between In-Core and Out-of-Core modes ; 2 - Out-of-Core

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Analysis, numerical factorization, and solution
    phase = 13;  // Analysis + numerical factorization + solve
    
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, values.data(), rowIndex.data(), columns.data(),
            NULL, &nrhs, iparm, &msglvl, b.data(), x.data(), &error);
    
    if (error != 0) {
        std::cerr << "ERROR during solution: " << error << std::endl;
        return 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    
    // Calculate error compared to known solution
    double error_norm = calculateRelativeError(x, knownSolution);
    
    // Output first and last elements for verification, plus error
    std::cout << "First element: " << x[0] << std::endl;
    std::cout << "Last element: " << x[n-1] << std::endl;
    std::cout << "Relative Error: " << error_norm << std::endl;
    std::cout << "Time (ms): " << duration.count() << std::endl;
    
    // Write solution to file
    writeVectorToFile(x, solnFile);
    
    // Release memory
    phase = -1;  // Release internal memory
    pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, values.data(), rowIndex.data(), columns.data(),
            NULL, &nrhs, iparm, &msglvl, b.data(), x.data(), &error);
    
    return 0;
}