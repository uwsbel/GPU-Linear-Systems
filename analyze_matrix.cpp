#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <chrono>

// Include Eigen for matrix analysis
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;

// Matrix analysis structure
struct MatrixAnalysis {
    int dimension;
    int nnz;
    double sparsity;
    double density;
    double condition_number;
    double min_eigenvalue;
    double max_eigenvalue;
    double spectral_radius;
    bool is_symmetric;
    bool is_positive_definite;
    bool is_diagonally_dominant;
    double diagonal_dominance_ratio;
    double max_off_diagonal;
    double min_diagonal;
    double max_diagonal;
    double frobenius_norm;
    double one_norm;
    double infinity_norm;
    vector<double> eigenvalue_distribution;
    map<int, int> row_nnz_distribution;
    map<int, int> column_nnz_distribution;
    vector<pair<int, int>> largest_elements;
    vector<pair<int, int>> smallest_elements;
};

/**
 * Reads a matrix in COO format from a file and converts it to CSR format.
 */
void readMatrixCSR(const string &filename,
                   vector<double> &values,
                   vector<int> &rowIndex,
                   vector<int> &columns,
                   int &n)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    // Read all triplets first to determine matrix size
    vector<tuple<int, int, double>> triplets;
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
        max_row = max(max_row, row);
        max_col = max(max_col, col);

        triplets.emplace_back(row, col, value);
    }

    // Matrix dimensions are max indices + 1 (since we converted to 0-based)
    n = max_row + 1;

    // Check if matrix is square
    if (max_row != max_col)
    {
        cerr << "Error: Matrix is not square. Rows: " << max_row + 1 << ", Cols: " << max_col + 1 << endl;
        exit(1);
    }

    // Sort triplets by row, then by column for CSR format
    sort(triplets.begin(), triplets.end());

    // Initialize CSR arrays
    values.resize(triplets.size());
    columns.resize(triplets.size());
    rowIndex.resize(n + 1, 0);

    // Fill in the CSR arrays
    int current_row = -1;
    for (size_t i = 0; i < triplets.size(); i++)
    {
        int row = get<0>(triplets[i]);
        int col = get<1>(triplets[i]);
        double val = get<2>(triplets[i]);

        // Update row index array
        while (current_row < row)
        {
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

/**
 * Converts CSR format to Eigen SparseMatrix
 */
Eigen::SparseMatrix<double> csrToEigenSparse(const vector<double> &values,
                                             const vector<int> &rowIndex,
                                             const vector<int> &columns,
                                             int n)
{
    Eigen::SparseMatrix<double> matrix(n, n);
    matrix.reserve(values.size());

    for (int i = 0; i < n; i++)
    {
        for (int j = rowIndex[i]; j < rowIndex[i + 1]; j++)
        {
            matrix.insert(i, columns[j]) = values[j];
        }
    }

    matrix.makeCompressed();
    return matrix;
}

/**
 * Analyzes matrix properties
 */
MatrixAnalysis analyzeMatrix(const Eigen::SparseMatrix<double> &matrix)
{
    MatrixAnalysis analysis;
    
    analysis.dimension = matrix.rows();
    analysis.nnz = matrix.nonZeros();
    analysis.sparsity = 1.0 - (double)analysis.nnz / (analysis.dimension * analysis.dimension);
    analysis.density = 1.0 - analysis.sparsity;
    
    // Check symmetry
    analysis.is_symmetric = (matrix - matrix.transpose()).norm() < 1e-12;
    
    // Analyze diagonal dominance
    analysis.is_diagonally_dominant = true;
    analysis.diagonal_dominance_ratio = 0.0;
    analysis.min_diagonal = numeric_limits<double>::max();
    analysis.max_diagonal = numeric_limits<double>::lowest();
    analysis.max_off_diagonal = 0.0;
    
    double total_diagonal_sum = 0.0;
    double total_off_diagonal_sum = 0.0;
    
    for (int i = 0; i < matrix.rows(); i++)
    {
        double row_diagonal = 0.0;
        double row_off_diagonal = 0.0;
        
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, i); it; ++it)
        {
            if (it.row() == it.col())
            {
                row_diagonal += abs(it.value());
                analysis.min_diagonal = min(analysis.min_diagonal, abs(it.value()));
                analysis.max_diagonal = max(analysis.max_diagonal, abs(it.value()));
            }
            else
            {
                row_off_diagonal += abs(it.value());
                analysis.max_off_diagonal = max(analysis.max_off_diagonal, abs(it.value()));
            }
        }
        
        total_diagonal_sum += row_diagonal;
        total_off_diagonal_sum += row_off_diagonal;
        
        if (row_off_diagonal > row_diagonal)
        {
            analysis.is_diagonally_dominant = false;
        }
    }
    
    analysis.diagonal_dominance_ratio = total_diagonal_sum / (total_diagonal_sum + total_off_diagonal_sum);
    
    // Calculate matrix norms
    analysis.frobenius_norm = matrix.norm();
    analysis.one_norm = matrix.colwise().norm().maxCoeff();
    analysis.infinity_norm = matrix.rowwise().norm().maxCoeff();
    
    // Analyze sparsity pattern
    for (int i = 0; i < matrix.rows(); i++)
    {
        int row_nnz = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, i); it; ++it)
        {
            row_nnz++;
        }
        analysis.row_nnz_distribution[row_nnz]++;
    }
    
    for (int j = 0; j < matrix.cols(); j++)
    {
        int col_nnz = 0;
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, j); it; ++it)
        {
            col_nnz++;
        }
        analysis.column_nnz_distribution[col_nnz]++;
    }
    
    // Find largest and smallest elements
    vector<pair<double, pair<int, int>>> elements;
    for (int i = 0; i < matrix.rows(); i++)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, i); it; ++it)
        {
            elements.push_back({abs(it.value()), {it.row(), it.col()}});
        }
    }
    
    sort(elements.begin(), elements.end(), greater<pair<double, pair<int, int>>>());
    
    for (int i = 0; i < min(10, (int)elements.size()); i++)
    {
        analysis.largest_elements.push_back(elements[i].second);
    }
    
    for (int i = max(0, (int)elements.size() - 10); i < elements.size(); i++)
    {
        analysis.smallest_elements.push_back(elements[i].second);
    }
    
    // Eigenvalue analysis (for smaller matrices or symmetric matrices)
    if (analysis.dimension <= 1000 || analysis.is_symmetric)
    {
        try
        {
            Eigen::MatrixXd dense_matrix = matrix.toDense();
            Eigen::EigenSolver<Eigen::MatrixXd> solver(dense_matrix);
            auto eigenvalues = solver.eigenvalues();
            
            analysis.min_eigenvalue = eigenvalues.real().minCoeff();
            analysis.max_eigenvalue = eigenvalues.real().maxCoeff();
            analysis.spectral_radius = eigenvalues.cwiseAbs().maxCoeff();
            
            // Check positive definiteness
            analysis.is_positive_definite = true;
            for (int i = 0; i < eigenvalues.size(); i++)
            {
                if (eigenvalues(i).real() <= 0)
                {
                    analysis.is_positive_definite = false;
                    break;
                }
            }
            
            // Calculate condition number
            if (abs(analysis.min_eigenvalue) > 1e-12)
            {
                analysis.condition_number = analysis.spectral_radius / abs(analysis.min_eigenvalue);
            }
            else
            {
                analysis.condition_number = numeric_limits<double>::infinity();
            }
            
            // Store eigenvalue distribution
            for (int i = 0; i < eigenvalues.size(); i++)
            {
                analysis.eigenvalue_distribution.push_back(eigenvalues(i).real());
            }
            sort(analysis.eigenvalue_distribution.begin(), analysis.eigenvalue_distribution.end());
        }
        catch (const exception &e)
        {
            cout << "Warning: Could not compute eigenvalues: " << e.what() << endl;
            analysis.min_eigenvalue = numeric_limits<double>::quiet_NaN();
            analysis.max_eigenvalue = numeric_limits<double>::quiet_NaN();
            analysis.spectral_radius = numeric_limits<double>::quiet_NaN();
            analysis.condition_number = numeric_limits<double>::quiet_NaN();
            analysis.is_positive_definite = false;
        }
    }
    else
    {
        cout << "Matrix too large for eigenvalue analysis. Skipping..." << endl;
        analysis.min_eigenvalue = numeric_limits<double>::quiet_NaN();
        analysis.max_eigenvalue = numeric_limits<double>::quiet_NaN();
        analysis.spectral_radius = numeric_limits<double>::quiet_NaN();
        analysis.condition_number = numeric_limits<double>::quiet_NaN();
        analysis.is_positive_definite = false;
    }
    
    return analysis;
}

/**
 * Prints detailed matrix analysis
 */
void printMatrixAnalysis(const MatrixAnalysis &analysis, const string &matrix_name)
{
    cout << "\n" << string(80, '=') << endl;
    cout << "MATRIX ANALYSIS: " << matrix_name << endl;
    cout << string(80, '=') << endl;
    
    cout << "\nBASIC PROPERTIES:" << endl;
    cout << "  Dimension: " << analysis.dimension << " x " << analysis.dimension << endl;
    cout << "  Non-zeros: " << analysis.nnz << endl;
    cout << "  Sparsity: " << fixed << setprecision(6) << analysis.sparsity * 100 << "%" << endl;
    cout << "  Density: " << fixed << setprecision(6) << analysis.density * 100 << "%" << endl;
    
    cout << "\nMATRIX CHARACTERISTICS:" << endl;
    cout << "  Symmetric: " << (analysis.is_symmetric ? "Yes" : "No") << endl;
    cout << "  Diagonally dominant: " << (analysis.is_diagonally_dominant ? "Yes" : "No") << endl;
    cout << "  Positive definite: " << (analysis.is_positive_definite ? "Yes" : "No") << endl;
    cout << "  Diagonal dominance ratio: " << fixed << setprecision(6) << analysis.diagonal_dominance_ratio << endl;
    
    cout << "\nDIAGONAL ANALYSIS:" << endl;
    cout << "  Min diagonal element: " << scientific << setprecision(6) << analysis.min_diagonal << endl;
    cout << "  Max diagonal element: " << scientific << setprecision(6) << analysis.max_diagonal << endl;
    cout << "  Max off-diagonal element: " << scientific << setprecision(6) << analysis.max_off_diagonal << endl;
    
    cout << "\nMATRIX NORMS:" << endl;
    cout << "  Frobenius norm: " << scientific << setprecision(6) << analysis.frobenius_norm << endl;
    cout << "  1-norm: " << scientific << setprecision(6) << analysis.one_norm << endl;
    cout << "  Infinity norm: " << scientific << setprecision(6) << analysis.infinity_norm << endl;
    
    if (!isnan(analysis.condition_number))
    {
        cout << "\nEIGENVALUE ANALYSIS:" << endl;
        cout << "  Min eigenvalue: " << scientific << setprecision(6) << analysis.min_eigenvalue << endl;
        cout << "  Max eigenvalue: " << scientific << setprecision(6) << analysis.max_eigenvalue << endl;
        cout << "  Spectral radius: " << scientific << setprecision(6) << analysis.spectral_radius << endl;
        cout << "  Condition number: " << scientific << setprecision(6) << analysis.condition_number << endl;
        
        if (!analysis.eigenvalue_distribution.empty())
        {
            cout << "  Eigenvalue range: [" << analysis.eigenvalue_distribution.front() 
                 << ", " << analysis.eigenvalue_distribution.back() << "]" << endl;
        }
    }
    
    cout << "\nSPARSITY PATTERN ANALYSIS:" << endl;
    cout << "  Row non-zero distribution:" << endl;
    for (const auto &pair : analysis.row_nnz_distribution)
    {
        cout << "    " << pair.first << " non-zeros: " << pair.second << " rows" << endl;
    }
    
    cout << "  Column non-zero distribution:" << endl;
    for (const auto &pair : analysis.column_nnz_distribution)
    {
        cout << "    " << pair.first << " non-zeros: " << pair.second << " columns" << endl;
    }
    
    cout << "\nLARGEST ELEMENTS (by magnitude):" << endl;
    for (size_t i = 0; i < analysis.largest_elements.size(); i++)
    {
        cout << "  (" << analysis.largest_elements[i].first << ", " 
             << analysis.largest_elements[i].second << ")" << endl;
    }
    
    cout << "\nSMALLEST ELEMENTS (by magnitude):" << endl;
    for (size_t i = 0; i < analysis.smallest_elements.size(); i++)
    {
        cout << "  (" << analysis.smallest_elements[i].first << ", " 
             << analysis.smallest_elements[i].second << ")" << endl;
    }
    
    cout << "\n" << string(80, '=') << endl;
}

/**
 * Saves analysis results to a file
 */
void saveAnalysisToFile(const MatrixAnalysis &analysis, const string &matrix_name, const string &filename)
{
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << filename << " for writing" << endl;
        return;
    }
    
    file << "Matrix Analysis Results for: " << matrix_name << endl;
    file << "Generated on: " << chrono::system_clock::now().time_since_epoch().count() << endl;
    file << endl;
    
    file << "Basic Properties:" << endl;
    file << "  Dimension: " << analysis.dimension << " x " << analysis.dimension << endl;
    file << "  Non-zeros: " << analysis.nnz << endl;
    file << "  Sparsity: " << fixed << setprecision(6) << analysis.sparsity * 100 << "%" << endl;
    file << "  Density: " << fixed << setprecision(6) << analysis.density * 100 << "%" << endl;
    file << endl;
    
    file << "Matrix Characteristics:" << endl;
    file << "  Symmetric: " << (analysis.is_symmetric ? "Yes" : "No") << endl;
    file << "  Diagonally dominant: " << (analysis.is_diagonally_dominant ? "Yes" : "No") << endl;
    file << "  Positive definite: " << (analysis.is_positive_definite ? "Yes" : "No") << endl;
    file << "  Diagonal dominance ratio: " << fixed << setprecision(6) << analysis.diagonal_dominance_ratio << endl;
    file << endl;
    
    file << "Diagonal Analysis:" << endl;
    file << "  Min diagonal element: " << scientific << setprecision(6) << analysis.min_diagonal << endl;
    file << "  Max diagonal element: " << scientific << setprecision(6) << analysis.max_diagonal << endl;
    file << "  Max off-diagonal element: " << scientific << setprecision(6) << analysis.max_off_diagonal << endl;
    file << endl;
    
    file << "Matrix Norms:" << endl;
    file << "  Frobenius norm: " << scientific << setprecision(6) << analysis.frobenius_norm << endl;
    file << "  1-norm: " << scientific << setprecision(6) << analysis.one_norm << endl;
    file << "  Infinity norm: " << scientific << setprecision(6) << analysis.infinity_norm << endl;
    file << endl;
    
    if (!isnan(analysis.condition_number))
    {
        file << "Eigenvalue Analysis:" << endl;
        file << "  Min eigenvalue: " << scientific << setprecision(6) << analysis.min_eigenvalue << endl;
        file << "  Max eigenvalue: " << scientific << setprecision(6) << analysis.max_eigenvalue << endl;
        file << "  Spectral radius: " << scientific << setprecision(6) << analysis.spectral_radius << endl;
        file << "  Condition number: " << scientific << setprecision(6) << analysis.condition_number << endl;
        file << endl;
    }
    
    file << "Sparsity Pattern Analysis:" << endl;
    file << "  Row non-zero distribution:" << endl;
    for (const auto &pair : analysis.row_nnz_distribution)
    {
        file << "    " << pair.first << " non-zeros: " << pair.second << " rows" << endl;
    }
    file << endl;
    
    file << "  Column non-zero distribution:" << endl;
    for (const auto &pair : analysis.column_nnz_distribution)
    {
        file << "    " << pair.first << " non-zeros: " << pair.second << " columns" << endl;
    }
    file << endl;
    
    file.close();
    cout << "Analysis results saved to: " << filename << endl;
}

int main(int argc, char* argv[])
{
    string matrix_file;
    string output_file = "matrix_analysis_results.txt";
    
    // Parse command line arguments
    if (argc >= 2)
    {
        matrix_file = argv[1];
    }
    else
    {
        // Default to the 80x80 matrix
        matrix_file = "data/ancf/refine2/80/solve_1001_0_Z.dat";
    }
    
    if (argc >= 3)
    {
        output_file = argv[2];
    }
    
    cout << "Matrix Analysis Tool" << endl;
    cout << "===================" << endl;
    cout << "Input matrix file: " << matrix_file << endl;
    cout << "Output file: " << output_file << endl;
    cout << endl;
    
    // Read matrix
    cout << "Reading matrix from file..." << endl;
    vector<double> values;
    vector<int> rowIndex;
    vector<int> columns;
    int n;
    
    auto start_time = chrono::high_resolution_clock::now();
    readMatrixCSR(matrix_file, values, rowIndex, columns, n);
    auto end_time = chrono::high_resolution_clock::now();
    
    auto read_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Matrix read in " << read_time.count() << " ms" << endl;
    cout << "Matrix dimension: " << n << " x " << n << endl;
    cout << "Number of non-zeros: " << values.size() << endl;
    cout << endl;
    
    // Convert to Eigen sparse matrix
    cout << "Converting to Eigen sparse matrix..." << endl;
    start_time = chrono::high_resolution_clock::now();
    Eigen::SparseMatrix<double> matrix = csrToEigenSparse(values, rowIndex, columns, n);
    end_time = chrono::high_resolution_clock::now();
    
    auto convert_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Conversion completed in " << convert_time.count() << " ms" << endl;
    cout << endl;
    
    // Analyze matrix
    cout << "Analyzing matrix properties..." << endl;
    start_time = chrono::high_resolution_clock::now();
    MatrixAnalysis analysis = analyzeMatrix(matrix);
    end_time = chrono::high_resolution_clock::now();
    
    auto analysis_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "Analysis completed in " << analysis_time.count() << " ms" << endl;
    cout << endl;
    
    // Print results
    printMatrixAnalysis(analysis, matrix_file);
    
    // Save results to file
    saveAnalysisToFile(analysis, matrix_file, output_file);
    
    cout << "\nAnalysis complete!" << endl;
    cout << "Total time: " << (read_time + convert_time + analysis_time).count() << " ms" << endl;
    
    return 0;
} 