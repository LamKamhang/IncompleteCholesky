#include "EvaluationFramework.h"

#include <Eigen/SparseCholesky>
#include <Eigen/src/IterativeLinearSolvers/IncompleteCholesky.h>

#include <iostream>

#include "CudaIncompleteCholesky.h"

using namespace Eigen;
using namespace std;

using namespace CudaIncompleteCholesky_;

#define EVALUATE_SOLVER(title, seed, matrix_type, solver_type, n, nnz)  \
  {                                                                     \
    info_msg(title);                                                    \
    srand(seed);                                                        \
    solver_type solver;                                                 \
    EvaluationFramework::matrix_type(solver, n, nnz);                   \
    cout << endl;                                                       \
  }

int main(int argc, char *argv[])
{
  auto seed = 1;

  int n = 3000;
  int nnz = 5000;
  EVALUATE_SOLVER("Sparse Cholesky solver", seed, random_SPD,
                  SimplicialCholesky<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Sparse Cholesky LLT solver", seed, random_SPD,
  //                 SimplicialLLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Sparse Cholesky LDLT solver", seed, random_SPD,
  //                 SimplicialLDLT<SparseMatrix<double> >, n, nnz);
  EVALUATE_SOLVER("Incomplete Cholesky solver", seed, random_SPD,
                  IncompleteCholesky<double>, n, nnz);
  EVALUATE_SOLVER("cuda Incomplete Cholesky solver", seed, random_SPD,
                  CudaIncompleteCholesky<double>, n, nnz);

  // nnz = 10000;
  // EVALUATE_SOLVER("Sparse Cholesky solver", seed, random_SPD,
  //                 SimplicialCholesky<SparseMatrix<double> >, n, nnz);
  // // EVALUATE_SOLVER("Sparse Cholesky LLT solver", seed, random_SPD,
  // //                 SimplicialLLT<SparseMatrix<double> >, n, nnz);
  // // EVALUATE_SOLVER("Sparse Cholesky LDLT solver", seed, random_SPD,
  // //                 SimplicialLDLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Incomplete Cholesky solver", seed, random_SPD,
  //                 IncompleteCholesky<double>, n, nnz);
  // EVALUATE_SOLVER("cuda Incomplete Cholesky solver", seed, random_SPD,
  //                 CudaIncompleteCholesky<double>, n, nnz);

  // nnz = 15000;
  // EVALUATE_SOLVER("Sparse Cholesky solver", seed, random_SPD,
  //                 SimplicialCholesky<SparseMatrix<double> >, n, nnz);
  // // EVALUATE_SOLVER("Sparse Cholesky LLT solver", seed, random_SPD,
  // //                 SimplicialLLT<SparseMatrix<double> >, n, nnz);
  // // EVALUATE_SOLVER("Sparse Cholesky LDLT solver", seed, random_SPD,
  // //                 SimplicialLDLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Incomplete Cholesky solver", seed, random_SPD,
  //                 IncompleteCholesky<double>, n, nnz);
  // EVALUATE_SOLVER("cuda Incomplete Cholesky solver", seed, random_SPD,
  //                 CudaIncompleteCholesky<double>, n, nnz);

  // nnz = 20000;
  // EVALUATE_SOLVER("Sparse Cholesky solver", seed, random_SPD,
  //                 SimplicialCholesky<SparseMatrix<double> >, n, nnz);
  // // EVALUATE_SOLVER("Sparse Cholesky LLT solver", seed, random_SPD,
  // //                 SimplicialLLT<SparseMatrix<double> >, n, nnz);
  // // EVALUATE_SOLVER("Sparse Cholesky LDLT solver", seed, random_SPD,
  // //                 SimplicialLDLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Incomplete Cholesky solver", seed, random_SPD,
  //                 IncompleteCholesky<double>, n, nnz);
  // EVALUATE_SOLVER("cuda Incomplete Cholesky solver", seed, random_SPD,
  //                 CudaIncompleteCholesky<double>, n, nnz);

  // nnz = 25000;
  // EVALUATE_SOLVER("Sparse Cholesky solver", seed, random_SPD,
  //                 SimplicialCholesky<SparseMatrix<double> >, n, nnz);
  // // EVALUATE_SOLVER("Sparse Cholesky LLT solver", seed, random_SPD,
  // //                 SimplicialLLT<SparseMatrix<double> >, n, nnz);
  // // EVALUATE_SOLVER("Sparse Cholesky LDLT solver", seed, random_SPD,
  // //                 SimplicialLDLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Incomplete Cholesky solver", seed, random_SPD,
  //                 IncompleteCholesky<double>, n, nnz);
  // EVALUATE_SOLVER("cuda Incomplete Cholesky solver", seed, random_SPD,
  //                 CudaIncompleteCholesky<double>, n, nnz);

  // nnz = 30000;
  // // EVALUATE_SOLVER("Sparse Cholesky LLT solver", seed, random_SPD,
  // //                 SimplicialLLT<SparseMatrix<double> >, n, nnz);
  // // EVALUATE_SOLVER("Sparse Cholesky LDLT solver", seed, random_SPD,
  // //                 SimplicialLDLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Incomplete Cholesky solver", seed, random_SPD,
  //                 IncompleteCholesky<double>, n, nnz);
  // EVALUATE_SOLVER("cuda Incomplete Cholesky solver", seed, random_SPD,
  //                 CudaIncompleteCholesky<double>, n, nnz);

  // nnz = 90000;
  // // EVALUATE_SOLVER("Sparse Cholesky LLT solver", seed, random_SPD,
  // //                 SimplicialLLT<SparseMatrix<double> >, n, nnz);
  // // EVALUATE_SOLVER("Sparse Cholesky LDLT solver", seed, random_SPD,
  // //                 SimplicialLDLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Incomplete Cholesky solver", seed, random_SPD,
  //                 IncompleteCholesky<double>, n, nnz);
  // EVALUATE_SOLVER("cuda Incomplete Cholesky solver", seed, random_SPD,
  //                 CudaIncompleteCholesky<double>, n, nnz);

  // nnz = 900000;
  // // EVALUATE_SOLVER("Sparse Cholesky LLT solver", seed, random_SPD,
  // //                 SimplicialLLT<SparseMatrix<double> >, n, nnz);
  // // EVALUATE_SOLVER("Sparse Cholesky LDLT solver", seed, random_SPD,
  // //                 SimplicialLDLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Incomplete Cholesky solver", seed, random_SPD,
  //                 IncompleteCholesky<double>, n, nnz);
  // EVALUATE_SOLVER("cuda Incomplete Cholesky solver", seed, random_SPD,
  //                 CudaIncompleteCholesky<double>, n, nnz);

  nnz = 4000000;
  // EVALUATE_SOLVER("Sparse Cholesky LLT solver", seed, random_SPD,
  //                 SimplicialLLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Sparse Cholesky LDLT solver", seed, random_SPD,
  //                 SimplicialLDLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Incomplete Cholesky solver", seed, random_SPD,
  //                 IncompleteCholesky<double>, n, nnz);
  EVALUATE_SOLVER("cuda Incomplete Cholesky solver", seed, random_SPD,
                  CudaIncompleteCholesky<double>, n, nnz);

  nnz = 9000000;
  // EVALUATE_SOLVER("Sparse Cholesky LLT solver", seed, random_SPD,
  //                 SimplicialLLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Sparse Cholesky LDLT solver", seed, random_SPD,
  //                 SimplicialLDLT<SparseMatrix<double> >, n, nnz);
  // EVALUATE_SOLVER("Incomplete Cholesky solver", seed, random_SPD,
  //                 IncompleteCholesky<double>, n, nnz);
  EVALUATE_SOLVER("cuda Incomplete Cholesky solver", seed, random_SPD,
                  CudaIncompleteCholesky<double>, n, nnz);
  return 0;
}
