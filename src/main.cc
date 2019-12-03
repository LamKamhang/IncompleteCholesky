#include "EvaluationFramework.h"

#include <Eigen/SparseCholesky>
#include <Eigen/src/IterativeLinearSolvers/IncompleteCholesky.h>

#include <iostream>

#include "CudaIncompleteCholesky.h"

using namespace Eigen;
using namespace std;

using namespace CudaIncompleteCholesky_;

int main(int argc, char *argv[])
{
  auto seed = 1;

  int n = 3000;
  int nnz = 5000;

  // EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky Solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky Solver", seed, random_SPD<>, 20);

  EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);

  nnz = 10000;
  EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);

  nnz = 15000;
  EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);

  nnz = 20000;
  EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);

  nnz = 25000;
  EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);

  nnz = 30000;
  // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);

  nnz = 90000;
  // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);

  nnz = 900000;
  // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);

  nnz = 4000000;
  // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);

  nnz = 9000000;
  // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", seed, nnzSparse<>, n, nnz);
  // EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  return 0;
}
