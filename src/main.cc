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

  // int n = 3000;
  // std::vector<int> nnzs = {
  //   5000, 10000, 15000, 20000, 25000, 30000, 90000, 900000, 4000000, 9000000
  // };
  // for (auto & nnz : nnzs )
  // {
  //   EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky solver", seed, nnzSparse<>, n, nnz);
  //   // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", seed, nnzSparse<>, n, nnz);
  //   // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", seed, nnzSparse<>, n, nnz);
  //   EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  //   EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, nnz);
  // }

  int dense = 2;
  std::vector<int> ns = {
    500, 1000, 5000, 10000, 200000
  };
  for (auto & n : ns)
  {
    EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky solver", seed, nnzSparse<>, n, dense*n);
    EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", seed, nnzSparse<>, n, dense*n);
    EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", seed, nnzSparse<>, n, dense*n);
  }
  return 0;
}
