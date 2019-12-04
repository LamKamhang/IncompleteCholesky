#include "EvaluationFramework.h"

#include <Eigen/SparseCholesky>
#include <Eigen/src/IterativeLinearSolvers/IncompleteCholesky.h>

#include <iostream>

#include "CudaIncompleteCholesky.h"
#include "../external/mat-rbf/rbf_SPD.h"

using namespace Eigen;
using namespace std;

using namespace CudaIncompleteCholesky_;

int main(int argc, char *argv[])
{
  using Vec = Eigen::VectorXd;
  using SMat = Eigen::SparseMatrix<double>;

  auto seed = 1;
  srand(seed);

  // different dense
  info_msg("###different dense test###");
  int n = 3000;
  std::vector<int> nnzs = {
    5000, 10000, 15000, 20000, 25000, 30000, 90000, 900000, 4000000, 9000000
  };
  for (auto & nnz : nnzs )
  {
    const SMat A = nnzSparse(n, nnz);
    const Vec b = Vec::Random(n);

    if (nnz < 90000)
      EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky solver", A, b);
    // EvaluateSolver<SimplicialLLT<SparseMatrix<double> > >("Sparse Cholesky LLT solver", A, b);
    // EvaluateSolver<SimplicialLDLT<SparseMatrix<double> > >("Sparse Cholesky LDLT solver", A, b);
    EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", A, b);
    EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", A, b);
  }

  // different dims
  info_msg("###different dims test###");
  int dense = 10;
  std::vector<int> ns = {
    500, 1000, 5000, 10000, 200000
  };
  for (auto & n : ns)
  {
    const SMat A = nnzSparse(n, dense * n);
    const Vec b = Vec::Random(n);

    if (dense * n < 90000)
      EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky solver", A, b);
    EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", A, b);
    EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", A, b);
  }

  // mat_rbf
  if (argc == 2)
  {
    info_msg("###mat_rbf test###");
    SMat A;
    Vec b;
    rbf_SPD(argv[1], A, b);
    info_msg("A.rows: %ld\tA.cols: %ld", A.rows(), A.cols());
    EvaluateSolver<SimplicialCholesky<SparseMatrix<double> > >("Sparse Cholesky solver", A, b);
    EvaluateSolver<IncompleteCholesky<double> >("Incomplete Cholesky solver", A, b);
    EvaluateSolver<CudaIncompleteCholesky<double> >("cuda Incomplete Cholesky solver", A, b);
  }



  return 0;
}
