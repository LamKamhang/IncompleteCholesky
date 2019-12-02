# Sparse Solver Concept

All these solvers follow the same general concept. Here is a typical and general example:

```c++
#include <RequiredModuleName>
// ...
SparseMatrix<double> A;
// fill A
VectorXd b, x;
// fill b
// solve Ax = b
SolverClassName<SparseMatrix<double> > solver;
solver.compute(A);
if(solver.info()!=Success) {
  // decomposition failed
  return;
}
x = solver.solve(b);
if(solver.info()!=Success) {
  // solving failed
  return;
}
// solve for another right hand side:
x1 = solver.solve(b1);
```

Thus, each sparse solver should at least implements such interfaces:

```c++
// compute
SolverClassReference compute(const Mat &A);

// solve
Vec solve(const Vec &b);

// optional interface >>> info
Status_t info() const;
```

