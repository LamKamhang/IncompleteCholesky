# Incomplete Cholesky Factorization
This project is a Performance Evaluation of cuSparse Incomplete Cholesky Method.

And the project evaluates it compared with Normal cuSparse Cholesky Factorization Method、Eigen Cholesky Factorization Method. We focus on three things, one of which is correctness, then accuracy and finally computational efficiency.

In the test cases, the matrix will diverse in dims. And the density will distribute from very sparse to half dense. Actually, we will use this method as a part of FEM solver, to speed up physics simulation. Thus, in such form $Ax = b$, the matrix A in the equation will be very sparse and if the material is non linear, the matrix A and the vector b are changeable.

> Should keep in mind that the matrix to be factorized should be symmetric positive define matrix(SPD).

## Test for Correctness

- [x]   TODO


## Test for Accuracy

- [x] TODO

## Test for Efficiency

The time overhead is mainly reflected in three parts in the GPU program.

- Memory Copy
- Calculation
- Synchronize

Thus, when evaluating cuSPARSE programs, the above three points need to be considered.

- [x] TODO

## Results

[result table](tests/table.html)