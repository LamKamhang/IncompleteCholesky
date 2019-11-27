// #include <iostream>
// #include "cuda_runtime.h"
// #include "device_functions.h"
// #include "device_launch_parameters.h"
// #include <fstream>
// #include <assert.h>
// #include <cmath>
// #include <vector>
// #include "cublas_v2.h"
// #include <thrust/sort.h>
// #include <thrust/device_vector.h>
// #include <thrust/copy.h>
// #include <utility>
// #include <cusparse.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda_runtime.h>
// #include "cusparse.h"
// #include "cusolver_common.h"
// #include "cusolverDn.h"
// #include "math_functions.h"
// #include <algorithm>
// #include <Eigen/Cholesky>
// #include <Eigen/Core>
// #include <Eigen/LU>
// #include <Eigen/Sparse>
// #include <Eigen/SparseCholesky>
// #include <windows.h>
#include "CudaIncompleteCholesky.h"
//#define CLEANUP(s)                                   \
//do {                                                 \
//    printf ("%s\n", s);                              \
//    if (yHostPtr)           free(yHostPtr);          \
//    if (zHostPtr)           free(zHostPtr);          \
//    if (xIndHostPtr)        free(xIndHostPtr);       \
//    if (xValHostPtr)        free(xValHostPtr);       \
//    if (cooRowIndexHostPtr) free(cooRowIndexHostPtr);\
//    if (cooColIndexHostPtr) free(cooColIndexHostPtr);\
//    if (cooValHostPtr)      free(cooValHostPtr);     \
//    if (y)                  cudaFree(y);             \
//    if (z)                  cudaFree(z);             \
//    if (xInd)               cudaFree(xInd);          \
//    if (xVal)               cudaFree(xVal);          \
//    if (csrRowPtr)          cudaFree(csrRowPtr);     \
//    if (cooRowIndex)        cudaFree(cooRowIndex);   \
//    if (cooColIndex)        cudaFree(cooColIndex);   \
//    if (cooVal)             cudaFree(cooVal);        \
//    if (descr)              cusparseDestroyMatDescr(descr);\
//    if (handle)             cusparseDestroy(handle); \
//    cudaDeviceReset();          \
//    fflush (stdout);                                 \
//} while (0)
//cudaError_t cudaStat1, cudaStat2, cudaStat3, cudaStat4, cudaStat5, cudaStat6;
//cusparseStatus_t status;
//cusparseHandle_t handle = 0;

//int linearSolverCHOL(
//	cusolverDnHandle_t handle,
//	int n,
//	const double* Acopy,
//	int lda,
//	const double* b,
//	double* x)
//{
//	int bufferSize = 0;
//	int* info = NULL;
//	double* buffer = NULL;
//	double* A = NULL;
//	int h_info = 0;
//	double start, stop;
//	double time_solve;
//	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
//
//	cusolverDnDpotrf_bufferSize(handle, uplo, n, (double*)Acopy, lda, &bufferSize);
//
//	cudaMalloc(&info, sizeof(int)));
//	cudaMalloc(&buffer, sizeof(double) * bufferSize));
//	cudaMalloc(&A, sizeof(double) * lda * n);
//
//
//	// prepare a copy of A because potrf will overwrite A with L
//	cudaMemcpy(A, Acopy, sizeof(double) * lda * n, cudaMemcpyDeviceToDevice);
//	cudaMemset(info, 0, sizeof(int));
//
//	start = second();
//	start = second();
//
//	cusolverDnDpotrf(handle, uplo, n, A, lda, buffer, bufferSize, info);
//
//	cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost);
//
//	if (0 != h_info) {
//		fprintf(stderr, "Error: Cholesky factorization failed\n");
//	}
//
//	cudaMemcpy(x, b, sizeof(double) * n, cudaMemcpyDeviceToDevice);
//
//	cusolverDnDpotrs(handle, uplo, n, 1, A, lda, x, n, info);
//
//	cudaDeviceSynchronize();
//	stop = second();
//
//	time_solve = stop - start;
//	fprintf(stdout, "timing: cholesky = %10.6f sec\n", time_solve);
//
//	if (info) { cudaFree(info); }
//	if (buffer) { cudaFree(buffer); }
//	if (A) { cudaFree(A); }
//
//	return 0;
//}
//__global__ void print(int * pos, int size) {
//	for (int i = 0; i < size; i++)
//		printf("pos %d %d\n", i, pos[i]);
//}
//__global__ void print(double* pos, int size) {
//	for (int i = 0; i < size; i++)
//		printf("pos %d %f\n", i, pos[i]);
//}

using namespace std;
using namespace Eigen;
string fileName;
int main() {
	int n, nnz;
	cin >> fileName;
	ifstream ifs(fileName);
	//getchar();
	ifs >> n >> nnz;
	vector<Triplet<double>> Q;
	int a, b;
	double z;
	for (int i = 0; i < nnz; i++) {
		ifs >> a >> b >> z;
		//	cout << a << b << z << endl;
		Q.push_back(Triplet<double>(a, b, z));
		//	cout << Q[i] << endl;
	}
	SparseMatrix<double> A(n, n);

	A.setFromTriplets(Q.begin(), Q.end());

	VectorXd B(n), X(n);

	CudaIncompleteCholesky::CudaIncompleteCholesky<double> solver;
	for (int i = 0; i < n; i++) {
		ifs >> B(i);
	}

	// LARGE_INTEGER t1, t2, tc;
	// QueryPerformanceFrequency(&tc);
	// QueryPerformanceCounter(&t1);

	solver.compute(A);
	X = solver.solve(B);

	// QueryPerformanceCounter(&t2);
	// printf("Use Time:%f\n", (t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart);
	std::cout << "Linear system solvers comparison " << std::endl;
	std::cout << "  Relative error |Ax - b| / |b|  " << std::endl;
	double relative_error_llt = (A * X - B).norm() / B.norm();
	cout << relative_error_llt << endl;
}

//Eigen::SparseMatrix<double> A;
//Eigen::SparseVector<double> B;
//void eigenOpen(int* &csrRowPtr, int* &csrColInd, double* &csrVal, double* &x, int& n, int& nnz){
//	using namespace std;
//	using namespace Eigen;
//	string fileName;
//	cin >> fileName;
//	ifstream ifs(fileName);
//	//cout << ifs.is_open() << endl;
//	//getchar();
//	ifs >> n >> nnz;
//	vector<Triplet<double>> Q;
//	int a, b;
//	double z;
//	x = new double[n];
//	for (int i = 0; i < nnz; i++) {
//		ifs >> a >> b >> z;
//		Q.push_back(Triplet<double>(a, b, z));
//	}
//	A = SparseMatrix<double>(n, n);
//	B = SparseVector<double>(n);
//	A.setFromTriplets(Q.begin(), Q.end());
//	csrRowPtr = new int[n + 1];
//	csrColInd = new int[nnz];
//	csrVal = new double[nnz];
//	cudaMemcpy(csrVal, A.valuePtr(), sizeof(double) * nnz, cudaMemcpyHostToHost);
//	cudaMemcpy(csrRowPtr, A.outerIndexPtr(), sizeof(int) * (n + 1), cudaMemcpyHostToHost);
//	cudaMemcpy(csrColInd, A.innerIndexPtr(), sizeof(int) * nnz, cudaMemcpyHostToHost);
//	for (int i = 0; i < n; i++) {
//		ifs >> x[i];
//		B.insert(i) = x[i];
//	}
//	//cout << B << endl;
//	//getchar();
//}
//
//double errorCheck(double* y, int size) {
//	using namespace Eigen;
//	SparseVector<double> X(size);
//	for (int i = 0; i < size; i++) {
//		X.insert(i) = y[i];
//	}
//	double error = (A * X - B).norm() / B.norm();
//	return error;
//}
//
//int main() {
//	cusparseHandle_t handle;
//	cusparseStatus_t status = cusparseCreate(&handle);
//	int *csrRowPtr,*csrColInd, *d_csrRowPtr, *d_csrColInd;
//	int m = 3, nnz = 3;
//	double* d_csrVal, * csrVal, *x, *y, *z, *d_x, *d_y, *d_z;
//
//	//csrRowPtr = new int[m+1];
//	//csrColInd = new int[nnz];
//	//csrVal = new double[nnz];
//	//x = new double[m];
//	//for (int i = 0; i < m; i++) {
//	//	csrVal[i] = 2;
//	//	csrRowPtr[i] = i;
//	//	csrColInd[i] = i;
//	//	x[i] = 2*(i+1);
//	//}
//	//csrRowPtr[m] = nnz;
//	eigenOpen(csrRowPtr, csrColInd, csrVal, x, m, nnz);
//	//printf("%d %d\n", csrRowPtr[0], csrRowPtr[1]);
//	cudaMalloc((void**)& d_csrRowPtr, sizeof(int)*(m+1));
//	cudaMalloc((void**)& d_csrColInd, sizeof(int)* nnz);
//	cudaMalloc((void**)& d_csrVal, sizeof(double)*nnz);
//	cudaMalloc((void**)& d_x, sizeof(double) * m);
//	cudaMalloc((void**)& d_y, sizeof(double) * m);
//	cudaMalloc((void**)& d_z, sizeof(double) * m);
//	cudaMemcpy(d_csrRowPtr, csrRowPtr, sizeof(int) * (m + 1), cudaMemcpyHostToDevice);
//
//	cudaMemcpy(d_csrColInd, csrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_csrVal, csrVal, sizeof(double) * nnz, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_x, x, sizeof(double) * m, cudaMemcpyHostToDevice);
//	//print << <1, 1 >> > (d_csrRowPtr, m + 1);
//	//print << <1, 1 >> > (d_csrVal, nnz);
//	//print << <1, 1 >> > (d_csrColInd, nnz);
//	// Suppose that A is m x m sparse matrix represented by CSR format,
//	// Assumption:
//	// - handle is already created by cusparseCreate(),
//	// - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of A on device memory,
//	// - d_x is right hand side vector on device memory,
//	// - d_y is solution vector on device memory.
//	// - d_z is intermediate result on device memory.
//
//	cusparseMatDescr_t descr_M = 0;
//	cusparseMatDescr_t descr_L = 0;
//	csric02Info_t info_M = 0;
//	csrsv2Info_t  info_L = 0;
//	csrsv2Info_t  info_Lt = 0;
//	int pBufferSize_M;
//	int pBufferSize_L;
//	int pBufferSize_Lt;
//	int pBufferSize;
//	void* pBuffer = 0;
//	int structural_zero;
//	int numerical_zero;
//	const double alpha = 1.;
//	const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
//	const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
//	const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
//	const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
//	const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;
//
//	// step 1: create a descriptor which contains
//	// - matrix M is base-1
//	// - matrix L is base-1
//	// - matrix L is lower triangular
//	// - matrix L has non-unit diagonal
//	LARGE_INTEGER t1, t2, tc;
//	QueryPerformanceFrequency(&tc);
//	QueryPerformanceCounter(&t1);
//
//	cusparseCreateMatDescr(&descr_M);
//	cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
//	cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
//
//	cusparseCreateMatDescr(&descr_L);
//	cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
//	cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
//	cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
//	cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);
//
//	// step 2: create a empty info structure
//	// we need one info for csric02 and two info's for csrsv2
//	cusparseCreateCsric02Info(&info_M);
//	cusparseCreateCsrsv2Info(&info_L);
//	cusparseCreateCsrsv2Info(&info_Lt);
//
//	// step 3: query how much memory used in csric02 and csrsv2, and allocate the buffer
//	cusparseDcsric02_bufferSize(handle, m, nnz,
//		descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &pBufferSize_M);
//	cusparseDcsrsv2_bufferSize(handle, trans_L, m, nnz,
//		descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, &pBufferSize_L);
//	cusparseDcsrsv2_bufferSize(handle, trans_Lt, m, nnz,
//		descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_Lt, &pBufferSize_Lt);
//
//	pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_Lt));
//
//	// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
//	cudaMalloc((void**)& pBuffer, pBufferSize);
//
//	// step 4: perform analysis of incomplete Cholesky on M
//	//         perform analysis of triangular solve on L
//	//         perform analysis of triangular solve on L'
//	// The lower triangular part of M has the same sparsity pattern as L, so
//	// we can do analysis of csric02 and csrsv2 simultaneously.
//
//	cusparseDcsric02_analysis(handle, m, nnz, descr_M,
//		d_csrVal, d_csrRowPtr, d_csrColInd, info_M,
//		policy_M, pBuffer);
//	status = cusparseXcsric02_zeroPivot(handle, info_M, &structural_zero);
//	if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
//		printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
//	}
//
//	cusparseDcsrsv2_analysis(handle, trans_L, m, nnz, descr_L,
//		d_csrVal, d_csrRowPtr, d_csrColInd,
//		info_L, policy_L, pBuffer);
//
//	cusparseDcsrsv2_analysis(handle, trans_Lt, m, nnz, descr_L,
//		d_csrVal, d_csrRowPtr, d_csrColInd,
//		info_Lt, policy_Lt, pBuffer);
//
//	// step 5: M = L * L'
//	cusparseDcsric02(handle, m, nnz, descr_M,
//		d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer);
//	status = cusparseXcsric02_zeroPivot(handle, info_M, &numerical_zero);
//	//print << <1, 1 >> > (d_csrRowPtr, m + 1);
//	//print << <1, 1 >> > (d_csrColInd, m);
//	//print << <1, 1 >> > (d_csrVal, m);
//	if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
//		printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
//	}
//
//	// step 6: solve L*z = x
//	cusparseDcsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L,
//		d_csrVal, d_csrRowPtr, d_csrColInd, info_L,
//		d_x, d_z, policy_L, pBuffer);
//
//	// step 7: solve L'*y = z
//	cusparseDcsrsv2_solve(handle, trans_Lt, m, nnz, &alpha, descr_L,
//		d_csrVal, d_csrRowPtr, d_csrColInd, info_Lt,
//		d_z, d_y, policy_Lt, pBuffer);
//	cudaThreadSynchronize();
//
//	QueryPerformanceCounter(&t2);
//	printf("Use Time:%f\n", (t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart);
//	//print << <1, 1 >> > (d_y, m);
//	y = new double[m];
//	cudaMemcpy(y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost);
//	double error=errorCheck(y, m);
//	std::cout << error << std::endl;
//	// step 6: free resources
//	cudaFree(pBuffer);
//	cusparseDestroyMatDescr(descr_M);
//	cusparseDestroyMatDescr(descr_L);
//	cusparseDestroyCsric02Info(info_M);
//	cusparseDestroyCsrsv2Info(info_L);
//	cusparseDestroyCsrsv2Info(info_Lt);
//	cusparseDestroy(handle);
//}
