#pragma once
#ifndef CudaIncompleteCholesky_H
#define CudaIncompleteCholesky_H
#include <iostream>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <assert.h>
#include <cmath>
#include <vector>
#include "cublas_v2.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <utility>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include "cusolver_common.h"
#include "cusolverDn.h"
#define __CUDA_INTERNAL_COMPILATION__
#include "math_functions.h"
#undef __CUDA_INTERNAL_COMPILATION__

#include <algorithm>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
// #include <windows.h>

namespace CudaIncompleteCholesky {
	using namespace Eigen;
	template <typename Scalar>
	class CudaIncompleteCholesky {
	private:
		cusparseHandle_t handle;
		cusparseStatus_t status;
		int* csrRowPtr, * csrColInd, * d_csrRowPtr, * d_csrColInd;
		int m = 3, nnz = 3;
		Scalar* d_csrVal, * csrVal, * x, * y, * z, * d_x, * d_y, * d_z;
		cusparseMatDescr_t descr_M = 0;
		cusparseMatDescr_t descr_L = 0;
		csric02Info_t info_M = 0;
		csrsv2Info_t  info_L = 0;
		csrsv2Info_t  info_Lt = 0;
		int pBufferSize_M;
		int pBufferSize_L;
		int pBufferSize_Lt;
		int pBufferSize;
		void* pBuffer = 0;
		int structural_zero;
		int numerical_zero;
		const double alpha = 1.;
		const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
		const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
		const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
		const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
		const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;
		// LARGE_INTEGER t1, t2, tc;
		template<typename MatrixType>
		void cudaMemInit(int*& csrRowPtr, int*& csrColInd, Scalar*& csrVal, Scalar*& x, int& m, int& nnz, const MatrixType& mat)
		{
			status = cusparseCreate(&handle);
			csrRowPtr = new int[m + 1];
			csrColInd = new int[nnz];
			csrVal = new Scalar[nnz];
			cudaMemcpy(csrVal, mat.valuePtr(), sizeof(Scalar) * nnz, cudaMemcpyHostToHost);
			cudaMemcpy(csrRowPtr, mat.outerIndexPtr(), sizeof(int) * (m + 1), cudaMemcpyHostToHost);
			cudaMemcpy(csrColInd, mat.innerIndexPtr(), sizeof(int) * nnz, cudaMemcpyHostToHost);

			cudaMalloc((void**)& d_csrRowPtr, sizeof(int) * (m + 1));
			cudaMalloc((void**)& d_csrColInd, sizeof(int) * nnz);
			cudaMalloc((void**)& d_csrVal, sizeof(Scalar) * nnz);
			cudaMalloc((void**)& d_x, sizeof(Scalar) * m);
			cudaMalloc((void**)& d_y, sizeof(Scalar) * m);
			cudaMalloc((void**)& d_z, sizeof(Scalar) * m);
			cudaMemcpy(d_csrRowPtr, csrRowPtr, sizeof(int) * (m + 1), cudaMemcpyHostToDevice);

			cudaMemcpy(d_csrColInd, csrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
			cudaMemcpy(d_csrVal, csrVal, sizeof(Scalar) * nnz, cudaMemcpyHostToDevice);
			//cudaMemcpy(d_x, x, sizeof(Scalar) * m, cudaMemcpyHostToDevice);
		}

	public:
		CudaIncompleteCholesky() {}
		/*typedef typename NumTraits<Scalar>::Real RealScalar;
		typedef _OrderingType OrderingType;
		typedef typename OrderingType::PermutationType PermutationType;
		typedef typename PermutationType::StorageIndex StorageIndex;
		typedef SparseMatrix<Scalar, ColMajor, StorageIndex> FactorType;
		typedef Matrix<Scalar, Dynamic, 1> VectorSx;
		typedef Matrix<RealScalar, Dynamic, 1> VectorRx;
		typedef Matrix<StorageIndex, Dynamic, 1> VectorIx;
		typedef std::vector<std::list<StorageIndex> > VectorList;*/
		template<typename MatrixType>
		void compute(const MatrixType& mat)
		{
			nnz = mat.nonZeros();
			m = mat.rows();
			cudaMemInit(csrRowPtr, csrColInd, csrVal, x, m, nnz, mat);

			// Suppose that A is m x m sparse matrix represented by CSR format,
			// Assumption:
			// - handle is already created by cusparseCreate(),
			// - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of A on device memory,
			// - d_x is right hand side vector on device memory,
			// - d_y is solution vector on device memory.
			// - d_z is intermediate result on device memory.


			// step 1: create a descriptor which contains
			// - matrix M is base-1
			// - matrix L is base-1
			// - matrix L is lower triangular
			// - matrix L has non-unit diagonal

			// QueryPerformanceFrequency(&tc);
			// QueryPerformanceCounter(&t1);

			cusparseCreateMatDescr(&descr_M);
			cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
			cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

			cusparseCreateMatDescr(&descr_L);
			cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
			cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
			cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
			cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);

			// step 2: create a empty info structure
			// we need one info for csric02 and two info's for csrsv2
			cusparseCreateCsric02Info(&info_M);
			cusparseCreateCsrsv2Info(&info_L);
			cusparseCreateCsrsv2Info(&info_Lt);

			// step 3: query how much memory used in csric02 and csrsv2, and allocate the buffer
			cusparseDcsric02_bufferSize(handle, m, nnz,
				descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &pBufferSize_M);
			cusparseDcsrsv2_bufferSize(handle, trans_L, m, nnz,
				descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, &pBufferSize_L);
			cusparseDcsrsv2_bufferSize(handle, trans_Lt, m, nnz,
				descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_Lt, &pBufferSize_Lt);

			pBufferSize = std::max(pBufferSize_M, std::max(pBufferSize_L, pBufferSize_Lt));

			// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
			cudaMalloc((void**)& pBuffer, pBufferSize);

			// step 4: perform analysis of incomplete Cholesky on M
			//         perform analysis of triangular solve on L
			//         perform analysis of triangular solve on L'
			// The lower triangular part of M has the same sparsity pattern as L, so
			// we can do analysis of csric02 and csrsv2 simultaneously.

			cusparseDcsric02_analysis(handle, m, nnz, descr_M,
				d_csrVal, d_csrRowPtr, d_csrColInd, info_M,
				policy_M, pBuffer);
			status = cusparseXcsric02_zeroPivot(handle, info_M, &structural_zero);
			if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
				printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
			}

			cusparseDcsrsv2_analysis(handle, trans_L, m, nnz, descr_L,
				d_csrVal, d_csrRowPtr, d_csrColInd,
				info_L, policy_L, pBuffer);

			cusparseDcsrsv2_analysis(handle, trans_Lt, m, nnz, descr_L,
				d_csrVal, d_csrRowPtr, d_csrColInd,
				info_Lt, policy_Lt, pBuffer);

			// step 5: M = L * L'
			cusparseDcsric02(handle, m, nnz, descr_M,
				d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer);
			status = cusparseXcsric02_zeroPivot(handle, info_M, &numerical_zero);
			if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
				printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
			}
		}

		template<typename VectorType>
		VectorType solve(const VectorType& vec) {
			cudaMemcpy(d_x, &(vec(0)), sizeof(Scalar) * m, cudaMemcpyHostToDevice);
			// step 6: solve L*z = x
			cusparseDcsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L,
				d_csrVal, d_csrRowPtr, d_csrColInd, info_L,
				d_x, d_z, policy_L, pBuffer);

			// step 7: solve L'*y = z
			cusparseDcsrsv2_solve(handle, trans_Lt, m, nnz, &alpha, descr_L,
				d_csrVal, d_csrRowPtr, d_csrColInd, info_Lt,
				d_z, d_y, policy_Lt, pBuffer);
			cudaThreadSynchronize();

			// QueryPerformanceCounter(&t2);
			//printf("Use Time:%f\n", (t2.QuadPart - t1.QuadPart) * 1.0 / tc.QuadPart);
			//print << <1, 1 >> > (d_y, m);
			y = new double[m];
			cudaMemcpy(y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost);
			VectorType ret = Map<VectorXd>(y, m);
			return ret;
		}
		template<typename MatrixType>
		CudaIncompleteCholesky(const MatrixType& matrix) {
			compute(matrix);
		}
		~CudaIncompleteCholesky() {

			// step 6: free resources
			cudaFree(pBuffer);
			cusparseDestroyMatDescr(descr_M);
			cusparseDestroyMatDescr(descr_L);
			cusparseDestroyCsric02Info(info_M);
			cusparseDestroyCsrsv2Info(info_L);
			cusparseDestroyCsrsv2Info(info_Lt);
			cusparseDestroy(handle);
		}


	};
}

#endif
