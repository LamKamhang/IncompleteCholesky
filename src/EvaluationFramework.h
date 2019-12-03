/**
 * @file EvaluationFramework.h
 * @author LamKamhang (Cool_Lam@outlook.com)
 * @brief A Documented file.
 * @version 1.0
 * @date Mon Dec 2 20:45:50 CST 2019
 *
 * Detailed description
 *
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

#include "../external/log_utils/log_utils.h"
#include "../external/time_utils/time_utils.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <vector>

template<typename T> inline constexpr T scalar_max() {return std::numeric_limits<T>::max() / 10;}
template<typename T> inline constexpr T scalar_eps() {return 1;}

template<> inline constexpr double scalar_eps() {return 1e-9f;}
template<> inline constexpr float scalar_eps() {return 1e-5f;}

template<class T>
Eigen::SparseMatrix<T> Dense2Sparse(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Dense,
                                    const T tol = scalar_eps<T>())
{
  using Tri = Eigen::Triplet<T>;

  std::vector<Tri> triplets;
  for (int i = 0; i < Dense.rows(); ++i)
    for (int j = 0; j < Dense.cols(); ++j)
      if (std::abs(Dense(i, j)) >= tol)
        triplets.push_back(Tri(i, j, Dense(i, j)));

  Eigen::SparseMatrix<T> Sparse(Dense.rows(), Dense.cols());
  Sparse.setFromTriplets(triplets.begin(), triplets.end());

  return Sparse;
}

template<class T = double>
Eigen::SparseMatrix<T> nnzSparse(int n, int nnz)
{
  using Tri = Eigen::Triplet<T>;
  std::vector<Tri> triplets;

  for (int i = 0; i < (nnz - n) / 2; ++i)
    {
      int row = std::abs(rand()) % n;
      int col = row == 0 ? 0 : std::abs(rand()) % row;
      double value = (rand() % 100) / 100.0;
      triplets.push_back(Tri(row, col, value));
      triplets.push_back(Tri(col, row, value));
    }
  for (int i = 0; i < n; ++i)
    triplets.push_back(Tri(i, i, n));

  Eigen::SparseMatrix<T> Sparse(n, n);
  Sparse.setFromTriplets(triplets.begin(), triplets.end());

  return Sparse;
}

template<class T = double>
Eigen::SparseMatrix<T> random_SPD(int n)
{
  using Mat = Eigen::MatrixXd;
  using Vec = Eigen::VectorXd;
  using SMat = Eigen::SparseMatrix<double>;

  // construct SPD A
  Mat A;
  A = Mat::Random(n, n);
  A = 0.5 * (A + A.transpose()) + Mat::Identity(n, n) * n;
  return Dense2Sparse(A, std::abs(Mat::Random(1,1)(0, 0)));
}

template<class SparseSolver>
void EvaluateSolver(const std::string &title, const Eigen::SparseMatrix<double> &SA, const Eigen::VectorXd &b)
{
  using Vec = Eigen::VectorXd;

  info_msg("%s", title.c_str());

  SparseSolver solver;

  info_msg("Sparse Matrix number of non-zero elements %lu", SA.nonZeros());

  time_utils::time_point_id_t compute_id;
  TIMER_BEGIN(compute_id);
  solver.compute(SA);
  TIMER_END(compute_id, "framework compute");

  // solve Ax = b
  time_utils::time_point_id_t solve_id;
  TIMER_BEGIN(solve_id);
  Vec x = solver.solve(b);
  TIMER_END(solve_id, "framework solve");
  // error metric: ||Ax - b||
  info_msg("framework error: %lf", (SA*x - b).norm());
  printf("\n");
}

// template<class SparseSolver, typename SPD, typename ... Args>
// void EvaluateSolver(const std::string &title, int seed, SPD&& spd, int n, Args&& ... args)
// {
//   using Vec = Eigen::VectorXd;
//   using SMat = Eigen::SparseMatrix<double>;

//   info_msg("%s", title.c_str());
//   std::srand(seed);

//   SparseSolver solver;

//   SMat SA = std::bind(std::forward<SPD>(spd), n, std::forward<Args>(args)...)();

//   info_msg("Sparse Matrix number of non-zero elements %lu", SA.nonZeros());

//   time_utils::time_point_id_t compute_id;
//   TIMER_BEGIN(compute_id);
//   solver.compute(SA);
//   TIMER_END(compute_id, "framework compute");

//   // construct vector b
//   double error = 0;
//   // int iters = 1;
//   // for (int i = 0; i < iters; ++i)
//   //   {
//   const Vec b = Vec::Random(n);

//   time_utils::time_point_id_t solve_id;
//   TIMER_BEGIN(solve_id);
//   Vec x = solver.solve(b);
//   TIMER_END(solve_id, "framework solve");
//   // error metric: ||Ax - b||
//   error += (SA*x - b).norm();
//   // }
//   // error /= iters;
//   info_msg("framework error: %lf", error);
//   printf("\n");
// }
