#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <limits>

using CalcType = float;
using IndexType = int64_t;

using Mat = Eigen::MatrixXf;
using SpMat = Eigen::SparseMatrix<CalcType, Eigen::ColMajor, IndexType>;
using T = Eigen::Triplet<CalcType, IndexType>;

namespace dr_bcg_cpu {
int dr_bcg(const SpMat &A, Mat &X, const Mat &B,
           CalcType tolerance = std::numeric_limits<CalcType>::epsilon(),
           int max_iterations = 1000);
};
