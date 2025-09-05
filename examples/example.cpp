#include "dr_bcg_cpu/dr_bcg_cpu.h"
#include "dr_bcg_cpu/profiler.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <suitesparse_matrix.h>

void print_error(const SpMat &A, const Mat &X, const Mat &B) {
    const int m = B.rows();
    const int n = B.cols();

    if (m <= 0 || n <= 0) {
        return;
    }

    Mat product = A * X;

    CalcType total_error = 0;
    CalcType min_error = std::abs(B(0, 0) - product(0, 0));
    CalcType max_error = min_error;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            CalcType error = std::abs(B(i, j) - product(i, j));
            total_error += error;
            min_error = std::min(min_error, error);
            max_error = std::max(max_error, error);
        }
    }

    std::cout << "Error:" << std::endl;
    std::cout << "Min Error: " << min_error << std::endl;
    std::cout << "Max Error: " << max_error << std::endl;
    std::cout << "Average Error: " << total_error / B.size() << std::endl;
}

SpMat sparse_matlab_to_eigen(SuiteSparseMatrix &ssm) {
    const int n = ssm.rows();

    SpMat A(n, n);
    std::vector<T> triplet_list(n);

    for (int k = 0; k < ssm.nnz(); ++k) {
        // Find the column for the k-th non-zero element
        int col = 0;
        while (col < n && ssm.jc()[col + 1] <= k) {
            col++;
        }
        triplet_list.push_back({static_cast<int64_t>(ssm.ir()[k]), col,
                                static_cast<float>(ssm.data()[k])});
    }
    A.setFromTriplets(triplet_list.begin(), triplet_list.end());

    return A;
}

int main(int argc, char *argv[]) {
    SuiteSparseMatrix ssm(argv[1]);
    const int n = ssm.rows();
    constexpr int s = 4;

    std::cout << "n: " << n << "\ns: " << s << '\n' << std::endl;

    SpMat A = sparse_matlab_to_eigen(ssm);
    Mat X = Mat::Constant(n, s, 0);
    Mat B = Mat::Constant(n, s, 1);

    constexpr CalcType TOLERANCE = 0.001;
    constexpr int MAX_ITERATIONS = 100;

    int iterations =
        dr_bcg_cpu::dr_bcg(A, X, B, TOLERANCE, MAX_ITERATIONS);

    std::cout << "Iterations: " << iterations << std::endl;

    std::cout << std::endl;
    print_error(A, X, B);

    return 0;
}
