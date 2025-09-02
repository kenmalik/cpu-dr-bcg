#include "dr_bcg_cpu/dr_bcg_cpu.h"
#include <iostream>
#include <suitesparse_matrix.h>

void verify(const SpMat &A, const Mat &X, const Mat &B) {
    std::cout << "Verification that A * X = B" << std::endl;

    std::cout << "A * X:\n" << A * X << std::endl;
    std::cout << "B:\n" << B << std::endl;
}

int main(int argc, char *argv[]) {
    SuiteSparseMatrix ssm(argv[1]);
    const int n = ssm.rows();
    constexpr int s = 4;

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

    Mat X = Mat::Constant(n, s, 0);
    Mat B = Mat::Constant(n, s, 1);

    std::cout << "A:\n"
              << Eigen::MatrixXf(A).format({2, 0, ", ", "\n"}) << std::endl;
    std::cout << "X:\n" << X << std::endl;
    std::cout << "B:\n" << B << std::endl;

    int iterations = dr_bcg_cpu::dr_bcg(A, X, B, 0.001);

    std::cout << "X Final:\n" << X << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;

    verify(A, X, B);

    return 0;
}
