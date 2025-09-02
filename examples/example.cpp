#include "dr_bcg_cpu/dr_bcg_cpu.h"
#include <iostream>

void verify(const SpMat &A, const Mat &X, const Mat &B) {
    std::cout << "Verification that A * X = B" << std::endl;

    std::cout << "A * X:\n" << A * X << std::endl;
    std::cout << "B:\n" << B << std::endl;
}

int main(int argc, char *argv[]) {
    constexpr int n = 8;
    constexpr int s = 4;

    SpMat A(n, n);
    std::vector<T> triplet_list(n);
    for (int i = 0; i < n; i++) {
        T t{i, i, 1};
        triplet_list.push_back(t);
    }
    A.setFromTriplets(triplet_list.begin(), triplet_list.end());

    Mat X(n, s);
    Mat B = Mat::Constant(n, s, 1);

    std::cout << "A:\n" << A << std::endl;
    std::cout << "X:\n" << X << std::endl;
    std::cout << "B:\n" << B << std::endl;

    int iterations = dr_bcg_cpu::dr_bcg(A, X, B);

    std::cout << "X Final:\n" << X << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;

    verify(A, X, B);

    return 0;
}
