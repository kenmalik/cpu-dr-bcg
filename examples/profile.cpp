#include "dr_bcg_cpu/dr_bcg_cpu.h"
#include "dr_bcg_cpu/profiler.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <ratio>
#include <string>
#include <suitesparse_matrix.h>
#include <unordered_map>

void print_timings(const SolverTimings &timings, int iterations) {
    std::unordered_map<Event, std::string> event_names{
        {Event::GetXi, "GetXi"},
        {Event::GetX, "GetX"},
        {Event::GetResidual, "GetResidual"},
        {Event::GetWZeta, "GetWZeta"},
        {Event::GetS, "GetS"},
        {Event::GetSigma, "GetSigma"}};

    using FpMilliseconds = std::chrono::duration<double, std::milli>;

    std::cout << "event,total(ms),average(ms)" << std::endl;
    for (int i = 0; i < static_cast<int>(Event::Count); ++i) {
        const auto &e = event_names[static_cast<Event>(i)];
        const auto &t = timings.totals[i];
        const auto &ms = std::chrono::duration_cast<FpMilliseconds>(t);
        std::cout << e << ',' << ms.count() << ','
                  << ms.count() / static_cast<double>(iterations) << std::endl;
    };
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
    if (argc < 3) {
        std::cerr << "Usage: profile [MAT_FILE] [BLOCK_SIZE]" << std::endl;
        return 1;
    }

    if (!std::filesystem::exists(argv[1])) {
        std::cerr << argv[1] << " does not exist" << std::endl;
        return 1;
    }

    const int s = atoi(argv[2]);
    if (s <= 0) {
        std::cerr << argv[2] << " is an invalid s value" << std::endl;
        return 1;
    }

    SuiteSparseMatrix ssm(argv[1]);
    const int n = ssm.rows();
    std::cerr << n << ' ' << s << std::endl;

    SpMat A = sparse_matlab_to_eigen(ssm);
    Mat X = Mat::Constant(n, s, 0);
    Mat B = Mat::Constant(n, s, 1);

    constexpr CalcType TOLERANCE = 0.001;
    constexpr int MAX_ITERATIONS = 100;

    SolverTimings timings;
    int iterations =
        dr_bcg_cpu::dr_bcg(A, X, B, TOLERANCE, MAX_ITERATIONS, &timings);

    print_timings(timings, iterations);

    return 0;
}
