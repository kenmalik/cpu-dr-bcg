#include "dr_bcg_cpu/dr_bcg_cpu.h"
#include <Eigen/QR>
#include <cmath>

#ifdef DEBUG

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

void check_nan(const Mat &mat, const std::string step) {
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            if (std::isnan(mat(i, j))) {
                std::ostringstream oss;
                oss << "Nan detected at (" << i << "," << j << ") after '"
                    << step << "'";
                throw std::runtime_error(oss.str());
            }
        }
    }
}

#define CHECK_NAN(mat, iteration)                                              \
    do {                                                                       \
        check_nan(mat,                                                         \
                  "iteration " + std::to_string(iteration) + ": " + #mat);     \
    } while (0);

#else

#define CHECK_NAN(mat, iteration)                                              \
    do {                                                                       \
    } while (0);

#endif

inline void reduced_QR(const Mat &A, Mat &Q, Mat &R) {
    const Eigen::Index m = A.rows();
    const Eigen::Index n = A.cols();
    Eigen::ColPivHouseholderQR<Mat> qr(A);
    Q = qr.householderQ() * Mat::Identity(m, n);
    R = qr.matrixQR().topLeftCorner(n, n).triangularView<Eigen::Upper>();
}

int dr_bcg_cpu::dr_bcg(const SpMat &A, Mat &X, const Mat &B, float tolerance,
                       int max_iterations) {
    int iterations = 0;

    Mat R = B - A * X;

    Mat w, sigma;
    reduced_QR(R, w, sigma);

    Mat s = w;

    Mat xi, zeta;

    for (iterations = 0; iterations < max_iterations; ++iterations) {
        xi.noalias() = (s.transpose() * A * s).inverse();
        CHECK_NAN(xi, iterations);

        X.noalias() += s * xi * sigma;
        CHECK_NAN(X, iterations);

        if ((B.col(0) - A * X.col(0)).norm() / B.col(0).norm() < tolerance) {
            ++iterations;
            break;
        } else {
            reduced_QR(w - A * s * xi, w, zeta);
            CHECK_NAN(w, iterations);
            CHECK_NAN(zeta, iterations);

            s.noalias() = w + s * zeta.transpose();
            CHECK_NAN(s, iterations);

            sigma.noalias() = zeta * sigma;
            CHECK_NAN(sigma, iterations);
        }
    }

    return iterations;
}
