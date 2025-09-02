#include "dr_bcg_cpu/dr_bcg_cpu.h"
#include <Eigen/QR>

/*
function [X_final, iterations] = DR_BCG(A, B, X, tol, maxit)
    iterations = 0;
    R = B - A * X;
    [w, sigma] = qr(R,'econ');
    s = w;

    for k = 1:maxit
        iterations = iterations + 1;
        xi = (s' * A * s)^-1;
        X = X + s * xi * sigma;
        if (norm(B(:,1) - A * X(:,1)) / norm(B(:,1))) < tol
            break
        else
            [w, zeta] = qr(w - A * s * xi,'econ');
            s = w + s * zeta';
            sigma = zeta * sigma;
        end
    end
    X_final = X;
end
 */

void dr_bcg_cpu::dr_bcg(const SpMat &A, Mat &X, const Mat &B, float tolerance,
                        int max_iterations) {
    int iterations = 0;
    Mat R = B - A * X;
    Eigen::HouseholderQR<Mat> qr(R);
    Mat w = qr.householderQ();
    Mat sigma = qr.matrixQR().triangularView<Eigen::Upper>();
    Mat s = w;

    for (int i = 0; i < max_iterations; ++i) {
        ++iterations;

        Mat xi = (s.transpose() * A * s).inverse();
        X = X + s * xi * sigma;
        if ((B.col(0) - A * X.col(0)).norm() / B.col(0).norm() < tolerance) {
            break;
        } else {
            Eigen::HouseholderQR<Mat> qr(w - A * s * xi);
            w = qr.householderQ();
            Mat zeta = qr.matrixQR().triangularView<Eigen::Upper>();
            s = w + s * zeta.transpose();
            sigma = zeta * sigma;
        }
    }
}
