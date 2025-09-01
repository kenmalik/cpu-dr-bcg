#include "dr_bcg_cpu/dr_bcg_cpu.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/QR>

using Eigen::Vector3f;

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

void dr_bcg_cpu::dr_bcg() {
  Vector3f v(1.3, 2.23, 3.4321);
  std::cout << v << std::endl;
  std::cout << "DR-BCG" << std::endl;
}
