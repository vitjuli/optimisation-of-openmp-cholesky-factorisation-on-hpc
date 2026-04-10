/**
 * @file test_n2.cpp
 * @brief Minimal smoke test for the 2x2 example.
 *
 * This test exercises @ref mphil_dis_cholesky on the coursework’s 2x2
 * reference matrix \f$C=\begin{bmatrix}4&2\\2&26\end{bmatrix}\f$ and checks
 * that the in-place result matches the expected stored output
 * \f$\begin{bmatrix}2&1\\1&5\end{bmatrix}\f$ within \f$10^{-12}\f$.
 *
 * The implementation stores \f$L\f$ in the lower triangle and mirrors it into
 * the upper triangle (i.e. the array becomes symmetric on success), hence the
 * expected flattened buffer is [2, 1, 1, 5] rather than a strictly
 * lower-triangular layout.
 *s
 * The test also verifies that the return value is non-negative (successful
 * factorisation with a valid timing measurement).
 *
 * This file is compiled into the CholeskyN2 ctest target and serves as a
 * fast go/no-go check before running the full suite in @ref test_cholesky.cpp.
 */
#include "mphil_dis_cholesky.h"
#include <cmath>
#include <cstdio>

static bool nearly_equal(double a, double b, double tol = 1e-12) {
  return std::fabs(a - b) <= tol;
}

int main() {
  // C2 = [[4,2],[2,26]]
  double c[4] = {4.0, 2.0, 2.0, 26.0};

  double t = mphil_dis_cholesky(c, 2);
  if (t < 0.0) {
    std::fprintf(stderr, "FAIL: mphil_dis_cholesky returned %.3f\n", t);
    return 1;
  }

  // Expected after Cholesky:
  // [[2,1],[1,5]]
  if (!nearly_equal(c[0], 2.0) || !nearly_equal(c[1], 1.0) ||
      !nearly_equal(c[2], 1.0) || !nearly_equal(c[3], 5.0)) {
    std::fprintf(stderr, "FAIL: got [%.6f %.6f; %.6f %.6f]\n", c[0], c[1], c[2],
                 c[3]);
    return 1;
  }

  std::printf("PASS test_n2 (t=%.9f s)\n", t);
  return 0;
}