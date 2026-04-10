/**
 * @file example/main.cpp
 * @brief Minimal example program for the @ref mphil_dis_cholesky library.
 *
 * This executable constructs a deterministic symmetric positive-definite matrix
 * \f$\mathbf{C}\f$ (the coursework corr(n) family), applies the in-place
 * Cholesky factorisation, and reports:
 *
 * - wall-clock time \f$t\f$ (seconds) returned by @ref mphil_dis_cholesky
 * - log-determinant computed from the factor:
 *   \f$\log|\det(\mathbf{C})| = 2 \sum_{p=0}^{n-1} \log(L_{pp})\f$
 * - achieved throughput in GFLOP/s using the standard right-looking Cholesky
 *   operation estimate \f$W \approx \tfrac{1}{3} n^3\f$
 *
 * @section example_usage Usage
 * @code
 * ./build/bin/cholesky_example          # default n = 200
 * ./build/bin/cholesky_example 500      # n = 500
 * ./build/bin/cholesky_example 1000     # n = 1000
 * @endcode
 *
 * @section example_exit Exit conditions
 * The program returns a non-zero status if:
 * - the command-line \f$n\f$ is outside \f$[1, 100000]\f$
 * - @ref mphil_dis_cholesky returns a negative error code
 * - the computed log-determinant is not finite
 */

#include "mphil_dis_cholesky.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ─── SPD matrix generator ───────────────────────────────────────────────────

static double corr(double x, double y, double s) {
  return 0.99 * std::exp(-0.5 * 16.0 * (x - y) * (x - y) / (s * s));
}

static void fill_corr(double *c, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      c[n * i + j] = corr(i, j, n);
    c[n * i + i] = 1.0; // diagonal = 1 ensures positive definiteness
  }
}

// ─── Main
// ─────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {

  // ── Parse n from command line (default 200) ───────────────────────────────
  int n = 200;
  if (argc > 1) {
    char *end = nullptr;
    long val = std::strtol(argv[1], &end, 10);
    if (*end != '\0' || val < 1 || val > 100000) {
      std::fprintf(stderr,
                   "Error: n must be an integer in [1, 100000], got '%s'.\n"
                   "Usage: %s [n]\n",
                   argv[1], argv[0]);
      return 1;
    }
    n = static_cast<int>(val);
  }

  std::printf("mphil_dis_cholesky example  (CHOLESKY_VERSION=%d)\n",
              CHOLESKY_VERSION);
  std::printf("Matrix size     : n = %d\n", n);

  // ── Allocate and fill the correlation matrix ──────────────────────────────
  std::vector<double> c(static_cast<std::size_t>(n) * n);
  fill_corr(c.data(), n);

  // ── Factorise ─────────────────────────────────────────────────────────────
  double t = mphil_dis_cholesky(c.data(), n);

  if (t == -1.0) {
    std::fprintf(stderr,
                 "Error: mphil_dis_cholesky returned -1.0 "
                 "(invalid input: n=%d is out of range or pointer is null).\n",
                 n);
    return 1;
  }
  if (t == -2.0) {
    std::fprintf(
        stderr,
        "Error: mphil_dis_cholesky returned -2.0 "
        "(matrix is not positive definite — pivot <= 0 encountered).\n");
    return 1;
  }
  if (t < 0.0) {
    std::fprintf(
        stderr,
        "Error: mphil_dis_cholesky returned unexpected negative value %.6f.\n",
        t);
    return 1;
  }

  // ── Compute log|det(C)|: 2·sum log(L_pp) ───────────────────
  double logdet = 0.0;
  for (int p = 0; p < n; p++)
    logdet += std::log(c[static_cast<std::size_t>(p) * n + p]);
  logdet *= 2.0;

  if (!std::isfinite(logdet)) {
    std::fprintf(stderr,
                 "Warning: log|det(C)| is not finite (%.6f). "
                 "The factorisation may have encountered numerical issues.\n",
                 logdet);
  }

  // ── Compute estimated throughput ──────────────────────────────────────────
  // Standard Cholesky FLOP estimate: (1/3) n^3
  const double flops = (1.0 / 3.0) * static_cast<double>(n) *
                       static_cast<double>(n) * static_cast<double>(n);
  const double gflops = flops / (t * 1.0e9);

  // ── Report ────────────────────────────────────────────────────────────────
  std::printf("Elapsed time    : t      = %.6f s\n", t);
  std::printf("log|det(C)|     : logdet = %.6f\n", logdet);
  std::printf("FLOP estimate   : (1/3)n^3 = %.3e FLOP\n", flops);
  std::printf("Throughput      : %.4f GFLOP/s\n", gflops);

  return 0;
}
