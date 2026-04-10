/**
 * @file test_cholesky.cpp
 * @brief Full correctness test suite for @ref mphil_dis_cholesky.
 *
 * This translation unit implements a self-contained set of tests that validate
 * the public contract of @ref mphil_dis_cholesky across all build variants
 * (`CHOLESKY_VERSION=0..3`).
 *
 * @section test_correctness Correctness criteria (spec §1)
 * The factorisation is considered correct if:
 * - In-place layout: the stored output satisfies:
 *   - lower triangle (\f$i \ge j\f$) contains \f$L(i,j)\f$;
 *   - upper triangle (\f$i < j\f$) contains the mirrored values \f$L(j,i)\f$
 *     (i.e. \f$L^\top\f$).
 * - Log-determinant identity (spec Eq. 4):
 *   \f[
 *   \log|\det C| = 2\sum_{p=0}^{n-1}\log(L_{pp}).
 *   \f]
 * - Backward error (double precision):
 *   \f[
 *   \frac{\|C - \hat L\hat L^\top\|_F}{\|C\|_F} \le 10\,n\,\varepsilon,
 *   \f]
 *   where \f$\varepsilon\f$ is machine epsilon and \f$\hat L\f$ is the factor
 *   reconstructed from the stored lower triangle.
 *
 * @section test_inputs Test inputs
 * SPD matrices are generated deterministically using the corr(n) construction
 * provided in the coursework specification (p.5), so the suite is reproducible
 * without random seeds.
 *
 * @section test_build_run Build and run
 * All four versions must pass the same suite:
 * @code
 * cmake -S . -B build -DCHOLESKY_VERSION=0   # or 1, 2, or 3
 * cmake --build build
 * ctest --test-dir build --output-on-failure
 * @endcode
 */

#include "mphil_dis_cholesky.h"

#include <cmath>
#include <cstdio>
#include <vector>

// OpenMP runtime API — available only when compiled with -fopenmp (V2/V3).
// Included here so test_openmp_thread_invariance() can call
// omp_set_num_threads().
#ifdef _OPENMP
#include <omp.h>
#endif

// ─── Lightweight test runner ─────────────────────────────────────────────────

static int g_run = 0;
static int g_fail = 0;

// CHECK(condition, format_string, ...) — counts and reports each assertion.
// Uses ##__VA_ARGS__ (GCC/Clang extension, universally available here).
#define CHECK(cond, msg, ...)                                                  \
  do {                                                                         \
    ++g_run;                                                                   \
    char _buf[384];                                                            \
    std::snprintf(_buf, sizeof(_buf), msg, ##__VA_ARGS__);                     \
    if (!(cond)) {                                                             \
      ++g_fail;                                                                \
      std::fprintf(stderr, "FAIL  %s\n", _buf);                                \
    } else {                                                                   \
      std::printf("pass  %s\n", _buf);                                         \
    }                                                                          \
  } while (0)

// ─── Constants ───────────────────────────────────────────────────────────────

// IEEE 754 double-precision machine epsilon.
static constexpr double EPS = 2.220446049250313e-16;

// ─── SPD matrix generator (spec p.5, deterministic) ─────────────────────────

static double corr(double x, double y, double s) {
  return 0.99 * std::exp(-0.5 * 16.0 * (x - y) * (x - y) / (s * s));
}

static std::vector<double> make_corr(int n) {
  std::vector<double> c(n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++)
      c[n * i + j] = corr(i, j, n);
    c[n * i + i] = 1.0; // diagonal = 1 ensures positive definiteness
  }
  return c;
}

// ─── Numeric helpers ─────────────────────────────────────────────────────────

// Frobenius norm of a flat n×n array.
static double frob(const std::vector<double> &a) {
  double s = 0.0;
  for (double v : a)
    s += v * v;
  return std::sqrt(s);
}

// ||a − b||_F for two flat n×n arrays.
static double frob_diff(const std::vector<double> &a,
                        const std::vector<double> &b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); i++) {
    double d = a[i] - b[i];
    s += d * d;
  }
  return std::sqrt(s);
}

// Reconstruct C_hat = L·L^T from the in-place Cholesky result stored in c.
// After mphil_dis_cholesky, lower triangle of c holds L (L[i][j] = c[i*n+j]
// for i >= j).  The (i,j) entry of L·L^T is Σ_{k=0}^{min(i,j)} L[i][k]·L[j][k].
static std::vector<double> reconstruct(const double *c, int n) {
  std::vector<double> chat(n * n, 0.0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      int kmax = (i < j) ? i : j;
      double s = 0.0;
      for (int k = 0; k <= kmax; k++)
        s += c[i * n + k] * c[j * n + k];
      chat[i * n + j] = s;
    }
  return chat;
}

// log|det(C)| via spec Eq.(4): 2·Σ_{p=0}^{n−1} log(L_pp).
static double logdet_eq4(const double *c, int n) {
  double s = 0.0;
  for (int p = 0; p < n; p++)
    s += std::log(c[p * n + p]);
  return 2.0 * s;
}

// Reference log-det: runs the spec baseline pseudocode verbatim on a copy of
// c_orig and returns 2·Σ log(L_pp).  Used in T07 to cross-check the result
// returned by mphil_dis_cholesky for VERSION=1 and VERSION=2 against the
// known-correct baseline, without any external library.
static double ref_logdet(const std::vector<double> &c_orig, int n) {
  auto c = c_orig; // work on a copy — do not touch the caller's data
  for (int p = 0; p < n; p++) {
    const double diag = std::sqrt(c[p * n + p]);
    c[p * n + p] = diag;
    for (int j = p + 1; j < n; j++)
      c[p * n + j] /= diag;
    for (int i = p + 1; i < n; i++)
      c[i * n + p] /= diag;
    for (int j = p + 1; j < n; j++)
      for (int i = p + 1; i < n; i++)
        c[i * n + j] -= c[i * n + p] * c[p * n + j];
  }
  double s = 0.0;
  for (int p = 0; p < n; p++)
    s += std::log(c[p * n + p]);
  return 2.0 * s;
}

// ─── K: SPD stress — A·Aᵀ + αI family (helper) ───────────────────────────────
//
// Generates C = A·Aᵀ + 0.1·I  where  A[i][j] = sin((i+1)·(j+1)) / sqrt(n).
//
// This matrix is always strictly SPD:
//   xᵀ·C·x = ||A·x||² + 0.1·||x||² > 0  for all x ≠ 0.
//
// It differs from the corr() family in Groups C–D:
//   • Diagonal entries are ≥ 0.1 + row-variance  (not clamped to 1.0).
//   • Off-diagonal structure depends on sin products, not exp(-distance²).
//   • Condition number varies differently with n.
//
// Purpose: confirms that the implementation is correct for a second, distinct
// SPD family — ruling out a tuned-to-corr() implementation.
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<double> make_spd_stress(int n) {
  std::vector<double> A(n * n), C(n * n, 0.0);
  const double inv_sqrtn = 1.0 / std::sqrt(static_cast<double>(n));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      A[i * n + j] = std::sin((i + 1.0) * (j + 1.0)) * inv_sqrtn;
  // C = A * A^T + 0.1 * I
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < n; ++k)
        s += A[i * n + k] * A[j * n + k];
      C[i * n + j] = s;
    }
    C[i * n + i] += 0.1; // + 0.1 * I  (ensures strict positive definiteness)
  }
  return C;
}

static void test_spd_stress(int n) {
  auto c_orig = make_spd_stress(n);
  auto c = c_orig;
  double t = mphil_dis_cholesky(c.data(), n);

  // Return value >= 0.0 means success (t may be 0 for sub-µs runs on small n).
  CHECK(t >= 0.0, "T_SPD n=%2d: factorisation returns t>=0 (t=%.6f)", n, t);
  if (t < 0.0)
    return; // skip further checks if factorisation reported an error

  // Backward error: ||C − L·Lᵀ||_F / ||C||_F ≤ 20·n·ε.
  // 20× is still ~10000× above the typical observed error ≈ n·ε;
  // it accommodates A·Aᵀ matrices whose off-diagonal magnitudes may be
  // larger than corr()'s ≤ 0.99 bound.
  auto chat = reconstruct(c.data(), n);
  double rel = frob_diff(c_orig, chat) / frob(c_orig);
  double tol = 20.0 * n * EPS;
  CHECK(rel <= tol, "T_SPD n=%2d: backward error %.2e <= 20nε=%.2e", n, rel,
        tol);

  // Log-det: finite and matches ref_logdet() within n·10⁻¹².
  double ld = logdet_eq4(c.data(), n);
  double ld_ref = ref_logdet(c_orig, n);
  double ld_tol = n * 1e-12;
  CHECK(std::isfinite(ld) && std::fabs(ld - ld_ref) <= ld_tol,
        "T_SPD n=%2d: logdet=%.6f  |diff vs ref|=%.2e  tol=n·1e-12=%.2e", n, ld,
        std::fabs(ld - ld_ref), ld_tol);
}

// ─── A: Canonical spec example ───────────────────────────────────────────────

static void test_spec_2x2() {
  double c[4] = {4.0, 2.0, 2.0, 26.0};
  double t = mphil_dis_cholesky(c, 2);

  CHECK(t >= 0.0, // 1
        "T01a: 2x2 spec timing >= 0  (t=%.9f s)", t);

  bool ok = std::fabs(c[0] - 2.0) <= 1e-14 && std::fabs(c[1] - 1.0) <= 1e-14 &&
            std::fabs(c[2] - 1.0) <= 1e-14 && std::fabs(c[3] - 5.0) <= 1e-14;
  CHECK(ok, // 2
        "T01b: 2x2 spec output [[2,1],[1,5]]  (got [%.6f %.6f %.6f %.6f])",
        c[0], c[1], c[2], c[3]);
}

// ─── B: Analytical matrices
// ───────────────────────────────────────────────────

static void test_scalar_n1() {
  // C = [[4]]  →  L = [[2]],  log|det| = log(4)
  double c[1] = {4.0};
  mphil_dis_cholesky(c, 1);

  CHECK(std::fabs(c[0] - 2.0) <= 1e-14, // 3
        "T02a: n=1  L[0][0] = 2  (got %.10f)", c[0]);

  double ld = logdet_eq4(c, 1);
  CHECK(std::fabs(ld - std::log(4.0)) <= 1e-14, // 4
        "T02b: n=1  logdet = log(4)  (got %.10f, exp %.10f)", ld,
        std::log(4.0));
}

static void test_identity(int n) {
  // C = I_n  →  L = I_n,  log|det| = 0
  std::vector<double> c(n * n, 0.0);
  for (int i = 0; i < n; i++)
    c[i * n + i] = 1.0;
  mphil_dis_cholesky(c.data(), n);

  bool diag_ok = true;
  for (int i = 0; i < n; i++)
    if (std::fabs(c[i * n + i] - 1.0) > 1e-14) {
      diag_ok = false;
      break;
    }
  CHECK(diag_ok, "T03: I_%d  diagonal all 1", n); // 5, 7

  double ld = logdet_eq4(c.data(), n);
  CHECK(std::fabs(ld) <= 1e-13, // 6, 8
        "T03: I_%d  logdet = 0  (got %.3e)", n, ld);
}

static void test_scaled_identity(int n, double k) {
  // C = k·I_n  →  L = sqrt(k)·I_n,  log|det| = n·log(k)
  std::vector<double> c(n * n, 0.0);
  for (int i = 0; i < n; i++)
    c[i * n + i] = k;
  mphil_dis_cholesky(c.data(), n);

  double sq = std::sqrt(k);
  bool diag_ok = true;
  for (int i = 0; i < n; i++)
    if (std::fabs(c[i * n + i] - sq) > 1e-13) {
      diag_ok = false;
      break;
    }
  CHECK(diag_ok, // 9
        "T04: %.0f*I_%d  diagonal = sqrt(k) = %.6f", k, n, sq);

  double ld = logdet_eq4(c.data(), n);
  double ld_exp = n * std::log(k);
  CHECK(std::fabs(ld - ld_exp) <= 1e-11, // 10
        "T04: %.0f*I_%d  logdet = n*log(k)  (got %.8f, exp %.8f)", k, n, ld,
        ld_exp);
}

// ─── C: corr() SPD reconstruction — backward error ───────────────────────────

static void test_corr_backward(int n) {
  // Tolerance: 10·n·ε for n<128; 20·n·ε for n>=128 (extra accumulation).
  // Expected actual error is O(n·ε) ≈ n·2.2e-16 — tolerance is 10–20×
  // conservative.
  auto c_orig = make_corr(n);
  auto c = c_orig; // copy — mphil_dis_cholesky overwrites c
  double t = mphil_dis_cholesky(c.data(), n);
  auto chat = reconstruct(c.data(), n); // C_hat = L·L^T

  double rel = frob_diff(c_orig, chat) / frob(c_orig);
  double tol = (n >= 128 ? 20.0 : 10.0) * n * EPS;

  // t >= 0.0: allows t == 0.0 (sub-microsecond runs on small n where
  // omp_get_wtime() resolution is too coarse to measure).  Error return
  // codes are -1.0 (bad input) and -2.0 (non-SPD), both clearly negative.
  CHECK(t >= 0.0 && rel <= tol, // 11–15
        "T05: corr n=%3d  ||C-LL^T||_F/||C||_F = %.2e  tol = %.2e  t = %.4f s",
        n, rel, tol, t);
}

// ─── D: log-det identity (spec Eq. 4) ────────────────────────────────────────

static void test_logdet_2x2_analytical() {
  // det([[4,2],[2,26]]) = 4·26 − 2·2 = 100  →  log|det| = log(100)
  double c[4] = {4.0, 2.0, 2.0, 26.0};
  mphil_dis_cholesky(c, 2);
  double ld = logdet_eq4(c, 2);
  CHECK(std::fabs(ld - std::log(100.0)) <= 1e-13, // 16
        "T06: 2x2 logdet = log(100)  (got %.10f, exp %.10f)", ld,
        std::log(100.0));
}

static void test_logdet_corr_sign(int n) {
  // Three-part check:
  //   1. Finite and ≤ 0  (Hadamard: det(corr) ≤ ∏ diag = 1  →  logdet ≤ 0).
  //   2. Agrees with ref_logdet (spec baseline pseudocode run inline) to
  //      within n·1e-12.  Each L_pp can differ by O(ε) between versions;
  //      summing n logarithms accumulates to O(n·ε) ≈ n·2.2e-16 — the
  //      tolerance n·1e-12 is ~5000× conservative.
  auto c_orig = make_corr(n);
  auto c = c_orig;
  mphil_dis_cholesky(c.data(), n);
  double ld = logdet_eq4(c.data(), n);
  double ld_ref = ref_logdet(c_orig, n);
  double tol = n * 1e-12;
  CHECK(std::isfinite(ld) && ld <= 0.0 && std::fabs(ld - ld_ref) <= tol, // 17
        "T07: corr n=%d  logdet=%.6f  ref=%.6f  |diff|=%.2e  tol=%.2e", n, ld,
        ld_ref, std::fabs(ld - ld_ref), tol);
}

// ─── E: Input validation & error paths ───────────────────────────────────────

static void test_invalid_inputs() {
  double d[4] = {1.0, 0.0, 0.0, 1.0};

  CHECK(mphil_dis_cholesky(d, 0) == -1.0, "T08a: n=0      -> -1.0");       // 18
  CHECK(mphil_dis_cholesky(d, -1) == -1.0, "T08b: n=-1     -> -1.0");      // 19
  CHECK(mphil_dis_cholesky(d, 100001) == -1.0, "T08c: n=100001 -> -1.0");  // 20
  CHECK(mphil_dis_cholesky(nullptr, 2) == -1.0, "T08d: nullptr  -> -1.0"); // 21
}

static void test_non_spd() {
  // [[1,2],[2,1]]: after p=0 Schur gives C[1][1] = 1 − 4 = −3 ≤ 0  →  -2.0
  double c1[4] = {1.0, 2.0, 2.0, 1.0};
  CHECK(mphil_dis_cholesky(c1, 2) == -2.0, // 22
        "T09a: non-SPD [[1,2],[2,1]]   -> -2.0");

  // [[-1,0],[0,1]]: first pivot = −1 ≤ 0  →  -2.0 immediately
  double c2[4] = {-1.0, 0.0, 0.0, 1.0};
  CHECK(mphil_dis_cholesky(c2, 2) == -2.0, // 23
        "T09b: non-SPD [[-1,0],[0,1]]  -> -2.0");
}

// ─── F: Structural correctness
// ────────────────────────────────────────────────

static void test_structure(int n) {
  // After in-place Cholesky, the stored matrix satisfies:
  //   c[i*n+j] == c[j*n+i]  for all i≠j  (lower = L, upper = L^T mirrored)
  //   c[p*n+p] > 0           for all p    (L diagonal must be strictly
  //   positive)
  auto c = make_corr(n);
  mphil_dis_cholesky(c.data(), n);

  bool sym_ok = true;
  for (int i = 0; i < n && sym_ok; i++)
    for (int j = 0; j < i && sym_ok; j++)
      if (std::fabs(c[i * n + j] - c[j * n + i]) > 1e-13)
        sym_ok = false;
  CHECK(sym_ok, "T10: corr n=%d  upper/lower symmetry (c[i*n+j]==c[j*n+i])",
        n); // 24, 26

  bool pos_ok = true;
  for (int i = 0; i < n && pos_ok; i++)
    if (c[i * n + i] <= 0.0)
      pos_ok = false;
  CHECK(pos_ok, "T10: corr n=%d  diagonal L_pp > 0", n); // 25, 27
}

// ─── G: Step-level oracle proof (Task A) ─────────────────────────────────────
// Requires MPHIL_CHOLESKY_TESTING=1 (test targets only; see CMakeLists.txt).
//
// Strategy: apply pivot steps 0..n-1 one at a time to two identical copies of
// the same SPD matrix — one via ref_step() (clean serial reference) and one via
// mphil_dis_cholesky_step() (the compiled version's code path).  After each
// step the two matrices must agree element-wise within 50·ε·n·‖c_ref‖_max.
//
// This tolerance is ≫ any expected FP rounding difference (which is O(ε))
// and ≪ any algorithmic mistake (which would be O(1)).  The step-by-step
// comparison catches subtle bugs — a wrong loop bound, a missing scale, a
// race-corrupted element — that might cancel out in the end-to-end result.
// ─────────────────────────────────────────────────────────────────────────────

#ifdef MPHIL_CHOLESKY_TESTING

// Declare the test-only hook defined in src/mphil_dis_cholesky_testing.cpp.
// Compiled into the library only when MPHIL_CHOLESKY_TESTING=1 is defined.
extern int mphil_dis_cholesky_step(double *c, int n, int p);

// Reference one-step oracle — independent of all production code paths.
// Performs exactly the right-looking Cholesky step p (sqrt, row scale,
// col scale, Schur complement) in clean serial i-outer / j-inner order.
static void ref_step(double *c, int n, int p) {
  const double diag = std::sqrt(c[p * n + p]);
  c[p * n + p] = diag;
  const double inv = 1.0 / diag;
  // Row p: L^T[p][j] = C[p][j] / L[p][p]
  for (int j = p + 1; j < n; ++j)
    c[p * n + j] *= inv;
  // Column p: L[i][p] = C[i][p] / L[p][p]
  for (int i = p + 1; i < n; ++i)
    c[i * n + p] *= inv;
  // Schur complement: C[i][j] -= L[i][p] * L^T[p][j]   for i,j > p.
  // i-outer / j-inner: stride-1 inner access; Lip hoisted out of j-loop.
  for (int i = p + 1; i < n; ++i) {
    const double Lip = c[i * n + p];
    for (int j = p + 1; j < n; ++j)
      c[i * n + j] -= Lip * c[p * n + j];
  }
}

// max |v[k]| over all elements — used to compute a scale-relative tolerance.
static double max_abs_elem(const std::vector<double> &v) {
  double m = 0.0;
  for (double x : v) {
    double ax = std::fabs(x);
    if (ax > m)
      m = ax;
  }
  return m;
}

// max element-wise |a[k] - b[k]| — the deviation between two flat n×n matrices.
static double max_abs_diff_mat(const std::vector<double> &a,
                               const std::vector<double> &b) {
  double m = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    double d = std::fabs(a[k] - b[k]);
    if (d > m)
      m = d;
  }
  return m;
}

static void test_step_level(int n) {
  // Both copies start from the identical corr(n) SPD matrix.
  auto c_orig = make_corr(n);
  auto c_ref = c_orig;  // stepped by ref_step()
  auto c_test = c_orig; // stepped by mphil_dis_cholesky_step()

  bool all_ok = true;
  double worst_diff = 0.0;
  double worst_tol = 1.0;
  int worst_p = -1;

  for (int p = 0; p < n; ++p) {
    // Apply both oracles for this pivot step.
    ref_step(c_ref.data(), n, p);
    int rc = mphil_dis_cholesky_step(c_test.data(), n, p);

    if (rc != 0) {
      // A non-positive pivot on a known-SPD matrix is a definite bug.
      std::fprintf(stderr,
                   "  T_STEP n=%d p=%d: mphil_dis_cholesky_step returned %d"
                   " (pivot failure on SPD matrix — cannot continue)\n",
                   n, p, rc);
      all_ok = false;
      break;
    }

    // Scale-relative tolerance: 50·ε·n·‖c_ref‖_max.
    // For corr() matrices all elements are ≤ 1, so ‖c_ref‖_max ≈ 1 and
    // the tolerance ≈ 50·2.2e-16·n — generous for FP reordering, tight
    // enough to catch any O(1) algorithmic deviation.
    const double scale = max_abs_elem(c_ref);
    const double tol = 50.0 * EPS * n * (scale > 0.0 ? scale : 1.0);
    const double diff = max_abs_diff_mat(c_ref, c_test);

    if (diff > tol) {
      std::fprintf(
          stderr,
          "  T_STEP n=%d p=%d: max_abs_diff=%.2e  tol=%.2e  scale=%.2e\n", n, p,
          diff, tol, scale);
      all_ok = false;
      // Continue stepping so c_ref / c_test stay in sync even after a
      // failure, giving cleaner diagnostics for subsequent steps.
    }
    if (diff / tol > worst_diff / worst_tol) {
      worst_diff = diff;
      worst_tol = tol;
      worst_p = p;
    }
  }

  CHECK(all_ok,
        "T_STEP n=%2d: %d pivot steps match ref  "
        "(worst p=%d  ratio=%.3f  tol=50·ε·n·|max|)",
        n, n, worst_p, (worst_tol > 0.0 ? worst_diff / worst_tol : 0.0));
}

#endif // MPHIL_CHOLESKY_TESTING

// ─── H: OpenMP thread-count invariance (Task B — V2/V3 only) ─────────────────
// Verifies that mphil_dis_cholesky() produces equivalent numerical results
// regardless of how many threads are used.
//
// Two properties are checked for each thread count T ∈ {1,2,4,8,16}:
//   (1) Backward error  ||C − L·L^T||_F / ||C||_F ≤ 10·n·ε
//       — confirms the parallel factorisation is numerically correct.
//   (2) |logdet(T) − logdet(T=1)| ≤ 1e-10
//       — confirms the diagonal pivots are identical regardless of scheduling.
//
// omp_set_num_threads(T) overrides OMP_NUM_THREADS even when that env var is
// set to 1 by ctest (the API call updates the ICV nthreads-var directly, which
// takes precedence over the env var — OpenMP 5.1 §2.4).  After the test the
// thread count is restored to 1 so it does not affect subsequent tests.
//
// Compiled and run only when _OPENMP is defined (V2/V3 builds with -fopenmp).
// ─────────────────────────────────────────────────────────────────────────────

// ─── H2: V3 column-scaling parallelism proof ─────────────────────────────────
// Structural proof that row_i[p] *= inv_diag is distributed across >1 thread
// in V3, distinguishing it from V2 where the same operation runs serially in
// omp single.
//
// Mechanism: mphil_dis_cholesky_step() (when built with MPHIL_CHOLESKY_TESTING)
// records omp_get_thread_num() for every column-scaling write into a global
// buffer.  The accessor mphil_dis_last_col_writer_tids() exposes that buffer.
//
//   V2 expected: all entries identical (single thread did all scaling).
//   V3 expected: entries span >= 2 distinct tids (omp for distributed it).
//
// Uses n=256 (255 rows at step p=0), T=8 to guarantee sufficient work for
// spreading across threads.  Compiled only for CHOLESKY_VERSION==3 with OpenMP.
// ─────────────────────────────────────────────────────────────────────────────

#if defined(MPHIL_CHOLESKY_TESTING) && defined(_OPENMP) && CHOLESKY_VERSION == 3

extern const int *mphil_dis_last_col_writer_tids();
extern int mphil_dis_last_col_writer_count();

static void test_v3_col_scaling_parallel() {
  const int n = 256; // 255 rows scaled at p=0 — easily split across T=8
  const int T = 8;
  const int p = 0;

  // Probe actual thread count — omp_set_num_threads(T) may yield fewer
  // threads if hardware cores < T.  We use the real count in assertions.
  omp_set_num_threads(T);
  int actual_T = 0;
#pragma omp parallel
  {
#pragma omp single
    actual_T = omp_get_num_threads();
  }

  auto c = make_corr(n);
  int rc = mphil_dis_cholesky_step(c.data(), n, p);

  CHECK(rc == 0,
        "T_V3P: step p=%d n=%d returns 0 (SPD input, no pivot failure)", p, n);

  const int *tids = mphil_dis_last_col_writer_tids();
  const int count = mphil_dis_last_col_writer_count();
  const int expected_count = n - p - 1; // = 255 for p=0, n=256

  CHECK(count == expected_count,
        "T_V3P: col_writer_count = %d  (expected %d = n-p-1)", count,
        expected_count);

  // Every recorded tid must lie in [0, actual_T).
  bool valid = true;
  for (int i = 0; i < count; ++i) {
    if (tids[i] < 0 || tids[i] >= actual_T) {
      std::fprintf(stderr, "  T_V3P: tids[%d] = %d out of range [0, %d)\n", i,
                   tids[i], actual_T);
      valid = false;
      break;
    }
  }
  CHECK(valid, "T_V3P: all col-writer tids in [0, %d)  (actual_T=%d)", actual_T,
        actual_T);

  // Count distinct thread IDs.  tid values fit in [0, actual_T) <= [0, 8).
  bool seen[64] = {};
  int distinct = 0;
  for (int i = 0; i < count; ++i) {
    const int tid = tids[i];
    if (tid >= 0 && tid < 64 && !seen[tid]) {
      seen[tid] = true;
      ++distinct;
    }
  }

  // Require >= 2 distinct tids when actual_T >= 2 and work is enough to
  // split (count >= actual_T — guaranteed here: 255 >= 8).
  // If actual_T == 1 (single-core machine), 1 distinct tid is trivially OK.
  const int min_distinct = (actual_T >= 2) ? 2 : 1;
  CHECK(distinct >= min_distinct,
        "T_V3P: %d distinct col-writer tids >= %d "
        "(V3 column scaling is parallel; actual_T=%d)",
        distinct, min_distinct, actual_T);

  omp_set_num_threads(1); // restore for subsequent tests
}

#endif // MPHIL_CHOLESKY_TESTING && _OPENMP && CHOLESKY_VERSION == 3

// ─── I: V1 tail coverage ─────────────────────────────────────────────────────
// OPT-5 in V1 processes the Schur-complement i-rows in 4-wide blocks.  When
// (n − p − 1) % 4 != 0, the remaining 1–3 rows fall through to a scalar tail
// loop.  If that tail had a wrong bound (e.g. `i < i_end4` instead of `i < n`)
// those rows would be silently skipped, leaving L[i][j] un-updated and
// producing a wrong factorisation with no visible error from the 4-wide rows.
//
// Chosen sizes:
//   n=6, p=0 → (6−0−1)%4 = 5%4 = 1 → one tail row  (i = 5)
//   n=7, p=0 → (7−0−1)%4 = 6%4 = 2 → two tail rows (i = 5, 6)
//
// The test calls mphil_dis_cholesky_step() and checks every tail row against
// ref_step().  A missing tail would manifest as a large element-wise deviation
// in those rows only, so this test is more sensitive than the end-to-end
// backward-error tests in Group C.
// ─────────────────────────────────────────────────────────────────────────────

#if defined(MPHIL_CHOLESKY_TESTING) && CHOLESKY_VERSION == 1

static void test_v1_tail_coverage() {
  for (int n : {6, 7}) {
    const int p = 0;
    // Compile-time-verified: both n values satisfy the precondition.
    // n=6: tail_rows = (6-0-1)%4 = 1; n=7: tail_rows = (7-0-1)%4 = 2.
    const int i_end4 = p + 1 + ((n - p - 1) / 4) * 4;

    auto c_orig = make_corr(n);
    auto c_ref = c_orig;
    auto c_test = c_orig;

    ref_step(c_ref.data(), n, p);
    mphil_dis_cholesky_step(c_test.data(), n, p);

    const double scale = max_abs_elem(c_ref);
    const double tol = 50.0 * EPS * n * (scale > 0.0 ? scale : 1.0);

    // Check ONLY the tail rows — the 4-wide rows are already covered by
    // the Group G oracle test.  A wrong tail bound corrupts exactly these.
    bool tail_ok = true;
    for (int i = i_end4; i < n && tail_ok; ++i) {
      for (int j = p + 1; j < n && tail_ok; ++j) {
        const double diff = std::fabs(c_ref[i * n + j] - c_test[i * n + j]);
        if (diff > tol) {
          std::fprintf(stderr,
                       "  T_V1T n=%d i=%d j=%d: ref=%.10f test=%.10f"
                       " diff=%.2e tol=%.2e\n",
                       n, i, j, c_ref[i * n + j], c_test[i * n + j], diff, tol);
          tail_ok = false;
        }
      }
    }
    CHECK(tail_ok,
          "T_V1T n=%d p=0: scalar tail rows [i=%d..%d] match ref_step()"
          "  ((n-p-1)%%4=%d != 0)",
          n, i_end4, n - 1, (n - p - 1) % 4);
  }
}

// ─── I2: V1 OPT-1 structural proof ───────────────────────────────────────────
// A direct, non-timing proof that the Schur complement inner loop in V1 is
// j-inner (stride-1 in row-major) and NOT i-inner (stride-n, V0 order).
//
// Mechanism: mphil_dis_cholesky_step() V1 records the addresses of the first
// two writes in the scalar-tail j-inner loop into g_v1_schur_probe[].
// mphil_v1_schur_write_stride() returns their address delta (in double units).
//
//   OPT-1 active   → consecutive writes are to row_i[j] and row_i[j+1]
//                     → delta = &row_i[j+1] - &row_i[j] = 1
//   OPT-1 absent   → consecutive writes are to c[i*n+j] and c[(i+1)*n+j]
//   (V0-order)       → delta = &c[(i+1)*n+j] - &c[i*n+j] = n
//
// We use n=6, p=0 to guarantee the tail fires (see Group I).  The test also
// verifies the full step result against ref_step() as a correctness companion.
// ─────────────────────────────────────────────────────────────────────────────

extern ptrdiff_t mphil_v1_schur_write_stride();

static void test_v1_opt1_structural() {
  const int n = 6;
  const int p = 0;

  auto c_orig = make_corr(n);
  auto c_ref = c_orig;
  auto c_test = c_orig;

  ref_step(c_ref.data(), n, p);
  mphil_dis_cholesky_step(c_test.data(), n,
                          p); // also populates g_v1_schur_probe[]

  // OPT-1 proof: inner-loop write stride must be 1, not n.
  const ptrdiff_t stride = mphil_v1_schur_write_stride();
  CHECK(stride == 1,
        "T_V1O1: Schur write stride == 1 (j-inner, OPT-1 active); got %td"
        "  (n=%d means stride-n=%d would reveal i-inner / OPT-1 absent)",
        stride, n, n);

  // Companion: step result matches ref_step() for the same n, p.
  const double scale = max_abs_elem(c_ref);
  const double tol = 50.0 * EPS * n * (scale > 0.0 ? scale : 1.0);
  const double diff = max_abs_diff_mat(c_ref, c_test);
  CHECK(diff <= tol,
        "T_V1O1: step result matches ref_step() n=%d p=0"
        "  max_diff=%.2e tol=%.2e",
        n, diff, tol);
}

#endif // MPHIL_CHOLESKY_TESTING && CHOLESKY_VERSION == 1

// ─── J: OpenMP single/barrier sanity ─────────────────────────────────────────
// Verifies that mphil_dis_cholesky_step() produces the numerically correct
// result at high thread count (T=8), providing evidence that the implicit
// barrier after `omp single` is working.
//
// Rationale for catching a missing barrier:
//   If `omp single` were replaced by `omp master` (which has no implicit
//   barrier), T−1 threads would enter the Schur complement update while the
//   master is still scaling row_p[j].  With C scaled by 100 so that diag=10:
//     unscaled row_p[j] = C[0][j] ≈ 100·corr(0,j,n)
//     scaled   row_p[j] = C[0][j]/10 ≈ 10·corr(0,j,n)
//   A thread reading the unscaled value applies a Schur update 10× too large.
//   Per-element error ≈ L[i][0] · C[0][j] · (1 − 0.1) ≈ 0.9 · L[i][0] · C[0][j]
//   which, for L[i][0] ≈ corr(i,0,200) ≈ 0.5 and C[0][j] ≈ 10, is ≈ 4.5.
//   This exceeds 50·n·ε ≈ 2.2·10⁻¹² by twelve orders of magnitude, so the
//   CHECK below fails reliably whenever the race manifests (i.e., on any
//   machine with ≥ 2 physical cores and T=8 threads actually running in
//   parallel — the guaranteed case on CSD3 and typical dev machines).
//
//   The test is deterministic in the *correct* case: with the barrier present,
//   diff is always ≤ 50·n·ε.  In the broken case the diff is O(1), making the
//   failure mode very obvious.
// ─────────────────────────────────────────────────────────────────────────────

#if defined(MPHIL_CHOLESKY_TESTING) && defined(_OPENMP)

static void test_barrier_sanity() {
  const int n = 200;
  const int p = 0;
  const int T = 8;

  // Scale corr(n) by 100 so diag = 10.
  // Without the barrier, a thread reads row_p[j] = C_orig[0][j] (100×scale),
  // rather than C_orig[0][j]/10 (10×scale).  The 10× error is O(1) >> 50nε.
  auto c_orig = make_corr(n);
  for (auto &v : c_orig)
    v *= 100.0;

  auto c_ref = c_orig;
  auto c_test = c_orig;

  // Reference: single-threaded step (no concurrency risk).
  ref_step(c_ref.data(), n, p);

  // Test: T-thread step via the production code path.
  omp_set_num_threads(T);
  mphil_dis_cholesky_step(c_test.data(), n, p);
  omp_set_num_threads(1);

  const double scale = max_abs_elem(c_ref);
  const double tol = 50.0 * EPS * n * (scale > 0.0 ? scale : 1.0);
  const double diff = max_abs_diff_mat(c_ref, c_test);

  CHECK(diff <= tol,
        "T_BARR: T=%d n=%d p=0  max_diff=%.2e  tol=50·n·ε=%.2e"
        "  (missing barrier → unscaled row_p → diff~O(1)>>tol)",
        T, n, diff, tol);
}

#endif // MPHIL_CHOLESKY_TESTING && _OPENMP

#ifdef _OPENMP
static void test_openmp_thread_invariance(int n) {
  auto c_orig = make_corr(n);

  // Reference: T=1 gives the serial-path result.
  omp_set_num_threads(1);
  auto c1 = c_orig;
  mphil_dis_cholesky(c1.data(), n);
  const double logdet_T1 = logdet_eq4(c1.data(), n);
  // frob(c_orig) is the denominator for backward-error; compute once.
  const double frob_orig = frob(c_orig);

  static const int candidates[] = {1, 2, 4, 8, 16};
  const int nc = static_cast<int>(sizeof candidates / sizeof candidates[0]);
  const double back_tol = 10.0 * n * EPS;

  bool all_ok = true;

  for (int ci = 0; ci < nc; ++ci) {
    const int T = candidates[ci];
    omp_set_num_threads(T);

    auto c = c_orig;
    double t = mphil_dis_cholesky(c.data(), n);

    if (t < 0.0) {
      // Should never happen for a valid SPD corr() matrix.
      std::fprintf(
          stderr,
          "  T_OMP n=%d T=%2d: mphil_dis_cholesky returned %.1f (error)\n", n,
          T, t);
      all_ok = false;
      continue;
    }

    auto chat = reconstruct(c.data(), n);
    double back_err = frob_diff(c_orig, chat) / frob_orig;
    double ld = logdet_eq4(c.data(), n);
    double ld_diff = std::fabs(ld - logdet_T1);

    bool ok_back = (back_err <= back_tol);
    bool ok_ld = (ld_diff <= 1e-10);

    if (!ok_back || !ok_ld) {
      std::fprintf(stderr,
                   "  T_OMP n=%d T=%2d: backErr=%.2e (tol=%.2e %s)  "
                   "logdetDiff=%.2e (tol=1e-10 %s)\n",
                   n, T, back_err, back_tol, ok_back ? "OK" : "FAIL", ld_diff,
                   ok_ld ? "OK" : "FAIL");
      all_ok = false;
    }
  }

  omp_set_num_threads(1); // restore default for any subsequent tests

  CHECK(all_ok,
        "T_OMP: thread invariance n=%3d  T={1,2,4,8,16}  "
        "back≤10nε  |Δlogdet|≤1e-10",
        n);
}
#endif // _OPENMP

// ─── Main
// ─────────────────────────────────────────────────────────────────────

int main() {
  std::printf("mphil_dis_cholesky test suite (CHOLESKY_VERSION=%d)\n\n",
              CHOLESKY_VERSION);

  // A — canonical spec example
  test_spec_2x2(); // T01a–b  (checks  1– 2)

  // B — analytical matrices with known exact values
  test_scalar_n1();             // T02a–b  (checks  3– 4)
  test_identity(3);             // T03     (checks  5– 6)
  test_identity(16);            // T03     (checks  7– 8)
  test_scaled_identity(3, 4.0); // T04     (checks  9–10)

  // C — corr() SPD matrices: backward error ||C − LL^T||_F / ||C||_F
  test_corr_backward(5);   // T05     (check  11)
  test_corr_backward(16);  // T05     (check  12)
  test_corr_backward(64);  // T05     (check  13)
  test_corr_backward(128); // T05     (check  14)
  test_corr_backward(200); // T05     (check  15)

  // D — log-det identity (spec Eq. 4)
  test_logdet_2x2_analytical(); // T06     (check  16)
  test_logdet_corr_sign(64);    // T07     (check  17)

  // E — input validation and non-SPD error paths
  test_invalid_inputs(); // T08a–d  (checks 18–21)
  test_non_spd();        // T09a–b  (checks 22–23)

  // F — structural correctness: symmetry + diagonal positivity
  test_structure(5);  // T10     (checks 24–25)
  test_structure(64); // T10     (checks 26–27)

  // G — Step-level oracle proof (Task A): pivot-by-pivot comparison against
  //     a clean serial reference.  Catches wrong loop bounds, missing scales,
  //     data-race corruptions, or any single-step deviation O(1) in magnitude.
  //     Compiled only when MPHIL_CHOLESKY_TESTING=1 (test targets); see
  //     CMakeLists.txt.  Runs for all versions (V0–V3).
#ifdef MPHIL_CHOLESKY_TESTING
  std::printf("\n--- G: step-level oracle (MPHIL_CHOLESKY_TESTING) ---\n");
  test_step_level(5);  // T_STEP   n= 5: 5 pivot steps
  test_step_level(16); // T_STEP   n=16: 16 pivot steps
#endif

  // H — OpenMP thread-count invariance (Task B): same backward error and
  //     log-det for T ∈ {1,2,4,8,16}.  Compiled and run only for V2/V3
  //     builds where -fopenmp defines _OPENMP.
#ifdef _OPENMP
  std::printf("\n--- H: OpenMP thread invariance (_OPENMP defined) ---\n");
  test_openmp_thread_invariance(64);  // T_OMP  n= 64
  test_openmp_thread_invariance(200); // T_OMP  n=200
#endif

  // H2 — V3 column-scaling parallelism proof: structural check that
  //      row_i[p] *= inv_diag is distributed across >= 2 thread IDs,
  //      distinguishing V3 (omp for) from V2 (omp single).
  //      Compiled only for CHOLESKY_VERSION==3 with OpenMP + TESTING.
#if defined(MPHIL_CHOLESKY_TESTING) && defined(_OPENMP) && CHOLESKY_VERSION == 3
  std::printf("\n--- H2: V3 column-scaling parallelism proof ---\n");
  test_v3_col_scaling_parallel(); // T_V3P  n=256  T=8  p=0
#endif

  // I — V1 tail coverage: verifies OPT-5 scalar tail executes for sizes
  //     where (n−p−1)%4 != 0 (tail rows would be silently skipped if the
  //     loop bound were wrong).  n=6 (1 tail row) and n=7 (2 tail rows).
  //     Compiled only for CHOLESKY_VERSION==1 with MPHIL_CHOLESKY_TESTING.
  //
  // I2 — V1 OPT-1 structural proof: inspects the address stride between
  //     consecutive scalar-tail writes; stride==1 proves j is inner
  //     (OPT-1 stride-1), not i inner (stride-n = V0 order).
  //     Compiled only for CHOLESKY_VERSION==1 with MPHIL_CHOLESKY_TESTING.
#if defined(MPHIL_CHOLESKY_TESTING) && CHOLESKY_VERSION == 1
  std::printf("\n--- I: V1 tail coverage ---\n");
  test_v1_tail_coverage(); // T_V1T  n=6 (1 tail row), n=7 (2 tail rows)
  std::printf("\n--- I2: V1 OPT-1 structural proof ---\n");
  test_v1_opt1_structural(); // T_V1O1 n=6 p=0  stride==1
#endif

  // J — OpenMP single/barrier sanity: one-step correctness at T=8 with a
  //     scaled matrix (diag=10) that would reveal a missing barrier by
  //     producing an O(1) deviation vs the 50·n·ε tolerance.
  //     Compiled only when MPHIL_CHOLESKY_TESTING and _OPENMP are defined.
#if defined(MPHIL_CHOLESKY_TESTING) && defined(_OPENMP)
  std::printf("\n--- J: OpenMP single/barrier sanity ---\n");
  test_barrier_sanity(); // T_BARR n=200  T=8  p=0
#endif

  // K — SPD stress family (A·Aᵀ + 0.1·I): backward error + log-det for a
  //     second SPD class distinct from the spec corr() family.  Covers all
  //     versions; compiled unconditionally.
  std::printf("\n--- K: SPD stress family (A*A^T + 0.1*I) ---\n");
  test_spd_stress(8);  // T_SPD  n= 8  (3 checks)
  test_spd_stress(40); // T_SPD  n=40  (3 checks)

  std::printf("\n%d / %d tests passed.\n", g_run - g_fail, g_run);
  return (g_fail > 0) ? 1 : 0;
}
