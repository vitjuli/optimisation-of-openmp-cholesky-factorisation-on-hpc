/**
 * @file mphil_dis_cholesky_testing.cpp
 *
 * @brief Test only hooks and instrumentation for the Optimisation of OpenMP
 * Cholesky Factorisation on CSD3
 *
 * This translation unit is compiled only into the CMake target
 * cholesky_testing_hooks and is linked exclusively to the test executables
 * cholesky_test and cholesky_test_n2. It is never linked into the
 * production-facing executables (`cholesky_example`, `cholesky_benchmark`).
 *
 * The production library (`cholesky_lib`, implemented in
 * `src/mphil_dis_cholesky.cpp`) contains no test instrumentation and makes no
 * reference to MPHIL_CHOLESKY_TESTING.
 *
 * @section testing_contents Contents
 *
 * @subsection testing_step Step-level reference hook
 * mphil_dis_cholesky_step(c, n, p)
 *
 * Executes exactly one pivot step p of the right-looking (outer-product)
 * Cholesky factorisation on the in-place n×n matrix c[]. The implementation is
 * selected at compile time via CHOLESKY_VERSION, and follows the same code path
 * as the corresponding production version:
 *
 * - V0: baseline Schur update with j outer / i inner loop order (stride-`n`).
 * - V1: i outer / j inner loop order with the serial optimisations (OPT-1..5),
 *         including 4-wide unrolling in the Schur update.
 * - V2: OpenMP version with a persistent parallel region; column scaling
 * performed in omp single.
 * - V3: OpenMP version with a persistent parallel region; column scaling moved
 *         into omp for.
 *
 * @return 0 on success.
 * @return -2 if the pivot is non-positive (`c[p*n + p] <= 0`), indicating a
 * non-SPD input (or numerical breakdown).
 *
 * The caller is responsible for invoking steps p = 0, 1, …, n-1 in order.
 *
 * @subsection testing_v23_tid V2/V3 column-scaling thread-ID recorder
 *
 * During mphil_dis_cholesky_step(), the array g_col_writer_tids[] records the
 * value of omp_get_thread_num() for each write to the scaled column entry c[i*n
 * + p] *= inv_diag. The accessors mphil_dis_last_col_writer_tids() and
 * mphil_dis_last_col_writer_count() expose the recorded IDs. This
 * instrumentation is used by test Group H2 to verify that column scaling is
 * executed in parallel in V3 (as opposed to V2, where it is executed in the omp
 * single region).
 *
 * @subsection testing_v1_stride V1 OPT-1 write-stride probe
 *
 * The array g_v1_schur_probe[2] records the first two addresses written to
 * &row_i[j] inside the scalar-tail `j`-inner loop of the Schur update. The
 * address delta distinguishes the loop order:
 *
 * - delta == 1 confirms contiguous `j`-inner writes (OPT-1 active).
 * - delta == n would indicate a stride-`n` access pattern consistent with the
 * V0 loop order.
 *
 * The accessor mphil_v1_schur_write_stride() returns the measured delta.
 * This probe is used by test Group I2.
 */

#include <cmath>   // std::sqrt
#include <cstddef> // ptrdiff_t

// CHOLESKY_VERSION is supplied by CMake via -DCHOLESKY_VERSION=N on this TU
// (inherited from cholesky_lib PUBLIC compile-definitions).
#ifndef CHOLESKY_VERSION
#define CHOLESKY_VERSION 0
#endif

// omp.h is needed only for V2/V3 (omp_get_thread_num(), pragmas).
#if CHOLESKY_VERSION == 2 || CHOLESKY_VERSION == 3
#include <omp.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// V2 / V3 — column-scaling thread-ID recorder
// ─────────────────────────────────────────────────────────────────────────────
#if CHOLESKY_VERSION == 2 || CHOLESKY_VERSION == 3

// Global arrays are implicitly shared across OpenMP threads (§2.19.1).
// COL_TID_BUF_CAP of 2048 covers n up to ~2001 at step p=0.
static constexpr int COL_TID_BUF_CAP = 2048;
static int g_col_writer_tids[COL_TID_BUF_CAP];
static int g_col_writer_count = 0;

const int *mphil_dis_last_col_writer_tids() { return g_col_writer_tids; }
int mphil_dis_last_col_writer_count() { return g_col_writer_count; }

#endif // V2 || V3

// ─────────────────────────────────────────────────────────────────────────────
// V1 — OPT-1 write-stride probe
// ─────────────────────────────────────────────────────────────────────────────
#if CHOLESKY_VERSION == 1

// Records the addresses of the first two writes in the scalar-tail j-inner
// loop during a mphil_dis_cholesky_step() call.  Reset to 0 at the top of
// each V1 step call.
static const double *g_v1_schur_probe[2] = {nullptr, nullptr};
static int g_v1_schur_probe_count = 0;

ptrdiff_t mphil_v1_schur_write_stride() {
  if (g_v1_schur_probe_count < 2 || !g_v1_schur_probe[0] ||
      !g_v1_schur_probe[1])
    return 0;
  return g_v1_schur_probe[1] - g_v1_schur_probe[0];
}

#endif // V1

// ─────────────────────────────────────────────────────────────────────────────
// mphil_dis_cholesky_step — single-pivot-step oracle
// ─────────────────────────────────────────────────────────────────────────────

int mphil_dis_cholesky_step(double *c, int n, int p) {

// ── V0: j-outer / i-inner (direct pseudocode) ────────────────────────────────
#if CHOLESKY_VERSION == 0

  const double pivot = c[p * n + p];
  if (pivot <= 0.0)
    return -2;
  const double diag = std::sqrt(pivot);
  c[p * n + p] = diag;
  for (int j = p + 1; j < n; ++j)
    c[p * n + j] /= diag;
  for (int i = p + 1; i < n; ++i)
    c[i * n + p] /= diag;
  // Schur: j-outer / i-inner — same stride-n access as the V0 production code.
  for (int j = p + 1; j < n; ++j)
    for (int i = p + 1; i < n; ++i)
      c[i * n + j] -= c[i * n + p] * c[p * n + j];
  return 0;

// ── V1: OPT-1..5 (i-outer/j-inner, 4-wide unrolled) ─────────────────────────
#elif CHOLESKY_VERSION == 1

  // Reset OPT-1 probe so each call captures fresh write addresses.
  g_v1_schur_probe_count = 0;

  double *const row_p = c + p * n;
  const double pivot = row_p[p];
  if (pivot <= 0.0)
    return -2;
  const double diag = std::sqrt(pivot);
  const double inv_diag = 1.0 / diag;
  row_p[p] = diag;
  for (int j = p + 1; j < n; ++j)
    row_p[j] *= inv_diag; // OPT-3 row
  for (int i = p + 1; i < n; ++i)
    c[i * n + p] *= inv_diag; // OPT-3 col

  // Schur: 4-wide i-loop unrolling (OPT-5) with scalar tail.
  const int i_end4 = p + 1 + ((n - p - 1) / 4) * 4;
  int i = p + 1;
  for (; i < i_end4; i += 4) {
    double *const ri0 = c + i * n;
    double *const ri1 = ri0 + n;
    double *const ri2 = ri1 + n;
    double *const ri3 = ri2 + n;
    const double L0 = ri0[p], L1 = ri1[p], L2 = ri2[p], L3 = ri3[p];
    for (int j = p + 1; j < n; ++j) {
      const double rpj = row_p[j];
      ri0[j] -= L0 * rpj;
      ri1[j] -= L1 * rpj;
      ri2[j] -= L2 * rpj;
      ri3[j] -= L3 * rpj;
    }
  }
  // Scalar tail: remaining 0–3 rows.
  for (; i < n; ++i) {
    double *const row_i = c + i * n;
    const double Lip = row_i[p];
    for (int j = p + 1; j < n; ++j) {
      // OPT-1 probe: record first two write addresses (fires at most twice).
      if (g_v1_schur_probe_count < 2)
        g_v1_schur_probe[g_v1_schur_probe_count++] = &row_i[j];
      row_i[j] -= Lip * row_p[j];
    }
  }
  return 0;

// ── V2: persistent omp parallel region (one step, column scaling serial) ─────
#elif CHOLESKY_VERSION == 2

  int err = 0;
  // Reset column-writer recorder before the parallel region.
  g_col_writer_count =
      (n - p - 1 < COL_TID_BUF_CAP) ? (n - p - 1) : COL_TID_BUF_CAP;

#pragma omp parallel default(none) shared(c, n, p, err)
  {
#pragma omp single
    {
      const double pivot = c[p * n + p];
      if (pivot <= 0.0) {
        err = -2;
      } else {
        const double diag = std::sqrt(pivot);
        const double inv_diag = 1.0 / diag;
        double *const row_p = c + p * n;
        row_p[p] = diag;
        for (int j = p + 1; j < n; ++j)
          row_p[j] *= inv_diag;
        // Column scaling: serial (inside omp single).
        // Record tid for each row — proves single-thread ownership (H2).
        const int single_tid = omp_get_thread_num();
        for (int i = p + 1; i < n; ++i) {
          c[i * n + p] *= inv_diag;
          const int idx = i - p - 1;
          if (idx < COL_TID_BUF_CAP)
            g_col_writer_tids[idx] = single_tid; // same tid (serial)
        }
      }
    }
    // Implicit barrier after omp single — row_p and column p fully written.
    if (!err) {
      const double *const row_p = c + p * n;
#pragma omp for schedule(static)
      for (int i = p + 1; i < n; ++i) {
        double *const row_i = c + i * n;
        const double Lip = row_i[p];
        double *__restrict__ ri = row_i;
        const double *__restrict__ rp = row_p;
#pragma omp simd
        for (int j = p + 1; j < n; ++j)
          ri[j] -= Lip * rp[j];
      }
    }
  }
  return err;

// ── V3: persistent omp parallel region (column scaling parallelised)
// ──────────
#elif CHOLESKY_VERSION == 3

  int err = 0;
  double inv_diag = 0.0; // shared: written by single, read by omp for
  // Reset column-writer recorder before the parallel region.
  g_col_writer_count =
      (n - p - 1 < COL_TID_BUF_CAP) ? (n - p - 1) : COL_TID_BUF_CAP;

#pragma omp parallel default(none) shared(c, n, p, err, inv_diag)
  {
#pragma omp single
    {
      const double pivot = c[p * n + p];
      if (pivot <= 0.0) {
        err = -2;
      } else {
        const double diag = std::sqrt(pivot);
        inv_diag = 1.0 / diag;
        double *const row_p = c + p * n;
        row_p[p] = diag;
        for (int j = p + 1; j < n; ++j)
          row_p[j] *= inv_diag;
        // Column scaling removed — parallelised in omp for below.
      }
    }
    // Implicit barrier: inv_diag and row p are written and visible.
    if (!err) {
      const double *const row_p = c + p * n;
// schedule(static,1) interleaves rows 1-by-1 across threads, maximising the
// number of distinct tids recorded in g_col_writer_tids[].  This makes the
// Group H2 assertion (>= 2 distinct tids) deterministic even at small n,
// where a large default chunk size might accidentally assign all rows to
// thread 0.  Production code in mphil_dis_cholesky() keeps schedule(static)
// for optimal load balance; this change affects only the step-oracle path.
#pragma omp for schedule(static, 1)
      for (int i = p + 1; i < n; ++i) {
        double *const row_i = c + i * n;
        // Column scaling (parallelised — the V3 change).
        // Compute Lip first to avoid a redundant store-load on row_i[p].
        const double Lip = row_i[p] * inv_diag;
        row_i[p] = Lip;
        const int idx = i - p - 1;
        if (idx < COL_TID_BUF_CAP)
          g_col_writer_tids[idx] = omp_get_thread_num(); // varies
        double *__restrict__ ri = row_i;
        const double *__restrict__ rp = row_p;
#pragma omp simd
        for (int j = p + 1; j < n; ++j)
          ri[j] -= Lip * rp[j];
      }
    }
  }
  return err;

#else
#error                                                                         \
    "Unknown CHOLESKY_VERSION in mphil_dis_cholesky_testing.cpp (valid: 0..3)"
#endif // per-version dispatch
}
