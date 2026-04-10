/**
 * @file mphil_dis_cholesky.cpp
 * @brief Optimisation of OpenMP Cholesky Factorisation on CSD3
 *
 * This file implements an in-place Cholesky factorisation for symmetric
 * positive definite (SPD) matrices stored in row-major order. Four variants are
 * provided and selected at compile time via -DCHOLESKY_VERSION=N.
 *
 * @section cho_versions Build-time variants (CHOLESKY_VERSION)
 *
 * @par VERSION 0 — Serial reference (baseline)
 * Direct transcription of the coursework pseudocode; used as a correctness and
 * performance reference.  \n
 * Git tag: v0_serial_correct.
 *
 * @par VERSION 1 — Serial optimised
 * Same algorithm as VERSION 0, with five local optimisations:  \n
 * - OPT-1: Loop interchange in the Schur update (i-outer, j-inner) to obtain
 *   unit-stride traversal in row-major storage.  \n
 * - OPT-2: Hoist p*n and row pointers to reduce repeated index arithmetic.  \n
 * - OPT-3: Replace repeated divisions by one reciprocal per pivot step and
 *   multiply thereafter.  \n
 * - OPT-4: Hoist the invariant pivot-column entry L(i,p) into a scalar for
 *   the inner j-loop.  \n
 * - OPT-5: Unroll the outer i-loop by 4 rows to increase reuse of the pivot row
 *   and expose additional instruction-level parallelism.  \n
 * Git tag: v1_serial_optimised.
 *
 * @par VERSION 2 — OpenMP parallel
 * A persistent OpenMP parallel region encloses the pivot loop to avoid repeated
 * fork/join overhead. For each pivot step:  \n
 * - (a) #pragma omp single performs pivot check, square root, and scaling of
 *   the pivot row and pivot column.  \n
 * - (b) #pragma omp for schedule(static) parallelises the Schur update across
 *   i-rows (each thread writes disjoint rows, so the update is race-free).  \n
 * Timing uses omp_get_wtime().  \n
 * Git tag: v2_openmp_parallel.
 *
 * @par VERSION 3 — OpenMP tuned (parallel column scaling)
 * As VERSION 2, except the pivot-column scaling c[i*n+p] *= inv_diag (for
 * i>p) is moved from omp single into the omp for body. This reduces the
 * serial component associated with stride-n access in row-major storage, while
 * keeping pivot-row scaling serial (the Schur update requires a fully updated
 * pivot row).  \n
 * Git tag: v3_column_scaling.
 *
 * @section cho_layout Storage layout
 * Input and output share the same array c (in-place):  \n
 * - Input: c[i*n + j] = C(i,j) in row-major order.  \n
 * - Output: lower triangle stores L(i,j) for i >= j.  \n
 * - Output: upper triangle stores the mirrored values L(j,i) (i.e. `L^T`).  \n
 *
 * @section cho_return Return value
 * - t >= 0.0: wall-clock seconds for a successful factorisation (SPD input). \n
 * - -1.0: invalid input (`c == nullptr`, n <= 0, or `n > 100000`).  \n
 * - -2.0: matrix is not SPD (a non-positive pivot encountered).  \n
 *
 * @section cho_testing Test instrumentation
 * Test-only hooks (step-level oracle and lightweight probes) live in
 * src/mphil_dis_cholesky_testing.cpp and are compiled only into the
 * cholesky_testing_hooks target, which is linked exclusively to test
 * executables. The production library contains no test code and does not depend
 * on MPHIL_CHOLESKY_TESTING.
 */

#include "mphil_dis_cholesky.h"

#include <chrono> // std::chrono::high_resolution_clock
#include <cmath>  // std::sqrt

// Default to the baseline when the macro is not supplied by the build system.
#ifndef CHOLESKY_VERSION
#define CHOLESKY_VERSION 0
#endif

// <omp.h> is required only for VERSION 2 and 3 (omp_get_wtime() + directives).
// Excluding it for VERSION 0/1 keeps those builds free of any OpenMP
// dependency, so they compile on machines without libomp installed.
#if CHOLESKY_VERSION == 2 || CHOLESKY_VERSION == 3
#include <omp.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
#if CHOLESKY_VERSION == 0
// ─── Version 0: serial baseline ──────────────────────────────────────────────
//
// Direct transcription of the coursework pseudocode.  Intentionally unoptimised
// so that every improvement in v1 has a clear, measurable before/after.
// ─────────────────────────────────────────────────────────────────────────────

double mphil_dis_cholesky(double *c, int n) {
  if (c == nullptr || n <= 0 || n > 100000)
    return -1.0; // invalid input

  const auto t_start = std::chrono::high_resolution_clock::now();

  for (int p = 0; p < n; ++p) {
    // Diagonal pivot: must be positive for SPD matrices.
    const double pivot = c[p * n + p];
    if (pivot <= 0.0)
      return -2.0; // not SPD / numerical breakdown

    const double diag = std::sqrt(pivot);
    c[p * n + p] = diag;

    // Update row p to the right of the diagonal (upper triangle = L^T).
    for (int j = p + 1; j < n; ++j)
      c[p * n + j] /= diag;

    // Update column p below the diagonal (lower triangle = L).
    for (int i = p + 1; i < n; ++i)
      c[i * n + p] /= diag;

    // Trailing submatrix update (Schur complement).
    // Loop order: j outer, i inner — inner access c[i*n+j] is stride-n
    // (column-wise) in row-major storage.  Kept as-is for the baseline.
    for (int j = p + 1; j < n; ++j)
      for (int i = p + 1; i < n; ++i)
        c[i * n + j] -= c[i * n + p] * c[p * n + j];
  }

  const auto t_end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(t_end - t_start).count();
}

// ─────────────────────────────────────────────────────────────────────────────
#elif CHOLESKY_VERSION == 1
// ─── Version 1: serial optimised ─────────────────────────────────────────────
//
// Mathematically identical to v0.  Five optimisations are applied and
// annotated below; each targets a distinct bottleneck visible in v0.
// ─────────────────────────────────────────────────────────────────────────────

double mphil_dis_cholesky(double *c, int n) {
  if (c == nullptr || n <= 0 || n > 100000)
    return -1.0; // invalid input

  const auto t_start = std::chrono::high_resolution_clock::now();

  for (int p = 0; p < n; ++p) {
    // OPT-2: hoist p*n once per pivot column — eliminates one integer
    // multiply per element in the row update, column update, and Schur
    // complement inner loop.
    const int pn = p * n;
    double *const row_p = c + pn; // pointer to start of row p

    // Diagonal pivot: must be positive for SPD matrices.
    const double pivot = row_p[p];
    if (pivot <= 0.0)
      return -2.0; // not SPD / numerical breakdown

    const double diag = std::sqrt(pivot);
    row_p[p] = diag;

    // OPT-3: compute the reciprocal once; replace n-p-1 divisions in the
    // row update and n-p-1 divisions in the column update with multiplies.
    // Division is ~4-10x slower than multiplication on modern FP units.
    const double inv_diag = 1.0 / diag;

    // Update row p (upper triangle = L^T entries).
    // OPT-3: *= inv_diag instead of /= diag.
    // OPT-2: row_p pointer avoids recomputing p*n each iteration.
    for (int j = p + 1; j < n; ++j)
      row_p[j] *= inv_diag;

    // Update column p (lower triangle = L entries).
    // OPT-3: *= inv_diag instead of /= diag.
    for (int i = p + 1; i < n; ++i)
      c[i * n + p] *= inv_diag;

    // Trailing submatrix update (Schur complement):
    //   C[i][j] -= L[i][p] * L^T[p][j]   for all i,j > p.
    //
    // OPT-1: i-outer / j-inner loop order -> inner accesses stride-1.
    // OPT-2: row pointers hoisted -> no i*n multiply per j iteration.
    // OPT-4: Lip scalar hoist -> L[i][p] read once per outer-i, not per j.
    //
    // OPT-5: 4-wide i-loop unrolling.
    //   Naive: for each i, load row_p[j] once and update 1 row element.
    //   Unrolled: load row_p[j] once and update 4 row elements.
    //   This quadruples arithmetic intensity (4 FMAs per load of row_p[j])
    //   and exposes ILP: the 4 subtract-multiply operations are independent
    //   so the out-of-order core can issue them concurrently across its
    //   multiple FP execution units.
    //   Row pointer arithmetic uses pointer addition (ri1 = ri0 + n) rather
    //   than index multiplication — consistent with OPT-2.
    {
      const int i_end4 =
          p + 1 +
          ((n - p - 1) / 4) * 4; // largest i divisible into 4-wide blocks
      int i = p + 1;

      // Main 4-wide loop.
      for (; i < i_end4; i += 4) {
        double *const ri0 = c + i * n; // OPT-2
        double *const ri1 = ri0 + n;
        double *const ri2 = ri1 + n;
        double *const ri3 = ri2 + n;
        const double L0 = ri0[p]; // OPT-4
        const double L1 = ri1[p];
        const double L2 = ri2[p];
        const double L3 = ri3[p];
        // OPT-1 (stride-1): j-inner order makes all four row accesses
        // contiguous. ri0[j]..ri3[j] == c[i*n+j]..(c[(i+3)*n+j]): each advances
        // by sizeof(double) per j step.  row_p[j] == c[p*n+j]: also stride-1.
        // Both source and destination are read/written sequentially in memory,
        // enabling hardware prefetch and full-width SIMD loads/stores.
        for (int j = p + 1; j < n; ++j) {
          const double rpj = row_p[j]; // loaded once, used 4x (OPT-5)
          ri0[j] -= L0 * rpj;
          ri1[j] -= L1 * rpj;
          ri2[j] -= L2 * rpj;
          ri3[j] -= L3 * rpj;
        }
      }

      // Scalar tail: remaining 0–3 rows.
      for (; i < n; ++i) {
        double *const row_i = c + i * n; // OPT-2
        const double Lip = row_i[p];     // OPT-4
        // OPT-1 (stride-1): row_i[j] == c[i*n+j] and row_p[j] == c[p*n+j]
        // both advance by sizeof(double) per j step — stride-1 in row-major
        // layout.  Contrast with V0's j-outer loop where the inner access
        // c[i*n+j] stepped by n*sizeof(double), causing one cache miss per
        // element at large n.
        for (int j = p + 1; j < n; ++j)
          row_i[j] -= Lip * row_p[j];
      }
    }
  }

  const auto t_end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(t_end - t_start).count();
}

// ─────────────────────────────────────────────────────────────────────────────
#elif CHOLESKY_VERSION == 2
// ─── Version 2: OpenMP parallel ──────────────────────────────────────────────
//
// Same outer-product algorithm as v0/v1.  A single persistent parallel region
// wraps the entire p loop, so the thread team is created once and reused for
// every pivot step to avoid O(n) fork/join events.
//
// Per-step structure:
//   (a) omp single  — pivot check, sqrt, 1/diag, row/column scaling  [O(n)]
//   (b) omp for     — trailing Schur complement update                [O(n^2)]
//
// Implicit barriers preserve the dependency chain:
//   - Barrier after omp single  -> all threads see updated row_p before (b).
//   - Barrier after omp for     -> Schur update for step p is complete before
//                                  the next omp single reads c[(p+1)*n+(p+1)].
//
// schedule(static): each i-row costs exactly (n-p-1) FMA operations — uniform
// work per iteration -> static chunk assignment has optimal load balance with
// zero dynamic-scheduling overhead.
//
// Race-freedom: the omp for distributes distinct i-rows across threads; no two
// threads write the same element c[i*n+j].  row_p is read-only in the Schur
// loop (all threads read the same values safely without synchronisation).
//
// (git tag v2_openmp_parallel)
// ─────────────────────────────────────────────────────────────────────────────

double mphil_dis_cholesky(double *c, int n) {
  if (c == nullptr || n <= 0 || n > 100000)
    return -1.0; // invalid input

  // Shared error flag: 0 = ok, -2 = non-SPD pivot encountered.
  // Written only inside omp single (one thread at a time) -> no data race.
  int err = 0;

  // omp_get_wtime(): portable OpenMP wall-clock timer (seconds).
  // Preferred over std::chrono in OpenMP builds, same API already on the
  // thread team, consistent with the parallel timing model.
  // Measured just before and after the parallel region so all factorisation
  // work, including the initial thread-team synchronisation, is captured.
  const double t_start = omp_get_wtime();

// ── Persistent parallel region ────────────────────────────────────────────
// #pragma omp parallel: forks the thread team once here.
// Rationale: creating/destroying threads at every pivot step would add O(n)
// fork/join overheads; one region amortises that cost over all n steps
//
// default(none): forces every variable used inside to be explicitly scoped —
// best practice to prevent accidental sharing bugs.
//
// shared(c, n, err):
//   c   — the matrix array; all threads read/write distinct rows.
//   n   — matrix dimension; read-only by all threads.
//   err — error flag; written by single, read by all threads.
//
// All other variables (p, row_p, diag, inv_diag, row_i, Lip, ri, rp_local,
// i, j, pivot) are declared inside the region and are therefore automatically
// thread-private (each thread has its own copy on its stack).
#pragma omp parallel default(none) shared(c, n, err)
  {
    // Each thread executes its own copy of the sequential p loop.
    // Barriers inside (after omp single, after omp for) ensure every thread
    // advances to the same p before either block executes.
    for (int p = 0; p < n; ++p) {

// ── (a) Serial pivot section ──────────────────────────────────────
// #pragma omp single: exactly one thread (whichever arrives first)
// executes the enclosed block; all remaining threads wait at the
// implicit barrier at the closing brace.
//
// Why single?
//   (i)  The pivot check and std::sqrt must execute once, running
//        them on every thread would redundantly update c[p*n+p] n_t
//        times (wrong) and cause write-write races.
//   (ii) The implicit barrier guarantees that row_p (the pivot row)
//        and c[i*n+p] (the pivot column) are fully scaled before any
//        thread enters the Schur complement loop below.
#pragma omp single
      {
        const double pivot = c[p * n + p];

        if (pivot <= 0.0) {
          // Non-positive pivot: matrix is not SPD (or numerical breakdown).
          // Set the shared flag so every thread exits the p loop cleanly.
          err = -2;
        } else {
          const double diag = std::sqrt(pivot);
          // OPT-3: one reciprocal per pivot column; row/column updates
          //        use multiply instead of divide (~4-10x faster on FP units).
          const double inv_diag = 1.0 / diag;
          // OPT-2: hoist p*n once; row_p pointer used for both row
          //        update and as the read source in the Schur loop.
          double *const row_p = c + p * n;

          row_p[p] = diag;

          // Scale row p right of the diagonal (upper triangle = L^T).
          for (int j = p + 1; j < n; ++j)
            row_p[j] *= inv_diag; // OPT-3

          // Scale column p below the diagonal (lower triangle = L).
          for (int i = p + 1; i < n; ++i)
            c[i * n + p] *= inv_diag; // OPT-3
        }
      }
      // Implicit barrier: all threads have observed the single block's
      // writes to c[p*n+j] and c[i*n+p] before the Schur update begins.

      // If a non-SPD pivot was flagged, all threads exit together.
      // Checking after the barrier (end of single) ensures no thread reads
      // a stale err written by a concurrent single in a hypothetical future
      // p — safe because only the current p's single wrote to err.
      if (err)
        break;

      // OPT-2: hoist the pivot-row pointer out of the omp for body.
      // c + p*n is invariant over all i-iterations for this p step;
      // computing it once here avoids one integer multiply per i-row.
      // Declared const: the Schur complement only reads row p, never writes it.
      // All threads in the team use this pointer concurrently — safe because
      // it is read-only and points into already-updated memory (written by
      // the omp single block above, visible to all threads after its barrier).
      const double *const row_p = c + p * n;

// ── (b) Parallel Schur complement: C[i][j] -= L[i][p] * L^T[p][j] ─
//
// #pragma omp for: distributes the i-loop iterations across the
// thread team for this p step.
//
// Correctness:
//   • Each thread owns a contiguous range of i-rows.
//   • No two threads write to the same c[i*n+j] (distinct rows).
//   • row_p points to row p (read-only); all threads read it safely
//     without further synchronisation (no writes to row p here).
//
// schedule(static): assigns equal-sized contiguous chunks of i-rows
// to each thread at compile time (no runtime queue).  Justified
// because each row i costs exactly (n-p-1) FMA ops — the work is
// perfectly uniform, so static gives optimal balance with zero
// overhead compared to dynamic.
//
// Implicit barrier at the end of omp for: guarantees that every
// thread has completed its share of the Schur update before the next
// iteration's omp single reads c[(p+1)*n+(p+1)] as the new pivot.
#pragma omp for schedule(static)
      for (int i = p + 1; i < n; ++i) {
        double *const row_i = c + i * n; // OPT-2: hoist i*n multiply
        const double Lip = row_i[p];     // OPT-4: L[i][p] per i

        // __restrict__-qualified local aliases assert to the compiler
        // that ri and rp_local point to non-overlapping memory regions:
        //   ri       — writable alias for row i  (c + i*n, i > p)
        //   rp_local — read-only alias for row p (c + p*n, p < i)
        // These are provably disjoint rows of the same flat array, but
        // the compiler cannot deduce that from raw pointer arithmetic
        // alone.  __restrict__ supplies the proof and, combined with
        // #pragma omp simd below, lets the compiler emit fully vectorised
        // code without inserting a runtime alias-check or scalar fallback.
        // __restrict__ is a supported extension on both GCC and Apple Clang.
        double *__restrict__ ri = row_i;
        const double *__restrict__ rp_local = row_p;

// #pragma omp simd: instructs the compiler to emit SIMD (vector)
// instructions for the j-loop.
//
// Why safe here?
//   (1) No aliasing: ri (row i) and rp_local (row p, p < i) are
//       different rows of c[]; the __restrict__ qualifiers above
//       make this explicit, removing the need for versioned loops.
//   (2) No loop-carried dependency: ri[j] depends only on
//       ri[j] and rp_local[j] at the same j — iterations are
//       fully independent.
//   (3) Lip is loop-invariant — broadcast once as a scalar.
//
// Why not rely on auto-vectorisation with -O3 alone?
//   -O3 auto-vectorises only when it can prove safety, for
//   pointer-derived arrays it often emits a versioned loop or
//   scalar fallback.  #pragma omp simd is the standard OpenMP
//   mechanism to assert vectorisability explicitly, giving the
//   compiler permission to use the widest available SIMD unit:
//   AVX2 (4 doubles/iteration) on this machine, or AVX-512
//   (8 doubles/iteration) on CSD3 icelake (-march=icelake-server).
//
// Hardware impact (innermost hot path — O(n^3) total work):
//   Every doubling of SIMD width halves the time spent here,
//   compounding across all n^2 (i,j) pairs.
#pragma omp simd
        for (int j = p + 1; j < n; ++j)
          ri[j] -= Lip * rp_local[j]; // OPT-1: stride-1 (row-major)
      }
      // Implicit barrier after omp for — next p cannot begin until here.
    }
  }
  // End of persistent parallel region; thread team joins back to master.

  const double t_end =
      omp_get_wtime(); // wall-clock end: factorisation complete

  if (err)
    return -2.0;
  return t_end - t_start;
}

// ─────────────────────────────────────────────────────────────────────────────
#elif CHOLESKY_VERSION == 3
// ─── Version 3: OpenMP parallel — column scaling parallelised ────────────────
//
// Mathematically identical to VERSION 2.  One targeted change:
//   c[i*n+p] *= inv_diag  (column scaling, O(n) per step, O(n^2) total)
//   is moved OUT of the serial `omp single` block INTO the `omp for` body.
//
// Motivation:
//   In VERSION 2, omp single executes both the O(1) pivot/sqrt and the O(n)
//   column-scaling loop serially.  This O(n) serial work is bounded by
//   Amdahl's law: at small n (e.g. n=500) where the Schur complement per step
//   is small, the serial fraction is relatively large and caps speedup.
//   Parallelising column scaling reduces the serial fraction by moving O(n^2)
//   work (summed across all n steps) from serial to the thread team.
//
// Safety argument:
//   (i)  Each thread owns distinct i-rows; no two threads write the same
//        c[i*n+p] element, so no write/write race.
//   (ii) c[i*n+p] is only read within the same iteration, after it
//        has already been scaled, so ordering within a single thread is
//        preserved.
//   (iii) inv_diag is written in omp single (with its implicit barrier before
//        omp for), so all threads see the correct value before the omp for
//        body.
//
// inv_diag must be declared OUTSIDE the parallel region and added to shared()
// so that omp single can write it and omp for can read it.
//
// (git tag v3_column_scaling)
// ─────────────────────────────────────────────────────────────────────────────

double mphil_dis_cholesky(double *c, int n) {
  if (c == nullptr || n <= 0 || n > 100000)
    return -1.0;

  int err = 0;
  // inv_diag: declared outside the parallel region so it can be listed in
  // shared().  Written by omp single once per pivot step; read by omp for
  // in the same step after the implicit barrier that follows omp single.
  double inv_diag = 0.0;

  const double t_start = omp_get_wtime();

// Persistent parallel region: same rationale as VERSION 2.
// inv_diag added to shared() — required for the column-scaling change.
#pragma omp parallel default(none) shared(c, n, err, inv_diag)
  {
    for (int p = 0; p < n; ++p) {

// omp single: pivot check, sqrt, row scaling only.
// Column scaling (c[i*n+p] *= inv_diag) is removed from here
// and moved to the omp for body below.
// inv_diag is written to shared storage so the omp for can read it.
#pragma omp single
      {
        const double pivot = c[p * n + p];
        if (pivot <= 0.0) {
          err = -2;
        } else {
          const double diag = std::sqrt(pivot);
          inv_diag = 1.0 / diag; // write shared: read by omp for
          double *const row_p = c + p * n;
          row_p[p] = diag;
          // Scale row p right of diagonal (upper triangle = L^T).
          // Remains serial — row p must be fully written before
          // any thread reads rp_local[j] in the Schur complement.
          for (int j = p + 1; j < n; ++j)
            row_p[j] *= inv_diag;
          // Column scaling REMOVED from here — see omp for below.
        }
      }
      // Implicit barrier: inv_diag and row p are written and visible.

      if (err)
        break;

      const double *const row_p = c + p * n; // read-only pivot row

// omp for: Schur complement update + column scaling (parallelised).
//
// Each i-iteration now does three things in order:
//   (1) c[i*n+p] *= inv_diag  — column scaling for this row (was in single)
//   (2) Lip = row_i[p]        — read the freshly scaled L[i][p]
//   (3) ri[j] -= Lip * rp[j]  — Schur complement (unchanged from V2)
//
// Step (1) is safe here because:
//   - Each thread owns distinct i-rows, so no conflicts on c[i*n+p].
//   - c[i*n+p] is only read after being scaled (step 2 follows step 1).
//   - inv_diag is read-only after the omp single barrier above.
#pragma omp for schedule(static)
      for (int i = p + 1; i < n; ++i) {
        double *const row_i = c + i * n;

        // (1) Column scaling — parallelised (was serial in VERSION 2).
        // Compute Lip before writing row_i[p] to avoid a redundant store-load.
        // OPT-4: Lip holds L[i][p] once per outer-i iteration.
        const double Lip = row_i[p] * inv_diag;
        row_i[p] = Lip;

        // (2) Schur complement: C[i][j] -= L[i][p] * L^T[p][j]
        double *__restrict__ ri = row_i;
        const double *__restrict__ rp_local = row_p;
#pragma omp simd
        for (int j = p + 1; j < n; ++j)
          ri[j] -= Lip * rp_local[j];
      }
    }
  }

  const double t_end = omp_get_wtime();
  if (err)
    return -2.0;
  return t_end - t_start;
}

// ─────────────────────────────────────────────────────────────────────────────
#else
#error                                                                         \
    "Unknown CHOLESKY_VERSION. Valid values: 0 (baseline), 1 (optimised), 2 (OpenMP), 3 (tuned)."
#endif

// Test-only hooks (mphil_dis_cholesky_step, column-writer recorder, OPT-1
// probe) are in src/mphil_dis_cholesky_testing.cpp, compiled only into
// cholesky_testing_hooks and linked exclusively to test executables.
// This production file contains no test instrumentation.
