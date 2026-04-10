/**
 * @file benchmark/benchmark.cpp
 * @brief CSV benchmark harness for @ref mphil_dis_cholesky.
 *
 * This program benchmarks the in-place Cholesky factorisation over a sweep of
 * matrix sizes and (requested) thread counts. For each \f$(n, T)\f$ setting it
 * executes one warm-up call (not recorded) followed by multiple timed repeats,
 * and prints one CSV row per repeat plus a summary row containing the median
 * timing.
 *
 * @section bench_csv CSV output
 * The benchmark writes CSV to stdout with the following columns:
 *
 * version,n,threads,run,time_s,gflops,hostname
 *
 * - version  : compile-time @c CHOLESKY_VERSION (0, 1, 2, or 3)
 * - n        : matrix dimension
 * - threads  : actual thread count used (may be capped)
 * - run      : 0..(repeats-1) for timed repeats; -1 for the median-summary row
 * - time_s   : wall-clock seconds returned by @ref mphil_dis_cholesky
 *                (negative values are library error codes: -1.0 or -2.0)
 * - gflops   : \f$\frac{1}{3}n^3 / (time_s \cdot 10^9)\f$; 0.0 if @c time_s < 0
 * - hostname : result of @c gethostname() (identifies the CSD3 node/partition)
 *
 * The output is preceded by metadata comment lines starting with #. CSV
 * parsers that ignore comment lines can ingest the output directly.
 *
 * @section bench_usage Usage
 * @code
 * ./build/bin/cholesky_benchmark [OPTIONS]
 * @endcode
 *
 * Options:
 * - --sizes   "n1,n2,..."  Matrix sizes to benchmark
 *   (default: 500,1000,2000,4000)
 * - --threads "t1,t2,..."  Thread counts to sweep
 *   (default: 1,2,4,8,16,32,64,76)
 * - --repeats N            Timed repeats per \f$(n,T)\f$ (default: 3)
 * - --no-header            Suppress the CSV header line
 *
 * @section bench_threads Thread-count policy
 * - Versions 2/3 (OpenMP): @c omp_set_num_threads(T) is called before each run;
 *   the request is capped by @c omp_get_max_threads(). The threads column
 *   reports the capped value.
 * - Versions 0/1 (serial): OpenMP is not enabled; the effective thread count is
 *   always 1. Requested thread values are still iterated so the CSV shape is
 *   consistent; repeated rows indicate the expected lack of scaling.
 *
 * @section bench_method Measurement protocol
 * - For each \f$n\f$, the deterministic SPD input @c corr(n) is constructed
 * once as @c C_orig (no RNG).
 * - Before each timed repeat, @c C_orig is copied into a working buffer because
 *   @ref mphil_dis_cholesky overwrites its input in-place.
 * - One un-timed warm-up call precedes each \f$(n,T)\f$ block to reduce
 *   first-iteration overhead (e.g. OpenMP runtime initialisation and cache
 *   cold-start effects).
 * - stdout is flushed after each \f$(n,T)\f$ block so partial results remain
 *   available if a SLURM job is terminated early.
 */

#include "mphil_dis_cholesky.h"

#include <algorithm> // std::sort, std::min, std::copy, std::swap
#include <cmath>     // std::exp
#include <cstdio>    // std::printf, std::fprintf, std::fflush
#include <cstdlib>   // std::strtol
#include <cstring>   // std::strcmp
#include <string>    // std::string, std::to_string
#include <unistd.h>  // gethostname (POSIX; available on Linux and macOS)
#include <vector>    // std::vector

// OpenMP API is only available when the library was built with VERSION 2.
// Guard every omp_* call so that serial builds (VERSION 0/1) compile without
// libomp installed.
#ifdef _OPENMP
#include <omp.h> // omp_set_num_threads, omp_get_max_threads
#endif

// ─────────────────────────────────────────────────────────────────────────────
// SPD matrix generator — verbatim from coursework spec (p.5).
//
//   corr(x, y, s) = 0.99 · exp(−0.5 · 16 · (x−y)^2 / s^2)
//   diagonal      = 1.0   (ensures strict positive definiteness)
//
// Using the same generator as the test suite and example program guarantees
// that benchmark matrices are identical to those used in correctness checks.
// ─────────────────────────────────────────────────────────────────────────────

static double corr(double x, double y, double s) {
  return 0.99 * std::exp(-0.5 * 16.0 * (x - y) * (x - y) / (s * s));
}

static void fill_corr(double *c, int n) {
  const std::size_t sn = static_cast<std::size_t>(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      c[sn * static_cast<std::size_t>(i) + static_cast<std::size_t>(j)] =
          corr(static_cast<double>(i), static_cast<double>(j),
               static_cast<double>(n));
    c[sn * static_cast<std::size_t>(i) + static_cast<std::size_t>(i)] = 1.0;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small utilities
// ─────────────────────────────────────────────────────────────────────────────

static std::string get_hostname() {
  char buf[256];
  buf[0] = '\0';
  if (gethostname(buf, sizeof(buf)) == 0 && buf[0] != '\0')
    return std::string(buf);
  return std::string("unknown");
}

// Parse a comma-separated list of positive integers from a C string.
// Stops at first non-digit, non-comma character.
static std::vector<int> parse_int_list(const char *s) {
  std::vector<int> result;
  const char *p = s;
  while (*p != '\0') {
    char *end;
    long v = std::strtol(p, &end, 10);
    if (end == p)
      break; // no digit parsed — stop
    result.push_back(static_cast<int>(v));
    p = end;
    if (*p == ',')
      ++p; // skip separator
  }
  return result;
}

// GFLOP/s from the same FLOP estimate used in the report: (1/3) n^3.
// Returns 0.0 when time_s <= 0.0 (error codes are negative; time_s can be 0.0
// for very small n if the timer resolution is too coarse)
static double gflops_from(int n, double time_s) {
  if (time_s <= 0.0)
    return 0.0;
  const double flops = (1.0 / 3.0) * static_cast<double>(n) *
                       static_cast<double>(n) * static_cast<double>(n);
  return flops / (time_s * 1.0e9);
}

// Upper-median of a sorted range.
// For repeats=3 this returns the true median (index 1 of 3).
// For even repeats it returns the upper of the two middle elements.
static double sorted_median(std::vector<double> &v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2]; // integer division: e.g. 3/2 = 1
}

// Build a comma-separated string from a vector<int> (for metadata comments).
static std::string int_list_str(const std::vector<int> &v) {
  std::string s;
  for (std::size_t i = 0; i < v.size(); ++i) {
    s += std::to_string(v[i]);
    if (i + 1 < v.size())
      s += ',';
  }
  return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Usage
// ─────────────────────────────────────────────────────────────────────────────

static void print_usage(const char *prog) {
  std::fprintf(
      stderr,
      "Usage: %s [OPTIONS]\n"
      "\n"
      "Options:\n"
      "  --sizes   \"n1,n2,...\"   matrix sizes  (default: "
      "500,1000,2000,4000)\n"
      "  --threads \"t1,t2,...\"   thread counts (default: "
      "1,2,4,8,16,32,64,76)\n"
      "  --repeats N             timed repeats per (n,threads)  (default: 3)\n"
      "  --no-header             omit the CSV header line\n"
      "\n"
      "Output CSV columns:\n"
      "  version,n,threads,run,time_s,gflops,hostname\n"
      "  run = 0..repeats-1 per individual timed run.\n"
      "  run = -1 for the median-summary row.\n"
      "\n"
      "Lines beginning with '#' are metadata comments (not CSV data rows).\n"
      "A negative time_s value indicates a library error (-1.0 or -2.0).\n",
      prog);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {

  // ── Default sweep parameters ──────────────────────────────────────────────
  std::vector<int> sizes = {500, 1000, 2000, 4000};
  std::vector<int> threads = {1, 2, 4, 8, 16, 32, 64, 76};
  int repeats = 3;
  bool print_hdr = true;

  // ── Command-line parsing ──────────────────────────────────────────────────
  for (int a = 1; a < argc; ++a) {
    if (std::strcmp(argv[a], "--sizes") == 0 && a + 1 < argc) {
      sizes = parse_int_list(argv[++a]);
      if (sizes.empty()) {
        std::fprintf(stderr, "Error: --sizes produced an empty list.\n");
        return 1;
      }
    } else if (std::strcmp(argv[a], "--threads") == 0 && a + 1 < argc) {
      threads = parse_int_list(argv[++a]);
      if (threads.empty()) {
        std::fprintf(stderr, "Error: --threads produced an empty list.\n");
        return 1;
      }
    } else if (std::strcmp(argv[a], "--repeats") == 0 && a + 1 < argc) {
      char *end;
      long v = std::strtol(argv[++a], &end, 10);
      if (*end != '\0' || v < 1 || v > 100) {
        std::fprintf(
            stderr,
            "Error: --repeats must be an integer in [1,100], got '%s'.\n",
            argv[a]);
        return 1;
      }
      repeats = static_cast<int>(v);
    } else if (std::strcmp(argv[a], "--no-header") == 0) {
      print_hdr = false;
    } else if (std::strcmp(argv[a], "--help") == 0 ||
               std::strcmp(argv[a], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr,
                   "Error: unknown option '%s'. Use --help for usage.\n",
                   argv[a]);
      return 1;
    }
  }

  // ── Validate sizes ────────────────────────────────────────────────────────
  for (int n : sizes) {
    if (n <= 0 || n > 100000) {
      std::fprintf(
          stderr, "Error: --sizes contains n=%d which is outside [1,100000].\n",
          n);
      return 1;
    }
  }

  // ── Determine the thread ceiling ──────────────────────────────────────────
  // For VERSION 2 (OpenMP): omp_get_max_threads() reflects OMP_NUM_THREADS
  //   or the hardware thread count if the env var is unset.  We cap every
  //   requested thread count at this ceiling.
  // For VERSION 0/1 (no OpenMP): _OPENMP is not defined; ceiling = 1.
#ifdef _OPENMP
  const int max_threads = omp_get_max_threads();
#else
  const int max_threads = 1;
#endif

  // ── Metadata comment block ────────────────────────────────────────────────
  // Written to stdout BEFORE the CSV header so it travels with the data file.
  const std::string hostname = get_hostname();
  const std::string sizes_str = int_list_str(sizes);
  const std::string threads_str = int_list_str(threads);

  std::printf("# cholesky_benchmark  CHOLESKY_VERSION=%d\n", CHOLESKY_VERSION);
  std::printf("# hostname=%s\n", hostname.c_str());
  std::printf("# max_threads=%d\n", max_threads);
  std::printf("# sizes=%s\n", sizes_str.c_str());
  std::printf("# threads=%s\n", threads_str.c_str());
  std::printf("# repeats=%d\n", repeats);

  // ── CSV header ────────────────────────────────────────────────────────────
  if (print_hdr)
    std::printf("version,n,threads,run,time_s,gflops,hostname\n");

  // ── Main sweep: sizes × threads ──────────────────────────────────────────
  for (const int n : sizes) {

    // Allocate both buffers as flat row-major arrays.
    // Use size_t arithmetic to prevent integer overflow for large n.
    const std::size_t n2 =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::vector<double> C_orig(n2);
    std::vector<double> C_work(n2);

    // Build the SPD reference matrix once per n.
    // Deterministic: same corr() as the test suite — no RNG involved.
    fill_corr(C_orig.data(), n);

    for (const int t_req : threads) {

      // Cap the requested thread count at the runtime ceiling.
      // For serial builds (max_threads=1) this is always 1.
      // A stderr warning is emitted when capping occurs so the user
      // knows why threads < requested in the CSV output.
      const int actual_t = std::min(t_req, max_threads);
      if (actual_t < t_req) {
        std::fprintf(stderr,
                     "# Note: n=%d threads=%d capped to %d (max_threads=%d)\n",
                     n, t_req, actual_t, max_threads);
      }

#ifdef _OPENMP
      // Set the thread count for the OpenMP parallel regions inside
      // mphil_dis_cholesky for the upcoming runs on this (n,threads) block.
      omp_set_num_threads(actual_t);
#endif

      // ── Warm-up run (not recorded) ────────────────────────────────────
      // One un-timed factorisation before the measured repeats:
      //   (a) Initialises the OpenMP thread team (avoids first-parallel-
      //       region overhead bleeding into run=0 timing).
      //   (b) Pulls C_orig into the last-level cache, so all three timed
      //       repeats start with a warm cache rather than a cold one.
      // The warm-up result is discarded; errors here are silently ignored
      // because the matrix is known-SPD (corr() is always SPD).
      {
        std::copy(C_orig.cbegin(), C_orig.cend(), C_work.begin());
        (void)mphil_dis_cholesky(C_work.data(), n);
      }

      // ── Timed repeats ─────────────────────────────────────────────────
      std::vector<double> times(static_cast<std::size_t>(repeats));
      bool had_error = false;

      for (int r = 0; r < repeats; ++r) {
        // Restore the original SPD matrix for each repeat so every run
        // starts from identical input (mphil_dis_cholesky is in-place).
        std::copy(C_orig.cbegin(), C_orig.cend(), C_work.begin());

        const double t = mphil_dis_cholesky(C_work.data(), n);

        if (t < 0.0) {
          // Library error (-1.0 = invalid input, -2.0 = not SPD).
          // Emit a CSV row with the error code in time_s and gflops=0,
          // then skip the remaining repeats for this (n,threads) pair.
          // Policy: continue to the next (n,threads) combination rather
          // than aborting, so a single bad matrix size does not stop
          // the entire sweep.
          std::printf("%d,%d,%d,%d,%.9g,%.6g,%s\n", CHOLESKY_VERSION, n,
                      actual_t, r, t, 0.0, hostname.c_str());
          std::fprintf(stderr,
                       "# Error: mphil_dis_cholesky returned %.1f for "
                       "n=%d threads=%d run=%d — skipping block.\n",
                       t, n, actual_t, r);
          had_error = true;
          break;
        }

        times[static_cast<std::size_t>(r)] = t;
        std::printf("%d,%d,%d,%d,%.9g,%.6g,%s\n", CHOLESKY_VERSION, n, actual_t,
                    r, t, gflops_from(n, t), hostname.c_str());
      }

      if (had_error) {
        std::fflush(stdout);
        continue;
      }

      // ── Median-summary row (run = -1) ─────────────────────────────────
      // Compute the median of all timed values.
      // For repeats=3 this is the true (middle) value; for larger repeats
      // it is the upper of the two middle values (upper-median).
      // The median is more robust than the mean against occasional OS
      // scheduling jitter or cache warm-up effects.
      const double t_med = sorted_median(times);
      const double gf_med = gflops_from(n, t_med);
      std::printf("%d,%d,%d,%d,%.9g,%.6g,%s\n", CHOLESKY_VERSION, n, actual_t,
                  -1, t_med, gf_med, hostname.c_str());

      // Flush after every (n,threads) block so partial output is not
      // lost if the SLURM job hits its wall clock time limit.
      std::fflush(stdout);
    }
  }

  return 0;
}
