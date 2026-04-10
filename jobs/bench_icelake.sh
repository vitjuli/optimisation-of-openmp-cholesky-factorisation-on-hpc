#!/bin/bash
# =============================================================================
# bench_icelake.sh - SLURM benchmark job for CSD3 icelake partition
#
# Partition : icelake  (Intel Xeon Platinum 8360Y, 76 cores)
# Builds    : CHOLESKY_VERSION=2 with -march=icelake-server -O3
# Sweeps    : matrix sizes {500,1000,2000,4000,8000} x
#             thread counts {1,2,4,8,16,32,48,64,76}
#
# Output:
#   SLURM log  : bench_icelake_<JOBID>.log  (build output, env, diagnostics)
#   CSV data   : results/bench_icelake_<JOBID>_<TIMESTAMP>.csv
#
# Override sweep parameters by setting environment variables before calling sbatch:
#   BENCH_SIZES="1000,2000" BENCH_THREADS="1,4,8,16,32,64,76" sbatch jobs/bench_icelake.sh
# =============================================================================

#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=cholesky_icelake
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cholesky_icelake-%j.out
#SBATCH --error=logs/cholesky_icelake-%j.err

set -euo pipefail

# ── Reproducibility header ────────────────────────────────────────────────────
echo "================================================================="
echo " cholesky_benchmark  -  icelake partition"
echo " SLURM_JOB_ID      = ${SLURM_JOB_ID}"
echo " SLURM_JOB_NODELIST= ${SLURM_JOB_NODELIST}"
echo " Date/time         = $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================="
hostname
echo ""

# ── Module setup ──────────────────────────────────────────────────────────────
# Load GCC 11 (provides OpenMP 4.5 support and AVX-512 vectorisation).
# cmake is available in the default CSD3 environment; load it if needed.
module purge
module load gcc/11
module list 2>&1

echo ""
echo "--- Compiler ---"
gcc --version | head -1
g++ --version | head -1
echo ""

# ── CPU information ───────────────────────────────────────────────────────────
# Captured in the log for reproducibility.  Confirms correct partition/ISA.
echo "--- CPU info (lscpu) ---"
lscpu
echo ""

# ── OpenMP environment ────────────────────────────────────────────────────────
# OMP_NUM_THREADS: sets the ceiling used by omp_get_max_threads() so that
#   the benchmark's capping logic correctly limits to the allocated core count.
# OMP_PROC_BIND=close: keep each thread on the same physical core it started on,
#   avoiding costly cross-NUMA migration (two-socket node).
# OMP_PLACES=cores: bind to physical cores (not hardware threads).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-76}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "--- OpenMP environment ---"
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
echo "OMP_PROC_BIND   = ${OMP_PROC_BIND}"
echo "OMP_PLACES      = ${OMP_PLACES}"
echo ""

# ── Navigate to repo root ─────────────────────────────────────────────────────
# sbatch is expected to be submitted from the repository root directory.
# $SLURM_SUBMIT_DIR is set automatically by SLURM.
cd "${SLURM_SUBMIT_DIR}"
echo "Working directory: $(pwd)"
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
# -march=icelake-server: enables AVX-512 and icelake-specific tuning.
#   Preferred over -march=native for CSD3 so binaries are reproducible across
#   all nodes in the same partition (all icelake nodes have the same ISA).
# CHOLESKY_VERSION=2: OpenMP parallel implementation.
# Release build: -O3 (applied by CMakeLists.txt CHOLESKY_COMPILE_OPTIONS).
echo "--- CMake configure ---"
cmake -S . -B build \
      -DCHOLESKY_VERSION=2 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=icelake-server"

echo ""
echo "--- CMake build ---"
cmake --build build --clean-first -j"${SLURM_CPUS_PER_TASK:-4}"
echo ""

# ── Correctness check before benchmarking ────────────────────────────────────
# Run with 1 thread (matches the test suite env setting) to confirm correctness
# before spending time on the full benchmark sweep.
echo "--- Correctness tests (OMP_NUM_THREADS=1) ---"
OMP_NUM_THREADS=1 ctest --test-dir build --output-on-failure
echo ""

# ── Output file setup ─────────────────────────────────────────────────────────
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTDIR="${SLURM_SUBMIT_DIR}/results"
mkdir -p "${OUTDIR}"
CSV_FILE="${OUTDIR}/bench_icelake_${SLURM_JOB_ID}_${TIMESTAMP}.csv"

echo "--- Benchmark CSV will be written to ---"
echo "    ${CSV_FILE}"
echo ""

# ── Sweep parameters ──────────────────────────────────────────────────────────
# Override by setting BENCH_SIZES / BENCH_THREADS in the environment before
# calling sbatch:
#   BENCH_SIZES="1000,2000,4000" sbatch jobs/bench_icelake.sh
BENCH_SIZES="${BENCH_SIZES:-500,1000,2000,4000,8000}"
BENCH_THREADS="${BENCH_THREADS:-1,2,4,8,16,32,48,64,76}"
BENCH_REPEATS="${BENCH_REPEATS:-3}"

echo "--- Benchmark parameters ---"
echo "BENCH_SIZES   = ${BENCH_SIZES}"
echo "BENCH_THREADS = ${BENCH_THREADS}"
echo "BENCH_REPEATS = ${BENCH_REPEATS}"
echo ""

# ── Run the benchmark ─────────────────────────────────────────────────────────
# stdout (CSV) is redirected to the results file.
# stderr (metadata notes, any capping warnings) goes to the SLURM log.
echo "--- Starting benchmark sweep ---"
date '+%Y-%m-%d %H:%M:%S'

./build/bin/cholesky_benchmark \
    --sizes   "${BENCH_SIZES}"   \
    --threads "${BENCH_THREADS}" \
    --repeats "${BENCH_REPEATS}" \
    > "${CSV_FILE}"

echo ""
echo "--- Benchmark complete ---"
date '+%Y-%m-%d %H:%M:%S'
echo "CSV written to: ${CSV_FILE}"
echo "Rows in output: $(wc -l < "${CSV_FILE}")"
echo ""

# ── Quick summary of median GFLOP/s from CSV ──────────────────────────────────
echo "--- Median rows (run=-1) from CSV ---"
grep -v '^#' "${CSV_FILE}" | awk -F',' '$4 == -1 {printf "  n=%-5s  threads=%-3s  gflops=%s\n", $2, $3, $6}'
echo ""

echo "================================================================="
echo " Job finished successfully."
echo "================================================================="
