#!/bin/bash
# =============================================================================
# bench_cclake.sh - SLURM benchmark job for CSD3 cclake partition
#
# Partition : cclake  (Intel Xeon Gold 6248R, Cascade Lake, 56 cores/node)
# Builds    : CHOLESKY_VERSION=2 with -march=cascadelake -O3
# Sweeps    : matrix sizes {500,1000,2000,4000,8000} 
#             thread counts {1,2,4,8,16,32,48,56}
#
# Output:
#   SLURM log  : bench_cclake_<JOBID>.log  (build output, env, diagnostics)
#   CSV data   : results/bench_cclake_<JOBID>_<TIMESTAMP>.csv
#
# Change sweep parameters:
#   BENCH_SIZES="1000,2000" BENCH_THREADS="1,4,8,28,56" sbatch jobs/bench_cclake.sh
# =============================================================================

#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=cholesky_cclake
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cholesky_cclake-%j.out
#SBATCH --error=logs/cholesky_cclake-%j.err

set -euo pipefail

# ── Reproducibility header ────────────────────────────────────────────────────
echo "================================================================="
echo " cholesky_benchmark  -  cclake partition"
echo " SLURM_JOB_ID      = ${SLURM_JOB_ID}"
echo " SLURM_JOB_NODELIST= ${SLURM_JOB_NODELIST}"
echo " Date/time         = $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================="
hostname
echo ""

# ── Module setup ──────────────────────────────────────────────────────────────
module purge
module load gcc/11
module list 2>&1

echo ""
echo "--- Compiler ---"
gcc --version | head -1
g++ --version | head -1
echo ""

# ── CPU information ───────────────────────────────────────────────────────────
echo "--- CPU info (lscpu) ---"
lscpu
echo ""

# ── OpenMP environment ────────────────────────────────────────────────────────
# Identical rationale to bench_icelake.sh: bind threads close to avoid
# cross-NUMA overhead on the two-socket Cascade Lake node.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-56}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "--- OpenMP environment ---"
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
echo "OMP_PROC_BIND   = ${OMP_PROC_BIND}"
echo "OMP_PLACES      = ${OMP_PLACES}"
echo ""

# ── Navigate to repo root ─────────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR}"
echo "Working directory: $(pwd)"
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
# -march=cascadelake: enables AVX-512 (VNNI) and Cascade Lake-specific tuning.
#   Preferred over -march=native for reproducibility across cclake nodes.
# Note: cclake supports AVX-512 via Intel Cascade Lake microarchitecture.
echo "--- CMake configure ---"
cmake -S . -B build \
      -DCHOLESKY_VERSION=2 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=cascadelake"

echo ""
echo "--- CMake build ---"
cmake --build build --clean-first -j"${SLURM_CPUS_PER_TASK:-4}"
echo ""

# ── Correctness check ─────────────────────────────────────────────────────────
echo "--- Correctness tests (OMP_NUM_THREADS=1) ---"
OMP_NUM_THREADS=1 ctest --test-dir build --output-on-failure
echo ""

# ── Output file setup ─────────────────────────────────────────────────────────
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTDIR="${SLURM_SUBMIT_DIR}/results"
mkdir -p "${OUTDIR}"
CSV_FILE="${OUTDIR}/bench_cclake_${SLURM_JOB_ID}_${TIMESTAMP}.csv"

echo "--- Benchmark CSV will be written to ---"
echo "    ${CSV_FILE}"
echo ""

# ── Sweep parameters ──────────────────────────────────────────────────────────
BENCH_SIZES="${BENCH_SIZES:-500,1000,2000,4000,8000}"
BENCH_THREADS="${BENCH_THREADS:-1,2,4,8,16,32,48,56}"
BENCH_REPEATS="${BENCH_REPEATS:-3}"

echo "--- Benchmark parameters ---"
echo "BENCH_SIZES   = ${BENCH_SIZES}"
echo "BENCH_THREADS = ${BENCH_THREADS}"
echo "BENCH_REPEATS = ${BENCH_REPEATS}"
echo ""

# ── Run the benchmark ─────────────────────────────────────────────────────────
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

# ── Quick summary ─────────────────────────────────────────────────────────────
echo "--- Median rows (run=-1) from CSV ---"
grep -v '^#' "${CSV_FILE}" | awk -F',' '$4 == -1 {printf "  n=%-5s  threads=%-3s  gflops=%s\n", $2, $3, $6}'
echo ""

echo "================================================================="
echo " Job finished successfully."
echo "================================================================="
