#!/bin/bash
# =============================================================================
# bench_cclake_numa.sh - NUMA placement experiment on cclake
#
# Runs VERSION 2 twice on n=2000 with different OpenMP thread placement:
#   Run A: OMP_PROC_BIND=close   (fill socket 0 first)
#   Run B: OMP_PROC_BIND=spread  (distribute evenly across sockets)
#
# The cclake n=2000 performance drop observed at T>32 is caused
# by inter-socket (NUMA) traffic when OMP_PROC_BIND=close forces threads onto
# a second socket.  If correct, OMP_PROC_BIND=spread should give better or
# equal performance at T=48,56.
#
# cclake node: Intel Xeon Gold 6248R, 2 x 28 cores per socket.
# NUMA boundary: T > 28 crosses to second socket with OMP_PROC_BIND=close.
#
# Outputs:
#   results/bench_cclake_numa_close_<JOBID>_<TIMESTAMP>.csv
#   results/bench_cclake_numa_spread_<JOBID>_<TIMESTAMP>.csv
#
# Compare median rows: awk -F',' '$4==-1' both CSV files; look at T=32,48,56 columns.
# =============================================================================

#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=cholesky_numa_cclake
#SBATCH --partition=cclake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cholesky_numa_cclake-%j.out
#SBATCH --error=logs/cholesky_numa_cclake-%j.err

set -euo pipefail

echo "================================================================="
echo " NUMA placement experiment  -  cclake partition"
echo " SLURM_JOB_ID      = ${SLURM_JOB_ID}"
echo " SLURM_JOB_NODELIST= ${SLURM_JOB_NODELIST}"
echo " Date/time         = $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " HYPOTHESIS: cclake n=2000 drop at T>32 is NUMA-induced."
echo "================================================================="
hostname
echo ""

module purge
module load gcc/11
module list 2>&1

echo ""
echo "--- CPU info (lscpu) ---"
lscpu
echo ""

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-56}
export OMP_PLACES=cores

cd "${SLURM_SUBMIT_DIR}"
echo "Working directory: $(pwd)"
echo ""

echo "--- CMake configure (VERSION 2) ---"
cmake -S . -B build \
      -DCHOLESKY_VERSION=2 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=cascadelake"

echo ""
echo "--- CMake build ---"
cmake --build build --clean-first -j"${SLURM_CPUS_PER_TASK:-4}"
echo ""

echo "--- Correctness tests (OMP_NUM_THREADS=1) ---"
OMP_NUM_THREADS=1 ctest --test-dir build --output-on-failure
echo ""

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTDIR="${SLURM_SUBMIT_DIR}/results"
mkdir -p "${OUTDIR}"

CLOSE_CSV="${OUTDIR}/bench_cclake_numa_close_${SLURM_JOB_ID}_${TIMESTAMP}.csv"
SPREAD_CSV="${OUTDIR}/bench_cclake_numa_spread_${SLURM_JOB_ID}_${TIMESTAMP}.csv"

# ── Run A: OMP_PROC_BIND=close ─────────────────────────────────────────────
# Threads fill socket 0 first; NUMA boundary crossed at T>28.
echo "--- Run A: OMP_PROC_BIND=close ---"
echo "OMP_PROC_BIND=close, OMP_PLACES=cores"
echo "CSV: ${CLOSE_CSV}"
date '+%Y-%m-%d %H:%M:%S'

OMP_PROC_BIND=close OMP_PLACES=cores \
    ./build/bin/cholesky_benchmark \
        --sizes   "2000" \
        --threads "1,4,8,16,24,28,32,40,48,56" \
        --repeats "5" \
        > "${CLOSE_CSV}"

echo "Done. Rows: $(wc -l < "${CLOSE_CSV}")"
echo ""

# ── Run B: OMP_PROC_BIND=spread ────────────────────────────────────────────
# Threads spread evenly across both sockets from the start.
# With spread, 2 threads = 1 per socket; T=8 = 4 per socket, etc.
echo "--- Run B: OMP_PROC_BIND=spread ---"
echo "OMP_PROC_BIND=spread, OMP_PLACES=cores"
echo "CSV: ${SPREAD_CSV}"
date '+%Y-%m-%d %H:%M:%S'

OMP_PROC_BIND=spread OMP_PLACES=cores \
    ./build/bin/cholesky_benchmark \
        --sizes   "2000" \
        --threads "1,4,8,16,24,28,32,40,48,56" \
        --repeats "5" \
        > "${SPREAD_CSV}"

echo "Done. Rows: $(wc -l < "${SPREAD_CSV}")"
echo ""

# ── Summary comparison ─────────────────────────────────────────────────────
echo "--- Comparison: median GFLOP/s at T=24,28,32,40,48,56 ---"
echo ""
echo "CLOSE binding:"
grep -v '^#' "${CLOSE_CSV}" | awk -F',' '$4 == -1 && ($3==24 || $3==28 || $3==32 || $3==40 || $3==48 || $3==56) {printf "  n=%-5s  threads=%-3s  gflops=%s\n", $2, $3, $6}'
echo ""
echo "SPREAD binding:"
grep -v '^#' "${SPREAD_CSV}" | awk -F',' '$4 == -1 && ($3==24 || $3==28 || $3==32 || $3==40 || $3==48 || $3==56) {printf "  n=%-5s  threads=%-3s  gflops=%s\n", $2, $3, $6}'

echo ""
echo "================================================================="
echo " NUMA experiment finished."
echo " If close < spread at T>=32: NUMA hypothesis CONFIRMED."
echo " If close >= spread at T>=32: NUMA hypothesis REFUTED."
echo "================================================================="
