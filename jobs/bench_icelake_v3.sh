#!/bin/bash
# =============================================================================
# bench_icelake_v3.sh - SLURM benchmark for VERSION 3 (column scaling tuning)
#
# Builds CHOLESKY_VERSION=3 (VERSION 2 and column scaling moved from omp single
# to omp for) and runs the full thread sweep to compare against VERSION 2.
#
# Compare results against bench_icelake_<JOBID>.csv (VERSION 2) to quantify
# the effect of parallelising the O(n) column-scaling step.
#
# Output:
#   results/bench_icelake_v3_<JOBID>_<TIMESTAMP>.csv
# =============================================================================

#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=cholesky_v3_icelake
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cholesky_v3_icelake-%j.out
#SBATCH --error=logs/cholesky_v3_icelake-%j.err

set -euo pipefail

echo "================================================================="
echo " cholesky_benchmark  -  VERSION 3  -  icelake partition"
echo " SLURM_JOB_ID      = ${SLURM_JOB_ID}"
echo " SLURM_JOB_NODELIST= ${SLURM_JOB_NODELIST}"
echo " Date/time         = $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "================================================================="
hostname
echo ""

module purge
module load gcc/11
module list 2>&1

echo ""
echo "--- Compiler ---"
gcc --version | head -1
g++ --version | head -1
echo ""

echo "--- CPU info (lscpu) ---"
lscpu
echo ""

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-76}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "--- OpenMP environment ---"
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
echo "OMP_PROC_BIND   = ${OMP_PROC_BIND}"
echo "OMP_PLACES      = ${OMP_PLACES}"
echo ""

cd "${SLURM_SUBMIT_DIR}"
echo "Working directory: $(pwd)"
echo ""

echo "--- CMake configure (VERSION 3) ---"
cmake -S . -B build \
      -DCHOLESKY_VERSION=3 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=icelake-server"

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
CSV_FILE="${OUTDIR}/bench_icelake_v3_${SLURM_JOB_ID}_${TIMESTAMP}.csv"

echo "--- Benchmark parameters ---"
echo "CHOLESKY_VERSION = 3  (column scaling parallelised)"
echo "Sizes            = 500,1000,2000,4000,8000"
echo "Threads          = 1,2,4,8,16,32,48,64,76"
echo "Repeats          = 3"
echo "CSV output       = ${CSV_FILE}"
echo ""

echo "--- Starting benchmark ---"
date '+%Y-%m-%d %H:%M:%S'

./build/bin/cholesky_benchmark \
    --sizes   "500,1000,2000,4000,8000" \
    --threads "1,2,4,8,16,32,48,64,76" \
    --repeats "3" \
    > "${CSV_FILE}"

echo ""
echo "--- Benchmark complete ---"
date '+%Y-%m-%d %H:%M:%S'
echo "CSV written to: ${CSV_FILE}"
echo "Rows: $(wc -l < "${CSV_FILE}")"
echo ""

echo "--- Median rows (run=-1) ---"
grep -v '^#' "${CSV_FILE}" | awk -F',' '$4 == -1 {printf "  n=%-5s  threads=%-3s  gflops=%s\n", $2, $3, $6}'

echo "================================================================="
echo " Job finished successfully."
echo "================================================================="
