#!/bin/bash
# =============================================================================
# bench_icelake_v1.sh - SLURM serial optimised (VERSION 1) on icelake
#
# Builds CHOLESKY_VERSION=1 (loop swap, pointer hoisting, reciprocal
# multiply, scalar hoist, 4-wide unrolling) and measures single-thread
# performance across all sizes.
#
# Run after bench_icelake_v0.sh to obtain the V0 vs V1 speedup table.
#
# IMPORTANT: This script uses build_v1/ (not build/) so that it can be
# submitted concurrently with bench_icelake_v0.sh without build-directory
# conflicts on the shared CSD3 filesystem.
#
# Output:
#   results/bench_icelake_v1_<JOBID>_<TIMESTAMP>.csv
# =============================================================================

#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=cholesky_v1_icelake
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cholesky_v1_icelake-%j.out
#SBATCH --error=logs/cholesky_v1_icelake-%j.err

set -euo pipefail

echo "================================================================="
echo " cholesky_benchmark  -  VERSION 1  -  icelake partition"
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

cd "${SLURM_SUBMIT_DIR}"
echo "Working directory: $(pwd)"
echo ""

echo "--- CMake configure ---"
cmake -S . -B build_v1 \
      -DCHOLESKY_VERSION=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=icelake-server"

echo ""
echo "--- CMake build ---"
cmake --build build_v1 --clean-first -j1
echo ""

echo "--- Correctness tests ---"
ctest --test-dir build_v1 --output-on-failure
echo ""

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTDIR="${SLURM_SUBMIT_DIR}/results"
mkdir -p "${OUTDIR}"
CSV_FILE="${OUTDIR}/bench_icelake_v1_${SLURM_JOB_ID}_${TIMESTAMP}.csv"

echo "--- Benchmark parameters ---"
echo "CHOLESKY_VERSION = 1"
echo "Sizes            = 500,1000,2000,4000,8000"
echo "Threads          = 1  (serial build)"
echo "Repeats          = 3"
echo "CSV output       = ${CSV_FILE}"
echo ""

echo "--- Starting benchmark ---"
date '+%Y-%m-%d %H:%M:%S'

./build_v1/bin/cholesky_benchmark \
    --sizes   "500,1000,2000,4000,8000" \
    --threads "1" \
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
