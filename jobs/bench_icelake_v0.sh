#!/bin/bash
# =============================================================================
# bench_icelake_v0.sh - SLURM serial baseline (VERSION 0) on icelake
#
# Builds CHOLESKY_VERSION=0 and measures single-thread performance for comparison with VERSIONs 1 and 2.
#
# n=8000 is excluded: VERSION 0 has O(n^3) cache-unfriendly access and would
# exceed the time limit.  Use bench_icelake_v1.sh for n=8000 data.
#
# IMPORTANT: This script uses build_v0/ so that it can be
# submitted concurrently with bench_icelake_v1.sh without build-directory
# conflicts on the shared CSD3 filesystem.
#
# Output:
#   results/bench_icelake_v0_<JOBID>_<TIMESTAMP>.csv
# =============================================================================

#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=cholesky_v0_icelake
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=logs/cholesky_v0_icelake-%j.out
#SBATCH --error=logs/cholesky_v0_icelake-%j.err

set -euo pipefail

echo "================================================================="
echo " cholesky_benchmark  -  VERSION 0  -  icelake partition"
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

# Build VERSION 0 with Release flags.
# No OpenMP needed: VERSION 0 is serial.
echo "--- CMake configure ---"
cmake -S . -B build_v0 \
      -DCHOLESKY_VERSION=0 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=icelake-server"

echo ""
echo "--- CMake build ---"
cmake --build build_v0 --clean-first -j1
echo ""

# Correctness check before benchmarking.
echo "--- Correctness tests ---"
ctest --test-dir build_v0 --output-on-failure
echo ""

# Output CSV
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTDIR="${SLURM_SUBMIT_DIR}/results"
mkdir -p "${OUTDIR}"
CSV_FILE="${OUTDIR}/bench_icelake_v0_${SLURM_JOB_ID}_${TIMESTAMP}.csv"

echo "--- Benchmark parameters ---"
echo "CHOLESKY_VERSION = 0"
echo "Sizes            = 500,1000,2000,4000  (n=8000 excluded: too slow for V0)"
echo "Threads          = 1  (serial build; all thread values cap to 1)"
echo "Repeats          = 3"
echo "CSV output       = ${CSV_FILE}"
echo ""

echo "--- Starting benchmark ---"
date '+%Y-%m-%d %H:%M:%S'

./build_v0/bin/cholesky_benchmark \
    --sizes   "500,1000,2000,4000" \
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
