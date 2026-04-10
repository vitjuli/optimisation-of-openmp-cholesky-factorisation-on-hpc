#!/bin/bash
# =============================================================================
# bench_icelake_v0_n8000.sh — VERSION 0 at n=8000 only, icelake partition
#
# Partition : icelake  (Intel Xeon Platinum 8360Y, 76 cores / node)
# Version   : CHOLESKY_VERSION=0  (serial baseline, j-outer/i-inner Schur)
# Purpose   : Single data point completing the V0 size sweep.  n=8000 is
#             excluded from bench_icelake_v0.sh (wall-time breach risk);
#             this dedicated job uses 1 repeat and a 2h30 time limit.
#
# Output:
#   results/bench_icelake_v0_n8000_<JOBID>_<TIMESTAMP>.csv
# =============================================================================

#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=cholesky_v0_n8000
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --output=logs/cholesky_v0_n8000-%j.out
#SBATCH --error=logs/cholesky_v0_n8000-%j.err

set -euo pipefail

echo "================================================================="
echo " cholesky_benchmark  -  VERSION 0  -  n=8000 only"
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

cd "${SLURM_SUBMIT_DIR}"
echo "Working directory: $(pwd)"
echo ""

echo "--- CMake configure ---"
cmake -S . -B build_v0_n8000 \
      -DCHOLESKY_VERSION=0 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=icelake-server"

echo "--- CMake build ---"
cmake --build build_v0_n8000 --clean-first -j1
echo ""

echo "--- Correctness tests ---"
ctest --test-dir build_v0_n8000 --output-on-failure
echo ""

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTDIR="${SLURM_SUBMIT_DIR}/results"
mkdir -p "${OUTDIR}"
CSV_FILE="${OUTDIR}/bench_icelake_v0_n8000_${SLURM_JOB_ID}_${TIMESTAMP}.csv"

echo "--- Starting benchmark (n=8000, T=1, 1 repeat) ---"
echo "    (1 repeat sufficient: variance negligible at this scale; avoids wall-time breach)"
date '+%Y-%m-%d %H:%M:%S'

./build_v0_n8000/bin/cholesky_benchmark \
    --sizes   "8000" \
    --threads "1" \
    --repeats "1" \
    > "${CSV_FILE}"

echo ""
echo "--- Benchmark complete ---"
date '+%Y-%m-%d %H:%M:%S'
echo "CSV written to: ${CSV_FILE}"
echo ""

echo "--- Result (run=-1 row) ---"
grep -v '^#' "${CSV_FILE}" | awk -F',' '$4 == -1 {printf "  n=%-5s  threads=%-3s  gflops=%s\n", $2, $3, $6}'

echo "================================================================="
echo " Job finished successfully."
echo "================================================================="
