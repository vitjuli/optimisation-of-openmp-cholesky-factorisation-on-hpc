#!/bin/bash
# =============================================================================
# template_icelake.sh is a template for CSD3 scripts for cholesky_example (icelake)
#
# Builds the example program (cholesky_example) with CHOLESKY_VERSION=2 and
# runs it for several matrix sizes, demonstrating the library on a CSD3 node.
#
# Output: logs/cholesky_example_icelake-<JOBID>.out
# =============================================================================

#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=cholesky_example_icelake
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/cholesky_example_icelake-%j.out
#SBATCH --error=logs/cholesky_example_icelake-%j.err

set -euo pipefail

echo "================================================================="
echo " cholesky_example  -  icelake partition"
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
echo ""

# Thread pinning: keep all threads on the same NUMA domain for small matrices.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-76}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "--- OpenMP environment ---"
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"
echo "OMP_PROC_BIND   = ${OMP_PROC_BIND}"
echo "OMP_PLACES      = ${OMP_PLACES}"
echo ""

cd "${SLURM_SUBMIT_DIR}"

echo "--- CMake configure (VERSION 2) ---"
cmake -S . -B build \
      -DCHOLESKY_VERSION=2 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=icelake-server"

echo ""
echo "--- CMake build ---"
cmake --build build --clean-first -j"${SLURM_CPUS_PER_TASK:-4}"
echo ""

echo "--- Correctness tests ---"
OMP_NUM_THREADS=1 ctest --test-dir build --output-on-failure
echo ""

# ── Run the example program at several matrix sizes ────────────────────────
# cholesky_example builds a corr() SPD matrix of dimension n, factorises it
# in-place, and reports wall-clock time, log|det(C)|, and GFLOP/s.
#
# The log|det(C)| value verifies the factorisation: a user of this library
# in a downstream GP or covariance application would use this quantity.
echo "================================================================="
echo " Running cholesky_example (CHOLESKY_VERSION=2, all available threads)"
echo "================================================================="
echo ""

for N in 200 500 1000 2000 4000; do
    echo "--- n = ${N} ---"
    ./build/bin/cholesky_example "${N}"
    echo ""
done

echo "================================================================="
echo " Example program finished successfully."
echo "================================================================="
