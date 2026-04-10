#!/bin/bash
# =============================================================================
# compare_v2v3_icelake.sh - V2 vs V3 comparison on icelake
#
# Builds CHOLESKY_VERSION=2 and CHOLESKY_VERSION=3 in the same job allocation,
# runs both benchmarks on the same node with OMP env.
#
# Focus: n=2000,4000 x threads=1,32,48,64,76 x 5 repeats (median reported).
# n=2000 provides a crosscheck.
#
# Outputs:
#   results/compare_v2_<JOBID>_<HOST>.csv
#   results/compare_v3_<JOBID>_<HOST>.csv
# =============================================================================

#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH --job-name=chol_cmp_v2v3
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/compare_v2v3-%j.out
#SBATCH --error=logs/compare_v2v3-%j.err

set -euo pipefail

# ── Reproducibility header ────────────────────────────────────────────────────
echo "================================================================="
echo " cholesky_benchmark  -  V2 vs V3 apples-to-apples comparison"
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

echo "--- CPU info ---"
lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA'
echo ""

# ── Provenance strings (embedded into CSV comment headers) ────────────────────
cd "${SLURM_SUBMIT_DIR}"

HOST=$(hostname -s)
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_TAGS=$(git tag --points-at HEAD 2>/dev/null | tr '\n' ',' | sed 's/,$//' || echo "none")
GCC_VER=$(gcc --version | head -1)
CMAKE_FLAGS_V2="-DCHOLESKY_VERSION=2 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=icelake-server"
CMAKE_FLAGS_V3="-DCHOLESKY_VERSION=3 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=icelake-server"

echo "Working directory : $(pwd)"
echo "Git hash          : ${GIT_HASH}"
echo "Git tags          : ${GIT_TAGS}"
echo ""

# ── OpenMP environment — identical for both V2 and V3 runs ───────────────────
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# OMP_NUM_THREADS is NOT set - benchmark controls thread count per measurement.

echo "--- OpenMP environment ---"
echo "OMP_PROC_BIND = ${OMP_PROC_BIND}"
echo "OMP_PLACES    = ${OMP_PLACES}"
echo ""

# ── Build VERSION 2 ───────────────────────────────────────────────────────────
echo "================================================================="
echo " Building VERSION 2"
echo "================================================================="
cmake -S . -B build_cmp_v2 \
      -DCHOLESKY_VERSION=2 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=icelake-server"
cmake --build build_cmp_v2 --clean-first -j1

echo ""
echo "--- CMakeCache key flags (V2) ---"
grep -E 'CHOLESKY_VERSION|BUILD_TYPE|CXX_FLAGS' build_cmp_v2/CMakeCache.txt | grep -v '//'
echo ""

# ── Build VERSION 3 ───────────────────────────────────────────────────────────
echo "================================================================="
echo " Building VERSION 3"
echo "================================================================="
cmake -S . -B build_cmp_v3 \
      -DCHOLESKY_VERSION=3 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=icelake-server"
cmake --build build_cmp_v3 --clean-first -j1

echo ""
echo "--- CMakeCache key flags (V3) ---"
grep -E 'CHOLESKY_VERSION|BUILD_TYPE|CXX_FLAGS' build_cmp_v3/CMakeCache.txt | grep -v '//'
echo ""

# ── Correctness — full test suite for V3 (V2 assumed correct from prior runs) ─
echo "================================================================="
echo " Correctness tests - VERSION 3"
echo "================================================================="
ctest --test-dir build_cmp_v3 --output-on-failure
echo ""

# ── Output paths ──────────────────────────────────────────────────────────────
OUTDIR="${SLURM_SUBMIT_DIR}/results"
mkdir -p "${OUTDIR}"
CSV_V2="${OUTDIR}/compare_v2_${SLURM_JOB_ID}_${HOST}.csv"
CSV_V3="${OUTDIR}/compare_v3_${SLURM_JOB_ID}_${HOST}.csv"

# ── Benchmark VERSION 2 ───────────────────────────────────────────────────────
echo "================================================================="
echo " Benchmarking VERSION 2"
echo "================================================================="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

{
  echo "# job_id=${SLURM_JOB_ID}"
  echo "# hostname=${HOST}"
  echo "# git_hash=${GIT_HASH}"
  echo "# git_tags=${GIT_TAGS}"
  echo "# compiler=${GCC_VER}"
  echo "# cmake_flags=${CMAKE_FLAGS_V2}"
  echo "# omp_proc_bind=${OMP_PROC_BIND}"
  echo "# omp_places=${OMP_PLACES}"
  echo "# timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  ./build_cmp_v2/bin/cholesky_benchmark \
      --sizes   "2000,4000" \
      --threads "1,32,48,64,76" \
      --repeats "5"
} > "${CSV_V2}"

echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CSV: ${CSV_V2}  ($(wc -l < "${CSV_V2}") rows)"
echo ""

# ── Benchmark VERSION 3 (back-to-back, same node, same allocation) ────────────
echo "================================================================="
echo " Benchmarking VERSION 3"
echo "================================================================="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

{
  echo "# job_id=${SLURM_JOB_ID}"
  echo "# hostname=${HOST}"
  echo "# git_hash=${GIT_HASH}"
  echo "# git_tags=${GIT_TAGS}"
  echo "# compiler=${GCC_VER}"
  echo "# cmake_flags=${CMAKE_FLAGS_V3}"
  echo "# omp_proc_bind=${OMP_PROC_BIND}"
  echo "# omp_places=${OMP_PLACES}"
  echo "# timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  ./build_cmp_v3/bin/cholesky_benchmark \
      --sizes   "2000,4000" \
      --threads "1,32,48,64,76" \
      --repeats "5"
} > "${CSV_V3}"

echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "CSV: ${CSV_V3}  ($(wc -l < "${CSV_V3}") rows)"
echo ""

# ── Side-by-side summary ──────────────────────────────────────────────────────
echo "================================================================="
echo " RESULTS SUMMARY (median rows, run=-1)"
echo "================================================================="
printf "%-8s  %-8s  %-14s  %-14s  %s\n" \
       "version" "n" "threads" "GFLOP/s" "speedup_over_V2"

declare -A v2_gflops
while IFS=',' read -r ver n thr run time gflops rest; do
    [[ "$run" == "-1" ]] && v2_gflops["${n}_${thr}"]="${gflops}"
done < <(grep -v '^#' "${CSV_V2}")

while IFS=',' read -r ver n thr run time gflops rest; do
    if [[ "$run" == "-1" ]]; then
        base="${v2_gflops["${n}_${thr}"]:-0}"
        if (( $(echo "$base > 0" | bc -l) )); then
            ratio=$(echo "scale=3; $gflops / $base" | bc)
        else
            ratio="-"
        fi
        printf "V2  vs V3  n=%-6s T=%-4s  V2=%7.3f  V3=%7.3f  ratio=%s\n" \
               "${n}" "${thr}" "${base}" "${gflops}" "${ratio}"
    fi
done < <(grep -v '^#' "${CSV_V3}")

echo ""
echo "================================================================="
echo " Job finished successfully."
echo "================================================================="
