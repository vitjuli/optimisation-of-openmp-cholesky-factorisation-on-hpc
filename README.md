# MPhil DIS C2 High Performance Scientific Computing:
##  Optimisation of OpenMP Cholesky Factorisation on CSD3

**Iuliia Vitiugova** \
MPhil in Data Intensive Science \
University of Cambridge \
Lent Term 2026


## Overview

This library implements an Cholesky factorisation in C++17,
developed incrementally from a serial baseline through serial optimisation to
an OpenMP parallel version for deployment on CSD3.

Given a symmetric positive definite (SPD) matrix **$C$**, the algorithm computes
the lower triangular factor **$L$** such that **$C = LL^T$**, overwriting **$C$**
in place.  The log-determinant follows directly from the diagonal of **$L$**:

$$
\log|C| = 2 \cdot \sum_{p=0}^{n-1} \log(L_{pp})
$$

---

## Repository structure

```
include/mphil_dis_cholesky.h           — public API header
src/mphil_dis_cholesky.cpp             — implementation (VERSION 0/1/2/3)
src/mphil_dis_cholesky_testing.cpp     — test-only instrumentation hooks (step oracle, TID recorder, stride probe)
example/main.cpp                       — usage demonstration (corr() matrix, timing, log-det, GFLOP/s)
test/test_cholesky.cpp                 — full correctness suite (Groups A–K, 35–42 assertions by version)
test/test_n2.cpp                       — standalone 2x2 minimal test
benchmark/benchmark.cpp               — CSV benchmark harness (sizes × threads sweep)
jobs/bench_icelake.sh                  — SLURM job: icelake partition (76 cores, V2)
jobs/bench_cclake.sh                   — SLURM job: cclake partition (56 cores, V2)
jobs/bench_cclake_numa.sh              — NUMA placement experiment (close vs spread, cclake)
jobs/compare_v2v3_icelake.sh           — V2 vs V3 head-to-head comparison (icelake)
jobs/bench_icelake_v{0,1,3}.sh         — per-version baseline benchmarks
docs/Doxyfile                          — Doxygen configuration (HTML output to docs/html/)
report/                                — report
report/figures                         — figures
CMakeLists.txt                         — build system (CMake >= 3.16)
```
## Documentation

HTML documentation is generated from source via Doxygen.  It is **optional** -
the primary submission artefact is `report/report.pdf`.

### Generate and open locally

```bash
# Configure (Doxygen auto-detected, build succeeds even if not installed)
cmake -S . -B build

# Generate docs
cmake --build build --target docs

# Open in browser (no server needed — all files are local)
open docs/html/index.html          # macOS
xdg-open docs/html/index.html      # Linux / CSD3
```
Or run Doxygen directly from the repo root without CMake:

```bash
doxygen docs/Doxyfile
open docs/html/index.html
```

## Function interface

```cpp
#include "mphil_dis_cholesky.h"

double mphil_dis_cholesky(double* c, int n);
```

| Parameter | Description |
|-----------|-------------|
| `c` | Pointer to an `nxn` row-major array of doubles, modified in place |
| `n` | Matrix dimension |

**Input:** `c[i*n + j]` stores `C(i,j)`.

**Output (in place):**
- Lower triangle (`i >= j`): `L(i,j)`
- Upper triangle (`i < j`): `L^T(i,j)` = `L(j,i)` (mirrored)

**Return value:**

| Value | Description |
|-------|---------|
| `t > 0.0` | Wall-clock seconds spent on the factorisation (valid input) |
| `-1.0` | Invalid input: `c == nullptr`, `n <= 0`, or `n > 100000` |
| `-2.0` | Matrix is not SPD: a non-positive pivot was encountered |

**Example**:
```
Input C:   4  2      Output (in place):   2  1
           2  26                          1  5
```

---

## Implementation versions

Select the version at CMake configure time via `-DCHOLESKY_VERSION=<n>`:

| Version | Flag | Description | Git tag |
|---------|------|-------------|---------|
| 0 | `-DCHOLESKY_VERSION=0` | Serial baseline — direct transcription of spec pseudocode | `v0_serial_correct` |
| 1 | `-DCHOLESKY_VERSION=1` | Serial optimised — loop reorder, reciprocal, hoisting | `v1_serial_optimised` |
| 2 | `-DCHOLESKY_VERSION=2` | OpenMP parallel — persistent thread team | `v2_openmp_parallel` |
| 3 | `-DCHOLESKY_VERSION=3` | OpenMP tuned — column scaling moved into `omp for` | `v3_column_scaling` |

Version 0 is the default.  Versions 0 and 1 require no OpenMP installation.
Versions 2 and 3 require OpenMP; on macOS install `libomp` via Homebrew.

---

## Building

### Prerequisites

| Platform | Compiler | OpenMP |
|----------|----------|--------|
| Linux / CSD3 | GCC ≥ 9 | `module load gcc/11` |
| macOS | Apple Clang ≥ 14 | `brew install libomp` (version 2 only) |

CMake >= 3.16 is required on all platforms.

### Configure and build

```bash
# Serial baseline (default)
cmake -S . -B build -DCHOLESKY_VERSION=0
cmake --build build

# Serial optimised
cmake -S . -B build -DCHOLESKY_VERSION=1
cmake --build build

# OpenMP parallel
cmake -S . -B build -DCHOLESKY_VERSION=2
cmake --build build

# OpenMP tuned (column scaling parallelised)
cmake -S . -B build -DCHOLESKY_VERSION=3
cmake --build build
```

To additionally enable CPU-specific tuning (recommended on CSD3 after
selecting the appropriate partition):

```bash
# Generic native tuning
cmake -S . -B build -DCHOLESKY_VERSION=2 -DCHOLESKY_MARCH_NATIVE=ON

# CSD3 icelake partition (recommended over -march=native for reproducibility)
cmake -S . -B build -DCHOLESKY_VERSION=2 \
      -DCMAKE_CXX_FLAGS="-march=icelake-server"
```
---

## Example usage

The example program in `example/main.cpp` generates a deterministic SPD
correlation matrix using the `corr()` function from the coursework spec,
factorises it, and reports elapsed time, log-determinant, and throughput.

### Build and run

```bash
# Build (VERSION=0 default)
cmake -S . -B build -DCHOLESKY_VERSION=0
cmake --build build

# Default matrix size (n=200)
./build/bin/cholesky_example

# Custom matrix size
./build/bin/cholesky_example 500
./build/bin/cholesky_example 1000
```

### Example output

```
mphil_dis_cholesky example  (CHOLESKY_VERSION=0)
Matrix size     : n = 200
Elapsed time    : t      = 0.004751 s
log|det(C)|     : logdet = -877.102809
FLOP estimate   : (1/3)n^3 = 2.667e+06 FLOP
Throughput      : 0.5613 GFLOP/s
```

### With OpenMP (VERSION=2 or VERSION=3)

```bash
cmake -S . -B build -DCHOLESKY_VERSION=2
cmake --build build
OMP_NUM_THREADS=4 ./build/bin/cholesky_example 1000
```

The output fields are identical across all four versions; the elapsed time
and throughput will differ as parallelism increases.

---

## Expected outputs / sanity checks

All four versions implement the same algorithm; they must produce identical
`logdet` values and identical in-place `L` factors (to floating-point
rounding) for the same input.  Only timing and throughput differ.

Run with `n=200` (`corr()` SPD matrix, spec p.5, deterministic seed) to
verify a build before benchmarking:

```bash
cmake -S . -B build -DCHOLESKY_VERSION=<v> -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/bin/cholesky_example 200
```

### VERSION 0 — serial baseline

```
mphil_dis_cholesky example  (CHOLESKY_VERSION=0)
Matrix size     : n = 200
Elapsed time    : t      = 0.004751 s
log|det(C)|     : logdet = -877.102809
FLOP estimate   : (1/3)n^3 = 2.667e+06 FLOP
Throughput      : 0.5613 GFLOP/s
```

### VERSION 1 — serial optimised

```
mphil_dis_cholesky example  (CHOLESKY_VERSION=1)
Matrix size     : n = 200
Elapsed time    : t      = 0.000621 s   (hardware-dependent; ~5–8× faster than V0)
log|det(C)|     : logdet = -877.102809
FLOP estimate   : (1/3)n^3 = 2.667e+06 FLOP
Throughput      : 4.29 GFLOP/s
```

### VERSION 2 — OpenMP parallel (`OMP_NUM_THREADS=4`)

```
mphil_dis_cholesky example  (CHOLESKY_VERSION=2)
Matrix size     : n = 200
Elapsed time    : t      = 0.000298 s   (hardware-dependent; ~2–4× faster than V1 at T=4)
log|det(C)|     : logdet = -877.102809
FLOP estimate   : (1/3)n^3 = 2.667e+06 FLOP
Throughput      : 8.94 GFLOP/s
```

### VERSION 3 — OpenMP tuned (`OMP_NUM_THREADS=4`)

```
mphil_dis_cholesky example  (CHOLESKY_VERSION=3)
Matrix size     : n = 200
Elapsed time    : t      = 0.000261 s   (hardware-dependent; slightly faster than V2)
log|det(C)|     : logdet = -877.102809
FLOP estimate   : (1/3)n^3 = 2.667e+06 FLOP
Throughput      : 10.21 GFLOP/s
```

**Invariant:** `logdet = -877.102809` must be identical across all four
versions for the same `n=200` `corr()` matrix.  If it differs by more than
`~1e-8`, the build or input is wrong.  Timings and GFLOP/s are
hardware-dependent; the values shown above are from a local macOS run
(Apple M-series, 4 threads for V2/V3) and will differ on CSD3.

---

## Running the tests

### Quick run (all tests, recommended)

```bash
ctest --test-dir build --output-on-failure
```

### Verbose output (see every pass/fail line)

```bash
./build/bin/cholesky_test
```

### Run for every version

```bash
for v in 0 1 2 3; do
    cmake -S . -B build -DCHOLESKY_VERSION=$v -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j4
    echo "=== VERSION $v ==="
    ./build/bin/cholesky_test
done
```

Expected results: **V0 → 35/35 passed; V1 → 39/39 passed; V2 → 38/38 passed; V3 → 42/42 passed.**
(Test groups compile selectively: G always runs via the testing hook library;
H and J require OpenMP (V2/V3); H2 requires V3; I and I2 require V1.
Group K — SPD stress family — compiles and runs for all versions.)

---

## Test suite description

The suite in `test/test_cholesky.cpp` contains **35–42 assertions** depending
on the version (35 for V0; 39 for V1; 38 for V2; 42 for V3); each corresponds
to a distinct correctness property.  No external framework is used; the runner
is a single `CHECK` macro that counts pass/fail and prints a descriptive line
for every test.

All SPD matrices are generated deterministically using the `corr()` function
given on p. 5 of the coursework specification:

```cpp
corr(x, y, s) = 0.99 · exp(−0.5 · 16 · (x−y)² / s²),   diagonal = 1.0
```

This is the same function the markers use to assess correctness, so there
is no ambiguity about which matrices are being tested.

### Group A — Basic example (checks 1–2)

Verifies the 2x2 example:

$$C = \begin{pmatrix} 4 & 2 \\ 2 & 26 \end{pmatrix}  →  L·L^T = \begin{pmatrix} 2 & 1 \\ 1 & 5 \end{pmatrix}$$

- **T01a**: the returned wall-clock time is non-negative (`t >= 0.0`); valid
  input must not return an error code (which would be negative).
- **T01b**: all four elements of the in-place result match the expected values
  to within 10⁻¹⁴.

### Group B — Analytical matrices (checks 3–10)

Tests where the Cholesky factor and log-determinant are known exactly,
providing ground truth without any external library:

| Test | Matrix | Property verified |
|------|--------|-------------------|
| T02a | `C = [[4]]` (n=1) | `L[0][0] = 2` exactly |
| T02b | `C = [[4]]` (n=1) | `logdet = log(4)` via Eq. 4 |
| T03 (×2) | `I₃`, `I₁₆` | All diagonal entries of L equal 1 |
| T03 (×2) | `I₃`, `I₁₆` | `logdet = 0` via Eq. 4 (analytical: `log(det(I)) = 0`) |
| T04a | `4·I₃` | Diagonal of L equals `sqrt(4) = 2` |
| T04b | `4·I₃` | `logdet = 3·log(4)` via Eq. 4 (analytical: `log(det(4I₃)) = 3log4`) |

### Group C — SPD reconstruction, backward error (checks 11–15)

For each of five matrix sizes (n = 5, 16, 64, 128, 200) using `corr()`:

1. A copy of the original **C** is saved before the call.
2. `mphil_dis_cholesky` is called; the in-place result holds **L** (lower)
   and **L^T** (upper).
3. **C̃ = L·L^T** is reconstructed explicitly (O(n³) dense multiply).
4. The relative backward error is computed and bounded:
$$\|C − C̃\|_F / \|C\|_F  ≤  k·n·ε
$$

where $ε = 2.22×10^{-16}$ (double machine epsilon) and $k = 10$ ($k = 20$ for
n ≥ 128 to account for extra accumulation).  The actual errors observed
are typically 50–500x below the tolerance, confirming the algorithm is
numerically stable.

The returned timing value is also checked to be strictly positive within the
same assertion.

**Expected output (VERSION 0, n = 64):**
```
pass  T05: corr n= 64  ||C-LL^T||_F/||C||_F = 1.98e-16  tol = 1.42e-13  t = 0.0001 s
```

### Group D — Log-determinant identity, spec Eq. 4 (checks 16–17)

- **T06**: For the 2×2 spec matrix, `det = 4·26 − 2·2 = 100`, so
  `log|det| = log(100)`.  The value from `2·Σ log(L_pp)` is compared to
  this analytical result to within 10⁻¹³.

- **T07**: For the `corr()` matrix at n = 64, three properties are asserted
  jointly:
  - `logdet` is finite (no NaN or Inf).
  - `logdet ≤ 0` (Hadamard's inequality: $det(C) ≤ ∏ C_{ii} = 1$).
  - `|logdet − logdet_ref| ≤ n·10⁻¹²`, where `logdet_ref` is computed by an
    inline copy of the spec baseline pseudocode embedded in the test file —
    no external library required.  For VERSION 0 the difference is exactly 0;
    for VERSION 1 and 2 floating-point reordering may shift each `L_pp` by
    O(ε), accumulating to O(n·ε) ≈ 1.4×10⁻¹⁴ for n = 64, well within the
    tolerance of 6.4×10⁻¹¹.

### Group E — Input validation and error paths (checks 18–23)

Verifies every documented error code:

| Test | Input | Expected return |
|------|-------|----------------|
| T08a | `n = 0` | `−1.0` |
| T08b | `n = −1` | `−1.0` |
| T08c | `n = 100001` (exceeds limit) | `−1.0` |
| T08d | `c = nullptr`, `n = 2` | `−1.0` |
| T09a | `[[1,2],[2,1]]` (not PD: Schur gives `C[1][1] = 1 − 4 = −3`) | `−2.0` |
| T09b | `[[−1,0],[0,1]]` (first pivot `= −1 ≤ 0`) | `−2.0` |

### Group F — Structural correctness (checks 24–27)

After the in-place call on `corr()` matrices at n = 5 and n = 64:

- **Symmetry** (T10, ×2): `c[i·n+j] == c[j·n+i]` for all `i ≠ j` to within
  10⁻¹³.  This confirms the upper triangle stores exactly `L^T` (the mirror
  of the lower triangle `L`), as required by the spec.
- **Diagonal positivity** (T10, ×2): all `L_{pp} > 0`.  A necessary condition
  for the Cholesky factor to be valid; a zero or negative diagonal entry
  indicates a silent numerical failure.

### Group G — Step-level oracle proof (checks 28–29, all versions)

Applies pivot steps 0..n−1 one at a time, comparing `mphil_dis_cholesky_step()`
against an independent serial reference (`ref_step()`) element-wise after each
step.  Tolerance: `50·ε·n·‖c‖_max`.  Catches wrong loop bounds, missing
scales, or O(1) race-corrupted elements that might cancel in the end-to-end
result.  Run for n = 5 and n = 16.  Requires `MPHIL_CHOLESKY_TESTING=1`
(always set by the CMake test targets).

### Group K — SPD stress family (last 6 checks, all versions)

Tests the factorisation on a second, structurally distinct SPD matrix family:
`C = A·Aᵀ + 0.1·I` where `A[i][j] = sin((i+1)(j+1)) / √n`.  This matrix is
always strictly SPD regardless of n, and its off-diagonal structure differs
from `corr()`.  Three assertions per size (n = 8 and n = 40): timing ≥ 0,
backward error `≤ 20·n·ε`, and log-det within `n·10⁻¹²` of a reference run.
The purpose is to rule out an implementation tuned specifically to the `corr()`
family.

---

## Expected test output (VERSION 0)

```
mphil_dis_cholesky test suite (CHOLESKY_VERSION=0)

pass  T01a: 2x2 spec timing >= 0  (t=0.000000365 s)
pass  T01b: 2x2 spec output [[2,1],[1,5]]  (got [2.000000 1.000000 1.000000 5.000000])
pass  T02a: n=1  L[0][0] = 2  (got 2.0000000000)
pass  T02b: n=1  logdet = log(4)  (got 1.3862943611, exp 1.3862943611)
pass  T03: I_3  diagonal all 1
pass  T03: I_3  logdet = 0  (got 0.000e+00)
pass  T03: I_16  diagonal all 1
pass  T03: I_16  logdet = 0  (got 0.000e+00)
pass  T04: 4*I_3  diagonal = sqrt(k) = 2.000000
pass  T04: 4*I_3  logdet = n*log(k)  (got 4.15888308, exp 4.15888308)
pass  T05: corr n=  5  ||C-LL^T||_F/||C||_F = 2.53e-17  tol = 1.11e-14  t = 0.0000 s
pass  T05: corr n= 16  ||C-LL^T||_F/||C||_F = 8.39e-17  tol = 3.55e-14  t = 0.0000 s
pass  T05: corr n= 64  ||C-LL^T||_F/||C||_F = 1.98e-16  tol = 1.42e-13  t = 0.0001 s
pass  T05: corr n=128  ||C-LL^T||_F/||C||_F = 2.83e-16  tol = 5.68e-13  t = 0.0028 s
pass  T05: corr n=200  ||C-LL^T||_F/||C||_F = 3.54e-16  tol = 8.88e-13  t = 0.0026 s
pass  T06: 2x2 logdet = log(100)  (got 4.6051701860, exp 4.6051701860)
pass  T07: corr n=64  logdet=-258.904341  ref=-258.904341  |diff|=0.00e+00  tol=6.40e-11
pass  T08a: n=0      -> -1.0
pass  T08b: n=-1     -> -1.0
pass  T08c: n=100001 -> -1.0
pass  T08d: nullptr  -> -1.0
pass  T09a: non-SPD [[1,2],[2,1]]   -> -2.0
pass  T09b: non-SPD [[-1,0],[0,1]]  -> -2.0
pass  T10: corr n=5  upper/lower symmetry (c[i*n+j]==c[j*n+i])
pass  T10: corr n=5  diagonal L_pp > 0
pass  T10: corr n=64  upper/lower symmetry (c[i*n+j]==c[j*n+i])
pass  T10: corr n=64  diagonal L_pp > 0

--- G: step-level oracle (MPHIL_CHOLESKY_TESTING) ---
pass  T_STEP n= 5:  5 pivot steps match ref  (worst p=-1  ratio=0.000  tol=50·ε·n·|max|)
pass  T_STEP n=16: 16 pivot steps match ref  (worst p=-1  ratio=0.000  tol=50·ε·n·|max|)

--- K: SPD stress family (A*A^T + 0.1*I) ---
pass  T_SPD n= 8: factorisation returns t>=0 (t=...)
pass  T_SPD n= 8: backward error ... <= 20nε=3.55e-14
pass  T_SPD n= 8: logdet=...  |diff vs ref|=0.00e+00  tol=n·1e-12=8.00e-12
pass  T_SPD n=40: factorisation returns t>=0 (t=...)
pass  T_SPD n=40: backward error ... <= 20nε=1.78e-13
pass  T_SPD n=40: logdet=...  |diff vs ref|=0.00e+00  tol=n·1e-12=4.00e-11

35 / 35 tests passed.
```

Timing values (`t=...`) and backward-error magnitudes are hardware-dependent;
tolerances (`20nε`, `n·1e-12`) are deterministic and match what is shown.

---

## Performant use

- **Thread count**: set `OMP_NUM_THREADS` to match the number of physical
  cores available.  On CSD3 icelake nodes (76 cores) use
  `OMP_NUM_THREADS=76`.  Hyperthreading typically does not help for this
  memory-bandwidth-bound kernel.

- **Thread affinity**: bind threads to cores to avoid NUMA effects:
  ```bash
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
  ```

- **Problem size**: the routine is designed for large matrices.  For n < 64
  the parallel overhead typically exceeds the computation time; use VERSION 1
  for small matrices and VERSION 2 for n ≥ 500.

- **Compilation**: `-O3` is applied by default.  For CSD3 icelake nodes,
  add `-march=icelake-server` (preferred over `-march=native` for
  reproducibility across nodes in the same partition).

---

## CSD3 quick-start

```bash
module load gcc/11
cmake -S . -B build \
      -DCHOLESKY_VERSION=2 \
      -DCMAKE_CXX_FLAGS="-march=icelake-server"
cmake --build build -j4

export OMP_NUM_THREADS=76
export OMP_PROC_BIND=close
export OMP_PLACES=cores
ctest --test-dir build --output-on-failure
```

---

## Benchmark

The benchmark driver (`benchmark/benchmark.cpp`) sweeps matrix sizes × thread
counts, emits one CSV row per timed run, and appends a median-summary row per
`(n, threads)` pair.

### Build and run locally

```bash
# Build (VERSION=2 recommended for scaling results)
cmake -S . -B build -DCHOLESKY_VERSION=2 -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Default sweep: n ∈ {500,1000,2000,4000} × threads ∈ {1,2,4,8,16,32,64,76}
OMP_NUM_THREADS=8 ./build/bin/cholesky_benchmark > results.csv

# Custom sweep
OMP_NUM_THREADS=8 ./build/bin/cholesky_benchmark \
    --sizes   "500,1000,2000"  \
    --threads "1,2,4,8"        \
    --repeats 5                \
    > results.csv

# Suppress CSV header (useful when appending from multiple runs)
./build/bin/cholesky_benchmark --no-header >> results.csv
```

`OMP_NUM_THREADS` controls the ceiling for thread-count capping.  Set it to
the number of physical cores available.

### Run on CSD3 via SLURM

```bash
# icelake partition (76 cores, AVX-512, -march=icelake-server)
sbatch jobs/bench_icelake.sh

# cclake partition (56 cores, AVX-512, -march=cascadelake)
sbatch jobs/bench_cclake.sh
```

Each script:
1. Loads `gcc/11` and configures cmake with the partition-specific march flag.
2. Builds from `$SLURM_SUBMIT_DIR` (the directory where you call `sbatch`).
3. Runs the correctness tests before benchmarking.
4. Writes the CSV to `results/bench_<partition>_<JOBID>_<TIMESTAMP>.csv`.
5. Prints a GFLOP/s summary table to the SLURM log.

Override the sweep without editing the scripts:

```bash
BENCH_SIZES="1000,2000,4000,8000" \
BENCH_THREADS="1,2,4,8,16,32,48,64,76" \
sbatch jobs/bench_icelake.sh
```

### CSV column reference

| Column    | Type    | Description |
|-----------|---------|-------------|
| `version` | integer | `CHOLESKY_VERSION` (0 = baseline, 1 = optimised, 2 = OpenMP, 3 = tuned) |
| `n`       | integer | Matrix dimension |
| `threads` | integer | Actual thread count used (≤ `OMP_NUM_THREADS`) |
| `run`     | integer | 0, 1, 2, … for individual timed runs; **−1 for the median-summary row** |
| `time_s`  | float   | Wall-clock seconds returned by `mphil_dis_cholesky`; negative = error code |
| `gflops`  | float   | `(1/3)·n³ / (time_s·10⁹)`; 0.0 on error |
| `hostname`| string  | Result of `gethostname()` — identifies the CSD3 partition/node |

Lines beginning with `#` are metadata comments (version, hostname, sizes,
threads, repeats) and are not CSV data rows.  Most parsers ignore them:

```python
import pandas as pd
df = pd.read_csv("results.csv", comment='#')
medians = df[df['run'] == -1]
```

---

### Use of LLMs

LLMs supported editorial work on the report/README (clarity, structure, and English), helped standardise figure presentation (caption wording, consistent axis naming, and layout of result subsections), and helped with the generation of the Doxygen documentation and docstrings in code. Code, benchmarks, and all reported numbers were implemented and verified by the author.
