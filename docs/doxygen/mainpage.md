# mphil_dis_cholesky

Single-function C library computing the Cholesky factorisation
\f$C = LL^T\f$ in place, for an \f$n \times n\f$ symmetric
positive-definite (SPD) matrix stored in row-major order.  Four
compile-time versions (V0–V3) progress from a serial pseudocode
baseline to an OpenMP-parallel implementation with AVX-512 SIMD
vectorisation, targeting CSD3 HPC nodes.

## Quick start

```cmake
cmake -S . -B build -DCHOLESKY_VERSION=2 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## Minimal usage

```c
#include "mphil_dis_cholesky.h"

/* C: row-major n*n SPD matrix (n*n doubles), modified in place. */
double t = mphil_dis_cholesky(C, n);

if      (t == -1.0) { /* invalid input: NULL, n<=0, or n>100000 */ }
else if (t == -2.0) { /* non-positive pivot: matrix is not SPD   */ }
else                { /* t >= 0 seconds; C holds L (lower) and L^T (upper) */ }
```

After a successful call, `log|det C| = 2 * sum_{p} log(C[p*n+p])`.

## Pages

- @subpage build         — CMake options, version table, OpenMP requirements
- @subpage testing       — test executables, groups A–K, how to run
- @subpage benchmarking  — CLI options, CSV format, GFLOP/s definition, SLURM job table
- @subpage pipeline      — end-to-end workflow: SLURM → CSV → plots → tables
- @subpage plots         — all figures and summary tables with descriptions

See also: @ref mphil_dis_cholesky "API reference (mphil_dis_cholesky)".
