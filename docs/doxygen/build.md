# Build & Configuration {#build}

## CMake options

| Option | Values | Default | Effect |
|--------|--------|---------|--------|
| `CHOLESKY_VERSION` | 0, 1, 2, 3 | 0 | Select implementation version (see table below) |
| `CMAKE_BUILD_TYPE` | `Release`, `Debug` | — | `Release` applies `-O3`; omit for CMake default |
| `CMAKE_CXX_FLAGS` | any | — | Append extra flags, e.g. `-march=icelake-server` |

Example:
```cmake
cmake -S . -B build \
      -DCHOLESKY_VERSION=2 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=icelake-server"
cmake --build build --clean-first -j$(nproc)
```

## Version table

This is the single canonical description of all four versions.

| Version | Git tag | Description |
|---------|---------|-------------|
| **V0** | `v0_serial_correct` | Serial baseline. Direct transcription of the coursework pseudocode. j-outer/i-inner Schur update (stride-n, cache-unfriendly). Reference for correctness and baseline timing. |
| **V1** | `v1_serial_optimised` | Serial optimised. Five targeted micro-optimisations: loop-order swap to j-inner (OPT-1), row-pointer hoisting (OPT-2), reciprocal multiply instead of division (OPT-3), `L[i][p]` scalar hoist (OPT-4), 4-wide i-loop unrolling (OPT-5). |
| **V2** | `v2_openmp_parallel` | OpenMP parallel. One persistent `#pragma omp parallel` region wraps the p-loop, eliminating O(n) fork/join overhead. Column scaling executes in `omp single`; Schur update distributed by row via `omp for schedule(static)`. |
| **V3** | `v3_column_scaling` | OpenMP tuned. Identical to V2 except column scaling is moved from `omp single` into the `omp for` body, eliminating the O(n²) serial stall at high thread counts caused by stride-n DRAM accesses on a single thread. |

## OpenMP dependency

| Versions | OpenMP required | Notes |
|----------|----------------|-------|
| 0, 1 | No | Pure serial; compile and run without any OpenMP installation. |
| 2, 3 | Yes | Use `#pragma omp` directives and `omp_get_wtime()`. |

**macOS.** Apple Clang does not bundle OpenMP. Install via Homebrew and
let CMakeLists.txt handle the link flags automatically:
```bash
brew install libomp
```

**CSD3 icelake / cclake.** Load the compiler module before configuring:
```bash
module purge
module load gcc/11
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

## Generating documentation

Requires Doxygen to be installed (`doxygen --version`).

```bash
cmake --build build --target docs
open docs/html/index.html        # macOS
xdg-open docs/html/index.html   # Linux / CSD3
```

Or run Doxygen directly from the repository root:
```bash
doxygen docs/Doxyfile
```
