# Testing {#testing}

## Test executables

| CTest name | Source | Description |
|------------|--------|-------------|
| `CholeskyN2` | `test/test_n2.cpp` | 2×2 smoke test (spec example). Runs first. |
| `Cholesky` | `test/test_cholesky.cpp` | Full suite, groups A–K. |

## Running

```bash
# All tests:
ctest --test-dir build --output-on-failure

# Single-threaded (V2/V3 — avoids OpenMP scheduling noise):
OMP_NUM_THREADS=1 ctest --test-dir build --output-on-failure

# Specific test only:
ctest --test-dir build -R CholeskyN2 --output-on-failure
```

## Test groups

| Group | Versions | What it checks |
|-------|----------|----------------|
| A | All | Invalid input: `NULL` pointer, `n <= 0`, `n > 100000` → returns -1.0 |
| B | All | 2×2 spec example: `[[4,2],[2,26]]` → `[[2,1],[1,5]]` within 10⁻¹² |
| C | All | `n = 1` edge case; timing return value `>= 0.0` |
| D | All | Backward error \f$\|LL^T - A\|_F / \|A\|_F \le 10^{-10}\f$ for n ∈ {5,16,64,128,256} |
| E | All | Log-determinant accuracy: \f$|{}\log|\det C|_\text{computed} - \text{ref}| \le 1.5 \times 10^{-10}\f$ |
| F | All | Non-SPD input → returns -2.0 |
| G | All | Step oracle: `mphil_dis_cholesky_step()` matches independent reference pivot-by-pivot for n=5 and n=16 |
| H | V2, V3 | Thread-count invariance: backward error and \f$|\Delta\log\det|\f$ within tolerance for T ∈ {1,2,4,8,16} |
| H2 | V3 only | Column-scaling parallelism proof: ≥ 2 distinct thread IDs recorded in column writes |
| I | V1 only | Scalar-tail coverage: correctness for n=6 (1 tail row) and n=7 (2 tail rows) |
| I2 | V1 only | OPT-1 structural proof: first two Schur write addresses have stride 1 (j-inner active) |
| J | V2, V3 | Barrier sanity: one-step correctness at T=8 with a scaled diagonal matrix |
| K | All | SPD stress family \f$(AA^T + 0.1I)\f$ backward error and log-det for n=8 and n=40 |

**Assertion counts:** V0 → 35, V1 → 39, V2 → 38, V3 → 42.

## Testing hooks

Groups G, H2, I, I2 use test-only functions from
`src/mphil_dis_cholesky_testing.cpp` (the `cholesky_testing_hooks`
static library).  That library is linked **only** to the test
executables — never to `cholesky_example` or `cholesky_benchmark`.
The production library contains no test code.
