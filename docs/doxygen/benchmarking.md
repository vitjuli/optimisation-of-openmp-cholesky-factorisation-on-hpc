# Benchmarking {#benchmarking}

## Running the benchmark

```bash
./build/bin/cholesky_benchmark \
    --sizes   "500,1000,2000,4000,8000" \
    --threads "1,2,4,8,16,32,64,76"    \
    --repeats "3"                       \
    > results/bench.csv
```

| Option | Default | Description |
|--------|---------|-------------|
| `--sizes` | `500,1000,2000,4000` | Comma-separated matrix dimensions n |
| `--threads` | `1,2,4,8,16,32,64,76` | Comma-separated thread counts; capped at `omp_get_max_threads()` |
| `--repeats` | `3` | Timed runs per (n, threads) block; one untimed warm-up precedes them |
| `--no-header` | off | Suppress the metadata comment block and CSV header row |

Stdout carries the CSV; stderr carries metadata notes and any thread-cap warnings.
Redirect stdout to file; inspect stderr freely.

## CSV format

```
# version=2  host=cpu-q-343  sizes=500,1000  threads=1,4  repeats=3
version,n,threads,run,t_sec,gflops_per_sec
2,1000,4,0,0.041,16.23
2,1000,4,1,0.040,16.60
2,1000,4,2,0.039,17.04
2,1000,4,-1,0.040,16.62
```

| Column | Type | Description |
|--------|------|-------------|
| `version` | int | `CHOLESKY_VERSION` compiled into the binary (0–3) |
| `n` | int | Matrix dimension |
| `threads` | int | Thread count requested for this block |
| `run` | int | Run index 0…(repeats-1); **-1 = median summary row** |
| `t_sec` | float | Wall-clock seconds (`omp_get_wtime()` for V2/V3; `chrono` for V0/V1) |
| `gflops_per_sec` | float | \f$\frac{n^3/3}{t \times 10^9}\f$ — dominant cost is the Schur complement |

Use the **median row** (`run = -1`) as the primary result; it is robust to OS scheduling jitter.

## GFLOP/s definition

The dominant kernel is the Schur complement update:
\f[
  C[i][j] \mathrel{-}= L[i][p] \cdot L^T[p][j] \quad \forall\, i,j > p
\f]
across all pivot steps p.  Total floating-point operations \f$\approx n^3/3\f$.
Hence:
\f[
  \text{GFLOP/s} = \frac{n^3 / 3}{t_\text{sec} \times 10^9}
\f]

## SLURM job scripts

| Script | Partition | Cores | Purpose |
|--------|-----------|-------|---------|
| `jobs/bench_icelake.sh` | icelake | 76 | V2 full sweep: n ∈ {500…8000}, T ∈ {1…76}; primary scaling dataset |
| `jobs/bench_icelake_v0.sh` | icelake | 1 | V0 baseline, n ≤ 4000 (n=8000 excluded: too slow at O(n³) cache-miss rate) |
| `jobs/bench_icelake_v0_n8000.sh` | icelake | 1 | V0 at n=8000 only; 1 repeat, 2h30 time limit |
| `jobs/bench_icelake_v1.sh` | icelake | 1 | V1 serial optimised, all sizes including n=8000 |
| `jobs/bench_icelake_v3.sh` | icelake | 76 | V3 full sweep (column-scaling parallelised); compare against V2 |
| `jobs/bench_cclake.sh` | cclake | 56 | V2 full sweep on Cascade Lake; cross-architecture comparison |
| `jobs/bench_cclake_numa.sh` | cclake | 56 | NUMA experiment: `OMP_PROC_BIND=close` vs `spread` at T > 28 |
| `jobs/compare_v2v3_icelake.sh` | icelake | 76 | Back-to-back V2 vs V3 on the same node; n=2000,4000, 5 repeats |
| `jobs/template_icelake.sh` | icelake | 76 | Runs `cholesky_example` at n ∈ {200…4000}; useful for sanity checks |

Submit from the repository root.  Override sweep parameters via environment variables:
```bash
BENCH_SIZES="1000,2000,4000" BENCH_THREADS="1,8,16,32,64,76" \
    sbatch jobs/bench_icelake.sh
```
