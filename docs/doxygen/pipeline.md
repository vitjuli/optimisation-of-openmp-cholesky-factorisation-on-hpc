# Benchmark & Plotting Pipeline {#pipeline}

End-to-end workflow from a CSD3 allocation to publication-ready figures.

```
SLURM job scripts (jobs/)
        │
        │  build + run cholesky_benchmark
        │  → results/<name>_<JOBID>_<TIMESTAMP>.csv
        ▼
results/*.csv
        │
        │  python3 scripts/plot_bench.py --results results/ --outdir report/
        │
        ├─→ report/figures/*.png   (16 plot files)
        └─→ report/tables/peak_summary.{csv,md}
```

---

## Step 1 — Run the SLURM jobs

Submit the benchmark jobs from the repository root.  Each script builds the
project inside the allocation, verifies correctness with `ctest`, then runs the
benchmark binary and writes a self-describing CSV to `results/`.

**Recommended submission order:**

```bash
# Serial baselines (concurrent: use separate build_v0/ / build_v1/ dirs)
sbatch jobs/bench_icelake_v0.sh
sbatch jobs/bench_icelake_v1.sh

# V0 at n=8000 separately (2h30 wall-time; 1 repeat only)
sbatch jobs/bench_icelake_v0_n8000.sh

# OpenMP sweep — primary dataset
sbatch jobs/bench_icelake.sh          # V2, icelake, 76 threads
sbatch jobs/bench_cclake.sh           # V2, cclake,  56 threads

# V3 and V2-vs-V3 comparison
sbatch jobs/bench_icelake_v3.sh
sbatch jobs/compare_v2v3_icelake.sh

# NUMA experiment (cclake only)
sbatch jobs/bench_cclake_numa.sh
```

See the @ref benchmarking page for the full job table and parameter overrides.

---

## Step 2 — Inspect the CSV output

Every CSV starts with `#`-prefixed metadata lines followed by the data rows:

```
# cholesky_benchmark  CHOLESKY_VERSION=2
# hostname=cpu-q-536
# max_threads=76
# sizes=500,1000,2000,4000,8000
# threads=1,2,4,8,16,32,48,64,76
# repeats=3
version,n,threads,run,time_s,gflops,hostname
2,500,1,0,0.0234,10.67,cpu-q-536
2,500,1,1,0.0231,10.80,cpu-q-536
2,500,1,2,0.0229,10.89,cpu-q-536
2,500,1,-1,0.0231,10.80,cpu-q-536   ← median summary row (run = -1)
```

| Column | Type | Description |
|--------|------|-------------|
| `version` | int | `CHOLESKY_VERSION` compiled into the binary (0–3) |
| `n` | int | Matrix dimension |
| `threads` | int | Thread count used |
| `run` | int | Repeat index 0…(repeats-1); **-1 = median summary** |
| `time_s` | float | Wall-clock seconds from `mphil_dis_cholesky()` |
| `gflops` | float | \f$n^3/(3 \cdot t \cdot 10^9)\f$ |
| `hostname` | string | CSD3 node (identifies partition) |

The plotting script uses only `run == -1` rows for all figures.

**Quick look at the median rows:**
```bash
awk -F',' '$4 == -1' results/bench_icelake_*.csv | column -t -s','
```

---

## Step 3 — Generate plots and tables

```bash
# From repo root (requires pandas, matplotlib; scipy optional for Amdahl fits)
pip install pandas matplotlib scipy

python3 scripts/plot_bench.py \
    --results results/ \
    --outdir  report/
```

All outputs are written into `report/figures/` and `report/tables/`.  The
`--outdir` argument controls the root; subdirectories are created automatically.

**Override partitions or versions:**
```bash
# Only icelake data
python3 scripts/plot_bench.py --results results/ --outdir report/ \
    --partitions icelake

# Only V2 and V3
python3 scripts/plot_bench.py --results results/ --outdir report/ \
    --versions 2,3
```

---

## Step 4 — Output inventory

### Figures (`report/figures/`)

See the @ref plots page for descriptions and sample images.

| File | Plot type | Data source |
|------|-----------|-------------|
| `gflops_icelake.png` | GFLOP/s vs threads | icelake V2 |
| `gflops_cclake.png` | GFLOP/s vs threads | cclake V2 |
| `speedup_icelake.png` | Speedup vs threads | icelake V2 |
| `speedup_cclake.png` | Speedup vs threads | cclake V2 |
| `efficiency_icelake.png` | Parallel efficiency (%) | icelake V2 |
| `efficiency_cclake.png` | Parallel efficiency (%) | cclake V2 |
| `speedup_heatmap_icelake.png` | Speedup heatmap (n × T) | icelake V2 |
| `speedup_heatmap_cclake.png` | Speedup heatmap (n × T) | cclake V2 |
| `v3_v2_heatmap.png` | V3/V2 improvement (%) heatmap | icelake V2 vs V3 |
| `amdahl_icelake.png` | Speedup + fitted Amdahl curve | icelake V2 |
| `amdahl_cclake.png` | Speedup + fitted Amdahl curve | cclake V2 |
| `time_vs_n.png` | Wall-clock vs n (log-log) | all versions, T=1 |
| `peak_bar_icelake.png` | Peak GFLOP/s bar, all versions | icelake |
| `peak_bar_cclake.png` | Peak GFLOP/s bar, all versions | cclake |
| `platform_comparison.png` | icelake vs cclake GFLOP/s | V2 both partitions |
| `numa_cclake.png` | NUMA close vs spread | cclake n=2000 |
| `numa_ratio_cclake.png` | NUMA close/spread ratio | cclake |
| `v0_v1_v2_icelake.png` | V0/V1/V2/V3 single-thread | icelake T=1 |
| `v2_v3_comparison.png` | V2 vs V3 tuning effect | icelake V2, V3 |

### Tables (`report/tables/`)

| File | Description |
|------|-------------|
| `peak_summary.csv` | Peak GFLOP/s per (partition, version, n), with thread count at peak |
| `peak_summary.md` | Markdown version of the same table, ready for inclusion in reports |

**Sample from `peak_summary.csv`:**
```
partition,version,n,peak_gflops,threads_peak
icelake,0,500,1.00,1
icelake,1,500,3.79,1
icelake,2,500,11.39,16
icelake,3,500,13.36,8
icelake,2,4000,68.45,64
icelake,3,4000,101.48,76
cclake,2,4000,36.68,56
```

---

## Results files in `results/`

| File | Partition | Version | Notes |
|------|-----------|---------|-------|
| `bench_icelake_*.csv` | icelake | V2 | Primary scaling study |
| `bench_icelake_v0_*.csv` | icelake | V0 | Serial baseline, n ≤ 4000 |
| `bench_icelake_v0_n8000_*.csv` | icelake | V0 | n=8000 only |
| `bench_icelake_v1_*.csv` | icelake | V1 | Serial optimised |
| `bench_icelake_v3_*.csv` | icelake | V3 | Column-scaling parallelised |
| `bench_cclake_*.csv` | cclake | V2 | Cross-platform comparison |
| `bench_cclake_numa_close_*.csv` | cclake | V2 | NUMA close binding |
| `bench_cclake_numa_spread_*.csv` | cclake | V2 | NUMA spread binding |
| `compare_v2_*.csv` | icelake | V2 | Back-to-back V2 vs V3 |
| `compare_v3_*.csv` | icelake | V3 | Back-to-back V2 vs V3 |
