#!/usr/bin/env python3
"""
scripts/plot_bench.py
=====================
Benchmark visualisation pipeline for optimisation of Cholesky factorisation.

Reads all CSV files from the results directory, generates plots and a summary
table, and writes outputs to --outdir.

Usage
-----
    python3 scripts/plot_bench.py --results results/ --outdir report/

Requires
--------
    pip install pandas matplotlib        (required)
    pip install scipy                    (optional - enables Amdahl curve fit)

Outputs (inside --outdir)
-------------------------
    figures/gflops_icelake.png          GFLOP/s vs threads, V2, icelake
    figures/gflops_cclake.png           GFLOP/s vs threads, V2, cclake
    figures/speedup_icelake.png         Speedup vs threads, V2, icelake
    figures/speedup_cclake.png          Speedup vs threads, V2, cclake
    figures/efficiency_icelake.png      Parallel efficiency (%) vs threads, V2, icelake
    figures/efficiency_cclake.png       Parallel efficiency (%) vs threads, V2, cclake
    figures/speedup_heatmap_icelake.png Speedup heatmap (n x threads), V2, icelake
    figures/speedup_heatmap_cclake.png  Speedup heatmap (n x threads), V2, cclake
    figures/v3_v2_heatmap.png           V3 vs V2 improvement (%) heatmap, icelake
    figures/amdahl_icelake.png          Speedup + fitted Amdahl curve, icelake
    figures/time_vs_n.png               Wall-clock time vs n (log-log), T=1, all versions
    figures/peak_bar_icelake.png        Peak GFLOP/s bar chart, all versions, icelake
    figures/platform_comparison.png     icelake vs cclake GFLOP/s, V2
    figures/numa_cclake.png             NUMA close vs spread (if data present)
    figures/numa_ratio_cclake.png       NUMA close/spread ratio vs threads
    figures/v0_v1_v2_icelake.png        Single-thread V0/V1/V2/V3 comparison
    figures/v2_v3_comparison.png        V2 vs V3 tuning effect (if data present)
    tables/peak_summary.csv             Peak GFLOP/s per (partition, n, version)
    tables/peak_summary.md              Markdown version of peak summary

CSV format expected
-------------------
    version,n,threads,run,time_s,gflops,hostname
    run == -1 rows are the median-summary rows used for all plots.
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')   # non-interactive backend for HPC/headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("Error: pandas required.  Install: pip install pandas")

try:
    from scipy.optimize import curve_fit as _scipy_curve_fit
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> pd.DataFrame:
    """Load all benchmark CSV files from results_dir into one DataFrame."""
    paths = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files in: {results_dir}")

    frames = []
    for path in paths:
        fname = os.path.basename(path).lower()
        try:
            df = pd.read_csv(path, comment='#')
        except Exception as exc:
            print(f"  Warning: skipping {fname}: {exc}", file=sys.stderr)
            continue

        # Partition from filename
        if 'icelake' in fname:
            df['partition'] = 'icelake'
        elif 'cclake' in fname:
            df['partition'] = 'cclake'
        else:
            df['partition'] = 'unknown'

        # NUMA experiment variant from filename
        if 'numa_close' in fname:
            df['placement'] = 'close'
            df['is_numa'] = True
        elif 'numa_spread' in fname:
            df['placement'] = 'spread'
            df['is_numa'] = True
        else:
            df['placement'] = 'close'
            df['is_numa'] = False

        df['source'] = os.path.basename(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No readable CSV files found.")

    data = pd.concat(frames, ignore_index=True)
    for col in ('version', 'n', 'threads', 'run', 'gflops', 'time_s'):
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data


def median_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only run == -1 (median-summary) rows."""
    return df[df['run'] == -1].copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MARCH = {'icelake': 'icelake-server', 'cclake': 'cascadelake'}
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
VERSION_LABELS = {0: 'V0 baseline', 1: 'V1 serial opt.',
                  2: 'V2 OpenMP', 3: 'V3 col-scale par.'}
VERSION_COLORS = {0: '#d62728', 1: '#ff7f0e', 2: '#1f77b4', 3: '#2ca02c'}
SIZE_COLORS = {500: '#1f77b4', 1000: '#ff7f0e',
               2000: '#2ca02c', 4000: '#d62728', 8000: '#9467bd'}


def _save(fig: plt.Figure, out_dir: str, subdir: str, fname: str) -> None:
    path = os.path.join(out_dir, subdir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _log2_xaxis(ax: plt.Axes) -> None:
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def _gflops_to_time(gflops: float, n: int) -> float:
    """Convert GFLOP/s and n to wall-clock seconds."""
    if gflops <= 0:
        return float('nan')
    return (n ** 3 / 3.0) / (gflops * 1e9)


# ---------------------------------------------------------------------------
# Plot 1 & 2: GFLOP/s and Speedup vs threads for V2
# ---------------------------------------------------------------------------

def plot_gflops_threads(med: pd.DataFrame, partition: str,
                        out_dir: str, version: int = 2,
                        n_highlight: tuple = (2000, 4000, 8000)) -> None:
    """GFLOP/s vs thread count for selected n values."""
    sub = med[(med['partition'] == partition) &
              (med['version'] == version) &
              (~med['is_numa'])]
    if sub.empty:
        print(f"  No data: partition={partition} version={version}", file=sys.stderr)
        return

    ns = [n for n in n_highlight if n in sub['n'].values]
    if not ns:
        ns = sorted(sub['n'].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, n in enumerate(ns):
        d = sub[sub['n'] == n].sort_values('threads')
        ax.plot(d['threads'], d['gflops'],
                marker=MARKERS[i % len(MARKERS)],
                color=SIZE_COLORS.get(n),
                label=f'n={n:,}')

    ax.set_xlabel('Thread count')
    ax.set_ylabel('GFLOP/s  (median of 3 runs)')
    ax.set_title(
        f'V{version} GFLOP/s vs threads - {partition}\n'
        f'GCC 11, -O3, -march={MARCH.get(partition, "native")}, '
        f'OMP_PROC_BIND=close'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    _log2_xaxis(ax)
    _save(fig, out_dir, 'figures', f'gflops_{partition}.png')


def plot_speedup_threads(med: pd.DataFrame, partition: str,
                         out_dir: str, version: int = 2,
                         n_highlight: tuple = (2000, 4000, 8000)) -> None:
    """Parallel speedup (relative to T=1) vs thread count."""
    sub = med[(med['partition'] == partition) &
              (med['version'] == version) &
              (~med['is_numa'])]
    if sub.empty:
        print(f"  No speedup data: partition={partition}", file=sys.stderr)
        return

    ns = [n for n in n_highlight if n in sub['n'].values]
    if not ns:
        ns = sorted(sub['n'].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    t_max = sub['threads'].max()
    ax.plot([1, t_max], [1, t_max], 'k--', lw=1, alpha=0.4, label='Ideal (linear)')

    for i, n in enumerate(ns):
        d = sub[sub['n'] == n].sort_values('threads')
        base = d.loc[d['threads'] == d['threads'].min(), 'gflops'].values
        if len(base) == 0 or base[0] <= 0:
            continue
        speedup = d['gflops'] / base[0]
        ax.plot(d['threads'], speedup,
                marker=MARKERS[i % len(MARKERS)],
                color=SIZE_COLORS.get(n),
                label=f'n={n:,}')

    ax.set_xlabel('Thread count')
    ax.set_ylabel('Speedup  (relative to T=1)')
    ax.set_title(f'V{version} parallel speedup - {partition}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _log2_xaxis(ax)
    _save(fig, out_dir, 'figures', f'speedup_{partition}.png')


# ---------------------------------------------------------------------------
# Plot 3: Single-thread V0 / V1 / V2 / V3 comparison
# ---------------------------------------------------------------------------

def plot_version_comparison(med: pd.DataFrame, out_dir: str,
                             partition: str = 'icelake') -> None:
    """Single-thread GFLOP/s for V0, V1, V2, V3 across n."""
    sub = med[(med['partition'] == partition) &
              (med['threads'] == 1) &
              (~med['is_numa'])]
    if sub.empty:
        print(f"  No single-thread data for {partition}; skipping V comparison.",
              file=sys.stderr)
        return

    versions = sorted(sub['version'].dropna().unique().astype(int))
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, v in enumerate(versions):
        d = sub[sub['version'] == v].sort_values('n')
        if d.empty:
            continue
        label = VERSION_LABELS.get(v, f'V{v}')
        ax.plot(d['n'], d['gflops'],
                marker=MARKERS[i % len(MARKERS)],
                color=VERSION_COLORS.get(v),
                label=label)

    ax.set_xlabel('Matrix dimension  n')
    ax.set_ylabel('GFLOP/s  (median of 3 runs, T=1)')
    ax.set_title(
        f'Single-thread performance: V0 → V1 → V2 → V3\n'
        f'{partition}, GCC 11, -O3, -march={MARCH.get(partition, "native")}'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, 'figures', 'v0_v1_v2_icelake.png')


# ---------------------------------------------------------------------------
# Plot 4: NUMA placement experiment
# ---------------------------------------------------------------------------

def plot_numa(df_all: pd.DataFrame, out_dir: str) -> None:
    """Close vs spread NUMA comparison on cclake, n=2000."""
    sub = df_all[(df_all['partition'] == 'cclake') &
                 (df_all['n'] == 2000) &
                 (df_all['run'] == -1) &
                 (df_all['is_numa'])]
    if sub.empty:
        print("  No NUMA experiment data; skipping numa plot.", file=sys.stderr)
        return

    placements = sorted(sub['placement'].unique())
    if len(placements) < 2:
        print("  Need both close and spread data for NUMA plot; skipping.",
              file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'close': '#1f77b4', 'spread': '#d62728'}
    for p in placements:
        d = sub[sub['placement'] == p].sort_values('threads')
        ax.plot(d['threads'], d['gflops'], marker='o',
                color=colors.get(p, None), label=f'OMP_PROC_BIND={p}')

    ax.axvline(x=28, color='grey', linestyle=':', alpha=0.7,
               label='Socket boundary (28 cores/socket)')

    ax.set_xlabel('Thread count')
    ax.set_ylabel('GFLOP/s  (median of 5 runs)')
    ax.set_title(
        'NUMA placement: OMP_PROC_BIND=close vs spread\n'
        'cclake, n=2000, V2, -march=cascadelake'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, 'figures', 'numa_cclake.png')


# ---------------------------------------------------------------------------
# Plot 5: V2 vs V3 tuning comparison
# ---------------------------------------------------------------------------

def plot_v2_v3(med: pd.DataFrame, out_dir: str,
               partition: str = 'icelake') -> None:
    """V2 vs V3 GFLOP/s comparison for n=2000 and n=4000."""
    sub = med[(med['partition'] == partition) &
              (med['version'].isin([2, 3])) &
              (~med['is_numa']) &
              (med['n'].isin([500, 1000, 2000, 4000]))]
    if sub.empty or sub['version'].nunique() < 2:
        print("  No V2+V3 comparison data; skipping.", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, n in zip(axes, [2000, 4000]):
        d = sub[sub['n'] == n].sort_values('threads')
        for v in [2, 3]:
            dv = d[d['version'] == v]
            if dv.empty:
                continue
            label = VERSION_LABELS.get(v, f'V{v}')
            ax.plot(dv['threads'], dv['gflops'],
                    marker='o', color=VERSION_COLORS.get(v), label=label)
        ax.set_title(f'n={n:,}')
        ax.set_xlabel('Thread count')
        ax.set_ylabel('GFLOP/s')
        ax.legend()
        ax.grid(True, alpha=0.3)
        _log2_xaxis(ax)

    fig.suptitle(
        f'V2 vs V3 (column scaling parallelised) - {partition}\n'
        f'-O3, -march={MARCH.get(partition, "native")}'
    )
    fig.tight_layout()
    _save(fig, out_dir, 'figures', 'v2_v3_comparison.png')


# ---------------------------------------------------------------------------
# Plot 6: Parallel efficiency (%)
# ---------------------------------------------------------------------------

def plot_efficiency(med: pd.DataFrame, partition: str, out_dir: str,
                    version: int = 2,
                    n_highlight: tuple = (500, 1000, 2000, 4000, 8000)) -> None:
    """Parallel efficiency = speedup / T * 100% vs thread count."""
    sub = med[(med['partition'] == partition) &
              (med['version'] == version) &
              (~med['is_numa'])]
    if sub.empty:
        return

    ns = [n for n in n_highlight if n in sub['n'].values]
    if not ns:
        ns = sorted(sub['n'].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(y=100, color='k', linestyle='--', lw=1, alpha=0.4,
               label='Ideal (100%)')

    for i, n in enumerate(ns):
        d = sub[sub['n'] == n].sort_values('threads')
        base = d.loc[d['threads'] == d['threads'].min(), 'gflops'].values
        if len(base) == 0 or base[0] <= 0:
            continue
        speedup = d['gflops'] / base[0]
        efficiency = speedup / d['threads'] * 100.0
        ax.plot(d['threads'], efficiency,
                marker=MARKERS[i % len(MARKERS)],
                color=SIZE_COLORS.get(n),
                label=f'n={n:,}')

    ax.set_xlabel('Thread count')
    ax.set_ylabel('Parallel efficiency  (%)')
    ax.set_title(
        f'V{version} parallel efficiency - {partition}\n'
        f'Efficiency = Speedup / T x 100%'
    )
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _log2_xaxis(ax)
    _save(fig, out_dir, 'figures', f'efficiency_{partition}.png')


# ---------------------------------------------------------------------------
# Plot 7: Speedup heatmap (n x threads)
# ---------------------------------------------------------------------------

def plot_speedup_heatmap(med: pd.DataFrame, partition: str,
                         out_dir: str, version: int = 2) -> None:
    """Heatmap of speedup values with n as rows and thread count as columns."""
    sub = med[(med['partition'] == partition) &
              (med['version'] == version) &
              (~med['is_numa'])]
    if sub.empty:
        return

    # Build pivot: rows = n (descending for top-to-bottom visual), cols = threads
    pivot_data = {}
    ns = sorted(sub['n'].unique())
    all_threads = sorted(sub['threads'].unique())

    for n in ns:
        d = sub[sub['n'] == n].sort_values('threads')
        base = d.loc[d['threads'] == d['threads'].min(), 'gflops'].values
        if len(base) == 0 or base[0] <= 0:
            continue
        row = {}
        for _, r in d.iterrows():
            row[int(r['threads'])] = r['gflops'] / base[0]
        pivot_data[n] = row

    if not pivot_data:
        return

    ns_with_data = sorted(pivot_data.keys(), reverse=True)  # largest n at top
    col_labels = all_threads
    matrix = np.full((len(ns_with_data), len(col_labels)), np.nan)

    for i, n in enumerate(ns_with_data):
        for j, t in enumerate(col_labels):
            if t in pivot_data[n]:
                matrix[i, j] = pivot_data[n][t]

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.8 + 2),
                                    max(4, len(ns_with_data) * 0.7 + 1.5)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=1)

    # Annotate cells
    for i in range(len(ns_with_data)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > matrix[~np.isnan(matrix)].max() * 0.75 else 'black'
                ax.text(j, i, f'{val:.1f}x', ha='center', va='center',
                        fontsize=8, color=text_color, fontweight='bold')

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([str(t) for t in col_labels])
    ax.set_yticks(range(len(ns_with_data)))
    ax.set_yticklabels([f'n={n:,}' for n in ns_with_data])
    ax.set_xlabel('Thread count')
    ax.set_title(f'V{version} speedup over T=1 - {partition}\n(colour = speedup factor)')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Speedup (x)')
    fig.tight_layout()
    _save(fig, out_dir, 'figures', f'speedup_heatmap_{partition}.png')


# ---------------------------------------------------------------------------
# Plot 8: V3 vs V2 improvement heatmap
# ---------------------------------------------------------------------------

def plot_v3_v2_heatmap(med: pd.DataFrame, out_dir: str,
                        partition: str = 'icelake') -> None:
    """Heatmap of V3 improvement over V2 (%) as a function of n and threads."""
    sub = med[(med['partition'] == partition) &
              (med['version'].isin([2, 3])) &
              (~med['is_numa'])]
    if sub.empty or sub['version'].nunique() < 2:
        print("  No V2+V3 data for improvement heatmap; skipping.", file=sys.stderr)
        return

    v2 = sub[sub['version'] == 2].set_index(['n', 'threads'])['gflops']
    v3 = sub[sub['version'] == 3].set_index(['n', 'threads'])['gflops']
    common = v2.index.intersection(v3.index)
    if common.empty:
        return

    pct = ((v3[common] - v2[common]) / v2[common] * 100).reset_index()
    pct.columns = ['n', 'threads', 'pct']

    ns = sorted(pct['n'].unique(), reverse=True)
    ts = sorted(pct['threads'].unique())
    matrix = np.full((len(ns), len(ts)), np.nan)

    for i, n in enumerate(ns):
        for j, t in enumerate(ts):
            row = pct[(pct['n'] == n) & (pct['threads'] == t)]
            if not row.empty:
                matrix[i, j] = row['pct'].values[0]

    # Symmetric colormap centred at 0
    vmax = np.nanmax(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(max(8, len(ts) * 0.8 + 2),
                                    max(4, len(ns) * 0.7 + 1.5)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn',
                   vmin=-vmax, vmax=vmax)

    for i in range(len(ns)):
        for j in range(len(ts)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:+.0f}%', ha='center', va='center',
                        fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([str(t) for t in ts])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([f'n={n:,}' for n in ns])
    ax.set_xlabel('Thread count')
    ax.set_title(
        f'V3 improvement over V2 (%) - {partition}\n'
        f'Green = V3 faster; Red = V3 slower'
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('GFLOP/s change (%)')
    fig.tight_layout()
    _save(fig, out_dir, 'figures', 'v3_v2_heatmap.png')


# ---------------------------------------------------------------------------
# Plot 9: Speedup with Amdahl fit overlay
# ---------------------------------------------------------------------------

def _amdahl_serial_fraction(threads_arr, speedup_arr):
    """
    Estimate Amdahl serial fraction s from (threads, speedup) data.

    If scipy is available, use curve_fit. Otherwise fall back to the
    analytical estimate s = (T/speedup - 1) / (T - 1) at the max-T point.
    """
    def _model(T, s):
        return 1.0 / (s + (1.0 - s) / T)

    if _SCIPY and len(threads_arr) >= 3:
        try:
            popt, _ = _scipy_curve_fit(_model, threads_arr, speedup_arr,
                                       p0=[0.05], bounds=(0, 1))
            return float(popt[0])
        except Exception:
            pass

    # Analytical fallback: use the highest-T measurement
    idx = np.argmax(threads_arr)
    T, S = threads_arr[idx], speedup_arr[idx]
    if T <= 1 or S <= 0:
        return None
    s = (T / S - 1.0) / (T - 1.0)
    return max(0.0, min(1.0, s))


def plot_amdahl(med: pd.DataFrame, partition: str, out_dir: str,
                version: int = 2,
                n_highlight: tuple = (500, 1000, 2000, 4000)) -> None:
    """Speedup vs threads with Amdahl's law curve overlaid."""
    sub = med[(med['partition'] == partition) &
              (med['version'] == version) &
              (~med['is_numa'])]
    if sub.empty:
        return

    ns = [n for n in n_highlight if n in sub['n'].values]
    if not ns:
        ns = sorted(sub['n'].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    t_max = int(sub['threads'].max())
    ax.plot([1, t_max], [1, t_max], 'k--', lw=1, alpha=0.3, label='Ideal (linear)')

    t_fine = np.linspace(1, t_max, 300)

    for i, n in enumerate(ns):
        d = sub[sub['n'] == n].sort_values('threads')
        base = d.loc[d['threads'] == d['threads'].min(), 'gflops'].values
        if len(base) == 0 or base[0] <= 0:
            continue
        speedup = (d['gflops'] / base[0]).values
        threads = d['threads'].values.astype(float)
        color = SIZE_COLORS.get(n)

        # Measured points
        ax.plot(threads, speedup, marker=MARKERS[i % len(MARKERS)],
                color=color, label=f'n={n:,} (measured)')

        # Fitted Amdahl curve
        s = _amdahl_serial_fraction(threads, speedup)
        if s is not None and 0 < s < 1:
            amdahl = 1.0 / (s + (1.0 - s) / t_fine)
            ax.plot(t_fine, amdahl, linestyle=':', color=color, alpha=0.7,
                    label=f'  Amdahl fit  s={s:.3f}')

    ax.set_xlabel('Thread count')
    ax.set_ylabel('Speedup  (relative to T=1)')
    ax.set_title(
        f'V{version} speedup + Amdahl fit - {partition}\n'
        f'Dotted = fitted curve  s = serial fraction'
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _log2_xaxis(ax)
    _save(fig, out_dir, 'figures', f'amdahl_{partition}.png')


# ---------------------------------------------------------------------------
# Plot 10: Wall-clock time vs n (log-log)
# ---------------------------------------------------------------------------

def plot_time_vs_n(med: pd.DataFrame, out_dir: str,
                   partition: str = 'icelake') -> None:
    """
    Wall-clock time vs n for V0, V1 (T=1) and V2/V3 at T=1 and peak T.
    Log-log axes to show O(n^3) scaling.
    """
    sub = med[(med['partition'] == partition) & (~med['is_numa'])]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # V0 and V1 at T=1
    for v in [0, 1]:
        d = sub[(sub['version'] == v) & (sub['threads'] == 1)].sort_values('n')
        if d.empty:
            continue
        times = [_gflops_to_time(g, n) for g, n in zip(d['gflops'], d['n'])]
        ax.plot(d['n'], times,
                marker=MARKERS[v % len(MARKERS)],
                color=VERSION_COLORS.get(v),
                label=f'{VERSION_LABELS[v]}  T=1')

    # V2 and V3 at T=1 and peak T
    for v in [2, 3]:
        dv = sub[sub['version'] == v]
        if dv.empty:
            continue
        # T=1
        d1 = dv[dv['threads'] == 1].sort_values('n')
        if not d1.empty:
            times = [_gflops_to_time(g, n) for g, n in zip(d1['gflops'], d1['n'])]
            ax.plot(d1['n'], times,
                    marker=MARKERS[v % len(MARKERS)],
                    color=VERSION_COLORS.get(v), linestyle='-',
                    label=f'{VERSION_LABELS[v]}  T=1')
        # Peak T (best GFLOP/s per n)
        peak_rows = dv.loc[dv.groupby('n')['gflops'].idxmax()].sort_values('n')
        if not peak_rows.empty:
            times_p = [_gflops_to_time(g, n)
                       for g, n in zip(peak_rows['gflops'], peak_rows['n'])]
            t_labels = ','.join(str(int(t)) for t in peak_rows['threads'])
            ax.plot(peak_rows['n'], times_p,
                    marker=MARKERS[v % len(MARKERS)],
                    color=VERSION_COLORS.get(v), linestyle='--',
                    label=f'{VERSION_LABELS[v]}  peak T ({t_labels})')

    # Reference O(n^3) slope
    ns_ref = np.array([500, 8000])
    t_ref = (ns_ref / 500.0) ** 3
    # Normalise to roughly align with V0 n=500 if available
    v0_500 = sub[(sub['version'] == 0) & (sub['threads'] == 1) & (sub['n'] == 500)]
    if not v0_500.empty:
        base_t = _gflops_to_time(float(v0_500['gflops'].values[0]), 500)
        if not np.isnan(base_t):
            t_ref = t_ref * base_t
            ax.plot(ns_ref, t_ref, 'k:', lw=1, alpha=0.4, label='O(n³) slope')

    ax.set_xscale('log', base=10)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Matrix dimension  n')
    ax.set_ylabel('Wall-clock time  (seconds)')
    ax.set_title(
        f'Wall-clock time vs n - {partition}\n'
        f'Solid = T=1; Dashed = peak thread count'
    )
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3, which='both')
    _save(fig, out_dir, 'figures', 'time_vs_n.png')


# ---------------------------------------------------------------------------
# Plot 11: Peak GFLOP/s bar chart - all versions
# ---------------------------------------------------------------------------

def plot_peak_bar(med: pd.DataFrame, out_dir: str,
                  partition: str = 'icelake') -> None:
    """Grouped bar chart: peak GFLOP/s per version, grouped by n."""
    sub = med[(med['partition'] == partition) & (~med['is_numa'])]
    if sub.empty:
        return

    versions = sorted(sub['version'].dropna().unique().astype(int))
    ns = sorted(sub['n'].unique().astype(int))

    # For V0/V1 (serial) only use T=1; for V2/V3 use best across all T
    def peak(df, v, n):
        dv = df[(df['version'] == v) & (df['n'] == n)]
        if v in (0, 1):
            dv = dv[dv['threads'] == 1]
        if dv.empty:
            return 0.0
        return float(dv['gflops'].max())

    x = np.arange(len(ns))
    width = 0.18
    offsets = np.linspace(-(len(versions) - 1) / 2,
                           (len(versions) - 1) / 2,
                           len(versions)) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for v, offset in zip(versions, offsets):
        heights = [peak(sub, v, n) for n in ns]
        bars = ax.bar(x + offset, heights, width,
                      label=VERSION_LABELS.get(v, f'V{v}'),
                      color=VERSION_COLORS.get(v), alpha=0.85)
        # Label bars with values (skip zeros)
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f'{h:.0f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'n={n:,}' for n in ns])
    ax.set_ylabel('Peak GFLOP/s')
    ax.set_title(
        f'Peak GFLOP/s by version and matrix size - {partition}\n'
        f'V0/V1: T=1 only (serial);  V2/V3: best thread count'
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    _save(fig, out_dir, 'figures', f'peak_bar_{partition}.png')


# ---------------------------------------------------------------------------
# Plot 12: icelake vs cclake platform comparison
# ---------------------------------------------------------------------------

def plot_platform_comparison(med: pd.DataFrame, out_dir: str,
                              version: int = 2,
                              ns: tuple = (2000, 4000)) -> None:
    """GFLOP/s vs threads for icelake and cclake on the same axes."""
    sub = med[(med['version'] == version) & (~med['is_numa'])]
    if sub.empty:
        return

    fig, axes = plt.subplots(1, len(ns), figsize=(6 * len(ns), 5), sharey=False)
    if len(ns) == 1:
        axes = [axes]

    part_colors = {'icelake': '#1f77b4', 'cclake': '#d62728'}
    part_markers = {'icelake': 'o', 'cclake': 's'}

    for ax, n in zip(axes, ns):
        for part in ('icelake', 'cclake'):
            d = sub[(sub['partition'] == part) & (sub['n'] == n)].sort_values('threads')
            if d.empty:
                continue
            ax.plot(d['threads'], d['gflops'],
                    marker=part_markers[part],
                    color=part_colors[part],
                    label=f'{part}  (-march={MARCH.get(part, "native")})')

        ax.set_title(f'n={n:,}')
        ax.set_xlabel('Thread count')
        ax.set_ylabel('GFLOP/s')
        ax.legend()
        ax.grid(True, alpha=0.3)
        _log2_xaxis(ax)

    fig.suptitle(
        f'Platform comparison: icelake vs cclake - V{version}\n'
        f'icelake: 2x38 cores (76 total);  cclake: 2x28 cores (56 total)'
    )
    fig.tight_layout()
    _save(fig, out_dir, 'figures', 'platform_comparison.png')


# ---------------------------------------------------------------------------
# Plot 13: NUMA close/spread ratio
# ---------------------------------------------------------------------------

def plot_numa_ratio(df_all: pd.DataFrame, out_dir: str) -> None:
    """Ratio close/spread GFLOP/s vs thread count on cclake, n=2000."""
    sub = df_all[(df_all['partition'] == 'cclake') &
                 (df_all['n'] == 2000) &
                 (df_all['run'] == -1) &
                 (df_all['is_numa'])]
    if sub.empty:
        return

    close = sub[sub['placement'] == 'close'].set_index('threads')['gflops']
    spread = sub[sub['placement'] == 'spread'].set_index('threads')['gflops']
    common_t = close.index.intersection(spread.index)
    if common_t.empty:
        return

    ratio = (close[common_t] / spread[common_t]).reset_index()
    ratio.columns = ['threads', 'ratio']
    ratio = ratio.sort_values('threads')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(y=1.0, color='k', linestyle='--', lw=1, alpha=0.5,
               label='Equal performance')
    ax.fill_between(ratio['threads'], 1.0, ratio['ratio'],
                    where=(ratio['ratio'] >= 1.0), alpha=0.15, color='#1f77b4',
                    label='close faster')
    ax.fill_between(ratio['threads'], ratio['ratio'], 1.0,
                    where=(ratio['ratio'] < 1.0), alpha=0.15, color='#d62728',
                    label='spread faster')
    ax.plot(ratio['threads'], ratio['ratio'], 'o-', color='#1f77b4')

    ax.axvline(x=28, color='grey', linestyle=':', alpha=0.7,
               label='Socket boundary (28 cores)')

    ax.set_xlabel('Thread count')
    ax.set_ylabel('GFLOP/s ratio  (close / spread)')
    ax.set_title(
        'NUMA close/spread ratio - cclake, n=2000, V2\n'
        'Values >1: close wins;  <1: spread wins'
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, 'figures', 'numa_ratio_cclake.png')


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def generate_summary(med: pd.DataFrame, out_dir: str) -> None:
    """Write peak GFLOP/s summary as CSV and Markdown."""
    non_numa = med[~med['is_numa']]
    if non_numa.empty:
        print("  No data for summary table.", file=sys.stderr)
        return

    rows = []
    for (part, n, ver), grp in non_numa.groupby(['partition', 'n', 'version']):
        best = grp.loc[grp['gflops'].idxmax()]
        rows.append({
            'partition':    part,
            'version':      int(ver),
            'n':            int(n),
            'peak_gflops':  round(float(best['gflops']), 2),
            'threads_peak': int(best['threads']),
        })

    rows.sort(key=lambda r: (r['partition'], r['version'], r['n']))

    csv_path = os.path.join(out_dir, 'tables', 'peak_summary.csv')
    fields = ['partition', 'version', 'n', 'peak_gflops', 'threads_peak']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {csv_path}")

    md_path = os.path.join(out_dir, 'tables', 'peak_summary.md')
    with open(md_path, 'w') as f:
        f.write('| Partition | Version | n | Peak GFLOP/s | Threads at peak |\n')
        f.write('|-----------|---------|--:|-------------:|----------------:|\n')
        for r in rows:
            f.write(f"| {r['partition']} | V{r['version']} | {r['n']:,} | "
                    f"{r['peak_gflops']:.2f} | {r['threads_peak']} |\n")
    print(f"  Saved: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MPhil DIS C2 Cholesky benchmark plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--results', default='results',
                        help='Directory containing CSV files  (default: results/)')
    parser.add_argument('--outdir', default='report',
                        help='Root output directory  (default: report/)')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.outdir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'tables'),  exist_ok=True)

    print(f"Loading CSVs from: {args.results}/")
    if not _SCIPY:
        print("  Note: scipy not found - Amdahl fit will use analytical fallback.")
    try:
        data = load_results(args.results)
    except FileNotFoundError as exc:
        sys.exit(f"Error: {exc}")

    med = median_rows(data)
    versions_found = sorted(data['version'].dropna().unique().astype(int))
    partitions_found = sorted(data['partition'].unique())
    print(f"  {len(data)} total rows, {len(med)} median rows, "
          f"{data['source'].nunique()} file(s)")
    print(f"  Versions  : {versions_found}")
    print(f"  Partitions: {partitions_found}")

    print("\n[ 1/13] GFLOP/s vs threads...")
    for part in ('icelake', 'cclake'):
        plot_gflops_threads(med, part, args.outdir)

    print("[ 2/13] Speedup vs threads...")
    for part in ('icelake', 'cclake'):
        plot_speedup_threads(med, part, args.outdir)

    print("[ 3/13] Parallel efficiency...")
    for part in ('icelake', 'cclake'):
        plot_efficiency(med, part, args.outdir)

    print("[ 4/13] Speedup heatmap...")
    for part in ('icelake', 'cclake'):
        plot_speedup_heatmap(med, part, args.outdir)

    print("[ 5/13] V3/V2 improvement heatmap...")
    plot_v3_v2_heatmap(med, args.outdir)

    print("[ 6/13] Amdahl fit overlay...")
    for part in ('icelake', 'cclake'):
        plot_amdahl(med, part, args.outdir)

    print("[ 7/13] Wall-clock time vs n (log-log)...")
    plot_time_vs_n(med, args.outdir)

    print("[ 8/13] Peak GFLOP/s bar chart...")
    for part in ('icelake', 'cclake'):
        plot_peak_bar(med, args.outdir, partition=part)

    print("[ 9/13] Platform comparison (icelake vs cclake)...")
    plot_platform_comparison(med, args.outdir)

    print("[10/13] V0/V1/V2/V3 single-thread comparison...")
    plot_version_comparison(med, args.outdir)

    print("[11/13] NUMA placement comparison...")
    plot_numa(data, args.outdir)

    print("[12/13] NUMA close/spread ratio...")
    plot_numa_ratio(data, args.outdir)

    print("[13/13] V2 vs V3 tuning comparison...")
    plot_v2_v3(med, args.outdir)

    print("[14/13] Peak summary table...")
    generate_summary(med, args.outdir)

    print(f"\nAll outputs written to: {args.outdir}/")
    print("  figures/ :")
    outputs = [
        "gflops_{icelake,cclake}.png",
        "speedup_{icelake,cclake}.png",
        "efficiency_{icelake,cclake}.png",
        "speedup_heatmap_{icelake,cclake}.png",
        "v3_v2_heatmap.png",
        "amdahl_{icelake,cclake}.png",
        "time_vs_n.png",
        "peak_bar_{icelake,cclake}.png",
        "platform_comparison.png",
        "v0_v1_v2_icelake.png",
        "numa_cclake.png",
        "numa_ratio_cclake.png",
        "v2_v3_comparison.png",
    ]
    for o in outputs:
        print(f"    {o}")
    print("  tables/  : peak_summary.{csv,md}")


if __name__ == '__main__':
    main()
