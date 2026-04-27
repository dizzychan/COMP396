#!/usr/bin/env python3
"""
Bollinger + RSI Strategy Hyperparameter Optimization
Grid search over RSI buy/sell thresholds, evaluated by PD Ratio.
"""

import os
import subprocess
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/Users/dizzychan/Desktop/BT396_Version_0.1.0")
TRAINING_DATA_DIR = BASE_DIR / "DATA" / "PART1" / "training"
OUTPUT_BASE_DIR = BASE_DIR / "tests"
MAIN_PY = BASE_DIR / "main.py"

RSI_BUY_VALS = [10, 20, 25, 30, 35, 40]
RSI_SELL_VALS = [60, 65, 70, 75, 80, 90]

FIXED_PARAMS = {
    'n': 20,
    'stake': 10,
    'k': 2.5,
    'allow_short': True,
    'allow_RSI': True,
    'RSI_period': 14,
    'exit_rule': True,
    'Stop_loss': 0.10,
    'take_profit': 0.20
}


def create_output_dir(rsi_buy, rsi_sell, timestamp):
    """Create a dedicated output directory for each parameter combination."""
    dir_name = f"RSI_buy{rsi_buy}_sell{rsi_sell}_{timestamp}"
    output_dir = OUTPUT_BASE_DIR / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_backtest(rsi_buy, rsi_sell, output_dir):
    """
    Run a single backtest.

    Args:
        rsi_buy: RSI buy threshold
        rsi_sell: RSI sell threshold
        output_dir: directory to write backtest output

    Returns:
        dict: performance metrics including PD Ratio, or None on failure
    """
    print(f"\n{'=' * 60}")
    print(f"Running backtest: RSI_buy={rsi_buy}, RSI_sell={rsi_sell}")
    print(f"Output dir: {output_dir}")
    print(f"{'=' * 60}")

    cmd = [
        "python3",
        str(MAIN_PY),
        "--strategy", "BB",
        "--data-dir", str(TRAINING_DATA_DIR),
        "--output-dir", str(output_dir),
        "--no-plot",
    ]

    cmd.extend(["--param", f"RSI_buy={rsi_buy}"])
    cmd.extend(["--param", f"RSI_sell={rsi_sell}"])

    for key, value in FIXED_PARAMS.items():
        if isinstance(value, bool):
            cmd.extend(["--param", f"{key}={str(value).lower()}"])
        else:
            cmd.extend(["--param", f"{key}={value}"])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(BASE_DIR)
        )

        if result.returncode != 0:
            print(f"ERROR: Backtest failed!")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return None

        metrics = parse_backtest_output(output_dir)

        if metrics:
            print(f"✓ Success! PD Ratio: {metrics.get('pd_ratio', 'N/A'):.4f}")
        else:
            print(f"✗ Failed to parse metrics")

        return metrics

    except subprocess.TimeoutExpired:
        print(f"ERROR: Backtest timed out after 300 seconds")
        return None
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def parse_backtest_output(output_dir):
    """
    Parse backtest results from the output directory.
    Reads run_summary.json as written by main.py.

    Args:
        output_dir: backtest output directory

    Returns:
        dict: performance metrics, or None if parsing fails
    """
    summary_file = output_dir / "run_summary.json"

    if not summary_file.exists():
        print(f"WARNING: run_summary.json not found in {output_dir}")
        return None

    try:
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        metrics = {
            'pd_ratio': summary.get('pd_ratio_portfolio'),
            'final_value': summary.get('final_value'),
            'bankrupt': summary.get('bankrupt', False),
            'activity_pct': summary.get('activity_pct')
        }

        return metrics

    except Exception as e:
        print(f"ERROR parsing run_summary.json: {str(e)}")
        return None


def run_grid_search():
    """Execute the full grid search over all RSI_buy / RSI_sell combinations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    main_output_dir = OUTPUT_BASE_DIR / f"optimization_{timestamp}"
    main_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"# Starting Grid Search Optimization")
    print(f"# Timestamp: {timestamp}")
    print(f"# Output Directory: {main_output_dir}")
    print(f"# Training Data: {TRAINING_DATA_DIR}")
    print(f"# Total Combinations: {len(RSI_BUY_VALS)} x {len(RSI_SELL_VALS)} = {len(RSI_BUY_VALS) * len(RSI_SELL_VALS)}")
    print(f"{'#' * 60}\n")

    results = []
    total = len(RSI_BUY_VALS) * len(RSI_SELL_VALS)
    current = 0

    for rsi_buy in RSI_BUY_VALS:
        for rsi_sell in RSI_SELL_VALS:
            current += 1
            print(f"\n[{current}/{total}] Testing RSI_buy={rsi_buy}, RSI_sell={rsi_sell}")

            if rsi_buy >= rsi_sell:
                print(f"⚠ Skipping invalid combination: RSI_buy ({rsi_buy}) >= RSI_sell ({rsi_sell})")
                results.append({
                    'RSI_buy': rsi_buy,
                    'RSI_sell': rsi_sell,
                    'pd_ratio': np.nan,
                    'final_value': np.nan,
                    'bankrupt': True,
                    'activity_pct': np.nan,
                    'output_dir': 'N/A (invalid params)'
                })
                continue

            output_dir = create_output_dir(rsi_buy, rsi_sell, timestamp)
            metrics = run_backtest(rsi_buy, rsi_sell, output_dir)

            if metrics and metrics.get('pd_ratio') is not None:
                results.append({
                    'RSI_buy': rsi_buy,
                    'RSI_sell': rsi_sell,
                    'pd_ratio': metrics['pd_ratio'],
                    'final_value': metrics.get('final_value', np.nan),
                    'bankrupt': metrics.get('bankrupt', False),
                    'activity_pct': metrics.get('activity_pct', np.nan),
                    'output_dir': str(output_dir)
                })
            else:
                results.append({
                    'RSI_buy': rsi_buy,
                    'RSI_sell': rsi_sell,
                    'pd_ratio': np.nan,
                    'final_value': np.nan,
                    'bankrupt': True,
                    'activity_pct': np.nan,
                    'output_dir': str(output_dir)
                })

    df_results = pd.DataFrame(results)
    results_csv = main_output_dir / "optimization_results.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to: {results_csv}")

    plot_heatmap(df_results, main_output_dir)
    print_best_parameters(df_results)

    return df_results, main_output_dir


def plot_heatmap(df_results, output_dir):
    """Plot a PD Ratio heatmap across all RSI_buy / RSI_sell combinations."""
    print("\nGenerating heatmap...")

    pivot_table = df_results.pivot(
        index='RSI_sell',
        columns='RSI_buy',
        values='pd_ratio'
    )

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'PD Ratio'},
        linewidths=0.5,
        linecolor='gray',
        square=True
    )

    plt.title('Bollinger + RSI Strategy: PD Ratio Heatmap\n(RSI Buy vs RSI Sell Thresholds)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('RSI_buy (Buy Threshold)', fontsize=13, fontweight='bold')
    plt.ylabel('RSI_sell (Sell Threshold)', fontsize=13, fontweight='bold')

    plt.text(0.5, -0.15,
             'Green = Positive PD Ratio (Better) | Red = Negative PD Ratio (Worse)',
             ha='center', transform=plt.gca().transAxes,
             fontsize=11, style='italic', color='gray')

    plt.tight_layout()

    heatmap_file = output_dir / "pd_ratio_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {heatmap_file}")

    plt.close()


def print_best_parameters(df_results):
    """Print ranked parameter combinations sorted by PD Ratio."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    df_sorted = df_results.dropna(subset=['pd_ratio']).sort_values('pd_ratio', ascending=False)

    if len(df_sorted) == 0:
        print("⚠ No valid results found!")
        return

    print("\n🏆 TOP 10 PARAMETER COMBINATIONS (by PD Ratio):")
    print("-" * 60)

    for i, (idx, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        print(f"\nRank {i}:")
        print(f"  RSI_buy: {int(row['RSI_buy'])}, RSI_sell: {int(row['RSI_sell'])}")
        print(f"  PD Ratio: {row['pd_ratio']:.4f}")
        if not pd.isna(row.get('final_value')):
            profit = row['final_value'] - 1_000_000
            pct = (profit / 1_000_000) * 100
            print(f"  Final Value: ${row['final_value']:,.2f} ({pct:+.2f}%)")
        if not pd.isna(row.get('activity_pct')):
            print(f"  Activity: {row['activity_pct']:.2f}%")
        if row.get('bankrupt'):
            print(f"  ⚠ BANKRUPT")

    print("\n" + "=" * 60)

    best = df_sorted.iloc[0]
    print("\n🎯 BEST PARAMETERS:")
    print(f"  RSI_buy = {int(best['RSI_buy'])}")
    print(f"  RSI_sell = {int(best['RSI_sell'])}")
    print(f"  PD Ratio = {best['pd_ratio']:.4f}")
    if not pd.isna(best.get('final_value')):
        profit = best['final_value'] - 1_000_000
        pct = (profit / 1_000_000) * 100
        print(f"  Final Value = ${best['final_value']:,.2f} ({pct:+.2f}%)")

    print("\n📊 STATISTICS:")
    print(f"  Valid combinations: {len(df_sorted)}")
    print(f"  Mean PD Ratio: {df_sorted['pd_ratio'].mean():.4f}")
    print(f"  Std PD Ratio: {df_sorted['pd_ratio'].std():.4f}")
    print(f"  Positive PD Ratios: {(df_sorted['pd_ratio'] > 0).sum()} / {len(df_sorted)}")

    print("=" * 60 + "\n")


def main():
    """Entry point — validates paths then runs the grid search."""
    if not TRAINING_DATA_DIR.exists():
        print(f"ERROR: Training data directory not found: {TRAINING_DATA_DIR}")
        print(f"Please check if the path exists: {TRAINING_DATA_DIR}")
        return

    if not MAIN_PY.exists():
        print(f"ERROR: main.py not found: {MAIN_PY}")
        return

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Configuration check:")
    print(f"  Base Directory: {BASE_DIR}")
    print(f"  Training Data: {TRAINING_DATA_DIR} (exists: {TRAINING_DATA_DIR.exists()})")
    print(f"  Main Script: {MAIN_PY} (exists: {MAIN_PY.exists()})")
    print(f"  Output Directory: {OUTPUT_BASE_DIR}")

    try:
        df_results, output_dir = run_grid_search()

        print("\n" + "=" * 60)
        print("✓ Optimization completed successfully!")
        print(f"✓ Results saved to: {output_dir}")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Optimization interrupted by user")

    except Exception as e:
        print(f"\n✗ ERROR during optimization: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()