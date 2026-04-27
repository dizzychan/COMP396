# opt_main.py — Bollinger + RSI Grid Search Optimizer

Runs an exhaustive grid search over RSI buy and sell thresholds for a Bollinger Band + RSI strategy. Each combination is evaluated by launching `main.py` as a subprocess and reading the resulting `run_summary.json`. Results are ranked by **PD Ratio** and visualised as a heatmap.

---

## Requirements

- Python 3.8+
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- A working `main.py` at `BASE_DIR` that accepts `--strategy`, `--data-dir`, `--output-dir`, `--no-plot`, and `--param` arguments and writes `run_summary.json` to the output directory

---

## Quick Start

```bash
python3 opt_main.py
```

No CLI arguments are required. All paths and search-space values are configured directly at the top of the script.

---

## Configuration (edit at top of script)

### Paths

| Constant | Default | Description |
|---|---|---|
| `BASE_DIR` | `/Users/dizzychan/Desktop/BT396_Version_0.1.0` | Project root directory |
| `TRAINING_DATA_DIR` | `BASE_DIR/DATA/PART1/training` | Directory containing training CSV files |
| `OUTPUT_BASE_DIR` | `BASE_DIR/tests` | Root directory for all optimizer output |
| `MAIN_PY` | `BASE_DIR/main.py` | Path to the backtest entry-point script |

### Search Space

```python
RSI_BUY_VALS  = [10, 20, 25, 30, 35, 40]   # RSI buy threshold candidates
RSI_SELL_VALS = [60, 65, 70, 75, 80, 90]   # RSI sell threshold candidates
```

The search runs all `len(RSI_BUY_VALS) × len(RSI_SELL_VALS)` combinations. Combinations where `RSI_buy >= RSI_sell` are automatically skipped as invalid.

### Fixed Parameters

Parameters held constant across all runs:

| Parameter | Value | Description |
|---|---|---|
| `n` | 20 | Bollinger Band period |
| `stake` | 10 | Position size (units) |
| `k` | 2.5 | Bollinger Band width multiplier |
| `allow_short` | True | Enable short selling |
| `allow_RSI` | True | Enable RSI filter |
| `RSI_period` | 14 | RSI calculation period |
| `exit_rule` | True | Enable exit rule |
| `Stop_loss` | 0.10 | Stop-loss threshold (10%) |
| `take_profit` | 0.20 | Take-profit threshold (20%) |

---

## Output Files

All output is written under `OUTPUT_BASE_DIR/optimization_<timestamp>/`.

| File | Description |
|---|---|
| `optimization_results.csv` | One row per parameter combination: `RSI_buy`, `RSI_sell`, `pd_ratio`, `final_value`, `bankrupt`, `activity_pct`, `output_dir` |
| `pd_ratio_heatmap.png` | 2D heatmap of PD Ratio with `RSI_buy` on the x-axis and `RSI_sell` on the y-axis (green = higher, red = lower) |

Each individual backtest also writes its own sub-directory under `OUTPUT_BASE_DIR`, named `RSI_buy<X>_sell<Y>_<timestamp>/`, containing whatever `main.py` produces (including `run_summary.json`).

---

## How It Works

1. **Path validation** — `main()` checks that `TRAINING_DATA_DIR` and `MAIN_PY` exist before starting.
2. **Grid search** — `run_grid_search()` iterates over every `(RSI_buy, RSI_sell)` pair. Invalid pairs (`buy >= sell`) are recorded as `NaN` and skipped.
3. **Subprocess backtest** — `run_backtest()` builds a `python3 main.py ...` command, passes all fixed and variable parameters via `--param key=value`, and runs it with a 300-second timeout.
4. **Result parsing** — `parse_backtest_output()` reads `run_summary.json` from the run's output directory and extracts `pd_ratio_portfolio`, `final_value`, `bankrupt`, and `activity_pct`.
5. **Ranking** — `print_best_parameters()` sorts all valid results by PD Ratio descending and prints the top 10 plus summary statistics.
6. **Heatmap** — `plot_heatmap()` pivots the results into a 2D table and renders a colour-coded heatmap saved as a 300 dpi PNG.

---

## Extending the Search Space

To add more RSI values or sweep additional parameters, edit the constants at the top of the script:

```python
# Widen the RSI grid
RSI_BUY_VALS  = [5, 10, 15, 20, 25, 30, 35, 40, 45]
RSI_SELL_VALS = [55, 60, 65, 70, 75, 80, 85, 90]

# Override a fixed parameter
FIXED_PARAMS['Stop_loss'] = 0.05
```

To search over a third parameter (e.g. `k`), add an outer loop in `run_grid_search()` and update `create_output_dir()` to include it in the directory name.
