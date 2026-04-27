# walkforward_optimize_trade_based.py — Trade-Count Walk-Forward Optimizer

A walk-forward optimization script for **low-frequency trading strategies**, where windows are defined by a **fixed number of trades** rather than calendar time. This avoids the common problem of time-based windows containing vastly different numbers of trades for infrequent strategies.

---

## Requirements

- Python 3.8+
- `numpy`, `pandas`, `backtrader`
- Internal framework modules: `framework.data_loader`, `framework.strategies_loader`, `framework.analyzers`, `framework.strategy_base`
- **Strategy requirement**: the strategy class must record each entry signal date into `self.signal_dates` inside its `next()` method — this list drives window generation.

---

## Usage

```bash
python walkforward_optimize_trade_based.py --config <CONFIG_FILE> [OPTIONS]
```

### Required Arguments

| Argument | Description |
|---|---|
| `--config` | Path to the JSON configuration file (default: `params_config.json`) |

### Optional Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | *(hardcoded path)* | Directory containing the CSV data files |
| `--output-dir` | *(hardcoded path)* | Directory to write output files |
| `--s-mult` | `2.0` | Slippage multiplier passed to `COMP396BrokerConfig` |
| `--commission` | `0.0` | Commission rate (set `0.0` for no commission) |
| `--cash` | `1,000,000` | Starting portfolio cash |
| `--seed` | `42` | Random seed for reproducibility |

---

## Examples

```bash
# Minimal run using default paths from the config
python walkforward_optimize_trade_based.py --config params_config.json

# Custom data and output directories
python walkforward_optimize_trade_based.py \
    --config params_config.json \
    --data-dir ./DATA/Cleaned_Data \
    --output-dir ./output/walkforward_run1

# Custom slippage and starting cash
python walkforward_optimize_trade_based.py \
    --config params_config.json \
    --s-mult 1.5 \
    --cash 500000

# Fixed random seed for reproducible results
python walkforward_optimize_trade_based.py \
    --config params_config.json \
    --seed 123
```

---

## Configuration File (`params_config.json`)

The JSON config file controls window sizing, fixed parameters, and the multi-stage optimization search space.

### Top-level structure

```json
{
  "window": { ... },
  "fixed_params": { ... },
  "stages": [ ... ]
}
```

### `window` block

Defines how walk-forward windows are constructed from signal dates.

```json
"window": {
  "train_trades": 40,
  "test_trades": 10,
  "buffer_days": 5
}
```

| Key | Description |
|---|---|
| `train_trades` | Number of entry signals in the training window |
| `test_trades` | Number of entry signals in the test window |
| `buffer_days` | Extra calendar days padded around each window's start and end dates |

Windows are **non-overlapping**: each successive window advances by `train_trades` signals.

### `fixed_params` block

Parameters that are held constant across all windows and stages — not optimized.

```json
"fixed_params": {
  "printlog": false,
  "risk_budget_pct": 0.5
}
```

### `stages` block

A list of sequential optimization stages. Each stage searches a subset of parameters independently, using the best result from the previous stage as its starting point.

```json
"stages": [
  {
    "name": "Risk control",
    "iterations": 100,
    "params": {
      "risk_per_trade_cap": { "type": "float", "low": 0.02, "high": 0.10, "step": 0.005 },
      "risk_atr_period":    { "type": "int",   "low": 14,   "high": 42,   "step": 2    }
    }
  },
  {
    "name": "Exit multipliers",
    "iterations": 80,
    "params": {
      "take_profit_mult": { "type": "float", "low": 2.5, "high": 6.0, "step": 0.25 },
      "stop_loss_mult":   { "type": "float", "low": 1.0, "high": 3.0, "step": 0.125 }
    }
  }
]
```

**Parameter definition fields:**

| Field | Required | Description |
|---|---|---|
| `type` | Yes | `"int"`, `"float"`, or `"choice"` |
| `low` | For `int`/`float` | Lower bound of the search range |
| `high` | For `int`/`float` | Upper bound of the search range |
| `step` | Optional | Grid step size; if omitted for `float`, samples uniformly at random |
| `choices` | For `choice` | List of discrete values to sample from |

---

## Output Files

All output files are written to the directory specified by `--output-dir`.

| File | Description |
|---|---|
| `window_results.csv` | Per-window results: dates, test metrics, and best parameters found for each window. Updated incrementally after every window completes. |
| `robust_params.json` | Final robust parameter set: the **median** of each parameter's per-window best value, with correct type casting (`int` → rounded, `float` → 4 decimal places). |

---

## How It Works

**Step 1 — Collect signal dates**
A baseline backtest is run using `fixed_params` to collect all entry signal dates from `strategy.signal_dates`. These dates drive window construction instead of calendar time.

**Step 2 — Generate windows**
Windows are created by slicing signal dates into consecutive, non-overlapping blocks of `train_trades` + `test_trades` signals each. A `buffer_days` calendar padding is added around each window boundary to ensure enough OHLCV data is loaded.

**Step 3 — Multi-stage optimization per window**
For each window, stages run in sequence. Within each stage:
- `iterations` parameter sets are sampled randomly (with step-grid snapping).
- All evaluations run in **parallel** using `ProcessPoolExecutor` (all available CPU cores).
- The primary metric is `true_pd_ratio`; `final_value` is used as a tiebreaker.
- The best parameter set from each stage seeds the next stage.

**Step 4 — Robust parameter aggregation**
After all windows are processed, the **median** of each optimized parameter across windows is computed and saved as `robust_params.json`. This median aggregation reduces sensitivity to any single window's idiosyncratic market regime.

---

## Notes

- **Strategy compatibility**: the strategy loaded by `run_backtest` is hardcoded to `"v8" / "TeamStrategy"`, while the baseline signal-date run uses `"v6" / "TeamStrategy"`. Update these `load_strategy_class` calls to match your actual strategy version.
- **Parallelism**: the evaluation function `_evaluate_params` is defined at module top level (not inside `main`) specifically to support `pickle`-based multiprocessing on all platforms including macOS/Windows.
- **CSV data**: the backtest loads up to 10 CSV files from `--data-dir`, sorted alphabetically.
