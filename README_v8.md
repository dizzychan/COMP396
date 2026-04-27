# optimize_v8.py — V8 Strategy Optimizer

A parameter optimization tool for trading strategies using Latin Hypercube Sampling, multi-stage candidate filtering, and strict generalization gap controls to produce robust, out-of-sample parameters.

---

## Requirements

- Python 3.8+
- `pandas`
- `backtrader`
- Internal framework modules: `framework.data_loader`, `framework.analyzers`, `framework.strategy_base`, `framework.strategies_loader`

---

## Usage

```bash
python optimize_v8.py --train-dir <TRAIN_DATA_DIR> --val-dir <VAL_DATA_DIR> [OPTIONS]
```

### Required Arguments

| Argument | Description |
|---|---|
| `--train-dir` | Path to the training data directory (e.g. `./DATA/PART1`) |
| `--val-dir` | Path to the validation data directory (e.g. `./DATA/PART2`) |

### Optional Arguments

| Argument | Default | Description |
|---|---|---|
| `--n-trials` | `3000` | Total number of random search trials |
| `--workers` | all CPUs | Number of parallel worker processes |
| `--output-dir` | `./opt_output` | Directory to write output files |
| `--seed` | random | Random seed for reproducibility |
| `--no-reverse` | — | Flag to skip reverse validation (already permanently disabled; kept for compatibility) |

---

## Examples

```bash
# Standard run
python optimize_v8.py --train-dir ./DATA/PART1 --val-dir ./DATA/PART2

# Custom number of search trials
python optimize_v8.py --train-dir ./DATA/PART1 --val-dir ./DATA/PART2 --n-trials 2000

# Fixed random seed for reproducibility
python optimize_v8.py --train-dir ./DATA/PART1 --val-dir ./DATA/PART2 --seed 42

# Limit parallel workers
python optimize_v8.py --train-dir ./DATA/PART1 --val-dir ./DATA/PART2 --workers 4

# Custom output directory
python optimize_v8.py --train-dir ./DATA/PART1 --val-dir ./DATA/PART2 --output-dir ./results/run1
```

---

## Output Files

All outputs are written to `./opt_output/` (or the directory specified by `--output-dir`).

| File | Description |
|---|---|
| `best_params.json` | Final best parameters (JSON, ready for strategy use) |
| `best_params_forward.json` | Forward-pass best parameters (reserved for future use) |
| `best_params_reverse.json` | Reverse-pass best parameters (reserved for future use) |
| `optimization_log.txt` | Full run log including all phase results and scoring details |

---

## How It Works

The optimizer runs in three sequential phases per direction:

**Phase 1 — Global Search (Training Set)**
Latin Hypercube Sampling generates `N_TRIALS` parameter sets spread evenly across the search space. Each is backtested on the training set and scored using the COMP396 rubric (Return 0–11 pts + PD Ratio 0–11 pts, max 22).

**Phase 2 — Candidate Filtering + Robustness Check**
The top `TOP_CANDIDATE_PCT` (default 15%) of valid training results form a candidate pool. Each candidate undergoes `ROBUSTNESS_N_PERTURB` joint multi-dimensional perturbations (±15% noise). Candidates whose score drops more than `ROBUSTNESS_MAX_DROP` (20%) in fewer than `ROBUSTNESS_MIN_PASS` (70%) of perturbations are eliminated as parameter islands.

**Phase 3 — Validation Set Scoring**
Surviving candidates are backtested on the validation set. Two hard filters apply:
- Eliminated if COMP score degrades more than `GENGAP_THRESHOLD` (50%) relative to training
- Eliminated if PD Ratio degrades more than `GENGAP_HARD_PD_DROP` (35%) relative to training

A soft penalty further reduces the score proportionally to the generalization gap. The candidate with the highest adjusted validation score is selected as the final result.

---

## Data Sequence & Reverse Validation

The expected data sequence is `PART1 → PART2 → PART3`:

- **PART1** — Training set
- **PART2** — Validation set (acts as the strongest generalization proxy for PART3)
- **PART3** — Final scoring set (never seen during optimization)

**Reverse validation is permanently disabled.** Training on PART2 and validating on PART1 introduces data leakage — the optimizer would implicitly embed future information into selected parameters, causing them to fail on PART3.

---

## Key Configuration (edit at top of script)

| Constant | Default | Description |
|---|---|---|
| `N_TRIALS` | `3000` | Global number of LHS search trials |
| `TOP_CANDIDATE_PCT` | `0.15` | Fraction of valid training results entering candidate pool |
| `MIN_TRADES` | `25` | Minimum closed trades required; below this a trial is eliminated |
| `N_WORKERS` | `None` (all CPUs) | Parallel worker count |
| `STARTING_CASH` | `1,000,000` | Starting portfolio cash for backtests |
| `ROBUSTNESS_NOISE_PCT` | `0.15` | Perturbation noise range (±15% of parameter range) |
| `ROBUSTNESS_MAX_DROP` | `0.20` | Maximum tolerated score drop per perturbation |
| `ROBUSTNESS_MIN_PASS` | `0.70` | Minimum perturbation pass rate to retain a candidate |
| `GENGAP_THRESHOLD` | `0.50` | Max COMP score degradation before hard elimination |
| `GENGAP_HARD_PD_DROP` | `0.35` | Max PD Ratio degradation before hard elimination |
| `GENGAP_PENALTY_SCALE` | `0.20` | Soft penalty multiplier for generalization gap |

---

## COMP396 Scoring Reference

| Metric | Condition | Points |
|---|---|---|
| **Return** | ≥ 30% | 11 |
| | ≥ 15% | 8 |
| | ≥ 0% | 4 |
| | < 0% | 0 |
| **PD Ratio** | ≥ 3.0 | 11 |
| | ≥ 2.0 | 8 |
| | ≥ 0.5 | 4 |
| | < 0.5 | 0 |
| **Total** | | max 22 |
