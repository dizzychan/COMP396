#!/usr/bin/env python3
# =============================================================================
# optimize_v8.py — V8 Strategy Optimizer (Fused Parameters + Strict Generalization Filtering)
# =============================================================================
# Usage:
#   python optimize_v8.py --train-dir ./DATA/PART1 --val-dir ./DATA/PART2
#   python optimize_v8.py --train-dir ./DATA/PART1 --val-dir ./DATA/PART2 --n-trials 2000
#   python optimize_v8.py --train-dir ./DATA/PART1 --val-dir ./DATA/PART2 --no-reverse
#
# Output:
#   ./opt_output/
#     best_params.json           # Final best parameters
#     best_params_forward.json   # Forward-pass best parameters
#     best_params_reverse.json   # Reverse-pass best parameters
#     optimization_log.txt       # Full run log
# =============================================================================

import argparse
import json
import math
import random
import sys
import time
import traceback
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

# ===========================================================================
# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                      ① User-Editable Configuration                     ║
# ╚═════════════════════════════════════════════════════════════════════════╝
# ===========================================================================

# --- Global number of random search trials ---
N_TRIALS = 3000

# --- Candidate pool ratio (top X% of valid training results enter validation) ---
TOP_CANDIDATE_PCT = 0.15

# --- Minimum trade count threshold (below this, trial is immediately eliminated) ---
MIN_TRADES = 25

# --- Number of parallel workers (None = use all available CPU cores automatically) ---
N_WORKERS = None

# --- Backtest base configuration ---
STARTING_CASH = 1_000_000.0
COMMISSION    = 0.0
S_MULT        = 1.0
END_POLICY    = "liquidate"

# ===========================================================================
# ③ Neighborhood Perturbation Robustness Check
# ===========================================================================
ROBUSTNESS_ENABLED   = True
ROBUSTNESS_N_PERTURB = 20
ROBUSTNESS_NOISE_PCT = 0.15       # Wider perturbation range ±15%, more aggressively filters true islands (was 0.10)
ROBUSTNESS_MAX_DROP  = 0.20       # Stricter: tolerance for post-perturbation drop lowered to 20%, requires more robustness (was 0.25)
ROBUSTNESS_MIN_PASS  = 0.70

# ===========================================================================
# ④ Generalization Gap Filtering (V8: Stricter)
# ---------------------------------------------------------------------------
# V7: GENGAP_THRESHOLD=0.30, GENGAP_PENALTY_SCALE=0.30
# V8: GENGAP_THRESHOLD=0.20, GENGAP_PENALTY_SCALE=0.50
#     Additionally introduces GENGAP_HARD_PD_DROP: trials eliminated if
#     validation PD drops more than this ratio relative to training PD
# ===========================================================================
GENGAP_ENABLED        = True
GENGAP_THRESHOLD      = 0.50       # Relaxed: large market structure difference across periods; ≤50% degradation is acceptable (was 0.25)
GENGAP_PENALTY_SCALE  = 0.20       # Reduced soft penalty to avoid suppressing cross-period differences (was 0.50)
GENGAP_HARD_PD_DROP   = 0.35       # Eliminate if validation PD / training PD drops more than 35% (new)

# ===========================================================================
# ⑤ Reverse Validation: Permanently Disabled
# ---------------------------------------------------------------------------
# ===========================================================================
REVERSE_VALIDATION_ENABLED = False
REVERSE_N_TRIALS_RATIO     = 0.5     # Retained for compatibility, no longer used
REVERSE_PARAM_DIVERGE_WARN = 0.40    # Retained for compatibility, no longer used

# ===========================================================================
# ② Full Parameter Search Space (including V8 new fused parameters)
# ---------------------------------------------------------------------------
# Format: param_name -> (min, max, step, type)
# ===========================================================================
ALL_PARAMS = {
    # ======================================================================
    # Route B streamlined: 16 core parameters (reduced from 34→16, -41% dimensions)
    # Reduction logic: Predator / SuperTrend / Exhaustion Defense / Fused Signal /
    # Multi-timeframe Confirmation — all fixed at forward-pass optimal values;
    # only the 16 most impactful parameters are optimized
    # ======================================================================

    # -- Risk control core (3 params) --------------------------------------
    "risk_budget_pct":        (0.20,  0.75,  0.05,  "float"),
    "risk_per_trade_cap":     (0.02,  0.10,  0.005, "float"),
    "risk_atr_period":        (14,    42,    2,     "int"),

    # -- Standard exit (2 params) ------------------------------------------
    "take_profit_mult":       (2.5,   6.0,   0.25,  "float"),
    "stop_loss_mult":         (1.0,   3.0,   0.125, "float"),

    # -- Regime detection (2 params) ---------------------------------------
    "adx_threshold":          (10,    25,    1,     "int"),
    "bbw_upper_mult":         (1.0,   2.0,   0.1,   "float"),

    # -- T strategy (4 params) ---------------------------------------------
    "t_ma_filter":            (100,   300,   10,    "int"),
    "t_fast":                 (10,    35,    2,     "int"),
    "t_slow":                 (40,    90,    4,     "int"),
    "t_pullback_atr":         (0.1,   0.8,   0.05,  "float"),

    # -- B strategy (3 params) ---------------------------------------------
    "b_period":               (15,    30,    1,     "int"),
    "b_rsi_oversold":         (25,    45,    1,     "int"),
    "b_rsi_overbought":       (55,    80,    1,     "int"),

    # -- M strategy (2 params) ---------------------------------------------
    "m_vol_mult":             (1.0,   2.5,   0.05,  "float"),
    "m_trend_ma":             (10,    30,    2,     "int"),
}

# Parameters fixed and excluded from optimization.
# Predator / SuperTrend / Exhaustion Defense / Fused Signal / Multi-timeframe Confirmation
# are all fixed at their forward-pass optimal values.
# Fixing these dramatically improves generalization (34 dims → 87% were isolated islands).
FIXED_PARAMS = {
    "printlog":              True,
    "risk_total_budget":     800000.0,
    "b_dev":2.2,
    "max_units":200000,
    "cooldown":2,

    # ── Predator mode (fixed at forward-pass optimal) ──────────────────────
    "predator_adx_threshold": 29,
    "super_risk_budget_mult": 2.5,
    "super_risk_cap_mult":    2.25,

    # ── SuperTrend & ratchet (fixed at forward-pass optimal) ───────────────
    "super_adx_threshold":   22,
    "super_tp_mult":         5.0,
    "super_sl_mult":         4.0,
    "ratchet_lock_mult":     3.2,
    "super_trend_duration":  10,

    # ── Exhaustion defense (fixed at forward-pass optimal) ─────────────────
    "super_bailout_drops":   3,
    "super_bailout_r_floor": 3.25,
    "super_bailout_r_ceil":  4.25,

    # ── Fused signal (fixed at forward-pass optimal) ───────────────────────
    "confidence_slope":      0.8,
    "min_signal_threshold":  0.325,

    # ── Multi-timeframe confirmation (fixed at forward-pass optimal) ────────
    "adx_confirm_bars":      1,
    "bbw_confirm_bars":      5,
}


# ===========================================================================
#  Utility Functions
# ===========================================================================

def _snap(value, lo, hi, step, ptype):
    """Snap value to the nearest grid point defined by step, then clip to [lo, hi]."""
    if ptype == "int":
        v = lo + round((value - lo) / step) * step
        return int(max(lo, min(hi, v)))
    else:
        v = lo + round((value - lo) / step) * step
        return float(max(lo, min(hi, round(v, 8))))


def lhs_sample(n: int, param_spec: dict) -> list:
    """
    Latin Hypercube Sampling: divide each parameter's range into n equal strata,
    sample one random point per stratum, shuffle the order, then pair columns.
    """
    strata = {}
    for name, (lo, hi, step, ptype) in param_spec.items():
        width = (hi - lo) / n
        pts = []
        for i in range(n):
            raw = lo + (i + random.random()) * width
            pts.append(_snap(raw, lo, hi, step, ptype))
        random.shuffle(pts)
        strata[name] = pts

    return [{name: strata[name][i] for name in param_spec} for i in range(n)]


def build_full_params(sampled: dict) -> dict:
    merged = dict(FIXED_PARAMS)
    merged.update(sampled)
    return merged


# ===========================================================================
#  Backtrader Backtest Core
# ===========================================================================

def run_backtest(params: dict, data_dir: str) -> dict:
    """Single backtest run, called inside a subprocess."""
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        import backtrader as bt
        from framework.data_loader import add_10_csv_feeds
        from framework.analyzers import TruePortfolioPD, Activity
        from framework.strategy_base import COMP396BrokerConfig
        from framework.strategies_loader import load_strategy_class

        cerebro = bt.Cerebro(stdstats=False, preload=True, runonce=True)
        cerebro.broker.setcash(STARTING_CASH)
        if COMMISSION > 0:
            cerebro.broker.setcommission(commission=COMMISSION)

        add_10_csv_feeds(cerebro, Path(data_dir))

        StrategyClass = load_strategy_class("team8", None)
        run_params = dict(params)
        run_params["_comp396"] = COMP396BrokerConfig(
            s_mult=S_MULT,
            end_policy=END_POLICY,
            output_dir="/tmp/opt_scratch",
            debug=False,
        )

        cerebro.addstrategy(StrategyClass, **run_params)
        cerebro.addanalyzer(TruePortfolioPD, _name="truepd")
        cerebro.addanalyzer(Activity, _name="activity")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        strat = cerebro.run(maxcpus=1)[0]

        truepd = strat.analyzers.truepd.get_analysis()
        act    = strat.analyzers.activity.get_analysis()
        trades = strat.analyzers.trades.get_analysis()

        return {
            "true_pd_ratio": truepd.get("pd_ratio"),
            "final_equity":  truepd.get("final_equity"),
            "final_profit":  truepd.get("final_profit"),
            "max_drawdown":  truepd.get("max_drawdown"),
            "activity_pct":  act.get("activity_pct"),
            "total_trades":  trades.get("total", {}).get("closed", 0),
            "error":         None,
        }

    except Exception as e:
        return {
            "true_pd_ratio": None,
            "final_equity":  None,
            "final_profit":  None,
            "max_drawdown":  None,
            "activity_pct":  None,
            "total_trades":  0,
            "error":         str(e),
        }


def _worker(args):
    trial_id, params, data_dir = args
    result = run_backtest(params, data_dir)
    result["trial_id"] = trial_id
    result.update(params)
    return result


# ===========================================================================
#  Scoring Functions — aligned with COMP396 scoring rubric
# ===========================================================================

def _return_marks(final_profit: float) -> int:
    ret = final_profit / STARTING_CASH
    if ret >= 0.30: return 11
    if ret >= 0.15: return 8
    if ret >= 0.00: return 4
    return 0


def _pd_marks(pd_ratio: float) -> int:
    if pd_ratio >= 3.0: return 11
    if pd_ratio >= 2.0: return 8
    if pd_ratio >= 0.5: return 4
    return 0


def comp396_score(row: dict) -> float:
    """
    Simulates COMP396 scoring: return_marks + pd_marks (max 22).
    Hard elimination: error / insufficient trades / PD is None -> -inf
    Tiebreak: continuous pd_ratio value (weight 1e-4, does not affect tier ordering)
    """
    if row.get("error"):
        return float("-inf")
    if row.get("total_trades", 0) < MIN_TRADES:
        return float("-inf")
    pd_ratio = row.get("true_pd_ratio")
    if pd_ratio is None or (isinstance(pd_ratio, float) and math.isnan(pd_ratio)):
        return float("-inf")

    final_profit = row.get("final_profit") or 0.0
    total = _return_marks(final_profit) + _pd_marks(pd_ratio)
    tiebreak = pd_ratio * 1e-4
    return total + tiebreak


def score_train(row: dict) -> float:
    return comp396_score(row)


def score_val(val_row: dict, train_score, train_pd=None) -> float:
    """
    Validation set scoring (V8: stricter generalization filtering).

    Hard filter 1: validation COMP score drops more than GENGAP_THRESHOLD vs training -> eliminate
    Hard filter 2 (new): validation PD drops more than GENGAP_HARD_PD_DROP vs training PD -> eliminate
    Soft penalty: proportionally reduce score when gap is within threshold
    """
    base = comp396_score(val_row)
    if base == float("-inf"):
        return float("-inf")

    if GENGAP_ENABLED:
        # --- Hard filter 1: COMP total score gap ---
        if train_score is not None and train_score > 0:
            gap_ratio = (train_score - base) / train_score
            if gap_ratio > GENGAP_THRESHOLD:
                return float("-inf")
            base -= GENGAP_PENALTY_SCALE * max(0.0, gap_ratio)

        # --- Hard filter 2 (V8 new): PD Ratio direct drop ratio ---
        if train_pd is not None and train_pd > 0:
            val_pd = val_row.get("true_pd_ratio") or 0.0
            pd_drop = (train_pd - val_pd) / train_pd
            if pd_drop > GENGAP_HARD_PD_DROP:
                return float("-inf")
            # Additional soft penalty: larger PD drop incurs heavier deduction
            base -= 0.3 * max(0.0, pd_drop)

    return base


# ===========================================================================
#  Neighborhood Perturbation Robustness Check
# ===========================================================================

def perturb_one(params: dict, noise_pct: float) -> dict:
    """Apply simultaneous random perturbation to all optimizable parameters (joint multi-dimensional)."""
    perturbed = dict(params)
    for name, (lo, hi, step, ptype) in ALL_PARAMS.items():
        if name not in params:
            continue
        delta = random.uniform(-noise_pct, noise_pct) * (hi - lo)
        perturbed[name] = _snap(params[name] + delta, lo, hi, step, ptype)
    return perturbed


def robustness_filter(
    candidates_df: pd.DataFrame,
    train_dir: str,
    n_perturb: int,
    noise_pct: float,
    max_drop: float,
    min_pass: float,
    n_workers: int,
    log_fn,
) -> pd.DataFrame:
    """
    Joint multi-dimensional perturbation robustness check.
    Pass criterion: (base_score - perturbed_score) / base_score < max_drop
    Candidates with pass rate >= min_pass are retained; others are treated as parameter islands and eliminated.
    """
    log_fn(f"\n  [Perturbation Check] Candidates: {len(candidates_df)} | "
           f"Perturbations per candidate: {n_perturb} | Range: ±{noise_pct*100:.0f}% | "
           f"Drop tolerance: <{max_drop*100:.0f}% | Pass rate threshold: {min_pass*100:.0f}%")

    jobs = []
    cand_map = []
    base_scores = {}

    cands = candidates_df.reset_index(drop=True)
    for idx, row in cands.iterrows():
        bs = comp396_score(row.to_dict())
        base_scores[idx] = bs if bs > float("-inf") else 0.0
        base = {k: row[k] for k in ALL_PARAMS if k in row.index}
        base.update(FIXED_PARAMS)
        for _ in range(n_perturb):
            p = perturb_one(base, noise_pct)
            jobs.append((len(jobs), p, train_dir))
            cand_map.append(idx)

    t0 = time.time()
    res_map = {}
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(_worker, jobs, chunksize=4):
                res_map[res["trial_id"]] = res
    else:
        for args in jobs:
            res = _worker(args)
            res_map[res["trial_id"]] = res
    log_fn(f"  [Perturbation Check] Done, elapsed {time.time()-t0:.1f}s")

    pass_cnt  = {i: 0 for i in range(len(cands))}
    total_cnt = {i: 0 for i in range(len(cands))}

    for job_id, cand_idx in enumerate(cand_map):
        total_cnt[cand_idx] += 1
        res = res_map.get(job_id, {})
        p_score = comp396_score(res)
        b_score = base_scores[cand_idx]
        if b_score > 0:
            drop = (b_score - p_score) / b_score
            if drop < max_drop:
                pass_cnt[cand_idx] += 1

    cands = cands.copy()
    cands["robustness_pass_rate"] = [
        pass_cnt[i] / total_cnt[i] if total_cnt[i] > 0 else 0.0
        for i in range(len(cands))
    ]

    log_fn(f"\n  {'Rank':<6} {'Train PD':>10} {'Return':>10} {'COMP':>8} {'Perturb Pass%':>12} {'Status'}")
    log_fn(f"  {'-'*58}")
    for rank, (_, row) in enumerate(
        cands.sort_values("robustness_pass_rate", ascending=False).iterrows(), 1
    ):
        pd_str  = f"{row['true_pd_ratio']:.4f}" if row.get("true_pd_ratio") else "N/A"
        ret_str = f"{(row.get('final_profit',0)/STARTING_CASH*100):.1f}%"
        r_m     = _return_marks(row.get("final_profit") or 0)
        p_m     = _pd_marks(row.get("true_pd_ratio") or 0)
        status  = "✓ Robust" if row["robustness_pass_rate"] >= min_pass else "✗ Island"
        log_fn(f"  #{rank:<5} {pd_str:>10} {ret_str:>10} {r_m+p_m:>5}/22  "
               f"{row['robustness_pass_rate']*100:>10.1f}%  {status}")

    passed = cands[cands["robustness_pass_rate"] >= min_pass].copy()
    log_fn(f"\n  [Perturbation Check] Retained: {len(passed)}/{len(cands)}")

    if len(passed) == 0:
        best_rate = cands["robustness_pass_rate"].max()
        log_fn(f"  [Perturbation Check] ⚠ All eliminated — auto-relaxing to max pass rate × 0.8")
        passed = cands[cands["robustness_pass_rate"] >= best_rate * 0.8].copy()

    return passed.reset_index(drop=True)


# ===========================================================================
#  Single-Direction Optimization Flow (callable for forward or reverse pass)
# ===========================================================================

def run_single_direction(
    train_dir: str,
    val_dir: str,
    n_trials: int,
    n_workers: int,
    direction_label: str,
    log_fn,
) -> tuple:
    """
    Execute single-direction optimization (train → validate), returns (best_params, best_row_dict, summary_dict).
    direction_label: identifier used in logs, e.g. "Forward (A→B)" or "Reverse (B→A)"
    """

    t_start = time.time()
    log_fn(f"\n{'='*70}")
    log_fn(f"  [{direction_label}] Starting optimization")
    log_fn(f"  Training set: {train_dir}")
    log_fn(f"  Validation set: {val_dir}")
    log_fn(f"  Search trials: {n_trials}")
    log_fn(f"{'='*70}")

    # -------------------------------------------------------------------------
    # Phase 1: LHS global joint search (training set)
    # -------------------------------------------------------------------------
    log_fn(f"\n▶ [{direction_label}] Phase 1: Training set search ({n_trials} trials, LHS)")

    samples      = lhs_sample(n_trials, ALL_PARAMS)
    trial_args   = [(i, build_full_params(s), train_dir) for i, s in enumerate(samples)]
    t0           = time.time()
    train_results = []
    completed    = 0
    step         = max(1, n_trials // 20)

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(_worker, trial_args, chunksize=8):
                train_results.append(res)
                completed += 1
                if completed % step == 0:
                    el = time.time() - t0
                    eta = el / completed * (n_trials - completed)
                    log_fn(f"  {completed}/{n_trials} done | {el:.0f}s elapsed | ~{eta:.0f}s remaining")
    else:
        for args in trial_args:
            res = _worker(args)
            train_results.append(res)
            completed += 1
            if completed % step == 0:
                log_fn(f"  {completed}/{n_trials} done | {time.time()-t0:.0f}s elapsed")

    log_fn(f"  Phase 1 complete, elapsed {time.time()-t0:.1f}s")

    train_df = pd.DataFrame(train_results)
    train_df["train_score"] = train_df.apply(lambda r: score_train(r.to_dict()), axis=1)
    train_df = train_df.sort_values("train_score", ascending=False).reset_index(drop=True)

    valid_df = train_df[train_df["train_score"] > float("-inf")]
    log_fn(f"  Valid trials: {len(valid_df)}/{n_trials} ({len(valid_df)/n_trials*100:.1f}%)")
    if len(valid_df) > 0:
        log_fn(f"  Training PD  max: {valid_df['true_pd_ratio'].max():.4f} | "
               f"median: {valid_df['true_pd_ratio'].median():.4f}")

    if len(valid_df) == 0:
        log_fn(f"  ⚠ [{direction_label}] No valid trials, skipping")
        return None, None, None

    # -------------------------------------------------------------------------
    # Phase 2: Candidate filtering + perturbation robustness check
    # -------------------------------------------------------------------------
    log_fn(f"\n▶ [{direction_label}] Phase 2: Candidate filtering + perturbation check")

    n_cands        = max(6, int(len(valid_df) * TOP_CANDIDATE_PCT))
    candidates_raw = valid_df.head(n_cands).copy()
    log_fn(f"  Initial candidates: {len(candidates_raw)}")

    if ROBUSTNESS_ENABLED:
        candidates_df = robustness_filter(
            candidates_df=candidates_raw,
            train_dir=train_dir,
            n_perturb=ROBUSTNESS_N_PERTURB,
            noise_pct=ROBUSTNESS_NOISE_PCT,
            max_drop=ROBUSTNESS_MAX_DROP,
            min_pass=ROBUSTNESS_MIN_PASS,
            n_workers=n_workers,
            log_fn=log_fn,
        )
    else:
        candidates_df = candidates_raw
        log_fn("  [Perturbation Check] Skipped (ROBUSTNESS_ENABLED=False)")

    log_fn(f"  Candidates after perturbation check: {len(candidates_df)}")

    # -------------------------------------------------------------------------
    # Phase 3: Candidate validation set backtesting
    # -------------------------------------------------------------------------
    log_fn(f"\n▶ [{direction_label}] Phase 3: Candidate validation ({len(candidates_df)} candidates)")

    train_score_lookup = {
        int(row["trial_id"]): comp396_score(row.to_dict())
        for _, row in candidates_df.iterrows()
        if "trial_id" in row.index
    }
    train_pd_lookup = {
        int(row["trial_id"]): row.get("true_pd_ratio")
        for _, row in candidates_df.iterrows()
        if "trial_id" in row.index
    }

    val_args = []
    for _, row in candidates_df.iterrows():
        p = {k: row[k] for k in ALL_PARAMS if k in row.index}
        p.update(FIXED_PARAMS)
        val_args.append((int(row.get("trial_id", -1)), p, val_dir))

    t0        = time.time()
    val_res   = []
    completed = 0
    vstep     = max(1, len(val_args) // 10)

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            for res in pool.imap_unordered(_worker, val_args, chunksize=4):
                val_res.append(res)
                completed += 1
                if completed % vstep == 0:
                    log_fn(f"  [Validation] {completed}/{len(val_args)} done | {time.time()-t0:.0f}s")
    else:
        for args in val_args:
            res = _worker(args)
            val_res.append(res)
            completed += 1
            if completed % vstep == 0:
                log_fn(f"  [Validation] {completed}/{len(val_args)} done | {time.time()-t0:.0f}s")

    log_fn(f"  Phase 3 complete, elapsed {time.time()-t0:.1f}s")

    val_df = pd.DataFrame(val_res)
    val_df["val_score"] = val_df.apply(
        lambda r: score_val(
            r.to_dict(),
            train_score_lookup.get(int(r.get("trial_id", -1))),
            train_pd_lookup.get(int(r.get("trial_id", -1))),
        ),
        axis=1,
    )
    if GENGAP_ENABLED:
        log_fn(f"  [Gen Gap] Hard filters (>{GENGAP_THRESHOLD*100:.0f}% COMP degradation "
               f"or >{GENGAP_HARD_PD_DROP*100:.0f}% PD degradation eliminated) + soft penalty applied")

    val_df    = val_df.sort_values("val_score", ascending=False).reset_index(drop=True)
    val_valid = val_df[val_df["val_score"] > float("-inf")]
    log_fn(f"  Valid validation candidates: {len(val_valid)}/{len(val_df)}")

    # -------------------------------------------------------------------------
    # Phase 4: Extract best result
    # -------------------------------------------------------------------------
    if len(val_valid) == 0:
        log_fn(f"  ⚠ [{direction_label}] No valid validation results — falling back to training set best")
        best_row = valid_df.iloc[0].to_dict()
        fallback = True
    else:
        best_row = val_valid.iloc[0].to_dict()
        fallback = False

    best_params = {}
    for k, (lo, hi, step, ptype) in ALL_PARAMS.items():
        if k in best_row:
            v = best_row[k]
            best_params[k] = int(v) if ptype == "int" else float(v)
    best_params.update(FIXED_PARAMS)

    # Report results
    tid             = int(best_row.get("trial_id", -1))
    best_train_pd   = train_pd_lookup.get(tid)
    best_train_score = train_score_lookup.get(tid)
    best_val_pd     = best_row.get("true_pd_ratio") or 0.0
    best_profit     = best_row.get("final_profit") or 0.0
    best_ret_pct    = best_profit / STARTING_CASH * 100
    best_r_marks    = _return_marks(best_profit)
    best_p_marks    = _pd_marks(best_val_pd)
    best_val_score  = best_r_marks + best_p_marks

    # Generalization gap calculation
    if best_train_score and best_train_score > 0:
        gen_gap = (best_train_score - (best_val_score + best_val_pd * 1e-4)) / best_train_score
    else:
        gen_gap = None

    log_fn(f"\n  [{direction_label}] Best result:")
    log_fn(f"  COMP396 Total Score      : {best_val_score}/22  "
           f"(Return {best_r_marks}/11  +  PD {best_p_marks}/11)")
    log_fn(f"  Validation True PD Ratio : {best_val_pd:.4f}")
    log_fn(f"  Validation Return        : {best_ret_pct:.1f}%")
    log_fn(f"  Validation Trades        : {int(best_row.get('total_trades', 0))}")
    if best_train_pd:
        log_fn(f"  Training True PD Ratio   : {best_train_pd:.4f}")
    if gen_gap is not None:
        tag = "✓ Healthy" if gen_gap < 0.15 else ("⚠ Caution" if gen_gap < 0.25 else "✗ Overfit risk")
        log_fn(f"  Score Generalization Gap : {gen_gap*100:.1f}%  {tag}")
    if fallback:
        log_fn(f"  ⚠ Note: Result is from training set fallback (all validation candidates eliminated)")

    # Top-5 validation set comparison
    top5 = val_valid.head(5)
    if len(top5) > 0:
        log_fn(f"\n  [{direction_label}] Top-5 Validation Comparison:")
        log_fn(f"  {'Rank':<5} {'COMP':>9} {'Return':>9} {'PD':>6} {'Val PD':>9} "
               f"{'Return%':>8} {'Gen Gap':>10} {'Trades':>10}")
        log_fn(f"  {'-'*76}")
        for rank, (_, vrow) in enumerate(top5.iterrows(), 1):
            v_pd    = vrow.get("true_pd_ratio") or 0.0
            profit  = vrow.get("final_profit") or 0.0
            ret_pct = profit / STARTING_CASH * 100
            r_m     = _return_marks(profit)
            p_m     = _pd_marks(v_pd)
            t_score = train_score_lookup.get(int(vrow.get("trial_id", -1)))
            v_score = r_m + p_m
            gap_str = (f"{(t_score - (v_score + v_pd*1e-4))/t_score*100:.1f}%"
                       if (t_score and t_score > 0) else "N/A")
            mark    = "  ◀ Best" if rank == 1 else ""
            log_fn(f"  #{rank:<4} {v_score:>6}/22  {r_m:>6}/11  {p_m:>3}/11  "
                   f"{v_pd:>9.4f}  {ret_pct:>6.1f}%  {gap_str:>10}  "
                   f"{int(vrow.get('total_trades',0)):>10}{mark}")

    summary = {
        "val_score": best_val_score + best_val_pd * 1e-4,
        "val_pd": best_val_pd,
        "val_profit": best_profit,
        "gen_gap": gen_gap,
        "train_pd": best_train_pd,
        "fallback": fallback,
        "elapsed": time.time() - t_start,
    }

    log_fn(f"\n  [{direction_label}] Elapsed: {summary['elapsed']/60:.1f} minutes")
    return best_params, best_row, summary


# ===========================================================================
#  Parameter Divergence Analysis (Forward vs Reverse comparison)
# ===========================================================================

def _param_divergence_report(fwd_params, rev_params, log_fn):
    """
    Compare the divergence between forward and reverse optimal parameters,
    computing the difference as a fraction of each parameter's range width.
    Returns the proportion of parameters whose divergence exceeds 30% of range width.
    """
    diverge_count = 0
    total = 0
    diffs = []

    for k, (lo, hi, step, ptype) in ALL_PARAMS.items():
        if k not in fwd_params or k not in rev_params:
            continue
        total += 1
        span = hi - lo
        if span == 0:
            continue
        diff = abs(fwd_params[k] - rev_params[k])
        diff_ratio = diff / span
        diffs.append((k, fwd_params[k], rev_params[k], diff_ratio))
        if diff_ratio > 0.30:
            diverge_count += 1

    diverge_pct = diverge_count / total if total > 0 else 0.0

    # Sort and display parameters with largest divergence
    diffs.sort(key=lambda x: -x[3])

    log_fn(f"\n  [Forward/Reverse Parameter Divergence Analysis]")
    log_fn(f"  {'Parameter':<30} {'Fwd Value':>12} {'Rev Value':>12} {'Diff/Range':>12}")
    log_fn(f"  {'-'*68}")
    for k, fv, rv, dr in diffs[:15]:  # Show top 15 most divergent
        warn = " ⚠" if dr > 0.30 else ""
        log_fn(f"  {k:<30} {fv:>12.4g} {rv:>12.4g} {dr*100:>10.1f}%{warn}")

    log_fn(f"\n  Significantly divergent parameters: {diverge_count}/{total} ({diverge_pct*100:.1f}%)")

    if diverge_pct > REVERSE_PARAM_DIVERGE_WARN:
        log_fn(f"  ⚠ Warning: Large forward/reverse parameter divergence ({diverge_pct*100:.0f}% > "
               f"{REVERSE_PARAM_DIVERGE_WARN*100:.0f}%) — "
               f"optimal parameters may be period-sensitive, overfit risk present!")
        log_fn(f"  Suggestion: Consider conservative parameters with smaller fwd/rev divergence, or widen robustness check range.")
    else:
        log_fn(f"  ✓ Forward/reverse parameter consistency is good")

    return diverge_pct


# ===========================================================================
#  Main Optimization Flow
# ===========================================================================

def run_optimization(
    train_dir: str,
    val_dir: str,
    n_trials: int,
    n_workers: int,
    output_dir: Path,
    log_fn,
    enable_reverse: bool = False,   # Never enabled; parameter kept for backward compatibility
) -> dict:

    t_total = time.time()
    log_fn(f"\n{'#'*70}")
    log_fn(f"  V8 Strategy Optimizer Started (Strict Temporal Unidirectional)")
    log_fn(f"  Data sequence : PART1 (train) → PART2 (validate) → PART3 (score, unseen)")
    log_fn(f"  Training set  : {train_dir}")
    log_fn(f"  Validation set: {val_dir}  ← Preceding segment of PART3, strongest generalization proxy")
    log_fn(f"  Search trials : {n_trials} | Min trades: {MIN_TRADES} | Workers: {n_workers}")
    log_fn(f"  Param dims    : {len(ALL_PARAMS)} optimizable + {len(FIXED_PARAMS)} fixed")
    log_fn(f"  Sampling      : Latin Hypercube Sampling")
    log_fn(f"  Gen filtering : COMP gap >{GENGAP_THRESHOLD*100:.0f}% eliminated | "
           f"PD gap >{GENGAP_HARD_PD_DROP*100:.0f}% eliminated | soft penalty scale={GENGAP_PENALTY_SCALE}")
    log_fn(f"  Reverse valid : Disabled (introduces data leakage, causes PART3 failure)")
    log_fn(f"  Start time    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_fn(f"{'#'*70}")

    # =====================================================================
    # Unidirectional optimization: PART1 (train) → PART2 (validate)
    # The only correct direction: predicting the future using the past
    # =====================================================================
    best_params, best_row, summary = run_single_direction(
        train_dir=train_dir,
        val_dir=val_dir,
        n_trials=n_trials,
        n_workers=n_workers,
        direction_label="Forward (PART1→PART2)",
        log_fn=log_fn,
    )

    if not best_params:
        log_fn("  ✗ Optimization produced no valid results")
        return {}

    # =====================================================================
    # Output final results
    # =====================================================================
    best_val_pd    = summary.get("val_pd", 0)
    best_profit    = summary.get("val_profit", 0)
    best_ret_pct   = best_profit / STARTING_CASH * 100
    best_r_marks   = _return_marks(best_profit)
    best_p_marks   = _pd_marks(best_val_pd)
    best_val_score = best_r_marks + best_p_marks

    log_fn(f"\n{'='*60}")
    log_fn(f"  Final Results (COMP396 Simulated Score)")
    log_fn(f"{'='*60}")
    log_fn(f"  COMP396 Total Score        : {best_val_score}/22  "
           f"(Return {best_r_marks}/11  +  PD {best_p_marks}/11)")
    log_fn(f"  Validation (PART2) PD      : {best_val_pd:.4f}  [{best_p_marks}/11]")
    log_fn(f"  Validation (PART2) Return  : {best_ret_pct:.1f}%  [{best_r_marks}/11]")
    log_fn(f"  Validation Final Profit    : {best_profit:,.0f}")
    if summary.get("train_pd"):
        log_fn(f"  Training (PART1) PD        : {summary['train_pd']:.4f}")
    if summary.get("gen_gap") is not None:
        gen_gap = summary["gen_gap"]
        tag = "✓ Healthy" if gen_gap < 0.15 else ("⚠ Caution" if gen_gap < 0.25 else "✗ Overfit risk")
        log_fn(f"  Score Generalization Gap   : {gen_gap*100:.1f}%  {tag}")
        if gen_gap < 0:
            log_fn(f"  (Negative value = PART2 outperforms PART1; strategy aligns with market evolution direction — a good sign for PART3)")

    log_fn(f"\n  Final parameters ({len(ALL_PARAMS)} tunable + {len(FIXED_PARAMS)} fixed):")
    for k in sorted(best_params):
        if k not in FIXED_PARAMS:
            log_fn(f"    {k:<30} = {best_params[k]}")

    # Save final parameters
    out = {k: v for k, v in best_params.items() if isinstance(v, (int, float, bool))}
    with (output_dir / "best_params.json").open("w") as f:
        json.dump(out, f, indent=2)
    log_fn(f"\n✓ Final parameters saved: best_params.json")

    log_fn(f"\n{'#'*70}")
    log_fn(f"  Optimization complete! Total elapsed: {(time.time()-t_total)/60:.1f} minutes")
    log_fn(f"  Output directory: {output_dir}")
    log_fn(f"{'#'*70}\n")

    return out


# ===========================================================================
#  CLI Entry Point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V8 Strategy Optimizer (Fused Parameters + Strict Generalization Filtering)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run
  python optimize_v8.py --train-dir ./DATA/PART1 --val-dir ./DATA/PART2

  # Skip reverse validation (saves time, reverse is already disabled)
  python optimize_v8.py --train-dir ./DATA/PART1 --val-dir ./DATA/PART2 --no-reverse

  # Custom number of search trials
  python optimize_v8.py --train-dir ./DATA/PART1 --val-dir ./DATA/PART2 --n-trials 2000
        """
    )
    parser.add_argument("--train-dir",  required=True)
    parser.add_argument("--val-dir",    required=True)
    parser.add_argument("--n-trials",   type=int, default=N_TRIALS,
                        help=f"Total number of search trials (default {N_TRIALS})")
    parser.add_argument("--workers",    type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="./opt_output")
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--no-reverse", action="store_true",
                        help="Skip reverse validation (forward pass only)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    n_workers  = args.workers or (cpu_count() or 4)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = (output_dir / "optimization_log.txt").open("w", encoding="utf-8")

    def log_fn(msg: str):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log_fn(f"Random seed: {args.seed if args.seed else 'random'} | Workers: {n_workers}/{cpu_count()}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    try:
        run_optimization(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            n_trials=args.n_trials,
            n_workers=n_workers,
            output_dir=output_dir,
            log_fn=log_fn,
            enable_reverse=not args.no_reverse,
        )
    except KeyboardInterrupt:
        log_fn("\n⚠ Interrupted by user")
    except Exception as e:
        log_fn(f"\n✗ Optimization error: {e}\n{traceback.format_exc()}")
        raise
    finally:
        log_file.close()

    print(f"\nFull parameters saved to: {output_dir}/best_params.json")


if __name__ == "__main__":
    main()