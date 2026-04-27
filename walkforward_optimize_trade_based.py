#!/usr/bin/env python3
"""
Trade-count-based walk-forward optimization script (designed for low-frequency strategies).
Windows are defined by a fixed number of trades rather than calendar time.
Supports multi-stage random search and parallel computation.
"""

import argparse
import json
import os
import sys
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import backtrader as bt

# Add project root to path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from framework.data_loader import _read_csv_safely, _mk_pandas_feed
from framework.strategies_loader import load_strategy_class
from framework.analyzers import OpenOpenPnL, RealizedPnL, PDRatio, Activity, TruePortfolioPD
from framework.strategy_base import COMP396BrokerConfig

# -----------------------------------------------------------------------------
# Global evaluation function (must be at module top level to be picklable)
# -----------------------------------------------------------------------------
def _evaluate_params(train_start, train_end, params_subset, current_best_params,
                     data_dir, s_mult, commission, cash):
    """
    Evaluate a set of parameters on the specified training window.
    Returns (true_pd_ratio, final_value, merged_params_dict).
    """
    # Merge parameters
    test_params = current_best_params.copy()
    test_params.update(params_subset)
    try:
        # Import run_backtest here since subprocesses may not see global variables
        from walkforward_optimize_trade_based import run_backtest
        strat = run_backtest(train_start, train_end, test_params, data_dir,
                             s_mult, commission, cash)
        truepd = strat.analyzers.truepd.get_analysis()
        act = strat.analyzers.activity.get_analysis()
        true_pd_ratio = truepd.get('pd_ratio', None)
        if true_pd_ratio is None or np.isnan(true_pd_ratio):
            true_pd_ratio = -9999.0
        final_value = strat.broker.getvalue()
        # activity_pct = act.get('activity_pct', 0.0)  # not needed currently
        return true_pd_ratio, final_value, test_params
    except Exception as e:
        print(f"      Subprocess evaluation failed: {e}")
        return -float('inf'), -float('inf'), test_params

# -----------------------------------------------------------------------------
# Helper: extract signal dates from strategy
# -----------------------------------------------------------------------------
def get_signal_dates_from_strategy(strategy_class, fixed_params, data_dir, s_mult, commission, cash):
    """
    Run one backtest with a fixed set of parameters and retrieve the signal_dates list
    from the strategy instance. The strategy must record entry signal dates into
    self.signal_dates inside its next() method.
    """
    cerebro = bt.Cerebro(stdstats=False, preload=True, runonce=True)
    cerebro.broker.setcash(cash)
    if commission > 0:
        cerebro.broker.setcommission(commission=commission)

    data_dir = Path(data_dir)
    csvs = sorted(data_dir.glob("*.csv"))[:10]
    for fp in csvs:
        df = _read_csv_safely(fp)
        feed = _mk_pandas_feed(df, name=fp.stem)
        cerebro.adddata(feed)

    params = fixed_params.copy()
    params['_comp396'] = COMP396BrokerConfig(
        s_mult=s_mult,
        end_policy='liquidate',
        output_dir='',
        debug=False
    )
    cerebro.addstrategy(strategy_class, **params)

    strat = cerebro.run(maxcpus=1)[0]
    if hasattr(strat, 'signal_dates'):
        return sorted(set(strat.signal_dates))  # deduplicate and sort
    else:
        raise RuntimeError("Strategy does not have a signal_dates attribute. "
                           "Please record entry signal dates into self.signal_dates inside next().")

# -----------------------------------------------------------------------------
# Generate windows from a fixed trade count
# -----------------------------------------------------------------------------
def generate_windows_from_trades(signal_dates, train_trades, test_trades, buffer_days):
    """
    Generate walk-forward windows from a list of signal dates and a fixed trade count.
    Returns a list of (train_start, train_end, test_start, test_end) tuples as datetime objects.
    Windows are non-overlapping: each window advances by train_trades signals.
    """
    n_signals = len(signal_dates)
    windows = []
    idx = 0
    while idx + train_trades + test_trades <= n_signals:
        train_signal_start = signal_dates[idx]
        train_signal_end = signal_dates[idx + train_trades - 1]
        test_signal_start = signal_dates[idx + train_trades]
        test_signal_end = signal_dates[idx + train_trades + test_trades - 1]

        train_start = train_signal_start - timedelta(days=buffer_days)
        train_end = train_signal_end + timedelta(days=buffer_days)
        test_start = test_signal_start - timedelta(days=buffer_days)
        test_end = test_signal_end + timedelta(days=buffer_days)

        # Ensure datetime type
        if not isinstance(train_start, datetime):
            train_start = datetime.combine(train_start, datetime.min.time())
        if not isinstance(train_end, datetime):
            train_end = datetime.combine(train_end, datetime.min.time())
        if not isinstance(test_start, datetime):
            test_start = datetime.combine(test_start, datetime.min.time())
        if not isinstance(test_end, datetime):
            test_end = datetime.combine(test_end, datetime.min.time())

        windows.append((train_start, train_end, test_start, test_end))
        idx += train_trades  # non-overlapping advance

    return windows

# -----------------------------------------------------------------------------
# Extract metrics from analyzers (simplified)
# -----------------------------------------------------------------------------
def extract_metrics(strat):
    truepd = strat.analyzers.truepd.get_analysis()
    true_pd_ratio = truepd.get('pd_ratio', None)
    if true_pd_ratio is None or np.isnan(true_pd_ratio):
        true_pd_ratio = -9999.0
    final_value = strat.broker.getvalue()
    return true_pd_ratio, final_value

# -----------------------------------------------------------------------------
# Core backtest function
# -----------------------------------------------------------------------------
def run_backtest(fromdate, todate, strategy_params, data_dir, s_mult=2.0, commission=0.0, starting_cash=1_000_000):
    cerebro = bt.Cerebro(stdstats=False, preload=True, runonce=True)
    cerebro.broker.setcash(starting_cash)
    if commission > 0:
        cerebro.broker.setcommission(commission=commission)

    data_dir = Path(data_dir)
    csvs = sorted(data_dir.glob("*.csv"))[:10]
    for fp in csvs:
        df = _read_csv_safely(fp)
        if fromdate or todate:
            mask = pd.Series(True, index=df.index)
            if fromdate:
                mask &= df.index.date >= fromdate.date() if isinstance(fromdate, datetime) else df.index.date >= fromdate
            if todate:
                mask &= df.index.date <= todate.date() if isinstance(todate, datetime) else df.index.date <= todate
            df = df.loc[mask]
        if df.empty:
            continue
        feed = _mk_pandas_feed(df, name=fp.stem)
        cerebro.adddata(feed)

    if len(cerebro.datas) == 0:
        raise ValueError("No data in the specified date range.")

    StrategyClass = load_strategy_class("v8", "TeamStrategy")
    params = strategy_params.copy()
    params['_comp396'] = COMP396BrokerConfig(
        s_mult=s_mult,
        end_policy='liquidate',
        output_dir='',
        debug=False
    )
    cerebro.addstrategy(StrategyClass, **params)

    cerebro.addanalyzer(TruePortfolioPD, _name="truepd")
    cerebro.addanalyzer(Activity, _name="activity")  # retained for potential downstream analysis

    strat = cerebro.run(maxcpus=1)[0]
    return strat

# -----------------------------------------------------------------------------
# Sample parameters (with step support)
# -----------------------------------------------------------------------------
def sample_params(param_defs, fixed_params):
    params = fixed_params.copy()
    for name, defn in param_defs.items():
        typ = defn['type']
        low = defn.get('low')
        high = defn.get('high')
        step = defn.get('step')
        if typ == 'int':
            if step is None:
                step = 1
            n_steps = (high - low) // step
            idx = random.randint(0, n_steps)
            value = low + idx * step
        elif typ == 'float':
            if step is None:
                value = random.uniform(low, high)
            else:
                n_steps = int((high - low) / step)
                idx = random.randint(0, n_steps)
                value = low + idx * step
        elif typ == 'choice':
            choices = defn['choices']
            value = random.choice(choices)
        else:
            raise ValueError(f"Unknown type {typ}")
        params[name] = value
    return params

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params_config.json', help='Path to JSON config file')
    parser.add_argument('--data-dir', default='/Users/dizzychan/Desktop/BT396_Version_0.1.3/DATA/Cleaned_Data', help='Data directory')
    parser.add_argument('--output-dir', default='/Users/dizzychan/Desktop/BT396_Version_0.1.3/output/walkforward_opt', help='Output directory')
    parser.add_argument('--s-mult', type=float, default=2.0, help='Slippage multiplier')
    parser.add_argument('--commission', type=float, default=0.0, help='Commission rate')
    parser.add_argument('--cash', type=float, default=1_000_000, help='Starting cash')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file {config_path} not found.")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Run one baseline backtest to collect signal dates
    # -------------------------------------------------------------------------
    print("Running baseline backtest to collect signal dates...")
    StrategyClass = load_strategy_class("v6", "TeamStrategy")
    fixed_default = config['fixed_params'].copy()
    signal_dates = get_signal_dates_from_strategy(
        StrategyClass, fixed_default, args.data_dir,
        args.s_mult, args.commission, args.cash
    )
    print(f"Collected {len(signal_dates)} signal dates.")

    if len(signal_dates) < config['window']['train_trades'] + config['window']['test_trades']:
        print(f"Error: not enough signals to form even one window. "
              f"Need at least {config['window']['train_trades'] + config['window']['test_trades']} signals.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 2: Generate windows (fixed trade count)
    # -------------------------------------------------------------------------
    train_trades = config['window']['train_trades']
    test_trades = config['window']['test_trades']
    buffer_days = config['window']['buffer_days']
    windows = generate_windows_from_trades(signal_dates, train_trades, test_trades, buffer_days)
    print(f"Generated {len(windows)} windows.")

    # -------------------------------------------------------------------------
    # Step 3: Multi-stage optimization for each window
    # -------------------------------------------------------------------------
    results = []
    fixed_params = config['fixed_params']

    for win_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"\n===== Window {win_idx+1}/{len(windows)} =====")
        print(f"Train: {train_start.date()} to {train_end.date()}")
        print(f"Test:  {test_start.date()} to {test_end.date()}")

        current_best_params = fixed_params.copy()

        for stage_idx, stage in enumerate(config['stages']):
            print(f"  Stage {stage_idx+1}: {stage['name']}")
            param_defs = stage['params']
            n_iter = stage.get('iterations', 100)

            best_stage_score = -float('inf')
            best_stage_params = None
            best_stage_final = -float('inf')

            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = []
                for _ in range(n_iter):
                    sampled = sample_params(param_defs, {})
                    # Submit task to process pool with all required arguments
                    future = executor.submit(
                        _evaluate_params,
                        train_start, train_end,
                        sampled,
                        current_best_params,
                        args.data_dir,
                        args.s_mult,
                        args.commission,
                        args.cash
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    true_pd, final_val, params_comb = future.result()
                    if true_pd > best_stage_score or (true_pd == best_stage_score and final_val > best_stage_final):
                        best_stage_score = true_pd
                        best_stage_params = params_comb
                        best_stage_final = final_val

            if best_stage_params is not None:
                current_best_params = best_stage_params
                print(f"    Stage best true_pd_ratio: {best_stage_score:.4f}")
            else:
                print(f"    No valid parameters found in stage — keeping previous values")

        # Evaluate on test window
        test_strat = run_backtest(test_start, test_end, current_best_params, args.data_dir,
                                  args.s_mult, args.commission, args.cash)
        test_true_pd, test_final = extract_metrics(test_strat)
        # Collect activity for logging
        test_act = test_strat.analyzers.activity.get_analysis().get('activity_pct', 0.0)
        print(f"  Test window: true_pd_ratio={test_true_pd:.4f}, final_value={test_final:.2f}, activity={test_act:.2f}")

        record = {
            'window': win_idx + 1,
            'train_start': train_start.date().isoformat(),
            'train_end': train_end.date().isoformat(),
            'test_start': test_start.date().isoformat(),
            'test_end': test_end.date().isoformat(),
            'test_true_pd': test_true_pd,
            'test_final_value': test_final,
            'test_activity': test_act,
        }
        for k, v in current_best_params.items():
            record[f'param_{k}'] = v
        results.append(record)

        df_res = pd.DataFrame(results)
        df_res.to_csv(out_dir / 'window_results.csv', index=False)

    # -------------------------------------------------------------------------
    # Step 4: Compute robust parameters (median across windows)
    # -------------------------------------------------------------------------
    df_final = pd.DataFrame(results)
    param_cols = [col for col in df_final.columns if col.startswith('param_')]
    median_params = {}
    for col in param_cols:
        median_params[col.replace('param_', '')] = df_final[col].median()

    # Build a mapping from parameter name to type for proper type conversion
    print("\n===== Final Robust Parameters (median of per-window best values) =====")

    # Build param name → type mapping from stage definitions
    param_type_map = {}
    for stage in config['stages']:
        for param_name, defn in stage['params'].items():
            param_type_map[param_name] = defn['type']
    # Fixed params are used as-is and don't need type conversion here;
    # only optimized params (defined in stages) require rounding/casting.

    median_params = {}
    for col in param_cols:
        param_name = col.replace('param_', '')
        median_val = df_final[col].median()
        # Apply type conversion if the param has a known type
        if param_name in param_type_map:
            typ = param_type_map[param_name]
            if typ == 'int':
                median_val = int(round(median_val))
            elif typ == 'float':
                median_val = round(median_val, 4)  # 4 decimal places
            # 'choice' params are already discrete values; no conversion needed
        median_params[param_name] = median_val
        print(f"{param_name}: {median_val} ({typ if param_name in param_type_map else 'unknown'})")

    # Save robust params using converted values
    with open(out_dir / 'robust_params.json', 'w') as f:
        json.dump(median_params, f, indent=2)

    print(f"\nOptimization complete. Results saved to {out_dir}")

if __name__ == '__main__':
    main()