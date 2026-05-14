#!/usr/bin/env python3
"""
tune.py — Dedicated autotune launcher
======================================
Runs grid-search autotune for any combination of strategies and coins.

Usage examples:
    # Tune SUI on all 4 strategies (sequential, each uses all CPU cores internally)
    python tune.py --coin sui

    # Tune a specific strategy only
    python tune.py --coin sui --strategy breakout
    python tune.py --coin sui --strategy calmar,regime

    # Tune all coins on all strategies (full run)
    python tune.py

    # Tune multiple coins
    python tune.py --coin sui,hype

    # After tuning, show current best params summary
    python tune.py --summary
"""

# ── env threads (must be before numpy) ───────────────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

STRATEGIES = ["breakout", "calmar", "regime", "martingale", "boll_scalp"]

ALL_COINS = ["btc", "eth", "sol", "hype", "sui"]

RESULTS_DIRS = {
    "breakout":   _ROOT / "results/breakout",
    "calmar":     _ROOT / "results/calmar",
    "regime":     _ROOT / "results/regime",
    "martingale": _ROOT / "results/martingale",
    "boll_scalp": _ROOT / "results/boll_scalp",
}


def _load_best(strategy: str) -> dict:
    p = RESULTS_DIRS[strategy] / "best_params.json"
    return json.loads(p.read_text()) if p.exists() else {}


def print_summary():
    """Print current best params & scores for all strategies and coins."""
    width = 80
    print("\n" + "=" * width)
    print(f"  BEST PARAMS SUMMARY  ({_ROOT.name})")
    print("=" * width)
    for strat in STRATEGIES:
        best = _load_best(strat)
        if not best:
            print(f"\n  [{strat.upper()}]  no results yet")
            continue
        print(f"\n  [{strat.upper()}]  results/{strat}/best_params.json")
        print(f"  {'Coin':<8}  {'Score/Return':>14}  {'MaxDD%':>7}  Key params")
        print(f"  {'-'*8}  {'-'*14}  {'-'*7}  {'-'*30}")
        for coin in ALL_COINS:
            entry = best.get(coin)
            if not entry:
                continue
            p = entry.get("params", {})
            # pick score label
            if "best_score" in entry:
                score_lbl = f"{entry['best_return'] if 'best_return' in entry else entry['best_score']:.4f}"
                score_key = "calmar/sharpe"
            else:
                score_lbl = f"{entry.get('best_return', 0)*100:.1f}%"
                score_key = "return"
            hold = entry.get("max_hold_ratio", 0) * 100
            lev  = p.get("LEVERAGE", "?")
            don  = p.get("DONCHIAN_PERIOD", "?")
            rr   = p.get("TP_RR", "?")
            sl   = p.get("SL_MULT", "?")
            adx  = p.get("ADX_MIN", "?")
            param_str = f"lev={lev} don={don} rr={rr} sl={sl} adx={adx}"
            print(f"  {coin.upper():<8}  {score_lbl:>14}  {hold:>6.1f}%  {param_str}")
    print("\n" + "=" * width)


def run_tune(strategies: list, coin_filter: list | None, n_trials: int = 1000):
    """Run autotune sequentially for each strategy (each uses all cores internally)."""
    # lazy import after env vars are set
    import importlib
    modules = {
        "breakout":   "backtest.backtest_breakout",
        "calmar":     "backtest.backtest_calmar",
        "regime":     "backtest.backtest_regime",
        "martingale": "backtest.backtest_martingale",
        "boll_scalp": "backtest.backtest_boll_scalp",
    }

    total = len(strategies)
    for idx, strat in enumerate(strategies, 1):
        print(f"\n{'#'*70}")
        print(f"  [{idx}/{total}]  Strategy: {strat.upper()}"
              + (f"  |  Coin filter: {[c for _, c in coin_filter]}" if coin_filter else "  |  All coins"))
        print(f"{'#'*70}")
        t0 = time.time()

        mod = importlib.import_module(modules[strat])
        # build coins list for this strategy
        if coin_filter:
            active = [(s, c) for s, c in mod.COINS if c in {c2 for _, c2 in coin_filter}]
            if not active:
                print(f"  WARNING: none of {coin_filter} found in {strat} COINS, skipping.")
                continue
        else:
            active = None  # use all

        extra = {"n_trials": n_trials} if strat == "martingale" else {}
        mod.auto_tune(coins=active, **extra)

        elapsed = time.time() - t0
        print(f"\n  [{strat.upper()}] done in {elapsed/60:.1f} min")

    print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Autotune launcher for crypto-futures-trading-lab"
    )
    parser.add_argument(
        "--coin", type=str, default=None,
        help=f"Comma-separated coin(s) to tune. Available: {', '.join(ALL_COINS)}"
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help=f"Comma-separated strategy/strategies. Available: {', '.join(STRATEGIES)}"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print current best params summary and exit (no tuning)"
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Number of Optuna trials per coin (martingale only, default: 1000)"
    )
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    # Validate strategies
    strats = STRATEGIES
    if args.strategy:
        requested = [s.strip().lower() for s in args.strategy.split(",")]
        invalid = [s for s in requested if s not in STRATEGIES]
        if invalid:
            print(f"ERROR: unknown strategy/strategies: {invalid}")
            print(f"Available: {STRATEGIES}")
            sys.exit(1)
        strats = requested

    # Validate coins — we need to resolve to (symbol, coin) pairs
    # Use breakout's COINS as the master list
    from backtest.backtest_breakout import COINS as _ALL_COIN_PAIRS
    coin_filter = None
    if args.coin:
        requested_coins = [c.strip().lower() for c in args.coin.split(",")]
        invalid_coins = [c for c in requested_coins if c not in {c2 for _, c2 in _ALL_COIN_PAIRS}]
        if invalid_coins:
            print(f"ERROR: unknown coin(s): {invalid_coins}")
            print(f"Available: {[c for _, c in _ALL_COIN_PAIRS]}")
            sys.exit(1)
        coin_filter = [(s, c) for s, c in _ALL_COIN_PAIRS if c in set(requested_coins)]

    run_tune(strats, coin_filter, n_trials=args.trials or 1000)


if __name__ == "__main__":
    main()
