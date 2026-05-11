"""
Martingale Backtest  —  equal-size averaging down
=============================================
Entry  : close crosses above EMA20 → long  (only when 1h trend is up)
         close crosses below EMA20 → short (only when 1h trend is down)
Trend filter: 1h close > EMA(1h, TREND_EMA_PERIOD) → trend_up; else trend_down
Martingale:
  - Each level has the SAME notional size
  - Add next level when unrealized loss on the sequence
    reaches N × margin  (1 margin = notional / leverage)
    i.e. price moves 1/leverage from last entry
  - TP exits ALL levels at once, based on average entry price
  - MAX_LEVELS hit: close everything, accept loss
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
BEST_PARAMS_FILE = RESULTS_DIR / "best_params.json"

# ── Parameters ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000
FEE_RATE        = 0.0005       # 0.05% per side (taker)
LEVERAGE        = 50

BASE_RISK       = 0.05         # 1% of capital per level (equal sizing, no doubling)
MAX_LEVELS      = 10            # max martingale adds (loss side)
MAX_PYRAMID_LEVELS = 10          # max pyramid adds (profit side)
GRID_STEP_RATE  = 0.02          # price drop % to trigger next loss-side add (e.g. 0.02 = 2%)
                               # independent of LEVERAGE; default = 1/LEVERAGE
TP_MARGIN_RATE  = 1.00         # TP when profit reaches 50% of total margin used
                               # price move needed = TP_MARGIN_RATE / LEVERAGE
SL_CAPITAL_RATE = 0.50         # SL when loss reaches X% of total capital
                               # e.g. 0.10 = stop when down $1000 on $10000 account

EMA_PERIOD       = 20           # (unused for entry, kept for reference)
BOLL_PERIOD      = 20           # Bollinger Bands MA period on 1h bars
BOLL_STD         = 2.0          # number of std devs for bands
TREND_EMA_PERIOD = 20           # 1d EMA for trend filter

# ── Auto-tuning ───────────────────────────────────────────────────────────────
AUTO_TUNE = True               # True = grid search; False = single run with above params

TUNE_SPACE = {
    "LEVERAGE":           [50],
    "BASE_RISK":          [0.025, 0.05],
    "MAX_LEVELS":         [10],
    "MAX_PYRAMID_LEVELS": [10],
    "GRID_STEP_RATE":     [0.02, 0.05],
    "TP_MARGIN_RATE":     [0.50, 1.00, 3.00],
    "SL_CAPITAL_RATE":    [0.20, 0.50],
    "BOLL_PERIOD":        [14, 20],
    "BOLL_STD":           [1.5, 2.0, 2.5],
    "TREND_EMA_PERIOD":   [10, 20, 30],
}


# ── Indicator ─────────────────────────────────────────────────────────────────
def prepare(df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()

    # Bollinger Bands on 1h bars
    mid   = df["close"].rolling(BOLL_PERIOD).mean()
    std   = df["close"].rolling(BOLL_PERIOD).std()
    upper = mid + BOLL_STD * std
    lower = mid - BOLL_STD * std
    # Entry: price was below lower band last bar, now closes back above → long (反弹确认)
    #        price was above upper band last bar, now closes back below → short
    df["bb_upper"]    = upper
    df["bb_lower"]    = lower
    df["bb_mid"]      = mid
    df["mid_cross_up"]   = (df["close"].shift(1) < mid.shift(1)) & (df["close"] >= mid)
    df["mid_cross_down"] = (df["close"].shift(1) > mid.shift(1)) & (df["close"] <= mid)
    df["entry_long"]  = (df["close"].shift(1) < lower.shift(1)) & (df["close"] > lower)
    df["entry_short"] = (df["close"].shift(1) > upper.shift(1)) & (df["close"] < upper)

    # 1d trend filter: close > EMA(TREND_EMA_PERIOD) → trend_up=True
    d1     = df_1d.copy()
    d1_ema = d1["close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()
    d1["trend_up"] = d1["close"] > d1_ema
    trend = d1["trend_up"].reindex(df.index, method="ffill")
    _t = trend.ffill()
    df["trend_up"] = np.where(_t.isna(), False, _t).astype(bool)
    return df


# ── Martingale position ───────────────────────────────────────────────────────
class Martin:
    def __init__(self, direction: str, price: float, notional: float, capital: float):
        self.direction      = direction
        self.level          = 0
        self.profit_level   = 0
        self.notional       = notional                # fixed per level
        self.entries        = [(price, notional)]
        self.grid_step      = notional / LEVERAGE     # 1× margin in $ = 1/leverage price move
        self.capital        = capital                 # capital at entry (for SL calc)

    def avg_entry(self) -> float:
        total_n = sum(n for _, n in self.entries)
        return sum(p * n for p, n in self.entries) / total_n

    def tp(self) -> float:
        """TP price: profit = TP_MARGIN_RATE × total_margin, so price move = TP_MARGIN_RATE/LEVERAGE."""
        avg  = self.avg_entry()
        move = TP_MARGIN_RATE / LEVERAGE
        return avg * (1 + move) if self.direction == "long" else avg * (1 - move)

    def next_add_price(self) -> float:
        """Price at which to add next level: last entry ± GRID_STEP_RATE."""
        last_price = self.entries[-1][0]
        return (last_price * (1 - GRID_STEP_RATE) if self.direction == "long"
                else last_price * (1 + GRID_STEP_RATE))

    def hard_sl(self) -> float:
        """SL price: back-solve for price where total PnL = -SL_CAPITAL_RATE * capital.
        For long:  exit_p = (sum(n) - target_loss) / sum(n/entry_p)
        For short: exit_p = (sum(n) + target_loss) / sum(n/entry_p)
        """
        target_loss  = SL_CAPITAL_RATE * self.capital
        sum_n        = sum(n for _, n in self.entries)
        sum_n_over_p = sum(n / p for p, n in self.entries)
        if self.direction == "long":
            return (sum_n - target_loss) / sum_n_over_p
        else:
            return (sum_n + target_loss) / sum_n_over_p

    def add_level(self, price: float) -> bool:
        self.level += 1
        if self.level > MAX_LEVELS:
            return False
        self.entries.append((price, self.notional))
        return True

    def add_pyramid_level(self, price: float) -> bool:
        """Pyramid add on profit side: decreasing notional."""
        if self.profit_level >= MAX_PYRAMID_LEVELS:
            return False
        size = self.notional / (self.profit_level + 2)  # 1/2, 1/3, 1/4 ...
        self.entries.append((price, size))
        self.profit_level += 1
        return True

    def pnl(self, exit_price: float) -> float:
        total = 0.0
        for entry_p, notional in self.entries:
            pct    = ((exit_price - entry_p) / entry_p if self.direction == "long"
                      else (entry_p - exit_price) / entry_p)
            total += notional * pct - notional * FEE_RATE * 2
        return total


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(symbol: str, coin: str):
    print(f"\n{'='*50}")
    print(f"  {symbol}")
    print(f"{'='*50}")

    df_1h = pd.read_csv(DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True)
    df_1d = pd.read_csv(DATA_DIR / f"{coin}_futures_1d.csv", index_col=0, parse_dates=True)
    df = prepare(df_1h, df_1d)

    capital  = float(INITIAL_CAPITAL)
    peak_cap = capital
    martin        = None
    trades        = []
    peak_loss_ratio = 0.0   # max unrealized_loss / total_margin for current trade

    _iter = tqdm(range(EMA_PERIOD + 2, len(df)), desc=f"{coin.upper()}",
                 unit="bar", file=sys.stdout,
                 disable=not sys.stdout.isatty(), dynamic_ncols=True)

    for i in _iter:
        row = df.iloc[i]
        ts  = df.index[i]

        if capital <= 10:
            break

        # ── Exit ─────────────────────────────────────────────────────────────
        if martin is not None:
            tp_price  = martin.tp()
            hard_sl   = martin.hard_sl()
            next_add  = martin.next_add_price()

            hit_tp   = (row["high"] >= tp_price  if martin.direction == "long" else row["low"]  <= tp_price)
            hit_sl   = (row["low"]  <= hard_sl   if martin.direction == "long" else row["high"] >= hard_sl)
            hit_add  = (martin.level < MAX_LEVELS and
                        (row["low"] <= next_add if martin.direction == "long" else row["high"] >= next_add))
            # pyramid: in profit + BB mid cross in same direction
            in_profit = martin.pnl(row["close"]) > 0
            hit_pyramid = (
                martin.profit_level < MAX_PYRAMID_LEVELS and in_profit and
                ((martin.direction == "long"  and row["mid_cross_up"]) or
                 (martin.direction == "short" and row["mid_cross_down"]))
            )

            # track unrealized loss ratio vs current capital this bar
            _unrealized = martin.pnl(row["close"])
            if _unrealized < 0:
                peak_loss_ratio = max(peak_loss_ratio, -_unrealized / capital)

            if hit_tp:
                pnl      = martin.pnl(tp_price)
                pnl      = max(pnl, -capital)
                capital += pnl
                peak_cap = max(peak_cap, capital)
                trades.append({"exit_time": ts, "direction": martin.direction,
                                "level": martin.level, "profit_level": martin.profit_level,
                                "exit_reason": "TP",
                                "notional": martin.entries[0][1],
                                "entry_capital": martin.capital,
                                "peak_loss_ratio": round(peak_loss_ratio, 6),
                                "pnl_usdt": round(pnl, 4), "capital": round(capital, 4),
                                "drawdown": round((peak_cap - capital) / peak_cap, 6)})
                peak_loss_ratio = 0.0
                martin = None

            elif hit_sl:
                pnl      = martin.pnl(hard_sl)
                pnl      = max(pnl, -capital)
                capital += pnl
                peak_cap = max(peak_cap, capital)
                trades.append({"exit_time": ts, "direction": martin.direction,
                                "level": martin.level, "profit_level": martin.profit_level,
                                "exit_reason": "MAX_SL",
                                "notional": martin.entries[0][1],
                                "entry_capital": martin.capital,
                                "peak_loss_ratio": round(peak_loss_ratio, 6),
                                "pnl_usdt": round(pnl, 4), "capital": round(capital, 4),
                                "drawdown": round((peak_cap - capital) / peak_cap, 6)})
                peak_loss_ratio = 0.0
                martin = None

            elif hit_add:
                martin.add_level(next_add)
            elif hit_pyramid:
                martin.add_pyramid_level(row["close"])

        # ── Entry ─────────────────────────────────────────────────────────────
        if martin is None:
            if row["entry_long"] and row["trend_up"]:
                notional = min(capital * BASE_RISK, capital * LEVERAGE)
                martin   = Martin("long",  row["close"], notional, capital)
            elif row["entry_short"] and not row["trend_up"]:
                notional = min(capital * BASE_RISK, capital * LEVERAGE)
                martin   = Martin("short", row["close"], notional, capital)

    # ── Summary ───────────────────────────────────────────────────────────────
    if not trades:
        print("  No trades.")
        return None

    t        = pd.DataFrame(trades)
    t.to_csv(RESULTS_DIR / f"{coin}_martin.csv", index=False)

    n        = len(t)
    wins     = (t["pnl_usdt"] > 0).sum()
    losses   = n - wins
    win_rate = wins / n * 100
    avg_win  = t.loc[t["pnl_usdt"] > 0,  "pnl_usdt"].mean() if wins   else 0
    avg_loss = t.loc[t["pnl_usdt"] <= 0, "pnl_usdt"].mean() if losses else 0
    pf       = (wins * avg_win / (-losses * avg_loss)
                if losses and avg_loss else float("inf"))
    final    = t["capital"].iloc[-1]

    # per-trade drawdown already recorded; also compute single-trade max loss
    max_single_loss = t.loc[t["pnl_usdt"] < 0, "pnl_usdt"].min() if losses else 0

    # max peak_loss_ratio across all trades (unrealized loss / total margin)
    max_hold_ratio = t["peak_loss_ratio"].max()

    print(f"  Trades        : {n}  ({wins}W / {losses}L,  {win_rate:.1f}%)")
    print(f"  TP / MAX_SL   : {(t['exit_reason']=='TP').sum()} / {(t['exit_reason']=='MAX_SL').sum()}")
    print(f"  Level dist    : {t['level'].value_counts().sort_index().to_dict()}")
    print(f"  Pyramid dist  : {t['profit_level'].value_counts().sort_index().to_dict()}")
    print(f"  Avg win/loss  : ${avg_win:.2f} / ${avg_loss:.2f}")
    print(f"  Max single loss: ${max_single_loss:.2f}")
    print(f"  Max hold ratio : {max_hold_ratio*100:.1f}%  (peak unrealized loss / capital)")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Total PnL     : ${t['pnl_usdt'].sum():.2f}")
    print(f"  Total return  : {(final - INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.1f}%")
    print(f"  Max drawdown  : {t['drawdown'].max()*100:.1f}%")
    print(f"  Final capital : ${final:.2f}")
    return t, (final - INITIAL_CAPITAL) / INITIAL_CAPITAL, max_hold_ratio


# ── Main ──────────────────────────────────────────────────────────────────────
def current_params() -> dict:
    return {
        "LEVERAGE": LEVERAGE, "BASE_RISK": BASE_RISK, "MAX_LEVELS": MAX_LEVELS,
        "MAX_PYRAMID_LEVELS": MAX_PYRAMID_LEVELS, "GRID_STEP_RATE": GRID_STEP_RATE,
        "TP_MARGIN_RATE": TP_MARGIN_RATE, "SL_CAPITAL_RATE": SL_CAPITAL_RATE,
        "BOLL_PERIOD": BOLL_PERIOD, "BOLL_STD": BOLL_STD,
        "TREND_EMA_PERIOD": TREND_EMA_PERIOD,
    }


COINS = [
    ("BTC/USDT:USDT", "btc"),
    ("ETH/USDT:USDT", "eth"),
    ("SOL/USDT:USDT", "sol"),
    ("HYPE/USDT:USDT", "hype"),
]


def run_once(verbose: bool = True) -> tuple[float, dict, dict]:
    """Run backtest on all coins with current global params.
    Returns (avg_return, coin_returns, coin_hold_ratios).
    """
    coin_returns: dict = {}
    coin_hold_ratios: dict = {}
    for symbol, coin in COINS:
        result = run_backtest(symbol, coin) if verbose else _run_silent(symbol, coin)
        if result is not None:
            _, ret, hold_ratio = result
            coin_returns[coin] = ret
            coin_hold_ratios[coin] = hold_ratio
    avg_ret = sum(coin_returns.values()) / len(coin_returns) if coin_returns else 0.0
    return avg_ret, coin_returns, coin_hold_ratios


def _run_silent(symbol: str, coin: str):
    """run_backtest with all stdout suppressed."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run_backtest(symbol, coin)


def _apply_params(p: dict):
    """Inject a param dict into global variables."""
    g = globals()
    for k, v in p.items():
        g[k] = v


def auto_tune():
    import itertools
    keys   = list(TUNE_SPACE.keys())
    values = list(TUNE_SPACE.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
    print(f"\n{'='*60}")
    print(f"  AUTO-TUNE  |  {total} combinations  |  {len(COINS)} coins each")
    print(f"{'='*60}")

    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}
    best_avg   = max((v.get("best_return", float("-inf")) for v in best.values()), default=float("-inf"))

    pbar = tqdm(combos, total=total, desc="AUTO-TUNE", unit="combo", ncols=90)
    for idx, combo in enumerate(pbar, 1):
        p = dict(zip(keys, combo))
        _apply_params(p)

        avg_ret, coin_returns, coin_hold_ratios = run_once(verbose=False)

        # per-coin best update
        updated = []
        for coin, ret in coin_returns.items():
            prev_ret = best.get(coin, {}).get("best_return", float("-inf"))
            if ret > prev_ret:
                best[coin] = {
                    "best_return": round(ret, 6),
                    "max_hold_ratio": round(coin_hold_ratios[coin], 6),
                    "params": current_params(),
                }
                updated.append(f"{coin.upper()} {ret*100:.1f}%")

        if updated:
            best_avg = max((v.get("best_return", float("-inf")) for v in best.values()), default=float("-inf"))
            BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
            pbar.write(f"  [{idx:>{len(str(total))}}/{total}]  avg {avg_ret*100:.1f}%  ★ {', '.join(updated)}"
                       f"  | lev={p['LEVERAGE']} risk={p['BASE_RISK']} ml={p['MAX_LEVELS']}"
                       f" tp={p['TP_MARGIN_RATE']} boll={p['BOLL_PERIOD']}/{p['BOLL_STD']}")
        elif idx % 50 == 0:
            pbar.write(f"  [{idx:>{len(str(total))}}/{total}]  avg {avg_ret*100:.1f}%  (no improvement)")

    print(f"\nTuning complete. Best per-coin results in {BEST_PARAMS_FILE}")


def main():
    if AUTO_TUNE:
        auto_tune()
        return

    print("Martingale Backtest  (equal-size averaging)")
    print(f"Capital ${INITIAL_CAPITAL:,}  |  Base risk {BASE_RISK*100}%/level  "
          f"|  Max levels {MAX_LEVELS}  |  Grid step 1/leverage={1/LEVERAGE*100:.1f}%")
    print(f"Leverage {LEVERAGE}×  |  TP {TP_MARGIN_RATE*100:.0f}% margin profit ({TP_MARGIN_RATE/LEVERAGE*100:.1f}% price)  "
          f"|  SL {SL_CAPITAL_RATE*100:.0f}% of capital  "
          f"|  BOLL({BOLL_PERIOD},{BOLL_STD}) on 1h  |  1d EMA{TREND_EMA_PERIOD} trend filter")

    avg_return, coin_returns, coin_hold_ratios = run_once(verbose=True)
    print(f"\nAvg return across coins: {avg_return*100:.1f}%")

    # ── Best params tracking (per-coin independent) ────────────────────────────
    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}

    for coin, ret in coin_returns.items():
        hold_ratio = coin_hold_ratios[coin]
        prev = best.get(coin, {})
        prev_ret = prev.get("best_return", float("-inf"))
        tag = ""
        if ret > prev_ret:
            best[coin] = {
                "best_return": round(ret, 6),
                "max_hold_ratio": round(hold_ratio, 6),
                "params": current_params(),
            }
            tag = f"  ★ new best (prev {prev_ret*100:.1f}%)"
        print(f"  {coin.upper()}: return {ret*100:.1f}%  |  hold ratio {hold_ratio*100:.1f}%"
              f"  |  best {best[coin]['best_return']*100:.1f}%{tag}")

    BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))

    print(f"\nLogs → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
