#!/usr/bin/env bash
set -e

PYTHON=/home/bear/Softwares/anaconda3/envs/crypto/bin/python

# --reset: wipe all state + trades before starting
if [[ "$1" == "--reset" ]]; then
  echo "Resetting all paper trading state and trade records..."
  rm -f paper/paper_state_*.json paper/paper_trades_*.csv
  echo "Reset complete."
fi

mkdir -p logs

nohup "$PYTHON" -u paper/paper_trade_breakout.py > logs/paper_breakout.log 2>&1 &
echo "breakout PID: $!"

nohup "$PYTHON" -u paper/paper_trade_calmar.py > logs/paper_calmar.log 2>&1 &
echo "calmar PID: $!"

nohup "$PYTHON" -u paper/paper_trade_regime.py > logs/paper_regime.log 2>&1 &
echo "regime PID: $!"

nohup "$PYTHON" -u paper/paper_trade_martingale.py > logs/paper_martingale.log 2>&1 &
echo "martingale PID: $!"

nohup "$PYTHON" -u paper/paper_trade_boll_scalp.py > logs/paper_boll_scalp.log 2>&1 &
echo "boll_scalp PID: $!"

echo "All five paper traders started. Check logs/ for output."

# ── Dashboard ──────────────────────────────────────────────────────────────────
nohup "$PYTHON" dashboard/app.py > logs/dashboard.log 2>&1 &
echo "dashboard PID: $!"
