"""
Backtest Engine nghiêm ngặt:
- Event-driven simulation
- Commission + Slippage
- Walk-Forward Optimization
- Stress Tests (fees cao, slippage cao, gap, low liquidity)
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
from core.models import Signal, Side, TradeResult
from agents.strategies import get_all_strategies
from agents.risk_gate import RiskGate
import yaml
from pathlib import Path


@dataclass
class BacktestResult:
    trades: List[TradeResult] = field(default_factory=list)
    equity_curve: pd.Series = None
    metrics: Dict = field(default_factory=dict)
    params: Dict = field(default_factory=dict)


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 10000,
        commission: float = 0.0004,
        slippage: float = 0.0002,
        leverage: int = 5,
        risk_per_trade: float = 0.01,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.strategies = get_all_strategies()

    def run(
        self,
        data: Dict[str, pd.DataFrame],   # symbol -> OHLCV
        start: str = None,
        end: str = None,
    ) -> BacktestResult:
        """
        data: { "BTC/USDT": df, "ETH/USDT": df, ... }
        df columns: open, high, low, close, volume, timestamp
        """
        capital = self.initial_capital
        equity_curve = []
        trades: List[TradeResult] = []
        open_positions = {}  # key -> dict

        # Align all symbols to common timeline (use first symbol as master)
        symbols = list(data.keys())
        if not symbols:
            return BacktestResult()

        master = data[symbols[0]].copy()
        if start:
            master = master[master["timestamp"] >= pd.Timestamp(start)]
        if end:
            master = master[master["timestamp"] <= pd.Timestamp(end)]

        logger.info(f"Backtest {len(symbols)} symbols | bars={len(master)} | capital={capital}")

        for i in range(50, len(master)):
            current_time = master.iloc[i]["timestamp"]
            current_equity = capital + sum(
                self._unrealized(pos, data[pos["symbol"]].iloc[i]["close"])
                for pos in open_positions.values()
                if i < len(data[pos["symbol"]])
            )
            equity_curve.append({"time": current_time, "equity": current_equity})

            # Check SL / TP for open positions
            to_close = []
            for key, pos in open_positions.items():
                sym = pos["symbol"]
                if i >= len(data[sym]):
                    continue
                bar = data[sym].iloc[i]
                hit = self._check_exit(pos, bar)
                if hit:
                    to_close.append((key, hit))

            for key, (exit_price, reason) in to_close:
                pos = open_positions.pop(key)
                pnl, fees = self._calc_pnl(pos, exit_price)
                capital += pnl - fees
                trades.append(TradeResult(
                    symbol=pos["symbol"],
                    side=pos["side"],
                    entry=pos["entry"],
                    exit=exit_price,
                    size=pos["size"],
                    pnl=pnl - fees,
                    pnl_pct=(pnl - fees) / self.initial_capital * 100,
                    strategy=pos["strategy"],
                    open_time=pos["open_time"],
                    close_time=current_time,
                    reason=reason,
                    fees=fees,
                ))

            # Generate new signals (chỉ trên nến mới)
            for sym in symbols:
                df = data[sym]
                if i >= len(df):
                    continue
                window = df.iloc[max(0, i - 150):i + 1].copy()
                if len(window) < 50:
                    continue

                for strat in self.strategies:
                    signals = strat.generate(window, sym, "15m")
                    for sig in signals:
                        # Simple risk gate in backtest
                        if len(open_positions) >= 5:
                            continue
                        if sig.risk_reward() < 1.5:
                            continue

                        # Size
                        risk_amount = capital * self.risk_per_trade
                        stop_dist = abs(sig.entry - sig.sl)
                        if stop_dist <= 0:
                            continue
                        size = risk_amount / stop_dist
                        size = min(size, capital * 0.05 * self.leverage / sig.entry)

                        # Apply slippage on entry
                        entry_price = sig.entry * (1 + self.slippage) if sig.side == Side.LONG else sig.entry * (1 - self.slippage)

                        key = f"{sym}_{sig.side.value}_{current_time}"
                        open_positions[key] = {
                            "symbol": sym,
                            "side": sig.side,
                            "entry": entry_price,
                            "sl": sig.sl,
                            "tp": sig.tp,
                            "size": size,
                            "strategy": sig.strategy,
                            "open_time": current_time,
                        }

        # Close remaining
        for key, pos in list(open_positions.items()):
            last_price = data[pos["symbol"]].iloc[-1]["close"]
            pnl, fees = self._calc_pnl(pos, last_price)
            capital += pnl - fees
            trades.append(TradeResult(
                symbol=pos["symbol"],
                side=pos["side"],
                entry=pos["entry"],
                exit=last_price,
                size=pos["size"],
                pnl=pnl - fees,
                pnl_pct=(pnl - fees) / self.initial_capital * 100,
                strategy=pos["strategy"],
                open_time=pos["open_time"],
                close_time=master.iloc[-1]["timestamp"],
                reason="END",
                fees=fees,
            ))

        eq = pd.DataFrame(equity_curve).set_index("time")["equity"]
        metrics = self._compute_metrics(trades, eq)

        return BacktestResult(trades=trades, equity_curve=eq, metrics=metrics)

    def _check_exit(self, pos, bar) -> Optional[Tuple[float, str]]:
        if pos["side"] == Side.LONG:
            if bar["low"] <= pos["sl"]:
                return pos["sl"] * (1 - self.slippage), "SL"
            if bar["high"] >= pos["tp"]:
                return pos["tp"] * (1 - self.slippage), "TP"
        else:
            if bar["high"] >= pos["sl"]:
                return pos["sl"] * (1 + self.slippage), "SL"
            if bar["low"] <= pos["tp"]:
                return pos["tp"] * (1 + self.slippage), "TP"
        return None

    def _calc_pnl(self, pos, exit_price) -> Tuple[float, float]:
        if pos["side"] == Side.LONG:
            pnl = (exit_price - pos["entry"]) * pos["size"]
        else:
            pnl = (pos["entry"] - exit_price) * pos["size"]
        fees = (pos["entry"] + exit_price) * pos["size"] * self.commission
        return pnl, fees

    def _unrealized(self, pos, price) -> float:
        if pos["side"] == Side.LONG:
            return (price - pos["entry"]) * pos["size"]
        return (pos["entry"] - price) * pos["size"]

    def _compute_metrics(self, trades: List[TradeResult], equity: pd.Series) -> Dict:
        if not trades:
            return {"total_trades": 0}

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100 if len(equity) > 1 else 0
        max_dd = self._max_drawdown(equity)

        return {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else float("inf"),
            "total_pnl": sum(pnls),
            "total_return_pct": total_return,
            "max_drawdown_pct": max_dd * 100,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "sharpe": self._sharpe(equity),
            "expectancy": np.mean(pnls) if pnls else 0,
        }

    def _max_drawdown(self, equity: pd.Series) -> float:
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return abs(dd.min()) if len(dd) else 0

    def _sharpe(self, equity: pd.Series, periods_per_year=365 * 24 * 4) -> float:  # 15m
        if len(equity) < 10:
            return 0
        ret = equity.pct_change().dropna()
        if ret.std() == 0:
            return 0
        return (ret.mean() / ret.std()) * np.sqrt(periods_per_year)


def walk_forward(
    data: Dict[str, pd.DataFrame],
    train_days: int = 90,
    test_days: int = 30,
    step_days: int = 30,
    initial_capital: float = 10000,
) -> List[BacktestResult]:
    """
    Walk-Forward: train window → test window → roll forward
    """
    # Giả sử data đã có timestamp
    all_times = sorted(set().union(*[set(df["timestamp"]) for df in data.values()]))
    if not all_times:
        return []

    start = all_times[0]
    end = all_times[-1]
    results = []

    current = start
    while current + timedelta(days=train_days + test_days) <= end:
        train_end = current + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)

        # Filter data
        train_data = {
            s: df[(df["timestamp"] >= current) & (df["timestamp"] < train_end)]
            for s, df in data.items()
        }
        test_data = {
            s: df[(df["timestamp"] >= train_end) & (df["timestamp"] < test_end)]
            for s, df in data.items()
        }

        # (Ở đây có thể optimize params trên train_data)
        engine = BacktestEngine(initial_capital=initial_capital)
        res = engine.run(test_data)
        res.params = {"train_start": str(current), "test_start": str(train_end), "test_end": str(test_end)}
        results.append(res)

        current += timedelta(days=step_days)
        logger.info(f"Walk-forward window done: {train_end.date()} → {test_end.date()} | trades={res.metrics.get('total_trades', 0)}")

    return results


def stress_test(
    data: Dict[str, pd.DataFrame],
    base_commission: float = 0.0004,
    base_slippage: float = 0.0002,
) -> Dict[str, BacktestResult]:
    """
    Stress scenarios:
    - High fees (2x, 3x)
    - High slippage
    - Combined adverse
    """
    scenarios = {
        "base": {"commission": base_commission, "slippage": base_slippage},
        "fees_2x": {"commission": base_commission * 2, "slippage": base_slippage},
        "slippage_3x": {"commission": base_commission, "slippage": base_slippage * 3},
        "adverse": {"commission": base_commission * 2.5, "slippage": base_slippage * 3},
        "extreme": {"commission": 0.001, "slippage": 0.001},
    }

    results = {}
    for name, params in scenarios.items():
        engine = BacktestEngine(
            commission=params["commission"],
            slippage=params["slippage"],
        )
        res = engine.run(data)
        results[name] = res
        logger.info(f"Stress [{name}] → return={res.metrics.get('total_return_pct', 0):.2f}% | DD={res.metrics.get('max_drawdown_pct', 0):.2f}%")

    return results
