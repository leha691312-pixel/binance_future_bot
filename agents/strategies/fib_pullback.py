"""
Fibonacci Retracement Swing (Strike.money)
Trong trend, chờ hồi về 38.2% / 50% / 61.8% rồi continuation.
"""
from typing import List
import pandas as pd
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class FibPullbackStrategy(BaseStrategy):
    name = "FibPullback"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = int(self.params.get("lookback", 60))
        self.levels = self.params.get("levels", [0.382, 0.5, 0.618])
        self.tol = float(self.params.get("tolerance", 0.004))  # 0.4% gần fib
        self.rr = float(self.params.get("rr", 2.5))
        self.atr_sl_mult = float(self.params.get("atr_sl_mult", 1.8))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.lookback + 5:
            return []

        df = df.copy()
        df["atr"] = self._atr(df, 14)
        window = df.iloc[-self.lookback:-1]
        last = df.iloc[-1]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        if atr <= 0:
            return []

        swing_high = float(window["high"].max())
        swing_low = float(window["low"].min())
        range_ = swing_high - swing_low
        if range_ <= 0 or range_ / swing_low < 0.015:
            return []

        # Xác định hướng swing gần nhất: nếu high gần cuối hơn → upswing vừa xong
        hi_idx = window["high"].idxmax()
        lo_idx = window["low"].idxmin()
        up_impulse = hi_idx > lo_idx  # low trước, high sau = up move

        price = float(last["close"])
        signals = []

        if up_impulse:
            # Fib từ low → high; chờ hồi xuống fib
            for lvl in self.levels:
                fib = swing_high - range_ * float(lvl)
                if abs(price - fib) / price <= self.tol and last["close"] > last["open"]:
                    entry = price
                    sl = min(swing_low * 0.998, entry - atr * self.atr_sl_mult)
                    risk = entry - sl
                    if risk > 0:
                        tp = min(swing_high, entry + risk * self.rr)
                        if tp > entry:
                            signals.append(Signal(
                                symbol=symbol,
                                side=Side.LONG,
                                entry=entry,
                                sl=sl,
                                tp=tp,
                                confidence=0.67,
                                strategy=self.name,
                                reason=f"Fib {lvl} pullback LONG | fib={fib:.6f}",
                                timeframe=timeframe,
                                extra={"fib_level": lvl, "fib_price": fib},
                            ))
                    break
        else:
            for lvl in self.levels:
                fib = swing_low + range_ * float(lvl)
                if abs(price - fib) / price <= self.tol and last["close"] < last["open"]:
                    entry = price
                    sl = max(swing_high * 1.002, entry + atr * self.atr_sl_mult)
                    risk = sl - entry
                    if risk > 0:
                        tp = max(swing_low, entry - risk * self.rr)
                        if tp < entry:
                            signals.append(Signal(
                                symbol=symbol,
                                side=Side.SHORT,
                                entry=entry,
                                sl=sl,
                                tp=tp,
                                confidence=0.67,
                                strategy=self.name,
                                reason=f"Fib {lvl} pullback SHORT | fib={fib:.6f}",
                                timeframe=timeframe,
                                extra={"fib_level": lvl, "fib_price": fib},
                            ))
                    break

        return signals[:1]

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
