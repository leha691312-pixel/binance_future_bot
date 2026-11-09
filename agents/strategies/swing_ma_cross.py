"""
Moving Average Crossover – Swing (Strike.money)
EMA/SMA chậm (vd 20/50) cho hold vài ngày–vài tuần.
"""
from typing import List
import pandas as pd
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class SwingMACrossStrategy(BaseStrategy):
    name = "SwingMACross"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.fast = int(self.params.get("ema_fast", 20))
        self.slow = int(self.params.get("ema_slow", 50))
        self.rr = float(self.params.get("rr", 2.5))
        self.atr_sl_mult = float(self.params.get("atr_sl_mult", 2.0))
        self.max_dist_pct = float(self.params.get("max_dist_pct", 0.03))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.slow + 5:
            return []

        df = df.copy()
        df["ema_f"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_s"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["atr"] = self._atr(df, 14)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        if atr <= 0:
            return []

        cross_up = prev["ema_f"] <= prev["ema_s"] and last["ema_f"] > last["ema_s"]
        cross_dn = prev["ema_f"] >= prev["ema_s"] and last["ema_f"] < last["ema_s"]

        # Giá không quá xa MA (setup yếu nếu đã chạy xa)
        dist = abs(float(last["close"] - last["ema_f"])) / float(last["close"])
        if dist > self.max_dist_pct:
            return []

        signals = []
        if cross_up and last["close"] > last["ema_s"]:
            entry = float(last["close"])
            swing_low = float(df["low"].iloc[-20:].min())
            sl = min(swing_low, entry - atr * self.atr_sl_mult)
            risk = entry - sl
            if risk > 0:
                tp = entry + risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.70,
                    strategy=self.name,
                    reason=f"Swing MA cross UP {self.fast}/{self.slow}",
                    timeframe=timeframe,
                    extra={"type": "golden_cross"},
                ))

        if cross_dn and last["close"] < last["ema_s"]:
            entry = float(last["close"])
            swing_high = float(df["high"].iloc[-20:].max())
            sl = max(swing_high, entry + atr * self.atr_sl_mult)
            risk = sl - entry
            if risk > 0:
                tp = entry - risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.70,
                    strategy=self.name,
                    reason=f"Swing MA cross DOWN {self.fast}/{self.slow}",
                    timeframe=timeframe,
                    extra={"type": "death_cross"},
                ))

        return signals[:1]

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
