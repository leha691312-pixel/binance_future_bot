"""
Pivot Point Scalping (Investopedia)
Pivot low → long khi giá vượt nhẹ; pivot high → short khi giá gãy xuống.
"""
from typing import List
import pandas as pd
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class PivotScalpStrategy(BaseStrategy):
    name = "PivotScalp"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.left = int(self.params.get("left", 2))
        self.right = int(self.params.get("right", 2))
        self.trigger_pct = float(self.params.get("trigger_pct", 0.0005))
        self.rr = float(self.params.get("rr", 1.6))
        self.atr_sl_mult = float(self.params.get("atr_sl_mult", 1.0))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        n = self.left + self.right + 1
        if len(df) < n + 10:
            return []

        df = df.copy()
        df["atr"] = self._atr(df, 14)
        last = df.iloc[-1]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        if atr <= 0:
            return []

        # Pivot tại index i = len-1-right (vừa confirm đủ right bars)
        i = len(df) - 1 - self.right
        if i < self.left:
            return []

        window_high = df["high"].iloc[i - self.left: i + self.right + 1]
        window_low = df["low"].iloc[i - self.left: i + self.right + 1]
        pivot_high = float(df["high"].iloc[i]) >= float(window_high.max())
        pivot_low = float(df["low"].iloc[i]) <= float(window_low.min())

        price = float(last["close"])
        ph = float(df["high"].iloc[i])
        pl = float(df["low"].iloc[i])
        signals = []

        if pivot_low and price > pl * (1 + self.trigger_pct) and last["close"] > last["open"]:
            entry = price
            sl = pl - atr * 0.2
            risk = entry - sl
            if risk > 0:
                tp = entry + risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.65,
                    strategy=self.name,
                    reason=f"Pivot LOW long | pivot={pl:.6f}",
                    timeframe=timeframe,
                    extra={"pivot": pl, "type": "pivot_low"},
                ))

        if pivot_high and price < ph * (1 - self.trigger_pct) and last["close"] < last["open"]:
            entry = price
            sl = ph + atr * 0.2
            risk = sl - entry
            if risk > 0:
                tp = entry - risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.65,
                    strategy=self.name,
                    reason=f"Pivot HIGH short | pivot={ph:.6f}",
                    timeframe=timeframe,
                    extra={"pivot": ph, "type": "pivot_high"},
                ))

        return signals[:1]

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
