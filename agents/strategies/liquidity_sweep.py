"""
Liquidity Sweep (approximation)
Tìm equal highs/lows → sweep (wick) → reclaim → entry.
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class LiquiditySweepStrategy(BaseStrategy):
    name = "LiquiditySweep"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 30)
        self.equal_tol = self.params.get("equal_tol", 0.0015)  # 0.15%
        self.rr = self.params.get("rr", 2.0)

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.lookback + 5:
            return []

        signals = []
        window = df.iloc[-self.lookback:]
        last = df.iloc[-1]

        # Equal highs
        highs = window["high"].values
        max_high = highs.max()
        equal_highs = np.where(np.abs(highs - max_high) / max_high < self.equal_tol)[0]

        if len(equal_highs) >= 2:
            # Sweep: last high > max_high nhưng close < max_high (wick)
            if last["high"] > max_high * (1 + self.equal_tol / 2) and last["close"] < max_high:
                entry = last["close"]
                sl = last["high"] * 1.001
                risk = sl - entry
                tp = entry - risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.70,
                    strategy=self.name,
                    reason="Liquidity Sweep Equal Highs → Reclaim",
                    timeframe=timeframe,
                ))

        # Equal lows
        lows = window["low"].values
        min_low = lows.min()
        equal_lows = np.where(np.abs(lows - min_low) / min_low < self.equal_tol)[0]

        if len(equal_lows) >= 2:
            if last["low"] < min_low * (1 - self.equal_tol / 2) and last["close"] > min_low:
                entry = last["close"]
                sl = last["low"] * 0.999
                risk = entry - sl
                tp = entry + risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.70,
                    strategy=self.name,
                    reason="Liquidity Sweep Equal Lows → Reclaim",
                    timeframe=timeframe,
                ))

        return signals
