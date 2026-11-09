"""
Opening Range Breakout (ORB)
Lấy range N nến đầu session, breakout + volume confirmation.
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side
from datetime import datetime


class ORBStrategy(BaseStrategy):
    name = "ORB"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.range_bars = self.params.get("range_bars", 4)      # 4 x 15m = 1h
        self.min_range_pct = self.params.get("min_range_pct", 0.003)
        self.volume_mult = self.params.get("volume_mult", 1.3)
        self.rr = self.params.get("rr", 2.0)

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.range_bars + 10:
            return []

        signals = []
        # Lấy range đầu
        orb_high = df["high"].iloc[:self.range_bars].max()
        orb_low = df["low"].iloc[:self.range_bars].min()
        range_pct = (orb_high - orb_low) / orb_low

        if range_pct < self.min_range_pct:
            return []

        # Nến hiện tại (last closed)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        avg_vol = df["volume"].iloc[-20:-1].mean()

        # Breakout up
        if last["close"] > orb_high and prev["close"] <= orb_high and last["volume"] > avg_vol * self.volume_mult:
            entry = last["close"]
            sl = orb_low
            risk = entry - sl
            tp = entry + risk * self.rr
            signals.append(Signal(
                symbol=symbol,
                side=Side.LONG,
                entry=entry,
                sl=sl,
                tp=tp,
                confidence=0.65,
                strategy=self.name,
                reason=f"ORB Breakout Up | range={range_pct:.2%}",
                timeframe=timeframe,
            ))

        # Breakout down
        if last["close"] < orb_low and prev["close"] >= orb_low and last["volume"] > avg_vol * self.volume_mult:
            entry = last["close"]
            sl = orb_high
            risk = sl - entry
            tp = entry - risk * self.rr
            signals.append(Signal(
                symbol=symbol,
                side=Side.SHORT,
                entry=entry,
                sl=sl,
                tp=tp,
                confidence=0.65,
                strategy=self.name,
                reason=f"ORB Breakout Down | range={range_pct:.2%}",
                timeframe=timeframe,
            ))

        return signals
