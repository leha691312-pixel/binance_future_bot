"""
Fair Value Gap (FVG) - ICT concept
FVG bullish: low[i] > high[i-2]  (gap lên)
FVG bearish: high[i] < low[i-2]  (gap xuống)
Entry khi price retest vào FVG zone.
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class FVGStrategy(BaseStrategy):
    name = "FVG"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 40)
        self.min_gap_pct = self.params.get("min_gap_pct", 0.0015)  # 0.15%
        self.max_age = self.params.get("max_age", 25)
        self.rr = self.params.get("rr", 2.2)
        self.require_displacement = self.params.get("require_displacement", True)

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.lookback + 5:
            return []

        signals = []
        last = df.iloc[-1]
        atr = self._atr(df, 14).iloc[-1]
        if pd.isna(atr) or atr == 0:
            return []

        # Quét FVG gần đây (từ mới đến cũ)
        for i in range(len(df) - 3, max(len(df) - self.max_age - 3, 2), -1):
            c0 = df.iloc[i]      # nến tạo gap (nến 3)
            c1 = df.iloc[i - 1]  # nến giữa
            c2 = df.iloc[i - 2]  # nến 1

            # --- Bullish FVG: low của nến 3 > high của nến 1 ---
            if c0["low"] > c2["high"]:
                gap_size = (c0["low"] - c2["high"]) / c2["high"]
                if gap_size < self.min_gap_pct:
                    continue

                # Displacement filter (nến giữa hoặc nến 3 mạnh)
                if self.require_displacement:
                    body = abs(c1["close"] - c1["open"])
                    if body < atr * 0.8 and abs(c0["close"] - c0["open"]) < atr * 0.8:
                        continue

                zone_low = c2["high"]
                zone_high = c0["low"]

                # Price đang retest zone (wick hoặc close trong zone)
                in_zone = (
                    (zone_low <= last["low"] <= zone_high) or
                    (zone_low <= last["close"] <= zone_high) or
                    (last["low"] <= zone_low and last["close"] > zone_low)
                )
                if in_zone and last["close"] > zone_low:
                    entry = last["close"]
                    sl = zone_low * 0.997
                    risk = entry - sl
                    if risk <= 0:
                        continue
                    tp = entry + risk * self.rr
                    signals.append(Signal(
                        symbol=symbol,
                        side=Side.LONG,
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        confidence=min(0.78, 0.60 + gap_size * 30),
                        strategy=self.name,
                        reason=f"Bullish FVG retest | gap={gap_size:.2%} | zone={zone_low:.4f}-{zone_high:.4f}",
                        timeframe=timeframe,
                        extra={"fvg_low": zone_low, "fvg_high": zone_high, "gap_pct": gap_size},
                    ))
                    break  # lấy FVG gần nhất

            # --- Bearish FVG: high của nến 3 < low của nến 1 ---
            if c0["high"] < c2["low"]:
                gap_size = (c2["low"] - c0["high"]) / c0["high"]
                if gap_size < self.min_gap_pct:
                    continue

                if self.require_displacement:
                    body = abs(c1["close"] - c1["open"])
                    if body < atr * 0.8 and abs(c0["close"] - c0["open"]) < atr * 0.8:
                        continue

                zone_high = c2["low"]
                zone_low = c0["high"]

                in_zone = (
                    (zone_low <= last["high"] <= zone_high) or
                    (zone_low <= last["close"] <= zone_high) or
                    (last["high"] >= zone_high and last["close"] < zone_high)
                )
                if in_zone and last["close"] < zone_high:
                    entry = last["close"]
                    sl = zone_high * 1.003
                    risk = sl - entry
                    if risk <= 0:
                        continue
                    tp = entry - risk * self.rr
                    signals.append(Signal(
                        symbol=symbol,
                        side=Side.SHORT,
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        confidence=min(0.78, 0.60 + gap_size * 30),
                        strategy=self.name,
                        reason=f"Bearish FVG retest | gap={gap_size:.2%} | zone={zone_low:.4f}-{zone_high:.4f}",
                        timeframe=timeframe,
                        extra={"fvg_low": zone_low, "fvg_high": zone_high, "gap_pct": gap_size},
                    ))
                    break

        return signals

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
