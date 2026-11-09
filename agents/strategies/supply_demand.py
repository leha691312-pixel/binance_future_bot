"""
Supply & Demand zones (simple version)
Zone tạo bởi nến base nhỏ + impulse mạnh đi ra.
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class SupplyDemandStrategy(BaseStrategy):
    name = "SupplyDemand"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = self.params.get("lookback", 40)
        self.base_max_body = self.params.get("base_max_body", 0.4)  # body < 40% of ATR
        self.impulse_min = self.params.get("impulse_min", 1.8)
        self.rr = self.params.get("rr", 2.0)

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.lookback + 5:
            return []

        df = df.copy()
        df["body"] = (df["close"] - df["open"]).abs()
        df["atr"] = self._atr(df, 14)
        signals = []
        last = df.iloc[-1]

        for i in range(len(df) - 15, len(df) - 3):
            if i < 3:
                continue
            row = df.iloc[i]
            atr = row["atr"]
            if pd.isna(atr) or atr == 0:
                continue

            # Demand: small base then strong up
            if row["body"] < atr * self.base_max_body:
                next_move = df.iloc[i+1:i+4]["close"].max() - row["close"]
                if next_move > atr * self.impulse_min:
                    zone_low = min(row["low"], df.iloc[i+1]["low"])
                    zone_high = max(row["high"], df.iloc[i+1]["high"])
                    if zone_low <= last["low"] <= zone_high * 1.002:
                        entry = last["close"]
                        sl = zone_low * 0.997
                        risk = entry - sl
                        if risk > 0:
                            tp = entry + risk * self.rr
                            signals.append(Signal(
                                symbol=symbol,
                                side=Side.LONG,
                                entry=entry,
                                sl=sl,
                                tp=tp,
                                confidence=0.62,
                                strategy=self.name,
                                reason="Demand zone retest",
                                timeframe=timeframe,
                            ))
                            break

            # Supply: small base then strong down
            if row["body"] < atr * self.base_max_body:
                next_move = row["close"] - df.iloc[i+1:i+4]["close"].min()
                if next_move > atr * self.impulse_min:
                    zone_low = min(row["low"], df.iloc[i+1]["low"])
                    zone_high = max(row["high"], df.iloc[i+1]["high"])
                    if zone_low * 0.998 <= last["high"] <= zone_high:
                        entry = last["close"]
                        sl = zone_high * 1.003
                        risk = sl - entry
                        if risk > 0:
                            tp = entry - risk * self.rr
                            signals.append(Signal(
                                symbol=symbol,
                                side=Side.SHORT,
                                entry=entry,
                                sl=sl,
                                tp=tp,
                                confidence=0.62,
                                strategy=self.name,
                                reason="Supply zone retest",
                                timeframe=timeframe,
                            ))
                            break

        return signals[:1]

    def _atr(self, df, period=14):
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
