"""
Order Block + Fair Value Gap (approximation of ICT/SMC concepts)
Tìm impulse mạnh → OB là nến đối ứng cuối cùng trước impulse.
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class OrderBlockStrategy(BaseStrategy):
    name = "OrderBlock"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.impulse_mult = self.params.get("impulse_mult", 2.2)  # body > ATR * mult
        self.lookback = self.params.get("lookback", 50)
        self.rr = self.params.get("rr", 2.5)
        self.max_ob_age = self.params.get("max_ob_age", 20)

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.lookback + 5:
            return []

        df = df.copy()
        df["body"] = (df["close"] - df["open"]).abs()
        df["atr"] = self._atr(df, 14)
        signals = []

        # Tìm impulse gần đây
        for i in range(len(df) - self.max_ob_age - 2, len(df) - 3):
            if i < 5:
                continue
            row = df.iloc[i]
            atr = row["atr"]
            if atr == 0 or pd.isna(atr):
                continue

            # Bullish impulse (strong green)
            if row["close"] > row["open"] and row["body"] > atr * self.impulse_mult:
                # OB = last red candle before impulse
                for j in range(i - 1, max(i - 6, 0), -1):
                    prev = df.iloc[j]
                    if prev["close"] < prev["open"]:  # bearish candle
                        ob_high = prev["high"]
                        ob_low = prev["low"]
                        # Price currently retesting OB zone
                        last = df.iloc[-1]
                        if ob_low <= last["low"] <= ob_high or ob_low <= last["close"] <= ob_high:
                            entry = last["close"]
                            sl = ob_low * 0.998
                            risk = entry - sl
                            if risk <= 0:
                                break
                            tp = entry + risk * self.rr
                            signals.append(Signal(
                                symbol=symbol,
                                side=Side.LONG,
                                entry=entry,
                                sl=sl,
                                tp=tp,
                                confidence=0.68,
                                strategy=self.name,
                                reason=f"Bullish OB retest @ {ob_low:.4f}-{ob_high:.4f}",
                                timeframe=timeframe,
                            ))
                        break

            # Bearish impulse
            if row["close"] < row["open"] and row["body"] > atr * self.impulse_mult:
                for j in range(i - 1, max(i - 6, 0), -1):
                    prev = df.iloc[j]
                    if prev["close"] > prev["open"]:
                        ob_high = prev["high"]
                        ob_low = prev["low"]
                        last = df.iloc[-1]
                        if ob_low <= last["high"] <= ob_high or ob_low <= last["close"] <= ob_high:
                            entry = last["close"]
                            sl = ob_high * 1.002
                            risk = sl - entry
                            if risk <= 0:
                                break
                            tp = entry - risk * self.rr
                            signals.append(Signal(
                                symbol=symbol,
                                side=Side.SHORT,
                                entry=entry,
                                sl=sl,
                                tp=tp,
                                confidence=0.68,
                                strategy=self.name,
                                reason=f"Bearish OB retest @ {ob_low:.4f}-{ob_high:.4f}",
                                timeframe=timeframe,
                            ))
                        break

        # Chỉ lấy signal mới nhất nếu có nhiều
        if signals:
            return [signals[-1]]
        return []

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
