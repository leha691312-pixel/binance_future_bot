"""
EMA Crossover + RSI filter (Investopedia scalping #7)
Fast/slow EMA cross + RSI > 50 long / < 50 short. Tight SL.
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class EMARSIScalpStrategy(BaseStrategy):
    name = "EMARSIScalp"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.fast = int(self.params.get("ema_fast", 8))
        self.slow = int(self.params.get("ema_slow", 21))
        self.rsi_period = int(self.params.get("rsi_period", 7))
        self.rr = float(self.params.get("rr", 1.5))
        self.atr_sl_mult = float(self.params.get("atr_sl_mult", 0.9))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.slow + 10:
            return []

        df = df.copy()
        df["ema_f"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_s"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["rsi"] = self._rsi(df["close"], self.rsi_period)
        df["atr"] = self._atr(df, 14)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        rsi = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
        if atr <= 0:
            return []

        cross_up = prev["ema_f"] <= prev["ema_s"] and last["ema_f"] > last["ema_s"]
        cross_dn = prev["ema_f"] >= prev["ema_s"] and last["ema_f"] < last["ema_s"]

        signals = []
        if cross_up and rsi > 50:
            entry = float(last["close"])
            sl = entry - atr * self.atr_sl_mult
            risk = entry - sl
            if risk > 0:
                tp = entry + risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.64,
                    strategy=self.name,
                    reason=f"EMA cross UP + RSI={rsi:.1f}",
                    timeframe=timeframe,
                    extra={"rsi": rsi},
                ))

        if cross_dn and rsi < 50:
            entry = float(last["close"])
            sl = entry + atr * self.atr_sl_mult
            risk = sl - entry
            if risk > 0:
                tp = entry - risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.64,
                    strategy=self.name,
                    reason=f"EMA cross DOWN + RSI={rsi:.1f}",
                    timeframe=timeframe,
                    extra={"rsi": rsi},
                ))

        return signals[:1]

    def _rsi(self, series: pd.Series, period: int = 7) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
