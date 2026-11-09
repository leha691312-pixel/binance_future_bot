"""
Stochastic + Bollinger Band scalping (Investopedia)
Entry: stoch turn from OB/OS + price near BB extreme.
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class StochBBStrategy(BaseStrategy):
    name = "StochBB"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.k_period = int(self.params.get("k_period", 5))
        self.d_period = int(self.params.get("d_period", 3))
        self.smooth = int(self.params.get("smooth", 3))
        self.bb_period = int(self.params.get("bb_period", 13))
        self.bb_std = float(self.params.get("bb_std", 3.0))
        self.os = float(self.params.get("oversold", 20))
        self.ob = float(self.params.get("overbought", 80))
        self.rr = float(self.params.get("rr", 1.6))
        self.atr_sl_mult = float(self.params.get("atr_sl_mult", 1.0))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        need = max(self.k_period, self.bb_period) + self.d_period + 5
        if len(df) < need:
            return []

        df = df.copy()
        low_min = df["low"].rolling(self.k_period).min()
        high_max = df["high"].rolling(self.k_period).max()
        raw_k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        df["stoch_k"] = raw_k.rolling(self.smooth).mean()
        df["stoch_d"] = df["stoch_k"].rolling(self.d_period).mean()

        mid = df["close"].rolling(self.bb_period).mean()
        std = df["close"].rolling(self.bb_period).std()
        df["bb_up"] = mid + self.bb_std * std
        df["bb_lo"] = mid - self.bb_std * std
        df["atr"] = self._atr(df, 14)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        if atr <= 0 or pd.isna(last["stoch_k"]):
            return []

        k, d = float(last["stoch_k"]), float(last["stoch_d"])
        pk, pd_ = float(prev["stoch_k"]), float(prev["stoch_d"])

        # Stoch turn up from oversold / turn down from overbought
        turn_up = pk < self.os and k > pk and k > d
        turn_dn = pk > self.ob and k < pk and k < d

        near_lo = last["low"] <= last["bb_lo"] * 1.002 or last["close"] <= last["bb_lo"] * 1.005
        near_up = last["high"] >= last["bb_up"] * 0.998 or last["close"] >= last["bb_up"] * 0.995

        signals = []
        if turn_up and near_lo:
            entry = float(last["close"])
            sl = entry - atr * self.atr_sl_mult
            risk = entry - sl
            if risk > 0:
                # TP hướng về mid band hoặc RR
                tp = min(entry + risk * self.rr, float(mid.iloc[-1]) if not pd.isna(mid.iloc[-1]) else entry + risk * self.rr)
                if tp > entry:
                    signals.append(Signal(
                        symbol=symbol,
                        side=Side.LONG,
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        confidence=0.68,
                        strategy=self.name,
                        reason=f"StochBB LONG | K={k:.1f} near lower BB",
                        timeframe=timeframe,
                        extra={"stoch_k": k},
                    ))

        if turn_dn and near_up:
            entry = float(last["close"])
            sl = entry + atr * self.atr_sl_mult
            risk = sl - entry
            if risk > 0:
                tp = max(entry - risk * self.rr, float(mid.iloc[-1]) if not pd.isna(mid.iloc[-1]) else entry - risk * self.rr)
                if tp < entry:
                    signals.append(Signal(
                        symbol=symbol,
                        side=Side.SHORT,
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        confidence=0.68,
                        strategy=self.name,
                        reason=f"StochBB SHORT | K={k:.1f} near upper BB",
                        timeframe=timeframe,
                        extra={"stoch_k": k},
                    ))

        return signals[:1]

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
