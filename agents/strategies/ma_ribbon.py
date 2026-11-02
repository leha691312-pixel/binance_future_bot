"""
Moving Average Ribbon Entry (Investopedia scalping)
SMA 5 / 8 / 13 — realign + spread = entry theo trend ngắn hạn.
"""
from typing import List
import pandas as pd
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class MARibbonStrategy(BaseStrategy):
    name = "MARibbon"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.p1 = int(self.params.get("sma1", 5))
        self.p2 = int(self.params.get("sma2", 8))
        self.p3 = int(self.params.get("sma3", 13))
        self.rr = float(self.params.get("rr", 1.8))
        self.atr_sl_mult = float(self.params.get("atr_sl_mult", 1.0))
        self.min_spread_pct = float(self.params.get("min_spread_pct", 0.0003))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.p3 + 5:
            return []

        df = df.copy()
        df["s1"] = df["close"].rolling(self.p1).mean()
        df["s2"] = df["close"].rolling(self.p2).mean()
        df["s3"] = df["close"].rolling(self.p3).mean()
        df["atr"] = self._atr(df, 14)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        if atr <= 0 or pd.isna(last["s3"]):
            return []

        # Ribbon aligned up / down
        bull_now = last["s1"] > last["s2"] > last["s3"]
        bear_now = last["s1"] < last["s2"] < last["s3"]
        bull_prev = prev["s1"] > prev["s2"] > prev["s3"]
        bear_prev = prev["s1"] < prev["s2"] < prev["s3"]

        spread = abs(float(last["s1"] - last["s3"])) / float(last["close"])
        if spread < self.min_spread_pct:
            return []

        # Realign: vừa xếp hàng lại sau khi flat/crisscross
        realign_up = bull_now and not bull_prev
        realign_dn = bear_now and not bear_prev

        # Price glued to fast SMA in strong trend
        near_fast_long = abs(last["close"] - last["s1"]) / last["close"] < 0.002
        near_fast_short = near_fast_long

        signals = []
        if (realign_up or (bull_now and near_fast_long and last["close"] > last["open"])):
            if bull_now:
                entry = float(last["close"])
                sl = min(float(last["s3"]), entry - atr * self.atr_sl_mult)
                risk = entry - sl
                if risk > 0:
                    tp = entry + risk * self.rr
                    signals.append(Signal(
                        symbol=symbol,
                        side=Side.LONG,
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        confidence=0.66,
                        strategy=self.name,
                        reason=f"MA Ribbon LONG 5-8-13 align | spread={spread:.3%}",
                        timeframe=timeframe,
                        extra={"type": "ribbon_up"},
                    ))

        if (realign_dn or (bear_now and near_fast_short and last["close"] < last["open"])):
            if bear_now:
                entry = float(last["close"])
                sl = max(float(last["s3"]), entry + atr * self.atr_sl_mult)
                risk = sl - entry
                if risk > 0:
                    tp = entry - risk * self.rr
                    signals.append(Signal(
                        symbol=symbol,
                        side=Side.SHORT,
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        confidence=0.66,
                        strategy=self.name,
                        reason=f"MA Ribbon SHORT 5-8-13 align | spread={spread:.3%}",
                        timeframe=timeframe,
                        extra={"type": "ribbon_down"},
                    ))

        return signals[:1]

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
