"""
Trend trading (Sarwa-style trend following)
- EMA fast/slow xác định trend
- ADX filter sức mạnh trend
- Entry khi giá/EMA confirm continuation
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class TrendMAStrategy(BaseStrategy):
    name = "TrendMA"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.fast = int(self.params.get("ema_fast", 9))
        self.slow = int(self.params.get("ema_slow", 21))
        self.adx_period = int(self.params.get("adx_period", 14))
        self.adx_min = float(self.params.get("adx_min", 25))
        self.rr = float(self.params.get("rr", 2.0))
        self.atr_sl_mult = float(self.params.get("atr_sl_mult", 1.5))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        need = max(self.slow, self.adx_period) + 5
        if len(df) < need:
            return []

        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["atr"] = self._atr(df, 14)
        df["adx"] = self._adx(df, self.adx_period)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        adx = float(last["adx"]) if not pd.isna(last["adx"]) else 0.0
        if atr <= 0 or adx < self.adx_min:
            return []

        signals = []

        # Bullish trend: fast > slow, ADX mạnh, giá trên fast, cross gần đây hoặc bounce
        bull_trend = last["ema_fast"] > last["ema_slow"]
        bear_trend = last["ema_fast"] < last["ema_slow"]

        # Cross up trong 2 nến gần nhất
        cross_up = prev["ema_fast"] <= prev["ema_slow"] and last["ema_fast"] > last["ema_slow"]
        cross_dn = prev["ema_fast"] >= prev["ema_slow"] and last["ema_fast"] < last["ema_slow"]

        # Price reclaim EMA fast trong trend
        bounce_long = (
            bull_trend
            and prev["low"] <= prev["ema_fast"]
            and last["close"] > last["ema_fast"]
            and last["close"] > last["open"]
        )
        bounce_short = (
            bear_trend
            and prev["high"] >= prev["ema_fast"]
            and last["close"] < last["ema_fast"]
            and last["close"] < last["open"]
        )

        if (cross_up or bounce_long) and bull_trend:
            entry = float(last["close"])
            sl = entry - atr * self.atr_sl_mult
            risk = entry - sl
            if risk > 0:
                tp = entry + risk * self.rr
                conf = min(0.82, 0.58 + (adx - self.adx_min) / 100)
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=conf,
                    strategy=self.name,
                    reason=f"TrendMA LONG | ADX={adx:.1f} | {'cross' if cross_up else 'bounce'}",
                    timeframe=timeframe,
                    extra={"adx": adx, "type": "cross" if cross_up else "bounce"},
                ))

        if (cross_dn or bounce_short) and bear_trend:
            entry = float(last["close"])
            sl = entry + atr * self.atr_sl_mult
            risk = sl - entry
            if risk > 0:
                tp = entry - risk * self.rr
                conf = min(0.82, 0.58 + (adx - self.adx_min) / 100)
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=conf,
                    strategy=self.name,
                    reason=f"TrendMA SHORT | ADX={adx:.1f} | {'cross' if cross_dn else 'bounce'}",
                    timeframe=timeframe,
                    extra={"adx": adx, "type": "cross" if cross_dn else "bounce"},
                ))

        return signals[:1]

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        up = high.diff()
        down = -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr.replace(0, np.nan)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        return dx.rolling(period).mean()
