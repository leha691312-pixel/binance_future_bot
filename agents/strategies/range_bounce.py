"""
Range trading (Sarwa-style)
- Xác định range bằng swing high/low gần đây
- Chỉ trade khi ADX thấp (không trend mạnh)
- Long gần support + RSI oversold; Short gần resistance + RSI overbought
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class RangeBounceStrategy(BaseStrategy):
    name = "RangeBounce"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = int(self.params.get("lookback", 40))
        self.adx_max = float(self.params.get("adx_max", 22))  # range khi ADX thấp
        self.zone_pct = float(self.params.get("zone_pct", 0.15))  # % chiều cao range
        self.rsi_low = float(self.params.get("rsi_low", 35))
        self.rsi_high = float(self.params.get("rsi_high", 65))
        self.rr = float(self.params.get("rr", 1.8))
        self.min_range_pct = float(self.params.get("min_range_pct", 0.008))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.lookback + 10:
            return []

        df = df.copy()
        df["rsi"] = self._rsi(df["close"], 14)
        df["adx"] = self._adx(df, 14)

        window = df.iloc[-self.lookback:-1]
        last = df.iloc[-1]
        adx = float(last["adx"]) if not pd.isna(last["adx"]) else 99.0
        rsi = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0

        # Chỉ range market
        if adx > self.adx_max:
            return []

        resistance = float(window["high"].max())
        support = float(window["low"].min())
        height = resistance - support
        if support <= 0 or height / support < self.min_range_pct:
            return []

        zone = height * (self.zone_pct / 100.0) if self.zone_pct > 1 else height * self.zone_pct
        # zone_pct config: 0.15 means 15% of range height if >1 else fraction
        if self.zone_pct >= 1:
            zone = height * (self.zone_pct / 100.0)
        else:
            zone = height * self.zone_pct

        price = float(last["close"])
        signals = []

        # Long gần support
        if price <= support + zone and rsi <= self.rsi_low and last["close"] > last["open"]:
            entry = price
            sl = support * 0.997
            risk = entry - sl
            if risk > 0:
                # TP hướng về giữa range hoặc resistance
                tp = min(entry + risk * self.rr, resistance - zone)
                if tp > entry:
                    signals.append(Signal(
                        symbol=symbol,
                        side=Side.LONG,
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        confidence=0.64,
                        strategy=self.name,
                        reason=f"Range LONG support={support:.4f} RSI={rsi:.1f} ADX={adx:.1f}",
                        timeframe=timeframe,
                        extra={"support": support, "resistance": resistance, "rsi": rsi},
                    ))

        # Short gần resistance
        if price >= resistance - zone and rsi >= self.rsi_high and last["close"] < last["open"]:
            entry = price
            sl = resistance * 1.003
            risk = sl - entry
            if risk > 0:
                tp = max(entry - risk * self.rr, support + zone)
                if tp < entry:
                    signals.append(Signal(
                        symbol=symbol,
                        side=Side.SHORT,
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        confidence=0.64,
                        strategy=self.name,
                        reason=f"Range SHORT resist={resistance:.4f} RSI={rsi:.1f} ADX={adx:.1f}",
                        timeframe=timeframe,
                        extra={"support": support, "resistance": resistance, "rsi": rsi},
                    ))

        return signals[:1]

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
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
