"""
RSI Swing Strategy (Strike.money)
- Continuation: RSI pullback 30-40 in uptrend / 60-70 in downtrend
- Reversal: RSI OS/OB + nến đảo chiều
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class SwingRSIStrategy(BaseStrategy):
    name = "SwingRSI"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.ema_trend = int(self.params.get("ema_trend", 50))
        self.rr = float(self.params.get("rr", 2.5))
        self.atr_sl_mult = float(self.params.get("atr_sl_mult", 2.0))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.ema_trend + 5:
            return []

        df = df.copy()
        df["rsi"] = self._rsi(df["close"], self.rsi_period)
        df["ema"] = df["close"].ewm(span=self.ema_trend, adjust=False).mean()
        df["atr"] = self._atr(df, 14)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        rsi = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
        if atr <= 0:
            return []

        uptrend = last["close"] > last["ema"] and last["ema"] > df["ema"].iloc[-5]
        downtrend = last["close"] < last["ema"] and last["ema"] < df["ema"].iloc[-5]
        bull_candle = last["close"] > last["open"]
        bear_candle = last["close"] < last["open"]

        signals = []

        # Continuation long: uptrend + RSI 30-45 + bounce candle
        if uptrend and 30 <= rsi <= 45 and bull_candle and prev["rsi"] <= rsi:
            entry = float(last["close"])
            sl = float(df["low"].iloc[-15:].min()) - atr * 0.3
            risk = entry - sl
            if risk > 0:
                tp = entry + risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.68,
                    strategy=self.name,
                    reason=f"Swing RSI pullback LONG | RSI={rsi:.1f}",
                    timeframe=timeframe,
                    extra={"rsi": rsi, "type": "continuation"},
                ))

        # Continuation short
        if downtrend and 55 <= rsi <= 70 and bear_candle and prev["rsi"] >= rsi:
            entry = float(last["close"])
            sl = float(df["high"].iloc[-15:].max()) + atr * 0.3
            risk = sl - entry
            if risk > 0:
                tp = entry - risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.68,
                    strategy=self.name,
                    reason=f"Swing RSI pullback SHORT | RSI={rsi:.1f}",
                    timeframe=timeframe,
                    extra={"rsi": rsi, "type": "continuation"},
                ))

        # Reversal long: RSI < 30 + bullish candle
        if rsi < 30 and bull_candle and prev["rsi"] < 35:
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
                    confidence=0.62,
                    strategy=self.name,
                    reason=f"Swing RSI OS reversal LONG | RSI={rsi:.1f}",
                    timeframe=timeframe,
                    extra={"rsi": rsi, "type": "reversal"},
                ))

        if rsi > 70 and bear_candle and prev["rsi"] > 65:
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
                    confidence=0.62,
                    strategy=self.name,
                    reason=f"Swing RSI OB reversal SHORT | RSI={rsi:.1f}",
                    timeframe=timeframe,
                    extra={"rsi": rsi, "type": "reversal"},
                ))

        return signals[:1]

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
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
