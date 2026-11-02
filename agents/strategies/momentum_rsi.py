"""
Momentum trading (Sarwa-style)
- RSI cross mid-line / extreme reclaim
- MACD histogram confirm
- ADX optional filter
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class MomentumRSIStrategy(BaseStrategy):
    name = "MomentumRSI"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.rsi_mid = float(self.params.get("rsi_mid", 50))
        self.macd_fast = int(self.params.get("macd_fast", 12))
        self.macd_slow = int(self.params.get("macd_slow", 26))
        self.macd_signal = int(self.params.get("macd_signal", 9))
        self.adx_min = float(self.params.get("adx_min", 20))
        self.rr = float(self.params.get("rr", 2.0))
        self.atr_sl_mult = float(self.params.get("atr_sl_mult", 1.4))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.macd_slow + self.macd_signal + 5:
            return []

        df = df.copy()
        df["rsi"] = self._rsi(df["close"], self.rsi_period)
        ema_f = df["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_s = df["close"].ewm(span=self.macd_slow, adjust=False).mean()
        df["macd"] = ema_f - ema_s
        df["macd_sig"] = df["macd"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_sig"]
        df["atr"] = self._atr(df, 14)
        df["adx"] = self._adx(df, 14)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        adx = float(last["adx"]) if not pd.isna(last["adx"]) else 0.0
        rsi = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
        if atr <= 0:
            return []
        if adx < self.adx_min:
            return []

        signals = []

        # RSI cross up mid + MACD hist > 0 và đang tăng
        rsi_up = prev["rsi"] <= self.rsi_mid and last["rsi"] > self.rsi_mid
        rsi_dn = prev["rsi"] >= self.rsi_mid and last["rsi"] < self.rsi_mid
        macd_bull = last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]
        macd_bear = last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"]

        if rsi_up and macd_bull and last["close"] > last["open"]:
            entry = float(last["close"])
            sl = entry - atr * self.atr_sl_mult
            risk = entry - sl
            if risk > 0:
                tp = entry + risk * self.rr
                conf = min(0.80, 0.55 + abs(rsi - 50) / 200 + adx / 200)
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=conf,
                    strategy=self.name,
                    reason=f"Momentum LONG | RSI={rsi:.1f} ADX={adx:.1f}",
                    timeframe=timeframe,
                    extra={"rsi": rsi, "adx": adx},
                ))

        if rsi_dn and macd_bear and last["close"] < last["open"]:
            entry = float(last["close"])
            sl = entry + atr * self.atr_sl_mult
            risk = sl - entry
            if risk > 0:
                tp = entry - risk * self.rr
                conf = min(0.80, 0.55 + abs(rsi - 50) / 200 + adx / 200)
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=conf,
                    strategy=self.name,
                    reason=f"Momentum SHORT | RSI={rsi:.1f} ADX={adx:.1f}",
                    timeframe=timeframe,
                    extra={"rsi": rsi, "adx": adx},
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
