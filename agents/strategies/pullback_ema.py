"""
Pullback trading (Sarwa-style)
- Trend xác định bằng EMA slow + ADX
- Chờ hồi về EMA fast / slow
- Confirm nến + volume rồi vào theo trend
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class PullbackEMAStrategy(BaseStrategy):
    name = "PullbackEMA"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.ema_fast = int(self.params.get("ema_fast", 21))
        self.ema_slow = int(self.params.get("ema_slow", 50))
        self.adx_min = float(self.params.get("adx_min", 25))
        self.vol_mult = float(self.params.get("vol_mult", 1.1))
        self.rr = float(self.params.get("rr", 2.2))
        self.atr_sl_mult = float(self.params.get("atr_sl_mult", 1.3))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.ema_slow + 10:
            return []

        df = df.copy()
        df["ema_f"] = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_s"] = df["close"].ewm(span=self.ema_slow, adjust=False).mean()
        df["atr"] = self._atr(df, 14)
        df["adx"] = self._adx(df, 14)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        adx = float(last["adx"]) if not pd.isna(last["adx"]) else 0.0
        vol_ma = float(last["vol_ma"]) if not pd.isna(last["vol_ma"]) else 0.0
        if atr <= 0 or adx < self.adx_min:
            return []

        bull = last["ema_f"] > last["ema_s"] and last["close"] > last["ema_s"]
        bear = last["ema_f"] < last["ema_s"] and last["close"] < last["ema_s"]
        vol_ok = vol_ma <= 0 or last["volume"] >= vol_ma * self.vol_mult

        signals = []

        # Pullback long: trend up, nến trước chạm EMA, nến hiện tại đóng trên EMA + bullish
        if bull and vol_ok:
            touched = prev["low"] <= prev["ema_f"] * 1.002
            reclaim = last["close"] > last["ema_f"] and last["close"] > last["open"]
            if touched and reclaim:
                entry = float(last["close"])
                sl = min(float(prev["low"]), entry - atr * self.atr_sl_mult)
                risk = entry - sl
                if risk > 0:
                    tp = entry + risk * self.rr
                    signals.append(Signal(
                        symbol=symbol,
                        side=Side.LONG,
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        confidence=min(0.78, 0.60 + adx / 200),
                        strategy=self.name,
                        reason=f"Pullback LONG to EMA{self.ema_fast} | ADX={adx:.1f}",
                        timeframe=timeframe,
                        extra={"adx": adx, "ema": self.ema_fast},
                    ))

        if bear and vol_ok:
            touched = prev["high"] >= prev["ema_f"] * 0.998
            reject = last["close"] < last["ema_f"] and last["close"] < last["open"]
            if touched and reject:
                entry = float(last["close"])
                sl = max(float(prev["high"]), entry + atr * self.atr_sl_mult)
                risk = sl - entry
                if risk > 0:
                    tp = entry - risk * self.rr
                    signals.append(Signal(
                        symbol=symbol,
                        side=Side.SHORT,
                        entry=entry,
                        sl=sl,
                        tp=tp,
                        confidence=min(0.78, 0.60 + adx / 200),
                        strategy=self.name,
                        reason=f"Pullback SHORT to EMA{self.ema_fast} | ADX={adx:.1f}",
                        timeframe=timeframe,
                        extra={"adx": adx, "ema": self.ema_fast},
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
