"""
Bollinger Bands Squeeze – Swing (Strike.money)
Bandwidth hẹp (squeeze) → breakout nến đóng ngoài band + volume.
"""
from typing import List
import pandas as pd
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class BBSqueezeStrategy(BaseStrategy):
    name = "BBSqueeze"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.bb_period = int(self.params.get("bb_period", 20))
        self.bb_std = float(self.params.get("bb_std", 2.0))
        self.squeeze_lookback = int(self.params.get("squeeze_lookback", 40))
        self.squeeze_pctile = float(self.params.get("squeeze_pctile", 0.25))  # bandwidth thấp nhất 25%
        self.vol_mult = float(self.params.get("vol_mult", 1.2))
        self.rr = float(self.params.get("rr", 2.5))
        self.atr_sl_mult = float(self.params.get("atr_sl_mult", 1.8))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < max(self.bb_period, self.squeeze_lookback) + 5:
            return []

        df = df.copy()
        mid = df["close"].rolling(self.bb_period).mean()
        std = df["close"].rolling(self.bb_period).std()
        df["bb_up"] = mid + self.bb_std * std
        df["bb_lo"] = mid - self.bb_std * std
        df["bw"] = (df["bb_up"] - df["bb_lo"]) / mid.replace(0, pd.NA)
        df["atr"] = self._atr(df, 14)
        df["vol_ma"] = df["volume"].rolling(20).mean()

        last = df.iloc[-1]
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
        bw = float(last["bw"]) if not pd.isna(last["bw"]) else 1.0
        if atr <= 0:
            return []

        # Squeeze: bandwidth hiện tại trong nhóm thấp của lookback
        hist_bw = df["bw"].iloc[-self.squeeze_lookback:-1].dropna()
        if len(hist_bw) < 10:
            return []
        threshold = float(hist_bw.quantile(self.squeeze_pctile))
        was_squeeze = bw <= threshold or float(df["bw"].iloc[-3:-1].min()) <= threshold

        if not was_squeeze:
            return []

        vol_ok = True
        if not pd.isna(last["vol_ma"]) and last["vol_ma"] > 0:
            vol_ok = last["volume"] >= last["vol_ma"] * self.vol_mult

        signals = []
        # Breakout up
        if last["close"] > last["bb_up"] and last["close"] > last["open"] and vol_ok:
            entry = float(last["close"])
            sl = float(mid.iloc[-1]) if not pd.isna(mid.iloc[-1]) else entry - atr * self.atr_sl_mult
            sl = min(sl, entry - atr * 0.5)
            risk = entry - sl
            if risk > 0:
                tp = entry + risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.69,
                    strategy=self.name,
                    reason=f"BB Squeeze breakout UP | bw={bw:.4f}",
                    timeframe=timeframe,
                    extra={"bandwidth": bw},
                ))

        if last["close"] < last["bb_lo"] and last["close"] < last["open"] and vol_ok:
            entry = float(last["close"])
            sl = float(mid.iloc[-1]) if not pd.isna(mid.iloc[-1]) else entry + atr * self.atr_sl_mult
            sl = max(sl, entry + atr * 0.5)
            risk = sl - entry
            if risk > 0:
                tp = entry - risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.69,
                    strategy=self.name,
                    reason=f"BB Squeeze breakout DOWN | bw={bw:.4f}",
                    timeframe=timeframe,
                    extra={"bandwidth": bw},
                ))

        return signals[:1]

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
