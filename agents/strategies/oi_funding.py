"""
Open Interest + Funding Rate Divergence Strategy
- OI tăng + price tăng → continuation (aggressive)
- OI tăng + price giảm → short squeeze potential / bearish divergence
- Funding cực dương + price cao → short bias (overleveraged longs)
- Funding cực âm + price thấp → long bias
"""
from typing import List, Optional
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class OIFundingStrategy(BaseStrategy):
    name = "OI_Funding"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.oi_lookback = self.params.get("oi_lookback", 12)       # bars
        self.funding_extreme = self.params.get("funding_extreme", 0.0008)  # 0.08%
        self.rr = self.params.get("rr", 2.0)
        self.min_oi_change = self.params.get("min_oi_change", 0.03)  # 3%

    def generate(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        oi_series: Optional[pd.Series] = None,
        funding_rate: Optional[float] = None,
    ) -> List[Signal]:
        """
        oi_series: optional Series of open interest (aligned with df)
        funding_rate: latest funding rate (float)
        Nếu không có OI/Funding → strategy bỏ qua (return [])
        """
        df = self._ensure_columns(df)
        if len(df) < 30:
            return []

        signals = []
        last = df.iloc[-1]
        atr = self._atr(df, 14).iloc[-1]
        if pd.isna(atr) or atr == 0:
            return []

        # Price change gần đây
        price_change = (last["close"] - df["close"].iloc[-self.oi_lookback]) / df["close"].iloc[-self.oi_lookback]

        # --- Funding extreme ---
        if funding_rate is not None:
            # Funding rất dương → longs overleveraged → bias SHORT
            if funding_rate > self.funding_extreme and price_change > 0.01:
                entry = last["close"]
                sl = entry + atr * 1.4
                risk = sl - entry
                tp = entry - risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.66,
                    strategy=self.name,
                    reason=f"Funding extreme long ({funding_rate:.4%}) + price up → SHORT bias",
                    timeframe=timeframe,
                    extra={"funding": funding_rate, "type": "funding_extreme"},
                ))

            # Funding rất âm → shorts overleveraged → bias LONG
            if funding_rate < -self.funding_extreme and price_change < -0.01:
                entry = last["close"]
                sl = entry - atr * 1.4
                risk = entry - sl
                tp = entry + risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.66,
                    strategy=self.name,
                    reason=f"Funding extreme short ({funding_rate:.4%}) + price down → LONG bias",
                    timeframe=timeframe,
                    extra={"funding": funding_rate, "type": "funding_extreme"},
                ))

        # --- OI Divergence ---
        if oi_series is not None and len(oi_series) >= self.oi_lookback:
            oi_now = oi_series.iloc[-1]
            oi_prev = oi_series.iloc[-self.oi_lookback]
            if oi_prev > 0:
                oi_change = (oi_now - oi_prev) / oi_prev
            else:
                oi_change = 0.0

            # OI tăng mạnh + price giảm → bearish divergence (possible continuation down hoặc squeeze)
            if oi_change > self.min_oi_change and price_change < -0.008:
                entry = last["close"]
                sl = entry + atr * 1.3
                risk = sl - entry
                tp = entry - risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.70,
                    strategy=self.name,
                    reason=f"OI↑ {oi_change:.1%} + Price↓ → bearish divergence",
                    timeframe=timeframe,
                    extra={"oi_change": oi_change, "type": "oi_divergence"},
                ))

            # OI tăng mạnh + price tăng → bullish continuation
            if oi_change > self.min_oi_change and price_change > 0.008:
                entry = last["close"]
                sl = entry - atr * 1.3
                risk = entry - sl
                tp = entry + risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.68,
                    strategy=self.name,
                    reason=f"OI↑ {oi_change:.1%} + Price↑ → bullish continuation",
                    timeframe=timeframe,
                    extra={"oi_change": oi_change, "type": "oi_continuation"},
                ))

            # OI giảm + price tăng → weak rally (possible short)
            if oi_change < -self.min_oi_change and price_change > 0.01:
                entry = last["close"]
                sl = entry + atr * 1.2
                risk = sl - entry
                tp = entry - risk * self.rr
                signals.append(Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=0.63,
                    strategy=self.name,
                    reason=f"OI↓ {oi_change:.1%} + Price↑ → weak rally",
                    timeframe=timeframe,
                    extra={"oi_change": oi_change, "type": "oi_weak_rally"},
                ))

        return signals[:1]

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
