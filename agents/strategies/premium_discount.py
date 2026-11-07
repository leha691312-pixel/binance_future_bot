"""
Premium / Discount Zone (ICT-style)

- Xác định range gần đây (swing high / swing low)
- Equilibrium (EQ) = 50% range
- Discount: giá dưới EQ → ưu tiên LONG
- Premium: giá trên EQ → ưu tiên SHORT
- Entry khi giá ở vùng sâu (discount ≤ 30% / premium ≥ 70%) + nến rejection
"""
from typing import List
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class PremiumDiscountStrategy(BaseStrategy):
    name = "PremiumDiscount"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.lookback = int(self.params.get("lookback", 50))
        self.discount_max = float(self.params.get("discount_max", 0.35))  # ≤ 35% range
        self.premium_min = float(self.params.get("premium_min", 0.65))  # ≥ 65% range
        self.rr = float(self.params.get("rr", 2.0))
        self.min_range_pct = float(self.params.get("min_range_pct", 0.008))  # 0.8%
        self.sl_buffer_atr = float(self.params.get("sl_buffer_atr", 0.3))
        self.require_rejection = bool(self.params.get("require_rejection", True))

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < self.lookback + 5:
            return []

        window = df.iloc[-self.lookback :]
        swing_high = float(window["high"].max())
        swing_low = float(window["low"].min())
        rng = swing_high - swing_low
        if rng <= 0:
            return []

        mid = (swing_high + swing_low) / 2.0
        range_pct = rng / mid if mid > 0 else 0.0
        if range_pct < self.min_range_pct:
            return []

        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(last["close"])
        high = float(last["high"])
        low = float(last["low"])
        open_ = float(last["open"])

        # Vị trí trong range: 0 = low, 1 = high
        pos = (close - swing_low) / rng
        atr = float(self._atr(df, 14).iloc[-1] or 0)
        if atr <= 0:
            atr = rng * 0.05

        signals: List[Signal] = []

        # --- DISCOUNT → LONG ---
        if pos <= self.discount_max:
            if self.require_rejection and not self._bullish_rejection(open_, high, low, close, prev):
                return []
            entry = close
            sl = swing_low - atr * self.sl_buffer_atr
            risk = entry - sl
            if risk <= 0:
                return []
            # TP ưu tiên EQ, tối thiểu RR
            tp_eq = mid
            tp_rr = entry + risk * self.rr
            tp = max(tp_eq, tp_rr) if tp_eq > entry else tp_rr
            # Không vượt quá swing high quá nhiều
            tp = min(tp, swing_high + atr * 0.2)

            depth = 1.0 - pos  # càng thấp càng sâu discount
            conf = min(0.88, 0.55 + depth * 0.25 + (0.05 if close > open_ else 0))
            signals.append(
                Signal(
                    symbol=symbol,
                    side=Side.LONG,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=conf,
                    strategy=self.name,
                    reason=(
                        f"DISCOUNT pos={pos:.2f} EQ={mid:.6g} "
                        f"range={swing_low:.6g}-{swing_high:.6g}"
                    ),
                    timeframe=timeframe,
                    extra={
                        "zone": "discount",
                        "eq": mid,
                        "swing_high": swing_high,
                        "swing_low": swing_low,
                        "pos": pos,
                    },
                )
            )
            return signals

        # --- PREMIUM → SHORT ---
        if pos >= self.premium_min:
            if self.require_rejection and not self._bearish_rejection(open_, high, low, close, prev):
                return []
            entry = close
            sl = swing_high + atr * self.sl_buffer_atr
            risk = sl - entry
            if risk <= 0:
                return []
            tp_eq = mid
            tp_rr = entry - risk * self.rr
            tp = min(tp_eq, tp_rr) if tp_eq < entry else tp_rr
            tp = max(tp, swing_low - atr * 0.2)

            depth = pos
            conf = min(0.88, 0.55 + depth * 0.25 + (0.05 if close < open_ else 0))
            signals.append(
                Signal(
                    symbol=symbol,
                    side=Side.SHORT,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    confidence=conf,
                    strategy=self.name,
                    reason=(
                        f"PREMIUM pos={pos:.2f} EQ={mid:.6g} "
                        f"range={swing_low:.6g}-{swing_high:.6g}"
                    ),
                    timeframe=timeframe,
                    extra={
                        "zone": "premium",
                        "eq": mid,
                        "swing_high": swing_high,
                        "swing_low": swing_low,
                        "pos": pos,
                    },
                )
            )
            return signals

        return []

    @staticmethod
    def _bullish_rejection(o, h, l, c, prev) -> bool:
        body = abs(c - o)
        lower_wick = min(o, c) - l
        full = h - l
        if full <= 0:
            return False
        # Pin bar / hammer hoặc bullish engulf
        pin = lower_wick >= body * 1.2 and lower_wick >= full * 0.4
        engulf = c > o and c > float(prev["open"]) and c > float(prev["close"]) and o <= float(prev["close"])
        close_strong = c > o and (c - l) / full >= 0.6
        return bool(pin or engulf or close_strong)

    @staticmethod
    def _bearish_rejection(o, h, l, c, prev) -> bool:
        body = abs(c - o)
        upper_wick = h - max(o, c)
        full = h - l
        if full <= 0:
            return False
        pin = upper_wick >= body * 1.2 and upper_wick >= full * 0.4
        engulf = c < o and c < float(prev["open"]) and c < float(prev["close"]) and o >= float(prev["close"])
        close_strong = c < o and (h - c) / full >= 0.6
        return bool(pin or engulf or close_strong)

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()
