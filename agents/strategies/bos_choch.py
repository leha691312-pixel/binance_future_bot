"""
BOS (Break of Structure) + CHoCH (Change of Character)
- Swing high/low detection
- BOS: phá structure theo hướng trend hiện tại (continuation)
- CHoCH: phá structure ngược hướng (đảo chiều tiềm năng)
"""
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from agents.strategies.base import BaseStrategy
from core.models import Signal, Side


class BOSCHOCHStrategy(BaseStrategy):
    name = "BOS_CHOCH"

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.swing_lookback = self.params.get("swing_lookback", 5)   # nến 2 bên để confirm swing
        self.rr = self.params.get("rr", 2.3)
        self.min_break_pct = self.params.get("min_break_pct", 0.001)
        self.use_choch = self.params.get("use_choch", True)
        self.use_bos = self.params.get("use_bos", True)

    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        df = self._ensure_columns(df)
        if len(df) < 60:
            return []

        swings = self._find_swings(df)
        if len(swings) < 4:
            return []

        signals = []
        last = df.iloc[-1]
        atr = self._atr(df, 14).iloc[-1]
        if pd.isna(atr) or atr == 0:
            return []

        # Xác định market structure gần nhất
        structure = self._analyze_structure(swings)
        if structure is None:
            return []

        trend = structure["trend"]          # "bullish" | "bearish"
        last_sh = structure["last_swing_high"]
        last_sl = structure["last_swing_low"]
        prev_sh = structure.get("prev_swing_high")
        prev_sl = structure.get("prev_swing_low")

        # ---------- CHoCH (Change of Character) ----------
        if self.use_choch:
            # Bullish CHoCH: đang bearish structure → phá last swing high
            if trend == "bearish" and last_sh is not None:
                if last["close"] > last_sh["price"] * (1 + self.min_break_pct):
                    # Xác nhận break mạnh
                    if last["close"] > last_sh["price"] + atr * 0.3:
                        entry = last["close"]
                        sl = last_sl["price"] if last_sl else entry - atr * 1.5
                        risk = entry - sl
                        if risk > 0:
                            tp = entry + risk * self.rr
                            signals.append(Signal(
                                symbol=symbol,
                                side=Side.LONG,
                                entry=entry,
                                sl=sl,
                                tp=tp,
                                confidence=0.74,
                                strategy=self.name,
                                reason=f"Bullish CHoCH | broke SH={last_sh['price']:.4f}",
                                timeframe=timeframe,
                                extra={"type": "CHoCH", "trend_before": "bearish"},
                            ))

            # Bearish CHoCH: đang bullish → phá last swing low
            if trend == "bullish" and last_sl is not None:
                if last["close"] < last_sl["price"] * (1 - self.min_break_pct):
                    if last["close"] < last_sl["price"] - atr * 0.3:
                        entry = last["close"]
                        sl = last_sh["price"] if last_sh else entry + atr * 1.5
                        risk = sl - entry
                        if risk > 0:
                            tp = entry - risk * self.rr
                            signals.append(Signal(
                                symbol=symbol,
                                side=Side.SHORT,
                                entry=entry,
                                sl=sl,
                                tp=tp,
                                confidence=0.74,
                                strategy=self.name,
                                reason=f"Bearish CHoCH | broke SL={last_sl['price']:.4f}",
                                timeframe=timeframe,
                                extra={"type": "CHoCH", "trend_before": "bullish"},
                            ))

        # ---------- BOS (Break of Structure - continuation) ----------
        if self.use_bos and not signals:  # ưu tiên CHoCH nếu có
            # Bullish BOS: trend bullish + phá higher high mới
            if trend == "bullish" and last_sh is not None and prev_sh is not None:
                if last["close"] > last_sh["price"] * (1 + self.min_break_pct):
                    entry = last["close"]
                    # SL dưới last higher low
                    sl = last_sl["price"] if last_sl else entry - atr * 1.2
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
                            reason=f"Bullish BOS | HH break {last_sh['price']:.4f}",
                            timeframe=timeframe,
                            extra={"type": "BOS", "trend": "bullish"},
                        ))

            # Bearish BOS
            if trend == "bearish" and last_sl is not None and prev_sl is not None:
                if last["close"] < last_sl["price"] * (1 - self.min_break_pct):
                    entry = last["close"]
                    sl = last_sh["price"] if last_sh else entry + atr * 1.2
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
                            reason=f"Bearish BOS | LL break {last_sl['price']:.4f}",
                            timeframe=timeframe,
                            extra={"type": "BOS", "trend": "bearish"},
                        ))

        return signals[:1]  # chỉ lấy signal mạnh nhất

    def _find_swings(self, df: pd.DataFrame) -> List[dict]:
        """Tìm swing high / swing low"""
        swings = []
        lb = self.swing_lookback
        for i in range(lb, len(df) - lb):
            window_high = df["high"].iloc[i - lb:i + lb + 1]
            window_low = df["low"].iloc[i - lb:i + lb + 1]
            if df["high"].iloc[i] >= window_high.max():
                swings.append({
                    "type": "high",
                    "price": df["high"].iloc[i],
                    "index": i,
                    "time": df.index[i] if hasattr(df.index, "__getitem__") else i,
                })
            if df["low"].iloc[i] <= window_low.min():
                swings.append({
                    "type": "low",
                    "price": df["low"].iloc[i],
                    "index": i,
                    "time": df.index[i] if hasattr(df.index, "__getitem__") else i,
                })
        # Sort theo index
        swings.sort(key=lambda x: x["index"])
        return swings

    def _analyze_structure(self, swings: List[dict]) -> Optional[dict]:
        """Xác định trend dựa trên higher high / higher low hoặc lower high / lower low"""
        highs = [s for s in swings if s["type"] == "high"]
        lows = [s for s in swings if s["type"] == "low"]

        if len(highs) < 2 or len(lows) < 2:
            return None

        last_sh = highs[-1]
        prev_sh = highs[-2]
        last_sl = lows[-1]
        prev_sl = lows[-2]

        # Bullish structure: HH + HL
        is_hh = last_sh["price"] > prev_sh["price"]
        is_hl = last_sl["price"] > prev_sl["price"]

        # Bearish structure: LH + LL
        is_lh = last_sh["price"] < prev_sh["price"]
        is_ll = last_sl["price"] < prev_sl["price"]

        if is_hh and is_hl:
            trend = "bullish"
        elif is_lh and is_ll:
            trend = "bearish"
        else:
            # Mixed → lấy hướng của swing gần nhất
            if last_sh["index"] > last_sl["index"]:
                trend = "bullish" if is_hh else "bearish"
            else:
                trend = "bearish" if is_ll else "bullish"

        return {
            "trend": trend,
            "last_swing_high": last_sh,
            "last_swing_low": last_sl,
            "prev_swing_high": prev_sh,
            "prev_swing_low": prev_sl,
        }

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
