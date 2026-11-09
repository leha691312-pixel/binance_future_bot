from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
from core.models import Signal


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, params: dict = None):
        self.params = params or {}

    @abstractmethod
    def generate(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Signal]:
        """
        Nhận OHLCV DataFrame (columns: open, high, low, close, volume)
        Trả về list Signal (có thể rỗng)
        """
        pass

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Missing column {c}")
        return df
