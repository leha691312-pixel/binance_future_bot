from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class Mode(str, Enum):
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    CLOSED = "CLOSED"

@dataclass
class Signal:
    symbol: str
    side: Side
    entry: float
    sl: float
    tp: float
    confidence: float
    strategy: str
    reason: str
    timeframe: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    extra: Dict[str, Any] = field(default_factory=dict)

    def risk_reward(self) -> float:
        risk = abs(self.entry - self.sl)
        if risk == 0:
            return 0.0
        reward = abs(self.tp - self.entry)
        return reward / risk

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "reason": self.reason,
            "timeframe": self.timeframe,
            "rr": round(self.risk_reward(), 2),
            "timestamp": self.timestamp.isoformat(),
        }

@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    size: float                 # in contracts / base asset
    sl: float
    tp: float
    order_id: str
    strategy: str
    open_time: datetime
    leverage: int = 1
    unrealized_pnl: float = 0.0
    status: OrderStatus = OrderStatus.OPEN

@dataclass
class TradeResult:
    symbol: str
    side: Side
    entry: float
    exit: float
    size: float
    pnl: float
    pnl_pct: float
    strategy: str
    open_time: datetime
    close_time: datetime
    reason: str                 # TP / SL / MANUAL / KILL
    fees: float = 0.0
