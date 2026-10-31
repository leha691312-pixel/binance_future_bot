"""
Risk Gate - BẮT BUỘC trước mọi lệnh.
Chống lệnh trùng, position size, RR, daily loss, kill switch.
"""
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from loguru import logger
from core.models import Signal, Side, Position
import yaml
from pathlib import Path


class RiskGate:
    def __init__(self, risk_config: dict = None, equity: float = 10000):
        if risk_config is None:
            with open(Path(__file__).parent.parent / "config" / "risk.yaml") as f:
                risk_config = yaml.safe_load(f)

        self.cfg = risk_config
        self.equity = equity
        self.daily_pnl = 0.0
        self.daily_reset_date = datetime.utcnow().date()
        self.open_positions: Dict[str, Position] = {}
        self.recent_orders: Dict[str, datetime] = {}   # key = symbol_side
        self.consecutive_losses = 0
        self.kill_switch = False
        self.kill_reason = ""
        self.api_errors = 0
        self.api_calls = 0

        logger.info("RiskGate initialized")

    def update_equity(self, equity: float):
        self.equity = equity

    def _reset_daily_if_needed(self):
        today = datetime.utcnow().date()
        if today != self.daily_reset_date:
            self.daily_pnl = 0.0
            self.daily_reset_date = today
            self.consecutive_losses = 0
            logger.info("Daily PnL reset")

    def sync_positions_from_exchange(self, exchange) -> int:
        """Đồng bộ toàn bộ vị thế đang mở trên sàn vào RiskGate."""
        try:
            positions = exchange.get_positions()
            self.open_positions.clear()

            for p in positions:
                size = abs(float(p.size or 0))
                if size <= 0:
                    continue

                # Key phải giống format trong register_order
                key = f"{p.symbol}_{p.side.value}"
                self.open_positions[key] = p

            count = len(self.open_positions)
            logger.info(f"Synced {count} open positions from exchange")
            return count
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
            return len(self.open_positions)

    def check(self, signal: Signal, current_price: float = None, current_positions=None) -> Tuple[bool, str]:
        """
        Trả về (approved: bool, reason: str)
        """
        self._reset_daily_if_needed()

        if self.kill_switch:
            return False, f"KILL_SWITCH: {self.kill_reason}"

        # 1. Daily loss limit
        if self.daily_pnl <= -self.cfg["max_daily_loss_pct"] * self.equity:
            self.activate_kill_switch("DAILY_LOSS_LIMIT")
            return False, "DAILY_LOSS_LIMIT"

        # 2. Max drawdown (cần equity peak tracking - simplified)
        # (implement peak equity nếu cần)

        # 3. Max open positions
        if len(self.open_positions) >= self.cfg["max_open_positions"]:
            return False, "MAX_OPEN_POSITIONS"

        # 3b. Symbol đã có vị thế trên sàn → không mở thêm
        symbol_keys = [k for k in self.open_positions if k.startswith(signal.symbol + "_")]
        if symbol_keys:
            return False, f"EXCHANGE ALREADY HAS POSITION ({signal.symbol})"

        # 4. Duplicate protection
        key = f"{signal.symbol}_{signal.side.value}"
        last_time = self.recent_orders.get(key)
        if last_time and (datetime.utcnow() - last_time).total_seconds() < self.cfg["duplicate_window_seconds"]:
            return False, "DUPLICATE_ORDER_WINDOW"

        # 5. Risk:Reward
        rr = signal.risk_reward()
        if rr < self.cfg["min_risk_reward"]:
            return False, f"RR_TOO_LOW ({rr:.2f} < {self.cfg['min_risk_reward']})"

        # 6. Position size check (sẽ tính ở Execution, nhưng pre-check)
        risk_amount = self.equity * self.cfg["risk_per_trade_pct"]
        stop_distance = abs(signal.entry - signal.sl)
        if stop_distance <= 0:
            return False, "INVALID_SL"

        # 7. Total exposure
        current_exposure = sum(p.size * p.entry_price for p in self.open_positions.values())
        max_exposure = self.equity * self.cfg["max_total_exposure_pct"]
        estimated_notional = risk_amount / (stop_distance / signal.entry)  # rough
        if current_exposure + estimated_notional > max_exposure:
            return False, "MAX_TOTAL_EXPOSURE"

        # 8. Consecutive losses
        if self.consecutive_losses >= self.cfg["kill_on_consecutive_losses"]:
            self.activate_kill_switch("CONSECUTIVE_LOSSES")
            return False, "CONSECUTIVE_LOSSES"

        return True, "APPROVED"

    def register_order(self, signal: Signal, order_id: str, size: float):
        key = f"{signal.symbol}_{signal.side.value}"
        self.recent_orders[key] = datetime.utcnow()
        self.open_positions[key] = Position(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=signal.entry,
            size=size,
            sl=signal.sl,
            tp=signal.tp,
            order_id=order_id,
            strategy=signal.strategy,
            open_time=datetime.utcnow(),
        )
        logger.info(f"RiskGate registered: {key} size={size}")

    def close_position(self, key: str, pnl: float):
        if key in self.open_positions:
            del self.open_positions[key]
        self.daily_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        logger.info(f"Position closed {key} | PnL={pnl:.2f} | Daily={self.daily_pnl:.2f}")

    def activate_kill_switch(self, reason: str):
        self.kill_switch = True
        self.kill_reason = reason
        logger.critical(f"=== KILL SWITCH ACTIVATED: {reason} ===")

    def reset_kill_switch(self):
        self.kill_switch = False
        self.kill_reason = ""
        logger.warning("Kill switch manually reset")

    def record_api_call(self, success: bool):
        self.api_calls += 1
        if not success:
            self.api_errors += 1
        if self.api_calls > 20:
            error_rate = self.api_errors / self.api_calls
            if error_rate > self.cfg["kill_on_api_error_rate"]:
                self.activate_kill_switch(f"HIGH_API_ERROR_RATE ({error_rate:.1%})")

    def status(self) -> dict:
        return {
            "kill_switch": self.kill_switch,
            "kill_reason": self.kill_reason,
            "daily_pnl": round(self.daily_pnl, 2),
            "open_positions": len(self.open_positions),
            "consecutive_losses": self.consecutive_losses,
            "equity": self.equity,
        }
