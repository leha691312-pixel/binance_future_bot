"""
Risk Gate - bat buoc truoc moi lenh.
Chong lenh trung, position size, RR, daily loss, kill switch.
Theo doi vi the dong (TP/SL) -> ghi PnL + huy lenh con lai.
Chi dong som theo TIMEOUT (max hold theo scalp/day/swing).
"""
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
from loguru import logger
from core.models import Signal, Side, Position
import yaml
from pathlib import Path


class RiskGate:
    def __init__(self, risk_config: dict = None, equity: float = 10000):
        if risk_config is None:
            with open(Path(__file__).parent.parent / "config" / "risk.yaml", encoding="utf-8") as f:
                risk_config = yaml.safe_load(f)

        self.cfg = risk_config
        self.equity = equity
        self.daily_pnl = 0.0
        self.daily_reset_date = datetime.utcnow().date()
        self.open_positions: Dict[str, Position] = {}
        self.protective_orders: Dict[str, dict] = {}
        self.recent_orders: Dict[str, datetime] = {}
        self.consecutive_losses = 0
        self.kill_switch = False
        self.kill_reason = ""
        self.api_errors = 0
        self.api_calls = 0

        logger.info(
            f"RiskGate initialized | max_open={self.cfg.get('max_open_positions', 10)} | "
            f"invalidation=TIMEOUT only"
        )

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
        try:
            positions = exchange.get_positions()
            self.open_positions.clear()
            for p in positions:
                size = abs(float(p.size or 0))
                if size <= 0:
                    continue
                key = f"{p.symbol}_{p.side.value}"
                self.open_positions[key] = p
            count = len(self.open_positions)
            logger.info(f"Synced {count} open positions from exchange")
            return count
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
            return len(self.open_positions)

    def handle_closed_positions(self, exchange, order_logger=None) -> int:
        """Vi the bien mat (cham TP hoac SL) -> huy lenh bao ve con lai + ghi PnL."""
        try:
            live_keys = set()
            for p in exchange.get_positions():
                side_val = p.side.value if hasattr(p.side, "value") else str(p.side)
                live_keys.add(f"{p.symbol}_{side_val}")

            closed = 0
            for key, pos in list(self.open_positions.items()):
                if key in live_keys:
                    continue
                symbol = pos.symbol
                try:
                    fn = getattr(exchange, "cancel_protective_orders", None) or exchange.cancel_all_orders
                    n = fn(symbol) or 0
                    logger.info(f"Canceled remaining TP/SL for {symbol} (ops={n})")
                except Exception as e:
                    logger.warning(f"Cancel remaining failed {symbol}: {e}")

                pnl = self._estimate_realized_pnl(exchange, pos)
                self.close_position(key, pnl)
                if order_logger:
                    try:
                        order_logger.log(
                            action="CLOSE",
                            symbol=symbol,
                            side=getattr(pos.side, "value", str(pos.side)),
                            size=getattr(pos, "size", 0),
                            entry=getattr(pos, "entry_price", 0),
                            status="CLOSED",
                            note="TP or SL hit - remaining protective canceled",
                            pnl=pnl,
                        )
                    except Exception:
                        pass
                closed += 1
            return closed
        except Exception as e:
            logger.error(f"handle_closed_positions: {e}")
            return 0

    def _estimate_realized_pnl(self, exchange, pos: Position) -> float:
        try:
            upnl = float(getattr(pos, "unrealized_pnl", 0) or 0)
            if upnl != 0:
                return upnl
        except Exception:
            pass
        try:
            ticker = exchange.fetch_ticker(pos.symbol) if hasattr(exchange, "fetch_ticker") else None
            last = float((ticker or {}).get("last") or 0)
            entry = float(pos.entry_price or 0)
            size = abs(float(pos.size or 0))
            if last > 0 and entry > 0 and size > 0:
                if pos.side == Side.LONG or str(pos.side).upper() in ("LONG", "BUY"):
                    return (last - entry) * size
                return (entry - last) * size
        except Exception:
            pass
        return 0.0

    def resolve_trade_style(self, strategy: str = "", timeframe: str = "") -> str:
        mapping = self.cfg.get("strategy_trade_style") or {}
        if strategy and strategy in mapping:
            return str(mapping[strategy]).lower()
        tf = (timeframe or "").lower().strip()
        if tf in ("1m", "3m", "5m"):
            return "scalp"
        if tf in ("15m", "30m", "1h"):
            return "day"
        if tf in ("4h", "1d", "1w"):
            return "swing"
        return "day"

    def _resolve_max_hold_minutes(self, pos: Position) -> Tuple[int, str]:
        style = (getattr(pos, "trade_style", None) or "").lower()
        if not style:
            style = self.resolve_trade_style(
                getattr(pos, "strategy", "") or "",
                getattr(pos, "timeframe", "") or "",
            )
        profiles = self.cfg.get("hold_profiles") or {}
        profile = profiles.get(style) or {}
        default_hold = {"scalp": 90, "day": 720, "swing": 4320}.get(style, 720)
        max_hold = int(profile.get("max_hold_minutes", default_hold))
        return max_hold, style

    def count_positions_by_style(self) -> Dict[str, int]:
        counts = {"scalp": 0, "day": 0, "swing": 0}
        for pos in self.open_positions.values():
            style = (getattr(pos, "trade_style", None) or "day").lower()
            if style not in counts:
                style = "day"
            counts[style] += 1
        return counts

    def check_invalidation(self, exchange, pos: Position) -> Tuple[bool, str]:
        """Chi TIMEOUT theo max hold cua style."""
        max_hold, style = self._resolve_max_hold_minutes(pos)
        open_time = getattr(pos, "open_time", None)
        if open_time is None:
            return False, "STILL_VALID"
        try:
            held = (datetime.utcnow() - open_time).total_seconds() / 60.0
        except Exception:
            return False, "STILL_VALID"
        if held > max_hold:
            return True, f"TIMEOUT_{style.upper()} ({held:.0f}min > {max_hold}min)"
        return False, "STILL_VALID"

    def monitor_invalidations(self, exchange, order_logger=None) -> int:
        """TIMEOUT only: market close + huy ca TP LIMIT va SL STOP_MARKET."""
        closed = 0
        for key, pos in list(self.open_positions.items()):
            should_close, reason = self.check_invalidation(exchange, pos)
            if not should_close:
                continue

            logger.warning(
                f"TIMEOUT {pos.symbol} {pos.side.value} -> {reason} | "
                f"close position + cancel TP&SL"
            )

            try:
                exchange.close_position(pos.symbol, pos.side)
            except Exception as e:
                logger.error(f"Close on timeout failed {pos.symbol}: {e}")
                continue

            try:
                n = exchange.cancel_all_orders(pos.symbol) or 0
                logger.info(f"Timeout cancel TP/SL {pos.symbol}: {n} order ops")
            except Exception as e:
                logger.warning(f"Cancel TP/SL after timeout failed {pos.symbol}: {e}")

            pnl = self._estimate_realized_pnl(exchange, pos)
            self.close_position(key, pnl)
            if order_logger:
                try:
                    order_logger.log(
                        action="CLOSE",
                        symbol=pos.symbol,
                        side=getattr(pos.side, "value", str(pos.side)),
                        size=getattr(pos, "size", 0),
                        entry=getattr(pos, "entry_price", 0),
                        status="TIMEOUT",
                        note=f"{reason} | TP+SL canceled",
                        pnl=pnl,
                    )
                except Exception:
                    pass
            closed += 1

        if closed:
            logger.info(f"Timeout-closed {closed} position(s) this cycle")
        return closed

    def check(self, signal: Signal, current_price: float = None, current_positions=None) -> Tuple[bool, str]:
        self._reset_daily_if_needed()

        if self.kill_switch:
            return False, f"KILL_SWITCH: {self.kill_reason}"

        if self.daily_pnl <= -self.cfg["max_daily_loss_pct"] * self.equity:
            self.activate_kill_switch("DAILY_LOSS_LIMIT")
            return False, "DAILY_LOSS_LIMIT"

        max_open = int(self.cfg.get("max_open_positions", 10))
        if len(self.open_positions) >= max_open:
            return False, f"MAX_OPEN_POSITIONS ({len(self.open_positions)}>={max_open})"

        symbol_keys = [k for k in self.open_positions if k.startswith(signal.symbol + "_")]
        if symbol_keys:
            return False, f"EXCHANGE ALREADY HAS POSITION ({signal.symbol})"

        key = f"{signal.symbol}_{signal.side.value}"
        last_time = self.recent_orders.get(key)
        if last_time and (datetime.utcnow() - last_time).total_seconds() < self.cfg["duplicate_window_seconds"]:
            return False, "DUPLICATE_ORDER_WINDOW"

        rr = signal.risk_reward()
        if rr < self.cfg["min_risk_reward"]:
            return False, f"RR_TOO_LOW ({rr:.2f} < {self.cfg['min_risk_reward']})"

        risk_amount = self.equity * self.cfg["risk_per_trade_pct"]
        stop_distance = abs(signal.entry - signal.sl)
        if stop_distance <= 0:
            return False, "INVALID_SL"

        current_exposure = sum(
            float(p.size or 0) * float(p.entry_price or 0) for p in self.open_positions.values()
        )
        max_exposure = self.equity * float(self.cfg.get("max_total_exposure_pct", 0.3))
        estimated_notional = risk_amount / (stop_distance / signal.entry) if signal.entry else 0
        if current_exposure + estimated_notional > max_exposure:
            return False, "MAX_TOTAL_EXPOSURE"

        if self.consecutive_losses >= self.cfg["kill_on_consecutive_losses"]:
            self.activate_kill_switch("CONSECUTIVE_LOSSES")
            return False, "CONSECUTIVE_LOSSES"

        return True, "APPROVED"

    def register_order(
        self,
        signal: Signal,
        order_id: str,
        size: float,
        sl_order_id: str = "",
        tp_order_id: str = "",
    ):
        key = f"{signal.symbol}_{signal.side.value}"
        self.recent_orders[key] = datetime.utcnow()
        style = self.resolve_trade_style(signal.strategy or "", getattr(signal, "timeframe", "") or "")
        max_hold, _ = self._resolve_max_hold_minutes(
            type("Tmp", (), {
                "trade_style": style,
                "strategy": signal.strategy,
                "timeframe": getattr(signal, "timeframe", ""),
                "open_time": datetime.utcnow(),
            })()
        )
        tf = getattr(signal, "timeframe", "") or ""
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
            timeframe=tf,
            trade_style=style,
        )
        if sl_order_id or tp_order_id:
            self.protective_orders[key] = {
                "sl_id": sl_order_id or "",
                "tp_id": tp_order_id or "",
            }
        logger.info(
            f"RiskGate registered: {key} size={size} | style={style} | "
            f"max_hold={max_hold}min | open={len(self.open_positions)}/{self.cfg.get('max_open_positions', 10)}"
        )

    def close_position(self, key: str, pnl: float):
        if key in self.open_positions:
            del self.open_positions[key]
        self.protective_orders.pop(key, None)
        self.daily_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        logger.info(f"Position closed {key} | PnL={pnl:.4f} | Daily={self.daily_pnl:.4f}")

    def activate_kill_switch(self, reason: str):
        self.kill_switch = True
        self.kill_reason = reason
        logger.critical(f"=== KILL SWITCH ACTIVATED: {reason} ===")

    def reset_kill_switch(self):
        self.kill_switch = False
        self.kill_reason = ""
        logger.warning("Kill switch manually reset")

    def record_api_call(self, success: bool, *, count_error: bool = True):
        """Ghi nhan API call. count_error=False: khong tinh loi soft (invalid symbol)."""
        self.api_calls += 1
        if not success and count_error:
            self.api_errors += 1
        if self.api_calls > 20:
            error_rate = self.api_errors / self.api_calls
            if error_rate > self.cfg.get("kill_on_api_error_rate", 0.15):
                self.activate_kill_switch(f"HIGH_API_ERROR_RATE ({error_rate:.1%})")

    def status(self) -> dict:
        by = self.count_positions_by_style()
        return {
            "kill_switch": self.kill_switch,
            "kill_reason": self.kill_reason,
            "daily_pnl": round(self.daily_pnl, 2),
            "open_positions": len(self.open_positions),
            "max_open_positions": int(self.cfg.get("max_open_positions", 10)),
            "positions_by_style": by,
            "consecutive_losses": self.consecutive_losses,
            "equity": self.equity,
        }
