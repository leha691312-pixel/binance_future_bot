"""
Risk Gate - bắt buộc trước mọi lệnh.
Chống lệnh trùng, position size, RR, daily loss, kill switch.
Theo dõi vị thế đóng (TP/SL) → ghi PnL + hủy lệnh còn lại.
Chỉ đóng sớm theo TIMEOUT (max hold theo scalp/day/swing).
Không còn BOS / ADVERSE_MOVE / LOW_REMAINING_RR / quota theo style.
"""
from datetime import datetime
from typing import Dict, Tuple, Optional, Any, List
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
        self.recent_orders: Dict[str, datetime] = {}
        self.consecutive_losses = 0
        self.kill_switch = False
        self.kill_reason = ""
        self.api_errors = 0
        self.api_calls = 0
        self.protective_orders: Dict[str, Dict[str, str]] = {}

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
            current: Dict[str, Position] = {}

            for p in positions:
                size = abs(float(p.size or 0))
                if size <= 0:
                    continue
                key = f"{p.symbol}_{p.side.value}"
                old = self.open_positions.get(key)
                if old:
                    p.strategy = getattr(old, "strategy", None) or getattr(p, "strategy", None)
                    if getattr(old, "sl", None):
                        p.sl = old.sl
                    if getattr(old, "tp", None):
                        p.tp = old.tp
                    if getattr(old, "order_id", None):
                        p.order_id = old.order_id
                    if getattr(old, "open_time", None):
                        p.open_time = old.open_time
                    if getattr(old, "timeframe", None):
                        p.timeframe = old.timeframe
                    if getattr(old, "trade_style", None):
                        p.trade_style = old.trade_style
                current[key] = p

            self.open_positions = current
            count = len(self.open_positions)
            logger.info(f"Synced {count} open positions from exchange")
            return count
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
            return len(self.open_positions)

    def handle_closed_positions(self, exchange, order_logger=None) -> int:
        """Vị thế mất trên sàn → hủy SL/TP còn lại + ghi PnL."""
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
                    exchange.cancel_all_orders(symbol)
                    logger.info(f"Canceled remaining orders for {symbol}")
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
                            note="TP/SL hit – protective canceled",
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

    def _classify_trade_style(self, strategy: str = "", timeframe: str = "") -> str:
        """Chỉ dùng để chọn max_hold_minutes theo scalp/day/swing."""
        strategy_raw = (strategy or "").strip()
        strategy = strategy_raw.lower()
        timeframe = (timeframe or "").lower().strip()

        style_map = self.cfg.get("strategy_trade_style") or {}
        lower_map = {str(k).lower(): str(v).lower() for k, v in style_map.items()}
        if strategy in lower_map:
            return lower_map[strategy]
        if strategy_raw in style_map:
            return str(style_map[strategy_raw]).lower()

        if any(k in strategy for k in ("orb", "opening", "scalp", "liquidity", "sweep")):
            return "scalp"
        if any(k in strategy for k in ("oi", "funding", "swing")):
            return "swing"

        tf_minutes = self._timeframe_to_minutes(timeframe)
        if tf_minutes <= 5:
            return "scalp"
        if tf_minutes <= 60:
            return "day"
        return "swing"

    def _timeframe_to_minutes(self, tf: str) -> int:
        tf = (tf or "").lower().strip()
        if not tf:
            return 15
        try:
            if tf.endswith("m"):
                return int(tf[:-1])
            if tf.endswith("h"):
                return int(tf[:-1]) * 60
            if tf.endswith("d"):
                return int(tf[:-1]) * 1440
            if tf.endswith("w"):
                return int(tf[:-1]) * 10080
        except ValueError:
            pass
        return 15

    def _resolve_max_hold_minutes(self, pos: Position) -> Tuple[int, str]:
        style = (getattr(pos, "trade_style", None) or "").lower()
        if not style:
            style = self._classify_trade_style(
                strategy=getattr(pos, "strategy", "") or "",
                timeframe=getattr(pos, "timeframe", "") or "",
            )

        profiles = self.cfg.get("hold_profiles") or {}
        profile = profiles.get(style) or {}
        max_min = profile.get("max_hold_minutes")

        if max_min is None:
            defaults = {"scalp": 90, "day": 720, "swing": 4320}
            max_min = defaults.get(style, int(self.cfg.get("max_hold_minutes", 720)))

        return int(max_min), style

    def count_positions_by_style(self) -> Dict[str, int]:
        counts = {"scalp": 0, "day": 0, "swing": 0}
        for pos in self.open_positions.values():
            style = (getattr(pos, "trade_style", None) or "").lower()
            if not style:
                style = self._classify_trade_style(
                    strategy=getattr(pos, "strategy", "") or "",
                    timeframe=getattr(pos, "timeframe", "") or "",
                )
            if style not in counts:
                counts[style] = 0
            counts[style] += 1
        return counts

    def check_invalidation(self, exchange, pos: Position) -> Tuple[bool, str]:
        """Chỉ TIMEOUT theo max hold của style. Không BOS / adverse / RR."""
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
        """Đóng sớm chỉ khi hết max hold time."""
        closed = 0
        for key, pos in list(self.open_positions.items()):
            should_close, reason = self.check_invalidation(exchange, pos)
            if not should_close:
                continue

            logger.warning(
                f"TIMEOUT {pos.symbol} {pos.side.value} → {reason} | closing"
            )

            try:
                exchange.close_position(pos.symbol, pos.side)
            except Exception as e:
                logger.error(f"Close on timeout failed {pos.symbol}: {e}")
                continue

            try:
                exchange.cancel_all_orders(pos.symbol)
            except Exception as e:
                logger.warning(f"Cancel after timeout failed {pos.symbol}: {e}")

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
                        note=reason,
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

        # Tổng tối đa 10 vị thế (không chia quota theo style)
        max_open = int(self.cfg.get("max_open_positions", 10))
        if len(self.open_positions) >= max_open:
            return False, "MAX_OPEN_POSITIONS"

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

        current_exposure = sum(p.size * p.entry_price for p in self.open_positions.values())
        max_exposure = self.equity * self.cfg["max_total_exposure_pct"]
        estimated_notional = risk_amount / (stop_distance / signal.entry) if signal.entry else 0
        if current_exposure + estimated_notional > max_exposure:
            return False, "MAX_TOTAL_EXPOSURE"

        if self.consecutive_losses >= self.cfg["kill_on_consecutive_losses"]:
            self.activate_kill_switch("CONSECUTIVE_LOSSES")
            return False, "CONSECUTIVE_LOSSES"

        return True, "APPROVED"

    def register_order(self, signal: Signal, order_id: str, size: float, sl_order_id: str = "", tp_order_id: str = ""):
        key = f"{signal.symbol}_{signal.side.value}"
        self.recent_orders[key] = datetime.utcnow()

        tf = getattr(signal, "timeframe", "") or ""
        style = self._classify_trade_style(strategy=signal.strategy or "", timeframe=tf)
        max_hold, _ = self._resolve_max_hold_minutes(
            Position(
                symbol=signal.symbol,
                side=signal.side,
                entry_price=signal.entry,
                size=size,
                sl=signal.sl,
                tp=signal.tp,
                order_id=order_id,
                strategy=signal.strategy or "",
                open_time=datetime.utcnow(),
                timeframe=tf,
                trade_style=style,
            )
        )

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

    def record_api_call(self, success: bool):
        self.api_calls += 1
        if not success:
            self.api_errors += 1
        if self.api_calls > 20:
            error_rate = self.api_errors / self.api_calls
            if error_rate > self.cfg["kill_on_api_error_rate"]:
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
