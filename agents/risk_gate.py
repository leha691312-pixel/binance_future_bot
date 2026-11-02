"""
Risk Gate - BẮT BUỘC trước mọi lệnh.
Chống lệnh trùng, position size, RR, daily loss, kill switch.
Theo dõi vị thế đóng (TP/SL) → ghi PnL + hủy lệnh còn lại.
Phát hiện invalidation / đảo chiều → đóng sớm.
max_hold_minutes + max positions theo scalp / day / swing.
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
                    p.strategy = old.strategy or p.strategy
                    if old.sl:
                        p.sl = old.sl
                    if old.tp:
                        p.tp = old.tp
                    if old.order_id:
                        p.order_id = old.order_id
                    if old.open_time:
                        p.open_time = old.open_time
                    if getattr(old, "timeframe", None):
                        p.timeframe = old.timeframe
                    if getattr(old, "trade_style", None):
                        p.trade_style = old.trade_style
                # Nếu chưa có style (sync từ sàn), gán theo strategy/timeframe
                if not getattr(p, "trade_style", None):
                    p.trade_style = self._classify_trade_style(
                        strategy=getattr(p, "strategy", "") or "",
                        timeframe=getattr(p, "timeframe", "") or "",
                    )
                current[key] = p

            self.open_positions = current
            count = len(self.open_positions)
            by = self.count_positions_by_style()
            logger.info(f"Synced {count} positions | by style={by}")
            return count
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
            return len(self.open_positions)

    def handle_closed_positions(self, exchange, order_logger=None) -> int:
        try:
            live = exchange.get_positions()
            live_keys = {f"{p.symbol}_{p.side.value}" for p in live if abs(float(p.size or 0)) > 0}

            closed_count = 0
            for key, pos in list(self.open_positions.items()):
                if key in live_keys:
                    continue

                closed_count += 1
                symbol = pos.symbol
                side = pos.side.value if hasattr(pos.side, "value") else str(pos.side)

                try:
                    exchange.cancel_all_orders(symbol)
                    logger.info(f"Canceled remaining protective orders for {symbol}")
                except Exception as e:
                    logger.warning(f"Cancel orders failed {symbol}: {e}")

                pnl = self._estimate_realized_pnl(exchange, pos)
                self.close_position(key, pnl)
                self.protective_orders.pop(key, None)

                if order_logger:
                    try:
                        order_logger.log(
                            action="CLOSE",
                            symbol=symbol,
                            side=side,
                            size=pos.size,
                            entry=pos.entry_price,
                            sl=pos.sl,
                            tp=pos.tp,
                            order_id=pos.order_id or "",
                            strategy=pos.strategy or "",
                            status="CLOSED",
                            mode=getattr(exchange.mode, "value", str(exchange.mode)),
                            note=f"TP/SL hit | PnL≈{pnl:.4f}",
                        )
                    except Exception as e:
                        logger.warning(f"OrderLogger CLOSE failed: {e}")

                logger.info(
                    f"CLOSED {side} {symbol} size={pos.size} entry={pos.entry_price} | "
                    f"PnL≈{pnl:.4f} | remaining orders canceled"
                )

            return closed_count
        except Exception as e:
            logger.error(f"handle_closed_positions failed: {e}")
            return 0

    def _estimate_realized_pnl(self, exchange, pos: Position) -> float:
        try:
            if getattr(exchange.mode, "value", "") == "paper":
                return 0.0
            if not getattr(exchange, "exchange", None):
                return 0.0

            trades = exchange.exchange.fetch_my_trades(pos.symbol, limit=20)
            if not trades:
                return 0.0

            close_side = "sell" if pos.side == Side.LONG else "buy"
            for t in reversed(trades):
                if str(t.get("side", "")).lower() != close_side:
                    continue
                info = t.get("info") or {}
                rpnl = info.get("realizedPnl") or info.get("realized_pnl") or t.get("realizedPnl")
                if rpnl is not None:
                    return float(rpnl)
                price = float(t.get("price") or 0)
                amount = float(t.get("amount") or 0)
                if price > 0 and amount > 0 and pos.entry_price > 0:
                    if pos.side == Side.LONG:
                        return (price - pos.entry_price) * amount
                    else:
                        return (pos.entry_price - price) * amount
            return 0.0
        except Exception as e:
            logger.debug(f"Estimate PnL failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # TRADE STYLE
    # ------------------------------------------------------------------
    def _classify_trade_style(self, strategy: str = "", timeframe: str = "") -> str:
        strategy_raw = (strategy or "").strip()
        strategy = strategy_raw.lower()
        timeframe = (timeframe or "").lower().strip()

        style_map = self.cfg.get("strategy_trade_style") or {}
        # case-insensitive lookup
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
            defaults = {"scalp": 45, "day": 360, "swing": 2880}
            max_min = defaults.get(style, int(self.cfg.get("max_hold_minutes", 360)))

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
                style = "day"
            counts[style] += 1
        return counts

    def _max_for_style(self, style: str) -> int:
        per = self.cfg.get("max_positions_per_style") or {}
        if style in per:
            return int(per[style])
        # fallback 5 theo yêu cầu
        return int(per.get(style, 5))

    # ------------------------------------------------------------------
    # INVALIDATION
    # ------------------------------------------------------------------
    def check_invalidation(self, exchange, pos: Position) -> Tuple[bool, str]:
        try:
            symbol = pos.symbol
            side = pos.side
            entry = float(pos.entry_price or 0)
            sl = float(pos.sl or 0)
            tp = float(pos.tp or 0)

            if entry <= 0:
                return False, "NO_ENTRY"

            lookback = int(self.cfg.get("invalidation_lookback_candles", 12))
            min_remaining_rr = float(self.cfg.get("min_remaining_rr", 0.4))
            adverse_pct = float(self.cfg.get("max_adverse_pct", 1.5))
            bos_buffer = float(self.cfg.get("bos_buffer_pct", 0.15)) / 100

            max_hold_min, style = self._resolve_max_hold_minutes(pos)

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe="15m", limit=max(lookback + 5, 30))
            if not ohlcv or len(ohlcv) < 5:
                return False, "NO_DATA"

            closes = [c[4] for c in ohlcv]
            highs = [c[2] for c in ohlcv]
            lows = [c[3] for c in ohlcv]
            current_price = closes[-1]

            recent_high = max(highs[-lookback:-1]) if len(highs) > lookback else max(highs[:-1])
            recent_low = min(lows[-lookback:-1]) if len(lows) > lookback else min(lows[:-1])

            if side == Side.LONG:
                if current_price < recent_low * (1 - bos_buffer):
                    return True, f"BOS_REVERSE_LONG (price={current_price:.6f} < low={recent_low:.6f})"
            else:
                if current_price > recent_high * (1 + bos_buffer):
                    return True, f"BOS_REVERSE_SHORT (price={current_price:.6f} > high={recent_high:.6f})"

            risk = abs(entry - sl) if sl > 0 else 0
            if risk > 0 and tp > 0:
                if side == Side.LONG:
                    remaining = tp - current_price
                else:
                    remaining = current_price - tp
                remaining_rr = remaining / risk if risk else 0
                if remaining_rr < min_remaining_rr:
                    return True, f"LOW_REMAINING_RR ({remaining_rr:.2f} < {min_remaining_rr})"

            open_time = pos.open_time or datetime.utcnow()
            hold_min = (datetime.utcnow() - open_time).total_seconds() / 60
            if hold_min > max_hold_min:
                return True, f"TIMEOUT_{style.upper()} ({hold_min:.0f}min > {max_hold_min}min)"

            if side == Side.LONG:
                move_pct = (entry - current_price) / entry * 100
            else:
                move_pct = (current_price - entry) / entry * 100
            if move_pct > adverse_pct:
                return True, f"ADVERSE_MOVE ({move_pct:.2f}% > {adverse_pct}%)"

            return False, f"STILL_VALID[{style}]"
        except Exception as e:
            logger.debug(f"check_invalidation error {pos.symbol}: {e}")
            return False, f"ERROR:{e}"

    def monitor_invalidations(self, exchange, order_logger=None) -> int:
        closed = 0
        for key, pos in list(self.open_positions.items()):
            should_close, reason = self.check_invalidation(exchange, pos)
            if not should_close:
                continue

            logger.warning(
                f"INVALIDATION {pos.symbol} {pos.side.value} → {reason} | closing early"
            )

            try:
                exchange.close_position(pos.symbol, pos.side)
            except Exception as e:
                logger.error(f"Close on invalidation failed {pos.symbol}: {e}")
                continue

            try:
                exchange.cancel_all_orders(pos.symbol)
            except Exception as e:
                logger.warning(f"Cancel after invalidation failed: {e}")

            pnl = self._estimate_realized_pnl(exchange, pos)
            if pnl == 0.0:
                try:
                    ticker = exchange.fetch_ticker(pos.symbol)
                    price = float(ticker.get("last") or pos.entry_price)
                    if pos.side == Side.LONG:
                        pnl = (price - pos.entry_price) * pos.size
                    else:
                        pnl = (pos.entry_price - price) * pos.size
                except Exception:
                    pass

            self.close_position(key, pnl)
            closed += 1

            if order_logger:
                try:
                    order_logger.log(
                        action="CLOSE",
                        symbol=pos.symbol,
                        side=pos.side.value,
                        size=pos.size,
                        entry=pos.entry_price,
                        sl=pos.sl,
                        tp=pos.tp,
                        order_id=pos.order_id or "",
                        strategy=pos.strategy or "",
                        status="INVALIDATED",
                        mode=getattr(exchange.mode, "value", ""),
                        note=f"Invalidation: {reason} | PnL≈{pnl:.4f}",
                    )
                except Exception as e:
                    logger.warning(f"OrderLogger INVALIDATE failed: {e}")

            logger.info(
                f"INVALIDATED & CLOSED {pos.side.value} {pos.symbol} | "
                f"reason={reason} | PnL≈{pnl:.4f}"
            )

        return closed

    def check(self, signal: Signal, current_price: float = None, current_positions=None) -> Tuple[bool, str]:
        self._reset_daily_if_needed()

        if self.kill_switch:
            return False, f"KILL_SWITCH: {self.kill_reason}"

        if self.daily_pnl <= -self.cfg["max_daily_loss_pct"] * self.equity:
            self.activate_kill_switch("DAILY_LOSS_LIMIT")
            return False, "DAILY_LOSS_LIMIT"

        # Tổng vị thế toàn bot
        if len(self.open_positions) >= self.cfg.get("max_open_positions", 15):
            return False, "MAX_OPEN_POSITIONS"

        # Limit theo style (scalp/day/swing) — mỗi loại tối đa 5
        style = self._classify_trade_style(
            strategy=signal.strategy or "",
            timeframe=getattr(signal, "timeframe", "") or "",
        )
        by_style = self.count_positions_by_style()
        max_style = self._max_for_style(style)
        if by_style.get(style, 0) >= max_style:
            return False, (
                f"MAX_{style.upper()}_POSITIONS "
                f"({by_style.get(style, 0)}>={max_style}) | "
                f"scalp={by_style.get('scalp', 0)} day={by_style.get('day', 0)} swing={by_style.get('swing', 0)}"
            )

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
        by = self.count_positions_by_style()
        logger.info(
            f"RiskGate registered: {key} size={size} | style={style} | "
            f"max_hold={max_hold}min | by_style={by}"
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
            "positions_by_style": by,
            "consecutive_losses": self.consecutive_losses,
            "equity": self.equity,
        }
