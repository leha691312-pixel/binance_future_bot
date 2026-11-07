"""
Execution Agent - entry market + TP LIMIT + SL STOP_MARKET
- Isolated margin + leverage
- ML on_open with risk_amount / notional for R-multiple labels
- TP: LIMIT reduceOnly
- SL: STOP_MARKET only (LIMIT SL fills immediately = bug)
- Size clamped to exchange min/max quantity (avoid -4005)
"""
from typing import Optional, Set, Tuple
from loguru import logger
from core.exchange import ExchangeWrapper
from core.models import Signal, Side, Mode
from agents.risk_gate import RiskGate


class ExecutionAgent:
    def __init__(
        self,
        exchange: ExchangeWrapper,
        risk_gate: RiskGate,
        default_leverage: int = 5,
        max_leverage: int = 10,
        order_logger=None,
        ml_agent=None,
        ohlcv_cache: dict = None,
    ):
        self.exchange = exchange
        self.risk_gate = risk_gate
        self.default_leverage = max(1, int(default_leverage))
        self.max_leverage = max(self.default_leverage, int(max_leverage))
        self.order_logger = order_logger
        self.ml_agent = ml_agent
        self.ohlcv_cache = ohlcv_cache if ohlcv_cache is not None else {}
        self.skip_symbols: Set[str] = set()

    def _choose_leverage(self, signal: Signal) -> int:
        lev = self.default_leverage
        try:
            entry = float(signal.entry)
            sl = float(signal.sl)
            if entry > 0 and sl > 0:
                stop_pct = abs(entry - sl) / entry * 100
                if stop_pct >= 4.0:
                    lev = max(2, self.default_leverage // 3)
                elif stop_pct >= 2.5:
                    lev = max(3, self.default_leverage // 2)
                elif stop_pct <= 0.6:
                    lev = min(self.max_leverage, self.default_leverage + 2)
        except Exception:
            pass
        return max(1, min(lev, self.max_leverage))

    def execute(self, signal: Signal) -> Optional[dict]:
        if signal.symbol in self.skip_symbols:
            logger.debug(f"Skip {signal.symbol} (in skip list)")
            return None

        approved, reason = self.risk_gate.check(signal)
        if not approved:
            logger.warning(f"RiskGate REJECTED {signal.symbol} {signal.side.value}: {reason}")
            return None

        equity = self.exchange.get_balance()
        self.risk_gate.update_equity(equity)
        risk_pct = self.risk_gate.cfg.get("risk_per_trade_pct", 0.01)
        risk_amount = equity * risk_pct

        stop_distance = abs(signal.entry - signal.sl)
        if stop_distance <= 0:
            logger.error("Zero stop distance")
            return None

        leverage = self._choose_leverage(signal)

        size = risk_amount / stop_distance
        max_notional = equity * self.risk_gate.cfg.get("max_position_pct", 0.05) * leverage
        if signal.entry > 0 and size * signal.entry > max_notional:
            size = max_notional / signal.entry

        size = self._clamp_amount(signal.symbol, size, price=float(signal.entry or 0))
        if size is None or size <= 0:
            logger.error(f"Size invalid after market limits: {signal.symbol}")
            return None

        side = "buy" if signal.side == Side.LONG else "sell"

        try:
            self.exchange.prepare_symbol(signal.symbol, leverage)

            order = self.exchange.create_order(
                symbol=signal.symbol,
                side=side,
                amount=size,
                order_type="market",
            )
            order_id = str(order.get("id", "paper"))
            fill_price = float(order.get("price") or order.get("average") or signal.entry)

            sl_ok, tp_ok, sl_id, tp_id = self._place_protective_orders(signal, size, fill_price)

            self.risk_gate.register_order(
                signal, order_id, size,
                sl_order_id=sl_id,
                tp_order_id=tp_id,
            )

            if self.ml_agent is not None:
                try:
                    df = (
                        self.ohlcv_cache.get(signal.symbol)
                        or self.ohlcv_cache.get(f"{signal.symbol}_{getattr(signal, 'timeframe', '')}")
                    )
                    actual_risk = abs(fill_price - float(signal.sl)) * size
                    notional = size * fill_price
                    self.ml_agent.on_open(
                        signal,
                        df,
                        key=f"{signal.symbol}_{signal.side.value}",
                        trade_meta={
                            "risk_amount": float(actual_risk or risk_amount),
                            "notional": float(notional),
                            "sl": float(signal.sl),
                            "tp": float(signal.tp),
                            "fill_price": float(fill_price),
                            "size": float(size),
                            "entry": float(fill_price),
                        },
                    )
                except Exception as e:
                    logger.warning(f"ML on_open skip: {e}")

            logger.info(
                f"EXECUTED {signal.side.value} {signal.symbol} size={size:.6f} lev={leverage}x "
                f"entry~={fill_price:.6f} SL={signal.sl:.6f} TP={signal.tp:.6f} "
                f"strategy={signal.strategy} | margin=ISOLATED | TP=LIMIT SL=STOP_MARKET"
            )

            if self.order_logger:
                try:
                    self.order_logger.log(
                        action="OPEN",
                        symbol=signal.symbol,
                        side=signal.side.value,
                        size=size,
                        entry=fill_price,
                        sl=signal.sl,
                        tp=signal.tp,
                        leverage=leverage,
                        strategy=signal.strategy,
                        order_id=order_id,
                        status="FILLED",
                        note=f"lev={leverage}x SL_ok={sl_ok} TP_ok={tp_ok} TP=LIMIT SL=STOP_MARKET",
                    )
                except Exception as e:
                    logger.debug(f"order_logger: {e}")

            self.risk_gate.record_api_call(True)
            return {
                "order_id": order_id,
                "symbol": signal.symbol,
                "side": signal.side.value,
                "size": size,
                "entry": fill_price,
                "leverage": leverage,
                "sl_placed": sl_ok,
                "tp_placed": tp_ok,
                "sl_order_id": sl_id,
                "tp_order_id": tp_id,
            }
        except Exception as e:
            err = str(e).lower()
            logger.error(f"Execution failed {signal.symbol}: {e}")
            soft = any(
                x in err
                for x in (
                    "invalid symbol",
                    "does not have market",
                    "invalid symbol status",
                    "symbol not found",
                    "market is closed",
                    "tradfi",
                    "agreement contract",
                    "sign tradfi",
                    "-1122",
                    "-1121",
                    "-4411",
                    "-2019",
                    "-4164",
                    "-4005",
                    "greater than max quantity",
                    "max quantity",
                )
            )
            if soft:
                self.skip_symbols.add(signal.symbol)
                self.risk_gate.record_api_call(False, count_error=False)
            else:
                self.risk_gate.record_api_call(False, count_error=True)
            return None

    def _clamp_amount(self, symbol: str, amount: float, price: float = 0.0) -> float:
        """Round + clamp to exchange min/max quantity & max notional (avoid -4005)."""
        try:
            markets = getattr(self.exchange.exchange, "markets", None) or {}
            m = markets.get(symbol) or {}
            if not m and hasattr(self.exchange, "exchange") and self.exchange.exchange:
                try:
                    m = self.exchange.exchange.market(symbol) or {}
                except Exception:
                    m = {}

            limits = (m.get("limits") or {})
            amt_lim = limits.get("amount") or {}
            cost_lim = limits.get("cost") or {}
            min_amt = float(amt_lim.get("min") or 0) or 0.0
            max_amt = amt_lim.get("max")
            max_cost = cost_lim.get("max")

            prec = (m.get("precision") or {}).get("amount")
            step = None
            if prec is not None:
                if isinstance(prec, int):
                    step = 10 ** (-prec) if prec >= 0 else None
                else:
                    try:
                        step = float(prec)
                    except (TypeError, ValueError):
                        step = None

            amount = float(amount or 0)
            if amount <= 0:
                return 0.0

            if max_amt is not None:
                try:
                    max_amt_f = float(max_amt)
                    if max_amt_f > 0 and amount > max_amt_f:
                        logger.warning(
                            f"Size {amount} > max qty {max_amt_f} for {symbol} -> clamp"
                        )
                        amount = max_amt_f
                except (TypeError, ValueError):
                    pass

            if price and max_cost is not None:
                try:
                    max_cost_f = float(max_cost)
                    if max_cost_f > 0 and amount * price > max_cost_f:
                        amount = max_cost_f / price
                        logger.warning(
                            f"Size notional > max cost {max_cost_f} for {symbol} -> clamp"
                        )
                except (TypeError, ValueError):
                    pass

            if step and step > 0:
                amount = (amount // step) * step
            elif isinstance(prec, int) and prec >= 0:
                amount = float(f"{amount:.{prec}f}")
            else:
                amount = round(amount, 6)

            if min_amt > 0 and amount < min_amt:
                logger.error(
                    f"Size {amount} < min qty {min_amt} for {symbol} - skip"
                )
                return 0.0
            return amount
        except Exception as e:
            logger.debug(f"_clamp_amount {symbol}: {e}")
            return round(float(amount or 0), 6)

    def _normalize_sl_tp(self, signal, fill_price: float):
        """Ensure SL/TP on correct side vs fill; stop not too tight."""
        entry = float(fill_price or signal.entry or 0)
        sl = float(signal.sl)
        tp = float(signal.tp)
        side = signal.side.value if hasattr(signal.side, "value") else str(signal.side)

        if entry <= 0:
            return sl, tp, True

        if side == "LONG":
            if sl >= entry:
                logger.error(f"Invalid SL for LONG: SL={sl} >= entry={entry}")
                return sl, tp, False
            if tp <= entry:
                logger.warning(f"TP for LONG below entry - check signal TP={tp} entry={entry}")
        else:
            if sl <= entry:
                logger.error(f"Invalid SL for SHORT: SL={sl} <= entry={entry}")
                return sl, tp, False
            if tp >= entry:
                logger.warning(f"TP for SHORT above entry - check signal TP={tp} entry={entry}")

        stop_pct = abs(entry - sl) / entry * 100
        if stop_pct < 0.15:
            logger.error(
                f"SL too tight ({stop_pct:.3f}% < 0.15%) {signal.symbol} - skip protective"
            )
            return sl, tp, False
        return sl, tp, True

    def _place_protective_orders(
        self, signal, size: float, fill_price: float = None
    ):
        """TP = LIMIT; SL = STOP_MARKET only."""
        if self.exchange.mode.value == "paper":
            logger.info(
                f"[PAPER] Protective | SL={signal.sl:.6f} TP={signal.tp:.6f} "
                f"(TP=LIMIT SL=STOP_MARKET)"
            )
            key = f"{signal.symbol}_{signal.side.value}"
            if key in self.exchange.paper_positions:
                self.exchange.paper_positions[key].sl = signal.sl
                self.exchange.paper_positions[key].tp = signal.tp
            return True, True, "paper-sl", "paper-tp"

        close_side = "sell" if signal.side.value == "LONG" else "buy"
        fill = float(fill_price or signal.entry)
        sl_px, tp_px, ok = self._normalize_sl_tp(signal, fill)
        if not ok:
            logger.error(f"Protective skipped - invalid SL/TP {signal.symbol}")
            return False, False, "", ""

        sl_ok, tp_ok = False, False
        sl_id, tp_id = "", ""

        try:
            tp_order = self.exchange.create_order(
                symbol=signal.symbol,
                side=close_side,
                amount=size,
                price=tp_px,
                order_type="limit",
                params={"reduceOnly": True, "timeInForce": "GTC"},
            )
            tp_id = str(tp_order.get("id", ""))
            tp_ok = True
            logger.info(f"TP LIMIT placed {signal.symbol} @ {tp_px}")
        except Exception as e:
            logger.error(f"TP LIMIT FAILED {signal.symbol}: {e}")
            try:
                tp_order = self.exchange.create_order(
                    symbol=signal.symbol,
                    side=close_side,
                    amount=size,
                    price=tp_px,
                    order_type="limit",
                    params={"reduceOnly": True},
                )
                tp_id = str(tp_order.get("id", ""))
                tp_ok = True
                logger.info(f"TP LIMIT retry OK {signal.symbol} @ {tp_px}")
            except Exception as e2:
                logger.error(f"TP LIMIT retry failed: {e2}")

        try:
            sl_order = self.exchange.create_order(
                symbol=signal.symbol,
                side=close_side,
                amount=size,
                order_type="STOP_MARKET",
                params={
                    "stopPrice": sl_px,
                    "reduceOnly": True,
                    "workingType": "CONTRACT_PRICE",
                },
            )
            sl_id = str(sl_order.get("id", ""))
            sl_ok = True
            logger.info(f"SL STOP_MARKET placed {signal.symbol} @ {sl_px}")
        except Exception as e:
            logger.warning(f"SL STOP_MARKET failed {signal.symbol}: {e} -> retry")
            try:
                sl_order = self.exchange.create_order(
                    symbol=signal.symbol,
                    side=close_side,
                    amount=size,
                    order_type="STOP_MARKET",
                    params={"stopPrice": sl_px, "reduceOnly": True},
                )
                sl_id = str(sl_order.get("id", ""))
                sl_ok = True
                logger.info(f"SL STOP_MARKET retry OK {signal.symbol} @ {sl_px}")
            except Exception as e2:
                logger.error(f"SL STOP_MARKET final failed: {e2}")

        return sl_ok, tp_ok, sl_id, tp_id
