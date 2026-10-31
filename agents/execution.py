"""
Execution Agent - đặt lệnh entry + TP/SL tự động (Binance Futures)
- Luôn set Isolated margin
- Tự chọn leverage hợp lý (theo config + ATR% nếu có)
- Bắt lỗi -4411 (TradFi-Perps agreement) và bỏ qua symbol đó.
"""
from typing import Optional, Set
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
    ):
        self.exchange = exchange
        self.risk_gate = risk_gate
        self.default_leverage = max(1, int(default_leverage))
        self.max_leverage = max(self.default_leverage, int(max_leverage))
        # Symbol bị lỗi agreement / không trade được → skip trong session
        self.skip_symbols: Set[str] = set()

    def _choose_leverage(self, signal: Signal) -> int:
        """
        Chọn leverage hợp lý:
        - Base = default_leverage (config)
        - Nếu stop distance % lớn (rủi ro cao) → giảm leverage
        - Không vượt max_leverage
        """
        lev = self.default_leverage
        try:
            entry = float(signal.entry)
            sl = float(signal.sl)
            if entry > 0 and sl > 0:
                stop_pct = abs(entry - sl) / entry * 100  # %
                # stop rộng (> 2.5%) → giảm leverage
                if stop_pct >= 4.0:
                    lev = max(2, self.default_leverage // 3)
                elif stop_pct >= 2.5:
                    lev = max(3, self.default_leverage // 2)
                elif stop_pct <= 0.6:
                    # stop rất hẹp, có thể tăng nhẹ nhưng vẫn ≤ max
                    lev = min(self.max_leverage, self.default_leverage + 2)
        except Exception:
            pass

        lev = max(1, min(lev, self.max_leverage))
        return lev

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

        # Leverage hợp lý theo signal
        leverage = self._choose_leverage(signal)

        size = risk_amount / stop_distance
        max_notional = equity * self.risk_gate.cfg.get("max_position_pct", 0.05) * leverage
        if size * signal.entry > max_notional:
            size = max_notional / signal.entry

        size = self._round_amount(signal.symbol, size)
        if size <= 0:
            logger.error(f"Size too small after rounding: {size}")
            return None

        side = "buy" if signal.side == Side.LONG else "sell"

        try:
            # Luôn Isolated + set leverage trước khi vào lệnh
            self.exchange.prepare_symbol(signal.symbol, leverage)

            order = self.exchange.create_order(
                symbol=signal.symbol,
                side=side,
                amount=size,
                order_type="market",
            )
            order_id = str(order.get("id", "paper"))
            fill_price = float(order.get("price") or order.get("average") or signal.entry)

            self.risk_gate.register_order(signal, order_id, size)

            logger.info(
                f"EXECUTED {signal.side.value} {signal.symbol} size={size} lev={leverage}x "
                f"entry≈{fill_price:.6f} SL={signal.sl:.6f} TP={signal.tp:.6f} "
                f"strategy={signal.strategy} | margin=ISOLATED"
            )

            sl_ok, tp_ok = self._place_protective_orders(signal, size, fill_price)

            return {
                "order_id": order_id,
                "signal": signal.to_dict(),
                "size": size,
                "entry": fill_price,
                "leverage": leverage,
                "margin_mode": "isolated",
                "sl_placed": sl_ok,
                "tp_placed": tp_ok,
                "status": "FILLED",
            }
        except Exception as e:
            err = str(e)
            # -4411: cần ký TradFi-Perps agreement → bỏ qua symbol này
            if "-4411" in err or "TradFi-Perps" in err or "agreement contract" in err.lower():
                self.skip_symbols.add(signal.symbol)
                logger.warning(
                    f"Skip {signal.symbol}: cần ký TradFi-Perps agreement trên Binance. "
                    f"Đã thêm vào skip list ({len(self.skip_symbols)} symbols)."
                )
                return None

            # Lỗi khác
            logger.error(f"Execution failed {signal.symbol}: {e}")
            self.risk_gate.record_api_call(False)
            return None

    def _round_amount(self, symbol: str, amount: float) -> float:
        try:
            if self.exchange.mode != Mode.PAPER and self.exchange.exchange:
                return float(self.exchange.exchange.amount_to_precision(symbol, amount))
        except Exception:
            pass
        if amount >= 1:
            return round(amount, 3)
        if amount >= 0.01:
            return round(amount, 4)
        return round(amount, 6)

    def _place_protective_orders(self, signal: Signal, size: float, entry_price: float) -> tuple:
        if self.exchange.mode == Mode.PAPER:
            logger.info(
                f"[PAPER] Protective orders simulated | "
                f"SL={signal.sl:.6f} TP={signal.tp:.6f} size={size}"
            )
            key = f"{signal.symbol}_{signal.side.value}"
            if key in self.exchange.paper_positions:
                self.exchange.paper_positions[key].sl = signal.sl
                self.exchange.paper_positions[key].tp = signal.tp
            return True, True

        close_side = "sell" if signal.side == Side.LONG else "buy"
        sl_ok, tp_ok = False, False

        try:
            sl_order = self.exchange.create_order(
                symbol=signal.symbol,
                side=close_side,
                amount=size,
                order_type="STOP_MARKET",
                params={
                    "stopPrice": float(signal.sl),
                    "reduceOnly": True,
                    "workingType": "MARK_PRICE",
                },
            )
            logger.info(f"SL placed: {signal.symbol} stop={signal.sl} id={sl_order.get('id')}")
            sl_ok = True
        except Exception as e:
            logger.error(f"SL order FAILED {signal.symbol}: {e}")
            try:
                sl_order = self.exchange.create_order(
                    symbol=signal.symbol,
                    side=close_side,
                    amount=size,
                    order_type="STOP_MARKET",
                    params={"stopPrice": float(signal.sl), "reduceOnly": True},
                )
                logger.info(f"SL placed (retry): {signal.symbol} id={sl_order.get('id')}")
                sl_ok = True
            except Exception as e2:
                logger.error(f"SL retry also failed: {e2}")

        try:
            tp_order = self.exchange.create_order(
                symbol=signal.symbol,
                side=close_side,
                amount=size,
                order_type="TAKE_PROFIT_MARKET",
                params={
                    "stopPrice": float(signal.tp),
                    "reduceOnly": True,
                    "workingType": "MARK_PRICE",
                },
            )
            logger.info(f"TP placed: {signal.symbol} stop={signal.tp} id={tp_order.get('id')}")
            tp_ok = True
        except Exception as e:
            logger.error(f"TP order FAILED {signal.symbol}: {e}")
            try:
                tp_order = self.exchange.create_order(
                    symbol=signal.symbol,
                    side=close_side,
                    amount=size,
                    order_type="TAKE_PROFIT_MARKET",
                    params={"stopPrice": float(signal.tp), "reduceOnly": True},
                )
                logger.info(f"TP placed (retry): {signal.symbol} id={tp_order.get('id')}")
                tp_ok = True
            except Exception as e2:
                logger.error(f"TP retry also failed: {e2}")

        if not sl_ok or not tp_ok:
            logger.warning(
                f"Protective incomplete for {signal.symbol}: SL={sl_ok} TP={tp_ok}"
            )
        return sl_ok, tp_ok
