"""
Exchange wrapper hỗ trợ 3 mode: paper / testnet / live
Sử dụng ccxt cho tính nhất quán.
"""
import os
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import ccxt
from loguru import logger
from core.models import Mode, Side, Position, OrderStatus


class ExchangeWrapper:
    def __init__(self, mode: str = "paper", config: dict = None):
        self.mode = Mode(mode)
        self.config = config or {}
        self.exchange = None
        self.paper_balance = float(self.config.get("initial_capital", 10000))
        self.paper_positions: Dict[str, Position] = {}
        self.paper_orders: Dict[str, dict] = {}
        self.paper_trades: List[dict] = []
        self._init_exchange()

    def _init_exchange(self):
        if self.mode == Mode.PAPER:
            logger.info("Exchange mode: PAPER (local simulation)")
            return

        options = {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        }

        if self.mode == Mode.TESTNET:
            # Binance đã bỏ sandbox/testnet cũ cho Futures.
            # Dùng Demo Trading mới: enable_demo_trading(True)
            # API key phải tạo tại https://demo.binance.com (Demo Trading API)
            self.exchange = ccxt.binanceusdm({
                "apiKey": os.getenv("BINANCE_TESTNET_API_KEY", "") or os.getenv("BINANCE_DEMO_API_KEY", ""),
                "secret": os.getenv("BINANCE_TESTNET_API_SECRET", "") or os.getenv("BINANCE_DEMO_API_SECRET", ""),
                "options": options,
                "enableRateLimit": True,
            })
            if hasattr(self.exchange, "enable_demo_trading"):
                self.exchange.enable_demo_trading(True)
            else:
                # Fallback cho bản ccxt cũ hơn
                try:
                    self.exchange.set_sandbox_mode(True)
                except Exception as e:
                    logger.error(f"Demo/Sandbox not supported: {e}")
                    raise
            logger.info("Exchange mode: DEMO TRADING (demo.binance.com)")
        else:  # LIVE
            self.exchange = ccxt.binanceusdm({
                "apiKey": os.getenv("BINANCE_API_KEY", ""),
                "secret": os.getenv("BINANCE_API_SECRET", ""),
                "options": options,
                "enableRateLimit": True,
            })
            logger.warning("Exchange mode: LIVE – REAL MONEY")

        # Load markets
        try:
            self.exchange.load_markets()
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 500) -> List[list]:
        if self.mode == Mode.PAPER:
            # In pure paper we still need real data for signals → use public endpoint
            public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
            return public.fetch_ohlcv(symbol, timeframe, limit=limit)
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> dict:
        if self.mode == Mode.PAPER:
            public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
            return public.fetch_ticker(symbol)
        return self.exchange.fetch_ticker(symbol)

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        if self.mode == Mode.PAPER:
            public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
            return public.fetch_order_book(symbol, limit)
        return self.exchange.fetch_order_book(symbol, limit)

    def fetch_funding_rate(self, symbol: str) -> dict:
        try:
            if self.mode == Mode.PAPER:
                public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
                return public.fetch_funding_rate(symbol)
            return self.exchange.fetch_funding_rate(symbol)
        except Exception:
            return {}

    def fetch_open_interest(self, symbol: str) -> dict:
        try:
            if self.mode == Mode.PAPER:
                public = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
                return public.fetch_open_interest(symbol)
            return self.exchange.fetch_open_interest(symbol)
        except Exception:
            return {}

        def _is_tradable_market(self, market: dict, quote: str = "USDT") -> bool:
        """Chỉ nhận perpetual USDT đang active + status TRADING."""
        if not market or not isinstance(market, dict):
            return False
        if not market.get("active", True):
            return False
        if market.get("spot"):
            return False
        # Bỏ delivery / quarterly (có expiry)
        if market.get("expiry") or market.get("expiryDatetime"):
            return False
        if market.get("swap") is False and market.get("future"):
            return False

        info = market.get("info") or {}
        status = str(info.get("status") or info.get("contractStatus") or "").upper()
        if status and status not in ("TRADING", "LIVE"):
            return False

        symbol = market.get("symbol") or ""
        if f"/{quote}" not in symbol and f":{quote}" not in symbol:
            return False

        q = quote.upper()
        quote_ccxt = (market.get("quote") or "").upper()
        settle = (market.get("settle") or "").upper()
        if quote_ccxt and quote_ccxt != q and settle != q:
            return False
        return True

    def _normalize_symbol(self, sym: str, quote: str = "USDT") -> str:
        """Chuẩn hóa về dạng CCXT perpetual: BTC/USDT:USDT."""
        if not sym:
            return sym
        if f":{quote}" in sym:
            return sym
        if sym.endswith(f"/{quote}"):
            return f"{sym}:{quote}"
        return sym

    def get_top_symbols(self, n: int = 20, quote: str = "USDT") -> List[str]:
        """Top N theo volume 24h — chỉ symbol TRADING + active trên đúng mode.

        - paper: markets + volume public mainnet
        - testnet/demo: chỉ symbol có trong markets của demo exchange
        - live: markets + volume live exchange
        """
        default_pairs = [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT",
            "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
            "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
            "DOT/USDT:USDT",
        ]

        markets: Dict[str, Any] = {}
        try:
            if self.mode == Mode.PAPER:
                public = ccxt.binanceusdm({
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                })
                public.load_markets()
                markets = public.markets or {}
            elif self.exchange:
                if not getattr(self.exchange, "markets", None):
                    self.exchange.load_markets()
                markets = self.exchange.markets or {}
        except Exception as e:
            logger.warning(f"load_markets for top symbols failed: {e}")

        tradable = set()
        for sym, m in (markets or {}).items():
            try:
                if self._is_tradable_market(m, quote=quote):
                    tradable.add(self._normalize_symbol(sym, quote))
            except Exception:
                continue

        tickers = {}
        try:
            if self.mode != Mode.PAPER and self.exchange:
                tickers = self.exchange.fetch_tickers() or {}
            else:
                public = ccxt.binanceusdm({
                    "enableRateLimit": True,
                    "options": {"defaultType": "future"},
                })
                tickers = public.fetch_tickers() or {}
        except Exception as e:
            logger.warning(f"fetch_tickers failed: {e}")
            try:
                if self.exchange:
                    tickers = self.exchange.fetch_tickers() or {}
            except Exception as e2:
                logger.warning(f"exchange fetch_tickers failed: {e2}")

        ranked = []
        for sym, t in tickers.items():
            norm = self._normalize_symbol(sym, quote)
            # Bắt buộc nằm trong markets TRADING của mode hiện tại
            if tradable and norm not in tradable and sym not in tradable:
                continue
            is_usdt = (
                sym.endswith(f"/{quote}")
                or sym.endswith(f"/{quote}:{quote}")
                or f"/{quote}:" in sym
            )
            if not is_usdt:
                continue
            vol = t.get("quoteVolume") or t.get("baseVolume") or 0
            try:
                vol_f = float(vol or 0)
            except (TypeError, ValueError):
                vol_f = 0.0
            if vol_f <= 0:
                continue
            ranked.append((norm, vol_f))

        ranked.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        result = []
        for s, _ in ranked:
            if s in seen:
                continue
            seen.add(s)
            result.append(s)
            if len(result) >= n:
                break

        if len(result) < n and tradable:
            for s in sorted(tradable):
                if s not in seen:
                    result.append(s)
                    seen.add(s)
                if len(result) >= n:
                    break

        if not result:
            logger.warning("Top symbols empty after TRADING filter → default majors")
            if tradable:
                result = [s for s in default_pairs if s in tradable][:n]
            if not result:
                result = default_pairs[:n]

        logger.info(
            f"Top symbols ({self.mode.value}): {len(result)}/{n} "
            f"(tradable_markets={len(tradable)})"
        )
        return result[:n]

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------
    def get_balance(self) -> float:
        if self.mode == Mode.PAPER:
            return self.paper_balance
        bal = self.exchange.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0))

    def get_positions(self) -> List[Position]:
        if self.mode == Mode.PAPER:
            return list(self.paper_positions.values())
        # Live / Testnet
        positions = []
        raw = self.exchange.fetch_positions()
        for p in raw:
            if float(p.get("contracts", 0)) > 0:
                side = Side.LONG if p["side"] == "long" else Side.SHORT
                lev = p.get("leverage") or 1
                try:
                    lev = int(float(lev))
                except (TypeError, ValueError):
                    lev = 1
                ep = p.get("entryPrice") or 0
                upnl = p.get("unrealizedPnl") or 0
                positions.append(Position(
                    symbol=p["symbol"],
                    side=side,
                    entry_price=float(ep) if ep else 0.0,
                    size=float(p.get("contracts") or 0),
                    sl=0.0,
                    tp=0.0,
                    order_id=str(p.get("id") or ""),
                    strategy="unknown",
                    open_time=datetime.utcnow(),
                    leverage=lev,
                    unrealized_pnl=float(upnl) if upnl else 0.0,
                ))
        return positions

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------
    def create_order(
        self,
        symbol: str,
        side: str,          # "buy" / "sell"
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
        params: dict = None,
    ) -> dict:
        params = params or {}
        if self.mode == Mode.PAPER:
            return self._paper_create_order(symbol, side, amount, price, order_type, params)

        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params,
            )
            logger.info(f"[{self.mode.value}] Order created: {symbol} {side} {amount}")
            return order
        except Exception as e:
            logger.error(f"Order failed: {e}")
            raise

    def _paper_create_order(self, symbol, side, amount, price, order_type, params):
        order_id = str(uuid.uuid4())[:12]
        ot = (order_type or "market").upper()
        params = params or {}

        # Protective orders (SL/TP) → giữ status open, không fill ngay
        if ot in ("STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT"):
            stop_price = params.get("stopPrice") or price
            order = {
                "id": order_id,
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": stop_price,
                "stopPrice": stop_price,
                "type": ot,
                "status": "open",
                "timestamp": int(time.time() * 1000),
                "info": {"paper": True, "reduceOnly": params.get("reduceOnly", True)},
            }
            self.paper_orders[order_id] = order
            logger.info(f"[PAPER] {ot} placed: {symbol} {side} {amount} @ stop={stop_price}")
            return order

        # Market / limit entry → fill ngay
        ticker = self.fetch_ticker(symbol)
        fill_price = price or ticker["last"]
        slip = 0.0002
        if side == "buy":
            fill_price *= (1 + slip)
        else:
            fill_price *= (1 - slip)

        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": fill_price,
            "type": ot,
            "status": "closed",
            "timestamp": int(time.time() * 1000),
            "info": {"paper": True},
        }
        self.paper_orders[order_id] = order
        logger.info(f"[PAPER] Order filled: {symbol} {side} {amount} @ {fill_price:.6f}")
        return order

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage cho symbol. Paper mode luôn OK."""
        if self.mode == Mode.PAPER:
            return True
        try:
            self.exchange.set_leverage(int(leverage), symbol)
            logger.info(f"Leverage set: {symbol} → {leverage}x")
            return True
        except Exception as e:
            logger.warning(f"Set leverage failed {symbol}: {e}")
            return False

    def set_margin_mode(self, symbol: str, mode: str = "isolated") -> bool:
        """
        Luôn dùng Isolated cho an toàn (mỗi position độc lập, không bị kéo theo position khác).
        mode: 'isolated' | 'cross'
        """
        if self.mode == Mode.PAPER:
            return True
        mode = (mode or "isolated").lower()
        if mode not in ("isolated", "cross"):
            mode = "isolated"
        try:
            # ccxt: set_margin_mode(marginMode, symbol, params={})
            self.exchange.set_margin_mode(mode, symbol)
            logger.info(f"Margin mode set: {symbol} → {mode.upper()}")
            return True
        except Exception as e:
            err = str(e).lower()
            # Binance trả lỗi nếu đã đúng mode rồi → coi như OK
            if "no need to change" in err or "already" in err or "margin type" in err:
                logger.debug(f"Margin mode already {mode} for {symbol}")
                return True
            logger.warning(f"Set margin mode failed {symbol}: {e}")
            return False

    def prepare_symbol(self, symbol: str, leverage: int) -> bool:
        """
        Chuẩn bị symbol trước khi vào lệnh:
        1. Set Isolated
        2. Set leverage
        """
        ok_margin = self.set_margin_mode(symbol, "isolated")
        ok_lev = self.set_leverage(symbol, leverage)
        return ok_margin and ok_lev

    def cancel_all_orders(self, symbol: str = None):
        if self.mode == Mode.PAPER:
            self.paper_orders.clear()
            return
        try:
            if symbol:
                self.exchange.cancel_all_orders(symbol)
            else:
                # cancel all open
                for pos in self.get_positions():
                    self.exchange.cancel_all_orders(pos.symbol)
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")

    def close_position(self, symbol: str, side: Side):
        """Market close position"""
        positions = self.get_positions()
        for p in positions:
            if p.symbol == symbol and p.side == side:
                close_side = "sell" if side == Side.LONG else "buy"
                return self.create_order(symbol, close_side, p.size, order_type="market")
        return None

    def flatten_all(self):
        """Kill switch: đóng hết position + hủy order"""
        logger.critical("FLATTEN ALL POSITIONS")
        self.cancel_all_orders()
        for p in self.get_positions():
            self.close_position(p.symbol, p.side)
        if self.mode == Mode.PAPER:
            self.paper_positions.clear()
