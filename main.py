"""
Main Orchestrator - Multi-agent trading bot
Modes: paper | testnet | live
Discord control: set DISCORD_BOT_TOKEN + DISCORD_ALLOWED_IDS in .env
"""
import time
import yaml
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from utils.logger import setup_logger
from utils.status_writer import write_status
from utils.order_logger import OrderLogger
from core.exchange import ExchangeWrapper
from agents.risk_gate import RiskGate
from agents.scanner import ScannerAgent
from agents.execution import ExecutionAgent
from agents.ml_agent import MLAgent
from agents.discord_control import control_state, start_discord_in_thread

load_dotenv()


def load_config():
    with open(Path(__file__).parent / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open(Path(__file__).parent / "config" / "risk.yaml") as f:
        risk = yaml.safe_load(f)
    return settings, risk


def main():
    settings, risk_cfg = load_config()
    setup_logger(settings.get("log_file", "logs/bot.log"), settings.get("log_level", "INFO"))

    mode = settings.get("mode", "paper")
    logger.info(f"=== Starting Bot in mode: {mode.upper()} ===")

    if mode == "live":
        logger.warning("LIVE MODE – Real money at risk. Make sure you understand the risks.")
        confirm = input("Type 'YES I UNDERSTAND' to continue: ")
        if confirm != "YES I UNDERSTAND":
            logger.info("Aborted.")
            return

    exchange = ExchangeWrapper(mode=mode, config=settings)
    risk_gate = RiskGate(risk_config=risk_cfg, equity=exchange.get_balance())

    order_logger = OrderLogger(path="data/orders.xlsx")

    synced = risk_gate.sync_positions_from_exchange(exchange)
    logger.info(f"Startup sync: {synced} positions loaded from exchange")

    scanner = ScannerAgent(
        exchange,
        top_n=settings.get("top_n", 20),
        timeframes=settings.get("timeframes", ["15m", "1h"]),
    )

    default_lev = int(risk_cfg.get("default_leverage", 5))
    max_lev = int(risk_cfg.get("max_leverage", 10))
    executor = ExecutionAgent(
        exchange,
        risk_gate,
        default_leverage=default_lev,
        max_leverage=max_lev,
        order_logger=order_logger,
    )
    logger.info(f"Execution leverage: default={default_lev}x | max={max_lev}x | margin=ISOLATED")

    ml_agent = MLAgent(settings.get("ml_model_path", "data/models/signal_filter.joblib"))

    # Discord control (optional)
    control_state.set_refs(risk_gate, executor, exchange, settings, risk_cfg, mode)
    start_discord_in_thread()

    logger.info("All agents ready. Starting main loop...")

    try:
        while True:
            if risk_gate.kill_switch:
                logger.critical("Kill switch active – flattening and pausing")
                try:
                    exchange.flatten_all()
                except Exception:
                    pass
                write_status(
                    mode=mode,
                    kill_switch=True,
                    kill_reason=risk_gate.kill_reason,
                    equity=risk_gate.equity,
                    daily_pnl=risk_gate.daily_pnl,
                    open_positions=len(risk_gate.open_positions),
                )
                time.sleep(60)
                continue

            # Pause từ Discord: vẫn monitor closed positions, không scan/execute
            if control_state.paused:
                closed = risk_gate.handle_closed_positions(exchange, order_logger)
                risk_gate.sync_positions_from_exchange(exchange)
                control_state.last_cycle_info = f"PAUSED | closed={closed}"
                write_status(
                    mode=mode,
                    kill_switch=False,
                    equity=risk_gate.equity,
                    daily_pnl=risk_gate.daily_pnl,
                    open_positions=len(risk_gate.open_positions),
                    note="paused_via_discord",
                )
                time.sleep(settings.get("scan_interval_seconds", 60))
                continue

            closed = risk_gate.handle_closed_positions(exchange, order_logger)
            if closed:
                logger.info(f"Handled {closed} closed position(s) this cycle")

            risk_gate.sync_positions_from_exchange(exchange)

            # Cập nhật top_n nếu Discord đã chỉnh
            if hasattr(scanner, "top_n"):
                scanner.top_n = int(settings.get("top_n", scanner.top_n))

            raw_signals = scanner.scan()
            signals = ml_agent.filter(raw_signals)

            executed = 0
            for sig in signals[:3]:
                result = executor.execute(sig)
                if result:
                    executed += 1

            pos_list = []
            try:
                for p in exchange.get_positions():
                    pos_list.append({
                        "symbol": p.symbol,
                        "side": p.side.value if hasattr(p.side, "value") else str(p.side),
                        "entry": p.entry_price,
                        "size": p.size,
                        "sl": p.sl,
                        "tp": p.tp,
                        "upnl": p.unrealized_pnl,
                    })
            except Exception:
                pass

            recent = [s.to_dict() for s in signals[:15]]

            write_status(
                mode=mode,
                kill_switch=risk_gate.kill_switch,
                kill_reason=risk_gate.kill_reason,
                equity=risk_gate.equity,
                daily_pnl=risk_gate.daily_pnl,
                open_positions=len(risk_gate.open_positions) or len(pos_list),
                consecutive_losses=risk_gate.consecutive_losses,
                raw_signals=len(raw_signals),
                filtered_signals=len(signals),
                executed=executed,
                skip_symbols=len(getattr(executor, "skip_symbols", [])),
                recent_signals=recent,
                positions=pos_list,
            )

            control_state.last_cycle_info = (
                f"raw={len(raw_signals)} filtered={len(signals)} "
                f"executed={executed} closed={closed}"
            )

            logger.info(
                f"Cycle done | raw={len(raw_signals)} | filtered={len(signals)} | "
                f"executed={executed} | closed={closed} | risk={risk_gate.status()}"
            )

            time.sleep(settings.get("scan_interval_seconds", 60))

    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C) – positions left open on exchange")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        write_status(mode=mode, last_error=str(e)[:200], kill_switch=risk_gate.kill_switch)


if __name__ == "__main__":
    main()
