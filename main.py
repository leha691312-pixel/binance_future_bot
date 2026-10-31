"""
Main Orchestrator - Multi-agent trading bot
Modes: paper | testnet | live
"""
import time
import yaml
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from utils.logger import setup_logger
from utils.status_writer import write_status
from core.exchange import ExchangeWrapper
from agents.risk_gate import RiskGate
from agents.scanner import ScannerAgent
from agents.execution import ExecutionAgent
from agents.ml_agent import MLAgent

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
    scanner = ScannerAgent(exchange, top_n=settings.get("top_n", 20), timeframes=settings.get("timeframes", ["15m", "1h"]))
    executor = ExecutionAgent(exchange, risk_gate, leverage=5)
    ml_agent = MLAgent(settings.get("ml_model_path", "data/models/signal_filter.joblib"))

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

            # 1. Scan
            raw_signals = scanner.scan()

            # 2. ML filter
            signals = ml_agent.filter(raw_signals)

            # 3. Execute top signals
            executed = 0
            for sig in signals[:3]:
                result = executor.execute(sig)
                if result:
                    executed += 1

            # Positions snapshot
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

            logger.info(
                f"Cycle done | raw={len(raw_signals)} | filtered={len(signals)} | "
                f"executed={executed} | risk={risk_gate.status()}"
            )

            time.sleep(settings.get("scan_interval_seconds", 60))

    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C)...")
        try:
            if mode != "paper":
                exchange.flatten_all()
        except Exception as e:
            logger.warning(f"Flatten on shutdown failed (ignored): {e}")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        write_status(mode=mode, last_error=str(e)[:200], kill_switch=risk_gate.kill_switch)
        try:
            exchange.flatten_all()
        except Exception as e2:
            logger.warning(f"Flatten on error failed (ignored): {e2}")


if __name__ == "__main__":
    main()
