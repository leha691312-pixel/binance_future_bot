"""
Main Orchestrator - Multi-agent trading bot
ML: R-label, time-split, collect_only, deploy gate
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
from agents.strategies import list_strategy_names

load_dotenv()


def load_config():
    with open(Path(__file__).parent / "config" / "settings.yaml") as f:
        settings = yaml.safe_load(f)
    with open(Path(__file__).parent / "config" / "risk.yaml") as f:
        risk = yaml.safe_load(f)
    return settings, risk


def _attach_ml_close_hook(risk_gate: RiskGate, ml_agent: MLAgent):
    orig = risk_gate.close_position

    def close_position(key: str, pnl: float):
        meta = {}
        pos = risk_gate.open_positions.get(key)
        if pos is not None:
            meta = {
                "strategy": getattr(pos, "strategy", "") or "",
                "timeframe": getattr(pos, "timeframe", "") or "",
                "trade_style": getattr(pos, "trade_style", "") or "",
                "symbol": getattr(pos, "symbol", ""),
                "size": float(getattr(pos, "size", 0) or 0),
                "entry": float(getattr(pos, "entry_price", 0) or 0),
                "sl": float(getattr(pos, "sl", 0) or 0),
            }
        orig(key, pnl)
        try:
            report = ml_agent.on_close(key, pnl, meta=meta)
            if report:
                if report.get("deployed"):
                    logger.info(f"ML retrain DEPLOYED: {report}")
                elif report.get("error"):
                    logger.warning(f"ML retrain skipped: {report}")
                else:
                    logger.info(f"ML retrain report: {report}")
        except Exception as e:
            logger.warning(f"ML on_close hook failed: {e}")

    risk_gate.close_position = close_position  # type: ignore
    risk_gate.ml_agent = ml_agent


def main():
    settings, risk_cfg = load_config()
    setup_logger(settings.get("log_file", "logs/bot.log"), settings.get("log_level", "INFO"))

    mode = settings.get("mode", "paper")
    logger.info(f"=== Starting Bot in mode: {mode.upper()} ===")

    if mode == "live":
        logger.warning("LIVE MODE – Real money at risk.")
        confirm = input("Type 'YES I UNDERSTAND' to continue: ")
        if confirm != "YES I UNDERSTAND":
            logger.info("Aborted.")
            return

    exchange = ExchangeWrapper(mode=mode, config=settings)
    risk_gate = RiskGate(risk_config=risk_cfg, equity=exchange.get_balance())
    order_logger = OrderLogger(path="data/orders.xlsx")

    synced = risk_gate.sync_positions_from_exchange(exchange)
    logger.info(f"Startup sync: {synced} positions")

    enabled = settings.get("enabled_strategies")
    if enabled is not None and len(enabled) == 0:
        enabled = None

    scanner = ScannerAgent(
        exchange,
        top_n=settings.get("top_n", 20),
        timeframes=settings.get("timeframes", ["15m", "1h"]),
        enabled_strategies=enabled,
        strategy_params=settings.get("strategy_params") or {},
    )
    logger.info(f"Active strategies: {[s.name for s in scanner.strategies]}")

    ml_agent = MLAgent(
        model_path=settings.get("ml_model_path", "data/models/signal_filter.joblib"),
        samples_path=settings.get("ml_samples_path", "data/models/train_samples.jsonl"),
        min_samples=int(settings.get("ml_min_samples", 80)),
        retrain_every_new=int(settings.get("ml_retrain_every_new", 25)),
        min_retrain_hours=float(settings.get("ml_retrain_hours", 12)),
        collect_only=bool(settings.get("ml_collect_only", True)),
        filter_enabled=bool(settings.get("ml_filter_enabled", True)),
        min_auc_deploy=float(settings.get("ml_min_auc_deploy", 0.55)),
        min_class_count=int(settings.get("ml_min_class_count", 20)),
        fee_rate=float(settings.get("ml_fee_rate", 0.0004)),
        min_r_multiple=float(settings.get("ml_min_r_multiple", 0.25)),
        prob_threshold=float(settings.get("ml_prob_threshold", 0.55)),
        test_ratio=float(settings.get("ml_test_ratio", 0.25)),
    )
    _attach_ml_close_hook(risk_gate, ml_agent)
    logger.info(f"ML status: {ml_agent.status()}")

    ohlcv_cache: dict = {}
    default_lev = int(risk_cfg.get("default_leverage", 5))
    max_lev = int(risk_cfg.get("max_leverage", 10))
    executor = ExecutionAgent(
        exchange,
        risk_gate,
        default_leverage=default_lev,
        max_leverage=max_lev,
        order_logger=order_logger,
        ml_agent=ml_agent,
        ohlcv_cache=ohlcv_cache,
    )

    control_state.set_refs(risk_gate, executor, exchange, settings, risk_cfg, mode)
    try:
        control_state.scanner = scanner
        control_state.ml_agent = ml_agent
    except Exception:
        pass
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

            if control_state.paused:
                closed = risk_gate.handle_closed_positions(exchange, order_logger)
                invalidated = risk_gate.monitor_invalidations(exchange, order_logger)
                risk_gate.sync_positions_from_exchange(exchange)
                control_state.last_cycle_info = f"PAUSED | closed={closed} invalidated={invalidated}"
                time.sleep(settings.get("scan_interval_seconds", 60))
                continue

            closed = risk_gate.handle_closed_positions(exchange, order_logger)
            invalidated = risk_gate.monitor_invalidations(exchange, order_logger)
            risk_gate.sync_positions_from_exchange(exchange)

            if hasattr(scanner, "top_n"):
                scanner.top_n = int(settings.get("top_n", scanner.top_n))
            new_enabled = settings.get("enabled_strategies")
            if new_enabled is not None:
                current_names = [s.name for s in scanner.strategies]
                want = new_enabled if new_enabled else list_strategy_names()
                if set(want) != set(current_names):
                    scanner.reload_strategies(
                        enabled=new_enabled if new_enabled else None,
                        strategy_params=settings.get("strategy_params") or {},
                    )

            raw_signals = scanner.scan()
            if hasattr(scanner, "last_dfs") and isinstance(scanner.last_dfs, dict):
                ohlcv_cache.clear()
                ohlcv_cache.update(scanner.last_dfs)

            signals = ml_agent.filter(raw_signals, dfs=ohlcv_cache if ohlcv_cache else None)

            executed = 0
            for sig in signals[:3]:
                if executor.execute(sig):
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

            ml_st = ml_agent.status()
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
                recent_signals=[s.to_dict() for s in signals[:15]],
                positions=pos_list,
                strategies=[s.name for s in scanner.strategies],
                ml=ml_st,
            )

            logger.info(
                f"Cycle | raw={len(raw_signals)} filtered={len(signals)} "
                f"exec={executed} closed={closed} inv={invalidated} ml={ml_st}"
            )
            time.sleep(settings.get("scan_interval_seconds", 60))

    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C) – positions left open")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        write_status(mode=mode, last_error=str(e)[:200], kill_switch=risk_gate.kill_switch)


if __name__ == "__main__":
    main()
