"""
Discord Control Interface for Binance Futures Bot
Cho phép điều khiển bot qua Discord: status, chỉnh thông số risk, kill switch, pause.

Cấu hình trong .env:
  DISCORD_BOT_TOKEN=...
  DISCORD_ALLOWED_IDS=123456789,987654321   # user id được phép (cách nhau bằng dấu phẩy)
  DISCORD_PREFIX=!                          # tùy chọn, mặc định !
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional, Set
from pathlib import Path

import yaml
from loguru import logger

try:
    import discord
    from discord.ext import commands
    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False
    discord = None
    commands = None


# Các key risk được phép chỉnh qua Discord (an toàn)
ALLOWED_RISK_KEYS = {
    "default_leverage": (int, 1, 20),
    "max_leverage": (int, 1, 50),
    "risk_per_trade_pct": (float, 0.001, 0.05),
    "max_position_pct": (float, 0.01, 0.2),
    "max_total_exposure_pct": (float, 0.05, 1.0),
    "max_open_positions": (int, 1, 20),
    "min_risk_reward": (float, 1.0, 5.0),
    "max_daily_loss_pct": (float, 0.005, 0.2),
    "duplicate_window_seconds": (int, 30, 3600),
    "kill_on_consecutive_losses": (int, 2, 20),
    "cooldown_seconds": (int, 0, 600),
}

ALLOWED_SETTINGS_KEYS = {
    "top_n": (int, 5, 100),
    "scan_interval_seconds": (int, 30, 600),
}


class BotControlState:
    """Shared state giữa main loop và Discord."""

    def __init__(self):
        self.risk_gate = None
        self.executor = None
        self.exchange = None
        self.settings: Dict[str, Any] = {}
        self.risk_cfg: Dict[str, Any] = {}
        self.mode: str = "paper"
        self.paused: bool = False          # True = không scan/execute
        self.running: bool = True
        self.last_cycle_info: str = ""

    def set_refs(self, risk_gate, executor, exchange, settings, risk_cfg, mode: str):
        self.risk_gate = risk_gate
        self.executor = executor
        self.exchange = exchange
        self.settings = settings or {}
        self.risk_cfg = risk_cfg or {}
        self.mode = mode


control_state = BotControlState()


def _parse_allowed_ids() -> Set[int]:
    raw = os.getenv("DISCORD_ALLOWED_IDS", "").strip()
    ids: Set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if part.isdigit():
            ids.add(int(part))
    return ids


def _is_authorized(ctx) -> bool:
    allowed = _parse_allowed_ids()
    if not allowed:
        # Không cấu hình → chỉ cho owner của bot (nếu biết) hoặc chặn hết
        return False
    return ctx.author.id in allowed


def _save_risk_yaml(risk_cfg: dict):
    path = Path(__file__).parent.parent / "config" / "risk.yaml"
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(risk_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info("risk.yaml saved via Discord")
    except Exception as e:
        logger.warning(f"Failed to save risk.yaml: {e}")


def _save_settings_yaml(settings: dict):
    path = Path(__file__).parent.parent / "config" / "settings.yaml"
    try:
        # Giữ comment khó → chỉ overwrite keys đơn giản bằng dump
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(settings, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info("settings.yaml saved via Discord")
    except Exception as e:
        logger.warning(f"Failed to save settings.yaml: {e}")


def create_bot() -> Optional[Any]:
    if not HAS_DISCORD:
        logger.warning("discord.py not installed – Discord control disabled")
        return None

    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        logger.info("DISCORD_BOT_TOKEN not set – Discord control disabled")
        return None

    prefix = os.getenv("DISCORD_PREFIX", "!").strip() or "!"
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix=prefix, intents=intents, help_command=None)

    @bot.event
    async def on_ready():
        logger.info(f"Discord control online as {bot.user} (prefix={prefix})")

    @bot.command(name="help")
    async def cmd_help(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Bạn không có quyền điều khiển bot.")
            return
        text = (
            "**Binance Futures Bot – Discord Control**\n"
            f"Prefix: `{prefix}`\n\n"
            f"`{prefix}status` – trạng thái bot, equity, PnL, positions\n"
            f"`{prefix}positions` – danh sách vị thế đang mở\n"
            f"`{prefix}params` – xem thông số risk hiện tại\n"
            f"`{prefix}set <key> <value>` – chỉnh thông số (vd: set default_leverage 3)\n"
            f"`{prefix}pause` – tạm dừng scan/execute\n"
            f"`{prefix}resume` – tiếp tục\n"
            f"`{prefix}kill` – bật kill switch (dừng + flatten)\n"
            f"`{prefix}unkill` – tắt kill switch\n"
            f"`{prefix}skip clear` – xóa danh sách symbol bị skip\n"
            f"`{prefix}help` – lệnh này\n"
        )
        await ctx.send(text)

    @bot.command(name="status")
    async def cmd_status(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        st = control_state
        rg = st.risk_gate
        if not rg:
            await ctx.send("Bot chưa sẵn sàng.")
            return
        equity = rg.equity
        try:
            if st.exchange:
                equity = st.exchange.get_balance()
                rg.update_equity(equity)
        except Exception:
            pass
        pos_n = len(rg.open_positions)
        lines = [
            f"**Mode:** `{st.mode}`",
            f"**Paused:** `{st.paused}`",
            f"**Kill switch:** `{rg.kill_switch}`" + (f" ({rg.kill_reason})" if rg.kill_switch else ""),
            f"**Equity:** `{equity:.2f}` USDT",
            f"**Daily PnL:** `{rg.daily_pnl:.4f}`",
            f"**Open positions:** `{pos_n}`",
            f"**Consecutive losses:** `{rg.consecutive_losses}`",
            f"**Leverage:** default=`{getattr(st.executor, 'default_leverage', '?')}` max=`{getattr(st.executor, 'max_leverage', '?')}`",
            f"**Scan interval:** `{st.settings.get('scan_interval_seconds', '?')}s` | top_n=`{st.settings.get('top_n', '?')}`",
        ]
        if st.last_cycle_info:
            lines.append(f"**Last cycle:** {st.last_cycle_info}")
        await ctx.send("\n".join(lines))

    @bot.command(name="positions")
    async def cmd_positions(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        st = control_state
        if not st.exchange:
            await ctx.send("Exchange chưa sẵn sàng.")
            return
        try:
            positions = st.exchange.get_positions()
        except Exception as e:
            await ctx.send(f"Lỗi lấy positions: {e}")
            return
        if not positions:
            await ctx.send("Không có vị thế đang mở.")
            return
        lines = ["**Open positions:**"]
        for p in positions:
            side = p.side.value if hasattr(p.side, "value") else str(p.side)
            lines.append(
                f"• `{p.symbol}` {side} size={p.size} entry={p.entry_price:.6g} "
                f"uPnL={getattr(p, 'unrealized_pnl', 0):.4f}"
            )
        await ctx.send("\n".join(lines[:25]))

    @bot.command(name="params")
    async def cmd_params(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        st = control_state
        rg = st.risk_gate
        cfg = (rg.cfg if rg else st.risk_cfg) or {}
        lines = ["**Risk params (có thể `set`):**"]
        for k in sorted(ALLOWED_RISK_KEYS.keys()):
            lines.append(f"• `{k}` = `{cfg.get(k, 'N/A')}`")
        lines.append("\n**Settings:**")
        for k in sorted(ALLOWED_SETTINGS_KEYS.keys()):
            lines.append(f"• `{k}` = `{st.settings.get(k, 'N/A')}`")
        await ctx.send("\n".join(lines))

    @bot.command(name="set")
    async def cmd_set(ctx, key: str = None, value: str = None):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        if not key or value is None:
            await ctx.send("Dùng: `!set <key> <value>` — xem `!params`")
            return
        key = key.strip().lower()
        st = control_state

        if key in ALLOWED_RISK_KEYS:
            typ, lo, hi = ALLOWED_RISK_KEYS[key]
            try:
                val = typ(value)
            except ValueError:
                await ctx.send(f"Giá trị không hợp lệ cho `{key}`")
                return
            if not (lo <= val <= hi):
                await ctx.send(f"`{key}` phải trong khoảng [{lo}, {hi}]")
                return
            if st.risk_gate:
                st.risk_gate.cfg[key] = val
            st.risk_cfg[key] = val
            # Đồng bộ leverage sang executor
            if key == "default_leverage" and st.executor:
                st.executor.default_leverage = int(val)
            if key == "max_leverage" and st.executor:
                st.executor.max_leverage = max(st.executor.default_leverage, int(val))
            _save_risk_yaml(st.risk_gate.cfg if st.risk_gate else st.risk_cfg)
            await ctx.send(f"✅ Đã set `{key}` = `{val}` (đã lưu risk.yaml)")
            return

        if key in ALLOWED_SETTINGS_KEYS:
            typ, lo, hi = ALLOWED_SETTINGS_KEYS[key]
            try:
                val = typ(value)
            except ValueError:
                await ctx.send(f"Giá trị không hợp lệ cho `{key}`")
                return
            if not (lo <= val <= hi):
                await ctx.send(f"`{key}` phải trong khoảng [{lo}, {hi}]")
                return
            st.settings[key] = val
            _save_settings_yaml(st.settings)
            await ctx.send(f"✅ Đã set `{key}` = `{val}` (đã lưu settings.yaml)")
            return

        await ctx.send(
            f"Key `{key}` không được phép chỉnh.\n"
            f"Risk: {', '.join(sorted(ALLOWED_RISK_KEYS))}\n"
            f"Settings: {', '.join(sorted(ALLOWED_SETTINGS_KEYS))}"
        )

    @bot.command(name="pause")
    async def cmd_pause(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        control_state.paused = True
        await ctx.send("⏸ Bot đã **pause** (không scan/execute). Vị thế vẫn giữ.")

    @bot.command(name="resume")
    async def cmd_resume(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        control_state.paused = False
        await ctx.send("▶️ Bot đã **resume**.")

    @bot.command(name="kill")
    async def cmd_kill(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        st = control_state
        if st.risk_gate:
            st.risk_gate.activate_kill_switch("DISCORD_KILL")
        await ctx.send("🛑 Kill switch **ON**. Bot sẽ flatten và tạm dừng.")

    @bot.command(name="unkill")
    async def cmd_unkill(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        st = control_state
        if st.risk_gate:
            st.risk_gate.reset_kill_switch()
        await ctx.send("✅ Kill switch **OFF**.")

    @bot.command(name="skip")
    async def cmd_skip(ctx, action: str = None):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        st = control_state
        if action == "clear" and st.executor:
            n = len(st.executor.skip_symbols)
            st.executor.skip_symbols.clear()
            await ctx.send(f"Đã xóa {n} symbol khỏi skip list.")
        else:
            n = len(getattr(st.executor, "skip_symbols", []) or [])
            await ctx.send(f"Skip list: {n} symbols. Dùng `!skip clear` để xóa.")

    return bot


def start_discord_in_thread() -> Optional[threading.Thread]:
    """Chạy Discord bot trong background thread. Trả về thread hoặc None."""
    if not HAS_DISCORD:
        return None
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        return None

    bot = create_bot()
    if bot is None:
        return None

    def _run():
        try:
            bot.run(token, log_handler=None)
        except Exception as e:
            logger.error(f"Discord bot crashed: {e}")

    t = threading.Thread(target=_run, name="discord-control", daemon=True)
    t.start()
    logger.info("Discord control thread started")
    return t
