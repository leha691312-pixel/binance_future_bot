"""
Discord Control Interface for Binance Futures Bot
Giao diện nút bấm (Buttons + Select) + vẫn hỗ trợ lệnh text cũ.

Cấu hình trong .env:
  DISCORD_BOT_TOKEN=...
  DISCORD_ALLOWED_IDS=123456789,987654321
  DISCORD_PREFIX=!                          # mặc định !

Cách dùng chính:
  !panel  hoặc  !menu  → hiện bảng điều khiển với nút
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
    from discord import app_commands
    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False
    discord = None
    commands = None
    app_commands = None


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
        self.paused: bool = False
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


def _user_authorized(user_id: int) -> bool:
    allowed = _parse_allowed_ids()
    if not allowed:
        return False
    return user_id in allowed


def _is_authorized(ctx) -> bool:
    return _user_authorized(ctx.author.id)


async def _check_interaction(interaction) -> bool:
    if not _user_authorized(interaction.user.id):
        await interaction.response.send_message(
            "⛔ Bạn không có quyền điều khiển bot.", ephemeral=True
        )
        return False
    return True


def _save_risk_yaml(risk_cfg: dict):
    path = Path(__file__).parent.parent / "config" / "risk.yaml"
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                risk_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
        logger.info("risk.yaml saved via Discord")
    except Exception as e:
        logger.warning(f"Failed to save risk.yaml: {e}")


def _save_settings_yaml(settings: dict):
    path = Path(__file__).parent.parent / "config" / "settings.yaml"
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                settings, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
        logger.info("settings.yaml saved via Discord")
    except Exception as e:
        logger.warning(f"Failed to save settings.yaml: {e}")


def _status_text() -> str:
    st = control_state
    rg = st.risk_gate
    if not rg:
        return "Bot chưa sẵn sàng."
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
        f"**Kill switch:** `{rg.kill_switch}`"
        + (f" ({rg.kill_reason})" if rg.kill_switch else ""),
        f"**Equity:** `{equity:.2f}` USDT",
        f"**Daily PnL:** `{rg.daily_pnl:.4f}`",
        f"**Open positions:** `{pos_n}` / max `{rg.cfg.get('max_open_positions', '?')}`",
        f"**Consecutive losses:** `{rg.consecutive_losses}`",
        f"**Leverage:** default=`{getattr(st.executor, 'default_leverage', '?')}` "
        f"max=`{getattr(st.executor, 'max_leverage', '?')}`",
        f"**Scan interval:** `{st.settings.get('scan_interval_seconds', '?')}s` | "
        f"top_n=`{st.settings.get('top_n', '?')}`",
    ]
    if st.last_cycle_info:
        lines.append(f"**Last cycle:** {st.last_cycle_info}")
    return "\n".join(lines)


def _positions_text() -> str:
    st = control_state
    if not st.exchange:
        return "Exchange chưa sẵn sàng."
    try:
        positions = st.exchange.get_positions()
    except Exception as e:
        return f"Lỗi lấy positions: {e}"
    if not positions:
        return "Không có vị thế đang mở."
    lines = ["**Open positions:**"]
    for p in positions[:25]:
        side = p.side.value if hasattr(p.side, "value") else str(p.side)
        lines.append(
            f"• `{p.symbol}` {side} size={p.size} entry={p.entry_price:.6g} "
            f"uPnL={getattr(p, 'unrealized_pnl', 0):.4f}"
        )
    return "\n".join(lines)


def _params_text() -> str:
    st = control_state
    rg = st.risk_gate
    cfg = (rg.cfg if rg else st.risk_cfg) or {}
    lines = ["**Risk params:**"]
    for k in sorted(ALLOWED_RISK_KEYS.keys()):
        lines.append(f"• `{k}` = `{cfg.get(k, 'N/A')}`")
    lines.append("\n**Settings:**")
    for k in sorted(ALLOWED_SETTINGS_KEYS.keys()):
        lines.append(f"• `{k}` = `{st.settings.get(k, 'N/A')}`")
    lines.append("\n*Chỉnh nhanh: menu bên dưới hoặc `!set key value`*")
    return "\n".join(lines)


def _apply_param(key: str, value: str) -> str:
    """Áp dụng 1 param; trả về message kết quả."""
    key = key.strip().lower()
    st = control_state

    if key in ALLOWED_RISK_KEYS:
        typ, lo, hi = ALLOWED_RISK_KEYS[key]
        try:
            val = typ(value)
        except ValueError:
            return f"Giá trị không hợp lệ cho `{key}`"
        if not (lo <= val <= hi):
            return f"`{key}` phải trong khoảng [{lo}, {hi}]"
        if st.risk_gate:
            st.risk_gate.cfg[key] = val
        st.risk_cfg[key] = val
        if key == "default_leverage" and st.executor:
            st.executor.default_leverage = int(val)
        if key == "max_leverage" and st.executor:
            st.executor.max_leverage = max(st.executor.default_leverage, int(val))
        _save_risk_yaml(st.risk_gate.cfg if st.risk_gate else st.risk_cfg)
        return f"✅ `{key}` = `{val}` (đã lưu risk.yaml)"

    if key in ALLOWED_SETTINGS_KEYS:
        typ, lo, hi = ALLOWED_SETTINGS_KEYS[key]
        try:
            val = typ(value)
        except ValueError:
            return f"Giá trị không hợp lệ cho `{key}`"
        if not (lo <= val <= hi):
            return f"`{key}` phải trong khoảng [{lo}, {hi}]"
        st.settings[key] = val
        _save_settings_yaml(st.settings)
        return f"✅ `{key}` = `{val}` (đã lưu settings.yaml)"

    return f"Key `{key}` không được phép chỉnh."


# ---------------------------------------------------------------------------
# UI Views
# ---------------------------------------------------------------------------

if HAS_DISCORD:

    class ConfirmKillView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=30)

        @discord.ui.button(label="Xác nhận KILL", style=discord.ButtonStyle.danger, emoji="🛑")
        async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            st = control_state
            if st.risk_gate:
                st.risk_gate.activate_kill_switch("DISCORD_KILL")
            await interaction.response.edit_message(
                content="🛑 Kill switch **ON**. Bot sẽ flatten và tạm dừng.",
                view=None,
            )

        @discord.ui.button(label="Hủy", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.edit_message(content="Đã hủy kill.", view=None)

    class ParamSelect(discord.ui.Select):
        def __init__(self):
            options = [
                discord.SelectOption(
                    label="default_leverage",
                    description="Leverage mặc định (1–20)",
                    value="default_leverage",
                ),
                discord.SelectOption(
                    label="max_leverage",
                    description="Leverage tối đa (1–50)",
                    value="max_leverage",
                ),
                discord.SelectOption(
                    label="risk_per_trade_pct",
                    description="Risk % mỗi lệnh",
                    value="risk_per_trade_pct",
                ),
                discord.SelectOption(
                    label="max_open_positions",
                    description="Số vị thế tối đa",
                    value="max_open_positions",
                ),
                discord.SelectOption(
                    label="min_risk_reward",
                    description="RR tối thiểu",
                    value="min_risk_reward",
                ),
                discord.SelectOption(
                    label="max_daily_loss_pct",
                    description="Giới hạn lỗ ngày",
                    value="max_daily_loss_pct",
                ),
                discord.SelectOption(
                    label="top_n",
                    description="Số symbol quét",
                    value="top_n",
                ),
                discord.SelectOption(
                    label="scan_interval_seconds",
                    description="Chu kỳ scan (giây)",
                    value="scan_interval_seconds",
                ),
            ]
            super().__init__(
                placeholder="Chọn thông số để chỉnh…",
                min_values=1,
                max_values=1,
                options=options,
                row=3,
            )

        async def callback(self, interaction: discord.Interaction):
            if not await _check_interaction(interaction):
                return
            key = self.values[0]
            await interaction.response.send_modal(SetParamModal(key))

    class SetParamModal(discord.ui.Modal):
        def __init__(self, key: str):
            super().__init__(title=f"Set {key}")
            self.key = key
            st = control_state
            cfg = (st.risk_gate.cfg if st.risk_gate else st.risk_cfg) or {}
            current = cfg.get(key, st.settings.get(key, ""))
            self.value_input = discord.ui.TextInput(
                label=f"Giá trị mới cho {key}",
                placeholder=str(current),
                default=str(current) if current != "" else None,
                required=True,
                max_length=32,
            )
            self.add_item(self.value_input)

        async def on_submit(self, interaction: discord.Interaction):
            if not _user_authorized(interaction.user.id):
                await interaction.response.send_message("⛔ Không có quyền.", ephemeral=True)
                return
            msg = _apply_param(self.key, str(self.value_input.value).strip())
            await interaction.response.send_message(msg, ephemeral=True)

    class ControlPanelView(discord.ui.View):
        """Bảng điều khiển chính – timeout dài để dùng lâu."""

        def __init__(self):
            super().__init__(timeout=600)
            self.add_item(ParamSelect())

        @discord.ui.button(label="Status", style=discord.ButtonStyle.primary, emoji="📊", row=0)
        async def btn_status(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_message(_status_text(), ephemeral=True)

        @discord.ui.button(label="Positions", style=discord.ButtonStyle.primary, emoji="📈", row=0)
        async def btn_positions(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_message(_positions_text(), ephemeral=True)

        @discord.ui.button(label="Params", style=discord.ButtonStyle.secondary, emoji="⚙️", row=0)
        async def btn_params(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_message(_params_text(), ephemeral=True)

        @discord.ui.button(label="Refresh panel", style=discord.ButtonStyle.secondary, emoji="🔄", row=0)
        async def btn_refresh(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.edit_message(
                content=_panel_header(),
                view=ControlPanelView(),
            )

        @discord.ui.button(label="Pause", style=discord.ButtonStyle.secondary, emoji="⏸", row=1)
        async def btn_pause(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            control_state.paused = True
            await interaction.response.send_message(
                "⏸ Bot đã **pause** (không scan/execute). Vị thế vẫn giữ.",
                ephemeral=True,
            )

        @discord.ui.button(label="Resume", style=discord.ButtonStyle.success, emoji="▶️", row=1)
        async def btn_resume(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            control_state.paused = False
            await interaction.response.send_message("▶️ Bot đã **resume**.", ephemeral=True)

        @discord.ui.button(label="Kill", style=discord.ButtonStyle.danger, emoji="🛑", row=1)
        async def btn_kill(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_message(
                "⚠️ Xác nhận bật kill switch (flatten + dừng)?",
                view=ConfirmKillView(),
                ephemeral=True,
            )

        @discord.ui.button(label="Unkill", style=discord.ButtonStyle.success, emoji="✅", row=1)
        async def btn_unkill(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            st = control_state
            if st.risk_gate:
                st.risk_gate.reset_kill_switch()
            await interaction.response.send_message("✅ Kill switch **OFF**.", ephemeral=True)

        @discord.ui.button(label="Clear skip", style=discord.ButtonStyle.secondary, emoji="🧹", row=2)
        async def btn_skip_clear(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            st = control_state
            if st.executor:
                n = len(st.executor.skip_symbols)
                st.executor.skip_symbols.clear()
                await interaction.response.send_message(
                    f"Đã xóa **{n}** symbol khỏi skip list.", ephemeral=True
                )
            else:
                await interaction.response.send_message("Executor chưa sẵn sàng.", ephemeral=True)


def _panel_header() -> str:
    st = control_state
    paused = "⏸ PAUSED" if st.paused else "▶️ RUNNING"
    kill = ""
    if st.risk_gate and st.risk_gate.kill_switch:
        kill = f" | 🛑 KILL: {st.risk_gate.kill_reason}"
    return (
        f"**Binance Futures Bot – Control Panel**\n"
        f"Mode: `{st.mode}` | {paused}{kill}\n"
        f"Bấm nút bên dưới để thao tác. Chỉ user được phép trong `DISCORD_ALLOWED_IDS`.\n"
        f"*Panel hết hạn sau ~10 phút → gõ `!panel` để mở lại.*"
    )


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

    @bot.command(name="panel", aliases=["menu", "ui"])
    async def cmd_panel(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Bạn không có quyền điều khiển bot.")
            return
        await ctx.send(content=_panel_header(), view=ControlPanelView())

    @bot.command(name="help")
    async def cmd_help(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Bạn không có quyền điều khiển bot.")
            return
        text = (
            "**Binance Futures Bot – Discord Control**\n"
            f"**Khuyến nghị:** `{prefix}panel` → bảng nút bấm\n\n"
            f"`{prefix}panel` / `{prefix}menu` – mở control panel (nút)\n"
            f"`{prefix}status` – trạng thái\n"
            f"`{prefix}positions` – vị thế\n"
            f"`{prefix}params` – thông số\n"
            f"`{prefix}set <key> <value>` – chỉnh thông số\n"
            f"`{prefix}pause` / `{prefix}resume`\n"
            f"`{prefix}kill` / `{prefix}unkill`\n"
            f"`{prefix}skip clear` – xóa skip list\n"
        )
        await ctx.send(text)

    @bot.command(name="status")
    async def cmd_status(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        await ctx.send(_status_text())

    @bot.command(name="positions")
    async def cmd_positions(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        await ctx.send(_positions_text())

    @bot.command(name="params")
    async def cmd_params(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        await ctx.send(_params_text())

    @bot.command(name="set")
    async def cmd_set(ctx, key: str = None, value: str = None):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.")
            return
        if not key or value is None:
            await ctx.send(f"Dùng: `{prefix}set <key> <value>` hoặc mở `{prefix}panel`")
            return
        await ctx.send(_apply_param(key, value))

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
        await ctx.send(
            "⚠️ Xác nhận bật kill switch?",
            view=ConfirmKillView(),
        )

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
            await ctx.send(f"Skip list: {n} symbols. Dùng `{prefix}skip clear` hoặc nút trên panel.")

    return bot


def start_discord_in_thread() -> Optional[threading.Thread]:
    """Chạy Discord bot trong background thread."""
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
