"""
Discord Control – UI tối ưu (Embed + Buttons + Select + Modal)

.env:
  DISCORD_BOT_TOKEN=...
  DISCORD_ALLOWED_IDS=123,456
  DISCORD_PREFIX=!

Dùng: !panel  (hoặc !menu / !ui)
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple
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

# Nhãn tiếng Việt cho select / modal
PARAM_LABELS = {
    "default_leverage": ("Đòn bẩy mặc định", "1–20"),
    "max_leverage": ("Đòn bẩy tối đa", "1–50"),
    "risk_per_trade_pct": ("Risk mỗi lệnh", "0.001–0.05 (vd 0.01 = 1%)"),
    "max_position_pct": ("% vốn 1 vị thế", "0.01–0.2"),
    "max_total_exposure_pct": ("Tổng exposure", "0.05–1.0"),
    "max_open_positions": ("Số vị thế tối đa", "1–20"),
    "min_risk_reward": ("RR tối thiểu", "1.0–5.0"),
    "max_daily_loss_pct": ("Lỗ ngày tối đa", "0.005–0.2"),
    "duplicate_window_seconds": ("Chống lệnh trùng (giây)", "30–3600"),
    "kill_on_consecutive_losses": ("Kill sau N lệnh thua", "2–20"),
    "cooldown_seconds": ("Cooldown sau lệnh (giây)", "0–600"),
    "top_n": ("Số symbol quét", "5–100"),
    "scan_interval_seconds": ("Chu kỳ scan (giây)", "30–600"),
}

COLOR_OK = 0x2ECC71
COLOR_WARN = 0xF1C40F
COLOR_DANGER = 0xE74C3C
COLOR_INFO = 0x3498DB
COLOR_IDLE = 0x95A5A6


class BotControlState:
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
    return bool(allowed) and user_id in allowed


def _is_authorized(ctx) -> bool:
    return _user_authorized(ctx.author.id)


async def _check_interaction(interaction) -> bool:
    if not _user_authorized(interaction.user.id):
        msg = "⛔ Bạn không có quyền điều khiển bot."
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)
        return False
    return True


def _save_risk_yaml(risk_cfg: dict):
    path = Path(__file__).parent.parent / "config" / "risk.yaml"
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                risk_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
    except Exception as e:
        logger.warning(f"Failed to save risk.yaml: {e}")


def _save_settings_yaml(settings: dict):
    path = Path(__file__).parent.parent / "config" / "settings.yaml"
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                settings, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
    except Exception as e:
        logger.warning(f"Failed to save settings.yaml: {e}")


def _snapshot() -> Dict[str, Any]:
    """Lấy snapshot trạng thái an toàn."""
    st = control_state
    rg = st.risk_gate
    equity = float(getattr(rg, "equity", 0) or 0) if rg else 0.0
    try:
        if st.exchange:
            equity = float(st.exchange.get_balance())
            if rg:
                rg.update_equity(equity)
    except Exception:
        pass

    pos_n = len(rg.open_positions) if rg else 0
    max_pos = (rg.cfg.get("max_open_positions") if rg else st.risk_cfg.get("max_open_positions")) or "?"
    daily = float(getattr(rg, "daily_pnl", 0) or 0) if rg else 0.0
    kill = bool(getattr(rg, "kill_switch", False)) if rg else False
    kill_reason = str(getattr(rg, "kill_reason", "") or "") if rg else ""
    losses = int(getattr(rg, "consecutive_losses", 0) or 0) if rg else 0
    dlev = getattr(st.executor, "default_leverage", st.risk_cfg.get("default_leverage", "?"))
    mlev = getattr(st.executor, "max_leverage", st.risk_cfg.get("max_leverage", "?"))

    return {
        "mode": st.mode,
        "paused": st.paused,
        "equity": equity,
        "daily_pnl": daily,
        "pos_n": pos_n,
        "max_pos": max_pos,
        "kill": kill,
        "kill_reason": kill_reason,
        "losses": losses,
        "dlev": dlev,
        "mlev": mlev,
        "scan": st.settings.get("scan_interval_seconds", "?"),
        "top_n": st.settings.get("top_n", "?"),
        "last_cycle": st.last_cycle_info or "—",
        "ready": rg is not None,
    }


def _panel_color(snap: dict) -> int:
    if not snap["ready"]:
        return COLOR_IDLE
    if snap["kill"]:
        return COLOR_DANGER
    if snap["paused"]:
        return COLOR_WARN
    return COLOR_OK


def _status_embed() -> "discord.Embed":
    snap = _snapshot()
    if not snap["ready"]:
        return discord.Embed(
            title="Bot chưa sẵn sàng",
            description="Đợi main loop khởi tạo xong rồi bấm **Làm mới**.",
            color=COLOR_IDLE,
        )

    if snap["kill"]:
        state = f"🛑 KILL — {snap['kill_reason'] or 'active'}"
    elif snap["paused"]:
        state = "⏸ Tạm dừng (pause)"
    else:
        state = "▶️ Đang chạy"

    pnl = snap["daily_pnl"]
    pnl_s = f"{'+' if pnl >= 0 else ''}{pnl:.4f}"

    emb = discord.Embed(
        title="📊 Trạng thái bot",
        color=_panel_color(snap),
        timestamp=datetime.now(timezone.utc),
    )
    emb.add_field(name="Trạng thái", value=state, inline=True)
    emb.add_field(name="Mode", value=f"`{snap['mode']}`", inline=True)
    emb.add_field(name="Equity", value=f"`{snap['equity']:.2f}` USDT", inline=True)
    emb.add_field(name="PnL ngày", value=f"`{pnl_s}`", inline=True)
    emb.add_field(
        name="Vị thế",
        value=f"`{snap['pos_n']}` / `{snap['max_pos']}`",
        inline=True,
    )
    emb.add_field(name="Thua liên tiếp", value=f"`{snap['losses']}`", inline=True)
    emb.add_field(
        name="Leverage",
        value=f"default `{snap['dlev']}` · max `{snap['mlev']}`",
        inline=True,
    )
    emb.add_field(
        name="Scan",
        value=f"mỗi `{snap['scan']}`s · top `{snap['top_n']}`",
        inline=True,
    )
    emb.add_field(
        name="Cycle gần nhất",
        value=str(snap["last_cycle"])[:200],
        inline=False,
    )
    emb.set_footer(text="Binance Futures Bot")
    return emb


def _positions_embed() -> "discord.Embed":
    st = control_state
    emb = discord.Embed(title="📈 Vị thế đang mở", color=COLOR_INFO, timestamp=datetime.now(timezone.utc))
    if not st.exchange:
        emb.description = "Exchange chưa sẵn sàng."
        emb.color = COLOR_IDLE
        return emb
    try:
        positions = st.exchange.get_positions()
    except Exception as e:
        emb.description = f"Lỗi: `{e}`"
        emb.color = COLOR_DANGER
        return emb
    if not positions:
        emb.description = "Không có vị thế nào."
        emb.color = COLOR_IDLE
        return emb

    total_upnl = 0.0
    lines = []
    for p in positions[:20]:
        side = p.side.value if hasattr(p.side, "value") else str(p.side)
        upnl = float(getattr(p, "unrealized_pnl", 0) or 0)
        total_upnl += upnl
        flag = "🟢" if side.upper() == "LONG" else "🔴"
        lines.append(
            f"{flag} `{p.symbol}` **{side}** size=`{p.size}`\n"
            f"　entry `{p.entry_price:.6g}` · uPnL `{upnl:+.4f}`"
        )
    emb.description = "\n".join(lines)
    emb.add_field(name="Tổng uPnL", value=f"`{total_upnl:+.4f}`", inline=True)
    emb.add_field(name="Số lệnh", value=f"`{len(positions)}`", inline=True)
    emb.set_footer(text="Tối đa 20 dòng")
    return emb


def _params_embed() -> "discord.Embed":
    st = control_state
    rg = st.risk_gate
    cfg = (rg.cfg if rg else st.risk_cfg) or {}
    emb = discord.Embed(
        title="⚙️ Thông số",
        description="Chọn trong menu **Chỉnh thông số** hoặc dùng preset đòn bẩy.",
        color=COLOR_INFO,
        timestamp=datetime.now(timezone.utc),
    )
    risk_lines = []
    for k in sorted(ALLOWED_RISK_KEYS.keys()):
        label = PARAM_LABELS.get(k, (k, ""))[0]
        risk_lines.append(f"• **{label}** (`{k}`): `{cfg.get(k, 'N/A')}`")
    emb.add_field(name="Risk", value="\n".join(risk_lines)[:1020], inline=False)
    set_lines = []
    for k in sorted(ALLOWED_SETTINGS_KEYS.keys()):
        label = PARAM_LABELS.get(k, (k, ""))[0]
        set_lines.append(f"• **{label}** (`{k}`): `{st.settings.get(k, 'N/A')}`")
    emb.add_field(name="Settings", value="\n".join(set_lines), inline=False)
    return emb


def _panel_embed() -> "discord.Embed":
    snap = _snapshot()
    if snap["kill"]:
        state = f"🛑 KILL ({snap['kill_reason'] or 'on'})"
    elif snap["paused"]:
        state = "⏸ PAUSED"
    elif snap["ready"]:
        state = "▶️ RUNNING"
    else:
        state = "⏳ STARTING"

    emb = discord.Embed(
        title="🎛️ Control Panel",
        description=(
            f"**{state}** · mode `{snap['mode']}`\n"
            f"Equity **{snap['equity']:.2f}** USDT · "
            f"PnL ngày `{snap['daily_pnl']:+.4f}` · "
            f"Positions **{snap['pos_n']}/{snap['max_pos']}**\n\n"
            f"Dùng nút bên dưới. Chỉ user trong `DISCORD_ALLOWED_IDS`."
        ),
        color=_panel_color(snap),
        timestamp=datetime.now(timezone.utc),
    )
    emb.add_field(
        name="Leverage",
        value=f"`{snap['dlev']}`x (max `{snap['mlev']}`)",
        inline=True,
    )
    emb.add_field(
        name="Scan",
        value=f"`{snap['scan']}`s · top `{snap['top_n']}`",
        inline=True,
    )
    emb.set_footer(text="Làm mới để cập nhật · Panel ~30 phút")
    return emb


def _apply_param(key: str, value: str) -> str:
    key = key.strip().lower()
    st = control_state

    if key in ALLOWED_RISK_KEYS:
        typ, lo, hi = ALLOWED_RISK_KEYS[key]
        try:
            val = typ(value)
        except ValueError:
            return f"❌ Giá trị không hợp lệ cho `{key}`"
        if not (lo <= val <= hi):
            return f"❌ `{key}` phải trong [{lo}, {hi}]"
        if st.risk_gate:
            st.risk_gate.cfg[key] = val
        st.risk_cfg[key] = val
        if key == "default_leverage" and st.executor:
            st.executor.default_leverage = int(val)
        if key == "max_leverage" and st.executor:
            st.executor.max_leverage = max(int(getattr(st.executor, "default_leverage", 1)), int(val))
        _save_risk_yaml(st.risk_gate.cfg if st.risk_gate else st.risk_cfg)
        label = PARAM_LABELS.get(key, (key,))[0]
        return f"✅ **{label}** = `{val}`"

    if key in ALLOWED_SETTINGS_KEYS:
        typ, lo, hi = ALLOWED_SETTINGS_KEYS[key]
        try:
            val = typ(value)
        except ValueError:
            return f"❌ Giá trị không hợp lệ cho `{key}`"
        if not (lo <= val <= hi):
            return f"❌ `{key}` phải trong [{lo}, {hi}]"
        st.settings[key] = val
        _save_settings_yaml(st.settings)
        label = PARAM_LABELS.get(key, (key,))[0]
        return f"✅ **{label}** = `{val}`"

    return f"❌ Key `{key}` không được phép."


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

if HAS_DISCORD:

    class ConfirmKillView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=45)

        @discord.ui.button(
            label="Xác nhận KILL", style=discord.ButtonStyle.danger, emoji="🛑"
        )
        async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            st = control_state
            if st.risk_gate:
                st.risk_gate.activate_kill_switch("DISCORD_KILL")
            emb = discord.Embed(
                title="🛑 Kill switch ON",
                description="Bot sẽ flatten và tạm dừng vòng lặp.",
                color=COLOR_DANGER,
            )
            await interaction.response.edit_message(content=None, embed=emb, view=None)

        @discord.ui.button(label="Hủy", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.edit_message(
                content="Đã hủy kill.", embed=None, view=None
            )

    class ParamSelect(discord.ui.Select):
        def __init__(self):
            options = []
            for key in [
                "default_leverage",
                "max_leverage",
                "risk_per_trade_pct",
                "max_open_positions",
                "min_risk_reward",
                "max_daily_loss_pct",
                "top_n",
                "scan_interval_seconds",
            ]:
                label, hint = PARAM_LABELS.get(key, (key, ""))
                st = control_state
                cfg = (st.risk_gate.cfg if st.risk_gate else st.risk_cfg) or {}
                cur = cfg.get(key, st.settings.get(key, "?"))
                options.append(
                    discord.SelectOption(
                        label=label[:100],
                        description=f"Hiện: {cur} · {hint}"[:100],
                        value=key,
                    )
                )
            super().__init__(
                placeholder="✏️ Chỉnh thông số…",
                min_values=1,
                max_values=1,
                options=options,
                row=3,
            )

        async def callback(self, interaction: discord.Interaction):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_modal(SetParamModal(self.values[0]))

    class SetParamModal(discord.ui.Modal):
        def __init__(self, key: str):
            label, hint = PARAM_LABELS.get(key, (key, ""))
            super().__init__(title=f"Chỉnh: {label}"[:45])
            self.key = key
            st = control_state
            cfg = (st.risk_gate.cfg if st.risk_gate else st.risk_cfg) or {}
            current = cfg.get(key, st.settings.get(key, ""))
            self.value_input = discord.ui.TextInput(
                label=f"Giá trị ({hint})"[:45],
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
            emb = discord.Embed(
                description=msg,
                color=COLOR_OK if msg.startswith("✅") else COLOR_DANGER,
            )
            await interaction.response.send_message(embed=emb, ephemeral=True)

    class ControlPanelView(discord.ui.View):
        """Panel chính – timeout 30 phút."""

        def __init__(self):
            super().__init__(timeout=1800)
            self.add_item(ParamSelect())

        async def on_timeout(self):
            # Không edit được message ở đây dễ dàng không có ref; user gõ !panel lại
            pass

        async def _refresh_panel(self, interaction: discord.Interaction):
            await interaction.response.edit_message(
                content=None,
                embed=_panel_embed(),
                view=ControlPanelView(),
            )

        @discord.ui.button(label="Làm mới", style=discord.ButtonStyle.primary, emoji="🔄", row=0)
        async def btn_refresh(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await self._refresh_panel(interaction)

        @discord.ui.button(label="Trạng thái", style=discord.ButtonStyle.secondary, emoji="📊", row=0)
        async def btn_status(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_message(embed=_status_embed(), ephemeral=True)

        @discord.ui.button(label="Vị thế", style=discord.ButtonStyle.secondary, emoji="📈", row=0)
        async def btn_positions(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_message(embed=_positions_embed(), ephemeral=True)

        @discord.ui.button(label="Thông số", style=discord.ButtonStyle.secondary, emoji="⚙️", row=0)
        async def btn_params(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_message(embed=_params_embed(), ephemeral=True)

        @discord.ui.button(label="Tạm dừng", style=discord.ButtonStyle.secondary, emoji="⏸", row=1)
        async def btn_pause(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            control_state.paused = True
            await interaction.response.edit_message(
                embed=_panel_embed(), view=ControlPanelView()
            )

        @discord.ui.button(label="Tiếp tục", style=discord.ButtonStyle.success, emoji="▶️", row=1)
        async def btn_resume(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            control_state.paused = False
            await interaction.response.edit_message(
                embed=_panel_embed(), view=ControlPanelView()
            )

        @discord.ui.button(label="Kill", style=discord.ButtonStyle.danger, emoji="🛑", row=1)
        async def btn_kill(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            emb = discord.Embed(
                title="⚠️ Xác nhận Kill",
                description=(
                    "Bật kill switch sẽ **flatten** vị thế và dừng bot.\n"
                    "Chỉ bấm xác nhận nếu bạn chắc chắn."
                ),
                color=COLOR_DANGER,
            )
            await interaction.response.send_message(
                embed=emb, view=ConfirmKillView(), ephemeral=True
            )

        @discord.ui.button(label="Gỡ Kill", style=discord.ButtonStyle.success, emoji="✅", row=1)
        async def btn_unkill(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            if control_state.risk_gate:
                control_state.risk_gate.reset_kill_switch()
            await interaction.response.edit_message(
                embed=_panel_embed(), view=ControlPanelView()
            )

        @discord.ui.button(label="Lev 3x", style=discord.ButtonStyle.secondary, row=2)
        async def btn_lev3(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            msg = _apply_param("default_leverage", "3")
            await interaction.response.send_message(msg, ephemeral=True)

        @discord.ui.button(label="Lev 5x", style=discord.ButtonStyle.secondary, row=2)
        async def btn_lev5(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            msg = _apply_param("default_leverage", "5")
            await interaction.response.send_message(msg, ephemeral=True)

        @discord.ui.button(label="Lev 10x", style=discord.ButtonStyle.secondary, row=2)
        async def btn_lev10(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            msg = _apply_param("default_leverage", "10")
            await interaction.response.send_message(msg, ephemeral=True)

        @discord.ui.button(
            label="Xóa skip", style=discord.ButtonStyle.secondary, emoji="🧹", row=2
        )
        async def btn_skip(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            st = control_state
            if st.executor:
                n = len(st.executor.skip_symbols)
                st.executor.skip_symbols.clear()
                await interaction.response.send_message(
                    f"🧹 Đã xóa **{n}** symbol khỏi skip list.", ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    "Executor chưa sẵn sàng.", ephemeral=True
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

    @bot.event
    async def on_command_error(ctx, error):
        if isinstance(error, commands.CommandNotFound):
            return
        logger.warning(f"Discord command error: {error}")
        try:
            await ctx.send(f"⚠️ Lỗi: `{error}`", delete_after=15)
        except Exception:
            pass

    @bot.command(name="panel", aliases=["menu", "ui"])
    async def cmd_panel(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Bạn không có quyền điều khiển bot.", delete_after=10)
            return
        await ctx.send(embed=_panel_embed(), view=ControlPanelView())

    @bot.command(name="help")
    async def cmd_help(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        emb = discord.Embed(
            title="Trợ giúp điều khiển",
            description=(
                f"**Khuyến nghị:** `{prefix}panel` — bảng nút + embed\n\n"
                f"`{prefix}panel` · `{prefix}menu` · `{prefix}ui`\n"
                f"`{prefix}status` · `{prefix}positions` · `{prefix}params`\n"
                f"`{prefix}set <key> <value>`\n"
                f"`{prefix}pause` · `{prefix}resume`\n"
                f"`{prefix}kill` · `{prefix}unkill`\n"
                f"`{prefix}skip clear`"
            ),
            color=COLOR_INFO,
        )
        await ctx.send(embed=emb)

    @bot.command(name="status")
    async def cmd_status(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        await ctx.send(embed=_status_embed())

    @bot.command(name="positions")
    async def cmd_positions(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        await ctx.send(embed=_positions_embed())

    @bot.command(name="params")
    async def cmd_params(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        await ctx.send(embed=_params_embed())

    @bot.command(name="set")
    async def cmd_set(ctx, key: str = None, value: str = None):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        if not key or value is None:
            await ctx.send(f"Dùng: `{prefix}set <key> <value>` hoặc `{prefix}panel`")
            return
        await ctx.send(_apply_param(key, value))

    @bot.command(name="pause")
    async def cmd_pause(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        control_state.paused = True
        await ctx.send("⏸ Bot đã **pause**.")

    @bot.command(name="resume")
    async def cmd_resume(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        control_state.paused = False
        await ctx.send("▶️ Bot đã **resume**.")

    @bot.command(name="kill")
    async def cmd_kill(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        emb = discord.Embed(
            title="⚠️ Xác nhận Kill",
            description="Bật kill = flatten + dừng.",
            color=COLOR_DANGER,
        )
        await ctx.send(embed=emb, view=ConfirmKillView())

    @bot.command(name="unkill")
    async def cmd_unkill(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        if control_state.risk_gate:
            control_state.risk_gate.reset_kill_switch()
        await ctx.send("✅ Kill switch **OFF**.")

    @bot.command(name="skip")
    async def cmd_skip(ctx, action: str = None):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        st = control_state
        if action == "clear" and st.executor:
            n = len(st.executor.skip_symbols)
            st.executor.skip_symbols.clear()
            await ctx.send(f"Đã xóa {n} symbol khỏi skip list.")
        else:
            n = len(getattr(st.executor, "skip_symbols", []) or [])
            await ctx.send(f"Skip: {n} symbols. `{prefix}skip clear` hoặc nút trên panel.")

    return bot


def start_discord_in_thread() -> Optional[threading.Thread]:
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
