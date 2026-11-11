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
import subprocess
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

# String settings (không dùng min/max số)
ALLOWED_SETTINGS_STR = {
    "margin_mode": ("isolated", "cross"),
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
    "margin_mode": ("Margin mode", "isolated | cross"),
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
        self.request_restart: bool = False
        self.last_cycle_info: str = ""
        self.scanner = None

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


def _normalize_symbol(sym: str, quote: str = "USDT") -> str:
    if not sym:
        return ""
    s = str(sym).strip().upper().replace(" ", "")
    if not s:
        return ""
    if f":{quote}" in s:
        return s
    if s.endswith(f"/{quote}"):
        return f"{s}:{quote}"
    if s.endswith(quote) and "/" not in s:
        base = s[: -len(quote)]
        if base:
            return f"{base}/{quote}:{quote}"
    if "/" in s and ":" not in s:
        return f"{s}:{quote}"
    if "/" not in s and ":" not in s:
        return f"{s}/{quote}:{quote}"
    return s


def _get_excluded_list() -> list:
    st = control_state
    raw = list(st.settings.get("excluded_symbols") or [])
    sc = getattr(st, "scanner", None)
    if sc is not None and hasattr(sc, "excluded_list_display"):
        try:
            live = sc.excluded_list_display()
            if live is not None:
                return list(live)
        except Exception:
            pass
    out, seen = [], set()
    for x in raw:
        n = _normalize_symbol(str(x))
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _persist_excluded(symbols: list) -> list:
    st = control_state
    cleaned, seen = [], set()
    for x in symbols:
        n = _normalize_symbol(str(x))
        if n and n not in seen:
            seen.add(n)
            cleaned.append(n)
    cleaned.sort()
    st.settings["excluded_symbols"] = cleaned
    _save_settings_yaml(st.settings)
    sc = getattr(st, "scanner", None)
    if sc is not None and hasattr(sc, "set_excluded_symbols"):
        sc.set_excluded_symbols(cleaned)
    return cleaned


def _format_excluded_msg() -> str:
    items = _get_excluded_list()
    if not items:
        return "📭 Blacklist trống — bot quét tất cả top_n."
    lines = "\n".join(f"• `{s}`" for s in items[:40])
    more = f"\n… +{len(items) - 40}" if len(items) > 40 else ""
    return f"🚫 **Blacklist ({len(items)}):**\n{lines}{more}"


def _repo_root() -> Path:
    """Thư mục gốc repo (chứa main.py, .git)."""
    return Path(__file__).resolve().parent.parent


def _git_pull() -> Tuple[bool, str]:
    """git pull --ff-only trong thư mục bot. Trả về (ok, log)."""
    root = _repo_root()
    if not (root / ".git").exists():
        return False, f"Không thấy .git tại `{root}` — bot chưa clone bằng git?"
    try:
        r = subprocess.run(
            ["git", "pull", "--ff-only", "origin", "main"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        if r.returncode != 0:
            # fallback: git pull không chỉ định remote/branch
            r2 = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            out2 = ((r2.stdout or "") + "\n" + (r2.stderr or "")).strip()
            if r2.returncode == 0:
                return True, out2[:1500] or "Already up to date."
            return False, (out2 or out or f"exit {r2.returncode}")[:1500]
        return True, (out or "Already up to date.")[:1500]
    except FileNotFoundError:
        return False, "Không tìm thấy lệnh `git` trên máy bot."
    except subprocess.TimeoutExpired:
        return False, "git pull timeout (>120s)."
    except Exception as e:
        return False, str(e)[:500]



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
        "excluded": len(_get_excluded_list()),
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
        name="Blacklist",
        value=f"`{snap.get('excluded', 0)}` symbol loại khỏi scan",
        inline=True,
    )
    emb.add_field(
        name="Cycle gần nhất",
        value=str(snap["last_cycle"])[:200],
        inline=False,
    )
    emb.set_footer(text="Binance Futures Bot")
    return emb


def _lookup_position_meta(symbol: str, side: str) -> dict:
    """Lấy strategy / trade_style từ RiskGate (lệnh bot mở)."""
    st = control_state
    rg = st.risk_gate
    if not rg:
        return {}
    side_u = (side or "").upper()
    key = f"{symbol}_{side_u}"
    pos = (rg.open_positions or {}).get(key)
    if pos is None:
        # fallback: match symbol only
        for k, p in (rg.open_positions or {}).items():
            if str(k).startswith(str(symbol) + "_"):
                pos = p
                break
    if pos is None:
        return {}
    strategy = getattr(pos, "strategy", None) or ""
    style = getattr(pos, "trade_style", None) or ""
    if not style and strategy and hasattr(rg, "resolve_trade_style"):
        try:
            style = rg.resolve_trade_style(strategy, getattr(pos, "timeframe", "") or "")
        except Exception:
            style = ""
    tf = getattr(pos, "timeframe", None) or ""
    return {"strategy": strategy, "trade_style": style, "timeframe": tf}


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
        flag = "🟢" if str(side).upper() == "LONG" else "🔴"
        meta = _lookup_position_meta(getattr(p, "symbol", ""), side)
        # strategy từ exchange object (hiếm) hoặc RiskGate
        strategy = (
            meta.get("strategy")
            or getattr(p, "strategy", None)
            or "—"
        )
        style = meta.get("trade_style") or getattr(p, "trade_style", None) or ""
        tf = meta.get("timeframe") or getattr(p, "timeframe", None) or ""
        strat_s = str(strategy) if strategy else "—"
        style_s = f" · `{style}`" if style else ""
        tf_s = f" · `{tf}`" if tf else ""
        lines.append(
            f"{flag} `{p.symbol}` **{side}** size=`{p.size}`\n"
            f"　strategy **`{strat_s}`**{style_s}{tf_s}\n"
            f"　entry `{p.entry_price:.6g}` · uPnL `{upnl:+.4f}`"
        )
    emb.description = "\n".join(lines)[:4000]
    emb.add_field(name="Tổng uPnL", value=f"`{total_upnl:+.4f}`", inline=True)
    emb.add_field(name="Số lệnh", value=f"`{len(positions)}`", inline=True)
    emb.set_footer(text="Strategy lấy từ RiskGate · tối đa 20 dòng")
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
    for k in sorted(ALLOWED_SETTINGS_STR.keys()):
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

    # String settings (vd: margin_mode)
    if key in ALLOWED_SETTINGS_STR:
        allowed = ALLOWED_SETTINGS_STR[key]
        val = str(value).strip().lower()
        if val not in allowed:
            return f"❌ `{key}` chỉ nhận: {', '.join(allowed)}"
        st.settings[key] = val
        # Áp dụng runtime cho exchange (lệnh mới)
        if st.exchange is not None and key == "margin_mode":
            try:
                st.exchange.margin_mode = val
                if hasattr(st.exchange, "config") and isinstance(st.exchange.config, dict):
                    st.exchange.config["margin_mode"] = val
            except Exception as e:
                logger.warning(f"Apply margin_mode runtime failed: {e}")
        _save_settings_yaml(st.settings)
        label = PARAM_LABELS.get(key, (key,))[0]
        note = ""
        if key == "margin_mode":
            note = (
                "\n⚠️ Chỉ áp dụng cho **vị thế mới**. "
                "Vị thế đang mở giữ nguyên margin type trên sàn."
            )
        return f"✅ **{label}** = `{val}`{note}"

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

    class ConfirmRestartView(discord.ui.View):
        """Xác nhận restart bot (không flatten vị thế)."""

        def __init__(self):
            super().__init__(timeout=45)

        @discord.ui.button(
            label="Xác nhận RESTART", style=discord.ButtonStyle.danger, emoji="🔁"
        )
        async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            control_state.request_restart = True
            emb = discord.Embed(
                title="🔁 Restart đã yêu cầu",
                description=(
                    "Bot sẽ **khởi động lại process** trong vài giây.\n"
                    "Vị thế trên sàn **không** bị đóng.\n"
                    "Discord có thể mất kết nối tạm rồi online lại."
                ),
                color=COLOR_WARN,
            )
            await interaction.response.edit_message(content=None, embed=emb, view=None)
            logger.warning(
                f"Discord restart requested by user {interaction.user.id}"
            )

        @discord.ui.button(label="Hủy", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.edit_message(
                content="Đã hủy restart.", embed=None, view=None
            )

    class ConfirmPullRestartView(discord.ui.View):
        """git pull rồi restart process."""

        def __init__(self):
            super().__init__(timeout=60)

        @discord.ui.button(
            label="Pull & Restart", style=discord.ButtonStyle.danger, emoji="📥"
        )
        async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.defer(ephemeral=True)
            ok, log = _git_pull()
            logger.info(f"Discord git pull ok={ok} | {log[:300]}")
            if not ok:
                emb = discord.Embed(
                    title="❌ git pull thất bại",
                    description=f"```\n{log}\n```\nBot **không** restart. Xử lý conflict trên máy bot rồi thử lại.",
                    color=COLOR_DANGER,
                )
                await interaction.followup.send(embed=emb, ephemeral=True)
                return
            control_state.request_restart = True
            emb = discord.Embed(
                title="📥 Pull OK — đang Restart",
                description=(
                    f"```\n{log}\n```\n"
                    "Bot sẽ **khởi động lại** trong vài giây.\n"
                    "Vị thế trên sàn **không** bị đóng."
                ),
                color=COLOR_OK,
            )
            await interaction.followup.send(embed=emb, ephemeral=True)
            logger.warning(f"Discord pull+restart by user {interaction.user.id}")

        @discord.ui.button(label="Hủy", style=discord.ButtonStyle.secondary)
        async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.edit_message(
                content="Đã hủy pull.", embed=None, view=None
            )

    class ExcludeSymbolModal(discord.ui.Modal):
        """Modal nhập symbol để thêm vào blacklist (không quét)."""

        def __init__(self):
            super().__init__(title="Loại symbol khỏi scan")
            self.symbols_input = discord.ui.TextInput(
                label="Symbol (cách nhau bằng dấu cách)",
                placeholder="BTC DOGE PEPE  hoặc  BTC/USDT:USDT",
                required=True,
                max_length=200,
                style=discord.TextStyle.short,
            )
            self.add_item(self.symbols_input)

        async def on_submit(self, interaction: discord.Interaction):
            if not _user_authorized(interaction.user.id):
                await interaction.response.send_message("⛔ Không có quyền.", ephemeral=True)
                return
            raw = str(self.symbols_input.value or "").strip()
            parts = [p for p in raw.replace(",", " ").split() if p.strip()]
            if not parts:
                await interaction.response.send_message(
                    "❌ Chưa nhập symbol nào.", ephemeral=True
                )
                return
            current = list(_get_excluded_list())
            added = []
            for p in parts:
                n = _normalize_symbol(p)
                if n and n not in current:
                    current.append(n)
                    added.append(n)
            _persist_excluded(current)
            if added:
                msg = (
                    f"✅ Đã loại **{len(added)}** symbol khỏi scan:\n"
                    + "\n".join(f"• `{s}`" for s in added)
                    + f"\n\n{_format_excluded_msg()}"
                )
                color = COLOR_OK
            else:
                msg = f"ℹ️ Symbol đã có trong blacklist.\n\n{_format_excluded_msg()}"
                color = COLOR_INFO
            emb = discord.Embed(description=msg[:4000], color=color)
            await interaction.response.send_message(embed=emb, ephemeral=True)

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

        # ---- Row 4: blacklist + restart ----
        @discord.ui.button(
            label="Loại symbol", style=discord.ButtonStyle.secondary, emoji="🚫", row=4
        )
        async def btn_exclude(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            await interaction.response.send_modal(ExcludeSymbolModal())

        @discord.ui.button(
            label="Xem blacklist", style=discord.ButtonStyle.secondary, emoji="📋", row=4
        )
        async def btn_exclude_list(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            emb = discord.Embed(
                title="Blacklist (không quét)",
                description=_format_excluded_msg(),
                color=COLOR_INFO,
            )
            await interaction.response.send_message(embed=emb, ephemeral=True)

        @discord.ui.button(
            label="Xóa blacklist", style=discord.ButtonStyle.secondary, emoji="♻️", row=4
        )
        async def btn_exclude_clear(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            _persist_excluded([])
            await interaction.response.send_message(
                "✅ Đã xóa toàn bộ blacklist — bot quét lại top_n.",
                ephemeral=True,
            )

        @discord.ui.button(
            label="Restart bot", style=discord.ButtonStyle.danger, emoji="🔁", row=4
        )
        async def btn_restart(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            emb = discord.Embed(
                title="⚠️ Xác nhận Restart",
                description=(
                    "Bot sẽ **tắt process và chạy lại** (load code/config mới).\n"
                    "• Vị thế trên sàn **không** bị đóng\n"
                    "• Discord offline vài giây rồi online lại\n"
                    "Chỉ dùng khi cần nạp lại code sau `git pull`."
                ),
                color=COLOR_WARN,
            )
            await interaction.response.send_message(
                embed=emb, view=ConfirmRestartView(), ephemeral=True
            )

        @discord.ui.button(
            label="Pull & Restart", style=discord.ButtonStyle.danger, emoji="📥", row=4
        )
        async def btn_pull_restart(self, interaction: discord.Interaction, button: discord.ui.Button):
            if not await _check_interaction(interaction):
                return
            emb = discord.Embed(
                title="⚠️ Xác nhận Pull & Restart",
                description=(
                    "Bot sẽ chạy **`git pull`** từ GitHub rồi **restart** process.\n"
                    "• Cần máy bot đã `git clone` repo và có mạng\n"
                    "• Vị thế trên sàn **không** bị đóng\n"
                    "• Nếu pull conflict → **không** restart, báo lỗi tại đây"
                ),
                color=COLOR_WARN,
            )
            await interaction.response.send_message(
                embed=emb, view=ConfirmPullRestartView(), ephemeral=True
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
                f"`{prefix}restart` — khởi động lại bot (có xác nhận)\n"
                f"`{prefix}pull` — git pull + restart (có xác nhận)\n"
                f"`{prefix}skip clear`\n"
                f"`{prefix}exclude list|add|remove|clear`\n\n"
                "**Trên panel:** 🚫 Loại symbol · 📋 Blacklist · "
                "🔁 Restart · 📥 Pull & Restart"
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

    @bot.command(name="restart")
    async def cmd_restart(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        emb = discord.Embed(
            title="⚠️ Xác nhận Restart",
            description=(
                "Bot sẽ tắt process và chạy lại.\n"
                "Vị thế trên sàn **không** bị đóng."
            ),
            color=COLOR_WARN,
        )
        await ctx.send(embed=emb, view=ConfirmRestartView())

    @bot.command(name="pull")
    async def cmd_pull(ctx):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        emb = discord.Embed(
            title="⚠️ Xác nhận Pull & Restart",
            description=(
                "Bot sẽ **`git pull`** từ GitHub rồi **restart**.\n"
                "Vị thế trên sàn **không** bị đóng.\n"
                "Conflict → không restart, báo lỗi."
            ),
            color=COLOR_WARN,
        )
        await ctx.send(embed=emb, view=ConfirmPullRestartView())

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
            await ctx.send(
                f"Skip runtime: **{n}**. `{prefix}skip clear` | `{prefix}exclude list`"
            )

    @bot.command(name="exclude", aliases=["blacklist", "ban"])
    async def cmd_exclude(ctx, action: str = None, *symbols: str):
        if not _is_authorized(ctx):
            await ctx.send("⛔ Không có quyền.", delete_after=10)
            return
        act = (action or "list").lower()
        current = list(_get_excluded_list())
        if act in ("list", "ls", "show"):
            await ctx.send(_format_excluded_msg())
            return
        if act == "clear":
            _persist_excluded([])
            await ctx.send("🧹 Đã xóa hết blacklist.")
            return
        if act in ("add", "+") or (action and act not in ("list", "remove", "clear", "add", "rm", "-")):
            syms = list(symbols) if act in ("add", "+") else ([action] + list(symbols or []))
            if act in ("add", "+") and not symbols:
                await ctx.send(f"Dùng: `{prefix}exclude add BTC DOGE`")
                return
            added = []
            for raw in syms:
                if not raw:
                    continue
                n = _normalize_symbol(raw)
                if n and n not in current:
                    current.append(n)
                    added.append(n)
            cleaned = _persist_excluded(current)
            msg = ", ".join(f"`{s}`" for s in added) if added else "đã có"
            await ctx.send(f"✅ Thêm: {msg} | tổng **{len(cleaned)}**")
            return
        if act in ("remove", "rm", "-"):
            if not symbols:
                await ctx.send(f"Dùng: `{prefix}exclude remove BTC`")
                return
            removed, targets, bases = [], set(), set()
            for raw in symbols:
                n = _normalize_symbol(raw)
                if n:
                    targets.add(n)
                    bases.add(n.split("/")[0])
                bases.add(str(raw).strip().upper())
            new_list = []
            for s in current:
                if s in targets or s.split("/")[0] in bases:
                    removed.append(s)
                else:
                    new_list.append(s)
            cleaned = _persist_excluded(new_list)
            msg = ", ".join(f"`{s}`" for s in removed) if removed else "—"
            await ctx.send(f"🗑️ Gỡ: {msg} | còn **{len(cleaned)}**")
            return
        await ctx.send(f"`{prefix}exclude list|add|remove|clear`")

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
