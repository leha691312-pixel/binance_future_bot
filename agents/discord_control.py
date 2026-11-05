"""
Discord Control – panel + exclude blacklist
.env: DISCORD_BOT_TOKEN, DISCORD_ALLOWED_IDS, DISCORD_PREFIX
"""
from __future__ import annotations
import os, threading
from datetime import datetime, timezone
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
PARAM_LABELS = {k: (k, "") for k in list(ALLOWED_RISK_KEYS) + list(ALLOWED_SETTINGS_KEYS)}
COLOR_OK, COLOR_WARN, COLOR_DANGER, COLOR_INFO, COLOR_IDLE = 0x2ECC71, 0xF1C40F, 0xE74C3C, 0x3498DB, 0x95A5A6

class BotControlState:
    def __init__(self):
        self.risk_gate = self.executor = self.exchange = self.scanner = None
        self.settings: Dict[str, Any] = {}
        self.risk_cfg: Dict[str, Any] = {}
        self.mode = "paper"
        self.paused = False
        self.running = True
        self.last_cycle_info = ""
    def set_refs(self, risk_gate, executor, exchange, settings, risk_cfg, mode: str):
        self.risk_gate, self.executor, self.exchange = risk_gate, executor, exchange
        self.settings, self.risk_cfg, self.mode = settings or {}, risk_cfg or {}, mode

control_state = BotControlState()

def _parse_allowed_ids() -> Set[int]:
    raw = os.getenv("DISCORD_ALLOWED_IDS", "").strip()
    return {int(p) for p in raw.replace(" ", "").split(",") if p.isdigit()}

def _user_authorized(user_id: int) -> bool:
    allowed = _parse_allowed_ids()
    return bool(allowed) and user_id in allowed

def _is_authorized(ctx) -> bool:
    return _user_authorized(ctx.author.id)

def _save_risk_yaml(risk_cfg: dict):
    path = Path(__file__).parent.parent / "config" / "risk.yaml"
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(risk_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception as e:
        logger.warning(f"Failed to save risk.yaml: {e}")

def _save_settings_yaml(settings: dict):
    path = Path(__file__).parent.parent / "config" / "settings.yaml"
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(settings, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
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

def _snapshot() -> Dict[str, Any]:
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
    return {
        "mode": st.mode, "paused": st.paused, "equity": equity,
        "daily_pnl": float(getattr(rg, "daily_pnl", 0) or 0) if rg else 0.0,
        "pos_n": pos_n, "max_pos": max_pos,
        "kill": bool(getattr(rg, "kill_switch", False)) if rg else False,
        "kill_reason": str(getattr(rg, "kill_reason", "") or "") if rg else "",
        "losses": int(getattr(rg, "consecutive_losses", 0) or 0) if rg else 0,
        "dlev": getattr(st.executor, "default_leverage", st.risk_cfg.get("default_leverage", "?")),
        "mlev": getattr(st.executor, "max_leverage", st.risk_cfg.get("max_leverage", "?")),
        "scan": st.settings.get("scan_interval_seconds", "?"),
        "top_n": st.settings.get("top_n", "?"),
        "excluded": len(_get_excluded_list()),
        "last_cycle": st.last_cycle_info or "—",
        "ready": rg is not None,
    }

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
        return f"✅ **{key}** = `{val}`"
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
        return f"✅ **{key}** = `{val}`"
    return f"❌ Key `{key}` không được phép."

def create_bot():
    if not HAS_DISCORD:
        return None
    prefix = os.getenv("DISCORD_PREFIX", "!").strip() or "!"
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix=prefix, intents=intents, help_command=None)

    @bot.event
    async def on_ready():
        logger.info(f"Discord control online as {bot.user}")

    @bot.command(name="help")
    async def cmd_help(ctx):
        if not _is_authorized(ctx):
            return
        await ctx.send(
            f"`{prefix}status` `{prefix}positions` `{prefix}params` `{prefix}set <k> <v>`\n"
            f"`{prefix}pause` `{prefix}resume` `{prefix}kill` `{prefix}unkill`\n"
            f"`{prefix}exclude list|add|remove|clear` — blacklist không quét"
        )

    @bot.command(name="status")
    async def cmd_status(ctx):
        if not _is_authorized(ctx):
            return
        s = _snapshot()
        await ctx.send(
            f"**{s['mode']}** equity={s['equity']:.2f} pos={s['pos_n']}/{s['max_pos']} "
            f"paused={s['paused']} kill={s['kill']} blacklist={s['excluded']}"
        )

    @bot.command(name="positions")
    async def cmd_positions(ctx):
        if not _is_authorized(ctx):
            return
        st = control_state
        if not st.exchange:
            await ctx.send("Exchange chưa sẵn sàng.")
            return
        try:
            positions = st.exchange.get_positions()
        except Exception as e:
            await ctx.send(f"Lỗi: {e}")
            return
        if not positions:
            await ctx.send("Không có vị thế.")
            return
        lines = []
        for p in positions[:15]:
            side = p.side.value if hasattr(p.side, "value") else str(p.side)
            lines.append(f"`{p.symbol}` {side} size={p.size} uPnL={getattr(p,'unrealized_pnl',0)}")
        await ctx.send("\n".join(lines))

    @bot.command(name="params")
    async def cmd_params(ctx):
        if not _is_authorized(ctx):
            return
        st = control_state
        cfg = (st.risk_gate.cfg if st.risk_gate else st.risk_cfg) or {}
        lines = [f"`{k}`={cfg.get(k)}" for k in sorted(ALLOWED_RISK_KEYS)]
        lines += [f"`{k}`={st.settings.get(k)}" for k in sorted(ALLOWED_SETTINGS_KEYS)]
        lines.append(f"blacklist={len(_get_excluded_list())}")
        await ctx.send("\n".join(lines))

    @bot.command(name="set")
    async def cmd_set(ctx, key: str = None, value: str = None):
        if not _is_authorized(ctx):
            return
        if not key or value is None:
            await ctx.send(f"Dùng: `{prefix}set <key> <value>`")
            return
        await ctx.send(_apply_param(key, value))

    @bot.command(name="pause")
    async def cmd_pause(ctx):
        if not _is_authorized(ctx):
            return
        control_state.paused = True
        await ctx.send("⏸ Paused")

    @bot.command(name="resume")
    async def cmd_resume(ctx):
        if not _is_authorized(ctx):
            return
        control_state.paused = False
        await ctx.send("▶️ Resumed")

    @bot.command(name="kill")
    async def cmd_kill(ctx):
        if not _is_authorized(ctx):
            return
        if control_state.risk_gate:
            control_state.risk_gate.activate_kill_switch("DISCORD_KILL")
        await ctx.send("🛑 Kill switch ON")

    @bot.command(name="unkill")
    async def cmd_unkill(ctx):
        if not _is_authorized(ctx):
            return
        if control_state.risk_gate:
            control_state.risk_gate.reset_kill_switch()
        await ctx.send("✅ Kill switch OFF")

    @bot.command(name="skip")
    async def cmd_skip(ctx, action: str = None):
        if not _is_authorized(ctx):
            return
        st = control_state
        if action == "clear" and st.executor:
            n = len(st.executor.skip_symbols)
            st.executor.skip_symbols.clear()
            await ctx.send(f"Cleared {n} runtime skips")
        else:
            n = len(getattr(st.executor, "skip_symbols", []) or [])
            await ctx.send(f"Runtime skip: {n}. `{prefix}exclude list` for blacklist")

    @bot.command(name="exclude", aliases=["blacklist", "ban"])
    async def cmd_exclude(ctx, action: str = None, *symbols: str):
        if not _is_authorized(ctx):
            return
        act = (action or "list").lower()
        current = list(_get_excluded_list())
        if act in ("list", "ls", "show"):
            await ctx.send(_format_excluded_msg())
            return
        if act == "clear":
            _persist_excluded([])
            await ctx.send("🧹 Blacklist cleared")
            return
        if act in ("add", "+") or (action and act not in ("list", "remove", "clear", "add", "rm", "-")):
            syms = list(symbols) if act in ("add", "+") else ([action] + list(symbols or []))
            if act in ("add", "+") and not symbols:
                await ctx.send(f"`{prefix}exclude add BTC DOGE`")
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
            msg = ", ".join(f"`{s}`" for s in added) if added else "already listed"
            await ctx.send(f"✅ Add: {msg} | total **{len(cleaned)}**")
            return
        if act in ("remove", "rm", "-"):
            if not symbols:
                await ctx.send(f"`{prefix}exclude remove BTC`")
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
            await ctx.send(f"🗑️ Remove: {msg} | left **{len(cleaned)}**")
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
