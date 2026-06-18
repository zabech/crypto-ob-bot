"""
Telegram Crypto Bot - SMC Scanner Pro
Indikator: OB + EMA + BOS/CHoCH + Liquidity Sweep + VWAP + FVG + ATR + MTF + Session Filter
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import ccxt
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ─── KONFIGURASI ─────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

TIMEFRAMES = ["4h", "1d"]
SCAN_INTERVAL_MINUTES = 30
MAX_SYMBOLS = 100

# OB
OB_MIN_BODY_RATIO = 0.6
OB_VOLUME_MULTIPLIER = 1.5

# EMA
EMA_FAST = 20
EMA_SLOW = 50

# BOS/CHoCH
STRUCTURE_LOOKBACK = 10

# Liquidity Sweep
SWEEP_TOLERANCE = 0.005

# FVG
FVG_MIN_GAP_PCT = 0.003   # Gap minimal 0.3%

# ATR
ATR_PERIOD = 14
ATR_MIN_MULTIPLIER = 0.5  # Volatilitas minimal = 0.5x ATR rata-rata

# Session Filter (UTC)
LONDON_START = 7    # 07:00 UTC
LONDON_END = 12     # 12:00 UTC
NY_START = 13       # 13:00 UTC
NY_END = 18         # 18:00 UTC

# MTF: timeframe lebih tinggi untuk konfirmasi trend
MTF_MAP = {"4h": "1d", "1d": "1w"}

# Minimum skor (maks 7)
MIN_SCORE = 3

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


# ─── FETCH DATA ───────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        logger.warning(f"Gagal fetch {symbol} {timeframe}: {e}")
        return None


def get_usdt_symbols() -> list:
    try:
        markets = exchange.load_markets()
        symbols = [s for s, m in markets.items() if s.endswith("/USDT") and m.get("active") and m.get("spot")]
        return symbols[:MAX_SYMBOLS] if MAX_SYMBOLS else symbols
    except Exception as e:
        logger.error(f"Gagal load markets: {e}")
        return []


# ─── INDIKATOR ────────────────────────────────────────────────────────────────
def calc_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].ewm(span=period, adjust=False).mean()


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_fvg(df: pd.DataFrame, ob_index: int, ob_type: str) -> tuple:
    """
    Fair Value Gap: gap antara candle 1 dan candle 3 yang tidak terisi candle 2.
    Bullish FVG: low candle 3 > high candle 1
    Bearish FVG: high candle 3 < low candle 1
    """
    fvgs = []
    search_start = max(1, ob_index - 20)
    search_end = min(len(df) - 1, ob_index + 5)

    for i in range(search_start, search_end - 1):
        c1 = df.iloc[i - 1]
        c3 = df.iloc[i + 1]

        if ob_type == "bullish":
            gap = c3["low"] - c1["high"]
            if gap > 0 and gap / c1["high"] >= FVG_MIN_GAP_PCT:
                fvgs.append({"top": c3["low"], "bottom": c1["high"], "gap_pct": round(gap / c1["high"] * 100, 2)})

        elif ob_type == "bearish":
            gap = c1["low"] - c3["high"]
            if gap > 0 and gap / c3["high"] >= FVG_MIN_GAP_PCT:
                fvgs.append({"top": c1["low"], "bottom": c3["high"], "gap_pct": round(gap / c3["high"] * 100, 2)})

    if not fvgs:
        return False, None

    # Ambil FVG terdekat dengan harga saat ini
    current_price = df["close"].iloc[-1]
    closest = min(fvgs, key=lambda x: abs(current_price - (x["top"] + x["bottom"]) / 2))

    # Cek apakah harga berada di dalam atau dekat FVG
    in_fvg = closest["bottom"] <= current_price <= closest["top"]
    near_fvg = abs(current_price - (closest["top"] + closest["bottom"]) / 2) / current_price <= 0.02

    if in_fvg or near_fvg:
        return True, f"FVG {closest['gap_pct']}% (${closest['bottom']:,.4f}–${closest['top']:,.4f})"

    return False, None


def detect_bos_choch(df: pd.DataFrame, ob_index: int, ob_type: str) -> tuple:
    lookback = STRUCTURE_LOOKBACK
    start = max(0, ob_index - lookback)
    pre_ob = df.iloc[start:ob_index]
    post_ob = df.iloc[ob_index + 1: ob_index + 6]

    if len(pre_ob) < 3 or len(post_ob) < 1:
        return False, None, None

    if ob_type == "bullish":
        prev_highs = pre_ob["high"].values
        is_downtrend = all(prev_highs[i] >= prev_highs[i+1] for i in range(len(prev_highs)-1)) if len(prev_highs) > 1 else False
        last_swing_high = pre_ob["high"].max()
        if post_ob["high"].max() > last_swing_high:
            label = "CHoCH" if is_downtrend else "BOS"
            return True, label, f"{label} Bullish (break ${last_swing_high:,.4f})"

    elif ob_type == "bearish":
        prev_lows = pre_ob["low"].values
        is_uptrend = all(prev_lows[i] <= prev_lows[i+1] for i in range(len(prev_lows)-1)) if len(prev_lows) > 1 else False
        last_swing_low = pre_ob["low"].min()
        if post_ob["low"].min() < last_swing_low:
            label = "CHoCH" if is_uptrend else "BOS"
            return True, label, f"{label} Bearish (break ${last_swing_low:,.4f})"

    return False, None, None


def detect_liquidity_sweep(df: pd.DataFrame, ob_index: int, ob_type: str) -> tuple:
    lookback = 15
    start = max(0, ob_index - lookback)
    window = df.iloc[start:ob_index + 3]
    if len(window) < 5:
        return False, None

    if ob_type == "bullish":
        prev_swing_low = window["low"].rolling(3).min().iloc[:-3].min()
        for i in range(len(window) - 2, max(0, len(window) - 6), -1):
            row = window.iloc[i]
            swept = row["low"] < prev_swing_low * (1 - SWEEP_TOLERANCE)
            recovered = row["close"] > prev_swing_low
            if swept and recovered:
                return True, f"Sweep Low ${prev_swing_low:,.4f} → Reversal"

    elif ob_type == "bearish":
        prev_swing_high = window["high"].rolling(3).max().iloc[:-3].max()
        for i in range(len(window) - 2, max(0, len(window) - 6), -1):
            row = window.iloc[i]
            swept = row["high"] > prev_swing_high * (1 + SWEEP_TOLERANCE)
            recovered = row["close"] < prev_swing_high
            if swept and recovered:
                return True, f"Sweep High ${prev_swing_high:,.4f} → Reversal"

    return False, None


def check_mtf_trend(symbol: str, current_tf: str, ob_type: str) -> tuple:
    """Cek trend di timeframe lebih tinggi."""
    higher_tf = MTF_MAP.get(current_tf)
    if not higher_tf:
        return False, None

    df_htf = fetch_ohlcv(symbol, higher_tf, limit=60)
    if df_htf is None or len(df_htf) < 55:
        return False, None

    ema_fast = calc_ema(df_htf, EMA_FAST).iloc[-1]
    ema_slow = calc_ema(df_htf, EMA_SLOW).iloc[-1]

    if ob_type == "bullish" and ema_fast > ema_slow:
        return True, f"HTF {higher_tf.upper()} uptrend (EMA{EMA_FAST}>{EMA_SLOW})"
    elif ob_type == "bearish" and ema_fast < ema_slow:
        return True, f"HTF {higher_tf.upper()} downtrend (EMA{EMA_FAST}<{EMA_SLOW})"

    return False, f"HTF {higher_tf.upper()} berlawanan arah"


def check_session() -> tuple:
    """Cek apakah saat ini dalam sesi London atau New York."""
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    if LONDON_START <= hour < LONDON_END:
        return True, f"Sesi London ({hour:02d}:00 UTC)"
    elif NY_START <= hour < NY_END:
        return True, f"Sesi New York ({hour:02d}:00 UTC)"
    return False, f"Di luar sesi utama ({hour:02d}:00 UTC)"


def check_atr_volatility(df: pd.DataFrame) -> tuple:
    """Cek apakah volatilitas cukup untuk entry."""
    atr = calc_atr(df, ATR_PERIOD)
    current_atr = atr.iloc[-1]
    avg_atr = atr.iloc[-20:].mean()
    ratio = current_atr / avg_atr if avg_atr > 0 else 0
    if ratio >= ATR_MIN_MULTIPLIER:
        return True, f"ATR volatilitas cukup ({ratio:.1f}x rata-rata)"
    return False, f"ATR rendah ({ratio:.1f}x rata-rata)"


def is_strong_candle(row: pd.Series, avg_volume: float) -> bool:
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return False
    body = abs(row["close"] - row["open"])
    return (body / candle_range) >= OB_MIN_BODY_RATIO and row["volume"] >= avg_volume * OB_VOLUME_MULTIPLIER


# ─── DETEKSI SINYAL ───────────────────────────────────────────────────────────
def detect_signals(df: pd.DataFrame, symbol: str, timeframe: str) -> list:
    signals = []
    if len(df) < 80:
        return signals

    df = df.copy()
    df["ema_fast"] = calc_ema(df, EMA_FAST)
    df["ema_slow"] = calc_ema(df, EMA_SLOW)
    df["vwap"] = calc_vwap(df)
    avg_volume = df["volume"].rolling(20).mean()

    current_price = df["close"].iloc[-1]
    current_ema_fast = df["ema_fast"].iloc[-1]
    current_ema_slow = df["ema_slow"].iloc[-1]
    current_vwap = df["vwap"].iloc[-1]

    # Cek session & ATR sekali saja per scan
    session_ok, session_desc = check_session()
    atr_ok, atr_desc = check_atr_volatility(df)

    for i in range(5, len(df) - 3):
        row = df.iloc[i]
        avg_vol = avg_volume.iloc[i]
        if not is_strong_candle(row, avg_vol):
            continue

        next_candles = df.iloc[i + 1: i + 4]

        for ob_type in ["bullish", "bearish"]:
            if ob_type == "bullish":
                if row["close"] >= row["open"]:
                    continue
                bullish_follow = sum(1 for _, c in next_candles.iterrows() if c["close"] > c["open"])
                impulse_move = (next_candles["high"].max() - row["low"]) / row["low"] * 100
                if bullish_follow < 2 or impulse_move < 1.0:
                    continue
                ob_high, ob_low = row["open"], row["close"]
                proximity = (current_price - ob_high) / ob_high * 100

            else:
                if row["close"] <= row["open"]:
                    continue
                bearish_follow = sum(1 for _, c in next_candles.iterrows() if c["close"] < c["open"])
                impulse_move = (row["high"] - next_candles["low"].min()) / row["high"] * 100
                if bearish_follow < 2 or impulse_move < 1.0:
                    continue
                ob_high, ob_low = row["close"], row["open"]
                proximity = (current_price - ob_low) / ob_low * 100

            if not (-2.0 <= proximity <= 2.0):
                continue

            score = 1
            confirmations = [f"✅ Order Block {ob_type.capitalize()}"]
            not_confirmed = []

            # EMA
            if (ob_type == "bullish" and current_ema_fast > current_ema_slow) or \
               (ob_type == "bearish" and current_ema_fast < current_ema_slow):
                score += 1
                trend = "uptrend" if ob_type == "bullish" else "downtrend"
                confirmations.append(f"✅ EMA{EMA_FAST}/{EMA_SLOW} {trend}")
            else:
                not_confirmed.append(f"❌ EMA berlawanan trend")

            # BOS/CHoCH
            bos_found, _, bos_desc = detect_bos_choch(df, i, ob_type)
            if bos_found:
                score += 1
                confirmations.append(f"✅ {bos_desc}")
            else:
                not_confirmed.append(f"❌ Belum ada BOS/CHoCH")

            # Liquidity Sweep
            sweep_found, sweep_desc = detect_liquidity_sweep(df, i, ob_type)
            if sweep_found:
                score += 1
                confirmations.append(f"✅ {sweep_desc}")
            else:
                not_confirmed.append(f"❌ Belum ada Liquidity Sweep")

            # VWAP
            if (ob_type == "bullish" and current_price < current_vwap) or \
               (ob_type == "bearish" and current_price > current_vwap):
                score += 1
                side = "di bawah" if ob_type == "bullish" else "di atas"
                confirmations.append(f"✅ VWAP: harga {side} (${current_vwap:,.4f})")
            else:
                not_confirmed.append(f"❌ VWAP tidak konfirmasi")

            # FVG
            fvg_found, fvg_desc = detect_fvg(df, i, ob_type)
            if fvg_found:
                score += 1
                confirmations.append(f"✅ {fvg_desc}")
            else:
                not_confirmed.append(f"❌ Tidak ada FVG")

            # ATR
            if atr_ok:
                score += 1
                confirmations.append(f"✅ {atr_desc}")
            else:
                not_confirmed.append(f"❌ {atr_desc}")

            # MTF
            mtf_ok, mtf_desc = check_mtf_trend(symbol, timeframe, ob_type)
            if mtf_ok:
                score += 1
                confirmations.append(f"✅ {mtf_desc}")
            else:
                not_confirmed.append(f"❌ {mtf_desc or 'MTF tidak tersedia'}")

            # Session
            if session_ok:
                score += 1
                confirmations.append(f"✅ {session_desc}")
            else:
                not_confirmed.append(f"❌ {session_desc}")

            if score >= MIN_SCORE:
                signals.append({
                    "type": ob_type,
                    "ob_high": ob_high, "ob_low": ob_low,
                    "current_price": current_price,
                    "proximity_pct": round(proximity, 2),
                    "impulse_move_pct": round(impulse_move, 2),
                    "candle_time": row["timestamp"],
                    "score": score, "max_score": 9,
                    "confirmations": confirmations,
                    "not_confirmed": not_confirmed,
                    "vwap": round(current_vwap, 4),
                })

    return signals


# ─── FORMAT ALERT ─────────────────────────────────────────────────────────────
def format_alert(symbol: str, timeframe: str, signal: dict) -> str:
    ob_type = signal["type"]
    emoji = "🟢" if ob_type == "bullish" else "🔴"
    direction = "BULLISH" if ob_type == "bullish" else "BEARISH"
    action = "Support / Potensi Naik ↑" if ob_type == "bullish" else "Resistance / Potensi Turun ↓"
    coin = symbol.replace("/USDT", "")
    prox = signal["proximity_pct"]
    prox_str = f"+{prox:.2f}%" if prox >= 0 else f"{prox:.2f}%"
    score = signal["score"]
    max_score = signal["max_score"]

    filled = round(score / max_score * 10)
    bar = "█" * filled + "░" * (10 - filled)
    stars = "⭐" * min(score, 9)

    confirm_text = "\n".join(signal["confirmations"])
    not_text = "\n".join(signal["not_confirmed"]) if signal["not_confirmed"] else "—"

    return (
        f"{emoji} *OB {direction} — {score}/{max_score}*\n"
        f"`[{bar}]` {stars}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 `{coin}/USDT`  ⏱ `{timeframe.upper()}`\n"
        f"📍 OB: `${signal['ob_low']:,.4f}` — `${signal['ob_high']:,.4f}`\n"
        f"💰 Harga: `${signal['current_price']:,.4f}` ({prox_str})\n"
        f"📊 VWAP: `${signal['vwap']:,.4f}`\n"
        f"⚡ Impulse: `{signal['impulse_move_pct']}%`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Konfirmasi:*\n{confirm_text}\n\n"
        f"*Belum:*\n{not_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 {action}\n"
        f"⚠️ _Bukan financial advice._"
    )


# ─── SCANNER ──────────────────────────────────────────────────────────────────
async def scan_all(bot: Bot):
    logger.info("Memulai scan...")
    symbols = get_usdt_symbols()
    alert_count = 0

    for symbol in symbols:
        for tf in TIMEFRAMES:
            df = fetch_ohlcv(symbol, tf, limit=200)
            if df is None or len(df) < 80:
                continue
            signals = detect_signals(df, symbol, tf)
            for signal in signals:
                msg = format_alert(symbol, tf, signal)
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
                alert_count += 1
                await asyncio.sleep(0.5)
            await asyncio.sleep(0.3)

    logger.info(f"Scan selesai. Alert: {alert_count}")
    if alert_count == 0:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"🔍 Scan selesai `{datetime.now().strftime('%H:%M %d/%m/%Y')}`\nTidak ada sinyal skor ≥{MIN_SCORE}/9.",
            parse_mode="Markdown"
        )


# ─── COMMANDS ─────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *SMC Scanner Pro aktif!*\n\n"
        "9 indikator: OB + EMA + BOS/CHoCH + Sweep + VWAP + FVG + ATR + MTF + Session\n\n"
        "/scan — Scan manual\n/status — Status\n/help — Panduan",
        parse_mode="Markdown"
    )

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Scan dimulai, harap tunggu...")
    await scan_all(context.bot)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session_ok, session_desc = check_session()
    symbols = get_usdt_symbols()
    await update.message.reply_text(
        f"✅ *Bot aktif*\n"
        f"📊 Simbol: `{len(symbols)}`\n"
        f"⏱ TF: `{', '.join(TIMEFRAMES)}`\n"
        f"🔄 Interval: `{SCAN_INTERVAL_MINUTES} menit`\n"
        f"🎯 Min skor: `{MIN_SCORE}/9`\n"
        f"🕐 Sesi: `{'🟢 ' + session_desc if session_ok else '🔴 ' + session_desc}`",
        parse_mode="Markdown"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Sistem Skor (maks 9)*\n\n"
        "1️⃣ Order Block — zona institusi\n"
        "2️⃣ EMA 20/50 — konfirmasi trend\n"
        "3️⃣ BOS/CHoCH — struktur market\n"
        "4️⃣ Liquidity Sweep — smart money\n"
        "5️⃣ VWAP — posisi vs rata-rata volume\n"
        "6️⃣ FVG — Fair Value Gap belum terisi\n"
        "7️⃣ ATR — volatilitas cukup\n"
        "8️⃣ MTF — konfirmasi timeframe lebih tinggi\n"
        "9️⃣ Session — sesi London/New York\n\n"
        f"*Alert dikirim jika skor ≥ {MIN_SCORE}/9*\n\n"
        "⚠️ _Bukan financial advice._",
        parse_mode="Markdown"
    )

async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    await scan_all(context.bot)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Set TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID di environment variables!")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help", cmd_help))
    app.job_queue.run_repeating(auto_scan_job, interval=SCAN_INTERVAL_MINUTES * 60, first=10)
    logger.info("Bot dimulai...")
    app.run_polling()

if __name__ == "__main__":
    main()
