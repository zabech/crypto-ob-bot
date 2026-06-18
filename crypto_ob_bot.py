"""
Telegram Crypto Bot - Order Block Scanner
Konfirmasi: RSI + EMA + Fibonacci + BOS/CHoCH + Liquidity Sweep
"""

import asyncio
import logging
import os
from datetime import datetime
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

# Order Block
OB_MIN_BODY_RATIO = 0.6
OB_VOLUME_MULTIPLIER = 1.5

# RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# EMA
EMA_FAST = 20
EMA_SLOW = 50

# Fibonacci
FIB_LEVELS = [0.382, 0.5, 0.618]
FIB_TOLERANCE = 0.015

# BOS/CHoCH
STRUCTURE_LOOKBACK = 10   # Candle untuk cek swing high/low

# Liquidity Sweep
SWEEP_TOLERANCE = 0.005   # 0.5% toleransi sweep

# Minimum skor untuk kirim alert (maks 6)
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
def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


def calc_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].ewm(span=period, adjust=False).mean()


def calc_fibonacci(df: pd.DataFrame, lookback: int = 50) -> dict:
    recent = df.tail(lookback)
    swing_high = recent["high"].max()
    swing_low = recent["low"].min()
    diff = swing_high - swing_low
    levels = {f"fib_{int(f*1000)}": swing_high - diff * f for f in FIB_LEVELS}
    levels["swing_high"] = swing_high
    levels["swing_low"] = swing_low
    return levels


def is_near_fib(price: float, fib_levels: dict) -> tuple:
    for key, level in fib_levels.items():
        if key.startswith("fib_"):
            if abs(price - level) / level <= FIB_TOLERANCE:
                fib_name = key.replace("fib_", "")
                val = int(fib_name) / 1000
                return True, val, level
    return False, None, None


# ─── BOS / CHoCH ─────────────────────────────────────────────────────────────
def detect_bos_choch(df: pd.DataFrame, ob_index: int, ob_type: str) -> tuple:
    """
    Deteksi Break of Structure (BOS) atau Change of Character (CHoCH).
    
    BOS Bullish: candle setelah OB menembus swing high sebelumnya
    CHoCH: perubahan struktur dari bearish ke bullish (atau sebaliknya)
    
    Return: (terdeteksi, jenis, deskripsi)
    """
    lookback = STRUCTURE_LOOKBACK
    start = max(0, ob_index - lookback)
    pre_ob = df.iloc[start:ob_index]
    post_ob = df.iloc[ob_index + 1: ob_index + 6]

    if len(pre_ob) < 3 or len(post_ob) < 1:
        return False, None, None

    if ob_type == "bullish":
        # Cek apakah sebelum OB ada downtrend (lower highs)
        prev_highs = pre_ob["high"].values
        is_downtrend = all(prev_highs[i] >= prev_highs[i+1] for i in range(len(prev_highs)-1)) if len(prev_highs) > 1 else False

        # CHoCH: candle setelah OB menembus swing high terakhir sebelum OB
        last_swing_high = pre_ob["high"].max()
        post_high = post_ob["high"].max()

        if post_high > last_swing_high:
            if is_downtrend:
                return True, "CHoCH", f"CHoCH Bullish (break ${last_swing_high:,.4f})"
            else:
                return True, "BOS", f"BOS Bullish (break ${last_swing_high:,.4f})"

    elif ob_type == "bearish":
        prev_lows = pre_ob["low"].values
        is_uptrend = all(prev_lows[i] <= prev_lows[i+1] for i in range(len(prev_lows)-1)) if len(prev_lows) > 1 else False

        last_swing_low = pre_ob["low"].min()
        post_low = post_ob["low"].min()

        if post_low < last_swing_low:
            if is_uptrend:
                return True, "CHoCH", f"CHoCH Bearish (break ${last_swing_low:,.4f})"
            else:
                return True, "BOS", f"BOS Bearish (break ${last_swing_low:,.4f})"

    return False, None, None


# ─── LIQUIDITY SWEEP ──────────────────────────────────────────────────────────
def detect_liquidity_sweep(df: pd.DataFrame, ob_index: int, ob_type: str) -> tuple:
    """
    Deteksi Liquidity Sweep sebelum harga masuk ke zona OB.
    
    Bullish: harga sweep ke bawah (ambil sell stop) lalu berbalik naik
    Bearish: harga sweep ke atas (ambil buy stop) lalu berbalik turun
    
    Return: (terdeteksi, deskripsi, level_sweep)
    """
    lookback = 15
    start = max(0, ob_index - lookback)
    window = df.iloc[start:ob_index + 3]

    if len(window) < 5:
        return False, None, None

    if ob_type == "bullish":
        # Cari swing low sebelumnya
        recent_lows = window["low"].rolling(3).min()
        prev_swing_low = recent_lows.iloc[:-3].min()

        # Cek apakah ada candle yang sweep di bawah swing low lalu close di atas
        for i in range(len(window) - 2, max(0, len(window) - 6), -1):
            row = window.iloc[i]
            next_row = window.iloc[i + 1] if i + 1 < len(window) else None

            swept = row["low"] < prev_swing_low * (1 - SWEEP_TOLERANCE)
            recovered = row["close"] > prev_swing_low if next_row is None else next_row["close"] > prev_swing_low

            if swept and recovered:
                return True, f"Sweep Low ${prev_swing_low:,.4f} → Reversal", prev_swing_low

    elif ob_type == "bearish":
        recent_highs = window["high"].rolling(3).max()
        prev_swing_high = recent_highs.iloc[:-3].max()

        for i in range(len(window) - 2, max(0, len(window) - 6), -1):
            row = window.iloc[i]
            next_row = window.iloc[i + 1] if i + 1 < len(window) else None

            swept = row["high"] > prev_swing_high * (1 + SWEEP_TOLERANCE)
            recovered = row["close"] < prev_swing_high if next_row is None else next_row["close"] < prev_swing_high

            if swept and recovered:
                return True, f"Sweep High ${prev_swing_high:,.4f} → Reversal", prev_swing_high

    return False, None, None


# ─── ORDER BLOCK ──────────────────────────────────────────────────────────────
def is_strong_candle(row: pd.Series, avg_volume: float) -> bool:
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return False
    body = abs(row["close"] - row["open"])
    return (body / candle_range) >= OB_MIN_BODY_RATIO and row["volume"] >= avg_volume * OB_VOLUME_MULTIPLIER


def detect_signals(df: pd.DataFrame) -> list:
    signals = []
    if len(df) < 80:
        return signals

    df = df.copy()
    df["rsi"] = calc_rsi(df, RSI_PERIOD)
    df["ema_fast"] = calc_ema(df, EMA_FAST)
    df["ema_slow"] = calc_ema(df, EMA_SLOW)
    avg_volume = df["volume"].rolling(20).mean()
    fib_levels = calc_fibonacci(df)
    current_price = df["close"].iloc[-1]
    current_rsi = df["rsi"].iloc[-1]
    current_ema_fast = df["ema_fast"].iloc[-1]
    current_ema_slow = df["ema_slow"].iloc[-1]

    for i in range(5, len(df) - 3):
        row = df.iloc[i]
        avg_vol = avg_volume.iloc[i]
        if not is_strong_candle(row, avg_vol):
            continue

        next_candles = df.iloc[i + 1: i + 4]

        # ── BULLISH OB ──────────────────────────────────────────────────────
        if row["close"] < row["open"]:
            bullish_follow = sum(1 for _, c in next_candles.iterrows() if c["close"] > c["open"])
            impulse_move = (next_candles["high"].max() - row["low"]) / row["low"] * 100
            if bullish_follow >= 2 and impulse_move >= 1.0:
                ob_high, ob_low = row["open"], row["close"]
                proximity = (current_price - ob_high) / ob_high * 100
                if -2.0 <= proximity <= 2.0:
                    score = 1
                    confirmations = ["✅ Order Block Bullish"]
                    not_confirmed = []

                    # RSI
                    if current_rsi <= RSI_OVERSOLD:
                        score += 1
                        confirmations.append(f"✅ RSI {current_rsi:.1f} (oversold)")
                    else:
                        not_confirmed.append(f"❌ RSI {current_rsi:.1f} (belum oversold)")

                    # EMA
                    if current_ema_fast > current_ema_slow:
                        score += 1
                        confirmations.append(f"✅ EMA{EMA_FAST} > EMA{EMA_SLOW} (uptrend)")
                    else:
                        not_confirmed.append(f"❌ EMA downtrend")

                    # Fibonacci
                    near_fib, fib_val, fib_price = is_near_fib(current_price, fib_levels)
                    if near_fib:
                        score += 1
                        confirmations.append(f"✅ Fib {fib_val:.3f} (${fib_price:,.4f})")
                    else:
                        not_confirmed.append(f"❌ Tidak di zona Fib")

                    # BOS / CHoCH
                    bos_found, bos_type, bos_desc = detect_bos_choch(df, i, "bullish")
                    if bos_found:
                        score += 1
                        confirmations.append(f"✅ {bos_desc}")
                    else:
                        not_confirmed.append(f"❌ Belum ada BOS/CHoCH")

                    # Liquidity Sweep
                    sweep_found, sweep_desc, sweep_level = detect_liquidity_sweep(df, i, "bullish")
                    if sweep_found:
                        score += 1
                        confirmations.append(f"✅ {sweep_desc}")
                    else:
                        not_confirmed.append(f"❌ Belum ada Liquidity Sweep")

                    if score >= MIN_SCORE:
                        signals.append({
                            "type": "bullish",
                            "ob_high": ob_high, "ob_low": ob_low,
                            "current_price": current_price,
                            "proximity_pct": round(proximity, 2),
                            "impulse_move_pct": round(impulse_move, 2),
                            "candle_time": row["timestamp"],
                            "score": score, "max_score": 6,
                            "confirmations": confirmations,
                            "not_confirmed": not_confirmed,
                            "rsi": round(current_rsi, 1),
                        })

        # ── BEARISH OB ──────────────────────────────────────────────────────
        elif row["close"] > row["open"]:
            bearish_follow = sum(1 for _, c in next_candles.iterrows() if c["close"] < c["open"])
            impulse_move = (row["high"] - next_candles["low"].min()) / row["high"] * 100
            if bearish_follow >= 2 and impulse_move >= 1.0:
                ob_high, ob_low = row["close"], row["open"]
                proximity = (current_price - ob_low) / ob_low * 100
                if -2.0 <= proximity <= 2.0:
                    score = 1
                    confirmations = ["✅ Order Block Bearish"]
                    not_confirmed = []

                    if current_rsi >= RSI_OVERBOUGHT:
                        score += 1
                        confirmations.append(f"✅ RSI {current_rsi:.1f} (overbought)")
                    else:
                        not_confirmed.append(f"❌ RSI {current_rsi:.1f} (belum overbought)")

                    if current_ema_fast < current_ema_slow:
                        score += 1
                        confirmations.append(f"✅ EMA{EMA_FAST} < EMA{EMA_SLOW} (downtrend)")
                    else:
                        not_confirmed.append(f"❌ EMA uptrend")

                    near_fib, fib_val, fib_price = is_near_fib(current_price, fib_levels)
                    if near_fib:
                        score += 1
                        confirmations.append(f"✅ Fib {fib_val:.3f} (${fib_price:,.4f})")
                    else:
                        not_confirmed.append(f"❌ Tidak di zona Fib")

                    bos_found, bos_type, bos_desc = detect_bos_choch(df, i, "bearish")
                    if bos_found:
                        score += 1
                        confirmations.append(f"✅ {bos_desc}")
                    else:
                        not_confirmed.append(f"❌ Belum ada BOS/CHoCH")

                    sweep_found, sweep_desc, sweep_level = detect_liquidity_sweep(df, i, "bearish")
                    if sweep_found:
                        score += 1
                        confirmations.append(f"✅ {sweep_desc}")
                    else:
                        not_confirmed.append(f"❌ Belum ada Liquidity Sweep")

                    if score >= MIN_SCORE:
                        signals.append({
                            "type": "bearish",
                            "ob_high": ob_high, "ob_low": ob_low,
                            "current_price": current_price,
                            "proximity_pct": round(proximity, 2),
                            "impulse_move_pct": round(impulse_move, 2),
                            "candle_time": row["timestamp"],
                            "score": score, "max_score": 6,
                            "confirmations": confirmations,
                            "not_confirmed": not_confirmed,
                            "rsi": round(current_rsi, 1),
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
    stars = "⭐" * score

    # Bar kekuatan sinyal
    filled = round(score / max_score * 10)
    bar = "█" * filled + "░" * (10 - filled)

    confirm_text = "\n".join(signal["confirmations"])
    not_confirm_text = "\n".join(signal["not_confirmed"]) if signal["not_confirmed"] else "—"

    return (
        f"{emoji} *OB {direction} — Skor {score}/{max_score}*\n"
        f"`[{bar}]` {stars}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 *Koin:* `{coin}/USDT`  ⏱ `{timeframe.upper()}`\n"
        f"📍 *Zona OB:* `${signal['ob_low']:,.4f}` — `${signal['ob_high']:,.4f}`\n"
        f"💰 *Harga:* `${signal['current_price']:,.4f}` ({prox_str})\n"
        f"⚡ *Impulse:* `{signal['impulse_move_pct']}%`  📊 *RSI:* `{signal['rsi']}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Konfirmasi:*\n{confirm_text}\n"
        f"*Belum:*\n{not_confirm_text}\n"
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
            signals = detect_signals(df)
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
            text=f"🔍 Scan selesai `{datetime.now().strftime('%H:%M %d/%m/%Y')}`\nTidak ada sinyal skor ≥{MIN_SCORE}/6.",
            parse_mode="Markdown"
        )


# ─── COMMANDS ─────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Bot SMC Scanner aktif!*\n\n"
        "Indikator: OB + RSI + EMA + Fib + BOS/CHoCH + Liquidity Sweep\n\n"
        "/scan — Scan manual\n/status — Status bot\n/help — Panduan",
        parse_mode="Markdown"
    )

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Scan dimulai, harap tunggu...")
    await scan_all(context.bot)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = get_usdt_symbols()
    await update.message.reply_text(
        f"✅ *Bot aktif*\n"
        f"📊 Simbol: `{len(symbols)}`\n"
        f"⏱ TF: `{', '.join(TIMEFRAMES)}`\n"
        f"🔄 Interval: `{SCAN_INTERVAL_MINUTES} menit`\n"
        f"🎯 Min skor: `{MIN_SCORE}/6`",
        parse_mode="Markdown"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Sistem Skor (maks 6)*\n\n"
        "1️⃣ *Order Block* — zona institusi\n"
        "2️⃣ *RSI* — oversold/overbought\n"
        "3️⃣ *EMA 20/50* — konfirmasi trend\n"
        "4️⃣ *Fibonacci* — level 38.2/50/61.8%\n"
        "5️⃣ *BOS/CHoCH* — konfirmasi struktur market\n"
        "6️⃣ *Liquidity Sweep* — smart money ambil likuiditas\n\n"
        f"*Alert dikirim jika skor ≥ {MIN_SCORE}/6*\n\n"
        "Semakin tinggi skor = sinyal makin kuat\n\n"
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
