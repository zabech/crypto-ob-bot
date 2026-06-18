"""
Telegram Crypto Bot - Order Block Scanner + RSI + EMA + Fibonacci
Konfirmasi multi-indikator untuk meningkatkan akurasi sinyal
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
RSI_OVERSOLD = 35       # Bullish OB: RSI di bawah ini
RSI_OVERBOUGHT = 65     # Bearish OB: RSI di atas ini

# EMA
EMA_FAST = 20
EMA_SLOW = 50

# Fibonacci levels (dari swing high/low 50 candle terakhir)
FIB_LEVELS = [0.382, 0.5, 0.618]
FIB_TOLERANCE = 0.015   # ±1.5% dari level fib

# Minimum skor konfirmasi untuk kirim alert (maks 4)
MIN_SCORE = 2

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


# ─── FETCH DATA ───────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 150) -> Optional[pd.DataFrame]:
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
                fib_name = key.replace("fib_", "0.")
                return True, float(fib_name) / 10, level
    return False, None, None


# ─── ORDER BLOCK ──────────────────────────────────────────────────────────────
def is_strong_candle(row: pd.Series, avg_volume: float) -> bool:
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return False
    body = abs(row["close"] - row["open"])
    return (body / candle_range) >= OB_MIN_BODY_RATIO and row["volume"] >= avg_volume * OB_VOLUME_MULTIPLIER


def detect_signals(df: pd.DataFrame) -> list:
    """Deteksi sinyal OB + konfirmasi multi-indikator."""
    signals = []
    if len(df) < 60:
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

    for i in range(2, len(df) - 3):
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
                    # Hitung skor konfirmasi
                    score = 1  # OB sendiri = 1
                    confirmations = []
                    reasons = []

                    # RSI oversold
                    if current_rsi <= RSI_OVERSOLD:
                        score += 1
                        confirmations.append(f"RSI {current_rsi:.1f} (oversold)")
                    else:
                        reasons.append(f"RSI {current_rsi:.1f}")

                    # EMA trend bullish
                    if current_ema_fast > current_ema_slow:
                        score += 1
                        confirmations.append(f"EMA{EMA_FAST} > EMA{EMA_SLOW} (uptrend)")
                    else:
                        reasons.append(f"EMA downtrend")

                    # Fibonacci
                    near_fib, fib_val, fib_price = is_near_fib(current_price, fib_levels)
                    if near_fib:
                        score += 1
                        confirmations.append(f"Fib {fib_val:.3f} (${fib_price:,.4f})")
                    else:
                        reasons.append("Tidak di zona Fib")

                    if score >= MIN_SCORE:
                        signals.append({
                            "type": "bullish",
                            "ob_high": ob_high,
                            "ob_low": ob_low,
                            "current_price": current_price,
                            "proximity_pct": round(proximity, 2),
                            "impulse_move_pct": round(impulse_move, 2),
                            "candle_time": row["timestamp"],
                            "score": score,
                            "confirmations": confirmations,
                            "reasons": reasons,
                            "rsi": round(current_rsi, 1),
                            "ema_fast": round(current_ema_fast, 4),
                            "ema_slow": round(current_ema_slow, 4),
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
                    confirmations = []
                    reasons = []

                    # RSI overbought
                    if current_rsi >= RSI_OVERBOUGHT:
                        score += 1
                        confirmations.append(f"RSI {current_rsi:.1f} (overbought)")
                    else:
                        reasons.append(f"RSI {current_rsi:.1f}")

                    # EMA trend bearish
                    if current_ema_fast < current_ema_slow:
                        score += 1
                        confirmations.append(f"EMA{EMA_FAST} < EMA{EMA_SLOW} (downtrend)")
                    else:
                        reasons.append(f"EMA uptrend")

                    # Fibonacci
                    near_fib, fib_val, fib_price = is_near_fib(current_price, fib_levels)
                    if near_fib:
                        score += 1
                        confirmations.append(f"Fib {fib_val:.3f} (${fib_price:,.4f})")
                    else:
                        reasons.append("Tidak di zona Fib")

                    if score >= MIN_SCORE:
                        signals.append({
                            "type": "bearish",
                            "ob_high": ob_high,
                            "ob_low": ob_low,
                            "current_price": current_price,
                            "proximity_pct": round(proximity, 2),
                            "impulse_move_pct": round(impulse_move, 2),
                            "candle_time": row["timestamp"],
                            "score": score,
                            "confirmations": confirmations,
                            "reasons": reasons,
                            "rsi": round(current_rsi, 1),
                            "ema_fast": round(current_ema_fast, 4),
                            "ema_slow": round(current_ema_slow, 4),
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

    # Bintang kekuatan sinyal
    stars = "⭐" * signal["score"]

    # Konfirmasi
    confirm_text = "\n".join([f"  ✅ {c}" for c in signal["confirmations"]]) if signal["confirmations"] else "  -"
    reason_text = "\n".join([f"  ❌ {r}" for r in signal["reasons"]]) if signal["reasons"] else "  -"

    return (
        f"{emoji} *Order Block {direction}*  {stars}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 *Koin:* `{coin}/USDT`\n"
        f"⏱ *Timeframe:* `{timeframe.upper()}`\n"
        f"📍 *Zona OB:* `${signal['ob_low']:,.4f}` — `${signal['ob_high']:,.4f}`\n"
        f"💰 *Harga:* `${signal['current_price']:,.4f}` ({prox_str} dari OB)\n"
        f"⚡ *Impulse:* `{signal['impulse_move_pct']}%`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Indikator:*\n"
        f"  RSI: `{signal['rsi']}`\n"
        f"  EMA{EMA_FAST}: `{signal['ema_fast']}` | EMA{EMA_SLOW}: `{signal['ema_slow']}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Konfirmasi ({signal['score']}/4):*\n"
        f"{confirm_text}\n"
        f"*Belum konfirmasi:*\n"
        f"{reason_text}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 *Sinyal:* {action}\n"
        f"⚠️ _Bukan financial advice._"
    )


# ─── SCANNER ──────────────────────────────────────────────────────────────────
async def scan_all(bot: Bot):
    logger.info("Memulai scan...")
    symbols = get_usdt_symbols()
    alert_count = 0

    for symbol in symbols:
        for tf in TIMEFRAMES:
            df = fetch_ohlcv(symbol, tf, limit=150)
            if df is None or len(df) < 60:
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
            text=f"🔍 Scan selesai `{datetime.now().strftime('%H:%M %d/%m/%Y')}`\nTidak ada sinyal yang memenuhi kriteria.",
            parse_mode="Markdown"
        )


# ─── COMMANDS ─────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Bot OB + RSI + EMA + Fibonacci aktif!*\n\n"
        "/scan — Scan manual\n/status — Status bot\n/help — Panduan indikator",
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
        f"🎯 Min skor: `{MIN_SCORE}/4`",
        parse_mode="Markdown"
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Panduan Indikator*\n\n"
        "*⭐ Sistem Skor (maks 4):*\n"
        "1️⃣ *Order Block* — zona institusi\n"
        "2️⃣ *RSI* — oversold (<35) atau overbought (>65)\n"
        "3️⃣ *EMA 20/50* — konfirmasi trend\n"
        "4️⃣ *Fibonacci* — level 38.2%, 50%, 61.8%\n\n"
        f"*Sinyal dikirim jika skor ≥ {MIN_SCORE}/4*\n\n"
        "⭐ = lemah  ⭐⭐ = cukup  ⭐⭐⭐ = kuat  ⭐⭐⭐⭐ = sangat kuat\n\n"
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
