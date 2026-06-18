"""
Telegram Crypto Bot - Order Block Scanner
Versi Railway-ready: konfigurasi via Environment Variables
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

# ─── KONFIGURASI (dari Environment Variable) ────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

TIMEFRAMES = ["4h", "1d"]
SCAN_INTERVAL_MINUTES = 30
MAX_SYMBOLS = 100

OB_LOOKBACK = 20
OB_MIN_BODY_RATIO = 0.6
OB_VOLUME_MULTIPLIER = 1.5

# ─── LOGGING ─────────────────────────────────────────────────────────────────
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
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


def is_strong_candle(row: pd.Series, avg_volume: float) -> bool:
    candle_range = row["high"] - row["low"]
    if candle_range == 0:
        return False
    body = abs(row["close"] - row["open"])
    return (body / candle_range) >= OB_MIN_BODY_RATIO and row["volume"] >= avg_volume * OB_VOLUME_MULTIPLIER


def detect_order_blocks(df: pd.DataFrame) -> dict:
    result = {"bullish_ob": [], "bearish_ob": []}
    if len(df) < OB_LOOKBACK + 5:
        return result

    avg_volume = df["volume"].rolling(20).mean()
    current_price = df["close"].iloc[-1]

    for i in range(2, len(df) - 3):
        row = df.iloc[i]
        avg_vol = avg_volume.iloc[i]
        if not is_strong_candle(row, avg_vol):
            continue

        next_candles = df.iloc[i + 1: i + 4]

        if row["close"] < row["open"]:  # Bullish OB
            bullish_follow = sum(1 for _, c in next_candles.iterrows() if c["close"] > c["open"])
            impulse_move = (next_candles["high"].max() - row["low"]) / row["low"] * 100
            if bullish_follow >= 2 and impulse_move >= 1.0:
                ob_high, ob_low = row["open"], row["close"]
                proximity = (current_price - ob_high) / ob_high * 100
                if -2.0 <= proximity <= 2.0:
                    result["bullish_ob"].append({"time": row["timestamp"], "ob_high": ob_high, "ob_low": ob_low,
                        "impulse_move_pct": round(impulse_move, 2), "proximity_pct": round(proximity, 2), "current_price": current_price})

        elif row["close"] > row["open"]:  # Bearish OB
            bearish_follow = sum(1 for _, c in next_candles.iterrows() if c["close"] < c["open"])
            impulse_move = (row["high"] - next_candles["low"].min()) / row["high"] * 100
            if bearish_follow >= 2 and impulse_move >= 1.0:
                ob_high, ob_low = row["close"], row["open"]
                proximity = (current_price - ob_low) / ob_low * 100
                if -2.0 <= proximity <= 2.0:
                    result["bearish_ob"].append({"time": row["timestamp"], "ob_high": ob_high, "ob_low": ob_low,
                        "impulse_move_pct": round(impulse_move, 2), "proximity_pct": round(proximity, 2), "current_price": current_price})

    return result


def format_alert(symbol: str, timeframe: str, ob_type: str, ob_data: dict) -> str:
    emoji = "🟢" if ob_type == "bullish" else "🔴"
    direction = "BULLISH" if ob_type == "bullish" else "BEARISH"
    action = "Support / Potensi Naik" if ob_type == "bullish" else "Resistance / Potensi Turun"
    coin = symbol.replace("/USDT", "")
    prox = ob_data["proximity_pct"]
    prox_str = f"+{prox:.2f}%" if prox >= 0 else f"{prox:.2f}%"
    return (
        f"{emoji} *Order Block {direction} Terdeteksi!*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 *Koin:* `{coin}/USDT`\n"
        f"⏱ *Timeframe:* `{timeframe.upper()}`\n"
        f"📍 *Zona OB:* `${ob_data['ob_low']:,.4f}` — `${ob_data['ob_high']:,.4f}`\n"
        f"💰 *Harga Saat Ini:* `${ob_data['current_price']:,.4f}`\n"
        f"📏 *Jarak ke OB:* `{prox_str}`\n"
        f"⚡ *Impulse Move:* `{ob_data['impulse_move_pct']}%`\n"
        f"🕐 *Candle OB:* `{ob_data['time'].strftime('%Y-%m-%d %H:%M')}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💡 *Sinyal:* {action}\n"
        f"⚠️ _Bukan financial advice._"
    )


async def scan_order_blocks(bot: Bot):
    logger.info("Memulai scan...")
    symbols = get_usdt_symbols()
    alert_count = 0
    for symbol in symbols:
        for tf in TIMEFRAMES:
            df = fetch_ohlcv(symbol, tf, limit=100)
            if df is None or len(df) < 30:
                continue
            ob_results = detect_order_blocks(df)
            for ob in ob_results["bullish_ob"] + ob_results["bearish_ob"]:
                ob_type = "bullish" if ob in ob_results["bullish_ob"] else "bearish"
                msg = format_alert(symbol, tf, ob_type, ob)
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
                alert_count += 1
                await asyncio.sleep(0.5)
            await asyncio.sleep(0.3)
    if alert_count == 0:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID,
            text=f"🔍 Scan selesai `{datetime.now().strftime('%H:%M %d/%m/%Y')}`\nTidak ada OB mendekati harga saat ini.", parse_mode="Markdown")
    logger.info(f"Scan selesai. Alert: {alert_count}")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 *Bot Order Block Scanner aktif!*\n\n/scan — Scan manual\n/status — Status\n/help — Panduan", parse_mode="Markdown")

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Scan dimulai, harap tunggu...")
    await scan_order_blocks(context.bot)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbols = get_usdt_symbols()
    await update.message.reply_text(f"✅ *Bot aktif*\n📊 Simbol: `{len(symbols)}`\n⏱ TF: `{', '.join(TIMEFRAMES)}`\n🔄 Interval: `{SCAN_INTERVAL_MINUTES} menit`", parse_mode="Markdown")

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📖 *Panduan*\n\n🟢 *Bullish OB:* Zona support → potensi naik\n🔴 *Bearish OB:* Zona resistance → potensi turun\n\n⚠️ _Bukan financial advice._", parse_mode="Markdown")

async def auto_scan_job(context: ContextTypes.DEFAULT_TYPE):
    await scan_order_blocks(context.bot)


def main():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise ValueError("Set TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID di environment variables Railway!")
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
