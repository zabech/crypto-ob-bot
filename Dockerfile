FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY crypto_ob_bot.py .

CMD ["python", "crypto_ob_bot.py"]
