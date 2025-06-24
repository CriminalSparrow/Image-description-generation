FROM python:3.12-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Устанавливаем необходимые пакеты
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt /app

# Устанавливаем зависимости
RUN pip3 install --no-cache-dir -r requirements.txt

# Копируем апишку
COPY app.py /app

COPY .env /app

# Открываем порт (5005)
EXPOSE 5005

# Запускаем FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005"]
