# Базовый образ с Python 3.10
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \           # Для обработки аудио/видео
    git \               # Для загрузки моделей из репозиториев
    libgl1 \            # Для работы с OpenCV
    && rm -rf /var/lib/apt/lists/*

# Настройка рабочей директории
WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .

# Установка Python-пакетов
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего проекта
COPY . .

# Создание папок для данных
RUN mkdir -p /app/input /app/output /app/temp /app/logs /app/models

# Настройка переменных окружения
ENV PYTHONUNBUFFERED=1
ENV LOG_PATH=/app/logs

# Команда запуска
CMD ["python", "main.py"]