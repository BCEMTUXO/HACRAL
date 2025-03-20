import os
import logging
from pathlib import Path
from core.audio_processing import extract_audio, transcribe_audio
from core.text_translator import TextTranslator
from core.tts_manager import TTSGenerator
from core.video_combiner import VideoCombiner

# === Настройка логирования ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Конфигурация ===
INPUT_DIR = "input"
OUTPUT_DIR = "output"
TEMP_DIR = "temp"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def process_video(input_path: str):
    """Основной процесс обработки видео."""
    try:
        logger.info(f"Начало обработки видео: {input_path}")
        
        # Шаг 1: Извлечение аудио
        logger.info("Извлечение аудио...")
        audio_path = extract_audio(input_path, TEMP_DIR)
        if not audio_path:
            logger.error("Не удалось извлечь аудио.")
            return

        # Шаг 2: Распознавание речи
        logger.info("Распознавание речи...")
        original_text = transcribe_audio(audio_path)
        if not original_text:
            logger.error("Не удалось распознать речь.")
            return

        # Шаг 3: Перевод текста
        logger.info("Перевод текста...")
        translator = TextTranslator()
        translated_text = translator.translate(original_text)
        if not translated_text:
            logger.error("Не удалось перевести текст.")
            return

        # Шаг 4: Синтез речи
        logger.info("Синтез речи...")
        tts_generator = TTSGenerator()
        tts_path = os.path.join(TEMP_DIR, "translated_audio.wav")
        if not tts_generator.generate_speech(translated_text, tts_path):
            logger.error("Не удалось синтезировать речь.")
            return

        # Шаг 5: Сборка финального видео
        logger.info("Сборка видео...")
        video_combiner = VideoCombiner()
        output_path = os.path.join(OUTPUT_DIR, "final_video.mp4")
        if video_combiner.combine(input_path, tts_path, output_path):
            logger.info(f"Готово! Видео сохранено в {output_path}")
        else:
            logger.error("Не удалось собрать видео.")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)

def main():
    """Точка входа в программу."""
    try:
        logger.info("Запуск main функции")
        # Поиск видео в папке input
        video_files = list(Path(INPUT_DIR).glob("*.mp4"))
        if not video_files:
            logger.error(f"В папке {INPUT_DIR} нет видеофайлов.")
            return

        # Обработка каждого видео
        for video_file in video_files:
            logger.info(f"Обработка видео: {video_file.name}")
            process_video(str(video_file))
    except Exception as e:
        logger.error(f"Ошибка в main: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logger.info("Запуск скрипта")
    main()
