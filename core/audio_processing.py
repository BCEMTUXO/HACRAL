import ffmpeg
import whisper
import os
import logging
from pathlib import Path
from typing import Optional
import torch

# === Настройка логирования ===
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # Увеличен уровень логирования до DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "audio_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Конфигурация ===
WHISPER_MODEL = "base"  # Модель для распознавания речи (base, small, medium)
SAMPLE_RATE = 16000     # Частота дискретизации аудио (Гц)
AUDIO_FORMAT = "wav"    # Формат выходного аудио

def extract_audio(video_path: str, output_dir: str = "temp") -> Optional[str]:
    """
    Извлекает аудиодорожку из видеофайла.
    
    Args:
        video_path (str): Путь к исходному видео.
        output_dir (str): Папка для сохранения аудио.
    
    Returns:
        Optional[str]: Путь к аудиофайлу или None при ошибке.
    """
    try:
        logger.info(f"Извлечение аудио из: {video_path}")
        
        # Проверка существования видеофайла
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Видеофайл {video_path} не найден!")
        
        # Создание папки для выходных данных
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(exist_ok=True, parents=True)
        
        # Генерация имени файла
        audio_filename = f"{Path(video_path).stem}_audio.{AUDIO_FORMAT}"
        audio_path = output_dir_path / audio_filename
        
        # Извлечение аудио через FFmpeg
        logger.debug(f"Команда FFmpeg: ffmpeg -i {video_path} -ac 1 -ar {SAMPLE_RATE} -acodec pcm_s16le {audio_path}")
        (
            ffmpeg
            .input(video_path)
            .output(
                str(audio_path),
                ac=1,              # Моно-канал
                ar=SAMPLE_RATE,    # Частота дискретизации
                acodec='pcm_s16le' # 16-битный PCM
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        
        logger.info(f"Аудио успешно сохранено: {audio_path}")
        return str(audio_path)
        
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode('utf-8')}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Ошибка извлечения аудио: {str(e)}", exc_info=True)
        return None

def transcribe_audio(audio_path: str, language: str = "es") -> Optional[str]:
    """
    Распознает речь из аудиофайла.
    
    Args:
        audio_path (str): Путь к аудиофайлу.
        language (str): Язык оригинала (ISO 639-1 код).
    
    Returns:
        Optional[str]: Распознанный текст или None при ошибке.
    """
    try:
        logger.info(f"Распознавание речи из: {audio_path}")
        
        # Проверка существования файла
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Аудиофайл {audio_path} не найден!")
        
        # Проверка наличия CUDA
        use_cuda = torch.cuda.is_available()
        logger.debug(f"CUDA доступен: {use_cuda}")
        
        # Загрузка модели Whisper
        model = whisper.load_model(WHISPER_MODEL)
        logger.info(f"Загружена модель Whisper: {WHISPER_MODEL}")
        
        # Распознавание речи
        result = model.transcribe(
            audio_path,
            language=language,
            verbose=False,
            fp16=use_cuda  # Использование GPU, если доступно
        )
        
        # Постобработка текста
        text = result["text"].strip()
        logger.info(f"Успешно распознано символов: {len(text)}")
        logger.debug(f"Пример текста: {text[:100]}...")
        
        return text
        
    except Exception as e:
        logger.error(f"Ошибка распознавания: {str(e)}", exc_info=True)
        return None
