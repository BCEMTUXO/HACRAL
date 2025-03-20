import torch
from TTS.api import TTS
import logging
import os
from pathlib import Path
from typing import Optional, Dict

# === Настройка логирования ===
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # Измените уровень на DEBUG для более подробного логирования
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "tts_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Конфигурация ===
DEFAULT_MODEL = "xtts_v2"  # xtts_v2 / your_tts / glados
VOICE_SAMPLES = {
    "rubius": "models/voices/rubius_ref.wav",
    "natasha": "models/voices/natasha_ref.wav"
}
MAX_TEXT_LENGTH = 500  # Максимальная длина текста для синтеза (символов)

class TTSGeneratorError(Exception):
    """Базовый класс для исключений в TTSGenerator."""
    pass

class ModelNotFoundError(TTSGeneratorError):
    """Исключение для случая, когда модель не найдена."""
    pass

class VoiceNotFoundError(TTSGeneratorError):
    """Исключение для случая, когда голос не найден."""
    pass

class FileCreationError(TTSGeneratorError):
    """Исключение для случая, когда файл не был создан."""
    pass

class TTSGenerator:
    """Класс для синтеза речи с поддержкой разных движков."""
    
    def __init__(self):
        logger.debug("Инициализация TTSGenerator")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = self._load_models()
        logger.info(f"Инициализирован TTSGenerator на устройстве: {self.device.upper()}")

    def _load_models(self) -> Dict[str, TTS]:
        """Загружает предварительно настроенные модели."""
        logger.debug("Загрузка моделей")
        models = {
            "xtts_v2": TTS("tts_models/multilingual/multi-dataset/xtts_v2"),
            "your_tts": TTS("tts_models/multilingual/multi-dataset/your_tts")
        }
        
        try:
            for model in models.values():
                model.to(self.device)
            logger.debug("Модели успешно загружены")
            return models
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {str(e)}", exc_info=True)
            raise TTSGeneratorError("Ошибка загрузки моделей") from e

    def _preprocess_text(self, text: str) -> str:
        """Очистка текста перед синтезом."""
        logger.debug(f"Предобработка текста: {text}")
        text = text.strip()[:MAX_TEXT_LENGTH]
        return text.replace("  ", " ")

    def generate_speech(
        self,
        text: str,
        output_path: str,
        voice_id: str = "rubius",
        model_id: str = DEFAULT_MODEL
    ) -> bool:
        """
        Генерирует речь из текста с клонированием голоса.
        
        Args:
            text (str): Входной текст
            output_path (str): Путь для сохранения аудио
            voice_id (str): Идентификатор голоса из VOICE_SAMPLES
            model_id (str): Идентификатор модели (xtts_v2/your_tts)

        Returns:
            bool: Успех операции
        """
        logger.debug("Начало генерации речи")
        try:
            # Валидация входных данных
            if not text:
                logger.error("Получен пустой текст")
                return False

            if model_id not in self.models:
                raise ModelNotFoundError(f"Модель {model_id} не найдена")

            if voice_id not in VOICE_SAMPLES:
                raise VoiceNotFoundError(f"Голос {voice_id} не найден")

            # Подготовка текста
            processed_text = self._preprocess_text(text)
            logger.info(f"Синтез речи: {processed_text[:50]}...")

            # Синтез через выбранную модель
            self.models[model_id].tts_to_file(
                text=processed_text,
                speaker_wav=VOICE_SAMPLES[voice_id],
                language="ru",
                file_path=output_path
            )

            # Проверка результата
            if not Path(output_path).exists():
                raise FileCreationError("Файл не был создан")
                
            logger.info(f"Аудио успешно сохранено: {output_path}")
            return True

        except ModelNotFoundError as e:
            logger.error(f"Ошибка модели: {str(e)}", exc_info=True)
        except VoiceNotFoundError as e:
            logger.error(f"Ошибка голоса: {str(e)}", exc_info=True)
        except FileCreationError as e:
            logger.error(f"Ошибка создания файла: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Неизвестная ошибка синтеза: {str(e)}", exc_info=True)
        return False

# Пример использования
if __name__ == "__main__":
    logger.debug("Запуск примера использования")
    tts = TTSGenerator()
    success = tts.generate_speech(
        text="Привет! Это тестовый пример синтеза речи.",
        output_path="output/test_voice.wav",
        voice_id="rubius"
    )
    logger.debug(f"Результат: {'Успех' if success else 'Ошибка'}")
    print(f"Результат: {'Успех' if success else 'Ошибка'}")
