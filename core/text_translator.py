from googletrans import Translator
import logging
import os
from pathlib import Path
from typing import Optional

# === Настройка логирования ===
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "text_translator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Конфигурация ===
DEFAULT_SRC_LANG = "es"   # Язык оригинала (испанский)
DEFAULT_DEST_LANG = "ru"  # Целевой язык (русский)
MAX_TEXT_LENGTH = 5000    # Максимальная длина текста для перевода (символов)

class TextTranslator:
    def __init__(self):
        self.translator = Translator()
        self._validate_translator()

    def _validate_translator(self) -> None:
        """Проверка работоспособности переводчика."""
        try:
            test_translation = self.translator.translate("test", src="en", dest="ru")
            if not test_translation.text:
                raise RuntimeError("Не удалось инициализировать переводчик.")
            logger.info("Переводчик успешно инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации переводчика: {str(e)}", exc_info=True)
            raise

    def translate(
        self, 
        text: str, 
        src_lang: str = DEFAULT_SRC_LANG, 
        dest_lang: str = DEFAULT_DEST_LANG
    ) -> Optional[str]:
        """
        Переводит текст с исходного языка на целевой.
        
        Args:
            text (str): Текст для перевода.
            src_lang (str): ISO-код языка оригинала (по умолчанию "es").
            dest_lang (str): ISO-код целевого языка (по умолчанию "ru").
        
        Returns:
            Optional[str]: Переведенный текст или None при ошибке.
        """
        try:
            # Валидация входных данных
            if not text:
                logger.warning("Получен пустой текст для перевода.")
                return None
                
            if len(text) > MAX_TEXT_LENGTH:
                logger.warning(f"Текст превышает лимит ({MAX_TEXT_LENGTH} символов).")
                text = text[:MAX_TEXT_LENGTH]
            
            logger.info(f"Начало перевода ({src_lang} → {dest_lang})...")
            
            # Выполнение перевода
            translated = self.translator.translate(
                text, 
                src=src_lang, 
                dest=dest_lang
            )
            
            # Постобработка
            result = translated.text.strip()
            logger.info(f"Успешно переведено символов: {len(result)}")
            logger.debug(f"Пример перевода: {result[:100]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка перевода: {str(e)}", exc_info=True)
            return None

# Пример использования
if __name__ == "__main__":
    translator = TextTranslator()
    sample_text = "Hola, ¿cómo estás?"
    translated_text = translator.translate(sample_text)
    print(f"Перевод: {translated_text}")
