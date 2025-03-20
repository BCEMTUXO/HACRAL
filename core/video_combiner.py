from moviepy.editor import VideoFileClip, AudioFileClip
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
        logging.FileHandler(log_dir / "video_combiner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Конфигурация ===
OUTPUT_CODEC = "libx264"    # Кодек видео (libx264, prores, mpeg4)
AUDIO_CODEC = "aac"         # Кодек аудио (aac, mp3)
OUTPUT_PRESET = "medium"    # Пресет FFmpeg (ultrafast, medium, slow)
THREADS = 4                 # Количество потоков для рендеринга

class VideoCombiner:
    """Класс для объединения видео и аудиодорожек."""
    
    def combine(
        self, 
        video_path: str, 
        audio_path: str, 
        output_dir: str = "output"
    ) -> Optional[str]:
        """
        Соединяет видео и аудио, сохраняя результат в output_dir.
        
        Args:
            video_path (str): Путь к исходному видео (без звука).
            audio_path (str): Путь к аудиофайлу для наложения.
            output_dir (str): Папка для сохранения результата.
            
        Returns:
            Optional[str]: Путь к итоговому файлу или None при ошибке.
        """
        try:
            # Валидация входных данных
            if not self._validate_paths(video_path, audio_path):
                return None

            logger.info(f"Начало обработки: {video_path}")

            # Загрузка клипов
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)

            # Синхронизация длительности
            audio_clip = self._sync_duration(audio_clip, video_clip.duration)

            # Наложение аудио
            final_clip = video_clip.set_audio(audio_clip)
            output_path = self._generate_output_path(video_path, output_dir)

            # Экспорт видео
            final_clip.write_videofile(
                output_path,
                codec=OUTPUT_CODEC,
                audio_codec=AUDIO_CODEC,
                threads=THREADS,
                preset=OUTPUT_PRESET
            )

            logger.info(f"Видео успешно собрано: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Ошибка комбинации: {str(e)}", exc_info=True)
            return None

        finally:
            # Закрытие клипов для освобождения ресурсов
            if 'video_clip' in locals() and video_clip:
                video_clip.close()
            if 'audio_clip' in locals() and audio_clip:
                audio_clip.close()

    def _validate_paths(self, video_path: str, audio_path: str) -> bool:
        """Проверяет существование файлов."""
        for path in [video_path, audio_path]:
            if not Path(path).exists():
                logger.error(f"Файл не найден: {path}")
                return False
        return True

    def _sync_duration(self, audio_clip: AudioFileClip, max_duration: float) -> AudioFileClip:
        """Обрезает аудио до длительности видео."""
        if audio_clip.duration > max_duration:
            logger.warning(f"Аудио ({audio_clip.duration:.1f} сек) длиннее видео ({max_duration:.1f} сек). Обрезка.")
            return audio_clip.subclip(0, max_duration)
        elif audio_clip.duration < max_duration:
            logger.warning(f"Аудио ({audio_clip.duration:.1f} сек) короче видео ({max_duration:.1f} сек).")
        return audio_clip

    def _generate_output_path(self, video_path: str, output_dir: str) -> str:
        """Генерирует путь для сохранения."""
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(exist_ok=True, parents=True)
        return str(output_dir_path / f"final_{Path(video_path).stem}.mp4")

# Пример использования
if __name__ == "__main__":
    combiner = VideoCombiner()
    result = combiner.combine(
        video_path="input/original.mp4",
        audio_path="temp/translated_audio.wav",
        output_dir="output"
    )
    print(f"Результат: {result}")
