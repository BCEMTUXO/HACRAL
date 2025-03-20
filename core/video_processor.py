import cv2
import numpy as np
from deepface import DeepFace
from moviepy.editor import VideoFileClip, concatenate_videoclips
import logging
import os
from pathlib import Path
from typing import List, Optional

# === Настройка логирования ===
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "video_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Конфигурация ===
FRAME_SKIP = 10                  # Анализировать каждый 10-й кадр (оптимизация)
EMOTION_THRESHOLD = 0.7          # Порог уверенности для учета эмоции
TARGET_EMOTIONS = ["happy", "surprise", "angry"]  # Целевые эмоции для нарезки
MIN_CLIP_DURATION = 5            # Минимальная длительность клипа (секунды)
MAX_CLIP_DURATION = 15           # Максимальная длительность клипа

class VideoProcessor:
    """Класс для анализа видео и автоматической нарезки по эмоциям."""
    
    def __init__(self):
        self.emotion_model = DeepFace.build_model("Emotion")
        logger.info("Инициализирован анализатор эмоций")

    def analyze_video_emotions(self, video_path: str) -> Optional[List[float]]:
        """
        Анализирует видео и возвращает временные метки эмоциональных пиков.
        
        Args:
            video_path (str): Путь к видеофайлу.
            
        Returns:
            Optional[List[float]]: Список временных меток в секундах.
        """
        try:
            logger.info(f"Анализ эмоций: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Не удалось открыть видео: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            timestamps = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Пропуск кадров для оптимизации
                if frame_count % FRAME_SKIP != 0:
                    frame_count += 1
                    continue

                # Анализ эмоций
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    models={"emotion": self.emotion_model},
                    enforce_detection=False,
                    silent=True
                )

                # Фильтрация результатов
                if results and results[0]["dominant_emotion"] in TARGET_EMOTIONS:
                    if results[0]["emotion"][results[0]["dominant_emotion"]] > EMOTION_THRESHOLD:
                        timestamp = frame_count / fps
                        timestamps.append(timestamp)
                        logger.debug(f"Обнаружена эмоция {results[0]['dominant_emotion']} на {timestamp:.2f} сек")

                frame_count += 1

            cap.release()
            logger.info(f"Найдено {len(timestamps)} эмоциональных моментов")
            return self._filter_timestamps(timestamps)

        except Exception as e:
            logger.error(f"Ошибка анализа: {str(e)}", exc_info=True)
            return None

    def auto_cut_video(
        self,
        input_path: str,
        output_path: str,
        timestamps: List[float]
    ) -> bool:
        """
        Создает видео из клипов вокруг эмоциональных моментов.
        
        Args:
            input_path (str): Исходное видео.
            output_path (str): Путь для сохранения результата.
            timestamps (List[float]): Временные метки от analyze_video_emotions.
            
        Returns:
            bool: Успех операции.
        """
        try:
            if not timestamps:
                logger.warning("Нет меток для нарезки")
                return False

            clip = VideoFileClip(input_path)
            clips = []

            for ts in timestamps:
                start = max(0, ts - MIN_CLIP_DURATION)
                end = min(ts + MIN_CLIP_DURATION, clip.duration)
                
                # Проверка длительности
                if (end - start) >= MIN_CLIP_DURATION:
                    subclip = clip.subclip(start, end)
                    clips.append(subclip)
                    logger.info(f"Добавлен клип: {start:.1f}-{end:.1f} сек")

            # Сборка финального видео
            if clips:
                final_clip = concatenate_videoclips(clips)
                final_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    threads=4
                )
                logger.info(f"Видео сохранено: {output_path}")
                return True
            return False

        except Exception as e:
            logger.error(f"Ошибка нарезки: {str(e)}", exc_info=True)
            return False

    def _filter_timestamps(self, timestamps: List[float]) -> List[float]:
        """Удаляет близкие дубликаты временных меток."""
        filtered = []
        prev = -999
        for ts in sorted(timestamps):
            if ts - prev > MIN_CLIP_DURATION:
                filtered.append(ts)
                prev = ts
        return filtered

# Пример использования
if __name__ == "__main__":
    vp = VideoProcessor()
    timestamps = vp.analyze_video_emotions("input/test.mp4")
    if timestamps:
        vp.auto_cut_video("input/test.mp4", "output/highlights.mp4", timestamps)

