
from dataclasses import dataclass
from typing import List
import logging
import time

from transformers import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Результаты анализа текста"""
    text: str
    sentiment: str
    entities: List[str]
    ner_time: float
    sentiment_time: float

    @property
    def processing_time(self) -> float:
        return self.ner_time + self.sentiment_time

    def display(self) -> None:
        """Вывод результатов анализа"""
        logger.info("="*60)
        logger.info("РЕЗУЛЬТАТ АНАЛИЗА")
        logger.info("="*60)
        logger.info("Текст: {}".format(self.text))
        logger.info("Тональность: {}".format(self.sentiment))
        logger.info("Сущности: {}".format(", ".join(self.entities) if self.entities else "Нет"))
        logger.info("Время анализа тональности: {:.4f} сек".format(self.sentiment_time))
        logger.info("Время анализа сущностей: {:.4f} сек".format(self.ner_time))
        logger.info("Общее время: {:.4f} сек".format(self.processing_time))
        logger.info("-"*60)


def analyze_with_huggingface(sentiment_pipeline: Pipeline, ner_pipeline: Pipeline, texts: List[str]) -> List[AnalysisResult]:
    """
    Анализ текстов с использованием реальных HuggingFace моделей
    
    :param texts: Список текстов для анализа
    :return: Список результатов анализа
    """
    logger.info("Запуск анализа с использованием реальных HuggingFace моделей")

    labels_map = {"label_0": "positive", 'label_1': "negative" }
    try:
        results = []
        for text in texts:
            start_time_sentiment = time.time()
            
            # Анализ тональности
            sentiment_result = sentiment_pipeline(text)
            sentiment = labels_map[sentiment_result[0]['label'].lower()]

            end_time_sentiment = time.time()
            sentiment_time = end_time_sentiment - start_time_sentiment
            # Извлечение сущностей
            try:
                ner_results = ner_pipeline(text)
                entities = list(set([entity['entity_group'] for entity in ner_results]))
            except Exception as e:
                logger.warning(f"Ошибка при извлечении сущностей: {e}")
                entities = []
            
            end_time = time.time()
            processing_time = end_time - start_time_sentiment
            
            results.append(AnalysisResult(
                text=text,
                sentiment=sentiment,
                entities=entities,
                sentiment_time=sentiment_time,
                ner_time=end_time - end_time_sentiment
            ))
        
        return results
    
    except ImportError as e:
        logger.error(f"Не удалось импортировать HuggingFace библиотеки: {e}")

