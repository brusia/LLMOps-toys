from jinja2.lexer import float_re

import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from transformers import pipeline
from openai import OpenAI
import os
from analyse import AnalysisResult
from data_engineering import clean_text
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Результаты работы моделей"""
    text: str
    hf_sentiment: str
    llm_sentiment: str
    hf_entities: List[str]
    llm_entities: List[str]
    hf_processing_time: float
    llm_processing_time: float
    hf_tp: int
    llm_tp: int

@dataclass
class ComparisonMetrics:
    accuracy_sentiment: float
    accuracy_entities: int | float
    avg_processing_time: float
    entity_count: int | float
    hf_accuracy: float = 0.0
    llm_accuracy: float = 0.0


class ModelComparison:
    """Класс для сравнения HuggingFace и LLM моделей"""
 
    def compare_model_results(self, model_results_1: List[AnalysisResult], model_results_2: List[AnalysisResult], marked_data: pd.DataFrame) -> List[ModelResults]:
        """Сравнение моделей на наборе текстов"""
        results = []

        dict_1 = {r.text: r for r in model_results_1}
        dict_2 = {r.text: r for r in model_results_2}

        common_texts = set(dict_1.keys()).intersection(set(dict_2.keys()))
        
        for text in sorted(common_texts):
            res1 = dict_1[text]
            res2 = dict_2[text]

            gt = marked_data[marked_data["review"] == text]["sentiment"].iloc[0] if not marked_data[marked_data["review"] == text].empty else None

            hf_tp = 0
            if gt is not None:
                if res1.sentiment == gt:
                    hf_tp = 1

            llm_tp = 0
            if gt is not None:
                if res2.sentiment == gt:
                    llm_tp = 1

            result = ModelResults(
                text=res1.text,
                hf_sentiment=res1.sentiment,
                llm_sentiment=res2.sentiment,
                hf_entities=res1.entities,
                llm_entities=res2.entities,
                hf_processing_time=res1.processing_time,
                llm_processing_time=res2.processing_time,
                hf_tp=hf_tp,
                llm_tp=llm_tp
                )

            results.append(result)
            
        return results
    
    def calculate_metrics(self, results: List[ModelResults]) -> ComparisonMetrics:
        """Расчет метрик сравнения"""
        # Точность тональности
        correct_sentiment = 0
        total_texts = len(results)
        
        for result in results:
            if result.hf_sentiment == result.llm_sentiment:
                correct_sentiment += 1
                
        accuracy_sentiment = correct_sentiment / total_texts if total_texts > 0 else 0
        
        # Точность извлечения сущностей (упрощенная версия)
        correct_entities = 0
        total_entities = 0
        
        for result in results:
            # Сравниваем количество совпадающих сущностей
            hf_set = set(result.hf_entities)
            llm_set = set(result.llm_entities)
            intersection = len(hf_set.intersection(llm_set))
            union = len(hf_set.union(llm_set))
            
            if union > 0:
                correct_entities += intersection / union
            total_entities += 1
            
        accuracy_entities = correct_entities / total_entities if total_entities > 0 else 0.0
        
        # Среднее время обработки
        hf_times = [r.hf_processing_time for r in results]
        llm_times = [r.llm_processing_time for r in results]
        avg_processing_time = (np.mean(hf_times) + np.mean(llm_times)) / 2
        
        # Количество найденных сущностей
        total_hf_entities = sum(len(r.hf_entities) for r in results)
        total_llm_entities = sum(len(r.llm_entities) for r in results)
        entity_count = (total_hf_entities + total_llm_entities) / 2
        
         # Точность на основе TP
        hf_tp_total = sum(r.hf_tp for r in results)
        llm_tp_total = sum(r.llm_tp for r in results)
        hf_accuracy = hf_tp_total / total_texts if total_texts > 0 else 0.0
        llm_accuracy = llm_tp_total / total_texts if total_texts > 0 else 0.0
        
        return ComparisonMetrics(
            accuracy_sentiment=accuracy_sentiment,
            accuracy_entities=accuracy_entities,
            avg_processing_time=avg_processing_time,
            entity_count=entity_count,
            hf_accuracy=hf_accuracy,
            llm_accuracy=llm_accuracy
        )
    
    def visualize_results(self, results: List[ModelResults], metrics: ComparisonMetrics):
        """Визуализация результатов сравнения"""
        # Подготовка данных для графиков
        texts = [r.text[:30] + "..." if len(r.text) > 30 else r.text for r in results]
        hf_times = [r.hf_processing_time for r in results]
        llm_times = [r.llm_processing_time for r in results]
        
        # График времени обработки
        plt.figure(figsize=(12, 8))
        
        # Время обработки
        plt.subplot(2, 2, 1)
        x = np.arange(len(texts))
        width = 0.35
        plt.bar(x - width/2, hf_times, width, label='HuggingFace')
        plt.bar(x + width/2, llm_times, width, label='LLM')
        plt.xlabel('Тексты')
        plt.ylabel('Время (с)')
        plt.title('Сравнение времени обработки')
        plt.xticks(x, texts, rotation=45)
        plt.legend()
        
        # Количество сущностей
        plt.subplot(2, 2, 2)
        hf_entities = [len(r.hf_entities) for r in results]
        llm_entities = [len(r.llm_entities) for r in results]
        plt.bar(x - width/2, hf_entities, width, label='HuggingFace')
        plt.bar(x + width/2, llm_entities, width, label='LLM')
        plt.xlabel('Тексты')
        plt.ylabel('Количество сущностей')
        plt.title('Сравнение количества извлеченных сущностей')
        plt.xticks(x, texts, rotation=45)
        plt.legend()
        
        # Точность по метрикам
        plt.subplot(2, 2, 3)
        metrics_data = [
            metrics.accuracy_sentiment,
            metrics.accuracy_entities
        ]
        metric_names = ['Точность тональности', 'Точность сущностей']
        bars = plt.bar(range(len(metric_names)), metrics_data)
        plt.xlabel('Метрики')
        plt.ylabel('Значение')
        plt.title('Сравнение точности')
        plt.xticks(range(len(metric_names)), metric_names)
        # Добавляем значения на столбцах
        for i, (bar, value) in enumerate(zip(bars, metrics_data)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
            
        # Точность на основе TP
        plt.subplot(2, 2, 4)
        tp_data = [
            sum(r.hf_tp for r in results) / len(results) if len(results) > 0 else 0,
            sum(r.llm_tp for r in results) / len(results) if len(results) > 0 else 0
        ]
        tp_names = ['HuggingFace TP', 'LLM TP']
        bars = plt.bar(range(len(tp_names)), tp_data)
        plt.xlabel('Модели')
        plt.ylabel('TP Accuracy')
        plt.title('Точность на основе TP')
        plt.xticks(range(len(tp_names)), tp_names)
        # Добавляем значения на столбцах
        for i, (bar, value) in enumerate(zip(bars, tp_data)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('comparison_results.png')
        plt.show()
        
    def generate_report(self, results: List[ModelResults], metrics: ComparisonMetrics):
        """Генерация отчета о сравнении"""
        logger.info("=" * 60)
        logger.info("ОТЧЕТ О СРАВНЕНИИ МОДЕЛЕЙ")
        logger.info("=" * 60)
        logger.info("Точность анализа тональности: {:.2%}".format(metrics.accuracy_sentiment))
        logger.info("Точность извлечения сущностей: {:.2%}".format(metrics.accuracy_entities))
        logger.info("Среднее время обработки: {:.3f} секунд".format(metrics.avg_processing_time))
        logger.info("Среднее количество найденных сущностей: {:.1f}".format(metrics.entity_count))
        logger.info("\nВыводы:")
        logger.info("-" * 60)
        
        if metrics.accuracy_sentiment > 0.7:
            logger.info("HuggingFace и LLM показывают высокую точность в анализе тональности")
        else:
            logger.info("Нуждается в улучшении точности анализа тональности")
            
        if metrics.accuracy_entities > 0.6:
            logger.info("Обе модели эффективны в извлечении сущностей")
        else:
            logger.info("Требуется улучшение качества извлечения сущностей")
            
        if metrics.avg_processing_time < 1.0:
            logger.info("Обе модели работают быстро")
        else:
            logger.info("Время обработки может быть оптимизировано")

        logger.info("Оба подхода просты в использовании.")