# src/performance_analyzer.py
"""Анализ производительности моделей."""

import time
import torch
from typing import List, Dict, Any
import logging
from ner.model_runner import LegalEntityExtractor
from ner.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Анализ производительности моделей."""
    
    def __init__(self):
        """Инициализирует анализатор производительности."""
        pass
    
    def compare_models(
        self,
        models_config: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        entity_types: List[str],
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """
        Сравнивает производительность разных моделей.
        
        :param models_config: Конфигурации моделей
        :param test_data: Тестовые данные
        :param entity_types: Список типов сущностей
        :param batch_size: Размер пакета
        :return: Результаты сравнения
        """
        results = {}
        
        for config in models_config:
            model_name = config["name"]
            logger.info(f"Тестирование модели: {model_name}")
            
            # Инициализация модели
            try:
                model_runner = LegalEntityExtractor(
                    model_name=config["model_path"],
                    device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
                )
                
                # Создание обработчика пакетов
                processor = BatchProcessor(model_runner)
                
                # Запуск теста
                start_time = time.time()
                stats = processor.optimize_inference(test_data, entity_types, batch_size)
                end_time = time.time()
                
                # Собираем метрики
                metrics = {
                    "model_name": model_name,
                    "processing_time": stats["processing_time_seconds"],
                    "throughput": stats["throughput_tokens_per_second"],
                    "total_contracts": stats["total_contracts"],
                    "tokens_processed": stats["tokens_processed"],
                    "batch_size": batch_size,
                    "memory_usage": self._get_memory_usage(),
                    "inference_time_per_contract": stats["processing_time_seconds"] / stats["total_contracts"]
                }
                
                results[model_name] = metrics
                
                logger.info(f"Модель {model_name} завершена")
                logger.info(f"Время: {metrics['processing_time']:.2f} секунд")
                logger.info(f"Пропускная способность: {metrics['throughput']:.2f} токенов/сек")
                
            except Exception as e:
                logger.error(f"Ошибка при тестировании модели {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """
        Получает информацию об использовании памяти.
        
        :return: Информация о памяти
        """
        memory_info = {}
        
        if torch.cuda.is_available():
            # Использование VRAM
            memory_info["vram_used_mb"] = torch.cuda.memory_allocated() / (1024**2)
            memory_info["vram_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
        else:
            # Использование RAM (пример)
            memory_info["ram_used_mb"] = 0.0
            
        return memory_info
    
    def generate_report(
        self,
        comparison_results: Dict[str, Any]
    ) -> str:
        """
        Генерирует отчет о сравнении моделей.
        
        :param comparison_results: Результаты сравнения
        :return: Отчет в виде строки
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ")
        report_lines.append("=" * 60)
        
        # Сортировка по скорости обработки
        sorted_results = sorted(
            comparison_results.items(),
            key=lambda x: x[1].get("throughput", 0),
            reverse=True
        )
        
        for model_name, metrics in sorted_results:
            if "error" in metrics:
                report_lines.append(f"\n{model_name}: ОШИБКА - {metrics['error']}")
                continue
                
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  Время обработки: {metrics['processing_time']:.2f} секунд")
            report_lines.append(f"  Пропускная способность: {metrics['throughput']:.2f} токенов/сек")
            report_lines.append(f"  Контрактов обработано: {metrics['total_contracts']}")
            report_lines.append(f"  Токенов обработано: {metrics['tokens_processed']}")
            report_lines.append(f"  Использование памяти: {metrics.get('memory_usage', {})}")
            report_lines.append(f"  Время на контракт: {metrics['inference_time_per_contract']:.2f} секунд")
        
        report_lines.append("\n" + "=" * 60)
        return "\n".join(report_lines)

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Конфигурации моделей для сравнения
    models_config = [
        {
            "name": "Llama-2-7B-Chat (FP16)",
            "model_path": "meta-llama/Llama-2-7b-chat-hf",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
    ]
    
    # Тестовые данные
    test_data = [
        {
            "contract_text": "В соответствии с условиями настоящего Договора, Компания А обязуется предоставить услуги по разработке программного обеспечения, а Компания Б обязуется оплатить эти услуги в размере 100 000 долларов США. Срок действия договора с 1 января 2024 года по 31 декабря 2024 года. Все споры будут разрешаться в суде г. Москвы.",
            "annotations": {}
        },
        {
            "contract_text": "Поставщик обязуется поставлять товары согласно спецификации, сроки поставки - до 30 дней после подписания контракта. Сумма оплаты составляет 50 000 евро. Споры решаются в арбитражном суде г. Санкт-Петербурга.",
            "annotations": {}
        }
    ]
    
    # Инициализация анализатора
    analyzer = PerformanceAnalyzer()
    
    # Сравнение моделей
    results = analyzer.compare_models(
        models_config=models_config,
        test_data=test_data,
        entity_types=["PERSON", "ORG", "MONEY", "DATE", "CONTRACT_TYPE", "OBLIGATION", "JURISDICTION"],
        batch_size=2
    )
    
    # Генерация отчета
    report = analyzer.generate_report(results)
    print(report)