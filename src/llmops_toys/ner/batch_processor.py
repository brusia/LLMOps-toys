# src/batch_processor.py (продолжение)
"""Обработка пакетов для повышения эффективности."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import logging
from ner.model_runner import LegalEntityExtractor

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Обработчик пакетов для извлечения сущностей."""
    
    def __init__(self, model_runner: LegalEntityExtractor):
        """
        Инициализирует обработчик пакетов.
        
        :param model_runner: Экземпляр LegalEntityExtractor
        """
        self.model_runner = model_runner
    
    async def process_batch_async(
        self,
        contracts: List[Dict[str, Any]],
        entity_types: List[str],
        batch_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Асинхронная обработка пакета контрактов.
        
        :param contracts: Список контрактов
        :param entity_types: Список типов сущностей
        :param batch_size: Размер пакета
        :return: Результаты обработки
        """
        results = []
        
        # Обрабатываем пакетами
        for i in range(0, len(contracts), batch_size):
            batch = contracts[i:i + batch_size]
            batch_results = []
            
            # Для простоты используем синхронную обработку внутри пакета
            for contract in batch:
                try:
                    contract_result = self.model_runner.extract_entities(
                        contract["contract_text"],
                        entity_types
                    )
                    batch_results.append({
                        "contract_id": id(contract),
                        "contract_text": contract["contract_text"][:100] + "...",
                        "results": contract_result
                    })
                except Exception as e:
                    logger.error(f"Ошибка при обработке контракта: {e}")
                    batch_results.append({
                        "contract_id": id(contract),
                        "contract_text": contract["contract_text"][:100] + "...",
                        "error": str(e)
                    })
            
            results.extend(batch_results)
            
        return results
    
    def process_batch_sync(
        self,
        contracts: List[Dict[str, Any]],
        entity_types: List[str],
        batch_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Синхронная обработка пакета контрактов.
        
        :param contracts: Список контрактов
        :param entity_types: Список типов сущностей
        :param batch_size: Размер пакета
        :return: Результаты обработки
        """
        results = []
        
        # Обрабатываем пакетами
        for i in range(0, len(contracts), batch_size):
            batch = contracts[i:i + batch_size]
            batch_results = []
            
            # Обработка пакета в потоке
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = []
                for contract in batch:
                    future = executor.submit(
                        self.model_runner.extract_entities,
                        contract["contract_text"],
                        entity_types
                    )
                    futures.append((future, contract))
                
                # Получаем результаты
                for future, contract in futures:
                    try:
                        contract_result = future.result(timeout=300)  # 5 минут таймаут
                        batch_results.append({
                            "contract_id": id(contract),
                            "contract_text": contract["contract_text"][:100] + "...",
                            "results": contract_result
                        })
                    except Exception as e:
                        logger.error(f"Ошибка при обработке контракта: {e}")
                        batch_results.append({
                            "contract_id": id(contract),
                            "contract_text": contract["contract_text"][:100] + "...",
                            "error": str(e)
                        })
            
            results.extend(batch_results)
            
        return results
    
    def optimize_inference(
        self,
        contracts: List[Dict[str, Any]],
        entity_types: List[str],
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """
        Оптимизация инференса с использованием пакетной обработки.
        
        :param contracts: Список контрактов
        :param entity_types: Список типов сущностей
        :param batch_size: Размер пакета
        :return: Статистика производительности
        """
        logger.info(f"Начинаем оптимизированную обработку с batch_size={batch_size}")
        
        start_time = time.time()
        
        # Используем синхронную обработку с пакетами
        results = self.process_batch_sync(
            contracts=contracts,
            entity_types=entity_types,
            batch_size=batch_size
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Подсчет статистики
        total_contracts = len(contracts)
        tokens_processed = sum(
            sum(result.get("tokens_generated", 0) 
                for result in contract_result.values()) 
            for contract_result in [r["results"] for r in results if "results" in r]
        )
        
        throughput = tokens_processed / processing_time if processing_time > 0 else 0
        
        stats = {
            "total_contracts": total_contracts,
            "processing_time_seconds": processing_time,
            "tokens_processed": tokens_processed,
            "throughput_tokens_per_second": throughput,
            "batch_size": batch_size,
            "results": results
        }
        
        logger.info(f"Обработка завершена за {processing_time:.2f} секунд")
        logger.info(f"Пропускная способность: {throughput:.2f} токенов/сек")
        
        return stats

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Инициализация модели
    extractor = LegalEntityExtractor()
    processor = BatchProcessor(extractor)
    
    # Пример данных
    sample_contracts = [
        {
            "contract_text": "В соответствии с условиями настоящего Договора, Компания А обязуется предоставить услуги по разработке программного обеспечения, а Компания Б обязуется оплатить эти услуги в размере 100 000 долларов США. Срок действия договора с 1 января 2024 года по 31 декабря 2024 года. Все споры будут разрешаться в суде г. Москвы.",
            "annotations": {}
        },
        {
            "contract_text": "Поставщик обязуется поставлять товары согласно спецификации, сроки поставки - до 30 дней после подписания контракта. Сумма оплаты составляет 50 000 евро. Споры решаются в арбитражном суде г. Санкт-Петербурга.",
            "annotations": {}
        }
    ]
    
    # Оптимизированная обработка
    entity_types = ["PERSON", "ORG", "MONEY", "DATE", "CONTRACT_TYPE", "OBLIGATION", "JURISDICTION"]
    stats = processor.optimize_inference(sample_contracts, entity_types, batch_size=2)
    
    print(f"Обработано {stats['total_contracts']} контрактов")
    print(f"Время обработки: {stats['processing_time_seconds']:.2f} секунд")
    print(f"Пропускная способность: {stats['throughput_tokens_per_second']:.2f} токенов/сек")