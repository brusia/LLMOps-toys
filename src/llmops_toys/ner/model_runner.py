# src/model_runner.py
"""Запуск локальных моделей для извлечения сущностей."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
import time
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class LegalEntityExtractor:
    """Извлекатель сущностей из юридических документов."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Инициализирует извлекатель сущностей.
        
        :param model_name: Название модели
        :param device: Устройство для запуска модели
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Загрузка модели {model_name} на устройство {device}")
        
        # Загрузка токенизатора и модели
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        
        # Установка параметров токенизатора
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info("Модель успешно загружена")
    
    @torch.no_grad()
    def extract_entities(
        self,
        contract_text: str,
        entity_types: List[str],
        max_new_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Извлекает сущности из текста контракта.
        
        :param contract_text: Текст контракта
        :param entity_types: Список типов сущностей для извлечения
        :param max_new_tokens: Максимальное количество токенов в ответе
        :return: Результаты извлечения
        """
        results = {}
        
        for entity_type in entity_types:
            logger.debug(f"Извлечение сущностей типа {entity_type}")
            
            # Формируем промпт
            from prompts import build_prompt
            prompt = build_prompt(entity_type, contract_text)
            
            # Токенизация
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Генерация ответа
            start_time = time.time()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            generation_time = time.time() - start_time
            
            # Декодирование результата
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Парсинг результата (упрощенный вариант)
            try:
                # В реальной реализации здесь будет более сложная логика парсинга
                results[entity_type] = {
                    "response": response,
                    "generation_time": generation_time,
                    "tokens_generated": len(outputs[0])
                }
            except Exception as e:
                logger.error(f"Ошибка при парсинге ответа для {entity_type}: {e}")
                results[entity_type] = {
                    "response": response,
                    "error": str(e),
                    "generation_time": generation_time,
                    "tokens_generated": len(outputs[0])
                }
                
        return results
    
    def benchmark_performance(
        self,
        contracts: List[Dict[str, Any]],
        entity_types: List[str]
    ) -> Dict[str, Any]:
        """
        Бенчмаркинг производительности модели.
        
        :param contracts: Список контрактов для тестирования
        :param entity_types: Список типов сущностей
        :return: Результаты бенчмарка
        """
        logger.info("Начинаем бенчмарк производительности...")
        
        total_time = 0
        total_tokens = 0
        total_examples = len(contracts)
        
        # Для каждого контракта извлекаем сущности
        for i, contract in enumerate(contracts):
            logger.debug(f"Обработка контракта {i+1}/{total_examples}")
            
            start_time = time.time()
            results = self.extract_entities(
                contract["contract_text"],
                entity_types
            )
            end_time = time.time()
            
            contract_time = end_time - start_time
            total_time += contract_time
            
            # Суммируем количество токенов
            for entity_result in results.values():
                total_tokens += entity_result.get("tokens_generated", 0)
            
            if i % 10 == 0:
                logger.info(f"Обработано {i+1} контрактов")
        
        avg_time_per_contract = total_time / total_examples
        avg_tokens_per_contract = total_tokens / total_examples
        
        performance_stats = {
            "total_contracts": total_examples,
            "total_time_seconds": total_time,
            "avg_time_per_contract_seconds": avg_time_per_contract,
            "total_tokens_generated": total_tokens,
            "avg_tokens_per_contract": avg_tokens_per_contract,
            "throughput_tokens_per_second": total_tokens / total_time if total_time > 0 else 0
        }
        
        logger.info("Бенчмарк завершен")
        return performance_stats

if __name__ == "__main__":
    # Пример использования
    extractor = LegalEntityExtractor()
    
    # Пример контракта
    sample_contract = {
        "contract_text": "В соответствии с условиями настоящего Договора, Компания А обязуется предоставить услуги по разработке программного обеспечения, а Компания Б обязуется оплатить эти услуги в размере 100 000 долларов США. Срок действия договора с 1 января 2024 года по 31 декабря 2024 года. Все споры будут разрешаться в суде г. Москвы.",
        "annotations": {
            "PERSON": ["Компания А", "Компания Б"],
            "ORG": ["Компания А", "Компания Б"],
            "MONEY": ["100 000 долларов США"],
            "DATE": ["1 января 2024 года", "31 декабря 2024 года"],
            "CONTRACT_TYPE": "Договор оказания услуг",
            "OBLIGATION": ["предоставить услуги по разработке программного обеспечения", "оплатить услуги"],
            "JURISDICTION": "суд г. Москвы"
        }
    }
    
    # Извлечение сущностей
    entity_types = ["PERSON", "ORG", "MONEY", "DATE", "CONTRACT_TYPE", "OBLIGATION", "JURISDICTION"]
    results = extractor.extract_entities(sample_contract["contract_text"], entity_types)
    
    print("Результаты извлечения:")
    for entity_type, result in results.items():
        print(f"{entity_type}: {result['response'][:200]}...")