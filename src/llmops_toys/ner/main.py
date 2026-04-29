# src/main.py
"""Основной скрипт для выполнения задачи по треку A."""

import argparse
import logging
from pathlib import Path
import torch
from typing import List, Dict, Any
import time
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_dataset(size: int = 1000) -> List[Dict[str, Any]]:
    """Создает образец набора данных для демонстрации."""
    sample_contracts = [
        {
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
        },
        {
            "contract_text": "Поставщик обязуется поставлять товары согласно спецификации, сроки поставки - до 30 дней после подписания контракта. Сумма оплаты составляет 50 000 евро. Споры решаются в арбитражном суде г. Санкт-Петербурга.",
            "annotations": {
                "PERSON": [],
                "ORG": ["Поставщик"],
                "MONEY": ["50 000 евро"],
                "DATE": ["до 30 дней после подписания контракта"],
                "CONTRACT_TYPE": "Договор поставки",
                "OBLIGATION": ["поставить товары согласно спецификации", "обеспечить сроки поставки"],
                "JURISDICTION": "арбитражный суд г. Санкт-Петербурга"
            }
        },
        {
            "contract_text": "Клиент обязуется оплатить услуги по разработке мобильного приложения в размере 250 000 рублей в течение 10 рабочих дней после получения счета. Срок выполнения работ - 3 месяца. Все споры разрешаются в суде г. Екатеринбурга.",
            "annotations": {
                "PERSON": [],
                "ORG": ["Клиент"],
                "MONEY": ["250 000 рублей"],
                "DATE": ["в течение 10 рабочих дней", "3 месяца"],
                "CONTRACT_TYPE": "Договор на оказание услуг",
                "OBLIGATION": ["оплатить услуги", "выполнить работы"],
                "JURISDICTION": "суд г. Екатеринбурга"
            }
        },
        {
            "contract_text": "Франчайзи обязуется соблюдать стандарты качества и маркетинговые требования franchisor'a. Сумма франшизы составляет 150 000 долларов США. Срок действия франчайзингового соглашения - 5 лет. Споры решаются в суде г. Нью-Йорка.",
            "annotations": {
                "PERSON": [],
                "ORG": ["franchisor", "Франчайзи"],
                "MONEY": ["150 000 долларов США"],
                "DATE": ["5 лет"],
                "CONTRACT_TYPE": "Франчайзинговое соглашение",
                "OBLIGATION": ["соблюдать стандарты качества", "соблюдать маркетинговые требования"],
                "JURISDICTION": "суд г. Нью-Йорка"
            }
        },
        {
            "contract_text": "Агент обязуется представлять интересы клиента при заключении сделок с поставщиками. Комиссионное вознаграждение составляет 5% от объема сделок. Срок действия агентского договора - 1 год. Все споры разрешаются в суде г. Казани.",
            "annotations": {
                "PERSON": [],
                "ORG": ["Агент", "клиент"],
                "MONEY": ["5% от объема сделок"],
                "DATE": ["1 год"],
                "CONTRACT_TYPE": "Агентский договор",
                "OBLIGATION": ["представлять интересы клиента", "заключать сделки с поставщиками"],
                "JURISDICTION": "суд г. Казани"
            }
        }
    ]
    
    # Расширяем набор случайными контрактами
    extended_contracts = []
    for i in range(size):
        # Выбираем случайный образец
        contract = sample_contracts[i % len(sample_contracts)]
        # Создаем уникальный контракт
        unique_contract = contract.copy()
        unique_contract["contract_text"] = contract["contract_text"].replace("Компания А", f"Компания {chr(65 + (i % 26))}")
        unique_contract["contract_text"] = unique_contract["contract_text"].replace("Компания Б", f"Компания {chr(66 + (i % 26))}")
        extended_contracts.append(unique_contract)
    
    return extended_contracts

def load_model(model_name: str, device: str = "cpu"):
    """
    Загружает модель с проверкой доступности.
    
    :param model_name: Название модели
    :param device: Устройство для запуска модели
    :return: Загруженная модель или None
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"Загрузка модели {model_name} на устройство {device}")
        
        # Проверяем доступность модели
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager"  # Для совместимости с Phi-3
        ).to(device)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("Модель успешно загружена")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {model_name}: {e}")
        return None, None

def build_prompt(entity_type: str, contract_text: str) -> str:
    """
    Строит промпт для извлечения конкретного типа сущности.
    
    :param entity_type: Тип сущности
    :param contract_text: Текст контракта
    :return: Сформированный промпт
    """
    prompts = {
        "PERSON": (
            "Извлеките все упоминания лиц (PERSON) из следующего юридического контракта:\n\n"
            "{contract_text}\n\n"
            "Ответьте в формате JSON с массивом объектов, где каждый объект содержит:\n"
            "- 'text': текст упоминания лица\n"
            "- 'start': позиция начала в тексте\n"
            "- 'end': позиция окончания в тексте"
        ),
        
        "ORG": (
            "Извлеките все упоминания организаций (ORG) из следующего юридического контракта:\n\n"
            "{contract_text}\n\n"
            "Ответьте в формате JSON с массивом объектов, где каждый объект содержит:\n"
            "- 'text': текст упоминания организации\n"
            "- 'start': позиция начала в тексте\n"
            "- 'end': позиция окончания в тексте"
        ),
        
        "MONEY": (
            "Извлеките все упоминания сумм денег (MONEY) из следующего юридического контракта:\n\n"
            "{contract_text}\n\n"
            "Ответьте в формате JSON с массивом объектов, где каждый объект содержит:\n"
            "- 'text': текст упоминания суммы\n"
            "- 'start': позиция начала в тексте\n"
            "- 'end': позиция окончания в тексте"
        ),
        
        "DATE": (
            "Извлеките все упоминания дат (DATE) из следующего юридического контракта:\n\n"
            "{contract_text}\n\n"
            "Ответьте в формате JSON с массивом объектов, где каждый объект содержит:\n"
            "- 'text': текст упоминания даты\n"
            "- 'start': позиция начала в тексте\n"
            "- 'end': позиция окончания в тексте"
        ),
        
        "CONTRACT_TYPE": (
            "Извлеките тип контракта (CONTRACT_TYPE) из следующего юридического контракта:\n\n"
            "{contract_text}\n\n"
            "Ответьте в формате JSON с ключом 'contract_type' и значением типа контракта."
        ),
        
        "OBLIGATION": (
            "Извлеките все обязательства (OBLIGATION) из следующего юридического контракта:\n\n"
            "{contract_text}\n\n"
            "Ответьте в формате JSON с массивом объектов, где каждый объект содержит:\n"
            "- 'text': текст обязательства\n"
            "- 'start': позиция начала в тексте\n"
            "- 'end': позиция окончания в тексте"
        ),
        
        "JURISDICTION": (
            "Извлеките информацию о юрисдикции (JURISDICTION) из следующего юридического контракта:\n\n"
            "{contract_text}\n\n"
            "Ответьте в формате JSON с ключом 'jurisdiction' и значением юрисдикции."
        )
    }
    
    prompt_template = prompts.get(entity_type, "")
    return prompt_template.format(contract_text=contract_text)

@torch.no_grad()
def extract_entities(model, tokenizer, contract_text: str, entity_types: List[str], device: str = "cpu") -> Dict[str, Any]:
    """
    Извлекает сущности из текста контракта с использованием модели.
    
    :param model: Загруженная модель
    :param tokenizer: Токенизатор
    :param contract_text: Текст контракта
    :param entity_types: Список типов сущностей для извлечения
    :param device: Устройство для запуска модели
    :return: Результаты извлечения
    """
    results = {}
    
    for entity_type in entity_types:
        logger.debug(f"Извлечение сущностей типа {entity_type}")
        
        # Формируем промпт
        prompt = build_prompt(entity_type, contract_text)
        
        # Токенизация
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(device)
        
        # Генерация ответа с правильными параметрами
        start_time = time.time()
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=1,
                early_stopping=True
            )
        except Exception as e:
            logger.warning(f"Ошибка при генерации для {entity_type}: {e}")
            # Используем упрощенный подход
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        generation_time = time.time() - start_time
        
        # Декодирование результата
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Имитация парсинга результата
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

def benchmark_performance(model, tokenizer, contracts: List[Dict[str, Any]], entity_types: List[str], device: str = "cpu") -> Dict[str, Any]:
    """
    Бенчмаркинг производительности модели.
    
    :param model: Загруженная модель
    :param tokenizer: Токенизатор
    :param contracts: Список контрактов для тестирования
    :param entity_types: Список типов сущностей
    :param device: Устройство для запуска модели
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
        results = extract_entities(model, tokenizer, contract["contract_text"], entity_types, device)
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

def main(args):
    """Главная функция."""
    logger.info("Начинаем выполнение задачи по извлечению именованных сущностей")
    
    # Шаг 1: Подготовка данных
    logger.info("1. Подготовка данных")
    
    # Создаем подвыборку данных
    contracts = create_sample_dataset(args.dataset_size)
    logger.info(f"Создано {len(contracts)} примеров контрактов")
    
    # Шаг 2: Инициализация модели
    logger.info("2. Инициализация модели")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используем устройство: {device}")
    
    model, tokenizer = load_model(args.model_name, device)
    
    if model is None:
        logger.error("Не удалось загрузить модель. Программа завершена.")
        return
    
    # Шаг 3: Извлечение сущностей
    logger.info("3. Извлечение сущностей")
    
    entity_types = ["PERSON", "ORG", "MONEY", "DATE", "CONTRACT_TYPE", "OBLIGATION", "JURISDICTION"]
    
    # Для демонстрации используем только первые несколько контрактов
    test_contracts = contracts[:min(args.demo_size, len(contracts))]
    
    # Используем реальную модель
    stats = benchmark_performance(model, tokenizer, test_contracts, entity_types, device)
    
    logger.info(f"Извлечены сущности из {stats['total_contracts']} контрактов")
    logger.info(f"Пропускная способность: {stats['throughput_tokens_per_second']:.2f} токенов/сек")
    
    # Шаг 4: Сохранение результатов
    logger.info("4. Сохранение результатов")
    
   # Сохраняем результаты в файл
    output_path = Path(f"data/ner/{Path(args.model_name).name}/entity_extraction_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_data = {
        "model_used": args.model_name,
        "entities_extracted": entity_types,
        "performance_metrics": {
            "processing_time_seconds": stats["total_time_seconds"],
            "throughput_tokens_per_second": stats["throughput_tokens_per_second"],
            "total_contracts_processed": stats["total_contracts"],
            "total_tokens_processed": stats["total_tokens_generated"]
        },
        "sample_results": []  # В реальной реализации здесь будут результаты извлечения
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Результаты сохранены в {output_path}")
    
    # Шаг 5: Генерация отчета
    logger.info("5. Генерация отчета")
    
    report = []
    report.append("=" * 60)
    report.append("ОТЧЕТ ПО ВЫПОЛНЕНИЮ ЗАДАЧИ ИЗВЛЕЧЕНИЯ СУЩНОСТЕЙ")
    report.append("=" * 60)
    report.append(f"Модель: {args.model_name}")
    report.append(f"Извлеченные сущности: {', '.join(entity_types)}")
    report.append(f"Обработано контрактов: {stats['total_contracts']}")
    report.append(f"Общее время обработки: {stats['total_time_seconds']:.2f} секунд")
    report.append(f"Пропускная способность: {stats['throughput_tokens_per_second']:.2f} токенов/сек")
    report.append(f"Обработано токенов: {stats['total_tokens_generated']}")
    report.append("=" * 60)
    
    logger.info("Генерация отчета завершена")
    print("\n".join(report))
    
    logger.info("Выполнение задачи завершено")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Извлечение сущностей из юридических документов")
    
    parser.add_argument("--model-name",
                        type=str,
                        default="Qwen/Qwen2-0.5B",
                        help="Модель для NER",
                        )
                        
    parser.add_argument("--dataset-size", 
                       type=int, 
                       default=100,
                       help="Размер подвыборки из датасета",
                       )
    
    parser.add_argument("--demo-size", 
                       type=int, 
                       default=10,
                       help="Размер демонстрационной выборки",
                       )
    
    parser.add_argument("--batch-size", 
                       type=int, 
                       default=5,
                       help="Размер пакета для batch processing",
                       )
    
    args = parser.parse_args()
    
    main(args)