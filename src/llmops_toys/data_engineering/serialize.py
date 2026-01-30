# de/serialize.py
import json
import csv
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingExample:
    """Пример для тренировки модели"""
    text: str
    sentiment: str
    entities: List[str]
    label: int  # 1 для положительной тональности, -1 для отрицательной, 0 для нейтральной

def save_to_jsonl(examples: List[TrainingExample], filepath: Path) -> None:
    """
    Сохранение данных в формате JSONL для OpenAI fine-tuning
    
    :param examples: Список примеров для тренировки
    :param filepath: Путь к файлу для сохранения
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for example in examples:
            # Создаем сообщения для OpenAI fine-tuning
            messages = [
                {
                    "role": "system",
                    "content": "Вы - эксперт по анализу текстов. Ваша задача - определить тональность текста и извлечь именованные сущности."
                },
                {
                    "role": "user",
                    "content": f"Проанализируйте тональность следующего текста на русском языке:\n\n{example.text}"
                },
                {
                    "role": "assistant",
                    "content": f"Тональность: {example.sentiment}\nСущности: {', '.join(example.entities) if example.entities else 'Нет сущностей'}"
                }
            ]
            
            # Формируем запись в формате JSONL
            record = {
                "messages": messages
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def save_to_csv(examples: List[TrainingExample], filepath: Path) -> None:
    """
    Сохранение данных в формате CSV для общего использования
    
    :param examples: Список примеров для тренировки
    :param filepath: Путь к файлу для сохранения
    """
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'sentiment', 'entities', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for example in examples:
            writer.writerow({
                'text': example.text,
                'sentiment': example.sentiment,
                'entities': ', '.join(example.entities) if example.entities else '',
                'label': example.label
            })

def load_from_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """
    Загрузка данных из файла JSONL
    
    :param filepath: Путь к файлу JSONL
    :return: Список записей
    """
    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def load_from_csv(filepath: Path) -> List[Dict[str, Any]]:
    """
    Загрузка данных из файла CSV
    
    :param filepath: Путь к файлу CSV
    :return: Список записей
    """
    examples = []
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            examples.append(row)
    return examples

def convert_training_data(raw_data: List[str], sentiments: List[int], entities_list: List[List[str]]) -> List[TrainingExample]:
    """
    Конвертация сырых данных в формат для тренировки
    
    :param raw_data: Список текстов
    :param sentiments: Список меток тональности (-1, 0, 1)
    :param entities_list: Список списков сущностей
    :return: Список объектов TrainingExample
    """
    examples = []
    
    # Преобразуем метки тональности в строки
    sentiment_map = {-1: "NEGATIVE", 0: "NEUTRAL", 1: "POSITIVE"}
    
    for i, text in enumerate(raw_data):
        sentiment = sentiment_map.get(sentiments[i], "NEUTRAL")
        entities = entities_list[i] if i < len(entities_list) else []
        
        example = TrainingExample(
            text=text,
            sentiment=sentiment,
            entities=entities,
            label=sentiments[i]
        )
        examples.append(example)
    
    return examples

def validate_jsonl_format(examples: List[Dict[str, Any]]) -> bool:
    """
    Проверка корректности формата JSONL для OpenAI
    
    :param examples: Список записей JSONL
    :return: True если формат корректен
    """
    for example in examples:
        if 'messages' not in example:
            return False
        
        messages = example['messages']
        if not isinstance(messages, list):
            return False
            
        # Проверяем, что есть системное и пользовательское сообщение
        roles = [msg['role'] for msg in messages]
        if 'system' not in roles or 'user' not in roles:
            return False
            
    return True

def validate_csv_format(examples: List[Dict[str, Any]]) -> bool:
    """
    Проверка корректности формата CSV
    
    :param examples: Список записей CSV
    :return: True если формат корректен
    """
    required_fields = ['text', 'sentiment', 'entities', 'label']
    
    for example in examples:
        for field in required_fields:
            if field not in example:
                return False
                
    return True

# Пример использования
if __name__ == "__main__":
    # Пример данных для тренировки
    raw_texts = [
        "отличный iphone 14 PRO!!!  купил в магазине  apple на тверской 😊. Камера супер",
        "УЖАСНОЕ обслуживание в сбербанке на красной площади.. менеджер иван петров вобще не помог(",
        "Спасибо огромное сотрудникам МТС за то что 3 часа держали меня в очереди! Просто восхитительно 👏"
    ]
    
    sentiments = [1, -1, -1]
    entities = [
        ["PRODUCT", "ORGANIZATION", "LOCATION"],
        ["ORGANIZATION", "PERSON", "LOCATION"],
        ["ORGANIZATION", "PERSON"]
    ]
    
    # Конвертируем данные
    training_examples = convert_training_data(raw_texts, sentiments, entities)
    
    # Сохраняем в JSONL формате
    save_to_jsonl(training_examples, Path("data/training_data.jsonl"))
    print("Данные сохранены в формате JSONL")
    
    # Сохраняем в CSV формате
    save_to_csv(training_examples, Path("data/training_data.csv"))
    print("Данные сохранены в формате CSV")
    
    # Загружаем и проверяем JSONL
    loaded_jsonl = load_from_jsonl(Path("data/training_data.jsonl"))
    print(f"Загружено {len(loaded_jsonl)} записей из JSONL")
    print(f"Формат JSONL корректен: {validate_jsonl_format(loaded_jsonl)}")
    
    # Загружаем и проверяем CSV
    loaded_csv = load_from_csv(Path("data/training_data.csv"))
    print(f"Загружено {len(loaded_csv)} записей из CSV")
    print(f"Формат CSV корректен: {validate_csv_format(loaded_csv)}")
