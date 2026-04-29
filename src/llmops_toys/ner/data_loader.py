# src/data_loader.py
"""Загрузчик данных для юридических контрактов."""

import json
from pathlib import Path
from typing import List, Dict, Any
import logging
import random
from datasets import load_dataset

logger = logging.getLogger(__name__)

def load_cuad_dataset_safe(
    subset_size: int = 1000,
    save_path: Path = Path("data/ner/cuad_subset.json"),
) -> List[Dict[str, Any]]:
    """
    Безопасная загрузка датасета CUAD с резервным вариантом.
    
    :param subset_size: Размер подвыборки
    :param save_path: Путь для сохранения подвыборки
    :return: Список примеров из датасета
    """
    logger.info("Попытка загрузки датасета CUAD...")
    
    try:
        # Попытка загрузки через Hugging Face
        dataset = load_dataset("theatticusproject/cuad", split="train")
        
        # Создаем подвыборку
        subset = dataset.select(range(min(subset_size, len(dataset))))
        
        # Преобразуем в список словарей
        examples = []
        for item in subset:
            examples.append({
                "contract_text": item["contract_text"],
                "annotations": item["annotations"]
            })
        
        # Сохраняем подвыборку
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Успешно загружено {len(examples)} примеров из датасета CUAD")
        return examples
        
    except Exception as e:
        logger.warning(f"Ошибка при загрузке датасета CUAD: {e}")
        raise e

def load_subset_from_file(path: Path) -> List[Dict[str, Any]]:
    """
    Загружает подвыборку из файла.
    
    :param path: Путь к файлу с данными
    :return: Список примеров
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)