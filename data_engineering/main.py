from compare import ModelComparison


import json
import typing
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import logging
import time

from transformers import pipeline, Pipeline

import torch

from analyse import analyze_with_huggingface
from data_engineering import markup_data, clean_text
from prompting import analyze_text_with_prompts

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data() -> List[str]:
    """Загрузка сырых данных

    :return: Полученные от DE сырые данные
    """
    raw_reviews = [
        # Простые случаи
        "отличный iphone 14 PRO!!!  купил в магазине  apple на тверской 😊. Камера супер",
        "УЖАСНОЕ обслуживание в сбербанке на красной площади.. менеджер иван петров вобще не помог(",

        # Сарказм и ирония (сложно для классических моделей)
        "Спасибо огромное сотрудникам МТС за то что 3 часа держали меня в очереди! Просто восхитительно 👏",
        "Какой замечательный сервис в Пятерочке - касса сломалась прямо передо мной, а персонал даже не извинился",

        # Смешанные эмоции
        "iPhone 13 хороший телефон, но цена кусается. В целом доволен покупкой в re:Store",
        "Ресторан Белуга красивый и атмосфера приятная, но официант Максим был невнимателен",

        # Сложная структура предложений
        "Хотя Tesla Model Y и дорогая машина, и сервис в Рольф Премиум иногда подводит, но в целом я очень доволен покупкой",
        "Не могу сказать что отель Ритц-Карлтон плохой, просто ожидал большего за такие деньги",

        # Контекстно-зависимые случаи
        "Заказал доставку в Яндекс.Еде из ресторана Дача на Рублевке - привезли холодное, но курьер Андрей был вежливый",
        "MacBook Pro 16 работает как часы уже год, покупал в iStore на Арбате у консультанта Елены",

        # Неоднозначные случаи
        "Сходил в кинотеатр Октябрь посмотреть новый фильм Marvel - ну такое себе, но попкорн вкусный был",
        "Обслуживание в банке ВТБ на Тверской оставляет желать лучшего, хотя менеджер Ольга старалась помочь",

        # Сложные именованные сущности
        "Купил новый Samsung Galaxy S24 Ultra в DNS на Ленинском проспекте, консультант Дмитрий Иванович всё объяснил",
        "Ужинал в ресторане White Rabbit на Смоленской площади - шеф-повар Владимир Мухин превзошел ожидания",

        # Опечатки и сленг
        "норм телек LG купил в эльдорадо, продавец норм чел был, всё рассказал про функции"
    ]
    return raw_reviews


def load_models(sentiment_model: str, ner_model: str) ->  Tuple[Pipeline, Pipeline]:
    """"""
    try:
        sentiment_pipe = pipeline(
                    "sentiment-analysis", 
                    model=SENTIMENT_MODEL, 
                    device=0 if torch.cuda.is_available() else -1
                )

        ner_pipe = pipeline(
                "ner", 
                model=NER_MODEL, 
                aggregation_strategy="simple", 
                device=0 if torch.cuda.is_available() else -1
            )
        print(f"Модели {SENTIMENT_MODEL}, {NER_MODEL} успешно загружены")
    except Exception as e:
        print(f"Ошибка при загрузке моделей: {e}")
        raise e
    
    return sentiment_pipe, ner_pipe


def display_cleaning_and_markup_comparison(raw_reviews: List[str], cleaned_reviews: List[str], marked_up_reviews: pd.DataFrame) -> None:
    """Отображает сравнение исходных данных, очищенных данных и размеченных данных

    :param raw_reviews: Исходные тексты отзывов
    :param cleaned_reviews: Очищенные тексты отзывов
    :param marked_up_reviews: Размеченные данные (сущности, эмоции и т.д.) в формате DataFrame
    """
    print("\n" + "="*100)
    print("СРАВНЕНИЕ: БЫЛО vs СТАЛО")
    print("="*100)

    for i, (raw, cleaned) in enumerate(zip(raw_reviews, cleaned_reviews), 1):
        print(f"\n{i}. ИСХОДНЫЙ ТЕКСТ:")
        print(f"   {raw}")

        print(f"\n   ОЧИЩЕННЫЙ ТЕКСТ:")
        print(f"   {cleaned}")

        # Получаем размеченные данные для текущего отзыва
        if i <= len(marked_up_reviews):
            marked_up = marked_up_reviews.iloc[i-1]
            print(f"\n   РАЗМЕЧЕННЫЕ ДАННЫЕ:")

            # Выводим только ключевые поля разметки
            if 'entities' in marked_up and marked_up['entities']:
                entities = marked_up['entities']
                print("   СУЩНОСТИ:")
                for entity in entities:
                    if isinstance(entity, dict):
                        print(f"     - {entity.get('word', '')} ({entity.get('entity_group', '')})")
                    else:
                        print(f"     - {entity}")

            if 'sentiment' in marked_up and marked_up['sentiment']:
                sentiment = marked_up['sentiment']
                print("   ЭМОЦИОНАЛЬНАЯ ОЦЕНКА:")
                if isinstance(sentiment, dict):
                    print(f"     - Тональность: {sentiment.get('label', 'не определена')}")
                    print(f"     - Уверенность: {sentiment.get('score', 0):.3f}")
                else:
                    print(f"     - {sentiment}")
        else:
            print("   Нет данных для разметки")

        print("-" * 100)


if __name__ == "__main__":
    # Загрузка моделей HuggingFace
    SENTIMENT_MODEL = "ai-forever/ruBert-base"
    NER_MODEL = "Gherman/bert-base-NER-Russian"
    
    sentiment_model, ner_model = load_models(SENTIMENT_MODEL, NER_MODEL)

    # Анализ отзывов

    raw_reviews = load_raw_data()
    cleaned_text = [clean_text(text) for text in raw_reviews]

    df = pd.DataFrame({"review": cleaned_text})
    df = markup_data(df)

    display_cleaning_and_markup_comparison(raw_reviews, cleaned_text, df)

    hf_results = analyze_with_huggingface(sentiment_model, ner_model, cleaned_text)
    logger.info("Hugging face results:")
    for hf_result in hf_results:
        hf_result.display()

    llm_results = []
    for cleaned_review in cleaned_text:
        llm_results.append(analyze_text_with_prompts(cleaned_review))
    
    logger.info("LLM results:")
    for llm_result in llm_results:
        llm_result.display()

    # Сравнение моделей
    comparator = ModelComparison()
    results = comparator.compare_model_results(hf_results, llm_results, df)
    metrics = comparator.calculate_metrics(results)
    comparator.visualize_results(results, metrics)
    comparator.generate_report(results, metrics)