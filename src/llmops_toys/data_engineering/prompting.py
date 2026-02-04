from data_engineering.analyse import AnalysisResult
import os
import time
from typing import Dict, Any, List
import openai
from openai import OpenAI
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prompts_for_llm() -> dict[str, str]:
    """
    Создание базовых промптов для разных задач (один промпт на задачу)
    
    :return: Словарь с промптами для NER и sentiment analysis
    """
    prompts = {
        "ner": (
            "Извлеките все именованные сущности из следующего текста на русском языке.\n\n"
            "Текст: {text}\n\n"
            "Ответ должен быть в формате JSON со списком сущностей:\n"
            "{{\n"
            '  "entities": ["сущность1", "сущность2", ...]\n'
            "}}"
        ),
        "sentiment": (
            "Проанализируйте тональность следующего текста на русском языке.\n\n"
            "Текст: {text}\n\n"
            "Ответ должен быть в формате JSON с одной из следующих тональностей:\n"
            "{{\n"
            '  "sentiment": "POSITIVE" или "NEGATIVE" или "NEUTRAL"\n'
            "}}"
        )
    }
    return prompts


def test_prompts():
    """Тестирование созданных промптов"""
    prompts = create_prompts_for_llm()
    
    print("Промпт для NER:")
    print(prompts["ner"])
    print("\n" + "="*50 + "\n")
    
    print("Промпт для sentiment analysis:")
    print(prompts["sentiment"])


def setup_openai_api():
    """Настройка OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    if not api_key:
        raise ValueError("Необходимо установить переменную окружения OPENAI_API_KEY")

    if not api_base:
        raise ValueError("Необходимо установить переменную окружения OPENAI_API_BASE")
    
    client = OpenAI(api_key=api_key, base_url=api_base)
    return client


def call_openai_api(client: OpenAI, prompt: str, text: str, model: str = "Qwen3-Coder-30B-A3B-Instruct-FP8", temperature=0.7, max_tokens=500) -> Dict[str, Any]:
    """
    Вызов OpenAI API с промптом
    
    :param client: Клиент OpenAI
    :param prompt: Промпт для модели
    :param text: Текст для анализа
    :param model: Модель для вызова
    :return: Ответ от API
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Вы - помощник по анализу текста."},
                {"role": "user", "content": str.format(prompt, text=text)}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        content = response.choices[0].message.content.strip()
        # Попытка парсинга JSON
        try:
            if content.startswith('```json'):
                content = content[7:]  # Убираем ```json
            if content.endswith('```'):
                content = content[:-3]  # Убираем ```

            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            # Если не удалось распарсить как JSON, возвращаем текст
            return {"raw_response": content}
            
    except Exception as e:
        logger.exception("Ошибка при вызове OpenAI API")


def analyze_text_with_prompts(text: str) -> AnalysisResult:
    """
    Анализ текста с использованием созданных промптов
    
    :param text: Текст для анализа
    :return: Результаты анализа
    """
    # Получаем промпты
    prompts = create_prompts_for_llm()
    
    # Настройка API
    client = setup_openai_api()
    
    results = {}
    
    # Анализ тональности
    start_time = time.time()
    sentiment_result = call_openai_api(client, prompts["sentiment"], text)["sentiment"].lower()
    sentiment_time = time.time() - start_time
    
    # Анализ NER
    start_time = time.time()
    ner_result = call_openai_api(client, prompts["ner"], text)
    ner_time = time.time() - start_time
    
    results = AnalysisResult(text=text,
                             sentiment=sentiment_result,
                             entities=ner_result,
                             ner_time=ner_time,
                             sentiment_time=sentiment_time)
    
    return results