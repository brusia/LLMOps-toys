from asyncio import run
import time
import json
from data_engineering.prompting import setup_openai_api, call_openai_api
from typing import Dict, Any
from datasets import load_dataset
import os
import requests
import mlflow
from infer.vllm_request import get_model_response

def evaluate_answer_with_judge(question: str, answer: str, judge_prompt: str) -> Dict[str, Any]:
    """
    Оценивает качество ответа с помощью LLM as a judge
    
    :param question: Вопрос
    :param answer: Ответ для оценки
    :param judge_prompt: Шаблон промпта для judge
    :return: Результат оценки
    """
    try:
        # Подготовка промпта для judge
        prompt = str.format(judge_prompt, question=question, answer=answer)
        
        # Используем OpenAI API для оценки
        openai_client = setup_openai_api()
        return call_openai_api(prompt=prompt, text="", client=openai_client)

    except Exception as e:
        return {"error": f"{e}"}


def run_evaluation(dataset_name: str, samples_count=10):
    """
    Запуск оценки моделей на датасете sberbank-ai/sberquad
    """
    # Загрузка датасета
    dataset = load_dataset(dataset_name)
    
    # Используем тестовую часть датасета
    test_data = dataset["test"]
    
    # Промпт для LLM as a judge
    judge_prompt = """Оцени качество следующего ответа на вопрос по шкале от 1 до 10:
Вопрос: {question}
Ответ: {answer}
Ответь только в формате JSON с полем "score" и пояснением "explanation".
Например: {{{{\"score\": 8, \"explanation\": \"Ответ точный и полезный\"}}}}"""
    
    # Списки для хранения результатов
    vllm_scores = []
    openai_scores = []
    
    # Обработка первых 10 примеров для демонстрации
    for i in range(min(samples_count, len(test_data))):
        with mlflow.start_run() as run:
            item = test_data[i] 
            question = item["question"]
            context = item["context"]
            
            # Формируем полный вопрос с контекстом
            full_question = f"Контекст: {context}\n\nВопрос: {question}"
            
            print(f"Обработка примера {i+1}: {question[:50]}...")
            
            # Получение ответов от обеих моделей
            vllm_start = time.time()
            vllm_result = get_model_response(full_question)
            vllm_end = time.time()
            
            openai_start = time.time()
            openai_client = setup_openai_api()
            openai_result = call_openai_api(prompt="Ответь на вопрос: {text}", text=full_question, client=openai_client)
            openai_end = time.time()

            # Оценка ответов
            vllm_score = {}
            if "error" not in vllm_result:
                vllm_score = evaluate_answer_with_judge(question, vllm_result.get("choices", [{}])[0].get("message", {}).get("content", ""), judge_prompt)
                vllm_scores.append(vllm_score)
                print(f"VLLM score: {vllm_score}")
            
            if isinstance(openai_result, dict) and "raw_response" in openai_result:
                openai_result = openai_result["raw_response"]
            if openai_result and not isinstance(openai_result, dict):
                openai_score = evaluate_answer_with_judge(question, openai_result, judge_prompt)
                openai_scores.append(openai_score)
                print(f"OpenAI score: {openai_score}")
            
            # Логирование в MLflow
            mlflow.log_metric("vllm_request_time", vllm_end - vllm_start)
            mlflow.log_metric("openai_request_time", openai_end - openai_start)

            if "score" in vllm_score:
                mlflow.log_metric("vllm_answer_score", vllm_score["score"])
                mlflow.log_param("vllm_answer_quality", vllm_score["explanation"])
                print(f"VLLM score: {vllm_score['score']}")
            else:
                print(f"VLLM score parsing error: {vllm_score}")
            
            openai_score = {}
            if openai_result and not isinstance(openai_result, dict):
                openai_score = evaluate_answer_with_judge(question, openai_result, judge_prompt)
                openai_scores.append(openai_score)
                
            if "score" in openai_score:
                mlflow.log_metric("openai_answer_score", openai_score["score"])
                mlflow.log_param("openai_answer_quality", openai_score["explanation"])
                print(f"OpenAI score: {openai_score['score']}")
            else:
                print(f"OpenAI score parsing error: {openai_score}")
    
    # Подведение итогов
    with mlflow.start_run(run_name=f"total_{min(samples_count, len(test_data))}") as run:
        if vllm_scores:
            avg_vllm_score = sum([s.get("score", 0) for s in vllm_scores if "score" in s]) / len(vllm_scores)
            mlflow.log_metric("avg_vllm_score", avg_vllm_score)
            print(f"Средний балл VLLM: {avg_vllm_score}")
        
        if openai_scores:
            avg_openai_score = sum([s.get("score", 0) for s in openai_scores if "score" in s]) / len(openai_scores)
            mlflow.log_metric("avg_openai_score", avg_openai_score)
            print(f"Средний балл OpenAI: {avg_openai_score}")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment("llm_as_a_judge")

    dataset_name = "kuznetsoffandrey/sberquad"
    run_evaluation(dataset_name, samples_count=50)