from difflib import SequenceMatcher
from asyncio import run
import time
import json
from data_engineering.prompting import setup_openai_api, call_openai_api
from typing import Dict, Any, List
from datasets import load_dataset
import os
import requests
import mlflow
from infer.vllm_request import get_model_response
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu
from dataclasses import dataclass
from rouge_score.rouge_scorer import RougeScorer

@dataclass
class BenchmarkMetrics:
    bleu: float
    semantic_similarity: float
    rouge1: float
    rouge2: float
    rougeL: float
    f1_score: float
    judge_score: float
    judge_quality: str
    exact_match: float
    request_time: float


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


def f1_score(blue: float, rouge: float) -> float:
    """
    Вычисляет F1-оценку на основании условных precision и recall (blue и rouge соответственно)
    
    :param blue: Значение метрики BLUE
    :param rouge: Значение метрики ROUGE
    :return: F1-оценка
    """
    return 2 * (blue * rouge) / (blue + rouge) if blue + rouge >= 1e-8 else 0.0


def bleu_score(answer: str, reference: str) -> float:
    """
    Вычисляет BLEU-оценку между ответом и эталоном
    
    :param answer: Ответ модели
    :param reference: Эталонный ответ
    :return: BLEU-оценка
    """
    # Разбиваем строки на токены
    answer_tokens = answer.split()
    reference_tokens = reference.split()
    
    # Вычисляем BLEU-оценку
    bleu_score = sentence_bleu([reference_tokens], answer_tokens)
    return bleu_score


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




def run_evaluation(dataset_name: str, samples_count=10) -> Dict[str, List[BenchmarkMetrics]]:
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

    judge_client = setup_openai_api()
    qwen_client = OpenAI(base_url="http://vllm-qwen.llmops-toys.orb.local/v1", api_key="")
    deepseek_client = OpenAI(base_url="http://vllm-deepseek.llmops-toys.orb.local/v1", api_key="")
    
    rouge_scorer = RougeScorer(
        ["rouge1", "rouge2", "rougeL"], 
        use_stemmer=True
    )

    models = {"alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B" : deepseek_client, "Qwen/Qwen2.5-0.5B-Instruct" : qwen_client }

    model_names_short = { "alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B": "deepseek", "Qwen/Qwen2.5-0.5B-Instruct": "qwen" }
    scores = { model_name: [] for model_name in models.keys()}


    # Обработка первых samples_count примеров для демонстрации
    # for i in range(min(samples_count, len(test_data))):
    for i in [9, 14, 32, 33]:
        with mlflow.start_run() as run:
            item = test_data[i] 
            question = item["question"]
            context = item["context"]
            
            # Формируем полный вопрос с контекстом
            full_question = f"Контекст: {context}\n\nВопрос: {question}"
            
            print(f"Обработка примера {i+1}: {question[:50]}...")

            # Получение эталонного ответа
            reference_result = call_openai_api(prompt="Ответь на вопрос: {text}", text=full_question, client=judge_client, max_tokens=500)
            if isinstance(reference_result, dict) and "raw_response" in reference_result:
                reference_result = reference_result["raw_response"]

            mlflow.log_param(f"reference", reference_result)
            # Получает ответ и вычисляет метрики для каждой модели
            for model_name, client in models.items():
                start_time = time.time()
                model_result = call_openai_api(prompt="Ответь на вопрос на русском языке: {text}", text=full_question, client=client, model=model_name)
                end_time = time.time()

                score = {}
                if isinstance(model_result, dict) and "raw_response" in model_result:
                    model_result = model_result["raw_response"]
                if model_result and not isinstance(model_result, dict):
                    score = evaluate_answer_with_judge(question, model_result, judge_prompt)

                # scores.append(deepseek_score)

                print("\n\n📊 Метрики LLM-as-a-judge:")
                print(f"{model_name} score: {score}")

                rouge_metric = rouge_scorer.score(reference_result, model_result)
                print(f"\n\n📊 Метрики ROUGE: {rouge_metric}")

                bleu = bleu_score(model_result, reference_result)
                print(f"BLEU: {bleu:.4f}")

                f1_score_value = f1_score(bleu, rouge_metric["rouge1"].fmeasure)
                print(f"F1-score: {f1_score_value:.4f}")

                semantic_similarity = SequenceMatcher(None, model_result.lower(), reference_result.lower()).ratio()
                print(f"semantic_similarity: {semantic_similarity:.4f}")

                exact_match = float(model_result.strip() == reference_result.strip())
                print(f"exact_match: {exact_match:.1f}")

                metrics = BenchmarkMetrics(bleu=bleu,
                    rouge1=rouge_metric["rouge1"].fmeasure,
                    rouge2=rouge_metric["rouge2"].fmeasure,
                    rougeL=rouge_metric["rougeL"].fmeasure,
                    request_time=end_time - start_time,
                    semantic_similarity=semantic_similarity,
                    exact_match=exact_match,
                    f1_score=f1_score_value,
                    judge_score=score.get("score", -1),
                    judge_quality=score.get("explanation", "Cannot parsed judge desicion.")
                )

                model_name_short = model_names_short.get(model_name)
                mlflow.log_metrics({
                    f"{model_name_short}_bleu": bleu,
                    f"{model_name_short}_rouge1": metrics.rouge1,
                    f"{model_name_short}_rouge2": metrics.rouge2,
                    f"{model_name_short}_rougeL": metrics.rougeL,
                    f"{model_name_short}_request_time": metrics.request_time,
                    f"{model_name_short}_semantic_similarity": metrics.semantic_similarity,
                    f"{model_name_short}_exact_match": exact_match,
                    f"{model_name_short}_f1_score": f1_score_value,
                    f"{model_name_short}_judge_score": metrics.judge_score})
                
                mlflow.log_param(f"{model_name_short}_judge_quality", metrics.judge_quality)
                mlflow.log_param(f"{model_name_short}_result", model_result)

                scores[model_name].append(metrics)

    return scores

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment("benchmarks")

    dataset_name = "kuznetsoffandrey/sberquad"
    run_evaluation(dataset_name, samples_count=50)