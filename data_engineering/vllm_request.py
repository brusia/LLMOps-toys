from prompting import call_openai_api, setup_openai_api
import time
import os
import requests
from typing import Dict, Any
import mlflow

def get_model_response(question: str, model = "Qwen3-Coder-30B-A3B-Instruct-FP8") -> Dict[str, Any]:
    """
    Отправляет вопрос модели через HTTP-запрос и возвращает ответ
    
    :param question: Вопрос для модели
    :return: Словарь с ответом модели
    """
    # URL API модели (пример)
    api_key = os.getenv("OPENAI_API_KEY")
    api_url = f"{os.getenv('OPENAI_API_BASE')}/chat/completions"
    
    # Проверка наличия необходимых переменных окружения
    if not api_key:
        return {"error": "Переменная окружения OPENAI_API_KEY не установлена"}
    
    if not api_url:
        return {"error": "Переменная окружения OPENAI_API_BASE не установлена"}
    
    # Параметры запроса
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    # Заголовки запроса
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=30,
        )
        
        # Проверка статуса ответа
        response.raise_for_status()
        
        # Возврат JSON-ответа
        return response.json()
        
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP ошибка: {response.status_code} - {e}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Ошибка запроса: {e}"}
    except Exception as e:
        return {"error": f"Неожиданная ошибка: {e}"}

# Пример использования
if __name__ == "__main__":
    question = "What is the capital of Germany?"

    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Инициализация эксперимента
    mlflow.set_experiment("vllm_test_experiment")

    request_vllm_start_time = time.time()
    request_result = get_model_response(question)
    request_vllm_end_time = time.time()
    print(request_result)
    mlflow.log_metric("vllm_request_time", request_vllm_end_time - request_vllm_start_time)
    mlflow.log_param("vllm_responce", request_result)

    openai_vllm_start_time = time.time()
    openai_client = setup_openai_api()
    openai_result = call_openai_api(prompt="Ответь на вопрос: {text}", text=question, client=openai_client)
    openai_vllm_end_time = time.time()
    mlflow.log_metric("openai_request_time", openai_vllm_end_time - openai_vllm_start_time)
    mlflow.log_param("openai_responce", openai_result)
    