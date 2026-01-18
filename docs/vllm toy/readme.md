## Упражнения с vllm

Добавлен скрипт data_engineering/vllm_request.py, в рамках которого реализовано обращение к развёрнутой c помощью vllm локальной модели как при помощи requests библиотеки, так и с помощью openAI-совместимого API. Вычислены метрики получения ответа, которые (вместе с полным ответом) залогированы в mlflow.

Сервис mlflow развёрнут при помощи docker-compose (соответствующий файл приложен).

## Пример вывода

После запуска скрипта будет выведен следующий вывод:

```
uv run data_engineering/vllm_request.py 
{'id': 'chatcmpl-56991658291a4a49b26c48513f6b3eda', 'created': 1768563851, 'model': 'hosted_vllm/Qwen3-Coder-30B-A3B-Instruct-FP8', 'object': 'chat.completion', 'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'content': 'The capital of Germany is Berlin.', 'role': 'assistant'}, 'provider_specific_fields': {'stop_reason': None, 'token_ids': None}}], 'usage': {'completion_tokens': 8, 'prompt_tokens': 15, 'total_tokens': 23}}
INFO:httpx:HTTP Request: POST https://llm-api.vllm_local/v1/chat/completions "HTTP/1.1 200 OK"
🏃 View run resilient-jay-444 at: http://localhost:5000/#/experiments/372331494521009433/runs/68213cf8523b49f8b268116cb8f5753b
🧪 View experiment at: http://localhost:5000/#/experiments/372331494521009433
```

## Скриншоты mlflow


[Experiments](docs/vllm toy/screenshots/mlflow_exp.png)

[Run_details](docs/vllm toy/screenshots/mlflow_res.png)

[Metrics](docs/vllm toy/screenshots/mlflow_metrics.png)

Заметно, что обращение при помощи requests существенно быстрее.