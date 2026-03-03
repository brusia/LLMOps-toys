# Benchmarks

Реализован трек оценки качества ответов на вопросы. В качестве датасета используется `sberbank-ai/sberquad` (открытый источник: huggingface)

## Изменения

Обновлён файл infer/metrics.py

В качестве эталонного ответа используется ответ Qwen3-Coder-30B-A3B-Instruct-FP8. Приводится сравнение с эталоном квантованных моделей alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B и Qwen/Qwen2.5-0.5B-Instruct.

docker-compose приложен.

## Запуск 

```bash
uv sync --group metrics --group mlflow
docker compose --env-file env_file --file docker-compose.mlflow-vllm.yaml up -d
uv run src/llmops_toys/infer/metrics.py
```

Результаты работы залогированы в mlflow (развёрнут локально из упомянутого compose-файла). Примеры экспериментов в mlflow:

[Experiment_list](bench_screens/experiments.png)
[Example1](bench_screens/example1.png)
[Example2](bench_screens/example2.png)
[Example3](bench_screens/example3.png)

Заметно, что квантованные модели плоховаты в вычленении основной информации. Также несмотря на прямой запрос выдавать ответ на русском языке они могут отвечать на английском. С увеличением количества параметров ситуация улучшается.

Кроме того видно, как небольшие модели начинают зацикливаться на повторении одного и того же при ответе на вопрос.

# Пример вывода модели

```
Обработка примера 14: Когда была получена лицензия на осуществление банк...
INFO:httpx:HTTP Request: POST https://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST http://vllm-deepseek.llmops-toys.orb.local/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"


📊 Метрики LLM-as-a-judge:
alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B score: {'score': 3, 'explanation': 'Ответ не отвечает на поставленный вопрос. Вопрос касается даты получения лицензии на осуществление банковских операций в иностранной валюте, а ответ посвящен созданию Банка Москвы и его участия вEstablishment of the stock exchange. Информация о лицензии отсутствует, и ответ содержит много лишних деталей, которые не относятся к теме вопроса.'}


📊 Метрики ROUGE: {'rouge1': Score(precision=0.003067484662576687, recall=0.5, fmeasure=0.006097560975609756), 'rouge2': Score(precision=0.0, recall=0.0, fmeasure=0.0), 'rougeL': Score(precision=0.003067484662576687, recall=0.5, fmeasure=0.006097560975609756)}
BLEU: 0.0000
F1-score: 0.0000
semantic_similarity: 0.0206
exact_match: 0.0
INFO:httpx:HTTP Request: POST http://vllm-qwen.llmops-toys.orb.local/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"


📊 Метрики LLM-as-a-judge:
Qwen/Qwen2.5-0.5B-Instruct score: {'score': 9, 'explanation': 'Ответ точный и содержит конкретную дату. Единственное улучшение — можно было бы указать, какой именно банк получил лицензию, чтобы сделать ответ ещё более информативным.'}


📊 Метрики ROUGE: {'rouge1': Score(precision=1.0, recall=1.0, fmeasure=1.0), 'rouge2': Score(precision=1.0, recall=1.0, fmeasure=1.0), 'rougeL': Score(precision=1.0, recall=1.0, fmeasure=1.0)}
BLEU: 0.7272
F1-score: 0.8421
semantic_similarity: 0.9806
exact_match: 0.0
🏃 View run overjoyed-cub-776 at: http://localhost:5000/#/experiments/710192957970364424/runs/5449c30c313544779487e39498433e47
🧪 View experiment at: http://localhost:5000/#/experiments/710192957970364424


Обработка примера 15: Когда было создано Акционерное общество открытого ...
INFO:httpx:HTTP Request: POST https://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST http://vllm-deepseek.llmops-toys.orb.local/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"


📊 Метрики LLM-as-a-judge:
alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B score: {'score': 3, 'explanation': "Ответ не соответствует вопросу и содержит множество ошибок и неуместной информации. Вопрос касается даты создания Акционерного общества открытого типа АКБ 'Московский кредитный банк', но в ответе упоминаются различные имена (Борис Борисов, В.А. Кухтаров) и даты, которые не связаны напрямую с созданием банка. Также присутствует бесполезная таблица с периодами, которая не добавляет ценности. Ответ не содержит корректной информации и не помогает в понимании вопроса."}


📊 Метрики ROUGE: {'rouge1': Score(precision=0.015873015873015872, recall=1.0, fmeasure=0.03125), 'rouge2': Score(precision=0.0, recall=0.0, fmeasure=0.0), 'rougeL': Score(precision=0.015873015873015872, recall=1.0, fmeasure=0.03125)}
BLEU: 0.0284
F1-score: 0.0298
semantic_similarity: 0.1966
exact_match: 0.0
INFO:httpx:HTTP Request: POST http://vllm-qwen.llmops-toys.orb.local/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"


📊 Метрики LLM-as-a-judge:
Qwen/Qwen2.5-0.5B-Instruct score: {'score': 6, 'explanation': 'Ответ содержит правильную дату создания банка (2004 год), но структура и формулировки некорректны и путаны. Упоминание о переходе к названию \'АО "МоскваНК"\' выглядит необоснованным и может быть ошибочным. Также есть повторяющиеся слова и лишние конструкции, снижающие ясность ответа.'}


📊 Метрики ROUGE: {'rouge1': Score(precision=0.0, recall=0.0, fmeasure=0.0), 'rouge2': Score(precision=0.0, recall=0.0, fmeasure=0.0), 'rougeL': Score(precision=0.0, recall=0.0, fmeasure=0.0)}
BLEU: 0.0000
F1-score: 0.0000
semantic_similarity: 0.5260
exact_match: 0.0
🏃 View run adventurous-fly-719 at: http://localhost:5000/#/experiments/710192957970364424/runs/3ea291ef62184e80ad6b2390a6d63ff6
🧪 View experiment at: http://localhost:5000/#/experiments/710192957970364424
Обработка примера 16: Кто возглавил совет директоров банка?...
INFO:httpx:HTTP Request: POST https://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST http://vllm-deepseek.llmops-toys.orb.local/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"


📊 Метрики LLM-as-a-judge:
alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B score: {'score': 2, 'explanation': 'Ответ не содержит корректной информации о том, кто возглавил совет директоров банка. Он представляет собой набор повторяющихся и бессмысленных данных, не отвечающих на поставленный вопрос. Ответ явно некачественный и непонятный.'}


📊 Метрики ROUGE: {'rouge1': Score(precision=0.0, recall=0.0, fmeasure=0.0), 'rouge2': Score(precision=0.0, recall=0.0, fmeasure=0.0), 'rougeL': Score(precision=0, recall=0, fmeasure=0)}
BLEU: 0.0000
F1-score: 0.0000
semantic_similarity: 0.0578
exact_match: 0.0
INFO:httpx:HTTP Request: POST http://vllm-qwen.llmops-toys.orb.local/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"
```