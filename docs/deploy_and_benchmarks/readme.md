## Упражнения с vllm

Добавлен скрипт infer/vllm_request.py, в рамках которого реализовано обращение к развёрнутой c помощью vllm локальной модели как при помощи requests библиотеки, так и с помощью openAI-совместимого API. Вычислены метрики получения ответа, которые (вместе с полным ответом) залогированы в mlflow.

Сервис mlflow развёрнут при помощи docker-compose (соответствующий файл приложен).

## Пример вывода

После запуска скрипта будет выведен следующий вывод:

```
uv run src/llmops_toys/infer/vllm_request.py 
{'id': 'chatcmpl-56991658291a4a49b26c48513f6b3eda', 'created': 1768563851, 'model': 'hosted_vllm/Qwen3-Coder-30B-A3B-Instruct-FP8', 'object': 'chat.completion', 'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'content': 'The capital of Germany is Berlin.', 'role': 'assistant'}, 'provider_specific_fields': {'stop_reason': None, 'token_ids': None}}], 'usage': {'completion_tokens': 8, 'prompt_tokens': 15, 'total_tokens': 23}}
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
🏃 View run resilient-jay-444 at: http://localhost:5000/#/experiments/372331494521009433/runs/68213cf8523b49f8b268116cb8f5753b
🧪 View experiment at: http://localhost:5000/#/experiments/372331494521009433
```

## Скриншоты mlflow


[Experiments](screenshots/mlflow_exp.png)

[Run_details](screenshots/mlflow_res.png)

[Metrics](screenshots/mlflow_metrics.png)

Заметно, что обращение при помощи requests существенно быстрее.

Добавлена также оценка модели LLM-as-a-judge для оценки качества ответов. Для этого добавлен скрипт infer/metrics.py, который оценивает ответы на основе датасета kuznetsoffandrey/sberquad, а также логирует результаты в mlflow.
Сревнение выполнено на одной и той же модели, с разным контекстным окном и температурой.

[Example_1](screenshots/mlflow_example_1.png)

[Example_2](screenshots/mlflow_example_2.png)

[Total_50](screenshots/mlflow_average.png)

Видно, что с меньшей длиной контекста наблюдается незначительное проседание в точности.

## Пример вывода модели

```
Обработка примера 24: Что затрудняет его использование в обучении в каче...
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
VLLM score: {'score': 7, 'explanation': 'Ответ содержит основную идею, что высокий порог вхождения затрудняет использование языка как первого в обучении. Однако он немного поверхностен и не раскрывает конкретные аспекты, делающие язык сложным для новичков, такие как синтаксис, семантика или отсутствие удобных инструментов для обучения.'}
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
OpenAI score: {'score': 6, 'explanation': 'Ответ содержит верную идею о высоком пороге вхождения, но слишком краткий и не раскрывает конкретные аспекты, которые затрудняют изучение языка. Не хватает примеров или пояснений.'}
VLLM score: 7
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
OpenAI score: 6
🏃 View run nimble-stag-498 at: http://localhost:5000/#/experiments/757769994549227305/runs/2cba5abc10154bc2a3ed3f6b890577ad
🧪 View experiment at: http://localhost:5000/#/experiments/757769994549227305


Обработка примера 25: Что в языке Си тяжело использовать?...
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
VLLM score: {'score': 6, 'explanation': 'Ответ содержит некоторые верные моменты, такие как сложность обучения и отсутствие встроенной поддержки современных парадигм, но он несколько преувеличивает проблемы языка C. Например, C поддерживает современные практики через библиотеки и стандарты, а также может быть безопасным при правильном использовании. Также в ответе есть стилистические и логические недочеты, например, утверждение о том, что C не поддерживает ООП, хотя это не совсем так. Общая оценка — средняя.'}
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
OpenAI score: {'score': 9, 'explanation': 'Ответ хорошо структурирован, содержит актуальные и важные аспекты, которые действительно могут вызывать сложности при использовании языка Си. Пояснения логичны и соответствуют реальности. Единственное улучшение — можно было бы добавить краткий пример или контекст для некоторых пунктов, чтобы сделать ответ ещё более информативным.'}
VLLM score: 6
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
OpenAI score: 9
🏃 View run sincere-lark-25 at: http://localhost:5000/#/experiments/757769994549227305/runs/32e4521cf2c4434d83c3b6ff8f5d8de3
🧪 View experiment at: http://localhost:5000/#/experiments/757769994549227305


Обработка примера 26: Где впервые появился язык Си?...
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
VLLM score: {'score': 9, 'explanation': "Ответ содержит точную информацию о месте и времени создания языка C, указывает на ключевых разработчиков и контекст создания. Небольшое уточнение касательно 'культурного контекста' может быть избыточным, но не является ошибкой. Общая структура и содержание высокого качества."}
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
OpenAI score: {'score': 6, 'explanation': 'Ответ показывает понимание того, что в предоставленном контексте нет информации о месте появления языка Си, что является корректным замечанием. Однако он не даёт точной информации о фактическом месте и времени создания языка Си, которое известно и может быть указано (Bell Labs, 1970-е годы). Ответ можно считать частично правильным, но недостаточно информативным для полного понимания вопроса.'}
VLLM score: 9
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
OpenAI score: 7
🏃 View run dashing-ray-418 at: http://localhost:5000/#/experiments/757769994549227305/runs/47a7fdd4e3a049438eb6c3b00c6b3869
🧪 View experiment at: http://localhost:5000/#/experiments/757769994549227305


Обработка примера 27: Чего не лишён этот язык?...
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
VLLM score: {'score': 10, 'explanation': 'Ответ полностью соответствует логике и содержанию текста. Указано, что язык не лишен недостатков, что прямо отвечает на вопрос. Также приведено объяснение, подтверждающее правильность ответа.'}
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
OpenAI score: {'score': 7, 'explanation': 'Ответ содержит правильное утверждение о том, что язык не лишен недостатков, но он слишком краткий и не раскрывает полностью суть вопроса. В ответе отсутствует конкретика и анализ, что снижает его качество. Более подробное объяснение с примерами или аргументами было бы предпочтительнее.'}
VLLM score: 10
INFO:httpx:HTTP Request: POST http://localhost:8000//v1/chat/completions "HTTP/1.1 200 OK"
OpenAI score: 7
🏃 View run dazzling-skink-89 at: http://localhost:5000/#/experiments/757769994549227305/runs/c29cae35e1584488b8c35ef676b5c2f0
🧪 View experiment at: http://localhost:5000/#/experiments/757769994549227305
```