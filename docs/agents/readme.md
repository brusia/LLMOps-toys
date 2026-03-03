# Агент для создания презентаций и текста на основе PDF-книги

## Описание проекта

Подпроект `src/agents` представляет собой систему агентов, которая автоматически создает презентации и подробные тексты на основе PDF-книг. Используется подход RAG (Retrieval-Augmented Generation) для извлечения информации из книг и LangGraph для управления последовательностью действий агента.

Реализована RAG-архитектура, которая загружает и обрабатывает PDF-файлы, создавая векторные представления текста с помощью библиотеки Sentence Transformers. Для хранения векторов используется база данных Chroma. Система также включает многоагентную архитектуру, где каждый агент выполняет свою роль: Researcher анализирует книгу и формирует план презентации, Writer создает содержание для слайдов, Presenter готовит финальную презентацию, а Detailed Text Creator генерирует подробный текст.

Для отслеживания метрик используется Langfuse, который позволяет мониторить использование токенов, отслеживать стоимость генерации и собирать данные о производительности. Выполнение всей генерации разбито на смысловые шаги, что позволяет профилировать и оценивать. стоимость каждого шага в отдельности, эффективно управляя и оптимизируя затраты при обучение.
Управление потоком выполнения осуществляется с помощью LangGraph, что обеспечивает сохранение состояния между шагами работы агентов.

Для оценки релевантности найденных документов используется LLM-as-a-judge подход, возвращённый score с объяснением от судьи также логируется в общей системе мониторинга Langfuse.

При реализации также использована система отслеживания и версионирования для prompt-ов (через langfuse). Соответствующим промптам назначается тег latest, production, и при изменении тега проект бесшовно подхватит новю версию, если она будет помечена соответствующим тегом (например, другой командой prompt-инженеров).

## Использование

### Настройка среды

```bash
uv sync agents
docker compose --env-file env_file --file docker-compose.langfuse.yaml up -d
```

### Запуск

```bash
uv run src/llmops_toys/agents/main.py --pdf ./data/book.pdf --title "Название презентации"
```

### Необходимые переменные окружения

```bash
### OpenAI
# OpenAI API settings
OPENAI_API_KEY=
OPENAI_API_BASE=

### Langfuse
# Postgres for Langfuse
POSTGRES_LANGFUSE_USER=
POSTGRES_LANGFUSE_PASSWORD=
POSTGRES_LANGFUSE_DB=

# Clickhoue for Langfuse
CLICKHOUSE_LANGFUSE_USER = 
CLICKHOUSE_LANGFUSE_PASSWORD=
CLICKHOUSE_LANGFUSE_DB = 

# Redis for Langfuse
REDIS_LANGFUSE_PASSWORD = 

# Langfuse API settings
LANGFUSE_SECRET_KEY = sk-lf-
LANGFUSE_PUBLIC_KEY = pk-lf-
LANGFUSE_BASE_URL = "http://localhost:3000"
```

## Метрики отслеживания

- Количество использованных токенов
- Стоимость генерации
- Длина выходных текстов
- Количество найденных документов
- Показатель релевантности запросу для найденных документов

## Скриншоты

- [Граф вычислений](screens/graph.png)
- [Trace выполнения](screens/trace.png)
- [Атомарные операции](screens/observations.png)
- [Дашборды](screens/dashboards.png)
- [Prompt registry](screens/prompts.png)
- [LLM as a judge](screens/llm-as-a-judge.png)

Примеры итоговой генерации для разных запросов приведены в директории [с результатами](results/).

Исходный документ для генерации доступен [по ссылке](../../data/agents/Manning.Distributed.Machine.Learning.Patterns.pdf).

### Вывод при генерации

```bash
uv run src/llmops_toys/agents/main.py --pdf data/agents/Manning.Distributed.Machine.Learning.Patterns.pdf --title Distributed\ Machine\ Learning\ Patterns 
2026-03-09 15:01:01,768 - INFO - Use pytorch device_name: mps
2026-03-09 15:01:01,768 - INFO - Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
2026-03-09 15:01:04,977 - INFO - Use pytorch device_name: mps
2026-03-09 15:01:04,977 - INFO - Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
2026-03-09 15:01:07,622 - INFO - ✅ Локальная embedding модель загружена
2026-03-09 15:01:07,623 - INFO - 🔄 Загрузка и обработка PDF: data/agents/Manning.Distributed.Machine.Learning.Patterns.pdf
2026-03-09 15:01:09,466 - INFO - ✅ Загружено 248 документов
2026-03-09 15:01:09,473 - INFO - ✅ Создано 242 чанков
2026-03-09 15:01:09,495 - INFO - Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
2026-03-09 15:01:11,486 - INFO - Книга Manning.Distributed.Machine.Learning.Patterns.pdf успешно обработана
2026-03-09 15:01:11,968 - INFO - Начало анализа книги
2026-03-09 15:01:42,018 - INFO - Создание содержания слайдов
2026-03-09 15:06:13,742 - INFO - Создание финальной презентации
2026-03-09 15:06:21,636 - INFO - Создание подробного текста презентации
```
