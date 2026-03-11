"""
Система агента для создания презентаций и текста на основе PDF-книги.
Использует RAG для извлечения информации и LangGraph для управления агентом.
"""
from re import M
from argparse import ArgumentParser
import uuid

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Annotated, Any, overload
from enum import Enum
import logging
from datetime import datetime
from agents.pdf_processor import PDFLoader
from langchain_core.documents import Document as LangChainDocument

# from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

from langfuse import observe

from langfuse.langchain import CallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langfuse import Langfuse

import tiktoken

from data_engineering.prompting import setup_openai_api
from embedder import DirectEmbeddingPipeline

logger = logging.getLogger(__name__)


class LangfuseCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs):
        # Подсчет входных токенов до выполнения запроса
        if prompts:
            prompt_text = prompts[0] if isinstance(prompts[0], str) else str(prompts[0])
            # Подсчет токенов с помощью tiktoken
            try:
                encoding = tiktoken.encoding_for_model('Qwen3-Coder-30B-A3B-Instruct-FP8')
                self.input_tokens_estimate = len(encoding.encode(prompt_text))
                self.prompt_tokens = self.input_tokens_estimate
            except Exception:
                # fallback на простой подсчет
                self.input_tokens_estimate = len(prompt_text.split()) * 1.3
                self.prompt_tokens = int(self.input_tokens_estimate)
        
        
    def on_llm_end(self, response, **kwargs):
        # Получаем информацию о токенах и стоимости
        if hasattr(response, 'llm_output'):
            llm_output = response.llm_output
            if isinstance(llm_output, dict):
                # Для OpenAI API
                if 'token_usage' in llm_output:
                    usage = llm_output['token_usage']
                    self.prompt_tokens = usage.get('prompt_tokens', 0)
                    self.completion_tokens = usage.get('completion_tokens', 0)
                    self.total_tokens = usage.get('total_tokens', 0)
                    
                    # Расчет стоимости (пример для GPT-3.5-turbo)
                    # Стоимость: $0.0015/1K токенов для входных токенов
                    # Стоимость: $0.002/1K токенов для выходных токенов
                    input_cost = usage.get('prompt_tokens', 0) * 0.0015 / 1000
                    output_cost = usage.get('completion_tokens', 0) * 0.002 / 1000
                    self.cost += input_cost + output_cost


class PresentationState(TypedDict):
    """Состояние агента презентации"""
    book_path: Path
    presentation_title: str
    presentation_outline: List[str]
    presentation_content: Dict[str, str]
    current_slide: int
    feedback: str
    final_presentation: str
    detailed_text: str
    thread_id: str
    tokens_used: int
    cost: float

class AgentRole(Enum):
    """Роли агента"""
    RESEARCHER = "researcher"
    WRITER = "writer"
    CORRECTOR = "corrector"
    PRESENTER = "presenter"
    DETAILED_TEXT_CREATOR = "detailed_text_creator"

class BookAnalyzer:
    """Анализатор PDF-книги для извлечения информации"""
    
    def __init__(self, book_path: Path, persist_directory: Path):
        self.book_path = book_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
    @observe(name="load_and_process_book", as_type="chain")
    def load_and_process_book(self) -> Chroma:
        """Загрузка и обработка PDF-книги"""
        try:
            loader = DirectEmbeddingPipeline(self.book_path)
            docs = loader.load_and_process()

            self.vectorstore = Chroma.from_documents(
                documents = docs,
                embedding=self.embedding_model,
                persist_directory=str(self.persist_directory)
            )
            
            logger.info(f"Книга {self.book_path.name} успешно обработана")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Ошибка при обработке книги: {e}")
            raise

    @observe(name="get_relevant_context", as_type="retriever")
    def get_relevant_context(self, query: str, k: int = 5) -> dict:
        """Получение релевантного контекста из книги"""
        if not self.vectorstore:
            raise ValueError("Векторное хранилище не инициализировано")
            
        results = self.vectorstore.similarity_search(query, k=k)
        # Отслеживаем количество найденных документов
        retrieved_docs_count = len(results)
        # Отслеживаем длину исходного текста
        original_text_length = sum(len(doc.page_content) for doc in results)

        return {
            "result": "\n".join([doc.page_content for doc in results]),
            "retrieved_documents_count": retrieved_docs_count,
            "original_text_length": original_text_length,
        }

class PresentationAgent:
    """Агент для создания презентации"""
    
    def _langfuse_prompt_to_langchain(self, prompt_name: str) -> ChatPromptTemplate:
        langfuse_prompt = self.langfuse.get_prompt(prompt_name).prompt

        system_prompt = None
        user_prompt = None
        placeholder = None

        for msg in langfuse_prompt:
            if msg['type'] == 'message':
                if msg['role'] == 'system':
                    system_prompt = msg['content']
                elif msg['role'] == 'user':
                    user_prompt = msg['content']
            elif msg['type'] == 'placeholder':
                placeholder = msg.get('name')

        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name=placeholder),
            HumanMessage(content=user_prompt)
        ])

    def __init__(self, book_analyzer: BookAnalyzer, langfuse_client: Langfuse, langfuse_callback: LangfuseCallbackHandler):
        self.book_analyzer = book_analyzer
        self.langfuse = langfuse_client
        self.langfuse_callback = langfuse_callback

        self.llm = ChatOpenAI(
            model="Qwen3-Coder-30B-A3B-Instruct-FP8",
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.7
        )

        # Шаблоны промптов
        self.research_prompt= self._langfuse_prompt_to_langchain("research_prompt")
        self.writer_prompt = self._langfuse_prompt_to_langchain("writer_prompt")
        self.presenter_prompt = self._langfuse_prompt_to_langchain("presentor_prompt")
        self.detailed_text_prompt = self._langfuse_prompt_to_langchain("detailed_text_prompt")
        
        self.research_chain = self.research_prompt | self.llm | StrOutputParser()
        self.writer_chain = self.writer_prompt | self.llm | StrOutputParser()
        self.presenter_chain = self.presenter_prompt | self.llm | StrOutputParser()
        self.detailed_text_chain = self.detailed_text_prompt |self.llm | StrOutputParser()
    
    @observe(name="research_book", as_type="generation")
    def research_book(self, state: PresentationState) -> PresentationState:
        """Поиск информации в книге"""
        logger.info("Начало анализа книги")
        
        context = self.book_analyzer.get_relevant_context(
            f"Основные темы и идеи книги {state['presentation_title']}", 
            k=3
        ).get("result")
        
        messages = [
            HumanMessage(content=f"Книга: {state['presentation_title']}")
        ]
        
        research_result = self.research_chain.invoke({
            "messages": messages,
            "book_title": state["presentation_title"],
            "context": context
        },
        callbacks=[self.langfuse_callback])
        
        # Генерация плана презентации
        outline = self._generate_outline(research_result)
        llm_as_a_judge_score = self._analyse_relevance(state["presentation_title"], context)

        self.langfuse.update_current_generation(
            usage_details={
                "input": self.langfuse_callback.prompt_tokens,
                "output": self.langfuse_callback.total_tokens,
                "completion_tokens": self.langfuse_callback.completion_tokens
                },

            cost_details={
                "input": 1,
                "output": 1,
            }
        )
        
        return {
            **state,
            "presentation_outline": outline,
            "feedback": "Информация о книге собрана",
            "relevance_judge_score": llm_as_a_judge_score.get("score", -1),
            "relevance_judge_explanation": llm_as_a_judge_score.get("explanation", "Cannot parsed judge desicion.")
        }
    

    @observe(name="_generate_outline", as_type="generation")
    def _generate_outline(self, research_result: str) -> List[str]:
        """Генерация плана презентации"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Создай структуру презентации на час-полтора на основе анализа книги"),
            HumanMessage(content=f"Создай 5-15 пунктов для презентации на основе этого анализа:\n{research_result}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({}, 
        callbacks=[self.langfuse_callback])
        
        self.langfuse.update_current_generation(
            usage_details={
                "input": self.langfuse_callback.prompt_tokens,
                "output": self.langfuse_callback.total_tokens,
                "completion_tokens": self.langfuse_callback.completion_tokens
                },

            cost_details={
                "input": 1,
                "output": 1,
            }
        )

        # Простая обработка результата
        lines = result.strip().split('\n')
        return [line.strip() for line in lines if line.strip()]

    
    @observe(name="analyse_relevance_for_founded_documents", as_type="generation")
    def _analyse_relevance(self, theme: str, document: str) -> dict:
        """Запуск LLM-as-a-Judge-агента для оценки релевантности извлечённого документа"""
        judge_promt = self._langfuse_prompt_to_langchain("judge_prompt")

        # ADR: вместо self.llm в качестве судьи в общем случае может быть использована другая модель.
        # ADR: для этого необходимо создать клиента self.llm_as_a_judge и направлять запросы ему
        judge_chain = judge_promt | self.llm | StrOutputParser()

        messages = [
                HumanMessage(content=f"Тема: {theme}, Документ: {document}")
            ]

        result = judge_chain.invoke({
            "messages": messages,
        },
        callbacks=[self.langfuse_callback])

        self.langfuse.update_current_generation(
            usage_details={
                "input": self.langfuse_callback.prompt_tokens,
                "output": self.langfuse_callback.total_tokens,
                "completion_tokens": self.langfuse_callback.completion_tokens
                },

            cost_details={
                "input": 1,
                "output": 1,
            }
        )

        try:
            if result.startswith('```json'):
                result = result[7:]  # Убираем ```json
            if result.endswith('```'):
                result = result[:-3]  # Убираем ```

            parsed = json.loads(result)
            return parsed
        except json.JSONDecodeError:
            # Если не удалось распарсить как JSON, возвращаем текст
            return {"raw_response": result}

        return parsed
    

    @observe(name="write_slide_content", as_type="generation")
    def write_slide_content(self, state: PresentationState) -> PresentationState:
        """Создание содержания для слайдов"""
        logger.info("Создание содержания слайдов")
        
        content = {}
        for i, topic in enumerate(state["presentation_outline"]):
            context = self.book_analyzer.get_relevant_context(topic, k=2).get("result")
            
            messages = [
                HumanMessage(content=f"Тема слайда: {topic}")
            ]
            
            slide_content = self.writer_chain.invoke({
                "messages": messages,
                "slide_number": i + 1,
                "topic": topic,
                "context": context
            },
        callbacks=[self.langfuse_callback])

        self.langfuse.update_current_generation(
            usage_details={
                "input": self.langfuse_callback.prompt_tokens,
                "output": self.langfuse_callback.total_tokens,
                "completion_tokens": self.langfuse_callback.completion_tokens
                },

            cost_details={
                "input": 1,
                "output": 1,
            }
        )
            
        content[f"slide_{i+1}"] = slide_content
        
        
        return {
            **state,
            "presentation_content": content,
            "current_slide": len(state["presentation_outline"]),
            "feedback": "Содержание слайдов создано",
        }
    

    @observe(name="create_final_presentation", as_type="generation")
    def create_final_presentation(self, state: PresentationState) -> PresentationState:
        """Создание финальной презентации"""
        logger.info("Создание финальной презентации")
        
        context = self.book_analyzer.get_relevant_context(
            state["presentation_title"], 
            k=5
        ).get("result")
        
        messages = [
            HumanMessage(content=f"Тема презентации: {state['presentation_title']}")
        ]
        
        final_presentation = self.presenter_chain.invoke({
            "messages": messages,
            "title": state["presentation_title"],
            "outline": "\n".join(state["presentation_outline"]),
            "context": context
        },
        callbacks=[self.langfuse_callback])

        self.langfuse.update_current_generation(
            usage_details={
                "input": self.langfuse_callback.prompt_tokens,
                "output": self.langfuse_callback.total_tokens,
                "completion_tokens": self.langfuse_callback.completion_tokens
                },

            cost_details={
                "input": 1,
                "output": 1,
            }
        )
        
        # Отслеживаем длину итогового текста
        output_length = len(final_presentation)
        # Отслеживаем входные данные
        input_data = {
            "title": state["presentation_title"],
            "outline_length": len(state["presentation_outline"]),
            "context_length": len(context)
        }

        return {
            **state,
            "final_presentation": final_presentation,
            "feedback": "Финальная презентация создана",

        }
    
    
    @observe(name="create_detailed_text", as_type="generation")
    def create_detailed_text(self, state: PresentationState) -> PresentationState:
        """Создание подробного текста всей презентации"""
        logger.info("Создание подробного текста презентации")
        
        context = self.book_analyzer.get_relevant_context(
            state["presentation_title"], 
            k=5
        ).get("result")
        
        messages = [
            HumanMessage(content=f"Тема презентации: {state['presentation_title']}")
        ]
        
        detailed_text = self.detailed_text_chain.invoke({
            "messages": messages,
            "title": state["presentation_title"],
            "outline": "\n".join(state["presentation_outline"]),
            "context": context
        },
        callbacks=[self.langfuse_callback])
        
        # Отслеживаем длину итогового текста
        output_length = len(detailed_text)
        # Отслеживаем входные данные
        input_data = {
            "title": state["presentation_title"],
            "outline_length": len(state["presentation_outline"]),
            "context_length": len(context)
        }
        
        # Получаем информацию о токенах и стоимости
        tokens_used = getattr(self.detailed_text_chain, 'tokens_used', 0)
        cost = getattr(self.detailed_text_chain, 'cost', 0.0)
        
        return {
            **state,
            "detailed_text": detailed_text,
            "feedback": "Подробный текст презентации создан",
            "tokens_used": tokens_used,
            "cost": cost
        }


@observe(name="create_agent_graph", as_type="span")
def create_agent_graph(book_path: Path, persist_directory: Path, langfuse_callback: LangfuseCallbackHandler) -> StateGraph:
    """Создание графа агента с использованием LangGraph"""
    
    # Инициализация компонентов
    book_analyzer = BookAnalyzer(book_path, persist_directory)
    langfuse_client = Langfuse()
    
    # Обработка книги
    vectorstore = book_analyzer.load_and_process_book()
    
    # Создание агента
    agent = PresentationAgent(book_analyzer, langfuse_client, langfuse_callback)
    
    # Создание графа состояний
    workflow = StateGraph(PresentationState)
    
    # Определение узлов
    workflow.add_node("research", agent.research_book)
    workflow.add_node("write", agent.write_slide_content)
    workflow.add_node("present", agent.create_final_presentation)
    workflow.add_node("create_detailed_text", agent.create_detailed_text)
    
    # Определение связей
    workflow.set_entry_point("research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "present")
    workflow.add_edge("present", "create_detailed_text")
    workflow.add_edge("create_detailed_text", END)
    
    # Создание чекпоинта для сохранения состояния
    memory = MemorySaver()
    
    # Компиляция графа
    compiled_workflow = workflow.compile(checkpointer=memory)
    
    return compiled_workflow


@observe(name="run_presentation_agent", as_type="span")
def run_presentation_agent(book_path: Path, title: str) -> Dict[str, str]:
    """
    Запуск агента для создания презентации
    
    :param book_path: Путь к PDF-файлу книги
    :param title: Название презентации
    :return: Словарь с финальной презентацией и подробным текстом
    """
    # Создание директории для хранения данных
    persist_directory = Path("data/agents/chroma_db")
    persist_directory.mkdir(exist_ok=True)
    
    # Создание графа агента

    langfuse_callback = LangfuseCallbackHandler()
    graph = create_agent_graph(book_path, persist_directory, langfuse_callback)

    # Запуск агента
    initial_state = {
        "book_path": book_path,
        "presentation_title": title,
        "presentation_outline": [],
        "presentation_content": {},
        "current_slide": 0,
        "feedback": "",
        "final_presentation": "",
        "detailed_text": "",
        "thread_id": str(uuid.uuid4()),
        "tokens_used": 0,
        "cost": 0.0
    }

    config = {
                "configurable": {
                    "thread_id": initial_state["thread_id"]
                },
                "callbacks":  [langfuse_callback],
            }
    
    try:
        # Выполнение графа
        result = graph.invoke(initial_state, config=config)
        return {
            "final_presentation": result["final_presentation"],
            "detailed_text": result["detailed_text"]
        }
    except Exception as e:
        logger.error(f"Ошибка при выполнении агента: {e}")
        raise

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    parser = ArgumentParser(description="Агент для создания презентаций и текста на основе PDF-книги")
    parser.add_argument("--pdf", required=True, help="Путь к PDF-файлу книги")
    parser.add_argument("--title", required=True, help="Название презентации")
    
    args = parser.parse_args()
    
    # Пример запуска
    try:
        # Укажите путь к вашей PDF-книге
        book_file = Path(args.pdf)
        presentation_title = args.title
        # Запуск агента
        result = run_presentation_agent(book_file, presentation_title)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"docs/agents/results/{presentation_title}")
        output_dir.mkdir(parents=True, exist_ok=True)
       
        with open(output_dir.joinpath("presentation.md"), "w", encoding="utf-8") as f:
            f.write("# " + presentation_title + "\n\n")
            f.write(result["final_presentation"])
        
        with open(output_dir.joinpath("detailed_text.md"), "w", encoding="utf-8") as f:
            f.write("# " + presentation_title + "\n\n")
            f.write(result["detailed_text"])

    except FileNotFoundError:
        print("Файл книги не найден. Пожалуйста, укажите правильный путь.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")