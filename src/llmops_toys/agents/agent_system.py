"""
Система агента для создания презентаций и текста на основе PDF-книги.
Использует RAG для извлечения информации и LangGraph для управления агентом.
"""
import uuid

import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Annotated, Any
from enum import Enum
import logging
from datetime import datetime
from agents.pdf_processor import PDFLoader
from langchain_core.documents import Document as LangChainDocument

# from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

import langfuse
from langfuse import observe

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

from data_engineering.prompting import setup_openai_api
from embedder import DirectEmbeddingPipeline

logger = logging.getLogger(__name__)

class PresentationState(TypedDict):
    """Состояние агента презентации"""
    book_path: Path
    presentation_title: str
    presentation_outline: List[str]
    presentation_content: Dict[str, str]
    current_slide: int
    feedback: str
    final_presentation: str
    thread_id: str

class AgentRole(Enum):
    """Роли агента"""
    RESEARCHER = "researcher"
    WRITER = "writer"
    CORRECTOR = "corrector"
    PRESENTER = "presenter"

class BookAnalyzer:
    """Анализатор PDF-книги для извлечения информации"""
    
    def __init__(self, book_path: Path, persist_directory: Path, langfuse_client: Langfuse):
        self.book_path = book_path
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.langfuse = langfuse_client
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

    @observe(name="get_relevant_context", as_type="retriver")
    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Получение релевантного контекста из книги"""
        if not self.vectorstore:
            raise ValueError("Векторное хранилище не инициализировано")
            
        results = self.vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in results])

class PresentationAgent:
    """Агент для создания презентации"""

    def _update_costs(self, responce):
        self.langfuse_client.update_current_generation(
            usage_details={
                "input": responce.usage.input_tokens,
                "output": responce.usage.output_tokens,
                "cache_read_input_tokens": responce.usage.cache_read_input_tokens
                },
                
            cost_details={
                # Here we assume the input and output cost are 1 USD each and half the price for cached tokens.
                "input": 1,
                "cache_read_input_tokens": 0.5,
                "output": 1,
            }
        )
    
    def __init__(self, book_analyzer: BookAnalyzer, langfuse_client: Langfuse):
        self.book_analyzer = book_analyzer
        self.langfuse_client = langfuse_client
        # self.langfuse_client.trace(name="PresentationAgent")
        self.llm = ChatOpenAI(
            model="Qwen3-Coder-30B-A3B-Instruct-FP8",
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.7
        )
        
        # Шаблоны промптов
        self.research_prompt = ChatPromptTemplate.from_messages(langfuse.get_prompt("research_prompt"))
        self.writer_prompt = ChatPromptTemplate.from_messages(langfuse.get_prompt("research_prompt"))

        self.presenter_prompt = ChatPromptTemplate.from_messages([langfuse.get_prompt("presentor_prompt")])
        
        self.research_chain = self.research_prompt | self.llm | StrOutputParser()
        self.writer_chain = self.writer_prompt | self.llm | StrOutputParser()
        self.presenter_chain = self.presenter_prompt | self.llm | StrOutputParser()

    @observe(name="research_book", as_type="generation")
    def research_book(self, state: PresentationState) -> PresentationState:
        """Поиск информации в книге"""
        logger.info("Начало анализа книги")
        
        context = self.book_analyzer.get_relevant_context(
            f"Основные темы и идеи книги {state['presentation_title']}", 
            k=3
        )
        
        messages = [
            HumanMessage(content=f"Книга: {state['presentation_title']}")
        ]
        
        research_result = self.research_chain.invoke({
            "messages": messages,
            "book_title": state["presentation_title"],
            "context": context
        })
        
        # Генерация плана презентации
        outline = self._generate_outline(research_result)

        self.langfuse.update_current_generation(
            usage_details={
                "input": slide_content.usage.input_tokens,
                "output": slide_content.usage.output_tokens,
                "cache_read_input_tokens": slide_content.usage.cache_read_input_tokens
                },
                
            cost_details={
                # Here we assume the input and output cost are 1 USD each and half the price for cached tokens.
                "input": 1,
                "cache_read_input_tokens": 0.5,
                "output": 1,
            }
        )
        
        return {
            **state,
            "presentation_outline": outline,
            "feedback": "Информация о книге собрана"
        }
    
    
    @observe(name="research_book", as_type="generation")
    def _generate_outline(self, research_result: str) -> List[str]:
        """Генерация плана презентации"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Создай структуру презентации на час-полтора на основе анализа книги"),
            HumanMessage(content=f"Создай 5-15 пунктов для презентации на основе этого анализа:\n{research_result}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({})

        self._update_costs(result)
        
        # Простая обработка результата
        lines = result.strip().split('\n')
        return [line.strip() for line in lines if line.strip()]
    

    @observe(name="write_slide_content", as_type="generation")
    def write_slide_content(self, state: PresentationState) -> PresentationState:
        """Создание содержания для слайдов"""
        logger.info("Создание содержания слайдов")
        
        content = {}
        for i, topic in enumerate(state["presentation_outline"]):
            context = self.book_analyzer.get_relevant_context(topic, k=2)
            
            messages = [
                HumanMessage(content=f"Тема слайда: {topic}")
            ]
            
            slide_content = self.writer_chain.invoke({
                "messages": messages,
                "slide_number": i + 1,
                "topic": topic,
                "context": context
            })
            
            content[f"slide_{i+1}"] = slide_content

            self.langfuse.update_current_generation(
                usage_details={
                    "input": slide_content.usage.input_tokens,
                    "output": slide_content.usage.output_tokens,
                    "cache_read_input_tokens": slide_content.usage.cache_read_input_tokens
                    },
                    
                cost_details={
                    # Here we assume the input and output cost are 1 USD each and half the price for cached tokens.
                    "input": 1,
                    "cache_read_input_tokens": 0.5,
                    "output": 1,
                }
            )
        
        return {
            **state,
            "presentation_content": content,
            "current_slide": len(state["presentation_outline"]),
            "feedback": "Содержание слайдов создано"
        }

    
    @observe(name="create_final_presentation", as_type="generation")
    def create_final_presentation(self, state: PresentationState) -> PresentationState:
        """Создание финальной презентации"""
        logger.info("Создание финальной презентации")
        
        context = self.book_analyzer.get_relevant_context(
            state["presentation_title"], 
            k=5
        )
        
        messages = [
            HumanMessage(content=f"Тема презентации: {state['presentation_title']}")
        ]
        
        final_presentation = self.presenter_chain.invoke({
            "messages": messages,
            "title": state["presentation_title"],
            "outline": "\n".join(state["presentation_outline"]),
            "context": context
        })

        self.langfuse.update_current_generation(
                usage_details={
                    "input": final_presentation.usage.input_tokens,
                    "output": final_presentation.usage.output_tokens,
                    "cache_read_input_tokens": final_presentation.usage.cache_read_input_tokens
                    },
                    
                cost_details={
                    # Here we assume the input and output cost are 1 USD each and half the price for cached tokens.
                    "input": 1,
                    "cache_read_input_tokens": 0.5,
                    "output": 1,
                }
            )
        
        return {
            **state,
            "final_presentation": final_presentation,
            "feedback": "Финальная презентация создана"
        }



@observe(name="create_final_presentation", as_type="chain")
def create_agent_graph(book_path: Path, persist_directory: Path) -> StateGraph:
    """Создание графа агента с использованием LangGraph"""
    
    # Инициализация компонентов
    langfuse_client = Langfuse()
    book_analyzer = BookAnalyzer(book_path, persist_directory, langfuse_client)
    
    # Обработка книги
    vectorstore = book_analyzer.load_and_process_book()
    
    # Создание агента
    agent = PresentationAgent(book_analyzer, langfuse_client)
    
    # Создание графа состояний
    workflow = StateGraph(PresentationState)
    
    # Определение узлов
    workflow.add_node("research", agent.research_book)
    workflow.add_node("write", agent.write_slide_content)
    workflow.add_node("present", agent.create_final_presentation)
    
    # Определение связей
    workflow.set_entry_point("research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "present")
    workflow.add_edge("present", END)
    
    # Создание чекпоинта для сохранения состояния
    memory = MemorySaver()
    
    # Компиляция графа
    compiled_workflow = workflow.compile(checkpointer=memory)
    
    return compiled_workflow


@observe(name="create_final_presentation", as_type="span")
def run_presentation_agent(book_path: Path, title: str) -> str:
    """
    Запуск агента для создания презентации
    
    :param book_path: Путь к PDF-файлу книги
    :param title: Название презентации
    :return: Финальная презентация в формате Markdown
    """
    # Создание директории для хранения данных
    persist_directory = Path("data/agents/chroma_db")
    persist_directory.mkdir(exist_ok=True)
    
    # Создание графа агента
    graph = create_agent_graph(book_path, persist_directory)
    
    # Запуск агента
    initial_state = {
        "book_path": book_path,
        "presentation_title": title,
        "presentation_outline": [],
        "presentation_content": {},
        "current_slide": 0,
        "feedback": "",
        "final_presentation": "",
        "thread_id": str(uuid.uuid4())
    }

    config = {
                "configurable": {
                    "thread_id": initial_state["thread_id"]
                }
            }
    
    try:
        # Выполнение графа
        result = graph.invoke(initial_state, config=config)
        return result["final_presentation"]
    except Exception as e:
        logger.error(f"Ошибка при выполнении агента: {e}")
        raise

# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Пример запуска
    try:
        # Укажите путь к вашей PDF-книге
        book_file = Path("example_book.pdf")
        presentation_title = "Анализ книги 'Принципы эффективного управления'"
        
        # Запуск агента
        presentation = run_presentation_agent(book_file, presentation_title)
        print(presentation)
        
    except FileNotFoundError:
        print("Файл книги не найден. Пожалуйста, укажите правильный путь.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")