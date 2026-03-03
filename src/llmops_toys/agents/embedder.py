# Файл: src/direct_embedding_pipeline.py

from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document as LlamaDocument
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import List, Optional
from pathlib import Path
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid

from langfuse import observe


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DirectEmbeddingPipeline:
    """Пайплайн с прямым созданием embedding-векторов"""
    
    def __init__(self, pdf_path: str, persist_directory: str = "data/agents/chroma_db"):
        self.pdf_path = Path(pdf_path)
        self.persist_directory = Path(persist_directory)
        self.chunk_overlap = 150
        self.chunk_size = 5000
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.logger = logging.getLogger(__name__)
        self.embedding_model = None
        self._setup_embedding_model()


    @observe(name="init_embedding_model", as_type="chain")
    def _setup_embedding_model(self):
        """Настройка локальной embedding модели"""
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.logger.info("✅ Локальная embedding модель загружена")
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise
    

    @observe(name="clean_text", as_type="chain")
    def clean_text(self, text: str) -> str:
        """Очистка текста"""
        if not isinstance(text, str):
            return ""
        
        # Удаляем непечатаемые символы
        cleaned = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Заменяем специальные символы
        cleaned = cleaned.replace('\x00', '').replace('\r', ' ').replace('\t', ' ')
        
        # Оставляем только ASCII символы
        ascii_only = ''.join(char if ord(char) < 128 else ' ' for char in cleaned)
        
        # Удаляем лишние пробелы
        cleaned = ' '.join(ascii_only.split())
        
        return cleaned
    

    @observe(name="load_data_and_process", as_type="chain")
    def load_and_process(self) -> List[LangChainDocument]:
        """Загрузка и обработка документов"""
        self.logger.info(f"🔄 Загрузка и обработка PDF: {self.pdf_path}")
        
        try:
            reader = SimpleDirectoryReader(
                input_files=[str(self.pdf_path)],
                required_exts=[".pdf"]
            )
            
            documents = reader.load_data()
            self.logger.info(f"✅ Загружено {len(documents)} документов")
            
            # Очищаем и создаем чанки
            all_chunks = []
            
            for i, doc in enumerate(documents):
                try:
                    text = doc.text if hasattr(doc, 'text') else str(doc)
                    # cleaned_text = self.clean_text(text)
                    
                    if len(text.strip()) > 5:
                        # Разбиваем на чанки
                        chunks = self.text_splitter.split_text(text)
                        
                        for j, chunk_text in enumerate(chunks):
                            # cleaned_chunk = self.clean_text(chunk_text)
                            
                            if len(chunk_text.strip()) > 3:
                                chunk_doc = LangChainDocument(
                                    page_content=chunk_text,
                                    metadata={
                                        **doc.metadata,
                                        'original_doc_id': i,
                                        'chunk_id': j,
                                        'chunk_length': len(chunk_text),
                                        'source_file': str(self.pdf_path)
                                    }
                                )
                                all_chunks.append(chunk_doc)
                                
                except Exception as e:
                    self.logger.warning(f"⚠️  Ошибка обработки документа {i}: {e}")
                    continue
            
            self.logger.info(f"✅ Создано {len(all_chunks)} чанков")
            return all_chunks
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки PDF: {e}")
            return []
    


    @observe(name="build_vectorstore_direct", as_type="span")
    def build_vectorstore_direct(self) -> Optional[Chroma]:
        """Построение векторного хранилища с прямыми embedding-векторами"""
        self.logger.info("🔄 Построение векторного хранилища с прямыми embedding")
        
        # Загружаем и обрабатываем документы
        documents = self.load_and_process()
        
        if not documents:
            self.logger.error("❌ Не удалось загрузить документы")
            return None

        try:
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=str(self.persist_directory)
            )

            self.logger.info(f"✅ Векторное хранилище создано c {len(documents)} документами)")
            return vectorstore
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания векторного хранилища: {e}")
            return None


def main():
    """Запуск прямого создания embedding"""
    logger = logging.getLogger(__name__)
    
    # Путь к PDF файлу
    pdf_path = "./data/Manning.Distributed.Machine.Learning.Patterns.pdf"
    
    logger.info(f"🚀 Запуск прямого создания embedding для: {pdf_path}")
    
    try:
        # Создаем пайплайн
        pipeline = DirectEmbeddingPipeline(pdf_path, "data/agents/chroma_db")
        
        # Построение векторного хранилища
        vectorstore = pipeline.build_vectorstore_direct()
        
        if vectorstore:
            logger.info("✅ Векторное хранилище успешно создано!")
            try:
                count = vectorstore._collection.count()
                logger.info(f"📊 В векторной базе данных {count} документов")
            except Exception as e:
                logger.warning(f"⚠️  Не удалось получить количество документов: {e}")
        else:
            logger.error("❌ Не удалось создать векторное хранилище")
            
    except Exception as e:
        logger.error(f"❌ Ошибка в пайплайне: {e}")

if __name__ == "__main__":
    main()