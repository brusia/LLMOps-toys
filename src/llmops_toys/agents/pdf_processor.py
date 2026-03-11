from llama_index.core import SimpleDirectoryReader, Document, ServiceContext
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional
from pathlib import Path
import logging
import re
import unicodedata

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PDFLoader:
    """Класс для обработки PDF файлов и извлечения текста"""
    
    def __init__(self, pdf_directory: str | Path = "./data"):
        self.pdf_directory = Path(pdf_directory)
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Извлечение текста из PDF файла
        
        :param pdf_path: Путь к PDF файлу
        :return: Извлеченный текст
        """
        try:
            # Используем SimpleDirectoryReader для чтения PDF
            reader = SimpleDirectoryReader(
                input_files=[pdf_path],
                required_exts=[".pdf"]
            )
            
            documents = reader.load_data()
            
            if documents:
                # Объединяем тексты из всех документов
                full_text = "\n".join([doc.text for doc in documents])
                self.logger.info(f"✅ Успешно извлечен текст из {pdf_path}")
                return full_text
            else:
                self.logger.warning(f"⚠️  Не удалось извлечь текст из {pdf_path}")
                return ""
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка при извлечении текста из {pdf_path}: {e}")
            return ""
    
    def extract_all_pdfs(self) -> List[Document]:
        """
        Извлечение текста из всех PDF файлов в директории
        
        :return: Список документов с текстом
        """
        try:
            # Читаем все PDF файлы из директории
            reader = SimpleDirectoryReader(
                input_dir=str(self.pdf_directory),
                required_exts=[".pdf"]
            )
            
            documents = reader.load_data()
            
            self.logger.info(f"✅ Успешно загружено {len(documents)} PDF документов")
            return documents
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка при загрузке PDF файлов: {e}")
            return []

    def is_valid_text(self, text: str) -> bool:
        """Проверка, является ли текст допустимым"""
        if not isinstance(text, str):
            return False
            
        # Проверяем, содержит ли текст только допустимые символы
        # Используем только буквы, цифры и основные знаки препинания
        allowed_pattern = re.compile(r'^[a-zA-Z0-9\s\.,!?;:()\-"\'/]+$')
        
        # Проверяем, что текст не слишком короткий
        if len(text.strip()) < 3:
            return False
            
        # Проверяем, что нет специальных токенов
        if re.search(r'\[unused\d+\]', text):
            return False
            
        # Проверяем, что нет странных Unicode символов
        if any(ord(char) > 127 and not char.isalpha() for char in text):
            # Проверяем, что это не просто пробелы или знаки препинания
            clean_text = re.sub(r'[^\w\s]', '', text)
            if len(clean_text.strip()) < 3:
                return False
                
        return True
    
    def clean_text_completely(self, text: str) -> str:
        """Полная очистка текста"""
        if not isinstance(text, str):
            return ""
        
        # Нормализация Unicode
        normalized = unicodedata.normalize('NFKD', text)
        
        # Удаление непечатаемых символов
        cleaned = ''.join(char for char in normalized if char.isprintable() or char.isspace())
        
        # Замена специальных символов
        cleaned = cleaned.replace('\x00', '').replace('\r', ' ').replace('\t', ' ')
        
        # Оставляем только ASCII символы и основные знаки препинания
        ascii_only = ''.join(char if ord(char) < 128 else ' ' for char in cleaned)
        
        # Удаляем специальные токены
        ascii_only = re.sub(r'\[unused\d+\]', ' ', ascii_only)
        
        # Удаляем лишние пробелы
        cleaned = re.sub(r'\s+', ' ', ascii_only).strip()
        
        # Ограничиваем длину
        if len(cleaned) > 80:
            cleaned = cleaned[:80]
        
        return cleaned

    def process_pdf_files(self, pdf_path: str | None = None) -> List[Document]:
        """
        Обработка PDF файлов и извлечение текста
        
        :return: Список документов с текстом
        """
        if pdf_path:
            self.pdf_directory = Path(pdf_path)
        self.logger.info("🔄 Начинаем обработку PDF файлов")
        
        # Извлекаем все PDF файлы
        documents = self.extract_all_pdfs()
        
        if not documents:
            self.logger.warning("⚠️  Не найдено PDF файлов для обработки")
            return []

        chunk_size: int = 200
        chunk_overlap: int = 5
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )

        # 3. Разбиение документов на чанки
        all_chunks = []
        
        for i, doc in enumerate(documents):
            # Получаем текст документа
            text = doc.text if hasattr(doc, 'text') else str(doc)
            cleaned_text = self.clean_text_completely(text)

            if len(cleaned_text.strip()) == 0:
                    self.logger.warning(f"⚠️  Пустой текст в документе {i}")
                    continue

            # Разбиваем на чанки
            chunks = text_splitter.split_text(cleaned_text)
            
            print(f"Документ {i+1}: {len(chunks)} чанков")
            
            # Создаем новые документы для каждого чанка
            for j, chunk in enumerate(chunks):
                chunk_doc = Document(
                    text=chunk,
                    metadata={**doc.metadata, 
                            'original_doc_id': i,
                            'chunk_id': j,
                            'chunk_length': len(chunk)}
                )
                all_chunks.append(chunk_doc)

        print(f"Всего чанков: {len(all_chunks)}")
        return all_chunks
    
def main():
    """Основная функция для обработки PDF файлов"""
    logger = logging.getLogger(__name__)
    
    # Создаем процессор
    processor = PDFLoader()
    
    # Обрабатываем PDF файлы
    documents = processor.process_pdf_files()
    
    if documents:
        logger.info("🔄 Создаем векторное хранилище...")
        vectorstore = processor.create_vectorstore_safe(documents, ignore_embedding_errors=True)
        logger.info("✅ Обработка завершена успешно!")
    else:
        logger.warning("⚠️  Нет документов для обработки")

if __name__ == "__main__":
    main()