#!/usr/bin/env python3
"""
Универсальный оценщик производительности для различных подходов поиска в Chroma
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from typing import List, Dict, Tuple, Callable, Any
import json

from confluence import CONFLUENCE_MLOPS_DOCS

class PerformanceEvaluator:
    """Универсальный оценщик производительности поиска в Chroma"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.test_queries = [
            "LLM deployment",
            "MLOps pipeline", 
            "Model monitoring",
            "LLM optimization",
            "CI/CD for AI"
        ]
    
    def setup_environment(self, documents: List[Dict], 
                         collection_name: str = "performance_test",
                         similarity_metric: str = "cosine") -> bool:
        """Настройка тестовой среды"""
        try:
            # Создание клиента с заданной метрикой схожести
            self.client = chromadb.Client(Settings())
        
            # Создание коллекции с заданной метрикой схожести
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Performance test collection", "similarity": similarity_metric}
            )
            
            # Загрузка модели эмбеддингов
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Вставка документов
            self._insert_documents(documents)
            
            print("✓ Тестовая среда готова")
            return True
            
        except Exception as e:
            print(f"✗ Ошибка настройки тестовой среды: {e}")
            return False
    
    def _insert_documents(self, documents: List[Dict]):
        """Вставка документов в коллекцию"""
        ids = [doc['id'] for doc in documents]
        contents = [doc['content'] for doc in documents]
        metadatas = []
        
        for doc in documents:
            metadata = {
                'title': doc['title'],
                'author': doc['author'],
                'company': doc['company'],
                'created_date': doc['created_date'],
                'status': doc['status'],
                'environment': doc['environment'],
                'difficulty': doc['difficulty'],
                'tags': doc['tags']
            }
            metadatas.append(metadata)
        
        # Создаем эмбеддинги
        embeddings = self.embedding_model.encode(contents)
        
        # Вставляем документы
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=contents,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"✓ Вставлено {len(documents)} документов")
    
    def evaluate_search_performance(self, 
                                  search_function: Callable[[str, Dict], dict],
                                  search_params: Dict[str, Any] = None,
                                  test_queries: List[str] = None,
                                  iterations: int = 1) -> Dict[str, Any]:
        """
        Универсальная функция оценки производительности поиска
        
        Args:
            search_function: функция поиска с параметрами
            search_params: параметры для функции поиска
            test_queries: список тестовых запросов
            iterations: количество итераций для усреднения
            
        Returns:
            Словарь с результатами оценки
        """
        if test_queries is None:
            test_queries = self.test_queries
            
        if search_params is None:
            search_params = {}
        
        total_execution_time = 0
        total_results_count = 0
        all_results = []
        
        print(f"🚀 Выполняется оценка производительности с параметрами: {search_params}")
        
        # Выполняем несколько итераций для усреднения
        for iteration in range(iterations):
            start_time = time.time()
            
            # Выполняем поиск для всех запросов
            iteration_results = []
            for query in test_queries:
                try:
                    # Вызываем функцию поиска с параметрами
                    result = search_function(query, search_params)
                    iteration_results.append(result)
                    total_results_count += len(result.get('ids', []))
                except Exception as e:
                    print(f"  ⚠️ Ошибка при поиске '{query}': {e}")
                    iteration_results.append({})
            
            end_time = time.time()
            iteration_time = end_time - start_time
            total_execution_time += iteration_time
            
            print(f"  Итерация {iteration + 1}: {iteration_time:.4f} секунд")
            all_results.append(iteration_results)
        
        # Вычисляем средние значения
        avg_execution_time = total_execution_time / iterations
        avg_results_per_query = total_results_count / (len(test_queries) * iterations)
        
        return {
            'execution_time': total_execution_time,
            'avg_execution_time': avg_execution_time,
            'iterations': iterations,
            'test_queries_count': len(test_queries),
            'total_results': total_results_count,
            'avg_results_per_query': avg_results_per_query,
            'search_params': search_params,
            'results': all_results
        }
    
    def basic_search(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Базовый поиск"""
        if params is None:
            params = {}
            
        n_results = params.get('n_results', 3)
        include = params.get('include', ["metadatas", "distances"])
        
        result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=include
        )
        return result
    
    def filtered_search(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Поиск с фильтрацией по метаданным"""
        if params is None:
            params = {}
            
        n_results = params.get('n_results', 3)
        tags_filter = params.get('tags_filter', ["LLM", "MLOps"])
        include = params.get('include', ["metadatas", "distances"])
        
        result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"tags": {"$in": tags_filter}},
            include=include
        )
        return result
    
    def limited_search(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Поиск с ограниченным количеством результатов"""
        if params is None:
            params = {}
            
        n_results = params.get('n_results', 1)
        include = params.get('include', ["metadatas", "distances"])
        
        result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=include
        )
        return result
    
    def distance_filtered_search(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Поиск с фильтрацией по дистанции"""
        if params is None:
            params = {}
            
        n_results = params.get('n_results', 5)
        max_distance = params.get('max_distance', 0.5)
        include = params.get('include', ["metadatas", "distances"])
        
        result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=include
        )
        
        # Фильтруем по дистанции
        if result.get('distances'):
            filtered_result = {
                'ids': [],
                'metadatas': [],
                'distances': []
            }
            
            for i, (id_list, meta_list, dist_list) in enumerate(zip(
                result['ids'], result['metadatas'], result['distances']
            )):
                for j, distance in enumerate(dist_list):
                    if distance <= max_distance:
                        filtered_result['ids'].append(id_list[j])
                        filtered_result['metadatas'].append(meta_list[j])
                        filtered_result['distances'].append(distance)
            
            return filtered_result
        
        return result
    
    def combined_search(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Комбинированный поиск"""
        if params is None:
            params = {}
            
        n_results = params.get('n_results', 3)
        tags_filter = params.get('tags_filter', ["LLM", "MLOps", "CI/CD"])
        max_distance = params.get('max_distance', 0.6)
        include = params.get('include', ["metadatas", "distances"])
        
        result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"tags": {"$in": tags_filter}},
            include=include
        )
        
        # Дополнительная фильтрация по дистанции
        if result.get('distances'):
            filtered_result = {
                'ids': [],
                'metadatas': [],
                'distances': []
            }
            
            for i, (id_list, meta_list, dist_list) in enumerate(zip(
                result['ids'], result['metadatas'], result['distances']
            )):
                for j, distance in enumerate(dist_list):
                    if distance <= max_distance:
                        filtered_result['ids'].append(id_list[j])
                        filtered_result['metadatas'].append(meta_list[j])
                        filtered_result['distances'].append(distance)
            
            return filtered_result
        
        return result

def run_performance_comparison(documents: List[Dict], similarity_metric: str = "cosine", 
                              collection_name: str = "performance_test") -> List[Dict]:
    """Функция для запуска сравнения производительности"""
    evaluator = PerformanceEvaluator()
    
    # Настройка тестовой среды
    if not evaluator.setup_environment(documents, collection_name=collection_name, 
                                       similarity_metric=similarity_metric):
        return []
    
    print("🚀 Запуск сравнения производительности...")
    print("=" * 70)
    
    # Определяем различные подходы
    approaches = [
        {
            'name': 'Базовый поиск',
            'function': evaluator.basic_search,
            'params': {'n_results': 3},
            'description': 'Стандартный поиск без фильтрации'
        },
        {
            'name': 'Поиск с фильтрацией',
            'function': evaluator.filtered_search,
            'params': {'n_results': 3, 'tags_filter': ['LLM', 'MLOps']},
            'description': 'Поиск с фильтрацией по тегам'
        },
        {
            'name': 'Ограниченные результаты',
            'function': evaluator.limited_search,
            'params': {'n_results': 1},
            'description': 'Поиск с ограниченным количеством результатов'
        },
        {
            'name': 'Фильтрация по дистанции',
            'function': evaluator.distance_filtered_search,
            'params': {'n_results': 5, 'max_distance': 1.5},
            'description': 'Поиск с фильтрацией по дистанции'
        },
        {
            'name': 'Комбинированная оптимизация',
            'function': evaluator.combined_search,
            'params': {'n_results': 5, 'max_distance': 2, 'tags_filter': ['LLM', 'MLOps', 'CI/CD']},
            'description': 'Комбинированная оптимизация'
        }
    ]
    
    results = []
    
    # Тестируем каждый подход
    for approach in approaches:
        print(f"\n🧪 Тест: {approach['name']}")
        print(f"   Описание: {approach['description']}")
        print(f"   Параметры: {approach['params']}")
        
        # Оцениваем производительность
        performance_result = evaluator.evaluate_search_performance(
            search_function=approach['function'],
            search_params=approach['params'],
            iterations=3  # 3 итерации для усреднения
        )
        
        performance_result['approach_name'] = approach['name']
        performance_result['description'] = approach['description']
        performance_result['similarity_metric'] = similarity_metric
        performance_result['collection_name'] = collection_name
        
        results.append(performance_result)
        
        print(f"   Время выполнения: {performance_result['execution_time']:.4f} секунд")
        print(f"   Среднее время: {performance_result['avg_execution_time']:.4f} секунд")
        print(f"   Результатов: {performance_result['total_results']}")
        print(f"   Среднее на запрос: {performance_result['avg_results_per_query']:.2f}")
    
    return results

def compare_results(results: List[Dict]) -> List[Dict]:
    """Сравнение результатов тестов"""
    print("\n" + "=" * 70)
    print("📊 Сравнение производительности")
    print("=" * 70)
    
    # Сортировка по времени выполнения
    sorted_results = sorted(results, key=lambda x: x['avg_execution_time'])
    
    print(f"{'№':<3} {'Подход':<25} {'Время (сек)':<12} {'Среднее (сек)':<12} {'Результатов':<10} {'Метрика':<10} {'Описание'}")
    print("-" * 100)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<3} {result['approach_name']:<25} {result['execution_time']:<12.4f} "
              f"{result['avg_execution_time']:<12.4f} {result['total_results']:<10} "
              f"{result.get('similarity_metric', 'cosine'):<10} {result['description']}")
    
    print("\n🏆 Рекомендации:")
    best_approach = sorted_results[0]
    print(f"   Лучший подход: '{best_approach['approach_name']}'")
    print(f"   Время выполнения: {best_approach['avg_execution_time']:.4f} секунд")
    print(f"   Метрика схожести: {best_approach.get('similarity_metric', 'cosine')}")
    
    return sorted_results

def main():
    """Основная функция"""
    # Импортируем документы
    
    # Определяем три разных окружения с разными метриками схожести
    environments = [
        {
            'name': 'Cosine Similarity Environment',
            'similarity_metric': 'cosine',
            'collection_name': 'cosine_env'
        },
        {
            'name': 'Euclidean Distance Environment', 
            'similarity_metric': 'euclidean',
            'collection_name': 'euclidean_env'
        },
        {
            'name': 'Dot Product Environment',
            'similarity_metric': 'dot',
            'collection_name': 'dot_env'
        }
    ]
    
    all_results = []
    
    # Запускаем тестирование для каждого окружения
    for env in environments:
        print(f"\n{'='*70}")
        print(f"Тестирование в окружении: {env['name']}")
        print(f"Метрика схожести: {env['similarity_metric']}")
        print(f"{'='*70}")
        
        # Запускаем сравнение производительности для текущего окружения
        results = run_performance_comparison(
            CONFLUENCE_MLOPS_DOCS, 
            similarity_metric=env['similarity_metric'],
            collection_name=env['collection_name']
        )
        all_results.extend(results)
    
    print("\n" + "=" * 70)
    print("📋 Общее сравнение производительности по всем окружениям")
    print("=" * 70)
    
    # Сравниваем результаты из всех окружений
    if all_results:
        # Группируем результаты по метрике схожести
        grouped_results = {}
        for result in all_results:
            metric = result.get('similarity_metric', 'cosine')
            if metric not in grouped_results:
                grouped_results[metric] = []
            grouped_results[metric].append(result)
        
        # Выводим результаты для каждого окружения
        for metric, results in grouped_results.items():
            print(f"\n📈 Результаты для метрики '{metric}':")
            print("-" * 50)
            sorted_results = sorted(results, key=lambda x: x['avg_execution_time'])
            for i, result in enumerate(sorted_results, 1):
                print(f"  {i}. {result['approach_name']}: "
                      f"{result['avg_execution_time']:.4f} секунд")
        
        # Находим лучший подход по каждой метрике
        print("\n🏆 Лучшие подходы по каждой метрике:")
        print("-" * 50)
        for metric, results in grouped_results.items():
            best_result = min(results, key=lambda x: x['avg_execution_time'])
            print(f"  {metric}: '{best_result['approach_name']}' ({best_result['avg_execution_time']:.4f} сек)")
    
    print("\n" + "=" * 70)
    print("📋 Рекомендации по выбору подхода")
    print("=" * 70)
    
    if all_results:
        # Находим самый быстрый общий результат
        overall_best = min(all_results, key=lambda x: x['avg_execution_time'])
        print(f"1. Для максимальной производительности используйте: '{overall_best['approach_name']}'")
        print(f"   - Время выполнения: {overall_best['avg_execution_time']:.4f} секунд")
        print(f"   - Метрика схожести: {overall_best.get('similarity_metric', 'cosine')}")
        
        # Сравниваем метрики схожести
        print("\n2. Сравнение метрик схожести:")
        metrics_performance = {}
        for result in all_results:
            metric = result.get('similarity_metric', 'cosine')
            if metric not in metrics_performance:
                metrics_performance[metric] = []
            metrics_performance[metric].append(result)
        
        # Выводим сравнение по метрикам
        for metric, results in metrics_performance.items():
            avg_time = sum(r['avg_execution_time'] for r in results) / len(results)
            print(f"   {metric}: среднее время {avg_time:.4f} секунд")
        
        # Рекомендации по метрикам
        fastest_metric = min(metrics_performance.keys(), 
                           key=lambda m: sum(r['avg_execution_time'] for r in metrics_performance[m]) / len(metrics_performance[m]))
        print(f"\n   Рекомендуемая метрика: '{fastest_metric}'")
        print("   Она обеспечивает наилучшую производительность в данном тесте")

if __name__ == "__main__":
    main()
