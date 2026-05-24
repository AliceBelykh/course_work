# sentiment_analyzer.py
# Расширенная версия с поддержкой нескольких моделей

import pandas as pd
import re
from transformers import pipeline
import torch
from typing import List, Dict, Union, Optional
from collections import Counter

# ============================================
# ДОСТУПНЫЕ МОДЕЛИ ДЛЯ РУССКОГО ЯЗЫКА
# ============================================
AVAILABLE_MODELS = {
    'tiny': {
        'name': 'cointegrated/rubert-tiny-sentiment-balanced',
        'description': 'Лёгкая (30 МБ), быстрая, хороший баланс скорости и качества',
        'size_mb': 30,
        'language': 'ru',
        'recommended': True
    },
    'base': {
        'name': 'blanchefort/rubert-base-cased-sentiment',
        'description': 'Точная (450 МБ), медленнее, но качественнее для сложных текстов',
        'size_mb': 450,
        'language': 'ru',
        'recommended': False
    },
    'large': {
        'name': 'ai-forever/ruBert-large-sentiment',
        'description': 'Максимальная точность (1.2 ГБ), требует много памяти',
        'size_mb': 1200,
        'language': 'ru',
        'recommended': False
    },
    'news': {
        'name': 'sberbank-ai/rubert-base',
        'description': 'Обучалась на новостях (450 МБ), хороша для формальных текстов',
        'size_mb': 450,
        'language': 'ru',
        'recommended': False
    },
    'multilingual': {
        'name': 'nlptown/bert-base-multilingual-uncased-sentiment',
        'description': 'Мультиязычная (700 МБ), поддерживает 6 языков',
        'size_mb': 700,
        'language': 'multi',
        'recommended': False
    },
    'distilled': {
        'name': 'distilbert-base-multilingual-cased-sentiment',
        'description': 'Дистиллированная (300 МБ), быстрая, поддерживает 8 языков',
        'size_mb': 300,
        'language': 'multi',
        'recommended': False
    }
}

class SentimentAnalyzer:
    """
    Класс для анализа тональности с поддержкой разных моделей
    """
    
    def __init__(self, model_id: str = 'tiny', device: Optional[str] = None, 
                 cache_dir: Optional[str] = None):
        """
        Инициализация анализатора с выбором модели
        
        Args:
            model_id: идентификатор модели ('tiny', 'base', 'large', 'news', 'multilingual', 'distilled')
                     или полное имя модели из HuggingFace
            device: 'cpu', 'cuda' или None (автоопределение)
            cache_dir: директория для кэширования модели (опционально)
        """
        # Определяем модель
        if model_id in AVAILABLE_MODELS:
            self.model_name = AVAILABLE_MODELS[model_id]['name']
            self.model_info = AVAILABLE_MODELS[model_id]
        else:
            # Пользовательская модель
            self.model_name = model_id
            self.model_info = {
                'name': model_id,
                'description': 'Пользовательская модель',
                'size_mb': '?',
                'language': 'unknown'
            }
        
        # Настройка устройства
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        elif device == 'cuda':
            self.device = 0 if torch.cuda.is_available() else -1
            if self.device != 0:
                print("⚠️ CUDA не доступна, использую CPU")
        else:
            self.device = -1
        
        print(f"\n{'='*60}")
        print(f"ИНИЦИАЛИЗАЦИЯ МОДЕЛИ")
        print(f"{'='*60}")
        print(f"Модель: {self.model_info['name']}")
        print(f"Описание: {self.model_info['description']}")
        print(f"Размер: {self.model_info['size_mb']} MB")
        print(f"Устройство: {'GPU' if self.device == 0 else 'CPU'}")
        print(f"{'='*60}\n")
        
        print("Загрузка модели...")
        
        # Загружаем модель с возможностью указания кэша
        load_kwargs = {
            'model': self.model_name,
            'tokenizer': self.model_name,
            'device': self.device,
            'truncation': True,
            'max_length': 512
        }
        
        if cache_dir:
            load_kwargs['model_kwargs'] = {'cache_dir': cache_dir}
        
        self.pipeline = pipeline('sentiment-analysis', **load_kwargs)
        
        print("✅ Модель загружена и готова к работе\n")
    
    @staticmethod
    def list_available_models() -> pd.DataFrame:
        """
        Возвращает DataFrame со списком доступных моделей
        """
        models_data = []
        for model_id, info in AVAILABLE_MODELS.items():
            models_data.append({
                'ID': model_id,
                'Модель': info['name'],
                'Описание': info['description'],
                'Размер (MB)': info['size_mb'],
                'Рекомендуемая': '✅' if info.get('recommended', False) else ''
            })
        
        return pd.DataFrame(models_data)
    
    def clean_text(self, text: str, level: str = 'standard') -> str:
        """
        Очистка текста от мусора
        
        Args:
            text: исходный текст
            level: уровень очистки ('minimal', 'standard', 'aggressive')
            
        Returns:
            очищенный текст
        """
        if not isinstance(text, str):
            text = str(text)
        
        if level == 'minimal':
            # Минимальная очистка - только удаление ссылок
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            return text.strip()
        
        # Стандартная очистка
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        if level == 'aggressive':
            # Агрессивная очистка - удаляем знаки препинания
            text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def split_long_text(self, text: str, max_chunk_size: int = 400) -> List[str]:
        """
        Разбивает длинный текст на части по предложениям
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        # Разбиваем по предложениям
        sentences = re.split(r'([.!?;:]+|\n+)', text)
        
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            delimiter = sentences[i+1] if i+1 < len(sentences) else ""
            full_sentence = sentence + delimiter
            
            if len(current_chunk + full_sentence) < max_chunk_size:
                current_chunk += full_sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = full_sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Принудительное разбиение слишком длинных частей
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                for i in range(0, len(chunk), max_chunk_size):
                    final_chunks.append(chunk[i:i+max_chunk_size])
            else:
                final_chunks.append(chunk)
        
        return final_chunks if final_chunks else [text[:max_chunk_size]]
    
    def analyze_single(self, text: str, clean: bool = True, 
                      clean_level: str = 'standard') -> Dict:
        """
        Анализ одного текста
        
        Args:
            text: текст для анализа
            clean:是否需要 очистка
            clean_level: уровень очистки ('minimal', 'standard', 'aggressive')
            
        Returns:
            словарь с результатами анализа
        """
        # Очищаем текст при необходимости
        original_length = len(text)
        if clean:
            text = self.clean_text(text, clean_level)
        
        if not text or len(text.strip()) == 0:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'text_length_original': original_length,
                'text_length_cleaned': 0,
                'chunks_analyzed': 0
            }
        
        # Разбиваем длинный текст
        chunks = self.split_long_text(text)
        
        if len(chunks) == 1:
            # Короткий текст - анализируем целиком
            try:
                result = self.pipeline(text[:512*4])[0]
                sentiment = result['label'].lower()
                confidence = result['score']
                method = 'full'
            except Exception as e:
                print(f"Ошибка анализа: {e}")
                sentiment = 'neutral'
                confidence = 0.0
                method = 'error'
        else:
            # Длинный текст - усредняем результаты по частям
            sentiments = []
            confidences = []
            
            for chunk in chunks:
                try:
                    result = self.pipeline(chunk[:512*4])[0]
                    sentiments.append(result['label'].lower())
                    confidences.append(result['score'])
                except Exception as e:
                    print(f"Ошибка анализа части: {e}")
                    sentiments.append('neutral')
                    confidences.append(0.0)
            
            # Определяем преобладающую тональность
            sentiment = Counter(sentiments).most_common(1)[0][0]
            confidence = sum(confidences) / len(confidences) if confidences else 0.0
            method = f'split_{len(chunks)}'
        
        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 3),
            'text_length_original': original_length,
            'text_length_cleaned': len(text),
            'chunks_analyzed': len(chunks),
            'method': method,
            'model_used': self.model_name
        }
    
    def analyze_batch(self, texts: List[str], clean: bool = True,
                     clean_level: str = 'standard', show_progress: bool = True) -> List[Dict]:
        """
        Анализ нескольких текстов
        
        Args:
            texts: список текстов для анализа
            clean:是否需要 очистка
            clean_level: уровень очистки
            show_progress: показывать прогресс
            
        Returns:
            список словарей с результатами
        """
        results = []
        total = len(texts)
        
        for idx, text in enumerate(texts):
            if show_progress and (idx + 1) % 50 == 0:
                print(f"Обработано {idx + 1}/{total} текстов")
            
            results.append(self.analyze_single(text, clean, clean_level))
        
        return results
    
    def analyze_from_csv(self, csv_path: str, text_column: str = 'text',
                        clean: bool = True, clean_level: str = 'standard',
                        output_path: str = None) -> pd.DataFrame:
        """
        Анализ тональности из CSV файла
        """
        print(f"📂 Загрузка данных из {csv_path}...")
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        if text_column not in df.columns:
            raise ValueError(f"Колонка '{text_column}' не найдена. Доступны: {df.columns.tolist()}")
        
        print(f"📊 Найдено {len(df)} записей для анализа")
        print(f"🔧 Используется модель: {self.model_name}")
        print(f"🧹 Очистка: {clean_level} уровень\n")
        
        # Анализируем тексты
        results = []
        for idx, text in enumerate(df[text_column].tolist()):
            if idx % 100 == 0 and idx > 0:
                print(f"⏳ Обработано {idx}/{len(df)} записей")
            
            result = self.analyze_single(text, clean, clean_level)
            results.append(result)
        
        # Добавляем результаты в DataFrame
        df['sentiment'] = [r['sentiment'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        
        # Сохраняем результат при необходимости
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n💾 Результаты сохранены в {output_path}")
        
        # Выводим статистику
        self._print_statistics(df)
        
        return df
    
    def _print_statistics(self, df: pd.DataFrame):
        """Выводит статистику анализа"""
        print("\n" + "="*60)
        print("📊 СТАТИСТИКА АНАЛИЗА")
        print("="*60)
        
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            emoji = '😊' if sentiment == 'positive' else ('😞' if sentiment == 'negative' else '😐')
            print(f"{emoji} {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        print(f"\n📈 Средняя уверенность: {df['confidence'].mean():.2f}")
        print(f"📉 Медианная уверенность: {df['confidence'].median():.2f}")
        print(f"✅ Текстов с высокой уверенностью (>0.8): {(df['confidence'] > 0.8).sum()}")
    
    def compare_models(self, texts: List[str], models: List[str] = None) -> pd.DataFrame:
        """
        Сравнивает результаты разных моделей на одних и тех же текстах
        
        Args:
            texts: список текстов для сравнения
            models: список ID моделей для сравнения (по умолчанию все)
            
        Returns:
            DataFrame с результатами сравнения
        """
        if models is None:
            models = ['tiny', 'base']
        
        print(f"🔄 Сравнение моделей: {', '.join(models)}")
        print(f"📝 Тестовых текстов: {len(texts)}\n")
        
        results = {}
        
        for model_id in models:
            print(f"🤖 Загрузка модели {model_id}...")
            analyzer = SentimentAnalyzer(model_id=model_id, device='cpu')
            
            model_results = []
            for text in texts:
                result = analyzer.analyze_single(text, clean=True)
                model_results.append({
                    'text': text[:50] + '...' if len(text) > 50 else text,
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence']
                })
            
            results[model_id] = model_results
        
        # Формируем DataFrame для сравнения
        comparison_data = []
        for i, text in enumerate(texts):
            row = {'Текст': texts[i][:50] + '...' if len(texts[i]) > 50 else texts[i]}
            for model_id in models:
                row[f'{model_id}_sentiment'] = results[model_id][i]['sentiment']
                row[f'{model_id}_confidence'] = results[model_id][i]['confidence']
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_sentiment_label(self, sentiment: str, lang: str = 'ru') -> str:
        """Возвращает человеко-читаемую метку тональности"""
        labels = {
            'ru': {'positive': 'Положительный', 'negative': 'Отрицательный', 'neutral': 'Нейтральный'},
            'en': {'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}
        }
        return labels.get(lang, labels['ru']).get(sentiment, sentiment)


# ============================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ============================================

if __name__ == "__main__":
    import sys
    
    # Показываем доступные модели
    print("\n📚 ДОСТУПНЫЕ МОДЕЛИ:")
    print(SentimentAnalyzer.list_available_models().to_string(index=False))
    print("\n")
    
    # Выбор модели через аргументы командной строки
    model_choice = 'tiny'
    csv_file = "reviews.csv"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        model_choice = sys.argv[2]
    
    try:
        # Инициализация с выбранной моделью
        analyzer = SentimentAnalyzer(model_id=model_choice)
        
        # Анализ CSV файла
        df = analyzer.analyze_from_csv(
            csv_path=csv_file,
            text_column='text',
            output_path=f'sentiment_results_{model_choice}.csv',
            clean=True,
            clean_level='standard'  # minimal, standard, aggressive
        )
        
        # Показываем пример результатов
        print("\n" + "="*60)
        print("📝 ПРИМЕР РЕЗУЛЬТАТОВ:")
        print("="*60)
        print(df[['text', 'sentiment', 'confidence']].head(10).to_string(index=False))
        
    except FileNotFoundError:
        print(f"\n❌ Файл {csv_file} не найден!")
        print("\n📖 ИНСТРУКЦИЯ:")
        print("python sentiment_analyzer.py [csv_file] [model_id]")
        print("\nПримеры:")
        print("  python sentiment_analyzer.py reviews.csv tiny")
        print("  python sentiment_analyzer.py reviews.csv base")
        print("  python sentiment_analyzer.py reviews.csv multilingual")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")