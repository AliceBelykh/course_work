# app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import os
import uuid
from datetime import datetime
import shutil
import glob
import re
from collections import Counter

from sentiment_analyzer import SentimentAnalyzer, AVAILABLE_MODELS


# Глобальные переменные для кэширования
movies_db = {}       # { "название": id, "id": название }
movies_info = {}     # { id: {"title": "", "year": "", "rating": ""} }
reviews_index = {}  # { movie_id: [список путей] }

# Создаём директории
os.makedirs("uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Инициализация приложения
app = FastAPI(title="Sentiment Analysis API", description="API для анализа тональности текстов")

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Шаблоны
templates = Jinja2Templates(directory="templates")

# Глобальный экземпляр анализатора (по умолчанию легкая модель)
analyzer = SentimentAnalyzer(model_id='tiny')

# Хранилище результатов загрузок
uploaded_files = {}

# Модели для API
class TextAnalysisRequest(BaseModel):
    text: str
    model_id: str = 'tiny'
    clean_level: str = 'standard'

class BatchAnalysisRequest(BaseModel):
    texts: List[str]
    model_id: str = 'tiny'
    clean_level: str = 'standard'

# ============================================
# ЗАГРУЗКА ДАННЫХ
# ============================================

def load_movies_database(csv_path="movies.csv"):
    """Загружает фильмы из вашего CSV (без изменений структуры)"""
    global movies_db, movies_info
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        for _, row in df.iterrows():
            movie_id = str(row['kinopoiskId']).strip()
            title = str(row['Title']).strip().lower()
            year = str(row.get('Year', '')) if pd.notna(row.get('Year', '')) else ''
            rating = str(row.get('Rating Kinopoisk', '')) if pd.notna(row.get('Rating Kinopoisk', '')) else ''
            
            # Для поиска по названию
            movies_db[title] = movie_id
            # Для обратного поиска по ID
            movies_db[movie_id] = title
            # Дополнительная информация
            movies_info[movie_id] = {
                'title': row['Title'],
                'year': year,
                'rating': rating
            }
        
        print(f"✅ Загружено {len(df)} фильмов")
        return True
    except Exception as e:
        print(f"⚠️ Ошибка загрузки movies.csv: {e}")
        return False

def get_review_files_list(folder_path="reviews"):
    """
    Возвращает словарь { movie_id: [список путей к файлам рецензий] }
    БЕЗ чтения содержимого файлов
    """
    reviews_index = {}
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return reviews_index
    
    files = glob.glob(f"{folder_path}/*.txt")
    print(f"📄 Найдено {len(files)} файлов с рецензиями (индексация без чтения)")
    
    for filepath in files:
        filename = os.path.basename(filepath)
        match = re.match(r'^(\d+)-', filename)
        if match:
            movie_id = match.group(1)
            if movie_id not in reviews_index:
                reviews_index[movie_id] = []
            reviews_index[movie_id].append(filepath)  # ← сохраняем ТОЛЬКО путь
    
    print(f"✅ Проиндексировано {len(reviews_index)} фильмов")
    return reviews_index

def load_review_text(filepath):
    """
    Загружает текст одного отзыва по пути
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"⚠️ Ошибка чтения {filepath}: {e}")
        return ""

def get_reviews_by_movie(movie_id, limit=None):
    """
    Получает отзывы для фильма (с постраничной загрузкой)
    limit - максимум отзывов (для тестирования, можно убрать)
    """
    filepaths = reviews_index.get(movie_id, [])
    
    if limit:
        filepaths = filepaths[:limit]
    
    reviews = []
    for filepath in filepaths:
        text = load_review_text(filepath)
        if text:
            reviews.append(text)
    
    return reviews

def get_reviews_count(movie_id):
    """Возвращает количество доступных отзывов для фильма"""
    return len(reviews_index.get(movie_id, []))

# Глобальная переменная теперь хранит ТОЛЬКО пути
reviews_index = {}  # { movie_id: [filepath1, filepath2, ...] }

# Загрузка индекса при старте
def init_reviews_index(folder_path="reviews"):
    global reviews_index
    reviews_index = get_review_files_list(folder_path)
    return reviews_index


# Загружаем данные при старте
print("\n📂 Загрузка данных...")
load_movies_database("movies_extended.csv")
init_reviews_index("reviews")
print("")

# ============================================
# ПОИСК ФИЛЬМОВ
# ============================================

def search_movie_by_title(query):
    """Ищет фильм по названию в вашей базе"""
    query_lower = query.lower().strip()
    results = []
    
    for title, movie_id in movies_db.items():
        if title.isdigit():
            continue
        if query_lower in title:
            info = movies_info.get(movie_id, {})
            results.append({
                'id': movie_id,
                'name': info.get('title', title.title()),
                'year': info.get('year', ''),
                'rating': info.get('rating', ''),
                'reviews_count': get_reviews_count(movie_id)  # ← быстрая операция
            })
    
    results.sort(key=lambda x: (x['name'].lower() != query_lower, x['name']))
    return results[:15]

# ============================================
# ВЕБ-ИНТЕРФЕЙС
# ============================================

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     """Главная страница с интерфейсом"""
#     from sentiment_analyzer import AVAILABLE_MODELS
#     return templates.TemplateResponse("index.html", {
#         "request": request,
#         "models": AVAILABLE_MODELS
#     })

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# ============================================
# API ДЛЯ РАБОТЫ С ТЕКСТОМ
# ============================================

@app.post("/api/analyze/text")
async def analyze_text(request: TextAnalysisRequest):
    """
    Анализ одного текста
    
    Пример запроса:
    {
        "text": "Отличный фильм, очень понравился!",
        "model_id": "tiny",
        "clean_level": "standard"
    }
    """
    try:
        # Если модель отличается от текущей - загружаем
        global analyzer
        if analyzer.model_name != request.model_id and request.model_id != 'tiny':
            # Динамическая загрузка другой модели
            temp_analyzer = SentimentAnalyzer(model_id=request.model_id)
            result = temp_analyzer.analyze_single(
                request.text, 
                clean=True, 
                clean_level=request.clean_level
            )
        else:
            result = analyzer.analyze_single(
                request.text, 
                clean=True, 
                clean_level=request.clean_level
            )
        
        return {
            "success": True,
            "data": {
                "text": request.text[:200] + "..." if len(request.text) > 200 else request.text,
                "sentiment": result['sentiment'],
                "sentiment_label": analyzer.get_sentiment_label(result['sentiment']),
                "confidence": result['confidence'],
                "model_used": result.get('model_used', analyzer.model_name),
                "text_length": result['text_length_cleaned'],
                "chunks_analyzed": result.get('chunks_analyzed', 1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/batch")
async def analyze_batch(request: BatchAnalysisRequest):
    """
    Анализ нескольких текстов
    
    Пример запроса:
    {
        "texts": ["Отличный фильм!", "Скучно..."],
        "model_id": "tiny",
        "clean_level": "standard"
    }
    """
    try:
        temp_analyzer = SentimentAnalyzer(model_id=request.model_id)
        results = temp_analyzer.analyze_batch(
            request.texts, 
            clean=True, 
            clean_level=request.clean_level
        )
        
        return {
            "success": True,
            "data": [
                {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": res['sentiment'],
                    "sentiment_label": temp_analyzer.get_sentiment_label(res['sentiment']),
                    "confidence": res['confidence']
                }
                for text, res in zip(request.texts, results)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# API ДЛЯ РАБОТЫ С CSV ФАЙЛАМИ
# ============================================

@app.post("/api/upload/csv")
async def upload_csv(
    file: UploadFile = File(...),
    model_id: str = Form('tiny'),
    text_column: str = Form('text'),
    clean_level: str = Form('standard')
):
    """
    Загрузка и анализ CSV файла
    
    Ожидаемый формат CSV:
    - Должен содержать колонку с текстами (по умолчанию 'text')
    - Другие колонки игнорируются
    - Кодировка: UTF-8
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
    
    try:
        # Генерируем уникальное имя файла
        file_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file_id}_{file.filename}"
        filepath = os.path.join("uploads", filename)
        
        # Сохраняем файл
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Загружаем CSV
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except:
            # Пробуем другие кодировки
            try:
                df = pd.read_csv(filepath, encoding='cp1251')
            except:
                df = pd.read_csv(filepath, encoding='latin1')
        
        # Проверяем наличие нужной колонки
        if text_column not in df.columns:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Колонка '{text_column}' не найдена. Доступные колонки: {df.columns.tolist()}"
                }
            )
        
        # Анализируем тексты
        temp_analyzer = SentimentAnalyzer(model_id=model_id)
        
        # Удаляем пустые строки
        df_clean = df[df[text_column].notna()].copy()
        
        results = []
        for idx, row in df_clean.iterrows():
            result = temp_analyzer.analyze_single(
                row[text_column],
                clean=True,
                clean_level=clean_level
            )
            results.append(result)
        
        # Добавляем результаты
        df_clean['sentiment'] = [r['sentiment'] for r in results]
        df_clean['confidence'] = [r['confidence'] for r in results]
        df_clean['sentiment_label'] = df_clean['sentiment'].apply(
            lambda x: temp_analyzer.get_sentiment_label(x)
        )
        
        # Сохраняем результат
        result_filename = f"result_{filename}"
        result_path = os.path.join("uploads", result_filename)
        df_clean.to_csv(result_path, index=False, encoding='utf-8-sig')
        
        # Статистика
        stats = {
            'total': len(df_clean),
            'positive': int((df_clean['sentiment'] == 'positive').sum()),
            'negative': int((df_clean['sentiment'] == 'negative').sum()),
            'neutral': int((df_clean['sentiment'] == 'neutral').sum()),
            'avg_confidence': float(df_clean['confidence'].mean())
        }
        
        # Сохраняем информацию для скачивания
        uploaded_files[result_filename] = result_path
        
        return {
            "success": True,
            "data": {
                "file_id": file_id,
                "original_filename": file.filename,
                "result_filename": result_filename,
                "stats": stats,
                "preview": df_clean[[text_column, 'sentiment_label', 'confidence']].head(10).to_dict('records')
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_result(filename: str):
    """Скачать результат анализа"""
    if filename in uploaded_files:
        return FileResponse(
            uploaded_files[filename],
            media_type='text/csv',
            filename=filename
        )
    
    filepath = os.path.join("uploads", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type='text/csv', filename=filename)
    
    raise HTTPException(status_code=404, detail="Файл не найден")

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ENDPOINTS
# ============================================

@app.get("/api/models")
async def get_models():
    """Получить список доступных моделей"""
    from sentiment_analyzer import AVAILABLE_MODELS
    return AVAILABLE_MODELS

@app.get("/api/health")
async def health_check():
    """Проверка работоспособности"""
    return {
        "status": "ok",
        "model_loaded": analyzer.model_name,
        "timestamp": datetime.now().isoformat()
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# ============================================
# API ДЛЯ РАБОТЫ С КИНОПОИСКОМ (ЛОКАЛЬНЫЙ ДАТАСЕТ)
# ============================================

class SearchMovieRequest(BaseModel):
    query: str

@app.post("/api/movies/search")
async def search_movie(request: SearchMovieRequest):
    """Поиск фильма по названию"""
    results = search_movie_by_title(request.query)
    if results:
        return {"success": True, "data": results}
    return {"success": False, "error": f"Фильм '{request.query}' не найден"}

@app.post("/api/movies/analyze")
async def analyze_movie(request: SearchMovieRequest):
    """Анализ всех рецензий фильма"""
    # Находим ID фильма
    query_lower = request.query.lower().strip()
    movie_id = None
    movie_title = None
    
    for title, mid in movies_db.items():
        if title.isdigit():
            continue
        if query_lower in title:
            movie_id = mid
            movie_title = movies_info.get(movie_id, {}).get('title', title.title())
            break
    
    if not movie_id:
        return {"success": False, "error": f"Фильм '{request.query}' не найден"}
    
    # Получаем рецензии
    # Получаем отзывы (только когда нужно)
    reviews = get_reviews_by_movie(movie_id)  # ← загружаем ТОЛЬКО для выбранного фильма
    
    if not reviews:
        return {"success": False, "error": f"Для фильма '{movie_title}' нет рецензий"}
    
    # Анализируем
    results = []
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for review in reviews:
        analysis = analyzer.analyze_single(review, clean=True, clean_level='standard')
        results.append({
            'text': review[:200] + "..." if len(review) > 200 else review,
            'sentiment': analysis['sentiment'],
            'sentiment_label': analyzer.get_sentiment_label(analysis['sentiment']),
            'confidence': analysis['confidence']
        })
        sentiment_counts[analysis['sentiment']] += 1
    
    # Сохраняем CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"reviews_{movie_id}_{timestamp}.csv"
    csv_path = os.path.join("uploads", csv_filename)
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    uploaded_files[csv_filename] = csv_path
    
    total = len(results)
    stats = {
        'total': total,
        'positive': sentiment_counts['positive'],
        'negative': sentiment_counts['negative'],
        'neutral': sentiment_counts['neutral'],
        'positive_percent': round(sentiment_counts['positive'] / total * 100, 1) if total else 0,
        'negative_percent': round(sentiment_counts['negative'] / total * 100, 1) if total else 0,
        'neutral_percent': round(sentiment_counts['neutral'] / total * 100, 1) if total else 0,
        'avg_confidence': round(sum(r['confidence'] for r in results) / total, 3) if total else 0
    }
    
    return {
        "success": True,
        "data": {
            "movie_name": movie_title,
            "movie_id": movie_id,
            "stats": stats,
            "reviews_preview": results[:20],
            "csv_filename": csv_filename
        }
    }

@app.get("/api/movies/list")
async def get_movies_list():
    """Список всех фильмов, у которых есть рецензии"""
    movies_list = []
    for movie_id, filepaths in reviews_index.items():  # ← работаем с индексам
        info = movies_info.get(movie_id, {})
        movies_list.append({
            'id': movie_id,
            'name': info.get('title', movie_id),
            'year': info.get('year', ''),
            'rating': info.get('rating', ''),
            'reviews_count': len(filepaths)  # ← просто длина списка, без чтения файлов
        })
    movies_list.sort(key=lambda x: x['name'])
    return {"success": True, "data": movies_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)