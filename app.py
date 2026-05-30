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
import re
from collections import Counter

# NLTK для стоп-слов (опционально, но рекомендуется)
try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK не установлен. Установите: pip install nltk")

from sentiment_analyzer import SentimentAnalyzer, AVAILABLE_MODELS

# Создаём директории
os.makedirs("uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Инициализация приложения
app = FastAPI(title="Sentiment Analysis API", description="API для анализа тональности текстов")

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Глобальный экземпляр анализатора (по умолчанию легкая модель)
analyzer = SentimentAnalyzer(model_id='tiny')

# Хранилище результатов загрузок
uploaded_files = {}

# Русские стоп-слова для частотного анализа
RUSSIAN_STOPWORDS = {
    'и', 'в', 'во', 'не', 'что', 'на', 'я', 'с', 'со', 'как', 'а', 'но', 'он', 'она', 'оно',
    'они', 'его', 'её', 'их', 'у', 'к', 'по', 'от', 'до', 'из', 'за', 'о', 'об', 'при', 'через',
    'бы', 'да', 'нет', 'так', 'вот', 'же', 'ли', 'ну', 'это', 'быть', 'весь', 'все', 'всё',
    'вся', 'всех', 'всем', 'всеми', 'этот', 'эта', 'эти', 'этих', 'этим', 'этими', 'тот', 'та',
    'те', 'тех', 'тем', 'теми', 'свой', 'своя', 'свое', 'свои', 'своего', 'своей', 'своих',
    'своим', 'своими', 'который', 'которая', 'которое', 'которые', 'которого', 'которой',
    'которых', 'которым', 'которыми', 'такой', 'такая', 'такое', 'такие', 'такого', 'такой',
    'таких', 'таким', 'такими', 'очень', 'весьма', 'более', 'менее', 'также', 'где', 'когда',
    'тогда', 'поэтому', 'потому', 'почему', 'зачем', 'откуда', 'куда', 'туда', 'сюда', 'тут',
    'там', 'здесь', 'теперь', 'уже', 'ещё', 'еще', 'даже', 'будто', 'словно', 'точно', 'прямо',
    'едва', 'лишь', 'только', 'чуть', 'вдруг', 'сразу', 'опять', 'снова', 'вновь', 'потом',
    'затем', 'сперва', 'сначала', 'наконец', 'действительно', 'неужели', 'разве', 'как-то',
    'где-то', 'кто-то', 'что-то', 'кое-как', 'кое-где', 'кое-кто', 'кое-что'
}

# Модели для API
class TextAnalysisRequest(BaseModel):
    text: str
    model_id: str = 'tiny'
    clean_level: str = 'standard'

class BatchAnalysisRequest(BaseModel):
    texts: List[str]
    model_id: str = 'tiny'
    clean_level: str = 'standard'

class SingleWordCloudRequest(BaseModel):
    text: str
    sentiment: str = 'neutral'

# ============================================
# ВЕБ-ИНТЕРФЕЙС
# ============================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Ошибка: файл templates/index.html не найден</h1>
                <p>Убедитесь, что файл index.html находится в папке templates</p>
            </body>
        </html>
        """, status_code=404)

# ============================================
# API ДЛЯ РАБОТЫ С ТЕКСТОМ
# ============================================

@app.post("/api/analyze/text")
async def analyze_text(request: TextAnalysisRequest):
    """
    Анализ одного текста
    """
    try:
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
                "model_used": result.get('model_used', 'tiny'),
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
    """
    try:
        results = analyzer.analyze_batch(
            request.texts, 
            clean=True, 
            clean_level=request.clean_level,
            show_progress=False
        )
        
        return {
            "success": True,
            "data": [
                {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "sentiment": res['sentiment'],
                    "sentiment_label": analyzer.get_sentiment_label(res['sentiment']),
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
        
        # Создаем анализатор с выбранной моделью
        temp_analyzer = SentimentAnalyzer(model_id=model_id)
        
        # Удаляем пустые строки
        df_clean = df[df[text_column].notna()].copy()
        # Удаляем пустые строки
        df_clean = df[df[text_column].notna()].copy()

        # ДОБАВЬТЕ ЭТУ СТРОКУ - удаляем пустые строки и строки с nan
        df_clean = df_clean[df_clean[text_column].astype(str).str.strip() != '']
        df_clean = df_clean[df_clean[text_column].astype(str) != 'nan']
        
        # Анализируем тексты
        results = []
        for idx, row in df_clean.iterrows():
            result = temp_analyzer.analyze_single(
                str(row[text_column]),
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
        
        # Создаем превью
        preview = []
        for _, row in df_clean.head(10).iterrows():
            preview.append({
                text_column: str(row[text_column])[:100],
                'sentiment_label': row['sentiment_label'],
                'confidence': row['confidence']
            })
                    # Рассчитываем среднюю уверенность по классам
        positive_conf = df_clean[df_clean['sentiment'] == 'positive']['confidence'].mean() if stats['positive'] > 0 else 0
        negative_conf = df_clean[df_clean['sentiment'] == 'negative']['confidence'].mean() if stats['negative'] > 0 else 0
        neutral_conf = df_clean[df_clean['sentiment'] == 'neutral']['confidence'].mean() if stats['neutral'] > 0 else 0
        
        return {
            "success": True,
            "data": {
                "file_id": file_id,
                "original_filename": file.filename,
                "result_filename": result_filename,
                "stats": stats,
                "preview": preview,
                "positive_confidence": float(positive_conf),
                "negative_confidence": float(negative_conf),
                "neutral_confidence": float(neutral_conf)
            }
        }
        
        return {
            "success": True,
            "data": {
                "file_id": file_id,
                "original_filename": file.filename,
                "result_filename": result_filename,
                "stats": stats,
                "preview": preview
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
# API ДЛЯ ОБЛАКА СЛОВ (НОВЫЕ ЭНДПОИНТЫ)
# ============================================

@app.post("/api/wordcloud/frequencies")
async def get_word_frequencies(
    file: UploadFile = File(...),
    text_column: str = Form('text'),
    clean_level: str = Form('standard')
):
    """
    Анализ частотности слов для построения облака слов
    Возвращает отдельно для всех, положительных, отрицательных и нейтральных отзывов
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
    
    try:
        # Сохраняем временный файл
        temp_path = os.path.join("uploads", f"temp_{uuid.uuid4()}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Загружаем CSV
        try:
            df = pd.read_csv(temp_path, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(temp_path, encoding='cp1251')
            except:
                df = pd.read_csv(temp_path, encoding='latin1')
        
        # Удаляем временный файл
        os.remove(temp_path)
        
        # Проверяем наличие колонки с текстом
        if text_column not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Колонка '{text_column}' не найдена. Доступны: {df.columns.tolist()}"}
            )
        
        # Проверяем наличие колонки с тональностью
        has_sentiment = 'sentiment' in df.columns
        
        # Русские стоп-слова
        stopwords = {
            'и', 'в', 'во', 'не', 'что', 'на', 'я', 'с', 'со', 'как', 'а', 'но', 'он', 'она', 'оно',
            'они', 'его', 'её', 'их', 'у', 'к', 'по', 'от', 'до', 'из', 'за', 'о', 'об', 'при', 'через',
            'бы', 'да', 'нет', 'так', 'вот', 'же', 'ли', 'ну', 'это', 'быть', 'весь', 'все', 'всё',
            'очень', 'также', 'где', 'когда', 'тогда', 'уже', 'ещё', 'даже', 'фильм', 'кино', 'потом',
            'потому', 'поэтому', 'зачем', 'почему', 'тут', 'там', 'здесь', 'теперь', 'еще'
        }
        
        # Функция очистки текста
        def clean_for_frequencies(text: str):
            if not isinstance(text, str):
                return []
            
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'[^\w\sа-яё]', '', text)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'\s+', ' ', text)
            
            words = text.split()
            filtered_words = [w for w in words if len(w) >= 3 and w not in stopwords]
            return filtered_words
        
        # Собираем частоты для каждой тональности
        all_words = []
        positive_words = []
        negative_words = []
        neutral_words = []
        
        for idx, row in df.iterrows():
            # Пропускаем пустые строки
            cell_value = row[text_column]
            if pd.isna(cell_value):
                continue
            if str(cell_value).strip() == '':
                continue
            
            text = str(cell_value)
            words = clean_for_frequencies(text)
            
            # Добавляем в общий счётчик
            all_words.extend(words)
            
            # Добавляем в счётчик по тональности (если есть колонка sentiment)
            if has_sentiment:
                sentiment = str(row.get('sentiment', 'neutral')).lower()
                if sentiment == 'positive':
                    positive_words.extend(words)
                elif sentiment == 'negative':
                    negative_words.extend(words)
                else:
                    neutral_words.extend(words)
            else:
                # Если нет колонки sentiment, добавляем все слова в neutral
                neutral_words.extend(words)
        
        # Считаем частоты
        from collections import Counter
        all_counter = Counter(all_words)
        positive_counter = Counter(positive_words)
        negative_counter = Counter(negative_words)
        neutral_counter = Counter(neutral_words)
        
        # Формируем результат
        result = {
            'all': [{'word': w, 'count': c} for w, c in all_counter.most_common(100) if w and w != 'nan'],
            'positive': [{'word': w, 'count': c} for w, c in positive_counter.most_common(100) if w and w != 'nan'],
            'negative': [{'word': w, 'count': c} for w, c in negative_counter.most_common(100) if w and w != 'nan'],
            'neutral': [{'word': w, 'count': c} for w, c in neutral_counter.most_common(100) if w and w != 'nan']
        }
        
        return {"success": True, "data": result}
        
    except Exception as e:
        print(f"Ошибка в wordcloud/frequencies: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/wordcloud/single")
async def get_word_frequencies_single(request: SingleWordCloudRequest):
    """
    Анализ частотности слов для одного текста
    """
    text = request.text
    sentiment = request.sentiment
    
    if not text or len(text) < 50:
        return {"success": True, "data": {"words": []}}
    
    # Очищаем текст
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\sа-яё]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Токенизация
    words = text.split()
    
    # Удаляем стоп-слова и короткие слова
    filtered_words = [w for w in words if w not in RUSSIAN_STOPWORDS and len(w) > 2]
    
    # Считаем частоты
    word_counts = Counter(filtered_words)
    
    # Берем топ-30 слов
    top_words = [{"word": word, "count": count, "sentiment": sentiment} 
                 for word, count in word_counts.most_common(30)]
    
    return {"success": True, "data": {"words": top_words}}

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ENDPOINTS
# ============================================

@app.get("/api/models")
async def get_models():
    """Получить список доступных моделей"""
    return AVAILABLE_MODELS

@app.get("/api/health")
async def health_check():
    """Проверка работоспособности"""
    return {
        "status": "ok",
        "model_loaded": analyzer.model_name,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    