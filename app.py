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

from sentiment_analyzer import SentimentAnalyzer, AVAILABLE_MODELS

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)