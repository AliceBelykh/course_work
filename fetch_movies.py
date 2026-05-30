import requests
import time
import pandas as pd
from typing import List, Dict, Any

# ============================================
# КОНФИГУРАЦИЯ (!!! ЗАМЕНИТЕ НА ВАШ КЛЮЧ !!!)
# ============================================
API_KEY = "T9TDW2N-A0X4JRK-JNAW3WD-5TVMBN2"  # <-- ВСТАВЬТЕ ЕГО СЮДА
BASE_URL = "https://api.poiskkino.dev/v1.4"  # ← НОВЫЙ АДРЕС

# Параметры поиска
SEARCH_PARAMS = {
    #"year": "2020-2025",
    #"rating.kp": "7-10",
    "type": "movie",
    "sortField": "votes.kp",      # Сортировка по количеству голосов на Кинопоиске
    "sortType": -1,                # -1 = по убыванию (самые популярные в начале)
    "isSeries": False,
    "limit": 250,
}

TOTAL_MOVIES_TO_FETCH = 3000

# ============================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С API
# ============================================
headers = {
    "accept": "application/json",
    "X-API-KEY": API_KEY,
}

def fetch_movies_page(page_num: int) -> List[Dict[str, Any]]:
    """Получает одну страницу с фильмами"""
    params = SEARCH_PARAMS.copy()
    params["page"] = page_num

    try:
        response = requests.get(f"{BASE_URL}/movie", headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("docs", [])
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе страницы {page_num}: {e}")
        return []

def fetch_all_movies(target_count: int = TOTAL_MOVIES_TO_FETCH) -> List[Dict[str, Any]]:
    """Собирает фильмы, пока не наберет нужное количество"""
    all_movies = []
    page = 1

    print(f"🚀 Начинаем сбор данных. Цель: {target_count} фильмов.")
    print(f"📋 API: {BASE_URL}")

    while len(all_movies) < target_count:
        print(f"  📄 Загружаем страницу {page}...")
        movies_on_page = fetch_movies_page(page)

        if not movies_on_page:
            print(f"  ⚠️ На странице {page} нет данных.")
            break

        new_movies = [m for m in movies_on_page if m not in all_movies]
        all_movies.extend(new_movies)

        print(f"  ✅ Добавлено {len(new_movies)} фильмов. Всего: {len(all_movies)}/{target_count}")

        if len(movies_on_page) < SEARCH_PARAMS["limit"]:
            print("  🏁 Последняя страница.")
            break

        page += 1
        time.sleep(0.2)

    print(f"🎉 Сбор данных завершен. Получено {len(all_movies)} фильмов.")
    return all_movies[:target_count]

def convert_to_dataframe(movies_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Преобразует данные в DataFrame"""
    processed_movies = []
    for movie in movies_data:
        # У нового API поле называется 'title'
        title = movie.get('title') or movie.get('name') or movie.get('alternativeName', 'Без названия')
        
        processed_movies.append({
            'Title': title,
            'kinopoiskId': movie.get('id'),
            'Year': movie.get('year'),
            'Rating Kinopoisk': movie.get('rating', {}).get('kp')
        })
    
    df = pd.DataFrame(processed_movies)
    initial_len = len(df)
    df = df.dropna(subset=['kinopoiskId'])
    print(f"🧹 Удалено {initial_len - len(df)} записей без ID.")
    return df

# ============================================
# ЗАПУСК
# ============================================
if __name__ == "__main__":
    if API_KEY == "ВАШ_API_КЛЮЧ_ОТ_TELEGRAM_БОТА":
        print("❌ ОШИБКА: Вы не указали API-ключ!")
        print("   Получите его в Telegram у бота @kinopoiskdev_bot")
    else:
        raw_movies = fetch_all_movies()
        if raw_movies:
            df_movies = convert_to_dataframe(raw_movies)
            output_filename = "movies_extended.csv"
            df_movies.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"\n✨ Готово! Файл: '{output_filename}'")
            print(f"📊 Всего: {len(df_movies)} фильмов")
            print(df_movies.head())
        else:
            print("❌ Не удалось получить данные.")