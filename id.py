https://giga.chat/link/gcsKxuRtuw
Отлично, это даже более удобный и надежный вариант! Работа с **ID фильма** (целым числом) гораздо предпочтительнее, чем с названием (строкой), потому что:
*   **Уникальность:** У двух разных фильмов могут быть одинаковые названия (например, "Побег" или "Гладиатор"), но ID всегда уникален.
*   **Отсутствие ошибок:** Легко ошибиться в названии ("Иллюзия обмана" или "Иллюзия обмана 2"), а ID — это просто число, которое не перепутаешь.

Ваша задача — модифицировать скрипт так, чтобы он принимал на вход **ID фильма** и анализировал только связанные с ним отзывы.

Вот как это сделать.

### Шаг 1: Подготовка структуры данных

Убедитесь, что ваш датасет (CSV-файл) содержит как минимум две колонки:
1.  `movie_id` — уникальный идентификатор фильма (например, `12345`).
2.  `text` или `review` — текст самого отзыва.

*Пример структуры вашего CSV (`reviews.csv`):*
| **movie_id** | **text** |
| :--- | :--- |
| 123 | "Фильм просто супер!" |
| 123 | "Не очень, ожидал большего." |
| 456 | "Лучший фильм года!" |

### Шаг 2: Модификация скрипта

Вам нужно внести несколько ключевых изменений в код.

#### 1. Добавление переменной для ID
В самом начале скрипта, где задаются параметры, добавьте переменную с ID фильма, который хотите проанализировать.

```python
# ID фильма, который мы хотим проанализировать
MOVIE_ID_TO_ANALYZE = 123 # Замените на нужный ID!
```

#### 2. Фильтрация DataFrame по ID
Сразу после загрузки данных из CSV добавьте блок фильтрации. Это самый важный шаг.

```python
# Загрузка данных из файла
df = pd.read_csv("reviews.csv", encoding="utf-8")

# --- НОВЫЙ БЛОК: Фильтрация по ID ---
# Проверяем, есть ли в датасете колонка с ID фильма
if 'movie_id' in df.columns:
    print(f"Фильтруем отзывы для фильма с ID: {MOVIE_ID_TO_ANALYZE}")
    # Фильтруем DataFrame, оставляя только строки с нужным movie_id
    df_filtered = df[df['movie_id'] == MOVIE_ID_TO_ANALYZE].copy()
    
    # Проверяем, нашли ли мы отзывы для этого ID
    if df_filtered.empty:
        print(f"Внимание! Отзывов для фильма с ID {MOVIE_ID_TO_ANALYZE} не найдено.")
        # Выходим из скрипта, чтобы не строить пустые графики
        # Если хотите анализировать все отзывы при ошибке, уберите этот exit()
        exit()
else:
    print("Колонка 'movie_id' не найдена. Анализ невозможен.")
    exit()
```
*Примечание: `exit()` останавливает выполнение скрипта, если отзывов не найдено. Это лучше, чем строить графики из ничего.*

#### 3. Использование отфильтрованных данных (`df_filtered`)
Теперь везде в коде, где раньше использовался `df`, замените его на `df_filtered`.

Найдите строки:
*   `df["clean"] = df["text"].apply(clean_text)` -> замените на `df_filtered["clean"] = df_filtered["text"].apply(clean_text)`
*   `for idx, text in enumerate(df["clean"].tolist()):` -> замените на `for idx, text in enumerate(df_filtered["clean"].tolist()):`
*   И так далее по всему коду анализа и построения графиков.

#### 4. Уникальное имя файла для графика
Чтобы графики не перезаписывались, давайте им имена на основе ID.

Найдите строку сохранения графика:
```python
plt.savefig("rubert_sentiment_analysis.png", dpi=150, bbox_inches='tight')
```
И замените её на:
```python
plt.savefig(f"rubert_sentiment_analysis_id_{MOVIE_ID_TO_ANALYZE}.png", dpi=150, bbox_inches='tight')
print(f"\n📊 График сохранен как rubert_sentiment_analysis_id_{MOVIE_ID_TO_ANALYZE}.png")
```

### Как теперь пользоваться скриптом?

1.  Откройте ваш CSV-файл и посмотрите, какой **ID** у интересующего вас фильма.
2.  Вставьте этот номер в переменную `MOVIE_ID_TO_ANALYZE = ...` в начале скрипта.
3.  Запустите скрипт.
4.  Он проанализирует только отзывы для этого конкретного ID и сохранит график с соответствующим именем файла.

Конечно, я помогу вам написать такой скрипт. Мы будем использовать библиотеку `requests` для загрузки веб-страницы и `BeautifulSoup` для извлечения названия из HTML-кода.

> **Важное предупреждение:** Парсинг (scraping) сайтов может нарушать их правила использования. "Кинопоиск" может блокировать IP-адреса, которые отправляют слишком много автоматических запросов. Используйте этот скрипт аккуратно, с задержками между запросами и только в образовательных целях.

### Шаг 1: Установка необходимых библиотек

Если у вас еще не установлены библиотеки для работы с сетью и HTML, откройте терминал в Visual Studio (где активировано ваше виртуальное окружение `venv`) и выполните команду:

```bash
pip install requests beautifulsoup4
```

### Шаг 2: Написание скрипта парсера

Создайте новый файл, например `kinopoisk_parser.py`, и вставьте в него следующий код.

```python
import requests
from bs4 import BeautifulSoup

def get_movie_title_by_id(movie_id):
    """
    Функция получает название фильма на Кинопоиске по его ID.
    Возвращает строку с названием или None, если фильм не найден.
    """
    # 1. Формируем URL-адрес
    url = f"https://www.kinopoisk.ru/film/{movie_id}/"
    
    # 2. Загружаем HTML-код страницы
    # Используем заголовок User-Agent, чтобы имитировать браузер
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # 3. Проверяем, успешно ли прошла загрузка
        if response.status_code == 200:
            # 4. Парсим HTML с помощью BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 5. Ищем элемент с названием фильма
            # Название находится в теге <h1> с классом 'moviename-big'
            title_element = soup.find('h1', class_='moviename-big')
            
            if title_element:
                # 6. Извлекаем и очищаем текст
                return title_element.get_text(strip=True)
            else:
                print(f"⚠️ Не удалось найти блок с названием для ID {movie_id}. Структура сайта могла измениться.")
                return None
                
        elif response.status_code == 404:
            print(f"❌ Фильм с ID {movie_id} не найден.")
            return None
        else:
            print(f"⚠️ Ошибка доступа к сайту. Код ответа: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Ошибка сети: {e}")
        return None

# --- Блок для запуска скрипта из командной строки ---
if __name__ == "__main__":
    # Проверяем, передал ли пользователь ID в аргументах командной строки
    import sys
    
    if len(sys.argv) != 2:
        print("Использование: python kinopoisk_parser.py <ID_ФИЛЬМА>")
        print("Пример: python kinopoisk_parser.py 447301")
        sys.exit(1)
        
    movie_id = sys.argv[1]
    
    # Проверяем, что аргумент является числом
    if not movie_id.isdigit():
        print("❌ Ошибка: ID фильма должен быть числом.")
        sys.exit(1)
        
    title = get_movie_title_by_id(movie_id)
    
    if title:
        print(f"✅ Название фильма: {title}")

```

### Как запустить этот скрипт?

1.  Сохраните код выше в файл `kinopoisk_parser.py`.
2.  Откройте терминал в Visual Studio (убедитесь, что окружение `venv` активно).
3.  Выполните команду, заменив `447301` на любой другой ID фильма.

```bash
python kinopoisk_parser.py 447301
```

**Пример вывода в консоль:**
```
✅ Название фильма: Начало
```

### Как использовать этот скрипт для создания вашего словаря?

Вы можете написать еще один маленький скрипт, который будет читать список ID из файла и сохранять результат в CSV.

**Пример (`create_dictionary.py`):**
```python
import csv

# Список ID фильмов, которые нужно обработать
movie_ids = [447301, 326, 880] 

with open('movies_dictionary.csv', 'w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(['movie_id', 'movie_title_ru']) # Записываем заголовки

    for movie_id in movie_ids:
        # Здесь мы вызываем функцию из нашего парсера
        from kinopoisk_parser import get_movie_title_by_id 
        
        title = get_movie_title_by_id(movie_id)
        if title:
            writer.writerow([movie_id, title])
            print(f"Добавлено: {movie_id} - {title}")
```
Запустив этот скрипт, вы автоматически получите файл `movies_dictionary.csv` с названиями фильмов.

Отличная задача! Чтобы автоматизировать создание датасета, нам нужно объединить ваш парсер с логикой чтения из одного файла и записи в другой.

Мы создадим скрипт, который:
1.  Берет список `ID` фильмов из исходного файла (например, из вашего датасета с отзывами).
2.  Для каждого `ID` находит название на Кинопоиске.
3.  Сохраняет результат в новый CSV-файл (`movies_dictionary.csv`).

### Шаг 1: Подготовка исходного файла

У вас должен быть файл с отзывами, где есть колонка `movie_id`.
Допустим, он называется `reviews.csv` и выглядит так:

| movie_id | text |
| :--- | :--- |
| 447301 | Отличный фильм! |
| 326 | Не понравилось. |
| 447301 | Смотрел на одном дыхании. |

Обратите внимание: `ID` 447301 повторяется. В итоговом словаре он должен быть только один раз.

### Шаг 2: Создание скрипта-сборщика (Dataset Builder)

Создайте новый файл, например `build_movies_dict.py`, и вставьте в него этот код.

```python
import pandas as pd
import csv
import time
import random

# --- 1. НАСТРОЙКИ СКРИПТА ---

# Путь к вашему исходному файлу с отзывами
INPUT_FILE = "reviews.csv" 

# Путь к файлу, который мы создадим (Словарь)
OUTPUT_FILE = "movies_dictionary.csv"

# Заголовки колонок (проверьте, как они называются у вас)
ID_COLUMN = "movie_id"
TITLE_COLUMN = "movie_title_ru"

# --- 2. ФУНКЦИЯ ПАРСЕРА (из предыдущего примера) ---

def get_movie_title_by_id(movie_id):
    """
    Получает название фильма по ID с помощью парсинга сайта.
    """
    url = f"https://www.kinopoisk.ru/film/{movie_id}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title_element = soup.find('h1', class_='moviename-big')
            
            if title_element:
                return title_element.get_text(strip=True)
            else:
                print(f"🤔 Блок с названием не найден для ID {movie_id}.")
                return None
        elif response.status_code == 404:
            print(f"🚫 ID {movie_id} не существует.")
            return None
    except Exception as e:
        print(f"⚠️ Ошибка сети для ID {movie_id}: {e}")
        return None

# --- 3. ОСНОВНАЯ ЛОГИКА СБОРКИ ДАТАСЕТА ---

def main():
    # Импортируем библиотеки здесь, чтобы скрипт был чище
    import requests
    from bs4 import BeautifulSoup

    try:
        # Читаем исходный файл с отзывами
        df = pd.read_csv(INPUT_FILE)
        
        # Проверяем, есть ли нужная колонка
        if ID_COLUMN not in df.columns:
            print(f"❗ Ошибка: В файле '{INPUT_FILE}' нет колонки '{ID_COLUMN}'.")
            return

        # Получаем список уникальных ID (чтобы не парсить один и тот же фильм 100 раз)
        unique_ids = df[ID_COLUMN].dropna().unique()
        
        print(f"🔍 Найдено {len(unique_ids)} уникальных ID для обработки.")
        
        # Открываем файл для записи (создаем его)
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow([ID_COLUMN, TITLE_COLUMN]) # Записываем заголовки

            # Проходим по каждому уникальному ID
            for i, movie_id in enumerate(unique_ids):
                try:
                    movie_id = int(movie_id) # Приводим к числу на всякий случай
                    print(f"📝 Обработка {i+1}/{len(unique_ids)}: ID {movie_id}...")
                    
                    title = get_movie_title_by_id(movie_id)
                    
                    if title:
                        writer.writerow([movie_id, title])
                        print(f"✅ Успех: {title}")
                    else:
                        writer.writerow([movie_id, "Название не найдено"])
                    
                    # ЗАДЕРЖКА! Это очень важно, чтобы Кинопоиск не заблокировал ваш IP.
                    # Ждем от 1 до 3 секунд перед следующим запросом.
                    time.sleep(random.uniform(1, 3))
                        
                except (ValueError, TypeError):
                    print(f"⚠️ Пропуск некорректного ID: {movie_id}")

        print(f"\n🎉 Готово! Датасет сохранен в '{OUTPUT_FILE}'")

    except FileNotFoundError:
        print(f"❗ Файл '{INPUT_FILE}' не найден. Убедитесь, что он в той же папке, что и скрипт.")

if __name__ == "__main__":
    main()
```

### Как это работает?

1.  **Чтение данных:** Скрипт открывает ваш `reviews.csv` и ищет все уникальные значения в колонке `movie_id`.
2.  **Цикл обработки:** Для каждого `ID` он вызывает функцию `get_movie_title_by_id`.
3.  **Запись:** Если название найдено, оно записывается в новую строку файла `movies_dictionary.csv` вместе с `ID`.
4.  **Защита от блокировки:** Команда `time.sleep()` делает паузу между запросами, имитируя поведение человека и защищая вас от временной блокировки со стороны сайта.

### Как запустить?

1.  Сохраните код в файл `build_movies_dict.py`.
2.  Убедитесь, что в той же папке лежит ваш файл с отзывами `reviews.csv`.
3.  Запустите скрипт из терминала VS Code:
    ```bash
    python build_movies_dict.py
    ```
4.  Подождите. Процесс может занять время (1 секунда на фильм + время загрузки страницы). Для 100 фильмов это займет около 2-3 минут.
5.  В итоге у вас появится файл `movies_dictionary.csv`.