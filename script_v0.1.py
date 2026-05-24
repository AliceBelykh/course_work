# # script_v0.1_rubert.py
# # Использование RuBERT для анализа тональности

# import pandas as pd
# import matplotlib.pyplot as plt
# import re
# import os

# # Установка: pip install pandas matplotlib transformers torch

# from transformers import pipeline
# import torch

# # 1. Инициализация модели RuBERT для русского языка
# print("Загружаю RuBERT модель...")
# # Используем легковесную модель (~30 МБ) для быстрой работы
# # Альтернативы: 'cointegrated/rubert-tiny-sentiment-balanced' (лучший баланс)
# # или 'blanchefort/rubert-base-cased-sentiment' (более точная, но тяжелая)
# MODEL_NAME = 'blanchefort/rubert-base-cased-sentiment'

# # Проверяем наличие GPU
# device = 0 if torch.cuda.is_available() else -1
# print(f"Использую {'GPU' if device == 0 else 'CPU'}")

# sentiment_pipeline = pipeline(
#     'sentiment-analysis',
#     model=MODEL_NAME,
#     tokenizer=MODEL_NAME,
#     device=device
# )

# # 2. Базовые эмодзи для ответов (можно расширить)
# def sentiment_to_emoji(sentiment):
#     emojis = {
#         'positive': '😊',
#         'negative': '😞',
#         'neutral': '😐'
#     }
#     return emojis.get(sentiment, '🤔')

# # 3. Очистка текста (минимальная, RuBERT сам хорошо обрабатывает шум)
# def clean_text(text):
#     text = str(text)
#     # Удаляем ссылки
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
#     # Удаляем упоминания и хэштеги (опционально)
#     text = re.sub(r'@\w+|#\w+', '', text)
#     # Убираем лишние пробелы
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text if text else "пустой отзыв"

# # 4. Загрузка данных (мини-набор для теста)
# SAMPLE_DATA = [
#     {"text": "Гениальный фильм, потрясающие актёры и музыка!", "label": "positive"},
#     {"text": "Скучно, сценарий предсказуемый, всё затянуто.", "label": "negative"},
#     {"text": "Нормальное кино, ничего особенного.", "label": "neutral"},
#     {"text": "Отличная атмосфера, рекомендую к просмотру.", "label": "positive"},
#     {"text": "Плохая игра актёров, диалоги неестественные.", "label": "negative"},
#     {"text": "Фильм неплохой, но есть над чем работать", "label": "mixed"},  # добавим для теста
#     {"text": "Отвратительный фильм, гавно", "label": "negative"}
# #     {"text": """
# # Иногда попадаются фильмы, после которых сидишь и еще долго не можешь переключиться на что-то другое. Вроде бы уже пошли титры, а мысли все крутятся вокруг того, что только что увидел. Для меня таким фильмом стал «Облачный атлас». Это не то кино, которое просто посмотрел вечером и забыл. Оно большое, сложное и очень всеобъемлющее.

# # Картина снята сразу тремя режиссерами — Лана Вачовски, Лилли Вачовски и Том Тыквер. И замах у них получился серьезный. В фильме сразу шесть историй, происходящих в разных эпохах — от XIX века до далекого будущего. Сначала кажется, что это просто разные сюжеты. Но постепенно понимаешь: между ними есть связь. Судьбы людей словно перекликаются через время, а поступки из одной эпохи могут отозваться в другой.

# # Структура у фильма, мягко говоря, непростая. Истории постоянно перебрасывают зрителя из одного времени в другое. Только привык к одной линии — и тебя уже уносят в другую. Но через какое-то время начинаешь втягиваться. Появляется ощущение, будто смотришь огромную мозаику, где кусочки постепенно складываются в цельную картину. И чем дальше, тем интереснее находить эти связи.

# # Отдельным удовольствием для меня стал актерский состав. В фильме собралась очень сильная команда: Том Хэнкс, Холли Берри, Хью Грант, Сьюзен Сарандон, Хьюго Уивинг, Джим Бродбент, Джим Стерджесс, Бен Уишоу, Пэ Ду-на, Кит Дэвид, Дэвид Джеси. И самое интересное — почти каждый из них играет сразу несколько ролей. Причем иногда узнать актера под гримом вообще невозможно. У меня был забавный момент: в линии про жителей Маона-Сол появляется охотник, и только уже после фильма я понял, что это Хью Грант. Пока не увидел титры — даже не догадался.

# # Каждая из историй затрагивает какую-то свою проблему. Где-то речь идет о положении женщин и борьбе за равные права. В другой линии поднимается тема отношения к пожилым людям и того, как общество иногда пытается их просто списать со счетов. Есть история про расизм и культурные различия. Все эти темы подаются через судьбы конкретных людей, поэтому воспринимаются не как лекция, а как живые истории.

# # Интересно и то, что фильм смотрит не только назад, но и вперед. Одна из линий разворачивается в будущем и поднимает довольно тревожные вопросы. Например: как люди будут относиться к клонам? Станут ли они для нас равными или человечество снова найдет повод разделить всех на «своих» и «чужих»? И еще одна мысль — про технологии. Что будет, если однажды они станут настолько сложными, что начнут казаться магией?

# # Сам сценарий получился насыщенным. Здесь много размышлений о свободе, о справедливости, о том, как люди связаны между собой. Фильм умеет и рассмешить, и ударить по эмоциям. После него остается странное чувство — легкая грусть. Грусть о том, через что человечество прошло, и о том, сколько проблем все еще остается.

# # Визуально картина тоже впечатляет. Каждая эпоха выглядит в картине по-своему: тут и морские путешествия XIX века есть, и неоновый футуристический мегаполис в Азии, и привычная современность, и далекое постапокалиптическое будущее. Реализовать постарались с размахом. Все это снято очень ярко и атмосферно. Плюс к этому — достойный саундтрек, который отлично связывает разные части фильма.

# # Но главное, что остается после просмотра — мысли. «Облачный атлас» предлагает подумать. Подумать о людях и о том, почему история так часто повторяет одни и те же ошибки. Из раза в раз. Каждый век. И вновь одни считают себя выше других. Фильм показывает, как легко рождаются предрассудки и вытекающая из них дискриминация. И одновременно напоминает, что всегда находятся люди, которые готовы с этим бороться.

# # Хочется верить, что когда-нибудь человечество все-таки научится быть добрее друг к другу. Пока это звучит скорее как мечта. Но такие фильмы, как «Облачный атлас», хотя бы заставляют об этом задуматься. А значит, они уже делают что-то важное.

# # 10 из 10""", "label": "positive"},
# ]

# # Если есть CSV, раскомментируйте строку ниже:
# # df = pd.read_csv("reviews_draft.csv", encoding="utf-8")
# df = pd.DataFrame(SAMPLE_DATA)

# # 5. Предобработка
# df["clean"] = df["text"].apply(clean_text)

# # 6. Предсказание тональности с помощью RuBERT
# print("Запускаю классификацию RuBERT...")
# results = []
# for idx, text in enumerate(df["clean"].tolist()):
#     try:
#         # Получаем предсказание
#         pred = sentiment_pipeline(text)[0]
#         sentiment = pred['label'].lower()
#         confidence = pred['score']

#         # Нормализуем метки (модель может возвращать 'positive'/'negative'/'neutral')
#         if sentiment == 'positive':
#             final_sentiment = 'positive'
#         elif sentiment == 'negative':
#             final_sentiment = 'negative'
#         else:
#             final_sentiment = 'neutral'

#         results.append({
#             'predicted': final_sentiment,
#             'confidence': confidence,
#             'emoji': sentiment_to_emoji(final_sentiment)
#         })

#         print(f"  {idx+1}/{len(df)}: {final_sentiment} ({confidence:.2f})")

#     except Exception as e:
#         print(f"Ошибка при обработке текста {idx}: {e}")
#         results.append({
#             'predicted': 'neutral',
#             'confidence': 0.0,
#             'emoji': '❓'
#         })

# df['predicted'] = [r['predicted'] for r in results]
# df['confidence'] = [r['confidence'] for r in results]
# df['emoji'] = [r['emoji'] for r in results]

# # 7. Визуализация результатов
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Первый график: распределение предсказанных тональностей
# counts_pred = df["predicted"].value_counts()
# colors_pred = {
#     'positive': '#4CAF50',
#     'negative': '#F44336',
#     'neutral': '#FFC107'
# }
# pred_colors = [colors_pred.get(label, '#999999') for label in counts_pred.index]

# axes[0].pie(counts_pred, labels=counts_pred.index, autopct='%1.0f%%', 
#             colors=pred_colors, startangle=90)
# axes[0].set_title("Распределение предсказанных тональностей", fontsize=12, pad=20)

# # Второй график: уверенность модели по классам
# df_grouped = df.groupby('predicted')['confidence'].agg(['mean', 'std']).reset_index()
# x_pos = range(len(df_grouped))
# axes[1].bar(x_pos, df_grouped['mean'], yerr=df_grouped['std'], 
#             capsize=5, color=[colors_pred.get(c, '#999999') for c in df_grouped['predicted']])
# axes[1].set_xticks(x_pos)
# axes[1].set_xticklabels(df_grouped['predicted'])
# axes[1].set_ylabel('Средняя уверенность')
# axes[1].set_ylim(0, 1)
# axes[1].set_title('Уверенность модели по классам', fontsize=12)
# axes[1].grid(True, alpha=0.3, axis='y')

# plt.tight_layout()
# plt.savefig("rubert_sentiment_analysis.png", dpi=150, bbox_inches='tight')
# print("\n📊 График сохранён как rubert_sentiment_analysis.png")

# # 8. Вывод результатов с эмодзи и уверенностью
# print("\n" + "="*80)
# print("РЕЗУЛЬТАТЫ АНАЛИЗА ТОНАЛЬНОСТИ")
# print("="*80)
# display_df = df[["text", "label", "predicted", "confidence", "emoji"]].copy()
# display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.2f}")
# print(display_df.to_string(index=False))

# # 9. Базовая оценка качества (если есть реальные метки)
# if "label" in df.columns:
#     from sklearn.metrics import classification_report, accuracy_score
    
#     # Фильтруем только валидные метки
#     valid_labels = ['positive', 'negative', 'neutral']
#     df_eval = df[df['label'].isin(valid_labels) & df['predicted'].isin(valid_labels)]
    
#     if len(df_eval) > 0:
#         print("\n" + "="*80)
#         print("ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
#         print("="*80)
#         accuracy = accuracy_score(df_eval['label'], df_eval['predicted'])
#         print(f"Accuracy: {accuracy:.2%}")
        
#         print("\nClassification Report:")
#         print(classification_report(
#             df_eval['label'], 
#             df_eval['predicted'],
#             labels=valid_labels,
#             target_names=valid_labels,
#             zero_division=0
#         ))

# # 10. Сохранение результатов в CSV
# output_file = "sentiment_results_rubert.csv"
# df.to_csv(output_file, index=False, encoding='utf-8-sig')
# print(f"\n💾 Результаты сохранены в {output_file}")

# # 11. Дополнительная статистика
# print("\n" + "="*80)
# print("СТАТИСТИКА ПО ТЕКСТАМ")
# print("="*80)
# print(f"Всего обработано: {len(df)} текстов")
# print(f"Средняя уверенность модели: {df['confidence'].mean():.2f}")
# print(f"Медианная уверенность: {df['confidence'].median():.2f}")
# print(f"Текстов с низкой уверенностью (<0.5): {(df['confidence'] < 0.5).sum()}")

# # TODO на следующие итерации:
# # 1. Добавить обработку больших файлов чанками (batch processing)
# # 2. Внедрить кэширование результатов для повторного использования
# # 3. Добавить визуализацию распределения уверенности (гистограмма)
# # 4. Реализовать экспорт в Excel с форматированием
# # 5. Добавить поддержку потоковой обработки через генераторы
# # 6. Настроить логирование ошибок в файл