# # script_v0.3_rubert_long_texts.py
# # Исправленная версия с поддержкой длинных текстов

# import pandas as pd
# import matplotlib.pyplot as plt
# import re
# import numpy as np
# from transformers import pipeline
# import torch

# # ============================================
# # 1. КЛАСС ДЛЯ ОБРАБОТКИ ДЛИННЫХ ТЕКСТОВ
# # ============================================
# class LongTextSentimentAnalyzer:
#     def __init__(self, model_name='cointegrated/rubert-tiny-sentiment-balanced', max_length=512):
#         self.max_length = max_length
#         self.device = 0 if torch.cuda.is_available() else -1

#         print(f"Загружаю модель {model_name}...")
#         self.pipeline = pipeline(
#             'sentiment-analysis',
#             model=model_name,
#             tokenizer=model_name,
#             device=self.device,
#             truncation=True,  # Обрезаем длинные тексты
#             max_length=max_length
#         )

#     def split_text(self, text, max_chunk_size=400):
#         """
#         Разбивает длинный текст на части по предложениям
#         """
#         # Разбиваем по предложениям (.!?;)
#         sentences = re.split(r'([.!?;]+)', str(text))
        
#         # Собираем предложения в чанки
#         chunks = []
#         current_chunk = ""
        
#         for i in range(0, len(sentences), 2):
#             sentence = sentences[i]
#             delimiter = sentences[i+1] if i+1 < len(sentences) else ""
#             full_sentence = sentence + delimiter
            
#             if len(current_chunk + full_sentence) < max_chunk_size:
#                 current_chunk += full_sentence
#             else:
#                 if current_chunk:
#                     chunks.append(current_chunk.strip())
#                 current_chunk = full_sentence
        
#         if current_chunk:
#             chunks.append(current_chunk.strip())
        
#         return chunks if chunks else [text[:max_chunk_size]]
    
#     def analyze_long_text(self, text, strategy='auto'):
#         """
#         Анализирует длинный текст с разными стратегиями
        
#         Стратегии:
#         - 'auto': автоматический выбор лучшей стратегии
#         - 'truncate': просто обрезать до 512 токенов (быстро)
#         - 'split': разбить на части и усреднить (точно)
#         - 'weighted': взвешенное усреднение по частям (рекомендуется)
#         """
#         text_length = len(text)
        
#         # Быстрая проверка длины токенов
#         tokens = self.pipeline.tokenizer.encode(text, truncation=False)
#         token_length = len(tokens)
        
#         print(f"   Длина текста: {text_length} символов, {token_length} токенов")
        
#         # Стратегия 1: Обрезание (если текст не сильно длинный)
#         if strategy == 'truncate' or (strategy == 'auto' and token_length < 600):
#             result = self.pipeline(text[:self.max_length * 4])[0]
#             return {
#                 'sentiment': result['label'].lower(),
#                 'confidence': result['score'],
#                 'method': 'truncate'
#             }
        
#         # Стратегия 2: Разбиение на части
#         if strategy == 'split' or strategy == 'weighted' or strategy == 'auto':
#             # Разбиваем текст на чанки
#             chunks = self.split_text(text, max_chunk_size=400)
#             print(f"   Текст разбит на {len(chunks)} частей")
            
#             # Анализируем каждый чанк
#             chunk_results = []
#             for i, chunk in enumerate(chunks):
#                 try:
#                     result = self.pipeline(chunk[:self.max_length * 4])[0]
#                     chunk_results.append({
#                         'sentiment': result['label'].lower(),
#                         'confidence': result['score'],
#                         'length': len(chunk)
#                     })
#                 except Exception as e:
#                     print(f"   Ошибка в чанке {i+1}: {e}")
#                     chunk_results.append({
#                         'sentiment': 'neutral',
#                         'confidence': 0.5,
#                         'length': len(chunk)
#                         })
            
#             if not chunk_results:
#                 return {'sentiment': 'neutral', 'confidence': 0.5, 'method': 'error'}
            
#             if strategy == 'weighted':
#                 # Взвешенное усреднение (длинные части важнее)
#                 total_weight = sum(r['length'] for r in chunk_results)
#                 sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
                
#                 for r in chunk_results:
#                     weight = r['length'] / total_weight
#                     if r['sentiment'] == 'positive':
#                         sentiment_scores['positive'] += weight
#                     elif r['sentiment'] == 'negative':
#                         sentiment_scores['negative'] += weight
#                     else:
#                         sentiment_scores['neutral'] += weight
                
#                 final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
#                 final_confidence = max(sentiment_scores.values())
                
#                 return {
#                     'sentiment': final_sentiment,
#                     'confidence': final_confidence,
#                     'method': f'weighted_avg({len(chunks)}_chunks)'
#                 }
#             else:
#                 # Простое усреднение
#                 sentiments = [r['sentiment'] for r in chunk_results]
#                 final_sentiment = max(set(sentiments), key=sentiments.count)
#                 avg_confidence = np.mean([r['confidence'] for r in chunk_results])
                
#                 return {
#                     'sentiment': final_sentiment,
#                     'confidence': avg_confidence,
#                     'method': f'split_avg({len(chunks)}_chunks)'
#                 }
        
#         return {'sentiment': 'neutral', 'confidence': 0.5, 'method': 'fallback'}

# # ============================================
# # 2. НЕГАТИВНЫЕ ПАТТЕРНЫ (как в прошлой версии)
# # ============================================
# NEGATIVE_PATTERNS = [
#     r'(плох[а-я]+)\s+(игр[а-я]+)',
#     r'(неестественн[а-я]+)\s+(диалог[а-я]+)',
#     r'(ужасн[а-я]+)\s+(игр[а-я]+)',
#     r'(сценар[а-я]+)\s+(плох[а-я]+|слаб[а-я]+)',
# ]

# STRONG_NEGATIVE_WORDS = ['плохой', 'ужасный', 'отвратительный', 'неестественный']

# def enhanced_analysis(text, analyzer):
#     """
#     Комбинированный анализ с обработкой длинных текстов
#     """
#     text_lower = str(text).lower()
    
#     # Проверка на негативные паттерны
#     # for pattern in NEGATIVE_PATTERNS:
#     #     if re.search(pattern, text_lower):
#     #         return {
#     #             'predicted': 'negative',
#     #             'confidence': 0.85,
#     #             'method': 'pattern_match'
#     #         }
    
#     # Анализ длинного текста
#     result = analyzer.analyze_long_text(text, strategy='weighted')
    
#     # Коррекция на основе негативных слов
#     negative_count = sum(1 for word in STRONG_NEGATIVE_WORDS if word in text_lower)
#     if negative_count > 0 and result['sentiment'] == 'positive' and result['confidence'] < 0.7:
#         result['sentiment'] = 'negative'
#         result['method'] += '_corrected'
    
#     return {
#         'predicted': result['sentiment'],
#         'confidence': result['confidence'],
#         'method': result.get('method', 'unknown')
#     }

# # ============================================
# # 3. ТЕСТОВЫЕ ДАННЫЕ (включая длинный отзыв)
# # ============================================
# # Генерируем длинный отзыв (1177 символов)
# LONG_REVIEW = """
# Фильм оставил двоякое впечатление. С одной стороны, операторская работа и визуальный ряд 
# выполнены на высоком уровне. Красивые пейзажи, интересные ракурсы, качественный монтаж. 
# Саундтрек тоже порадовал - музыка атмосферная и точно попадает в настроение сцен.

# Но теперь о главных недостатках. Игра актёров оставляет желать лучшего. Главный герой 
# выглядит неестественно, его диалоги звучат фальшиво. Особенно разочаровали сцены, где 
# требовалось показать сильные эмоции - актёр просто не справился. Второстепенные персонажи 
# тоже не спасают ситуацию.

# Сценарий - это отдельная боль. Он предсказуемый от начала и до конца. Все повороты сюжета 
# угадываются за десять минут до их наступления. Диалоги неестественные, многие фразы 
# выглядят так, будто их писал школьник. Юмор плоский, драматические моменты не трогают.

# Актёрский состав в целом подобран неудачно. Химии между персонажами нет, их отношения 
# выглядят натянутыми. Пары, которые должны были вызывать симпатию, раздражают.

# В итоге: красивый фасад не спасает пустое содержание. Жаль потраченного времени и денег.
# """

# TEST_DATA = [
#     {"text": "Гениальный фильм, потрясающие актёры и музыка!", "expected": "positive"},
#     {"text": "Плохая игра актёров, диалоги неестественные.", "expected": "negative"},
#     {"text": LONG_REVIEW, "expected": "negative"},  # Длинный отзыв 1177 символов
#     {"text": "Скучно, сценарий предсказуемый, всё затянуто.", "expected": "negative"},
#     {"text": "Нормальное кино, ничего особенного.", "expected": "neutral"},
# ]

# # ============================================
# # 4. ЗАПУСК АНАЛИЗА
# # ============================================
# print("="*80)
# print("АНАЛИЗ ТОНАЛЬНОСТИ С ПОДДЕРЖКОЙ ДЛИННЫХ ТЕКСТОВ")
# print("="*80)

# # Инициализируем анализатор
# analyzer = LongTextSentimentAnalyzer(max_length=512)

# # Анализируем каждый отзыв
# results = []
# df = pd.DataFrame(TEST_DATA)

# print("\n🔍 Запускаю анализ...\n")
# for idx, row in df.iterrows():
#     text = row['text']
#     expected = row['expected']

#     print(f"\n📝 Отзыв {idx+1}:")
#     print(f"   Длина: {len(text)} символов")

#     result = enhanced_analysis(text, analyzer)
#     results.append(result)

#     status = "✅" if result['predicted'] == expected else "❌"
#     print(f"   {status} Ожидалось: {expected} | Получено: {result['predicted']}")
#     print(f"   Уверенность: {result['confidence']:.2f} | Метод: {result['method']}")

# # Сохраняем результаты
# df['predicted'] = [r['predicted'] for r in results]
# df['confidence'] = [r['confidence'] for r in results]
# df['method'] = [r['method'] for r in results]

# # ============================================
# # 5. СРАВНЕНИЕ СТРАТЕГИЙ ДЛЯ ДЛИННОГО ТЕКСТА
# # ============================================
# print("\n" + "="*80)
# print("СРАВНЕНИЕ СТРАТЕГИЙ ДЛЯ ДЛИННОГО ОТЗЫВА")
# print("="*80)

# long_text = TEST_DATA[2]['text']
# strategies = ['truncate', 'split', 'weighted']

# for strategy in strategies:
#     result = analyzer.analyze_long_text(long_text, strategy=strategy)
#     print(f"\nСтратегия '{strategy}':")
#     print(f"  Результат: {result['sentiment']}")
#     print(f"  Уверенность: {result['confidence']:.3f}")
#     print(f"  Метод: {result.get('method', strategy)}")

# # ============================================
# # 6. ВИЗУАЛИЗАЦИЯ
# # ============================================
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # График 1: Сравнение точности
# accuracy = (df['expected'] == df['predicted']).mean()
# axes[0].bar(['Точность'], [accuracy], color='green' if accuracy > 0.8 else 'orange')
# axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
# axes[0].set_ylim(0, 1)
# axes[0].set_ylabel('Accuracy')
# axes[0].set_title(f'Общая точность: {accuracy:.1%}')
# axes[0].text(0, accuracy/2, f'{accuracy:.1%}', ha='center', fontsize=14, fontweight='bold')

# # График 2: Уверенность по разным стратегиям
# df_methods = df[df['method'].str.contains('chunks|weighted', na=False)]
# if len(df_methods) > 0:
#     methods = df_methods['method'].values
#     confidences = df_methods['confidence'].values
#     bars = axes[1].bar(range(len(methods)), confidences, color=['#4CAF50', '#FFC107'])
#     axes[1].set_xticks(range(len(methods)))
#     axes[1].set_xticklabels(methods, rotation=15, ha='right')
#     axes[1].set_ylabel('Уверенность')
#     axes[1].set_title('Сравнение стратегий обработки длинных текстов')
#     axes[1].set_ylim(0, 1)
    
#     # Добавляем значения на график
#     for bar, conf in zip(bars, confidences):
#         axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
#                     f'{conf:.2f}', ha='center', fontsize=10)

# plt.tight_layout()
# plt.savefig("long_text_sentiment_analysis.png", dpi=150, bbox_inches='tight')
# print(f"\n📊 График сохранён как long_text_sentiment_analysis.png")

# # ============================================
# # 7. ВЫВОД РЕЗУЛЬТАТОВ
# # ============================================
# print("\n" + "="*80)
# print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
# print("="*80)
# display_df = df[["text", "expected", "predicted", "confidence", "method"]].copy()
# display_df["text"] = display_df["text"].apply(lambda x: x[:80] + "..." if len(x) > 80 else x)
# display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.2f}")
# print(display_df.to_string(index=False))

# # Сохраняем результаты
# df.to_csv("long_text_results.csv", index=False, encoding='utf-8-sig')
# print(f"\n💾 Результаты сохранены в long_text_results.csv")