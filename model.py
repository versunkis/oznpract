import re
import numpy as np
import pandas as pd
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

morph = pymorphy3.MorphAnalyzer()

def lemmatize(text):
    text = text.lower()
    words = re.findall(r'\w+', text)
    lemmas = []
    for word in words:
        parsed = morph.parse(word)[0]
        lemmas.append(parsed.normal_form)
    return ' '.join(lemmas)

def read_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

try:
    law_text = read_file('docs\\115fz.txt')
    cb_text = read_file('docs\\regulation_860p.txt')
    bank_text = read_file('docs\\tbank_faq.txt')
except FileNotFoundError:
    print("Ошибка: Файлы не найдены.")
    exit()

def parse_law(text):
    elements = []
    current_chapter = ''
    current_article = ''
    article_text = ''
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        
        # Глава
        if re.match(r'^Глава\s+\d+', line):
            current_chapter = line
            
        # Статья
        elif re.match(r'^Статья\s+\d+', line):

            if current_article != '' and article_text.strip():
                paragraphs = article_text.strip().split('\n\n')
                for i, para in enumerate(paragraphs):
                    if len(para.strip()) > 50:
                        elements.append({
                            'source': '115-ФЗ',
                            'chapter': current_chapter,
                            'element': f'{current_article} (ч. {i+1})',
                            'text': para.strip()
                        })
            
            current_article = line
            article_text = ''
            
        else:
            article_text += '\n' + line

    if current_article != '' and article_text.strip():
        paragraphs = article_text.strip().split('\n\n')
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 50:
                elements.append({
                    'source': '115-ФЗ',
                    'chapter': current_chapter,
                    'element': f'{current_article} (ч. {i+1})',
                    'text': para.strip()
                })
    
    return elements

def parse_cb(text):
    elements = []
    current_chapter = ''
    current_point = ''
    point_text = ''
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if re.match(r'^Глава\s+\d+', line):
            current_chapter = line
        elif re.match(r'^\d+\.\d+', line):
            if current_point != '':
                elements.append({
                    'source': 'Положение ЦБ',
                    'chapter': current_chapter,
                    'element': current_point,
                    'text': point_text.strip()
                })
            current_point = line
            point_text = ''
        else:
            point_text += ' ' + line

    if current_point != '':
        elements.append({
            'source': 'Положение ЦБ',
            'chapter': current_chapter,
            'element': current_point,
            'text': point_text.strip()
        })
    return elements

def parse_bank(text):
    elements = []
    pattern = r'([^\n]+\?)\s*\n([\s\S]*?)(?=[^\n]+\?|$)'
    matches = re.findall(pattern, text)
    
    for question, answer in matches:
        elements.append({
            'source': 'Документ банка',
            'chapter': question.strip(),
            'element': question.strip(),
            'text': answer.strip()
        })
    return elements

law_elements = parse_law(law_text)
cb_elements = parse_cb(cb_text)
bank_elements = parse_bank(bank_text)

all_elements = law_elements + cb_elements + bank_elements
df = pd.DataFrame(all_elements)
df['lemmas'] = df['text'].apply(lemmatize)

law_df = df[df['source'] == '115-ФЗ']
cb_df = df[df['source'] == 'Положение ЦБ']
bank_df = df[df['source'] == 'Документ банка']

topics = [
    'Блокировка счетов',
    'Подозрительные операции',
    'Проверка клиента',
    'Срок проверки',
    'Персональные данные',
    'Идентификация клиента',
    'Финансовый мониторинг'
]
vectorizer = TfidfVectorizer()

all_lemmas = df['lemmas'].tolist()

vectorizer.fit(all_lemmas)

def find_best_match(topic, doc_df, vectorizer):
    doc_indices = doc_df.index.tolist()
    doc_lemmas = doc_df['lemmas'].tolist()
    
    if not doc_lemmas:
        return None, 0.0
    
    topic_lemma = lemmatize(topic)
    
    topic_vector = vectorizer.transform([topic_lemma])
    doc_vectors = vectorizer.transform(doc_lemmas)
    
    similarities = cosine_similarity(topic_vector, doc_vectors)[0]
    
    best_index_in_doc = np.argmax(similarities)
    best_similarity = similarities[best_index_in_doc]
    best_row = doc_df.iloc[best_index_in_doc]
    
    return best_row, best_similarity


for topic in topics:
    law_match, law_sim = find_best_match(topic, law_df, vectorizer)
    cb_match, cb_sim = find_best_match(topic, cb_df, vectorizer)
    bank_match, bank_sim = find_best_match(topic, bank_df, vectorizer)
    # ... остальной код

def detect_presence(similarity):
    if similarity < 0.15: return 'Отсутствует'
    elif similarity < 0.35: return 'Частично'
    else: return 'Присутствует'

results = []

for topic in topics:
    law_match, law_sim = find_best_match(topic, law_df, vectorizer)
    cb_match, cb_sim = find_best_match(topic, cb_df, vectorizer)
    bank_match, bank_sim = find_best_match(topic, bank_df, vectorizer)
    
    if law_match is None or cb_match is None or bank_match is None:
        continue

    ab_similarity = cosine_similarity(
        vectorizer.transform([lemmatize(law_match['text'])]),
        vectorizer.transform([lemmatize(cb_match['text'])])
    )[0][0]
    
    bc_similarity = cosine_similarity(
        vectorizer.transform([lemmatize(cb_match['text'])]),
        vectorizer.transform([lemmatize(bank_match['text'])])
    )[0][0]

    law_present = law_sim >= 0.25
    cb_present = cb_sim >= 0.25
    bank_present = bank_sim >= 0.25

    if ab_similarity >= 0.55:
        ab_meaning = 'смысл сохранился'
        ab_gap = 'отсутствует'
    # Если норма есть в одном, но нет в другом - Логический разрыв
    elif law_present != cb_present:
        ab_meaning = 'смысл не сохранился'
        ab_gap = 'Л'
    # Если сходство среднее, но тексты разной длины - Структурный
    else:
        len_ratio = max(len(law_match['text']), len(cb_match['text'])) / max(min(len(law_match['text']), len(cb_match['text'])), 1)
        if len_ratio > 2.5:
            ab_gap = 'С'
            ab_meaning = 'Частично (разная детализация)'
        # Если длина похожая, но слова разные - Терминологический
        elif ab_similarity < 0.45:  # 🔼 Подняли порог с 0.4 до 0.45
            ab_gap = 'Т'
            ab_meaning = 'Частично (другая формулировка)'
        else:
            ab_gap = 'отсутствует'
            ab_meaning = 'смысл сохранился'

    if bc_similarity >= 0.55:
        bc_meaning = 'смысл сохранился'
        bc_gap = 'отсутствует'
    elif cb_present != bank_present:
        bc_meaning = 'смысл не сохранился'
        bc_gap = 'Л'
    else:
        len_ratio = max(len(cb_match['text']), len(bank_match['text'])) / max(min(len(cb_match['text']), len(bank_match['text'])), 1)
        if len_ratio > 2.5:
            bc_gap = 'С'
            bc_meaning = 'Частично (разная детализация)'
        elif bc_similarity < 0.45:
            bc_gap = 'Т'
            bc_meaning = 'Частично (другая формулировка)'
        else:
            bc_gap = 'отсутствует'
            bc_meaning = 'смысл сохранился'

    results.append({
        'Понятие/требование': topic,
        'Как задано в A': f"{law_match['element']}. {law_match['text'][:350]}{'...' if len(law_match['text'])>350 else ''}",
        'Как раскрыто в B': f"{cb_match['element']}. {cb_match['text'][:350]}{'...' if len(cb_match['text'])>350 else ''}",
        'Как раскрыто в C': f"{bank_match['element']}. {bank_match['text'][:350]}{'...' if len(bank_match['text'])>350 else ''}",
        'Смысл между A и B совпадает?': ab_meaning,
        'Смысл между B и C совпадает?': bc_meaning,
        'Тип разрыва между A и B': ab_gap,
        'Тип разрыва между B и C': bc_gap,
        'Среднее сходство': round(((ab_similarity + bc_similarity) / 2) * 100, 2)
    })

result_df = pd.DataFrame(results)

cols_order = [
    'Понятие/требование',
    'Как задано в A',
    'Как раскрыто в B',
    'Как раскрыто в C',
    'Смысл между A и B совпадает?',
    'Смысл между B и C совпадает?',
    'Тип разрыва между A и B',
    'Тип разрыва между B и C',
    'Среднее сходство'
]

result_df = result_df.reindex(columns=cols_order)
result_df.to_excel('final_comparison.xlsx', index=False)

print("Готово! Результат анализа final_comparison.xlsx")
print("Распределение разрывов:")
print(result_df['Тип разрыва между A и B'].value_counts())