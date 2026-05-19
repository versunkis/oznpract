# Анализ нормативных актов: 115-ФЗ, ЦБ, Т-Банк
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

DOCS_CONFIG = {
    "115-ФЗ": {"path": "docs/115fz.txt", "type": "articles"},
    "Положение ЦБ 860-П": {"path": "docs/regulation_860p.txt", "type": "chapters"},
    "Т-Банк разъяснения": {"path": "docs/tbank_faq.txt", "type": "headers"}
}

THEMES = [
    "Блокировка (замораживание) средств",
    "Операции, подлежащие обязательному контролю",
    "Упрощённая идентификация клиента",
    "Подозрительные операции",
    "Право отказа в проведении операции",
    "Порядок проверки при блокировке счёта",
    "Внутренний контроль",
    "Оценка уровня риска клиента"
]

THRESHOLD_EXPLICIT = 0.4    # выше — норма "явно" присутствует
THRESHOLD_FULL_MATCH = 0.6  # выше — "полное" совпадение

# ПАРСИНГ ДОКУМЕНТОВ
def parse_legal_act(text: str, doc_name: str, structure_type: str) -> List[Dict]:
    elements = []
    text = re.sub(r'\r\n?', '\n', text).strip()
    
    # разбить по абзацам
    chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 80]
    # Если мало — по строкам
    if len(chunks) < 3:
        chunks = [c.strip() for c in text.split('\n') if len(c.strip()) > 80]
    
    # мусор
    skip_words = ['главное меню', 'поиск', 'версия для печати', 'base.garant', 'tbank.ru', 'cookie']
    clean_chunks = [c for c in chunks if not any(skip in c.lower() for skip in skip_words)]
    
    for i, chunk in enumerate(clean_chunks, 1):
        # номер статьи/пункта
        id_match = re.match(r'(?:Статья|ст\.|Глава|гл\.|Пункт|п\.)\s*(\d+)', chunk, re.IGNORECASE)
        elem_id = id_match.group(1) if id_match else str(i)
        
        elements.append({
            "doc": doc_name,
            "type": structure_type,
            "id": f"{doc_name[:10].replace(' ', '_')}_{elem_id}",
            "text": chunk[:500]
        })
    return elements

def load_documents(config: Dict[str, Dict]) -> List[Dict]:
    all_elements = []
    for doc_name, settings in config.items():
        path = settings["path"]
        if not os.path.exists(path):
            print(f" Файл не найден: {path}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            elements = parse_legal_act(text, doc_name, settings["type"])
            all_elements.extend(elements)
            print(f" {doc_name}: найдено {len(elements)} блоков")
        except Exception as e:
            print(f"Ошибка {doc_name}: {e}")
    return all_elements

# ПОИСК ПО КЛЮЧЕВЫМ СЛОВАМ
def search_by_keywords(elements: List[Dict], themes: List[str]) -> Dict[str, List[Dict]]:
    print("\n Поиск по ключевым словам")
    
    if not elements:
        return {theme: [] for theme in themes}
    
    element_texts = [elem["text"] for elem in elements]
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # слова и пары слов
        max_features=1000,    # топ-1000 самых важных признаков
        min_df=1,
        stop_words=None
    )
    
    try:
        elem_matrix = vectorizer.fit_transform(element_texts)
        theme_matrix = vectorizer.transform(themes)
    except ValueError:
        print("   Тексты слишком короткие, возвращаем первые блоки")
        return {theme: elements[:3] for theme in themes}
    
    results = {}
    for idx, theme in enumerate(themes):
        # сходство
        scores = cosine_similarity(theme_matrix[idx:idx+1], elem_matrix)[0]
        
        # топ-3 наиболее похожих
        top_indices = scores.argsort()[-3:][::-1]
        
        similar = []
        for elem_idx in top_indices:
            if scores[elem_idx] > 0.01:
                elem = elements[elem_idx]
                similar.append({
                    **elem, 
                    "similarity": round(float(scores[elem_idx]), 4)
                })
        
        results[theme] = similar
        print(f"  - {theme}: найдено {len(similar)} релевантных блоков")
    
    return results

# АНАЛИЗ
def analyze_theme_simple(theme: str, matches: List[Dict]) -> Dict:
    # Группируем лучшие совпадения по каждому документу
    docs_data = {}
    for doc_name in DOCS_CONFIG.keys():
        best = None
        for m in matches:
            if m["doc"] == doc_name:
                if best is None or m["similarity"] > best["similarity"]:
                    best = m
        docs_data[doc_name] = best

    def get_text(doc_key):
        m = docs_data.get(doc_key)
        if m and m["similarity"] > 0.15:
            txt = m["text"]
            clean = re.sub(r'\s+', ' ', txt).strip()
            return clean[:250] + ("..." if len(clean) > 250 else "")
        return "Информация в документе не найдена или не относится к теме."

    t_115 = get_text("115-ФЗ")
    t_cb = get_text("Положение ЦБ 860-П")
    t_tb = get_text("Т-Банк разъяснения")
    
    # Собираем все найденные тексты для анализа
    texts = [t for t in [t_115, t_cb, t_tb] if "не найдена" not in t]
    scores = [docs_data[d]["similarity"] if docs_data[d] else 0.0 for d in DOCS_CONFIG.keys()]
    
    # есть ли вообще данные
    if len(texts) < 2 or max(scores) < 0.2:
        disc_type = "Тема не представлена в документах"
        match_status = "Нет"
        
    else:
        # 2. Анализируем различия между текстами
        all_present = all(s >= 0.2 for s in scores)
        
        # Вычисляем длину текстов (для определения структурного разрыва)
        lengths = [len(t) for t in texts]
        max_len = max(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        length_ratio = max_len / min_len if min_len > 50 else 999
        
        # Проверяем, есть ли общие ключевые слова (для терминологического разрыва)
        if len(texts) >= 2:
            words1 = set(texts[0].lower().split())
            words2 = set(texts[-1].lower().split())
            common_words = words1.intersection(words2)
            overlap_ratio = len(common_words) / max(len(words1), len(words2)) if words1 or words2 else 0
        else:
            overlap_ratio = 0
        
        if all_present and overlap_ratio > 0.4 and length_ratio < 2:
            # Много общих слов, похожая длина → ТЕРМИНОЛОГИЧЕСКИЙ
            disc_type = "Терминологический. Различие в формулировках при сохранении одного и того же смысла"
            match_status = "Да"
            
        elif not all_present or overlap_ratio < 0.2:
            # Норма есть не везде или мало общих слов → ЛОГИЧЕСКИЙ
            missing = [k for k, v in docs_data.items() if not v or v["similarity"] < 0.2]
            present = [k for k, v in docs_data.items() if v and v["similarity"] >= 0.2]
            if missing:
                disc_type = f"Логический. Различие в содержании нормы: требование присутствует в {', '.join(present)}, но отсутствует в {', '.join(missing)}"
            else:
                disc_type = "Логический. Различие в содержании нормы (основания, условия или последствия отличаются)"
            match_status = "Частично"
            
        elif length_ratio > 2.5:
            # Сильная разница в длине → СТРУКТУРНЫЙ
            disc_type = "Структурный. Различие в способе изложения: один документ содержит общую норму, другой — детализированную процедуру"
            match_status = "Частично"
            
        else:
            # По умолчанию — терминологический
            disc_type = "Терминологический. Различие в формулировках при сохранении одного и того же смысла"
            match_status = "Да"

    return {
        "Понятие/требование": theme,
        "Документы сравнения": "115-ФЗ, №860-П, материалы Т-Банка",
        "Как задано в 115-ФЗ": t_115,
        "Как раскрыто в Положении ЦБ": t_cb,
        "Как раскрыто в материалах Т-Банка": t_tb,
        "Смысл совпадает?": match_status,
        "Тип разрыва": disc_type
    }

# 4. СБОРКА ТАБЛИЦЫ
import pandas as pd
from typing import List, Dict

def build_final_table(rows_data: List[Dict]) -> pd.DataFrame:
    flat_rows = [row["analysis"] for row in rows_data]
    
    df = pd.DataFrame(flat_rows)
    
    desired_cols = [
        "Понятие/требование",
        "Документы сравнения",
        "Как задано в 115-ФЗ",
        "Как раскрыто в Положении ЦБ",
        "Как раскрыто в материалах Т-Банка",
        "Смысл совпадает?",
        "Тип разрыва"
    ]
    
    df = df.reindex(columns=desired_cols).fillna("—")
    return df

# ГЛАВНЫЙ ЗАПУСК
def main():
    print(" Анализ нормативных актов: 115-ФЗ, ЦБ, Т-Банк\n")
    print(" Шаг 1: Чтение и разбиение документов")
    elements = load_documents(DOCS_CONFIG)
    if not elements:
        print(" Нет элементов для анализа. Проверь папку docs/ и кодировку файлов (UTF-8).")
        return
    print(f" Всего блоков для анализа: {len(elements)}\n")
    
    print(" Шаг 2: Поиск фрагментов")
    search_results = search_by_keywords(elements, THEMES)

    print("\n  Шаг 3: Анализ")
    final_results = []
    for theme in THEMES:
        print(f"    {theme}")
        similar_elems = search_results[theme]
        if not similar_elems:
            print(f"    Не найдено блоков")
            # пустой результат для таблицы
            analysis = analyze_theme_simple(theme, [])
            final_results.append({"theme": theme, "analysis": analysis})
            continue
        analysis = analyze_theme_simple(theme, similar_elems)
        final_results.append({"theme": theme, "analysis": analysis})
        print(f"    Готово")
    
    print("\n Шаг 4: Формирование отчёта")
    df = build_final_table(final_results)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 100)
    
    df.to_excel("analysis_results.xlsx", index=False)
    df.to_csv("analysis_results.csv", index=False, sep=';', encoding='utf-8-sig')
    
    print(" Результаты сохранены:")
    print("    - analysis_results.xlsx")
    print("    - analysis_results.csv")
    
    print("\n Статистика:")
    print(f"   - Проанализировано тем: {len(THEMES)}")
    print(f"   - Всего блоков в документах: {len(elements)}")
    gaps_count = sum(1 for r in final_results 
                    if any(kw in r['analysis'].get('Тип разрыва', '') 
                           for kw in ['Терминологический', 'Логический', 'Структурный', 'Логический разрыв']))
    print(f"   • Тем с выявленными разрывами: {gaps_count}")

main()