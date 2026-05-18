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
    "Блокирование (замораживание) средств",
    "Операции, подлежащие обязательному контролю",
    "Идентификация клиента",
    "Подозрительные операции",
    "Право отказа в проведении операции",
    "Упрощенный контроль (до 15000 рублей)",
    "Внутренний контроль"
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
def analyze_theme_simple(theme: str, elements: List[Dict]) -> Dict:
    comparison = []
    
    for elem in elements:
        score = elem["similarity"]
        
        # наличие нормы
        if score >= THRESHOLD_EXPLICIT:
            presence = "явно"
        elif score >= 0.2:
            presence = "неявно"
        else:
            presence = "отсутствует"
        
        # степень совпадения
        if score >= THRESHOLD_FULL_MATCH:
            match = "полное"
        elif score >= 0.3:
            match = "частичное"
        else:
            match = "отсутствие"
        
        comparison.append({
            "document": elem["doc"],
            "presence": presence,
            "match_degree": match,
            "score": score,
            "text_quote": elem["text"][:120]
        })
    
    # расхождения между документами
    discrepancies = []
    docs_with_norm = [c for c in comparison if c["presence"] != "отсутствует"]
    
    # если норма есть не во всех документах — логическое расхождение
    if 0 < len(docs_with_norm) < len(comparison):
        discrepancies.append({
            "type": "логический",
            "description": f"Норма {theme} присутствует не во всех документах",
            "docs": [c["document"] for c in comparison if c["presence"] == "отсутствует"]
        })
    
    # если степени совпадения разные — терминологическое расхождение
    match_degrees = set(c["match_degree"] for c in docs_with_norm)
    if len(match_degrees) > 1 and len(docs_with_norm) > 1:
        discrepancies.append({
            "type": "терминологический",
            "description": "Разная степень детализации нормы в документах",
            "docs": [c["document"] for c in comparison]
        })
    
    if not docs_with_norm:
        conclusion = f"По теме {theme}: норма не обнаружена в представленных фрагментах."
    elif len(discrepancies) == 0:
        conclusion = f"По теме {theme}: нормы в документах согласованы."
    else:
        conclusion = f"По теме {theme}: выявлены расхождения между документами (требуется ручная проверка)."
    
    # eсть ли противоречия
    has_contradictions = any(
        c["document"].lower().startswith("т-банк") and c["presence"] == "отсутствует"
        for c in comparison
    ) or len(discrepancies) > 1
    
    return {
        "comparison": comparison,
        "discrepancies": discrepancies,
        "conclusion": conclusion,
        "has_contradictions": has_contradictions
    }

# 4. СБОРКА ТАБЛИЦЫ
def build_final_table(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        row = {"Тема": res["theme"]}
        analysis = res["analysis"]
        
        for comp in analysis.get("comparison", []):
            short_doc = comp["document"][:18] + ".." if len(comp["document"]) > 18 else comp["document"]
            row[f"{short_doc} | Наличие"] = comp.get("presence", "н/д")
            row[f"{short_doc} | Совпадение"] = comp.get("match_degree", "н/д")
            row[f"{short_doc} | Балл"] = f"{comp.get('score', 0):.3f}"
        
        disc_list = analysis.get("discrepancies", [])
        row["Типы расхождений"] = ", ".join(set(d["type"] for d in disc_list)) if disc_list else "Нет"
        
        row["Противоречия"] = "Да" if analysis.get("has_contradictions") else "Нет"
        row["Вывод"] = analysis.get("conclusion", "")[:150]
        
        rows.append(row)
    
    return pd.DataFrame(rows)

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
    print(f"   - Найдено расхождений: {sum(1 for r in final_results if r['analysis']['discrepancies'])}")

main()