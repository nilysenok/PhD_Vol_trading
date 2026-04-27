"""
edit_dissertation_v2.py
Applies all remaining edits to Диссертация_Лысенок_UPDATED.docx:
1. Full rewrite of CONCLUSION (paragraphs 812-824)
2. Add S5 limitation paragraph after strategy limitations
3. Mark hourly table captions (4.2, 4.4, 4.9, 4.16, 4.21) with strikethrough + yellow
"""

import copy
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import RGBColor, Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

INPUT_PATH = '/Users/nilysenok/Downloads/Диссертация_Лысенок_UPDATED.docx'
OUTPUT_PATH = '/Users/nilysenok/Downloads/Диссертация_Лысенок_FINAL.docx'

doc = Document(INPUT_PATH)
print(f"Loaded document with {len(doc.paragraphs)} paragraphs")

# ─────────────────────────────────────────────────────────────
# Helper: set text of a paragraph with yellow highlight
# ─────────────────────────────────────────────────────────────
def set_paragraph_yellow(paragraph, text, font_name='Times New Roman', font_size=Pt(14),
                         strikethrough=False, red_ranges=None):
    """Replace paragraph text with yellow-highlighted text.
    red_ranges: list of (start, end) char positions to color red.
    """
    # Clear existing runs
    for run in paragraph.runs:
        run._element.getparent().remove(run._element)
    # Also clear any remaining text nodes
    for child in list(paragraph._element):
        if child.tag.endswith('}r'):
            paragraph._element.remove(child)

    if red_ranges is None:
        red_ranges = []

    # Find [XXX] placeholders and add them to red_ranges
    import re
    for m in re.finditer(r'\[XXX\]', text):
        red_ranges.append((m.start(), m.end()))

    # Build segments: split text into normal and red parts
    if red_ranges:
        segments = []
        prev = 0
        for start, end in sorted(red_ranges):
            if prev < start:
                segments.append((text[prev:start], False))
            segments.append((text[start:end], True))
            prev = end
        if prev < len(text):
            segments.append((text[prev:], False))
    else:
        segments = [(text, False)]

    for seg_text, is_red in segments:
        run = paragraph.add_run(seg_text)
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        run.font.name = font_name
        run.font.size = font_size
        if is_red:
            run.font.color.rgb = RGBColor(255, 0, 0)
        if strikethrough:
            run.font.strike = True


def clear_paragraph_text(paragraph):
    """Remove all text content from a paragraph while keeping the XML element."""
    for run in paragraph.runs:
        run._element.getparent().remove(run._element)
    for child in list(paragraph._element):
        if child.tag.endswith('}r'):
            paragraph._element.remove(child)


def insert_paragraph_after(paragraph, text, style='Normal',
                           font_name='Times New Roman', font_size=Pt(14),
                           yellow=True, strikethrough=False, red_ranges=None):
    """Insert a new paragraph after the given paragraph."""
    new_p = OxmlElement('w:p')
    # Copy paragraph properties from the reference paragraph's style
    paragraph._element.addnext(new_p)
    new_para = type(paragraph)(new_p, paragraph._parent)

    # Set style
    if style:
        new_para.style = doc.styles[style]

    # Set text
    run = new_para.add_run(text)
    run.font.name = font_name
    run.font.size = font_size
    if yellow:
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    if strikethrough:
        run.font.strike = True

    return new_para


# ─────────────────────────────────────────────────────────────
# 1. CONCLUSION — Full rewrite (paragraphs 812-824)
# ─────────────────────────────────────────────────────────────
print("\n=== 1. Rewriting CONCLUSION ===")

# Find conclusion start
conclusion_start = None
for i, p in enumerate(doc.paragraphs):
    if p.text.startswith("В диссертации оценена экономическая"):
        conclusion_start = i
        break

if conclusion_start is None:
    print("ERROR: Could not find conclusion start!")
else:
    print(f"Found conclusion start at paragraph {conclusion_start}")

# Find conclusion end (before "Список литературы" heading)
conclusion_end = None
for i in range(conclusion_start + 1, len(doc.paragraphs)):
    p = doc.paragraphs[i]
    if p.style and p.style.name.startswith('Heading') and 'Список литературы' in p.text:
        conclusion_end = i  # exclusive
        break
    # Also check for empty paragraph before heading
    if p.text.strip() == '' and i + 1 < len(doc.paragraphs):
        next_p = doc.paragraphs[i + 1]
        if next_p.style and next_p.style.name.startswith('Heading') and 'Список литературы' in next_p.text:
            conclusion_end = i  # include the empty paragraph
            break

print(f"Conclusion spans paragraphs {conclusion_start} to {conclusion_end - 1}")

# New conclusion text (each element = one paragraph)
new_conclusion = [
    "В диссертации оценена экономическая эффективность интеграции прогнозов реализованной волатильности в активные торговые стратегии на российском рынке акций. Исследование охватывает полный цикл: прогнозирование волатильности, формирование торговых стратегий, портфельная агрегация на трёх уровнях и оценка статистической и экономической значимости результатов.",

    "Гипотеза H1 подтверждена. Гибридная модель прогнозирования реализованной волатильности (HAR-J + градиентный бустинг) продемонстрировала наименьшее значение QLIKE на всех трёх горизонтах прогнозирования. Превосходство статистически значимо по тесту Дибольда–Мариано на уровне 1% для всех попарных сравнений. Снижение QLIKE по сравнению с базовой моделью HAR-J составило 25% на горизонте H = 1.",

    "Гипотеза H2 подтверждена. Интеграция прогнозов волатильности статистически значимо увеличивает коэффициент Шарпа мультистратегийного портфеля. На уровне отдельных инструментов (per-ticker) все три подхода интеграции описательно улучшают trading Sharpe: подход C (режимная фильтрация) повышает средний показатель с +0,52 до +0,69, подход D (таргетирование волатильности) — до +0,67. На уровне мультистратегийного портфеля бутстрэп-тест (N = 10 000) подтвердил значимость улучшения: подход D превосходит подход A с ΔSharpe = +1,53 (p < 0,001), подход C — с ΔSharpe = +1,08 (p < 0,001), подход B — с ΔSharpe = +0,11 (p = 0,046).",

    "Гипотеза H3 подтверждена. Мультистратегийный портфель с прогнозами волатильности (МСП-A, конфигурация TOP4 × MaxSharpe) обеспечил коэффициент Шарпа избыточной доходности 2,46 на контрольной выборке 2022–2025. Пассивная стратегия buy-and-hold на индексе IMOEX за тот же период продемонстрировала trading Sharpe = −0,28 и максимальную просадку −68,8%. Бутстрэп-тест подтвердил статистическую значимость превосходства (ΔSharpe = +2,05, p = 0,0004).",

    "Гипотеза H4 подтверждена. Прирост эквивалента определённости (ΔCE) подхода D относительно подхода A составил +3,36% годовых при γ = 10 (p = 0,001), что эквивалентно максимальной комиссии за доступ к прогнозам в размере 336 базисных пунктов. При годовых издержках внедрения 3,3 млн руб. минимальный капитал для окупаемости составляет [XXX] млн руб., что доступно для институциональных участников рынка.",

    "Гипотеза H5 подтверждена. Таргетирование волатильности (подход D) обеспечивает наибольшую экономическую ценность среди трёх подходов интеграции прогнозов. Прирост CE подхода D статистически значимо превосходит подход C (режимная фильтрация): ΔCE = +1,68% годовых при γ = 10 (p = 0,034). На уровне отдельных инструментов 7 из 10 протестированных фаз подхода D продемонстрировали положительный прирост trading Sharpe относительно базового подхода.",

    "Результаты обладают практической значимостью для институциональных инвесторов, управляющих компаний и профессиональных участников российского рынка акций. Разработанная методология позволяет количественно оценить экономическую целесообразность внедрения системы прогнозирования волатильности с учётом масштаба капитала и структуры издержек конкретного участника.",

    "Основные ограничения исследования связаны с выборкой (17 наиболее ликвидных акций MOEX) и периодом контрольной выборки (2022–2025), характеризующимся аномально высокой ключевой ставкой ЦБ (средняя 14,7%, максимум 21%). Перспективными направлениями являются: расширение на внутридневные стратегии с часовым таймфреймом, применение к другим развивающимся рынкам, а также исследование рекуррентных нейронных сетей для прогнозирования волатильности.",
]

# Strategy: clear old conclusion paragraphs, reuse them for new text,
# delete extras if new conclusion is shorter.
old_count = conclusion_end - conclusion_start  # 826 - 812 = 14 (includes empty para)
new_count = len(new_conclusion)

print(f"Old conclusion: {old_count} paragraphs, New conclusion: {new_count} paragraphs")

# Fill existing paragraphs with new text
for j in range(min(old_count, new_count)):
    idx = conclusion_start + j
    p = doc.paragraphs[idx]
    set_paragraph_yellow(p, new_conclusion[j])
    print(f"  Wrote new conclusion paragraph {j+1} at index {idx}")

# If new conclusion is shorter, clear the extra paragraphs
if new_count < old_count:
    for j in range(new_count, old_count):
        idx = conclusion_start + j
        p = doc.paragraphs[idx]
        clear_paragraph_text(p)
        # Set text to empty with a single empty run to keep formatting
        print(f"  Cleared extra paragraph at index {idx}")

# If new conclusion is longer, insert additional paragraphs
if new_count > old_count:
    # Insert after the last reused paragraph
    last_reused = conclusion_start + old_count - 1
    for j in range(old_count, new_count):
        insert_paragraph_after(
            doc.paragraphs[last_reused + (j - old_count)],
            new_conclusion[j],
            yellow=True
        )
        print(f"  Inserted new conclusion paragraph {j+1}")

print(f"Conclusion rewrite complete.")

# ─────────────────────────────────────────────────────────────
# 2. Add S5 limitation paragraph
# ─────────────────────────────────────────────────────────────
print("\n=== 2. Adding S5 limitation paragraph ===")

# Insert after paragraph 787 (last limitation in "Ограничения, связанные с торговыми стратегиями")
# Para 787 = "Учёт реальных спредов..."
s5_text = "Стратегия S5 (Pivot Points) демонстрирует экспозицию менее 2%, что отражает редкость формирования торговых сигналов на основе опорных уровней в текущих рыночных условиях."

# Find paragraph 787 by content
target_para = None
for i, p in enumerate(doc.paragraphs):
    if p.text.startswith("Учёт реальных спредов"):
        target_para = i
        break

if target_para is not None:
    print(f"Inserting S5 paragraph after paragraph {target_para}")
    insert_paragraph_after(doc.paragraphs[target_para], s5_text, yellow=True)
    print("  S5 limitation paragraph inserted.")
else:
    print("WARNING: Could not find 'Учёт реальных спредов' paragraph")

# ─────────────────────────────────────────────────────────────
# 3. Mark hourly table captions with strikethrough + yellow
# ─────────────────────────────────────────────────────────────
print("\n=== 3. Marking hourly table captions with strikethrough ===")

# Tables 4.2, 4.4, 4.9, 4.16, 4.21
hourly_tables = ['Таблица 4.2.', 'Таблица 4.4.', 'Таблица 4.9.', 'Таблица 4.16.', 'Таблица 4.21.']

for i, p in enumerate(doc.paragraphs):
    txt = p.text
    for tbl in hourly_tables:
        if tbl in txt:
            # Apply strikethrough to all runs
            for run in p.runs:
                run.font.strike = True
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
            print(f"  Marked [{i}] {txt[:80]}... with strikethrough + yellow")
            break

# ─────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────
print(f"\nSaving to {OUTPUT_PATH}...")
doc.save(OUTPUT_PATH)
print("Done!")

# Verify
doc2 = Document(OUTPUT_PATH)
print(f"\nVerification: {len(doc2.paragraphs)} paragraphs in output")

# Check conclusion
for i, p in enumerate(doc2.paragraphs):
    if 'ЗАКЛЮЧЕНИЕ' in p.text and p.style and p.style.name.startswith('Heading'):
        print(f"\nЗАКЛЮЧЕНИЕ heading at paragraph {i}")
        for j in range(i+1, min(i+12, len(doc2.paragraphs))):
            txt = doc2.paragraphs[j].text
            if txt.strip():
                has_yellow = any(r.font.highlight_color == WD_COLOR_INDEX.YELLOW for r in doc2.paragraphs[j].runs if r.font.highlight_color)
                has_strike = any(r.font.strike for r in doc2.paragraphs[j].runs)
                print(f"  [{j}] yellow={has_yellow} strike={has_strike} | {txt[:100]}")
        break

# Check S5
print("\nS5 paragraph check:")
for i, p in enumerate(doc2.paragraphs):
    if 'Стратегия S5 (Pivot Points) демонстрирует экспозицию' in p.text:
        has_yellow = any(r.font.highlight_color == WD_COLOR_INDEX.YELLOW for r in p.runs)
        print(f"  [{i}] yellow={has_yellow} | {p.text[:100]}")

# Check strikethrough tables
print("\nStrikethrough table captions:")
for i, p in enumerate(doc2.paragraphs):
    for tbl in hourly_tables:
        if tbl in p.text:
            has_strike = any(r.font.strike for r in p.runs)
            has_yellow = any(r.font.highlight_color == WD_COLOR_INDEX.YELLOW for r in p.runs if r.font.highlight_color)
            print(f"  [{i}] strike={has_strike} yellow={has_yellow} | {p.text[:80]}")
