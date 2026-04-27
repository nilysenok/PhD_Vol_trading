#!/usr/bin/env python3
"""
Create PhD defence presentation (18 slides) from HSE template.
v8 — final corrected numbers, 18 slides, all figures.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
import os

# ── paths ──────────────────────────────────────────────────────────────
TEMPLATE = '/Users/nilysenok/Downloads/02_HSE_Presentation_Shablon.pptx'
OUTPUT   = '/Users/nilysenok/Downloads/Презентация_Лысенок_защита_v8.pptx'

FIGS = {
    'imoex_rtsi':      '/Users/nilysenok/Desktop/PhD/article_figures_output/fig_1_imoex_rtsi.png',
    'qlike_comparison': '/Users/nilysenok/Desktop/PhD_Vol_trading/moex_volatility/results/final/figures/qlike_comparison.png',
    'msp_cumulative':  '/Users/nilysenok/Desktop/PhD_Vol_trading/moex_volatility/results/final/strategies/figures/fig_4_8.png',
    'ce_gamma':        '/Users/nilysenok/Desktop/PhD_Vol_trading/moex_volatility/results/final/strategies/figures/fig_4_9.png',
}

# ── colours ────────────────────────────────────────────────────────────
BLUE      = RGBColor(0x00, 0x20, 0x60)   # #002060
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
BLACK     = RGBColor(0x00, 0x00, 0x00)
LIGHT_GREY = RGBColor(0xF2, 0xF2, 0xF2)
GREEN     = RGBColor(0x00, 0x80, 0x00)
RED       = RGBColor(0xCC, 0x00, 0x00)
GREY      = RGBColor(0x66, 0x66, 0x66)

# ── layout indices (from HSE template) ────────────────────────────────
L_COVER  = 0   # Обложка
L_TEXT1  = 1   # Текст_1  (left text + right picture area)
L_TEXT2  = 2   # Текст_2  (full-width text)
L_TABLE1 = 7   # Таблица_1  (full-width table)
L_TABLE2 = 8   # Таблица_2  (table + right text)

# ── header texts ──────────────────────────────────────────────────────
HDR_13 = 'Факультет экономических наук НИУ ВШЭ'
HDR_14 = 'Базовая кафедра инфраструктуры\nфинансовых рынков'
HDR_15 = 'Москва, 2026'


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def remove_all_slides(prs):
    while len(prs.slides) > 0:
        rId = None
        for k in prs.slides._sldIdLst[0].attrib.keys():
            if k.endswith('}id'):
                rId = prs.slides._sldIdLst[0].get(k)
                break
        if rId:
            prs.part.drop_rel(rId)
        prs.slides._sldIdLst.remove(prs.slides._sldIdLst[0])


def set_ph(slide, idx, text, font_size=None, bold=False, color=None, alignment=None):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            ph.text = ''
            tf = ph.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = text
            if alignment:
                p.alignment = alignment
            run = p.runs[0] if p.runs else p.add_run()
            if not p.runs:
                run.text = text
            if font_size:
                run.font.size = Pt(font_size)
            if bold:
                run.font.bold = True
            if color:
                run.font.color.rgb = color
            return ph
    return None


def set_ph_rich(slide, idx, lines, font_size=14, color=BLACK):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            tf = ph.text_frame
            tf.clear()
            tf.word_wrap = True
            for i, item in enumerate(lines):
                if isinstance(item, str):
                    txt, bld, clr = item, False, color
                else:
                    txt = item[0]
                    bld = item[1] if len(item) > 1 else False
                    clr = item[2] if len(item) > 2 else color
                if i == 0:
                    p = tf.paragraphs[0]
                else:
                    p = tf.add_paragraph()
                p.space_before = Pt(2)
                p.space_after = Pt(2)
                run = p.add_run()
                run.text = txt
                run.font.size = Pt(font_size)
                run.font.bold = bld
                run.font.color.rgb = clr
            return ph
    return None


def add_headers(slide):
    set_ph(slide, 13, HDR_13, font_size=9, color=BLUE)
    set_ph(slide, 14, HDR_14, font_size=9, color=BLUE)
    set_ph(slide, 15, HDR_15, font_size=9, color=BLUE)


def add_image(slide, key, left=5.3, top=1.1, width=6.2, height=5.0):
    path = FIGS[key]
    if os.path.exists(path):
        slide.shapes.add_picture(path, Inches(left), Inches(top), Inches(width), Inches(height))
    else:
        print(f"  WARNING: {path} not found")


def make_table(slide, rows, cols, data, left, top, width, height,
               header_color=BLUE, body_size=11, header_size=12,
               col_widths=None):
    tbl_shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(top),
                                       Inches(width), Inches(height))
    tbl = tbl_shape.table
    if col_widths:
        for ci, cw in enumerate(col_widths):
            tbl.columns[ci].width = Inches(cw)
    for ri in range(rows):
        for ci in range(cols):
            cell = tbl.cell(ri, ci)
            cell.text = ''
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = str(data[ri][ci])
            p.alignment = PP_ALIGN.CENTER
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            run = p.runs[0] if p.runs else p.add_run()
            if not p.runs:
                run.text = str(data[ri][ci])
            if ri == 0:
                run.font.size = Pt(header_size)
                run.font.bold = True
                run.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_color
            else:
                run.font.size = Pt(body_size)
                run.font.color.rgb = BLACK
                if ri % 2 == 0:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = LIGHT_GREY
                else:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = WHITE
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
    return tbl_shape


def add_slide(prs, layout_idx):
    layout = prs.slide_layouts[layout_idx]
    return prs.slides.add_slide(layout)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def build():
    prs = Presentation(TEMPLATE)
    remove_all_slides(prs)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 1: Титульник
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_COVER)
    set_ph(s, 0,
           'Эффективность применения прогнозов\nволатильности в активных торговых\nстратегиях институциональных инвесторов\nна российском рынке акций',
           font_size=20, bold=True, color=BLUE)
    set_ph(s, 10, 'Лысенок Никита Ильич', font_size=14, color=BLUE)
    set_ph(s, 11, 'Научный руководитель:', font_size=10, color=BLUE)
    set_ph(s, 12, 'д.э.н., профессор Берзон Н. И.', font_size=10, color=BLUE)
    set_ph(s, 13, 'Диссертация на соискание учёной степени кандидата экономических наук\nJEL: G11, G17, G23, G32', font_size=11, color=BLUE)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 2: Актуальность
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TEXT1)
    add_headers(s)
    set_ph(s, 0, 'Актуальность исследования', font_size=24, bold=True, color=BLUE)

    set_ph_rich(s, 12, [
        ('1. IMOEX −28%, RTSI −30% (2022–2026).', True, BLUE),
        ('Пассивные стратегии принесли инвесторам убытки.', False),
        ('', False),
        ('2. КС ЦБ РФ 14,7–21,0% (2022–2025).', True, BLUE),
        ('Аномально высокая альтернативная стоимость капитала.', False),
        ('Направленные стратегии целесообразны лишь', False),
        ('при достаточно сильных сигналах.', False),
        ('', False),
        ('3. Парадокс высокой КС.', True, BLUE),
        ('Стратегии занимают позицию 7–16% времени.', False),
        ('Прогнозы волатильности отфильтровывают ложные', False),
        ('сигналы → только экономически оправданные входы.', False),
    ], font_size=12)

    add_image(s, 'imoex_rtsi', left=5.5, top=1.3, width=6.0, height=4.8)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 3: Цель и задачи
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TABLE1)
    add_headers(s)
    set_ph(s, 17, 'Цель и задачи исследования', font_size=20, bold=True, color=BLUE)

    data3 = [
        ['№', 'Задача'],
        ['1', 'Систематизировать подходы к прогнозированию волатильности и выявить пробелы на российском рынке'],
        ['2', 'Разработать гибридную модель (HAR-J + бустинг) и сравнить с HAR-J на трёх горизонтах'],
        ['3', 'Сформировать 6 стратегий × 4 подхода интеграции прогнозов волатильности'],
        ['4', 'Провести скользящую валидацию 24 комбинаций per-ticker с учётом комиссий'],
        ['5', 'Сформировать портфели стратегий (17 инструментов), 4 схемы взвешивания'],
        ['6', 'Сформировать МСП из 6 стратегий, сравнить с IMOEX buy-and-hold'],
        ['7', 'Оценить экономическую ценность (ΔCE), пороговый капитал Kmin'],
    ]
    make_table(s, 8, 2, data3, left=0.5, top=1.7, width=11.3, height=3.5,
               col_widths=[0.5, 10.8], body_size=11, header_size=12)

    set_ph_rich(s, 18, [
        ('Объект: активные торговые стратегии на российском рынке акций', False, GREY),
        ('Предмет: методы прогнозирования волатильности и оценка эффективности', False, GREY),
    ], font_size=10)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 4: Гипотезы
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TABLE1)
    add_headers(s)
    set_ph(s, 17, 'Гипотезы исследования', font_size=20, bold=True, color=BLUE)

    data4 = [
        ['', 'Формулировка', 'Метрика / тест'],
        ['H1', 'Гибридная модель значимо превосходит HAR-J по QLIKE', 'DM-тест'],
        ['H2', 'Интеграция прогнозов значимо увеличивает Sharpe МСП', 'Bootstrap ΔSharpe'],
        ['H3', 'МСП значимо превосходит IMOEX buy-and-hold', 'Bootstrap ΔSharpe'],
        ['H4', 'Затраты на интеграцию окупаются менее чем за 2 года', 'Kmin = ΔCost / ΔCE'],
        ['H5', 'Подходы C и D обеспечивают значимую ΔCE', 'Bootstrap ΔCE'],
    ]
    make_table(s, 6, 3, data4, left=0.5, top=2.0, width=11.3, height=3.2,
               col_widths=[0.8, 7.5, 3.0], body_size=12, header_size=12)
    set_ph(s, 18, '', font_size=9)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 5: Данные
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TEXT2)
    add_headers(s)
    set_ph(s, 0, 'Данные и выборка', font_size=24, bold=True, color=BLUE)

    set_ph_rich(s, 12, [
        ('Воронка отбора:', True, BLUE),
        ('100+ тикеров MOEX → 20 (история с 2014, 10-мин) → 17 (покрытие >99%)', False),
        ('Исключены: LSRG (587 пропусков), PHOR (272), OGKB (135)', False, GREY),
        ('', False),
        ('Параметры данных:', True, BLUE),
        ('Период: июнь 2014 – февраль 2026 (>11 лет)', False),
        ('Частота: 10-минутные свечи OHLCV', False),
        ('Сессия: 10:00–18:40 МСК (основная)', False),
        ('Источник: MOEX ISS API', False),
        ('Контроль: фильтрация аномальных баров, >80% покрытия дня', False),
        ('Комиссии: 0,05–0,06% за сторону сделки', False),
    ], font_size=13)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 6: Признаковое пространство
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TEXT2)
    add_headers(s)
    set_ph(s, 0, 'Выборка и признаковое пространство', font_size=24, bold=True, color=BLUE)

    data6 = [
        ['Группа признаков', 'Кол-во', 'Признаков'],
        ['Секторальные индексы MOEX', '11 × 9', '99'],
        ['Глобальные активы (Brent, VIX, S&P500…)', '10 × 8', '80'],
        ['Макроэкономические (КС, ФРС, ИПЦ, M2…)', '8 × 4', '32'],
        ['Собственные признаки акции', '', '23'],
        ['Итого', '', '234'],
    ]
    make_table(s, 6, 3, data6, left=0.5, top=2.5, width=11.3, height=2.8,
               col_widths=[7.0, 2.0, 2.3], body_size=13, header_size=13)

    set_ph_rich(s, 12, [
        ('17 тикеров: SBER, GAZP, LKOH, GMKN, ROSN, NVTK, MGNT, MTSS,', False),
        ('ALRS, CHMF, NLMK, PLZL, TATN, VTBR, MOEX, IRAO, SNGS', False),
        ('Покрытие >80% индекса IMOEX', False, GREY),
    ], font_size=12)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 7: Семь моделей
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TEXT1)
    add_headers(s)
    set_ph(s, 0, 'Сравнительный анализ моделей', font_size=24, bold=True, color=BLUE)

    data7 = [
        ['Модель', 'H=1', 'H=5', 'H=22'],
        ['Гибридная', '0,276', '0,373', '0,441'],
        ['LightGBM', '0,290', '0,382', '0,455'],
        ['XGBoost', '0,277', '0,396', '0,470'],
        ['HAR-J', '0,305', '0,424', '0,467'],
        ['LSTM', '0,372', '0,435', '0,485'],
        ['GRU', '0,399', '0,411', '0,503'],
        ['GJR-GARCH', '0,501', '0,567', '0,601'],
    ]
    make_table(s, 8, 4, data7, left=0.4, top=2.0, width=4.8, height=3.5,
               col_widths=[1.6, 1.0, 1.0, 1.0], body_size=11, header_size=11)

    set_ph_rich(s, 12, [
        ('QLIKE (тест 2019, 17 тикеров)', True, BLUE),
        ('', False),
        ('GJR-GARCH: +29–64% хуже HAR-J', False, RED),
        ('→ отсеяна', False, RED),
        ('', False),
        ('LSTM/GRU: +34–44% хуже бустинга', False, RED),
        ('→ отсеяны', False, RED),
    ], font_size=12)

    add_image(s, 'qlike_comparison', left=5.5, top=1.3, width=6.0, height=4.8)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 8: Гибрид + H1
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TEXT2)
    add_headers(s)
    set_ph(s, 0, 'Гибридная модель — H1 подтверждена', font_size=24, bold=True, color=BLUE)

    data8t = [
        ['', 'H=1', 'H=5', 'H=22'],
        ['HAR-J', '0,524', '0,796', '0,961'],
        ['Гибридная', '0,376', '0,762', '0,888'],
        ['DM t-stat', '13,55', '3,41', '7,03'],
        ['DM p-value', '<0,001', '<0,001', '<0,001'],
    ]
    make_table(s, 5, 4, data8t, left=0.5, top=2.3, width=5.5, height=2.2,
               col_widths=[1.7, 1.2, 1.2, 1.2], body_size=12, header_size=12)

    set_ph_rich(s, 12, [
        ('Гибрид = HAR-J + лучший бустинг', True, BLUE),
        ('Адаптивный выбор на валидации из 7 стратегий', False),
        ('Доля HAR-J: 21% на H=1 → 48% на H=22', False),
        ('', False),
        ('QLIKE скользящей валидации 2017–2026', True, BLUE),
        ('Улучшение: −28% (H=1), −4% (H=5), −8% (H=22)', False),
        ('', False),
        ('✅ H1 подтверждена: p<0,001 на всех горизонтах', True, GREEN),
    ], font_size=12)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 9: Стратегии и подходы
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TABLE1)
    add_headers(s)
    set_ph(s, 17, 'Стратегии и подходы интеграции прогнозов', font_size=20, bold=True, color=BLUE)

    data9 = [
        ['Код', 'Тип', 'Сигнал'],
        ['S1 Mean Reversion', 'Контртренд', 'Z-score'],
        ['S2 Bollinger Bands', 'Контртренд', 'Полосы Боллинджера'],
        ['S3 Donchian', 'Тренд', 'Пробой канала'],
        ['S4 Supertrend', 'Тренд', 'Адаптивный индикатор ATR'],
        ['S5 Pivot Points', 'Диапазон', 'Уровни поддержки/сопротивления'],
        ['S6 VWAP', 'Диапазон', 'Средневзвешенная по объёму'],
        ['', '', ''],
        ['Подход', '', 'Механизм интеграции'],
        ['A (базовый)', '', 'Без прогнозов волатильности'],
        ['B (адаптивное управл.)', '', 'SL/TP = f(σ_pred). Как торговать'],
        ['C (режимная фильтрация)', '', 'Маска режима. Когда торговать'],
        ['D (таргетирование вол.)', '', 'Vol-gate. Когда торговать'],
    ]
    make_table(s, 13, 3, data9, left=0.5, top=1.8, width=11.3, height=4.3,
               col_widths=[3.0, 2.0, 6.3], body_size=11, header_size=12)
    set_ph(s, 18, '', font_size=9)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 10: Per-ticker + портфели
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TABLE1)
    add_headers(s)
    set_ph(s, 17, 'От отдельных инструментов к портфелям стратегий', font_size=18, bold=True, color=BLUE)

    data10 = [
        ['', 'A', 'B', 'C', 'D'],
        ['S1 MeanRev', '2,00', '1,93', '2,57', '2,47'],
        ['S2 Bollinger', '1,98', '1,88', '2,79', '2,32'],
        ['S3 Donchian', '0,62', '0,65', '1,36', '1,81'],
        ['S4 Supertrend', '0,92', '1,01', '1,62', '1,96'],
        ['S5 Pivot', '1,28', '1,23', '1,39', '1,39'],
        ['S6 VWAP', '2,06', '2,03', '2,31', '2,47'],
        ['Среднее', '1,48', '1,45', '2,01', '2,07'],
    ]
    make_table(s, 8, 5, data10, left=0.5, top=1.7, width=11.3, height=3.3,
               col_widths=[2.8, 2.0, 2.0, 2.0, 2.0], body_size=13, header_size=13)

    set_ph_rich(s, 18, [
        ('Шарп портфелей стратегий (best weight, inst 0,05%). ', False),
        ('Средняя корреляция стратегий: 0,09. Контртренд vs тренд: от −0,24 до −0,11.', False),
    ], font_size=11)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 11: МСП — H2, H3
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TABLE1)
    add_headers(s)
    set_ph(s, 17, 'Мультистратегийный портфель vs IMOEX', font_size=20, bold=True, color=BLUE)

    data11 = [
        ['Подход', 'Шарп', 'Дох. %', 'Вол. %', 'MDD %', 'Эксп. %'],
        ['A (без прогнозов)', '1,32', '19,6', '4,4', '−2,8', '15,7'],
        ['B (адапт. управл.)', '1,47', '19,5', '3,8', '−1,6', '13,0'],
        ['C (режимн. фильтр)', '2,51', '21,2', '3,0', '−0,9', '7,2'],
        ['D (таргетир. вол.)', '2,54', '20,9', '2,9', '−1,1', '12,2'],
        ['IMOEX B&H', '−0,69', '−7,1', '30,8', '−70,3', '100'],
    ]
    make_table(s, 6, 6, data11, left=0.5, top=1.7, width=11.3, height=2.5,
               col_widths=[3.0, 1.5, 1.5, 1.5, 1.5, 1.5], body_size=13, header_size=12)

    set_ph_rich(s, 18, [
        ('Bootstrap: B>A ΔSh=+0,15 p=0,009***; C>A +1,19 p<0,001***; D>A +1,22 p<0,001***', False),
        ('МСП vs IMOEX: ΔSh=+2,05 p=0,001***', False),
        ('✅ H2 подтверждена   ✅ H3 подтверждена', True, GREEN),
    ], font_size=11)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 12: Кумулятивная доходность
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TEXT2)
    add_headers(s)
    set_ph(s, 0, 'Кумулятивная доходность МСП и IMOEX', font_size=22, bold=True, color=BLUE)

    # Full-slide image
    add_image(s, 'msp_cumulative', left=0.8, top=1.5, width=10.5, height=5.0)

    set_ph_rich(s, 12, [
        ('', False),
    ], font_size=9)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 13: CE — H5
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TEXT1)
    add_headers(s)
    set_ph(s, 0, 'Экономическая ценность — H5', font_size=24, bold=True, color=BLUE)

    data13 = [
        ['γ', 'B ΔCE', 'p', 'C ΔCE', 'p', 'D ΔCE', 'p'],
        ['3', '−0,09%', '0,59', '+1,77%', '0,03**', '+1,46%', '0,07*'],
        ['5', '−0,05%', '0,55', '+1,86%', '0,02**', '+1,56%', '0,05**'],
        ['10', '+0,04%', '0,45', '+2,07%', '0,01***', '+1,79%', '0,02**'],
    ]
    make_table(s, 4, 7, data13, left=0.3, top=2.2, width=5.0, height=1.8,
               col_widths=[0.5, 0.8, 0.6, 0.8, 0.7, 0.8, 0.7], body_size=10, header_size=10)

    set_ph_rich(s, 12, [
        ('Когда > как:', True, BLUE),
        ('C: +1,77–2,07% годовых', False, GREEN),
        ('D: +1,46–1,79% годовых', False, GREEN),
        ('B: не значим (−0,09–0,04%)', False, RED),
        ('', False),
        ('✅ H5 подтверждена', True, GREEN),
    ], font_size=12)

    add_image(s, 'ce_gamma', left=5.5, top=1.3, width=6.0, height=4.8)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 14: Kmin — H4
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TEXT2)
    add_headers(s)
    set_ph(s, 0, 'Пороговый капитал внедрения — H4', font_size=24, bold=True, color=BLUE)

    set_ph_rich(s, 12, [
        ('Kmin = ΔCost / ΔCE = 3,3 млн / 2,07% = 160 млн руб.', True, BLUE),
        ('', False),
        ('Структура затрат:', True, BLUE),
        ('Система A (без прогнозов): 5,7 млн/год', False),
        ('Маржинальные затраты на интеграцию: +3,3 млн/год', False),
        ('  — количественный исследователь 0,5 ставки', False, GREY),
        ('  — доплата управляющему 0,2', False, GREY),
        ('  — IT 0,3, серверы, данные', False, GREY),
        ('', False),
        ('По данным Банка России: доступно <3% активных', False),
        ('инвесторов с портфелем >1 млн руб.', False),
        ('Соответствует нижней границе институциональных', False),
        ('портфелей.', False),
        ('', False),
        ('✅ H4 подтверждена (окупаемость <1 года при K > Kmin)', True, GREEN),
    ], font_size=13)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 15: Итоги гипотез
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TABLE1)
    add_headers(s)
    set_ph(s, 17, 'Результаты проверки гипотез', font_size=20, bold=True, color=BLUE)

    data15 = [
        ['', 'Формулировка', 'Результат', ''],
        ['H1', 'Гибрид > HAR-J по QLIKE', 'p<0,001', '✅'],
        ['H2', 'Прогнозы ↑ Sharpe МСП', 'p<0,001 (C,D)', '✅'],
        ['H3', 'МСП > IMOEX', 'p=0,001', '✅'],
        ['H4', 'Окупаемость <2 лет', '<1 года', '✅'],
        ['H5', 'C и D значимы, B нет', 'p<0,02 (C,D)', '✅'],
    ]
    make_table(s, 6, 4, data15, left=0.5, top=1.8, width=11.3, height=2.5,
               col_widths=[0.8, 5.5, 2.5, 0.8], body_size=13, header_size=13)

    set_ph_rich(s, 18, [
        ('Управление размером позиции (когда торговать) обеспечивает +1,8–2,1% годовых.', True, BLUE),
        ('Корректировка параметров сделки (как торговать) значимой ценности не создаёт.', False),
    ], font_size=12)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 16: Научная новизна
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TEXT2)
    add_headers(s)
    set_ph(s, 0, 'Научная новизна', font_size=24, bold=True, color=BLUE)

    set_ph_rich(s, 12, [
        ('1. Гибридная модель HAR-J + бустинг для MOEX (DM p<0,001)', False),
        ('2. Систематическое сравнение 3 механизмов интеграции: 24 комбинации, 30 фаз', False),
        ('3. Полный цикл «прогноз → стратегия → портфель → МСП → значимость» для MOEX', False),
        ('4. Количественная оценка CE: ΔCE = +2,07% (C) и +1,79% (D) при γ=10', False),
        ('5. МСП: ΔSharpe = +2,05 vs IMOEX (p=0,001), прибылен 2022–2025', False),
        ('6. Kmin = 160 млн руб.', False),
        ('7. C и D значимы, B — нет. Когда > как.', True, BLUE),
        ('   Подтверждено на MOEX впервые.', True, BLUE),
    ], font_size=14)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 17: Ограничения
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_TEXT2)
    add_headers(s)
    set_ph(s, 0, 'Ограничения и направления', font_size=24, bold=True, color=BLUE)

    set_ph_rich(s, 12, [
        ('Ограничения:', True, BLUE),
        ('• Один рынок (MOEX), 17 акций', False),
        ('• Спотовый рынок (не фьючерсы)', False),
        ('• Контрольная выборка 4 года (2022–2025)', False),
        ('• Только направленные стратегии', False),
        ('• Субъективность оценки издержек (опрос 10 специалистов)', False),
        ('', False),
        ('Направления будущих исследований:', True, BLUE),
        ('• Международные рынки (NYSE, LSE, BSE, B3)', False),
        ('• Фьючерсы MOEX, тиковые данные', False),
        ('• Рыночно-нейтральные и опционные стратегии', False),
        ('• Трансформерные архитектуры, альтернативные данные', False),
        ('• Непрерывное масштабирование позиции', False),
    ], font_size=13)

    # ──────────────────────────────────────────────────────────────────
    # SLIDE 18: Спасибо
    # ──────────────────────────────────────────────────────────────────
    s = add_slide(prs, L_COVER)
    set_ph(s, 0, 'Спасибо за внимание!', font_size=28, bold=True, color=BLUE)
    set_ph(s, 10, 'Лысенок Никита Ильич', font_size=14, color=BLUE)
    set_ph(s, 13, '3 публикации: список C + 2×D ВШЭ\n5 конференций: Ясинская (2022–2024), Ломоносов, Архитектура финансов', font_size=11, color=BLUE)
    set_ph(s, 11, '', font_size=1)
    set_ph(s, 12, '', font_size=1)

    # ── Save ──────────────────────────────────────────────────────────
    prs.save(OUTPUT)
    print(f'Saved: {OUTPUT}')
    print(f'Total slides: {len(prs.slides)}')


if __name__ == '__main__':
    build()
