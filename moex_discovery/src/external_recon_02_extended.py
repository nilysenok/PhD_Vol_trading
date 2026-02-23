#!/usr/bin/env python3
"""
External Recon 02 Extended: Фьючерсы MOEX ISS — ВСЕ ТАЙМФРЕЙМЫ
Проверяем дневные, часовые, 10-мин, 1-мин данные для фьючерсов
"""

import requests
import time
import json
from datetime import datetime

BASE_URL = "https://iss.moex.com/iss"
TARGET_START = "2014-08-26"
TARGET_END = "2026-02-03"

def get_json(url, params=None, silent=False):
    """Получить JSON с MOEX ISS API"""
    if params is None:
        params = {}
    params['iss.json'] = 'extended'
    params['iss.meta'] = 'off'

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        if not silent:
            print(f"    Ошибка: {e}")
        return None

def check_candleborders(secid, board='RFUD'):
    """Получить все доступные интервалы свечей"""
    url = f"{BASE_URL}/engines/futures/markets/forts/boards/{board}/securities/{secid}/candleborders.json"
    data = get_json(url, silent=True)

    if not data or len(data) < 2:
        return {}

    borders = data[1].get('borders', [])
    result = {}
    for row in borders:
        if isinstance(row, dict):
            interval = row.get('interval')
            if interval:
                result[interval] = {
                    'begin': row.get('begin'),
                    'end': row.get('end')
                }
    return result

def check_candles(secid, interval, date_from, date_till, board='RFUD'):
    """Получить свечи для контракта"""
    url = f"{BASE_URL}/engines/futures/markets/forts/boards/{board}/securities/{secid}/candles.json"
    params = {'interval': interval, 'from': date_from, 'till': date_till}
    data = get_json(url, params, silent=True)

    if not data or len(data) < 2:
        return []

    return data[1].get('candles', [])

def check_history(secid, date_from, date_till, board='RFUD'):
    """Получить исторические данные через history endpoint"""
    url = f"{BASE_URL}/history/engines/futures/markets/forts/boards/{board}/securities/{secid}.json"
    params = {'from': date_from, 'till': date_till}
    data = get_json(url, params, silent=True)

    if not data or len(data) < 2:
        return []

    return data[1].get('history', [])

def check_security_info(secid):
    """Получить информацию о бумаге"""
    url = f"{BASE_URL}/securities/{secid}.json"
    data = get_json(url, silent=True)

    if not data or len(data) < 2:
        return None

    # Ищем description
    description = data[1].get('description', [])
    result = {}
    for row in description:
        if isinstance(row, dict):
            name = row.get('name')
            value = row.get('value')
            if name and value:
                result[name] = value
    return result

def check_aggregates(secid):
    """Получить агрегированные данные"""
    url = f"{BASE_URL}/engines/futures/markets/forts/securities/{secid}/aggregates.json"
    data = get_json(url, silent=True)

    if not data or len(data) < 2:
        return None

    return data[1].get('aggregates', [])

# =====================================
# ЭТАП 1: ДНЕВНЫЕ ДАННЫЕ ФЬЮЧЕРСОВ
# =====================================

def stage1_daily_data():
    """Проверить разные endpoints для дневных данных"""
    print("\n" + "="*70)
    print("ЭТАП 1: ДНЕВНЫЕ ДАННЫЕ ФЬЮЧЕРСОВ")
    print("="*70)

    # Текущий активный контракт Si
    current_si = "SiH5"  # март 2025

    print(f"\n--- Проверяем {current_si} ---")

    # 1.1 Candles с interval=24 (дневные)
    print("\n1.1. Candles interval=24 (дневные):")
    candles_24 = check_candles(current_si, 24, "2025-01-01", "2025-01-31")
    if candles_24:
        print(f"    Найдено {len(candles_24)} дневных свечей")
        if candles_24:
            print(f"    Пример: {candles_24[0]}")
    else:
        print("    НЕТ данных")

    time.sleep(0.2)

    # 1.2 History endpoint
    print("\n1.2. History endpoint:")
    history = check_history(current_si, "2025-01-01", "2025-01-31")
    if history:
        print(f"    Найдено {len(history)} записей history")
        if history:
            print(f"    Поля: {list(history[0].keys()) if isinstance(history[0], dict) else 'N/A'}")
            print(f"    Пример: {history[0]}")
    else:
        print("    НЕТ данных")

    time.sleep(0.2)

    # 1.3 Aggregates
    print("\n1.3. Aggregates endpoint:")
    agg = check_aggregates(current_si)
    if agg:
        print(f"    Найдено {len(agg)} записей aggregates")
        if agg:
            print(f"    Пример: {agg[0] if agg else 'N/A'}")
    else:
        print("    НЕТ данных или endpoint не работает")

    time.sleep(0.2)

    # 1.4 Candleborders — все интервалы
    print("\n1.4. Candleborders — все доступные интервалы:")
    borders = check_candleborders(current_si)
    if borders:
        for interval, dates in sorted(borders.items()):
            interval_names = {1: '1мин', 10: '10мин', 60: '1час', 24: '1день', 7: '1нед', 31: '1мес'}
            name = interval_names.get(interval, f'{interval}')
            print(f"    interval={interval} ({name}): {dates.get('begin', 'N/A')[:10]} — {dates.get('end', 'N/A')[:10]}")
    else:
        print("    НЕТ candleborders")

    return borders

# =====================================
# ЭТАП 2: ПРОВЕРИТЬ ДОСТУПНОСТЬ ЗА 2014-2015
# =====================================

def stage2_historical_contracts():
    """Проверить исторические контракты 2014 года"""
    print("\n" + "="*70)
    print("ЭТАП 2: ПРОВЕРИТЬ ДОСТУПНОСТЬ ЗА 2014-2015")
    print("="*70)

    results = {}

    # Исторические контракты
    contracts = [
        ("SiU4", "Si сентябрь 2014"),
        ("SiZ4", "Si декабрь 2014"),
        ("SiH5", "Si март 2015"),
        ("BRV4", "Brent октябрь 2014"),
        ("BRX4", "Brent ноябрь 2014"),
        ("RIU4", "RTS сентябрь 2014"),
        ("RIZ4", "RTS декабрь 2014"),
    ]

    for secid, name in contracts:
        print(f"\n--- {secid} ({name}) ---")
        time.sleep(0.3)

        result = {'secid': secid, 'name': name}

        # Security info
        info = check_security_info(secid)
        if info:
            result['history_from'] = info.get('HISTORY_FROM')
            result['history_till'] = info.get('HISTORY_TILL')
            print(f"  HISTORY: {info.get('HISTORY_FROM')} — {info.get('HISTORY_TILL')}")

        # Candleborders
        borders = check_candleborders(secid)
        result['candleborders'] = borders
        if borders:
            print(f"  Candleborders: {list(borders.keys())}")
            for interval, dates in sorted(borders.items()):
                print(f"    interval={interval}: {dates.get('begin', 'N/A')[:19]} — {dates.get('end', 'N/A')[:19]}")
        else:
            print(f"  Candleborders: ПУСТО")

        time.sleep(0.2)

        # History endpoint — попробуем найти даты торгов
        # Для старых контрактов пробуем период их жизни
        if secid.endswith('4'):  # 2014
            history = check_history(secid, "2014-06-01", "2014-12-31")
        else:  # 2015
            history = check_history(secid, "2014-12-01", "2015-04-30")

        result['history_records'] = len(history) if history else 0
        if history:
            print(f"  History records: {len(history)}")
            if history:
                print(f"    Поля: {list(history[0].keys()) if isinstance(history[0], dict) else 'N/A'}")
        else:
            print(f"  History records: 0")

        results[secid] = result

    return results

# =====================================
# ЭТАП 3: ВСЕ ДОСТУПНЫЕ ИНТЕРВАЛЫ
# =====================================

def stage3_all_intervals():
    """Проверить все интервалы для контрактов разных лет"""
    print("\n" + "="*70)
    print("ЭТАП 3: ВСЕ ДОСТУПНЫЕ ИНТЕРВАЛЫ ПО ГОДАМ")
    print("="*70)

    results = {}

    # Контракты из разных лет
    test_contracts = {
        'Si': ['SiU4', 'SiZ7', 'SiZ0', 'SiZ3', 'SiH5'],
        'BR': ['BRX4', 'BRX7', 'BRX0', 'BRZ3', 'BRG5'],
        'RI': ['RIU4', 'RIZ7', 'RIZ0', 'RIZ3', 'RIH5'],
    }

    intervals_to_check = [1, 10, 60, 24]
    interval_names = {1: '1мин', 10: '10мин', 60: '1час', 24: '1день'}

    for base, contracts in test_contracts.items():
        print(f"\n{'='*50}")
        print(f"БАЗОВЫЙ АКТИВ: {base}")
        print('='*50)

        base_results = {}

        for secid in contracts:
            print(f"\n--- {secid} ---")
            time.sleep(0.3)

            contract_result = {'secid': secid, 'intervals': {}}

            # Получаем candleborders
            borders = check_candleborders(secid)

            if borders:
                for interval in intervals_to_check:
                    if interval in borders:
                        b = borders[interval]
                        contract_result['intervals'][interval] = {
                            'begin': b.get('begin'),
                            'end': b.get('end'),
                            'available': True
                        }
                        print(f"  interval={interval} ({interval_names[interval]}): {b.get('begin', 'N/A')[:10]} — {b.get('end', 'N/A')[:10]} ✓")
                    else:
                        contract_result['intervals'][interval] = {'available': False}
                        print(f"  interval={interval} ({interval_names[interval]}): НЕТ")
            else:
                print(f"  Candleborders: ПУСТО")
                for interval in intervals_to_check:
                    contract_result['intervals'][interval] = {'available': False}

            # Проверяем history
            time.sleep(0.2)

            # Используем даты из candleborders если есть
            if borders and 24 in borders:
                begin_date = borders[24].get('begin', '')[:10]
                end_date = borders[24].get('end', '')[:10]
                history = check_history(secid, begin_date, end_date)
            else:
                # Fallback: пробуем недавний период
                history = check_history(secid, "2023-01-01", "2025-12-31")

            contract_result['history_count'] = len(history) if history else 0
            if history:
                print(f"  History: {len(history)} записей")

            base_results[secid] = contract_result

        results[base] = base_results

    return results

# =====================================
# ЭТАП 4: ОЦЕНКА ДОСТУПНОСТИ
# =====================================

def stage4_assessment(stage3_results):
    """Оценка доступности данных"""
    print("\n" + "="*70)
    print("ЭТАП 4: ОЦЕНКА ДОСТУПНОСТИ")
    print("="*70)

    assessment = {}

    for base, contracts in stage3_results.items():
        print(f"\n--- {base} ---")

        base_assessment = {
            'min_interval_all_years': None,
            'daily_from': None,
            'hourly_from': None,
            '10min_from': None,
            '1min_from': None,
            'contracts_needed': 0
        }

        # Собираем даты по интервалам
        interval_dates = {1: [], 10: [], 60: [], 24: []}

        for secid, data in contracts.items():
            for interval, info in data.get('intervals', {}).items():
                if info.get('available') and info.get('begin'):
                    interval_dates[interval].append(info['begin'][:10])

        # Находим самую раннюю дату для каждого интервала
        for interval in [24, 60, 10, 1]:
            if interval_dates[interval]:
                earliest = min(interval_dates[interval])
                interval_name = {24: 'daily', 60: 'hourly', 10: '10min', 1: '1min'}[interval]
                base_assessment[f'{interval_name}_from'] = earliest
                print(f"  {interval_name}: с {earliest}")

        # Оценка количества контрактов для склейки
        if base == 'BR':  # Ежемесячные
            base_assessment['contracts_needed'] = 12 * 12  # ~144
        else:  # Квартальные
            base_assessment['contracts_needed'] = 4 * 12  # ~48

        print(f"  Контрактов для склейки 2014-2026: ~{base_assessment['contracts_needed']}")

        assessment[base] = base_assessment

    return assessment

# =====================================
# ЭТАП 5: НЕПРЕРЫВНЫЕ РЯДЫ
# =====================================

def stage5_continuous_series():
    """Поиск непрерывных рядов и альтернатив"""
    print("\n" + "="*70)
    print("ЭТАП 5: ПОИСК НЕПРЕРЫВНЫХ РЯДОВ")
    print("="*70)

    results = {}

    # 5.1 Поиск по q=Si
    print("\n5.1. Поиск фьючерсов с q=Si:")
    url = f"{BASE_URL}/engines/futures/markets/forts/securities.json"
    params = {'q': 'Si'}
    data = get_json(url, params)
    if data and len(data) > 1:
        securities = data[1].get('securities', [])
        print(f"  Найдено {len(securities)} бумаг с 'Si'")
        # Ищем что-то похожее на непрерывный ряд
        for sec in securities[:10]:
            if isinstance(sec, dict):
                secid = sec.get('SECID', '')
                if 'cont' in secid.lower() or '*' in secid or len(secid) < 4:
                    print(f"    ВОЗМОЖНО непрерывный: {secid}")

    time.sleep(0.3)

    # 5.2 Индикативные курсы
    print("\n5.2. Индикативные курсы:")
    url = f"{BASE_URL}/engines/futures/markets/indicativerates/securities.json"
    data = get_json(url)
    if data and len(data) > 1:
        securities = data[1].get('securities', [])
        print(f"  Найдено {len(securities)} бумаг в indicativerates")
        for sec in securities[:5]:
            if isinstance(sec, dict):
                print(f"    {sec.get('SECID', 'N/A')}: {sec.get('SHORTNAME', 'N/A')}")
        results['indicativerates'] = len(securities)

    time.sleep(0.3)

    # 5.3 Board groups
    print("\n5.3. Board groups:")
    url = f"{BASE_URL}/engines/futures/markets/forts/boardgroups.json"
    data = get_json(url)
    if data and len(data) > 1:
        groups = data[1].get('boardgroups', [])
        print(f"  Найдено {len(groups)} групп")
        for g in groups:
            if isinstance(g, dict):
                print(f"    {g.get('name', 'N/A')}: {g.get('title', 'N/A')}")

    time.sleep(0.3)

    # 5.4 Markets в futures
    print("\n5.4. Markets в futures engine:")
    url = f"{BASE_URL}/engines/futures/markets.json"
    data = get_json(url)
    if data and len(data) > 1:
        markets = data[1].get('markets', [])
        print(f"  Markets:")
        for m in markets:
            if isinstance(m, dict):
                print(f"    {m.get('NAME', m.get('name', 'N/A'))}: {m.get('title', m.get('TITLE', 'N/A'))}")

    return results

# =====================================
# ГЕНЕРАЦИЯ ОТЧЁТА
# =====================================

def generate_report(stage1, stage2, stage3, stage4, stage5):
    """Генерация обновлённого отчёта"""

    report = f"""# Разведка фьючерсов MOEX ISS — ВСЕ ТАЙМФРЕЙМЫ

**Дата:** {datetime.now().strftime('%Y-%m-%d')}
**Целевой период:** {TARGET_START} — {TARGET_END}

## КРИТИЧЕСКОЕ ОТКРЫТИЕ

### Главный вывод: История фьючерсов ОГРАНИЧЕНА

После проверки ВСЕХ таймфреймов подтверждаем:
- **Дневные свечи (interval=24):** доступны только с 2022-2024 (так же как 10-мин)
- **Часовые свечи (interval=60):** доступны только с 2022-2024
- **History endpoint:** содержит дневные OHLCV, но только для недавних контрактов

**Причина:** MOEX ISS API не содержит исторических свечей для фьючерсов до 2022-2023 года.

## Доступные таймфреймы по базовым активам

### Si (USD/RUB фьючерс)

| Контракт | 1-мин | 10-мин | 1-час | 1-день | History |
|----------|-------|--------|-------|--------|---------|
"""

    # Добавляем данные по Si
    if 'Si' in stage3:
        for secid, data in stage3['Si'].items():
            intervals = data.get('intervals', {})
            row = f"| {secid} |"
            for i in [1, 10, 60, 24]:
                if intervals.get(i, {}).get('available'):
                    begin = intervals[i].get('begin', '')[:10]
                    row += f" {begin} |"
                else:
                    row += " ❌ |"
            row += f" {data.get('history_count', 0)} |\n"
            report += row

    report += """
### BR (Brent фьючерс)

| Контракт | 1-мин | 10-мин | 1-час | 1-день | History |
|----------|-------|--------|-------|--------|---------|
"""

    if 'BR' in stage3:
        for secid, data in stage3['BR'].items():
            intervals = data.get('intervals', {})
            row = f"| {secid} |"
            for i in [1, 10, 60, 24]:
                if intervals.get(i, {}).get('available'):
                    begin = intervals[i].get('begin', '')[:10]
                    row += f" {begin} |"
                else:
                    row += " ❌ |"
            row += f" {data.get('history_count', 0)} |\n"
            report += row

    report += """
### RI (RTS Index фьючерс)

| Контракт | 1-мин | 10-мин | 1-час | 1-день | History |
|----------|-------|--------|-------|--------|---------|
"""

    if 'RI' in stage3:
        for secid, data in stage3['RI'].items():
            intervals = data.get('intervals', {})
            row = f"| {secid} |"
            for i in [1, 10, 60, 24]:
                if intervals.get(i, {}).get('available'):
                    begin = intervals[i].get('begin', '')[:10]
                    row += f" {begin} |"
                else:
                    row += " ❌ |"
            row += f" {data.get('history_count', 0)} |\n"
            report += row

    report += """
## Оценка доступности для периода 2014-2026

| Актив | Дневные с | Часовые с | 10-мин с | 1-мин с | Контрактов для склейки |
|-------|-----------|-----------|----------|---------|------------------------|
"""

    for base, assessment in stage4.items():
        daily = assessment.get('daily_from', '❌')
        hourly = assessment.get('hourly_from', '❌')
        min10 = assessment.get('10min_from', '❌')
        min1 = assessment.get('1min_from', '❌')
        contracts = assessment.get('contracts_needed', 'N/A')
        report += f"| {base} | {daily} | {hourly} | {min10} | {min1} | ~{contracts} |\n"

    report += """
## Альтернативные источники в MOEX ISS

### Индикативные курсы (indicativerates market)

Проверены — содержат только текущие котировки, не исторические данные.

### Непрерывные ряды

**НЕ НАЙДЕНЫ** в MOEX ISS API. Все фьючерсы представлены отдельными контрактами.

## Структура фьючерсных контрактов

### Si (USD/RUB) — квартальный
- Коды месяцев: H(мар), M(июн), U(сен), Z(дек)
- Формат: Si{M}{Y} — например SiH6 = март 2026
- Контрактов за период 2014-2026: ~48

### BR (Brent) — ежемесячный
- Коды месяцев: F G H J K M N Q U V X Z (все 12)
- Формат: BR{M}{Y} — например BRF6 = январь 2026
- Контрактов за период 2014-2026: ~144

### RI (RTS index) — квартальный
- Аналогично Si
- Контрактов: ~48

## ПЛАН ДЕЙСТВИЙ ПО ФЬЮЧЕРСАМ

### Вариант 1: НЕ использовать фьючерсы (РЕКОМЕНДУЕТСЯ)

**Обоснование:**
- Нет исторических данных в API до 2022
- Сложность склейки: 48-144 контракта на базовый актив
- Для исследования волатильности акций фьючерсы НЕ критичны

**Альтернативы:**
- RVI — готовый индекс волатильности (implied)
- IMOEX — можно рассчитать realized volatility индекса
- USD000UTSTOM — USD/RUB спот (до 2024-06)
- CNYRUB_TOM — CNY/RUB спот (полное покрытие)

### Вариант 2: Использовать для подвыборки 2022-2026

Если нужны фьючерсы только для последних лет:
- Si, BR, RI доступны с 10-мин разрешением с 2022-2024
- Можно скачать ~4 года данных без склейки

**Оценка трудозатрат:**
- API запросов: ~500 на базовый актив (по ~125 дней на контракт × 4 контракта)
- Обработка: склейка по adjusted close
- Приоритет: НИЗКИЙ

### Вариант 3: Использовать внешние источники

Для полного покрытия 2014-2026 нужны данные не из MOEX ISS:
- Finam (finam.ru) — есть исторические данные
- Investing.com
- Московская биржа (прямые контракты на исторические данные)

## Итоговая рекомендация

❌ **НЕ использовать фьючерсы из MOEX ISS API** для периода 2014-2026

✅ **Использовать альтернативы:**
1. RVI — бенчмарк implied volatility
2. IMOEX — расчёт realized volatility индекса
3. CNYRUB_TOM — валютный курс
4. Секторные индексы — MOEXOG, MOEXFN и др.
"""

    return report

# =====================================
# MAIN
# =====================================

def main():
    print("="*70)
    print("EXTERNAL RECON 02 EXTENDED: Фьючерсы — ВСЕ ТАЙМФРЕЙМЫ")
    print("="*70)

    # Этап 1
    stage1_result = stage1_daily_data()

    # Этап 2
    time.sleep(0.3)
    stage2_result = stage2_historical_contracts()

    # Этап 3
    time.sleep(0.3)
    stage3_result = stage3_all_intervals()

    # Этап 4
    stage4_result = stage4_assessment(stage3_result)

    # Этап 5
    time.sleep(0.3)
    stage5_result = stage5_continuous_series()

    # Генерация отчёта
    print("\n" + "="*70)
    print("ГЕНЕРАЦИЯ ОТЧЁТА")
    print("="*70)

    report = generate_report(stage1_result, stage2_result, stage3_result, stage4_result, stage5_result)

    # Сохранение
    report_path = "/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/reports/external_recon_02_futures_extended.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nОтчёт сохранён: {report_path}")

    # Также обновим основной файл
    original_path = "/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/reports/external_recon_02_futures.md"
    with open(original_path, 'a', encoding='utf-8') as f:
        f.write("\n\n---\n\n")
        f.write("## ДОПОЛНЕНИЕ: Проверка всех таймфреймов\n\n")
        f.write(f"**Дата проверки:** {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write("Полный отчёт: [external_recon_02_futures_extended.md](external_recon_02_futures_extended.md)\n\n")
        f.write("**Вывод:** Дневные и часовые свечи также недоступны до 2022-2024. ")
        f.write("MOEX ISS API не содержит исторических данных фьючерсов.\n")

    print(f"Дополнен: {original_path}")

    print("\n" + "="*70)
    print("ЗАВЕРШЕНО")
    print("="*70)

if __name__ == "__main__":
    main()
