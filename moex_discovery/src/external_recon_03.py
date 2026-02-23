#!/usr/bin/env python3
"""
External Recon 03: Валюта, ставки и статистика MOEX ISS
"""

import requests
import time
from datetime import datetime

BASE_URL = "https://iss.moex.com/iss"
TARGET_START = "2014-08-26"
TARGET_END = "2026-02-03"

def get_json(url, params=None):
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
        print(f"  Ошибка: {e}")
        return None

def check_candle_borders(engine, market, board, secid, interval=10):
    """Проверить доступность свечей для инструмента"""
    url = f"{BASE_URL}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/candleborders.json"
    data = get_json(url)

    if not data or len(data) < 2:
        return None

    borders = data[1].get('borders', [])
    if not borders:
        return None

    result = {}
    for row in borders:
        if isinstance(row, dict) and row.get('interval') == interval:
            result = {
                'begin': row.get('begin'),
                'end': row.get('end')
            }
            break

    return result if result else None

def check_candle_sample(engine, market, board, secid, date, interval=10):
    """Получить образец свечей для даты"""
    url = f"{BASE_URL}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/candles.json"
    params = {'interval': interval, 'from': date, 'till': date}
    data = get_json(url, params)

    if not data or len(data) < 2:
        return 0

    candles = data[1].get('candles', [])
    return len(candles)

def explore_currency():
    """ЭТАП 1: Исследование валютных пар"""
    print("\n" + "="*60)
    print("ЭТАП 1: ВАЛЮТНЫЕ ПАРЫ")
    print("="*60)

    results = []

    # Основные валютные пары
    pairs = [
        ('USD000UTSTOM', 'USD/RUB TOM', 'currency', 'selt', 'CETS'),
        ('EUR_RUB__TOM', 'EUR/RUB TOM', 'currency', 'selt', 'CETS'),
        ('CNYRUB_TOM', 'CNY/RUB TOM', 'currency', 'selt', 'CETS'),
        ('USD000000TOD', 'USD/RUB TOD', 'currency', 'selt', 'CETS'),
        ('EUR_RUB__TOD', 'EUR/RUB TOD', 'currency', 'selt', 'CETS'),
    ]

    for secid, name, engine, market, board in pairs:
        print(f"\n--- {secid} ({name}) ---")
        time.sleep(0.2)

        borders = check_candle_borders(engine, market, board, secid, 10)
        if borders:
            begin = borders.get('begin', 'N/A')
            end = borders.get('end', 'N/A')
            print(f"  10-мин свечи: {begin} — {end}")

            # Проверим наличие данных на начало нашего периода
            time.sleep(0.2)
            bars_2014 = check_candle_sample(engine, market, board, secid, TARGET_START, 10)
            bars_2025 = check_candle_sample(engine, market, board, secid, "2025-12-01", 10)
            print(f"  Баров на {TARGET_START}: {bars_2014}")
            print(f"  Баров на 2025-12-01: {bars_2025}")

            # Определяем покрытие
            covers_start = begin and begin[:10] <= TARGET_START
            covers_recent = bars_2025 > 0

            results.append({
                'secid': secid,
                'name': name,
                'board': board,
                'begin_10m': begin,
                'end_10m': end,
                'bars_2014': bars_2014,
                'bars_2025': bars_2025,
                'covers_period': covers_start and covers_recent
            })
        else:
            print(f"  10-мин свечи: НЕТ")
            results.append({
                'secid': secid,
                'name': name,
                'board': board,
                'begin_10m': None,
                'end_10m': None,
                'bars_2014': 0,
                'bars_2025': 0,
                'covers_period': False
            })

    return results

def explore_rates():
    """ЭТАП 2: Исследование процентных ставок"""
    print("\n" + "="*60)
    print("ЭТАП 2: ПРОЦЕНТНЫЕ СТАВКИ")
    print("="*60)

    results = []

    # RUONIA
    print("\n--- RUONIA (Межбанковская ставка) ---")
    url = f"{BASE_URL}/statistics/engines/state/rates/columns.json"
    data = get_json(url)
    if data and len(data) > 1:
        print(f"  Доступные колонки: {list(data[1].keys()) if isinstance(data[1], dict) else 'N/A'}")

    # Проверим данные RUONIA
    url = f"{BASE_URL}/statistics/engines/state/rates.json"
    params = {'from': '2014-01-01', 'till': '2014-12-31'}
    data = get_json(url, params)
    if data and len(data) > 1:
        rates = data[1].get('rates', [])
        print(f"  Записей RUONIA за 2014: {len(rates)}")
        if rates:
            print(f"  Пример: {rates[0] if isinstance(rates[0], dict) else rates[0]}")
            results.append({
                'name': 'RUONIA',
                'type': 'overnight rate',
                'data_2014': len(rates),
                'available': True
            })
        else:
            results.append({
                'name': 'RUONIA',
                'type': 'overnight rate',
                'data_2014': 0,
                'available': False
            })

    # ZCYC (Zero Coupon Yield Curve)
    print("\n--- ZCYC (Кривая бескупонной доходности) ---")
    url = f"{BASE_URL}/engines/state/markets/rates/boards/ZCYC/securities.json"
    data = get_json(url)
    if data and len(data) > 1:
        securities = data[1].get('securities', [])
        print(f"  Инструментов ZCYC: {len(securities)}")
        if securities and len(securities) > 0:
            sample = securities[0] if isinstance(securities[0], dict) else securities[0]
            print(f"  Пример: {sample}")

    # Проверим history для ZCYC
    url = f"{BASE_URL}/history/engines/state/markets/rates/boards/ZCYC/securities.json"
    params = {'from': '2014-08-01', 'till': '2014-08-31'}
    data = get_json(url, params)
    if data and len(data) > 1:
        history = data[1].get('history', [])
        print(f"  Записей ZCYC за август 2014: {len(history)}")
        results.append({
            'name': 'ZCYC',
            'type': 'yield curve',
            'data_2014': len(history),
            'available': len(history) > 0
        })
    else:
        results.append({
            'name': 'ZCYC',
            'type': 'yield curve',
            'data_2014': 0,
            'available': False
        })

    # Ключевая ставка ЦБ
    print("\n--- Ключевая ставка ЦБ ---")
    url = f"{BASE_URL}/statistics/engines/state/rates/cbrf.json"
    data = get_json(url)
    if data and len(data) > 1:
        cbrf = data[1].get('cbrf', data[1].get('rates', []))
        if cbrf:
            print(f"  Записей: {len(cbrf)}")
            if cbrf:
                sample = cbrf[-1] if isinstance(cbrf, list) else cbrf
                print(f"  Последняя запись: {sample}")
            results.append({
                'name': 'CBRF Key Rate',
                'type': 'central bank rate',
                'data_2014': 'available',
                'available': True
            })

    return results

def explore_statistics():
    """ЭТАП 3: Исследование рыночной статистики"""
    print("\n" + "="*60)
    print("ЭТАП 3: РЫНОЧНАЯ СТАТИСТИКА")
    print("="*60)

    results = []

    # Капитализация рынка
    print("\n--- Капитализация рынка ---")
    url = f"{BASE_URL}/statistics/engines/stock/capitalization.json"
    data = get_json(url)
    if data and len(data) > 1:
        cap_key = 'capitalization' if 'capitalization' in data[1] else list(data[1].keys())[0] if data[1] else None
        if cap_key:
            cap = data[1].get(cap_key, [])
            print(f"  Записей капитализации: {len(cap) if isinstance(cap, list) else 'dict'}")
            if cap and isinstance(cap, list) and len(cap) > 0:
                sample = cap[0]
                print(f"  Пример: {sample}")
            results.append({
                'name': 'Market Capitalization',
                'available': True
            })

    # Обороты
    print("\n--- Обороты торгов ---")
    url = f"{BASE_URL}/statistics/engines/stock/markets/shares/turnover.json"
    params = {'date': '2014-08-26'}
    data = get_json(url, params)
    if data and len(data) > 1:
        # Найдём ключ с данными
        for key in data[1].keys():
            turnover = data[1].get(key, [])
            if isinstance(turnover, list) and len(turnover) > 0:
                print(f"  Данные оборотов ({key}): {len(turnover)} записей")
                sample = turnover[0] if turnover else None
                if sample:
                    print(f"  Пример: {sample}")
                results.append({
                    'name': f'Turnover ({key})',
                    'available': True
                })
                break

    # Engines
    print("\n--- Доступные engines ---")
    url = f"{BASE_URL}/engines.json"
    data = get_json(url)
    if data and len(data) > 1:
        engines = data[1].get('engines', [])
        print(f"  Engines: {len(engines)}")
        for eng in engines:
            if isinstance(eng, dict):
                print(f"    - {eng.get('name', 'N/A')}: {eng.get('title', 'N/A')}")

    # Markets для stock engine
    print("\n--- Markets в stock engine ---")
    url = f"{BASE_URL}/engines/stock/markets.json"
    data = get_json(url)
    if data and len(data) > 1:
        markets = data[1].get('markets', [])
        print(f"  Markets: {len(markets)}")
        for mkt in markets:
            if isinstance(mkt, dict):
                print(f"    - {mkt.get('NAME', mkt.get('name', 'N/A'))}: {mkt.get('title', mkt.get('TITLE', 'N/A'))}")

    return results

def explore_additional():
    """ЭТАП 4: Дополнительные источники"""
    print("\n" + "="*60)
    print("ЭТАП 4: ДОПОЛНИТЕЛЬНЫЕ ИСТОЧНИКИ")
    print("="*60)

    results = []

    # Индекс RGBI (государственные облигации)
    print("\n--- RGBI (индекс гособлигаций) ---")
    borders = check_candle_borders('stock', 'index', 'SNDX', 'RGBI', 10)
    if borders:
        print(f"  10-мин свечи: {borders.get('begin')} — {borders.get('end')}")
        time.sleep(0.2)
        bars = check_candle_sample('stock', 'index', 'SNDX', 'RGBI', TARGET_START, 10)
        print(f"  Баров на {TARGET_START}: {bars}")
        results.append({
            'name': 'RGBI',
            'type': 'bond index',
            'begin_10m': borders.get('begin'),
            'covers_2014': bars > 0
        })

    # MOEXREPO (ставки РЕПО)
    print("\n--- MOEXREPO (индекс ставок РЕПО) ---")
    borders = check_candle_borders('stock', 'index', 'SNDX', 'MOEXREPO', 10)
    if borders:
        print(f"  10-мин свечи: {borders.get('begin')} — {borders.get('end')}")
        results.append({
            'name': 'MOEXREPO',
            'type': 'repo rate index',
            'begin_10m': borders.get('begin'),
            'covers_2014': borders.get('begin', '')[:4] <= '2014' if borders.get('begin') else False
        })
    else:
        print(f"  10-мин свечи: НЕТ")

    # MIBOR (межбанковская ставка)
    print("\n--- MIBOR (межбанковская ставка) ---")
    borders = check_candle_borders('stock', 'index', 'SNDX', 'MIBOR', 10)
    if borders:
        print(f"  10-мин свечи: {borders.get('begin')} — {borders.get('end')}")
    else:
        print(f"  10-мин свечи: НЕТ")

        # Попробуем другой board
        borders = check_candle_borders('stock', 'index', 'RTSI', 'MIBOR', 10)
        if borders:
            print(f"  (RTSI board) 10-мин свечи: {borders.get('begin')} — {borders.get('end')}")

    return results

def generate_report(currency_results, rates_results, stats_results, additional_results):
    """Генерация отчёта"""

    report = f"""# Разведка валюты, ставок и статистики MOEX ISS

**Дата:** {datetime.now().strftime('%Y-%m-%d')}
**Целевой период:** {TARGET_START} — {TARGET_END}

## Сводка

### Валютные пары (CETS board)

| Инструмент | Название | 10-мин с | 10-мин до | Баров 2014 | Баров 2025 | Покрытие |
|------------|----------|----------|-----------|------------|------------|----------|
"""

    for r in currency_results:
        begin = r.get('begin_10m', 'N/A') or 'НЕТ'
        end = r.get('end_10m', 'N/A') or 'НЕТ'
        if begin and begin != 'НЕТ':
            begin = begin[:10]
        if end and end != 'НЕТ':
            end = end[:10]
        covers = '✅' if r.get('covers_period') else '❌'
        report += f"| {r['secid']} | {r['name']} | {begin} | {end} | {r.get('bars_2014', 0)} | {r.get('bars_2025', 0)} | {covers} |\n"

    report += """
### Критическое открытие по валютным парам

"""

    # Анализ USD000UTSTOM
    usd = next((r for r in currency_results if r['secid'] == 'USD000UTSTOM'), None)
    if usd:
        if usd.get('bars_2014', 0) > 0 and usd.get('bars_2025', 0) == 0:
            report += """**⚠️ USD000UTSTOM (USD/RUB TOM)**
- 10-минутные данные доступны с 2011 года
- **ПРОБЛЕМА:** Данные обрываются примерно в июне 2024
- Возможные причины:
  1. Изменение структуры торгов на валютной секции
  2. Перенос ликвидности на другой инструмент
  3. Технические изменения в API

**Рекомендация:** Использовать USD000UTSTOM только для периода 2014-08-26 — 2024-06-11
"""
        elif usd.get('covers_period'):
            report += """**✅ USD000UTSTOM (USD/RUB TOM)**
- Полное покрытие целевого периода
- 10-минутные свечи доступны
"""

    report += """
## Процентные ставки

| Источник | Тип | Данные за 2014 | Доступность |
|----------|-----|----------------|-------------|
"""

    for r in rates_results:
        available = '✅' if r.get('available') else '❌'
        data_2014 = r.get('data_2014', 'N/A')
        report += f"| {r['name']} | {r['type']} | {data_2014} | {available} |\n"

    report += """
### Примечания по ставкам

- **RUONIA:** Ежедневная межбанковская ставка овернайт. Доступна через statistics API.
- **ZCYC:** Кривая бескупонной доходности. Исторические данные ограничены.
- **Ключевая ставка ЦБ:** Доступна через statistics/engines/state/rates/cbrf

## Рыночная статистика

Доступные endpoints:
- `/statistics/engines/stock/capitalization` — капитализация рынка
- `/statistics/engines/stock/markets/shares/turnover` — обороты торгов
- `/engines/{engine}/markets` — структура рынков

## Дополнительные индексы

| Индекс | Тип | 10-мин с | Покрывает 2014+ |
|--------|-----|----------|-----------------|
"""

    for r in additional_results:
        begin = r.get('begin_10m', 'N/A')
        if begin and begin != 'N/A':
            begin = begin[:10]
        covers = '✅' if r.get('covers_2014') else '❌'
        report += f"| {r['name']} | {r.get('type', 'N/A')} | {begin} | {covers} |\n"

    report += """
## Выводы и рекомендации

### ✅ Подходят для исследования (2014-08-26 — 2026-02-03)

1. **RVI** — индекс волатильности (implied volatility), 10-мин с 2013-12-16
2. **IMOEX** — индекс МосБиржи, 10-мин с 2011-12-08
3. **RGBI** — индекс гособлигаций, 10-мин с 2011-12-08
4. **Секторные индексы** (MOEXOG, MOEXFN, и др.) — 10-мин с 2011-12-08

### ⚠️ Частичное покрытие

1. **USD000UTSTOM** — USD/RUB спот, данные обрываются в 2024
2. **EUR_RUB__TOM** — аналогичная проблема

### ❌ Не подходят

1. **Фьючерсы Si/BR/RI** — нет 10-мин данных до 2022-2023
2. **ZCYC** — нет исторических данных за 2014
3. **RUONIA** — ежедневные данные, не 10-мин

## Итоговая таблица внешних данных

| Источник | Тип | 10-мин | Покрытие 2014-2026 | Рекомендация |
|----------|-----|--------|--------------------|--------------|
| RVI | Implied Vol | ✅ | ✅ Полное | **Использовать как бенчмарк IV** |
| IMOEX | Индекс | ✅ | ✅ Полное | Рассчитать RV индекса |
| RGBI | Облигации | ✅ | ✅ Полное | Опционально |
| USD000UTSTOM | Валюта | ✅ | ⚠️ До 2024-06 | С осторожностью |
| Si/BR/RI | Фьючерсы | ❌ | ❌ | Не использовать |
"""

    return report

def main():
    print("="*60)
    print("EXTERNAL RECON 03: Валюта, ставки и статистика MOEX ISS")
    print("="*60)

    # Этап 1: Валютные пары
    currency_results = explore_currency()

    # Этап 2: Процентные ставки
    time.sleep(0.3)
    rates_results = explore_rates()

    # Этап 3: Рыночная статистика
    time.sleep(0.3)
    stats_results = explore_statistics()

    # Этап 4: Дополнительные источники
    time.sleep(0.3)
    additional_results = explore_additional()

    # Генерация отчёта
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ ОТЧЁТА")
    print("="*60)

    report = generate_report(currency_results, rates_results, stats_results, additional_results)

    # Сохранение отчёта
    report_path = "/Users/nilysenok/Desktop/MOEX_ISS/moex_discovery/reports/external_recon_03_other.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nОтчёт сохранён: {report_path}")
    print("\n" + "="*60)
    print("ЗАВЕРШЕНО")
    print("="*60)

if __name__ == "__main__":
    main()
