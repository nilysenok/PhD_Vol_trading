"""Д7.3: сборка календаря событий в data/calendar/*.csv (только GET, пауза >= 1 с).

Источники: cbr.ru, federalreserve.gov, bls.gov, moex.com. Росстат (SSL-цепочка не проверяется) и
дивиденды ISS (эндпоинт /securities/{T}/dividends.json возвращает описание бумаги, /cci/* требует логина)
недоступны -> файлы только с заголовком. Запуск: python3 d73_build_calendar.py [--cache DIR]
"""
import csv, datetime as dt, html, os, re, sys, time, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'calendar')
CACHE = sys.argv[sys.argv.index('--cache') + 1] if '--cache' in sys.argv else '/tmp/d73_raw'
TODAY = dt.date.today().isoformat()
HDR = ['date', 'event', 'source_url', 'retrieved_at', 'type', 'known_from', 'vneplan']
RU = {'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
      'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12}
EN = {m[:3]: i + 1 for i, m in enumerate('January February March April May June July August September October '
                                          'November December'.split())}
MON = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
Y0, Y1 = 2014, 2026


def get(name, url):
    p = os.path.join(CACHE, name)
    if not os.path.exists(p):
        os.makedirs(CACHE, exist_ok=True)
        time.sleep(1.2)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research; polite)'})
        open(p, 'wb').write(urllib.request.urlopen(req, timeout=60).read())
    return open(p, encoding='utf-8', errors='ignore').read()


def text(s):
    s = re.sub(r'<script.*?</script>|<style.*?</style>', '', s, flags=re.S)
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).replace('\xa0', ' ').strip()


def row(d, ev, url, typ, known, vn=''):
    return [d, ev, url, TODAY, typ, known, vn]


def plan_known(d):  # дата публикации плана не видна -> 1 января года события
    return '%s-01-01' % d[:4]


def cbr():
    url = 'https://www.cbr.ru/dkp/cal_mp/'
    s = get('cbr_cal.html', url)
    ty = {m.group(1): m.group(2) for m in re.finditer(r'href="#(t\d+)" data-text="(\d{4}) год"', s)}
    found = []
    for p in re.split(r'<div role="tabpanel"', s)[1:]:
        yr = ty.get(re.search(r'id="tab_content_(t\d+)"', p).group(1))
        if 'main-events_day' in p:
            for day in p.split('<div class="main-events_day">')[1:]:
                m = re.match(r'(\d+)\s+(\w+)\s*(\d{4})?', text(re.search(r'class="date[^"]*">(.*?)</div>', day, re.S).group(1)))
                for t in re.findall(r'<div class="title"><span[^>]*>(.*?)</span>', day, re.S):
                    found.append((dt.date(int(m.group(3) or yr), RU[m.group(2)], int(m.group(1))), text(t)))
        else:
            cur = None
            for tr in re.findall(r'<tr.*?</tr>', p, re.S):
                tds = [text(x) for x in re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)]
                if len(tds) == 2 and re.match(r'\d+ \w+$', tds[0]):
                    d, mn = tds[0].split()
                    cur = dt.date(int(yr), RU[mn], int(d))
                    tds = tds[1:]
                if cur and tds and re.match(r'(Внеочередное )?[Зз]аседание', tds[0]):
                    found.append((cur, tds[0].split('(')[0].strip()))
    rows = []
    for d, t in sorted(set(found)):
        if not re.match(r'(Внеочередное )?[Зз]аседание', t) or not Y0 <= d.year <= Y1:
            continue
        vn = t.startswith('Внеочередное')
        rows.append(row(d.isoformat(), 'CBR key-rate board meeting' + (' (внеочередное)' if vn else ''), url,
                        'факт' if vn else 'план', d.isoformat() if vn else plan_known(d.isoformat()), int(vn)))
    return rows


def fomc():
    mre = r'(%s)(?:/(%s))? (\d+)(?:-(\d+))?' % (MON, MON)
    rows = []

    def add(y, m1, m2, d1, d2, flag, url):
        d = dt.date(y, EN[(m2 or m1)[:3]], int(d2 or d1)).isoformat()
        vn = flag == 'unscheduled'
        ev = 'FOMC meeting' + (' (unscheduled)' if vn else ' (cancelled, planned)' if flag == 'cancelled' else '')
        rows.append(row(d, ev, url, 'факт' if vn else 'план', d if vn else plan_known(d), int(vn)))

    for y in range(2014, 2021):
        url = 'https://www.federalreserve.gov/monetarypolicy/fomchistorical%d.htm' % y
        t = text(get('fomc_%d.html' % y, url))
        for m in re.finditer(mre + r'(?: \((unscheduled|cancelled)\))?(?: Meeting)? - %d' % y, t):
            add(y, m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), url)
    url = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
    t = text(get('fomc_cal.html', url))
    for y in range(2021, 2027):
        i = t.find('%d FOMC Meetings' % y)
        sec = t[i:t.find('Meeting associated', i)]
        for m in re.finditer(mre + r'\*?', sec[len('%d FOMC Meetings' % y):]):
            if m.group(4):
                add(y, m.group(1), m.group(2), m.group(3), m.group(4), None, url)
    return sorted(set(map(tuple, rows)))


def cpi_us():
    rows = []
    for y in range(Y0, Y1 + 1):
        url = 'https://www.bls.gov/schedule/%d/home.htm' % y
        t = text(get('bls_%d.html' % y, url))
        for m in re.finditer(r'\w+day, (\w+) (\d+), (\d{4})(?: \d+:\d+ [AP]M)? Consumer Price Index for (\w+ \d{4})', t):
            d = dt.date(int(m.group(3)), EN[m.group(1)[:3]], int(m.group(2))).isoformat()
            lm = re.search(r'Last Modified Date: (\w+) (\d+), (\d{4})', t[m.end():])  # штамп страницы месяца
            lm = dt.date(int(lm.group(3)), EN[lm.group(1)[:3]], int(lm.group(2))).isoformat() if lm else '0'
            kf, note = plan_known(d), ''
            if lm[:4] == d[:4] and kf < lm <= d:  # график правился в течение года до выхода -> знание с правки
                kf, note = lm, ' [график правился %s]' % lm
            rows.append(row(d, 'US CPI release (%s)%s' % (m.group(4), note), url, 'план', kf, 0))
    return sorted(set(map(tuple, rows)))


def third(y, m, wd):
    d = dt.date(y, m, 1)
    return d + dt.timedelta((wd - d.weekday()) % 7 + 14)


def moex():
    url = 'https://www.moex.com/ru/factsheet/rebalance'
    exp = [row(third(y, m, 3).isoformat(), 'MOEX quarterly futures expiry (Si/RTS/MIX, third Thursday)',
               'https://iss.moex.com/iss/securities/SiH6.json (LSTTRADE; правило сверено 2019-2026)', 'план',
               '%d-01-01' % y, '') for y in range(Y0, Y1 + 1) for m in (3, 6, 9, 12)]
    reb = [row(third(y, m, 4).isoformat(), 'IMOEX rebalance effective (rule: third Friday)', url, 'план',
               '%d-01-01' % y, '') for y in range(Y0, Y1 + 1) for m in (3, 6, 9, 12)]
    t = text(get('rebal.html', url))
    t = t[t.find('Календарь Ребалансировок Индексов Акций'):t.find('Календарь ребалансировок Индексов облигаций')]
    off = []
    mm = r'(\d+)-(\d+) (\w+) (\d{4}) г\. (\d+) (\w+) (\d{4}) г\.'
    for m in re.finditer(mm, t):
        pub_end = dt.date(int(m.group(4)), RU[m.group(3)], int(m.group(2)))
        eff = dt.date(int(m.group(7)), RU[m.group(6)], int(m.group(5)))
        off.append(row(eff.isoformat(), 'IMOEX rebalance effective (official calendar)', url, 'факт',
                       pub_end.isoformat(), ''))
    return exp, reb, off


def write(name, rows):
    with open(os.path.join(OUT, name + '.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HDR)
        w.writerows(sorted(rows))
    print(name, len(rows))


def sources():
    """calendar_sources.csv: по годам - источник, число плановых событий, дата публикации календаря."""
    rows = []
    for n in ('cbr', 'fomc', 'cpi_us', 'cpi_ru', 'dividends_expected', 'dividends_actual', 'expirations', 'imoex_rebal'):
        df = list(csv.DictReader(open(os.path.join(OUT, n + '.csv'), encoding='utf-8')))
        for y in range(Y0, Y1 + 1):
            r = [x for x in df if x['date'][:4] == str(y)]
            pl = [x for x in r if x['vneplan'] != '1']
            url = r[0]['source_url'] if r else 'нет данных'
            rows.append([n, y, len(pl), len(r) - len(pl), url, 'не видна -> 1 января' if r else 'нет данных'])
    with open(os.path.join(OUT, 'calendar_sources.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'year', 'n_plan', 'n_vneplan', 'source_url', 'calendar_published'])
        w.writerows(rows)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    write('cbr', cbr())
    write('fomc', fomc())
    write('cpi_us', cpi_us())
    exp, reb, off = moex()
    write('expirations', exp)
    write('imoex_rebal', reb)
    write('imoex_rebal_official', off)
    for n in ('cpi_ru', 'dividends_expected', 'dividends_actual'):  # нет данных: только заголовок
        write(n, [])
    sources()
