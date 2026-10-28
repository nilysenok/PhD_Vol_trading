"""Тесты scripts/vygruzka_vselennaya.py на фикстуре ISS: сети нет, ISS подменён функцией get.

Мутации (mutations.md, строка добавляется писарем): (1) end >= 2022 проходит; (2) запись в каталог D25;
(3) пропуск страницы пагинации. Каждая подкладывается monkeypatch-ем в модуль, а тело теста обязано упасть.
"""
import hashlib
import importlib.util
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    'vg', Path(__file__).resolve().parents[1] / 'scripts' / 'vygruzka_vselennaya.py')
vg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vg)

BIG_MONTH = '2021-03-01'  # в этом месяце 1-мин баров 1200 = 500 + 500 + 200 (три страницы)


class FakeIss:
    """Ответ ISS: candles.json с пагинацией по 500; запросы записываются."""

    def __init__(self):
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        q = parse_qs(urlparse(url).query)
        iv, fr, till, start = int(q['interval'][0]), q['from'][0], q['till'][0], int(q['start'][0])
        n = 1200 if (iv == 1 and fr == BIG_MONTH) else 3
        t0 = pd.Timestamp(fr)
        step = {1: pd.Timedelta(minutes=1), 10: pd.Timedelta(minutes=10), 24: pd.Timedelta(days=1)}[iv]
        rows = [[str(t0 + k * step), 1.0, 2.0, 0.5, 1.5, 100 + k, 1000.0 + k] for k in range(n)]
        cols = ['begin', 'open', 'high', 'low', 'close', 'volume', 'value']
        return {'candles': {'columns': cols, 'data': rows[start:start + 500]}}


def _no_net(url):
    raise AssertionError(f'обращение к сети: {url}')


def _body_pagination(tmp):
    root = tmp / 'V40'
    vg.run(['SBER'], '2021-03-01', '2021-03-31', ['1m'], out=root, get=FakeIss(), root=root)
    df = pd.read_csv(root / 'bars1m' / 'SBER_2021.csv')
    assert len(df) == 1200 and df['begin'].is_unique
    assert df['volume'].tolist() == [100 + k for k in range(1200)]


def _body_end(tmp):
    root = tmp / 'V40'
    for bad in ('2022-01-01', '2024-12-31'):
        with pytest.raises(ValueError):
            vg.run(['SBER'], '2021-01-01', bad, ['1m', '10m', 'daily'], out=root, get=_no_net, root=root)
    assert not root.exists()


def _body_no_d25(tmp):
    root = tmp / 'V40'
    d25 = tmp / 'D25' / 'bars1m'
    with pytest.raises(ValueError):
        vg.run(['SBER'], '2021-03-01', '2021-03-31', ['1m'], out=d25, get=FakeIss(), root=root)
    assert not (tmp / 'D25').exists()


def test_pagination_collects_all_pages(tmp_path):
    _body_pagination(tmp_path)


def test_end_2022_refused_before_network(tmp_path):
    _body_end(tmp_path)
    vg.check_end('2021-12-31')


def test_d25_and_other_dirs_refused(tmp_path):
    _body_no_d25(tmp_path)
    root = tmp_path / 'V40'
    for bad in (tmp_path / 'moex_discovery', tmp_path / 'V40_x', tmp_path):
        with pytest.raises(ValueError):
            vg.check_out(bad, root)
    assert vg.check_out(root / 'bars1m', root) == (root / 'bars1m').resolve()


def test_mutation_end_2022_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(vg, 'check_end', lambda e: pd.Timestamp(e))
    with pytest.raises(BaseException):
        _body_end(tmp_path)


def test_mutation_write_to_d25_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(vg, 'check_out', lambda out, root=None: Path(out))
    with pytest.raises(BaseException):
        _body_no_d25(tmp_path)


def test_mutation_skip_page_caught(tmp_path, monkeypatch):
    def skipping(get, url):  # пропускает вторую страницу: после первой прыгает на start=1000
        rows, start = [], 0
        while True:
            c = get(f'{url}&start={start}')['candles']
            rows += [dict(zip(c['columns'], r)) for r in c['data']]
            if len(c['data']) < vg.PAGE:
                return rows
            start += 2 * vg.PAGE
    monkeypatch.setattr(vg, 'paginate', skipping)
    with pytest.raises(BaseException):
        _body_pagination(tmp_path)


def test_formats_sha_journal_and_resume(tmp_path):
    root, get = tmp_path / 'V40', FakeIss()
    args = (['SBER'], '2020-12-15', '2021-02-10', ['1m', '10m', 'daily'])
    vg.run(*args, out=root, get=get, root=root)
    b = pd.read_csv(root / 'bars1m' / 'SBER_2020.csv')
    assert list(b.columns) == vg.COLS and (root / 'bars1m' / 'SBER_2021.csv').exists()
    p = pd.read_parquet(root / 'candles_10m' / 'SBER.parquet')
    assert list(p.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    assert str(p['timestamp'].dtype).startswith('datetime64')
    d = pd.read_csv(root / 'daily_value' / 'SBER.csv')
    assert list(d.columns) == ['date', 'value']
    for u in get.calls:  # ни один запрос не выходит за период
        q = parse_qs(urlparse(u).query)
        assert '2020-12-15' <= q['from'][0] <= q['till'][0] <= '2021-02-10'
    j = pd.read_csv(root / 'sha_journal.csv')
    assert len(j) == 4 and (root / 'sha_journal.csv').exists()
    for _, r in j.iterrows():
        assert hashlib.sha256((root / r['file']).read_bytes()).hexdigest() == r['sha256']
    n = len(get.calls)
    vg.run(*args, out=root, get=get, root=root)  # докачка: всё готово, GET не нужны
    assert len(get.calls) == n


def test_retry_after_failure(monkeypatch):
    seq = iter([OSError('429'), OSError('503'), None])
    monkeypatch.setattr(vg.time, 'sleep', lambda s: None)

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"ok": 1}'

    def fake_open(url, timeout=0):
        e = next(seq)
        if e:
            raise e
        return R()
    monkeypatch.setattr(vg.urllib.request, 'urlopen', fake_open)
    assert vg.make_get()('http://x') == {'ok': 1}


def test_dry_run_prints_plan_without_network(tmp_path, capsys):
    existed = vg.ROOT.exists()
    vg.main(['--tickers', 'SBER,LKOH', '--end', '2021-12-31', '--dry-run'], get=_no_net)
    s = capsys.readouterr().out
    assert len(re.findall(r'^(SBER|LKOH)\t2015-01-01\.\.2021-12-31\t(1m|10m|daily)$', s, re.M)) == 6
    assert '6 заданий' in s and vg.ROOT.exists() == existed
    with pytest.raises(ValueError):
        vg.main(['--tickers', 'SBER', '--end', '2022-01-01', '--dry-run'], get=_no_net)
