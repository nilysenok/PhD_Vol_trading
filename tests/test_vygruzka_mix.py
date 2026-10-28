"""Тесты scripts/data/vygruzka_mix_imoex.py на синтетическом ответе ISS: сети нет, GET подменён opener-ом.

Мутации: (1) обрезка >= 2020 отключена (mutations.md № 36); (2) MXH9 с суффиксом / MXH5 без суффикса пропускаются;
(3) лимит запросов не действует. Мутация подкладывается monkeypatch-ем, тело теста обязано упасть.
"""
import csv
import importlib.util
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

_spec = importlib.util.spec_from_file_location(
    'vm', Path(__file__).resolve().parents[1] / 'scripts' / 'data' / 'vygruzka_mix_imoex.py')
vm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vm)

COLS = ['BOARDID', 'TRADEDATE', 'SECID', 'CLOSE']
DATES = [f'{y}-{m:02d}-{d:02d}' for y in (2019, 2020) for m in range(1, 13) for d in (3, 10, 17)]  # 72 даты, 36 из 2020


class FakeIss:
    """История: 72 строки с датами 2019-2020 (ISS «забыл» till), страницы по 100 -> 1 страница; big=True даёт 250 строк."""

    def __init__(self, big=False):
        self.calls, self.big = [], big

    def __call__(self, url):
        self.calls.append(url)
        secid = Path(urlparse(url).path).stem
        start = int(parse_qs(urlparse(url).query)['start'][0])
        dates = DATES if not self.big else sorted(DATES * 4)[:250]
        rows = [['X', d, secid, 1.0] for d in dates]
        return {'history': {'columns': COLS, 'data': rows[start:start + 100]}}


def _getter(fake, limit=vm.LIMIT):
    return vm.Getter(limit=limit, pause=0, opener=fake)


def _body_clip(tmp):
    fake = FakeIss()
    journal, n = vm.run(tmp, _getter(fake), ['IMOEX', 'MXH9'])
    assert n == 2
    for name, _, rows, dropped, _ in journal:
        assert int(rows) == 36 and int(dropped) == 36
        with open(tmp / name) as f:
            dates = [r['TRADEDATE'] for r in csv.DictReader(f)]
        assert len(dates) == 36 and max(dates) <= '2019-12-31'


def _body_codes(_tmp):
    assert len(vm.MIX) == 20 and len(set(vm.MIX)) == 20
    assert 'MXH9' in vm.MIX and 'MXH9_2019' not in vm.MIX and 'MXH5_2015' in vm.MIX
    for bad in ('MXH9_2019', 'MXH5', 'MXH5_2016'):
        with pytest.raises(ValueError):
            vm.build_url(bad)
    assert '/securities/MXH5_2015.json' in vm.build_url('MXH5_2015')
    assert 'till=2019-12-31' in vm.build_url('MXH9')


def _body_limit(tmp):
    fake = FakeIss(big=True)  # 3 страницы на код
    with pytest.raises(RuntimeError):
        vm.run(tmp, _getter(fake, limit=4), ['IMOEX', 'MXH9'])
    assert len(fake.calls) == 4  # пятый запрос не ушёл


def _body_paging(tmp):
    fake = FakeIss(big=True)
    vm.run(tmp, _getter(fake), ['IMOEX'])
    assert [parse_qs(urlparse(u).query)['start'][0] for u in fake.calls] == ['0', '100', '200']


def _body_from(tmp):
    assert 'from=2015-01-01&till=2019-12-31' in vm.build_url('IMOEX')  # по умолчанию без изменений
    u = vm.build_url('IMOEX', 0, '2014-01-01', '2014-12-31')
    assert 'from=2014-01-01&till=2014-12-31' in u
    with pytest.raises(ValueError):
        vm.build_url('IMOEX', 0, '2014-01-01', '2020-01-03')
    fake = FakeIss()
    j, n = vm.run(tmp, _getter(fake), ['IMOEX'], '2014-01-01', '2014-12-31', '_2014')
    assert 'from=2014-01-01&till=2014-12-31' in fake.calls[0] and n == 1
    assert j[0][0] == 'IMOEX_2014.csv' and (tmp / 'sha_journal_2014.csv').exists()
    assert not (tmp / 'sha_journal.csv').exists() and int(j[0][3]) == 36  # обрезка работает и с --from


@pytest.mark.parametrize('body', [_body_from, _body_clip, _body_codes, _body_limit, _body_paging])
def test_ok(tmp_path, body):
    body(tmp_path)


def test_mutatsiya_bez_obrezki_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, 'clip', lambda cols, rows: (rows, 0))
    with pytest.raises((AssertionError, RuntimeError)):  # RuntimeError: вторая защита write() перед записью
        _body_clip(tmp_path)


def test_mutatsiya_from_bez_obrezki_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, 'clip', lambda cols, rows: (rows, 0))
    with pytest.raises((AssertionError, RuntimeError)):
        _body_from(tmp_path)


def test_mutatsiya_sufiks_ne_togo_goda_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setitem(vm.SOURCES, 'MXH9_2019', ('futures', 'forts'))
    monkeypatch.setitem(vm.SOURCES, 'MXH5', ('futures', 'forts'))
    with pytest.raises((AssertionError, Failed)):
        _body_codes(tmp_path)


def test_mutatsiya_limit_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(vm.Getter, '__call__', lambda self, url: self.opener(url))
    with pytest.raises((AssertionError, Failed)):
        _body_limit(tmp_path)


Failed = pytest.fail.Exception
