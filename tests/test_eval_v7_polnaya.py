"""Период polnaya в источниках v7 (V7-8b): отметка замка обязательна; чтение файлов строго < 2025-01-01 (до акта владельца, Ж-155),
включая файлы индексов (раньше period=wf читал индексы до 2026-03-01); мутация «граница 2025 снята» падает. Синтетика в tmp, без сети."""
import json

import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850 as d18, eval_v7_ist as E, period_v2, vneshnie_1850 as v18

TK = ["AAA", "BBB", "CCC"]
DNI = pd.bdate_range("2024-11-01", "2025-01-10")            # хвост за границей 2025-01-01 лежит в файлах
PRE = pd.bdate_range("2014-09-01", periods=3)


def _bary(k: float) -> pd.DataFrame:
    ts = [d + pd.Timedelta(minutes=m) for d in DNI.append(PRE).sort_values() for m in range(600, 1130, 10)]
    c = k * (1 + 0.01 * np.sin(np.arange(len(ts)) / 7.0 + k))
    return pd.DataFrame({"timestamp": ts, "begin": ts, "open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 100})


@pytest.fixture
def main(tmp_path, monkeypatch):
    for i, t in enumerate(TK):
        p = tmp_path / "moex_discovery/data/raw/candles_10m"
        p.mkdir(parents=True, exist_ok=True)
        _bary(10.0 * (i + 1)).to_parquet(p / f"{t}.parquet")
    q = tmp_path / v18.PATHS["imoex"]
    q.parent.mkdir(parents=True, exist_ok=True)
    _bary(1000.0).to_parquet(q)
    karta = tmp_path / "karta.csv"
    pd.DataFrame({"ticker": TK, "indeks": ["MOEXFN", "MOEXFN", "MOEXOG"], "s": "2000-01-01"}).to_csv(karta, index=False)
    ot = tmp_path / "zamok.json"
    ot.write_text(json.dumps({"preregistraciya_v2": "x", "sha256": "0" * 64, "data_snyatiya": "2026-10-30"}), encoding="utf-8")
    monkeypatch.setattr(period_v2, "OTMETKA", ot)
    monkeypatch.setattr(E.otn, "KARTA_CSV", karta)
    orig = E.sobrat
    monkeypatch.setattr(E, "sobrat", lambda *a, **k: orig(*a, korp=lambda b, t: b, mnozh=lambda t, d, c: np.ones(len(c))))
    return tmp_path


def _chitaet(monkeypatch) -> list:
    """Шпион на оба читателя баров: максимальный timestamp/begin реально прочитанных строк."""
    seen = []
    r18, rv = d18.read_bars, v18.read_bars

    def w18(path, end=None):
        b = r18(path, end)
        seen.append(b["timestamp"].max())
        return b

    def wv(path, end=None):
        b = rv(path, end)
        seen.append(b["timestamp"].max())
        return b
    monkeypatch.setattr(d18, "read_bars", w18)
    monkeypatch.setattr(v18, "read_bars", wv)
    return seen


def test_polnaya_ne_chitaet_stroki_2025(main, monkeypatch):
    """G4: в файлах есть строки до 2025-01-10; прочитано (акции и IMOEX) и собрано — строго < 2025-01-01; первый день прогрева 2014-09."""
    seen = _chitaet(monkeypatch)
    s = E.zagruzit(E.POLNAYA, TK, main=main)
    assert len(seen) == 4 and max(seen) < pd.Timestamp("2025-01-01")          # 3 акции + IMOEX, оба читателя
    assert pd.Timestamp(s.panel.daty.max()) == pd.Timestamp("2024-12-31")
    assert pd.Timestamp(s.panel.daty.min()) == pd.Timestamp("2014-09-01")
    assert all(pd.Timestamp(b["date"].max()) < pd.Timestamp("2025-01-01") for b in s.b10.values())


def test_mutaciya_granica_2025_snyata_lovitsya(main, monkeypatch):
    """Мутант: DO_AKTA = конец wf + 1 день. Либо читаются строки 2025, либо вторая линия (zona.HARD) отказывает — тест выше на нём падает."""
    import dataclasses
    monkeypatch.setitem(period_v2.PERIODY, "polnaya", dataclasses.replace(period_v2.PERIODY["polnaya"], chtenie_do=None))
    seen = _chitaet(monkeypatch)
    try:
        E.zagruzit(E.POLNAYA, TK, main=main)
    except PermissionError:
        return
    assert max(seen) >= pd.Timestamp("2025-01-01")


def test_polnaya_bez_otmetki_otkaz_do_chteniya(main, monkeypatch, tmp_path):
    monkeypatch.setattr(period_v2, "OTMETKA", tmp_path / "net.json")
    seen = _chitaet(monkeypatch)
    with pytest.raises(period_v2.ZamokVladeltsa):
        E.zagruzit(E.POLNAYA, TK, main=main)
    assert seen == []


def test_ist7_nesyot_istochnik_otnositelno(main):
    """IstochnikiV7.ist7() отдаёт Ist7.otn (IMOEX панели): без него ось otnositelno на сервере — ValueError «нет otn»."""
    s = E.zagruzit(E.POLNAYA, TK, main=main)
    i7 = s.ist7()
    assert i7.otn is not None and i7.otn.imoex.index.equals(s.panel.imoex.index) and len(i7.otn.sha) == 64


def test_period_polnaya_opredelen_kak_v_progone():
    p = period_v2.PERIODY["polnaya"]
    assert p.progrev == pd.Timestamp("2014-08-26") and p.y0 == 2017 and p.konec == pd.Timestamp("2026-02-27")   # как в #911
    assert p.ocenka_do == pd.Timestamp("2025-12-31") and p.spravka_s == pd.Timestamp("2026-01-01")   # Ж-159: хвост справочный
    assert p.chtenie_do == period_v2.DO_AKTA == pd.Timestamp("2025-01-01")        # граница чтения до акта — одно место
    assert period_v2.chtenie_konec("polnaya") == pd.Timestamp("2025-01-01")
    assert period_v2.chtenie_konec("wf") == pd.Timestamp("2026-03-01")           # прочие периоды не тронуты
    assert period_v2.PERIODY["v2"].ocenka_do is None and period_v2.PERIODY["wf"].chtenie_do is None


def test_vechernyaya_sessiya_ne_vhodit_v_istochniki():
    """Бары с началом ≥ 18:50 (вечерняя сессия) не меняют ни 10-минутки, ни панель: sha тот же, слоты < 53."""
    base = {t: _bary(10.0 * (i + 1)) for i, t in enumerate(TK)}
    vech = {}
    for t, b in base.items():
        e = b[b["timestamp"].dt.minute == 0].copy()
        e["timestamp"] = e["timestamp"].dt.normalize() + pd.Timedelta(hours=19, minutes=10)
        e[["open", "high", "low", "close"]] *= 3.0
        vech[t] = pd.concat([b, e], ignore_index=True)
    im = pd.Series(1000.0 + np.arange(len(DNI.append(PRE))), index=DNI.append(PRE).sort_values())
    karta = pd.DataFrame({"ticker": TK, "indeks": ["MOEXFN", "MOEXFN", "MOEXOG"]})
    kw = dict(korp=lambda b, t: b, mnozh=lambda t, d, c: np.ones(len(c)))
    a = E.sobrat(base, im, karta, TK, "2014-01-01", "2025-01-01", **kw)
    b = E.sobrat(vech, im, karta, TK, "2014-01-01", "2025-01-01", **kw)
    assert a.sha == b.sha and max(x["slot"].max() for x in b.b10.values()) == 52
