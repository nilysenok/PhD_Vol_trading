"""V6 лист Z3 (Ж-230), G68: документ причин собирается из таблиц; подмена числа, статическое число вне таблиц, лишнее слово — ловятся. Сети и обучения нет."""
import re

import numpy as np
import pandas as pd

from models_v2.s13 import v6_why_data as DT, v6_why_doc2 as W, v6_why_docheck as DC, v6_why_stage as ST


def synth(seed=3):
    dates = pd.bdate_range("2017-01-02", "2026-02-27")
    n, rng = len(dates), np.random.default_rng(seed)
    k1 = np.exp(rng.normal(0, 0.4, n))
    f = {k: k1 * np.exp(np.clip(rng.normal(0, 0.15, n), -1, 1)) for k in DT.SER}
    f["K1"] = k1
    return dict(dates=dates, tp=dates + pd.offsets.BDay(1), rv=k1 * np.exp(rng.normal(0, 0.5, n)), f=f)


def tabs():
    return ST.tables(synth())


def test_dokument_iz_tablic_chisto_i_razdely():
    t = tabs()
    doc = W.build(t)
    assert DC.check_doc(t, doc) == []
    assert doc.startswith("Вывод (измеренные свойства") and DC.count_numbers(t) > 300
    assert DC.static_decimals(t) == set() and DC.check_static(t) == []


def test_mutatsiya_podmenennoe_chislo_v_dokumente_lovitsya():
    t = tabs()
    doc = W.build(t)
    toks = DC.tokens(doc)
    mid = toks[len(toks) // 2]
    bad = doc.replace(mid, mid[:-1] + str((int(mid[-1]) + 1) % 10), 1)
    assert bad != doc and any("≠ пересборке" in x for x in DC.check_doc(t, bad))
    first = re.sub(r"\d+,\d+", lambda m: m.group() + "1", doc.splitlines()[0], count=1)
    assert DC.check_doc(t, doc.replace(doc.splitlines()[0], first, 1))


def test_mutatsiya_chislo_vpisano_rukami_lovitsya(monkeypatch):
    t = tabs()
    monkeypatch.setattr(W, "NOT_CHECKED", W.NOT_CHECKED + "- Вписанное вручную значение 0,77 не из таблицы.\n")
    assert DC.check_static(t) and "0,77" in DC.check_static(t)[0]


def test_mutatsiya_tablica_podmenena_dokument_ne_sovpadaet():
    t = tabs()
    doc = W.build(t)
    u = {k: v.copy() for k, v in t.items()}
    u["WHY_YEARS"].loc[u["WHY_YEARS"].year == "total", "delta_pct"] += 0.5
    assert DC.check_doc(u, doc)


def test_dokument_bez_bps_i_rekomendacij_v7():
    doc = W.build(tabs())
    assert "bps" not in doc.lower() and "V7" not in doc
    assert "DEVELOPMENT / HISTORICAL WALK-FORWARD, диагностика, не отбор; первая цифра — потолок" in doc


def test_znak_pervoj_stroki_sleduet_tablice():
    t = tabs()
    t["WHY_YEARS"].loc[(t["WHY_YEARS"].hybrid == "A2") & (t["WHY_YEARS"].base == "K3") & (t["WHY_YEARS"].year == "total"), "delta_pct"] = -5.0
    assert "Δ вся история −5,00 %" in W.first_line(t)
