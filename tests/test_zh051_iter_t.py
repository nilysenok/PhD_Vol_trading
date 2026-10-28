"""Драйвер T16–T20 (К-T1) на синтетическом модуле событий с известным ответом. Данных не читает, в сеть не ходит.

Двойник оценщика: событие принимается всегда, исполняется каждое второе (по порядку d) — так «принято» ≠ «сделок».
Три мутации (каждая обязана ронять `_proverki*`): перезапись ledger, ворота по `prinyato`, PBO только по своей матрице.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from fastbt import etap1_poisk_vorota as V, zh051_iter_t as D, zh051_kriteriy as K

IDX = pd.bdate_range("2018-01-02", periods=520)
YEARS = {t: IDX.year.to_numpy() for t in ("AAA", "BBB")}
NOISE = np.random.default_rng(0).normal(0, 0.002, len(IDX))


class FakeTev:
    def __init__(self, shift=0.0):
        self.shift = shift

    def rows(self, events, tau, porog, stop_b=None, dnevnoy=False, lag=1):
        ex, rs, acc = NOISE.copy(), [], {}
        for t in YEARS:
            evs = sorted(ev for ev in events.get(t, []) if ev[0] + lag < len(IDX))      # как postroit_sig: e ≥ n — пропуск
            acc[t] = [(d, d + lag, s, 1.0) for d, s, _ in evs]
            ex_t = [a for a in acc[t][::2]]
            rs.append(NS(trades=[NS(entry_day=e) for _, e, _, _ in ex_t]))
            for _, e, s, _ in ex_t:
                ex[e:e + tau + 1] += 0.0005 * s * (1 + 0.05 * tau)
        if dnevnoy:
            ex = ex + self.shift
        return pd.DataFrame({"excess": ex}, index=IDX), rs, acc


class FakeMod:
    """Событие каждые 4 дня на тикер, сила = k % 6; порог — первая ось (не tau): сила ≥ порога."""
    @staticmethod
    def vhod():
        return None

    @staticmethod
    def sobytiya(inp, osi):
        thr = next(v for k, v in osi.items() if k != "tau")
        return [(t, d, 1 if (d // 4) % 2 else -1, float(k % 6))
                for t in YEARS for k, d in enumerate(range(1, 518, 4)) if k % 6 >= thr]


def _ctx(tmp: Path, shift=0.0) -> D.Ctx:
    (tmp / "SHA256SUMS.txt").write_text("", encoding="utf-8")
    return D.Ctx(FakeTev(shift), YEARS, tmp)


def test_plan_N_sovpadaet_s_csv():
    for it, n in {16: 38, 17: 29, 18: 14, 19: 11, 20: 29}.items():
        assert len(D.plan(it)) == n


def test_plan_s_lishnei_strokoi_stop(tmp_path):
    src = D.CSV.read_text("utf-8").splitlines()
    i = max(k for k, ln in enumerate(src) if ln.startswith("1.0,20,"))
    bad = tmp_path / "s.csv"
    bad.write_text("\n".join(src[:i] + src[i + 1:]) + "\n", encoding="utf-8")       # строка 20-й итерации выброшена
    with pytest.raises(SystemExit):
        D.plan(20, bad)


def test_vorota_po_realnym_sdelkam():
    ctx = D.Ctx(FakeTev(), YEARS)
    ev = D._by_t(FakeMod.sobytiya(None, {"g0": 4.0, "tau": 5}))
    res = D.ocenka(ctx, ev, 5, 1.0)
    for y in D.GODY:
        n_acc = sum(1 for t in ev for d, _, _ in ev[t] if int(YEARS[t][d + 1]) == y)
        assert res["prinyato_y"][y] == n_acc >= 30
        assert res["sdelok_y"][y] == sum(1 for t in ev for d, _, _ in sorted(ev[t])[::2] if int(YEARS[t][d + 1]) == y) < 30
    assert not D.vorota_sdelok(res)


def _proverki_vorota():
    ctx = D.Ctx(FakeTev(), YEARS)
    res = D.ocenka(ctx, D._by_t(FakeMod.sobytiya(None, {"g0": 4.0, "tau": 5})), 5, 1.0)
    assert res["sdelok_y"][2018] < 30 and not D.vorota_sdelok(res)


def test_mutaciya_vorota_po_prinyato_lovitsya(monkeypatch):
    _proverki_vorota()
    monkeypatch.setattr(D, "vorota_sdelok", lambda res: all(res["prinyato_y"][y] >= V.MIN_SDELOK for y in D.GODY))
    with pytest.raises(AssertionError):
        _proverki_vorota()


def _proverka_ledger(tmp: Path):
    ctx = _ctx(tmp)
    D.run_it(18, ctx, FakeMod, tmp / "tab")
    assert (tmp / "T18" / "ledger_it18.pkl").exists()
    try:
        D.run_it(18, ctx, FakeMod, tmp / "tab")
    except SystemExit:
        return
    raise AssertionError("ledger перезаписан")


def test_ledger_ne_perezapisyvaetsya(tmp_path):
    _proverka_ledger(tmp_path)
    led = K.load_ledger("T18", "ledger_it18.pkl", None, tmp_path)
    assert led.n_unique == 14 and all(k[0][0] == "T" for k in led.cache)


def test_mutaciya_perezapis_ledger_lovitsya(tmp_path, monkeypatch):
    def sloman(led, it, root):
        p = root / f"T{it}" / f"ledger_it{it}.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        p.write_bytes(pickle.dumps(led.cache))
    monkeypatch.setattr(D, "net_ledger", lambda it, root: None)
    monkeypatch.setattr(D, "zapisat_ledger", sloman)
    with pytest.raises(AssertionError):
        _proverka_ledger(tmp_path)


def _proverka_pbo(tmp: Path):
    ctx = _ctx(tmp)
    D.run_it(18, ctx, FakeMod, tmp / "tab")
    r = D.run_it(19, ctx, FakeMod, tmp / "tab")
    assert r["N_svoi"] == 11 and r["N_obsh"] == 14 + 11
    return r


def test_pbo_obedinennyi_ispolzuet_obshchuyu_matricu(tmp_path):
    r = _proverka_pbo(tmp_path)
    assert np.isfinite(r["pbo_svoi"]) and np.isfinite(r["pbo_obsh"])


def test_mutaciya_pbo_tolko_svoya_matritsa_lovitsya(tmp_path, monkeypatch):
    orig = D.pbo_obshchii
    monkeypatch.setattr(D, "pbo_obshchii", lambda it, led, root: orig(it, led, tmp_path / "net_takogo"))
    with pytest.raises(AssertionError):
        _proverka_pbo(tmp_path)


def test_kontroli_i_sverka_dnevnogo_puti(tmp_path):
    ctx = _ctx(tmp_path)
    ev = D._by_t(FakeMod.sobytiya(None, {"g0": 0.0, "tau": 5}))
    k = D.kontroli(ctx, ev, 5, 1.0, 18)
    assert len(k["a"]) == len(k["b"]) == 20
    assert D.p_kontrol([1.0] * 20, 0.5) == 21 / 21 and D.p_kontrol([-1.0] * 20, 0.5) == 1 / 21
    assert D.run_it(18, ctx, FakeMod, tmp_path / "t1")["G5_dnevnoy"] is True
    (tmp_path / "y").mkdir()
    ctx3 = _ctx(tmp_path / "y", shift=1e-6)          # дневной путь разошёлся на 1e-6 — ворота «воспроизводится» закрыты
    assert D.run_it(18, ctx3, FakeMod, tmp_path / "t3")["G5_dnevnoy"] is False
