"""Л5: репетиция (fwd_rehearsal). Известный ответ `rec`, состав таблицы, `--check` ничего не пишет, доступность баров к 18:40, одна обрезка целиком (obrezka).
Мутации: сдвиг строки на +1 день, допуск 0 при расхождении 1e-13, NaN вместо числа, узел выброшен, бары после d 18:40 в источнике (класс № 142 / Д2).
Цен новых файлов не читает (cut <= 2026-02-27). Сети нет."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

import torch  # noqa: F401  (первым: общий OpenMP)
from models_v2.s17 import fwd_reh_cut as RC, fwd_reh_ref as RR, fwd_rehearsal as RH, fwd_src as S

HAVE = RH.OUT.exists() and S.Paths().new_sber.exists()
D = pd.Timestamp("2026-01-15")


def digest(*dirs) -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for d in dirs for p in sorted(d.rglob("*")) if p.is_file()}


def test_rec_izvestnyj_otvet_i_mutacii():
    r = lambda g, f, tol=0.0: RR.rec(D, D, "x", g, f, tol, "obrezka")["ok"]  # noqa: E731
    assert r(1.5, 1.5) and r([1.0, np.nan], [1.0, np.nan]) and r(1.0 + 1e-13, 1.0, 1e-12)
    assert not r(1.0 + 1e-13, 1.0, 0.0)                        # допуск 0 — только побитно
    assert not r(1.0 + 1e-9, 1.0, 1e-12) and not r(-1.5, 1.5)  # перепутанный знак
    assert not r([1.0, np.nan], [1.0, 2.0]) and not r(np.nan, 1.0)
    assert RR.rec(D, D, "x", ["a"], ["b"], 0.0, "obrezka")["ok"] is False and RR.rec(D, D, "x", ["a", "b"], ["a", "b"], 0.0, "obrezka")["ok"] is True


def test_rec_mutaciya_stroka_dnya_plus_odin_lovitsya():
    h = pd.read_parquet(RR.PAN / "har_SBER.parquet").set_index("d")
    s0, s1 = h.loc[D, "sigma2"], h.loc[h.index[h.index.get_loc(D) + 1], "sigma2"]
    assert RR.rec(D, D, "K1", s0, s0, 0.0, "m")["ok"] and not RR.rec(D, D, "K1", s1, s0, 0.0, "m")["ok"]


def synth(skip=None, bad=None) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(RH.days()):
        thr = i in (0, 37)
        for n in RH.NODES + (RH.THROUGH_ONLY if thr else []):
            if n != skip or i:
                rows.append(RR.rec(d, d, n, 1.0, 2.0 if (n, i) == bad else 1.0, 0.0, "skvoznaya" if thr else "obrezka"))
    return pd.DataFrame(rows)


def test_problems_sostav_tablicy_i_mutacii():
    assert len(RH.days()) == 38 and RH.days()[0] == pd.Timestamp("2025-12-30") and RH.days()[-1] == pd.Timestamp("2026-02-26")
    assert RH.problems(synth()) == []
    assert any("нет узла K2" in p for p in RH.problems(synth(skip="K2")))                  # узел выброшен
    assert any("X1: got" in p for p in RH.problems(synth(bad=("X1", 5))))                  # расхождение
    full = synth()
    assert any("дни" in p for p in RH.problems(full[full["d"] != RH.days()[3].date()]))   # не все 38 дней


@pytest.mark.skipif(not HAVE, reason="нет REHEARSAL.csv")
def test_rehearsal_csv_38_dnej_0_rashozhdenij_i_check_nichego_ne_pishet():
    df = pd.read_csv(RH.OUT, comment="#")
    assert df["d"].nunique() == 38 and RH.problems(df) == [] and df["ok"].all()
    assert sorted(df[df["mode"] == "skvoznaya"]["d"].unique()) == ["2025-12-30", "2026-02-26"]
    assert (df[df["node"].isin(["K1", "cal_t_plus", "mkt17", "e_d", "s_d", "cfg8_lam"])]["tol"] == 0).all()   # допуск 1e-12 — только torch / xgboost / lightgbm
    assert set(df[(df["tol"] > 0) & (df["tol"] < 1e8)]["node"]) <= {"K2", "K3", "M", "z", "cfg8", "X1", "X1_state", "ZB", "ZB_state", "ZB0", "prefix_K2", "prefix_K3", "prefix_cfg8", "prefix_X1", "prefix_ZB", "prefix_ZB0"}
    before = digest(RH.OUT.parent, RH.PARTS)
    assert RH.check() == 0
    assert digest(RH.OUT.parent, RH.PARTS) == before


class Leaky(S.Src):
    """Мутация: источник игнорирует cut и отдаёт все бары архива (бары после d 18:40 попадают в признаки)."""

    def bars(self, inst, cut):
        return super().bars(inst, S.ARCHIVE_END)


def test_dostupnost_barov_k_18_40_i_mutaciya_lovitsya():
    ok, det = RC.avail(D, S.Src())
    assert all(ok) and len(ok) == 18 and "без бара в день d" in det
    bad, _ = RC.avail(D, Leaky())
    assert not all(bad)                                                                     # бары d+1…2026-02-27 позже 18:40 дня d


@pytest.fixture(scope="module")
def cut_rows():
    src = S.Src()
    return RC.run_cut(D, src, RR.Ref(src), through=False)


@pytest.mark.skipif(not HAVE, reason="нет новых begin / архива")
def test_obrezka_vse_uzly_ravny_arhivu(cut_rows):
    bad = [r["node"] for r in cut_rows if not r["ok"]]
    assert bad == [] and {r["node"] for r in cut_rows} == set(RH.NODES)
    assert next(r for r in cut_rows if r["node"] == "refit_cfg8")["got"] == "'2026-01-05'"
    assert next(r for r in cut_rows if r["node"] == "cal_t_plus")["bitwise"]
