"""Все читатели календаря берут на t одну версию экспирации MIX-3.22 (Д3, Д7 «одно определение»).

Читатели: d73_features (и d7_block через него), l2_lib.f_cal, d22_harxp.cal_window, d26_1.cal_lead,
r2_features.cal_window. Мутации — только в памяти: «читатель без known_to»."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import d22_harxp as P  # noqa: E402
import d26_1 as Z  # noqa: E402
import d73_features as D73  # noqa: E402
import l2_lib as L  # noqa: E402
import r2_features as R2  # noqa: E402

CAL = ROOT / "data" / "calendar"
T16 = [pd.Timestamp("2022-03-16")]


def _bez_kt(ev):
    """Мутация: читатель игнорирует known_to."""
    if isinstance(ev, pd.DataFrame):
        return ev.assign(known_to=pd.NaT)
    if isinstance(ev, dict):
        return {k: _bez_kt(v) for k, v in ev.items()}
    return ev[:2] + (np.full(len(ev[2]), np.datetime64("NaT")).astype(ev[2].dtype),)


def _pair(r, n, dte):
    return float(r[n]), float(r[dte])


def _d73(ev):
    f = D73.features("2022-03-16", 22, ev)
    return float(f["n_expirations_h22"]), float(f["dte_expirations"])


def _l2(ev):
    return _pair(L.f_cal(T16, 22, ev).iloc[0], "cal_n_expirations", "cal_dte_expirations")


def _d22(ev):
    return _pair(P.cal_window(T16, 1, 22, ev).iloc[0], "cal_n_expirations", "cal_dte_expirations")


def _d26(ev):
    return _pair(Z.cal_lead(T16, 1, ev).iloc[0], "cal_n_expirations", "cal_dte_expirations")


def _r2(ev):
    return _pair(R2.cal_window(T16, 1, 22, ev).iloc[0], "n_expirations", "dte_expirations")


def _chitateli():
    """{имя: (функция, вход, ожидаемое (n, dte) на 16.03.2022)}. d26.cal_lead(j=1) — окно одного дня: n=0, дней до 24.03 — 8."""
    P._EV.clear()
    p = P._events(CAL)
    P._EV.clear()
    r = pd.read_csv(CAL / "expirations.csv")
    r = pd.DataFrame({"date": pd.to_datetime(r["date"]), "known_from": pd.to_datetime(r["known_from"]),
                      "known_to": D73.known_to_col(r)})
    return {"d73": (_d73, D73.load_events(str(CAL), ("expirations",)), (1, 8)),
            "l2": (_l2, L.load_calendar(("expirations",), CAL), (1, 8)),
            "d22": (_d22, p, (1, 8)), "d26": (_d26, p, (0, 8)), "r2": (_r2, {"expirations": r}, (1, 8))}


def test_g5_16_marta_odna_ekspiraciya_u_kazhdogo_chitatelya() -> None:
    for name, (f, ev, ozh) in _chitateli().items():
        assert f(ev) == ozh, name


def test_g7_myutatsiya_bez_known_to_lovitsya_dlya_kazhdogo() -> None:
    for name, (f, ev, ozh) in _chitateli().items():
        assert f(_bez_kt(ev)) != ozh, name  # без known_to виден и срок 17.03 (до 24.03 — 1 день)


def test_do_2022_known_to_ne_zadan() -> None:
    """Разработка: до 2022-03 known_to не задан ни у одной строки — игнорирование known_to ничего не меняет."""
    d = pd.read_csv(CAL / "expirations.csv", parse_dates=["date", "known_to"])
    assert d[d["date"] < "2022-03-01"]["known_to"].isna().all()
