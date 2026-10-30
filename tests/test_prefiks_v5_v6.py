"""Префикс-свойство (§9.1 п. 1 предрегистрации, Д3): прогон до 2019 = префикс прогона до 2021 для оценщиков v5 и v6.
Причина прежнего нарушения (W1c): etap1_data.tick_of брал max шага цены по ВСЕМ годам окна, включая будущие.
Синтетика — шаг цены; реальные данные ≤ 2021 — 2–3 точки на Sk (малые, не счёт Соболя), как em_prefix.py W1c.
Мутации: «шаг по всему окну» (старое поведение) и «шаг включает год самого дня» ловятся."""
import dataclasses
import time

import numpy as np
import pandas as pd
import pytest

from fastbt import etap1, etap1_data as D, etap1_eval, period_v2, progon_sobol as ps

HV = 3                                       # последние бары окна платят выход — в сравнение не входят


def _bars(step_by_year: dict) -> pd.DataFrame:
    rows = []
    for y, st in step_by_year.items():
        d = pd.date_range(f"{y}-03-01", periods=40, freq="D")
        c = 100.0 + st * np.arange(40)
        rows.append(pd.DataFrame(dict(date=d, open=c, high=c + st, low=c - st, close=c + st)))
    return pd.concat(rows, ignore_index=True)


_TICK = D.tick_of


def _tick_vse_okno(b, do):                    # мутация: старое поведение — все годы окна
    return _TICK(b, pd.Timestamp("2100-01-01"))


def _tick_s_dnem_d(b, do):                    # мутация: прогрев включает последний год окна
    return _TICK(b, pd.Timestamp(b["date"].max()) + pd.Timedelta(days=1))


FULL = _bars({2015: 0.1, 2016: 0.1, 2017: 0.1, 2018: 0.1, 2019: 0.2, 2020: 0.2, 2021: 0.2})
DO = pd.Timestamp("2016-01-01")


def _prefix_ok(tick) -> bool:
    short = FULL[pd.to_datetime(FULL["date"]).dt.year <= 2018]
    return tick(short, DO) == tick(FULL, DO)


def test_tick_prichinnyj_ne_zavisit_ot_dliny_okna():
    assert _prefix_ok(D.tick_of)
    assert D.tick_of(FULL, DO) == pytest.approx(0.1)          # шаг 2019+ (0,2) в 2015 не просачивается


def test_tick_do_progreva_bez_bud_goda():
    pre = FULL[pd.to_datetime(FULL["date"]).dt.year <= 2016]
    assert D.tick_of(FULL, pd.Timestamp("2017-01-01")) == D.tick_of(pre, pd.Timestamp("2017-01-01"))


def test_tick_pozdnij_listing_pervyj_god():
    b = _bars({2018: 0.05, 2019: 0.1})                        # нет баров до do=2016: первый год данных
    assert D.tick_of(b, DO) == pytest.approx(0.05)
    assert D.tick_of(b.iloc[:0], DO) == 0.0


@pytest.mark.parametrize("mut", [_tick_vse_okno, _tick_s_dnem_d])
def test_mutaciya_tick_lovitsya(mut):
    assert not _prefix_ok(mut)


def test_build_days_peredayot_progrev(monkeypatch):
    seen = {}
    monkeypatch.setattr(D, "load_bars5m", lambda *a, **k: FULL.assign(pusto=False, ts=pd.to_datetime(FULL["date"])))
    monkeypatch.setattr(D, "tick_of", lambda b, do: seen.setdefault("do", do) and 0.0)
    D.build_days("T", np.array([np.datetime64("2019-12-30")]), np.array([1.0]), 0, 0, pd.Timestamp("2020-01-01"),
                 start=D.START_V1, last_year=2019)
    assert seen["do"] == pd.Timestamp("2016-01-01")


# --- реальные данные ≤ 2021 -------------------------------------------------------------------------------------
def _faza(sid, rz, y1, end, pts):
    old_p, old_y = dict(etap1_eval.PERIODS), dict(period_v2.PERIODY)
    etap1_eval.PERIODS["v2"] = (2015, y1)
    period_v2.PERIODY["v2"] = dataclasses.replace(period_v2.PERIODY["v2"], konec=pd.Timestamp(f"{y1}-12-31"))
    try:
        evr = ps.ocenshchik(sid, etap1._build(sid, None, pd.Timestamp(end), "v2"), rz)
        out = []
        for _i, pt in pts:
            r, _, _ = evr.evaluate_v2(pt, "v2")
            x, _tik, dts, _z, _cnt, _rd = evr._series("v2", evr.comm)
            out.append((r, pd.Series(np.asarray(x, float), index=pd.DatetimeIndex(dts))))
        return out
    finally:
        etap1_eval.PERIODS.clear(), etap1_eval.PERIODS.update(old_p)
        period_v2.PERIODY.clear(), period_v2.PERIODY.update(old_y)


def _tochki(sid, rz, idx):
    kub = ps.load_kub(sid, rz)
    out = [ps.tochki_v5(kub, i0, i0 + 20, rz) for i0 in idx]
    return [g[len(g) // 2] for g in out if g]


def _prefiks(sid, setka):
    """max|Δ| дней (без хвоста) и кварталов ≤ 2019Q3 между окном до 2019 и до 2021; статусы."""
    rz = ps.Rezhim(setka, "v2", "seed", setka == "v6", "t0d", "kontrol" if setka == "v6" else None)
    try:
        pts = _tochki(sid, rz, (0, 1000))
        a = _faza(sid, rz, 2019, "2020-01-01", pts)
        b = _faza(sid, rz, 2021, "2021-12-30", pts)
    except (FileNotFoundError, PermissionError, KeyError, ImportError) as e:
        pytest.skip(f"нет данных dev: {type(e).__name__}: {e}")
    dd, dq, ok = 0.0, 0.0, 0
    for (ra, sa), (rb, sb) in zip(a, b):
        if ra.get("status") != "ok" or rb.get("status") != "ok":
            continue
        ok += 1
        d = sa.iloc[:-HV]
        dd = max(dd, float(np.abs(d.to_numpy() - sb.reindex(d.index).to_numpy()).max()))
        dq = max([dq] + [abs(v - rb["by_quarter"][k]) for k, v in ra["by_quarter"].items()
                         if str(k) < "2019Q4" and k in rb["by_quarter"]])
    assert ok, "ни одной посчитанной точки"
    return dd, dq


@pytest.mark.parametrize("setka,sid", [("v5", "S3"), ("v5", "S4"), ("v6", "S1"), ("v6", "S2"), ("v6", "S6")])
def test_prefiks_realnye_dannye(setka, sid):
    t0 = time.perf_counter()
    dd, dq = _prefiks(sid, setka)
    assert dd <= 1e-12 and dq <= 1e-12, (setka, sid, dd, dq, time.perf_counter() - t0)


def test_mutaciya_tick_po_vsemu_oknu_lovitsya_na_realnyh(monkeypatch):
    monkeypatch.setattr(D, "tick_of", _tick_vse_okno)         # старое поведение: S3 v5 точка из W1c давала 1,3e-5
    dd, dq = _prefiks("S3", "v5")
    assert dd > 1e-12 or dq > 1e-12
