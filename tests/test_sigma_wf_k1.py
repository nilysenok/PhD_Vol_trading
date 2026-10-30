"""К-1 и К-4 (σ̂ C/D на wf, шаги 9–10): матрица без ДСВД и корень данных из FASTBT_MAIN.

Синтетика не читает данных проекта; тесты на реальной матрице пропускаются, если выхода К-1 нет
(путь — WF_K1_OUT, по умолчанию scratchpad листа; исходная матрица — FASTBT_MAIN или основная копия).
Мутации: «ДСВД в матрице», «ночь от воскресенья», «граница 2025 сдвинута», «переменная окружения игнорируется»."""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/dannye"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
import wf_baza_bez_dsvd as K  # noqa: E402

try:
    importlib.import_module("m0_pit_lib")
    LIB = True
except Exception:  # нет moex_volatility / конфига признаков рядом — окружение, не дефект
    LIB = False
ne_lib = pytest.mark.skipif(not LIB, reason="нет m0_pit_lib (moex_volatility / config признаков)")
MAIN = Path(os.environ.get("FASTBT_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))
OUT = Path(os.environ.get("WF_K1_OUT", "/private/tmp/claude-501/-Users-nilysenok-Desktop--------PhD-Vol-trading/"
                          "bd772d77-32ea-43cd-b4a4-016ac48df19f/scratchpad/wf_dannye/sigma910")) / "data/regen/features_pit_v3.parquet"
BASE = MAIN / "data/regen/features_pit_v3.parquet"
DSVD = K.daty_dsvd()
FRI, SAT, SUN, MON = (pd.Timestamp(f"2025-03-{d}") for d in (14, 15, 16, 17))
OWN = ["rv", "n_bars", "close"]


def proverit_net_dsvd(m, dsvd=DSVD):
    bad = m[m.date.isin(dsvd)]
    assert bad.empty, f"в матрице {len(bad)} строк дат ДСВД"


# ---------- календарь ----------
def test_kalendar_dsvd_bez_rabochey_subboty():
    assert len(DSVD) == 76 and SAT in DSVD and SUN in DSVD and FRI not in DSVD and MON not in DSVD
    assert pd.Timestamp("2025-11-01") not in DSVD  # рабочая суббота — основная сессия


# ---------- sobrat: синтетика ----------
def _base():
    dates = [pd.Timestamp(d) for d in ("2024-12-30", "2025-01-09", FRI, SAT, SUN, MON)]
    rows = [dict(date=d, ticker=t, rv=1.0 + i, n_bars=50, close=100.0 + i, ext=7.0, **{c: np.nan for c in K.TARGETS})
            for t in ("AA", "BB") for i, d in enumerate(dates)]
    return pd.DataFrame(rows)


def _own(base, rv=5.0):
    o = base[["date", "ticker"]].copy()
    o["rv"], o["n_bars"], o["close"] = rv, 40, 55.0
    return o


def _t(df):
    return df.assign(**{c: 0.0 for c in K.TARGETS})


def test_dsvd_data_net_v_matrice_i_konec_obrezan():
    b = _base()
    m = K.sobrat(b, _own(b), DSVD, pd.Timestamp("2025-03-16"), OWN, _t)
    proverit_net_dsvd(m)
    assert set(m.date) == {pd.Timestamp(d) for d in ("2024-12-30", "2025-01-09", FRI)}  # понедельник за концом


def test_mutatsiya_dsvd_v_matrice_lovitsya():
    b = _base()
    m = K.sobrat(b, _own(b), set(), pd.Timestamp("2025-03-17"), OWN, _t)  # мутация: ДСВД не исключены
    with pytest.raises(AssertionError):
        proverit_net_dsvd(m)


def test_do_2025_pobajtno_prezhnie_a_posle_pereschitany():
    b = _base()
    m = K.sobrat(b, _own(b), DSVD, pd.Timestamp("2025-03-17"), OWN, _t)
    old = b[b.date < K.GRANICA].reset_index(drop=True)
    assert_frame_equal(m[m.date < K.GRANICA].reset_index(drop=True), old, check_exact=True)
    assert (m.loc[m.date >= K.GRANICA, "rv"] == 5.0).all() and (m.loc[m.date < K.GRANICA, "rv"] < 5.0).all()


def test_mutatsiya_granitsa_sdvinuta_lovitsya(monkeypatch):
    b = _base()
    monkeypatch.setattr(K, "GRANICA", pd.Timestamp("2024-12-01"))  # мутация: пересчёт заходит в 2024
    m = K.sobrat(b, _own(b), DSVD, pd.Timestamp("2025-03-17"), OWN, _t)
    old = b[b.date < pd.Timestamp("2025-01-01")].reset_index(drop=True)
    with pytest.raises(AssertionError):
        assert_frame_equal(m[m.date < pd.Timestamp("2025-01-01")].reset_index(drop=True), old, check_exact=True)


# ---------- ночь понедельника: свечи ----------
def _svechi(d):
    rng = np.random.default_rng(7)
    rows, c = [], 100.0
    for day in (pd.Timestamp("2025-03-12"), pd.Timestamp("2025-03-13"), FRI, SAT, SUN, MON):
        if day in (SAT, SUN):
            c *= 1.25 if day == SAT else 0.85  # выходная сессия живёт своей жизнью
        for k in range(54):  # 09:50 … 18:40
            c *= float(np.exp(rng.normal(0, 0.002)))
            rows.append(dict(begin=day + pd.Timedelta(hours=9, minutes=50 + 10 * k), open=c, high=c, low=c, close=c, volume=10.0))
    pd.DataFrame(rows).assign(ticker="XX").drop(columns="ticker").to_parquet(d / "XX.parquet")
    return pd.DataFrame(rows)


def _rv_ponedelnika(tmp_path, dsvd):
    calc = K.svechi_bez_dsvd(tmp_path, dsvd, ot=pd.Timestamp("2025-03-01"))
    o = K.own_bez_dsvd(calc, ["XX"])
    return float(o.loc[o.date == MON, "rv"].iloc[0])


@ne_lib
def test_noch_ponedelnika_ot_pyatnicy(tmp_path):
    c = _svechi(tmp_path)
    c = c[~c.begin.dt.normalize().isin(DSVD)].reset_index(drop=True)
    r = np.diff(np.log(c.close.to_numpy()))
    mon = (c.begin.dt.normalize() == MON).to_numpy()[1:]
    ozhid = float((r[mon] ** 2).sum())  # первая доходность понедельника — от закрытия пятницы 18:40
    assert _rv_ponedelnika(tmp_path, DSVD) == pytest.approx(ozhid, rel=1e-12)


@ne_lib
def test_mutatsiya_noch_ot_voskresenya_lovitsya(tmp_path):
    c = _svechi(tmp_path)
    c = c[~c.begin.dt.normalize().isin(DSVD)].reset_index(drop=True)
    r = np.diff(np.log(c.close.to_numpy()))
    ozhid = float((r[(c.begin.dt.normalize() == MON).to_numpy()[1:]] ** 2).sum())
    mut = _rv_ponedelnika(tmp_path, set())  # мутация: бары ДСВД остались, ночь от закрытия воскресенья
    assert abs(mut - ozhid) > 0.5 * ozhid  # ночь Вс→Пн (−15 % за выходные) меняет rv в разы


# ---------- реальные данные ----------
real = pytest.mark.skipif(not (OUT.exists() and BASE.exists()), reason="нет выхода К-1 или исходной матрицы")


@real
def test_real_do_2025_pobajtno():
    a, b = pd.read_parquet(BASE), pd.read_parquet(OUT)
    g = K.GRANICA
    assert_frame_equal(a[a.date < g].reset_index(drop=True), b[b.date < g].reset_index(drop=True), check_exact=True)


@real
def test_real_net_dsvd_rabochaya_subbota_est_konec_0302():
    b = pd.read_parquet(OUT)
    proverit_net_dsvd(b)
    assert (b.date == "2025-11-01").sum() == 17 and b.date.max() == pd.Timestamp("2026-02-03")


@real
def test_real_den_sovpadaet_a_ponedelniki_posle_dsvd_menyayutsya():
    a, b = pd.read_parquet(BASE), pd.read_parquet(OUT)
    d = sorted(a.date.unique())
    posle = {x for p, x in zip(d[:-1], d[1:]) if p in DSVD}
    m = a[(a.date >= K.GRANICA) & ~a.date.isin(DSVD)].merge(b, on=["date", "ticker"], suffixes=("_a", "_b"))
    p = m.date.isin(posle)
    for c in ("rv", "bv", "jump", "rsv_pos", "rsv_neg", "n_bars", "close", "rv_annualized"):
        assert (m.loc[~p, c + "_a"] - m.loc[~p, c + "_b"]).abs().max() < 1e-12, c
    assert p.sum() == 595 and (m.loc[p, "rv_a"] - m.loc[p, "rv_b"]).abs().gt(1e-9).mean() > 0.9


# ---------- К-4: корень данных ----------
MODULI = {"r2_nogi": "DATA", "l2_lib": "DATA", "harx": "DATA", "d24_1": "R0", "r2_sborka_test": "MAIN", "vyr_etalony_finaly": "DATA_S"}
STARYJ = "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"


@pytest.mark.parametrize("mod,attr", MODULI.items())
def test_k4_put_iz_peremennoj(mod, attr, tmp_path):
    kod = ("import sys; sys.path.insert(0, %r); import %s as m; print(m.%s)" % (str(ROOT / "scripts/analysis"), mod, attr))
    env = {**os.environ, "FASTBT_MAIN": str(tmp_path), "NUMBA_NUM_THREADS": "1"}
    r = subprocess.run([sys.executable, "-c", kod], env=env, capture_output=True, text=True, cwd=tmp_path)
    if r.returncode and "ModuleNotFoundError" in r.stderr:
        pytest.skip(r.stderr.strip().splitlines()[-1])
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.strip().splitlines()[-1].startswith(str(tmp_path))


@pytest.mark.parametrize("mod", MODULI)
def test_k4_umolchanie_prezhnee_i_net_zhestkogo_puti(mod):
    src = (ROOT / f"scripts/analysis/{mod}.py").read_text()
    assert f"os.environ.get('FASTBT_MAIN', '{STARYJ}')" in src  # по умолчанию — прежний путь
    kod = [s for s in src.splitlines() if STARYJ in s and not s.lstrip().startswith("#")]
    assert all("FASTBT_MAIN" in s for s in kod), kod


def test_mutatsiya_k4_peremennaya_ignoriruetsya_lovitsya():
    src = (ROOT / "scripts/analysis/harx.py").read_text()
    mut = src.replace("os.environ.get('FASTBT_MAIN', '" + STARYJ + "')", "'" + STARYJ + "'")  # мутация: переменная игнорируется
    kod = [s for s in mut.splitlines() if STARYJ in s and not s.lstrip().startswith("#")]
    assert not all("FASTBT_MAIN" in s for s in kod)
