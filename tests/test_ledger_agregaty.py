"""Агрегаты точки v7 в ledger 2.2 (Ж-155 п. 3, Ж-156 п. 4). Синтетика с известным ответом.
Мутации: год по дате входа (сдвиг даты дня) вместо даты дня; режим с днём d (fazy._LAG = 0)."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fastbt.agregaty_v7 as A
import fastbt.ledger_v2 as L
from fastbt import fazy
from fastbt.metriki_e import msr, nedelnye_summy, sharpe, sr_dnevnoy


def test_mutaciya_bez_sqrt252_lovitsya(monkeypatch):
    monkeypatch.setattr(A, "SHKALA", 1.0)
    with pytest.raises(AssertionError):
        assert A.agregaty({"agr": AGR})["sharpe_itog"] == pytest.approx(sharpe(X), abs=1e-12)
    assert A.SHKALA == 1.0 and sr_dnevnoy(X) != sharpe(X)

DATY = np.arange(np.datetime64("2017-01-02"), np.datetime64("2026-03-01"))
DATY = DATY[np.is_busday(DATY)]
T = DATY.size
RNG = np.random.default_rng(7)
X = RNG.normal(2e-4, 1e-2, T)
POZ, SD = RNG.uniform(0, 1, T), RNG.integers(0, 3, T).astype(float)
REZ = RNG.integers(-1, 3, T)
AGR = A.sobrat(DATY, X, POZ, SD, REZ)
GOD = DATY.astype("datetime64[Y]").astype(int) + 1970


def test_sharpe_po_godam_iz_summ_raven_dnevnomu_ryadu():
    ag = A.agregaty({"agr": AGR})
    for g in A.GODY_V7:
        assert ag["sharpe_gody"][g] == pytest.approx(sharpe(X[GOD == g]), abs=1e-12)     # Д7: определение проекта, √252
        assert ag["sdelok_gody"][g] == SD[GOD == g].sum()
        assert ag["ekspoziciya_gody"][g] == pytest.approx(POZ[GOD == g].mean(), abs=1e-12)
    assert ag["sharpe_itog"] == pytest.approx(sharpe(X), abs=1e-12)
    for i, r in enumerate(A.REZHIMY):
        assert ag["sharpe_rezhimy"][r] == pytest.approx(sharpe(X[REZ == i]), abs=1e-12)   # дни без режима (−1) не входят


def test_msr_nedeli_iz_summ_raven_ryadu():
    w = nedelnye_summy(X, DATY)
    ag = A.agregaty({"agr": AGR})
    assert ag["msr_ned_itog"] == pytest.approx(msr(w, 52), rel=1e-8)
    # по годам: недели, чей последний торговый день в ряду лежит в году g
    s = pd.Series(X, index=pd.DatetimeIndex(DATY))
    gr = s.groupby(s.index.to_period("W-FRI"))
    wy = gr.apply(lambda v: v.index[-1].year).to_numpy()
    for g in (2018, 2021, 2025):
        assert ag["msr_ned_gody"][g] == pytest.approx(msr(w[wy == g], 52), rel=1e-8)


def test_rezhim_dnya_po_d_minus_1():
    daty = pd.bdate_range("2017-01-02", periods=300)
    c = pd.Series(100.0, index=daty)
    c.iloc[250:] = 1000.0                      # день 250: скачок → рост по закрытию d, но режим d знает только d−1
    r = A.rezhim_dnya(daty.to_numpy("datetime64[D]"), c)
    assert r[250] == 2 and r[251] == 0         # боковик в день скачка, рост — на следующий
    assert (r[:199] == -1).all()               # прогрев MA200 (+ лаг)


def test_mutaciya_rezhim_s_dnyom_d_lovitsya(monkeypatch):
    daty = pd.bdate_range("2017-01-02", periods=300)
    c = pd.Series(100.0, index=daty)
    c.iloc[250:] = 1000.0
    monkeypatch.setattr(fazy, "_LAG", 0)
    r = A.rezhim_dnya(daty.to_numpy("datetime64[D]"), c)
    with pytest.raises(AssertionError):
        assert r[250] == 2 and r[251] == 0


def test_mutaciya_god_po_date_vhoda_lovitsya(monkeypatch):
    """Год по более ранней дате (вход позиции) = дата дня, сдвинутая на неделю: хвост года уходит в следующий."""
    monkeypatch.setattr(A, "_god", lambda d: (np.asarray(d, "datetime64[D]") + np.timedelta64(7, "D")).astype("datetime64[Y]").astype(int) + 1970)
    ag = A.agregaty({"agr": A.sobrat(DATY, X, POZ, SD, REZ)})
    with pytest.raises(AssertionError):
        assert ag["sharpe_gody"][2018] == pytest.approx(sharpe(X[GOD == 2018]), abs=1e-12)


def test_staraya_stroka_22_chitaetsya_i_agr_prohodit_pisatel(tmp_path):
    kl = L.klyuch("S1", {"a": 1.0}, "wf", 0.0004, None)
    r = {"v": L.V22, "strategiya": "S1", "status": "ok", "klyuch": kl, "kod": "K", "vhody": "V", "klyuch_sha": L.klyuch_sha(kl, "K", "V"),
         "by_year": {"2017": 1.0}, "by_quarter": {"2017Q1": 1.0}, "setka": "v7", "rol": "sobol", "klyuch_n": L.klyuch_n("S1", kl)}
    assert A.agregaty(r) is None               # старая строка 2.2 — без agr
    L.proverit_stroku_22(r, "v7")
    L.proverit_stroku_22({**r, "agr": AGR}, "v7")
    with pytest.raises(ValueError, match="agr"):
        L.proverit_stroku_22({**r, "agr": {**AGR, "r": [[0.0] * 2] * 5}}, "v7")
    assert A.agregaty(json.loads(json.dumps({**r, "agr": AGR}))) == A.agregaty({"agr": AGR})   # JSON-круг без потерь


def test_otkazy_i_razmer():
    with pytest.raises(ValueError):
        A.sobrat(DATY, np.where(np.arange(T) == 3, np.nan, X), POZ, SD, REZ)      # nan в x
    with pytest.raises(ValueError):
        A.sobrat(DATY, X[:-1], POZ, SD, REZ)                                        # разная длина
    with pytest.raises(ValueError):
        A.sobrat(np.array(["2016-12-30"], "datetime64[D]"), X[:1], POZ[:1], SD[:1], REZ[:1])   # прогрев 2016 в период оценки не входит
    assert A.sharpe_iz_summ(0, 0, 0) != A.sharpe_iz_summ(0, 0, 0)                   # пустой год: nan, не 0
    n = len(json.dumps(AGR, separators=(",", ":")).encode())
    print(f"байт agr на точку: {n}")
    assert n < 6000


@pytest.mark.skipif(shutil.which("rsync") is None, reason="нет rsync")
def test_zabrat_agregaty_beryot_tolko_stroki(tmp_path):
    src, dst = tmp_path / "srv", tmp_path / "mak"
    d = src / "ledger/v7_s1/S1"
    for sub in ("port", "tik"):
        (d / "p00" / sub).mkdir(parents=True)
        (d / "p00" / sub / "00000.npy").write_text("x")
    (d / "p00/rows.jsonl").write_text("{}\n")
    (d / "manifest.json").write_text('{\n "v": "2.2"\n}\n')
    for k in ("tables", "logs"):
        (src / k).mkdir()
        (src / k / "f.txt").write_text("x")
    sh = Path(__file__).resolve().parents[1] / "scripts/server/perenos/zabrat.sh"
    r = subprocess.run(["bash", str(sh), "--agregaty", str(src), str(dst)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (dst / "ledger/v7_s1/S1/p00/rows.jsonl").exists() and (dst / "ledger/v7_s1/S1/manifest.json").exists()
    assert not list(dst.rglob("*.npy")), "port/tik (шарды) в режиме --agregaty не берутся"
    r0 = subprocess.run(["bash", str(sh), str(src), str(tmp_path / "all")], capture_output=True, text=True)
    assert r0.returncode == 0 and (tmp_path / "all/ledger/v7_s1/S1/p00/port/00000.npy").exists()   # прежний режим не сломан
