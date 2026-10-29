"""Синтетика для scripts/revers/r1_sverka_A.py (Ж-124 п. 2): маска строки 0, сходимость лестницы, чтение до 2025, переключатели."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
import r1_sverka_A as V  # noqa: E402

DNI = pd.bdate_range("2022-01-03", "2024-12-27")


def pozicii(d: Path, sdvig: dict) -> None:
    """Файл позиций на каждый прогон: 2 тикера × 2 семьи; прогоны различаются сдвигом доходности."""
    rng = np.random.default_rng(3)
    lr = rng.normal(0, .01, len(DNI))
    for n in V.RUNS:
        rows = []
        for t in ("X", "Y"):
            for s in ("S1", "S2"):
                pos = np.where(np.arange(len(DNI)) % 7 < 4, 1.0, -1.0) * (1 if s == "S1" else .5)
                rows.append(pd.DataFrame(dict(date=DNI, ticker=t, strategy=s, position=pos, position_tf=pos * .8,
                                              lr=lr * pos + sdvig[n])))
        pd.concat(rows).to_parquet(d / f"{n}.parquet")


@pytest.fixture
def papka(tmp_path, monkeypatch):
    sdvig = {n: .0002 * i for i, n in enumerate(V.RUNS)}
    pozicii(tmp_path, sdvig)
    monkeypatch.setattr(V, "stavka", lambda mode: pd.Series(.1 if mode == "pechat" else .12, index=DNI[:1]))
    return tmp_path


def test_stroka0_maska_1_na_0_daet_drugoy_sharp(papka):
    cache = {}
    r0 = V.evaluate(papka, V.BASE, cache)
    assert r0[2] == "R0"
    mut = V.evaluate(papka, dict(V.BASE, mask1=False), cache)        # мутация: строка 0 на маске [0]
    assert mut[2] == "P1" and abs(mut[0] - r0[0]) > 1e-3


def test_kazhdoe_sostoyanie_lestnicy_imeet_progon_i_itog_eto_a_zashch():
    lad = V.L.ladder(lambda s: (0.0, V.run_name(s) or pytest.fail(f"нет прогона: {s}")), V.BASE, V.STEPS, 0.0)
    fin = lad["state"]
    assert V.run_name(fin) == "Q6"
    assert fin == dict(mask1=False, sel=.0005, corp=True, div="full", ov=True, tf=True, comm=.0004, rate="cbr",
                       fwd=True, exit_fee=True, lev_cal=True)                 # конфигурация A s9 (Engine.port при 0,04 %)
    assert len(set(V.RUNS.values())) == len(V.RUNS)


def test_shag_vypal_lovitsya():
    bez_5a = [s for s in V.STEPS if s[0] != "5а"]                        # мутация: шаг «комиссия ряда» выпал
    fin = V.L.ladder(lambda s: (0.0,), V.BASE, bez_5a, 0.0)["state"]
    assert fin["comm"] == 0.0


def test_summa_delt_i_ostatok(papka):
    cache = {}
    ev = lambda s: V.evaluate(papka, s, cache)                         # noqa: E731
    fin = V.L.ladder(ev, V.BASE, V.STEPS, 0.0)
    tgt = fin["last"]
    lad = V.L.ladder(ev, V.BASE, V.STEPS, tgt)
    assert abs(sum(s[5] for s in lad["steps"]) + lad["rest"] - (tgt - lad["first"][0])) < 1e-12
    V.L.check_rest(lad, tgt)
    with pytest.raises(AssertionError):                                  # мутация: итог не тот (остаток 0,01)
        V.L.check_rest(V.L.ladder(ev, V.BASE, V.STEPS, tgt + .01), tgt + .01)


def test_po_godam_tolko_svoy_god(papka):
    p = V.evaluate(papka, V.BASE, {})[3]
    _, y = V.sharpe_po_godam(p)
    x = p["excess"]
    assert y[2023] == pytest.approx(V.S.sharpe(x[x.index.year == 2023]))
    assert y[2022] != pytest.approx(y[2023])


def test_pechat_ostanavlivaetsya_do_2025(tmp_path):
    f = tmp_path / "MSP_A_inst.csv"
    f.write_text("date,trading_ret,total_ret,excess_ret,exposure\n2024-12-27,0.01,0.02,0.015,0.1\n"
                 "2024-12-30,0.0,0.001,0.0005,0.1\n2025-01-03,9,9,9,9\nне,читается,вообще,дальше,\n", encoding="utf-8")
    d = V.chitat_do_2025(f)
    assert list(d.index) == list(pd.to_datetime(["2024-12-27", "2024-12-30"]))
    p = V.pechat(f)
    assert p["kr"].iloc[1] == pytest.approx((0.001 - 0.0005) / 3 * 365)          # дни (d_prev, d], как rate_restored
    f.write_text("date,x\n2024-01-03,1\n2024-01-02,2\n", encoding="utf-8")
    with pytest.raises(ValueError):                                      # не по возрастанию — отказ, а не «стоп раньше»
        V.chitat_do_2025(f)


def test_hold_out_v_poziciyah_otkaz():
    V.proverit_hold(pd.DataFrame(dict(date=pd.to_datetime(["2024-12-30"]))))
    with pytest.raises(PermissionError):
        V.proverit_hold(pd.DataFrame(dict(date=pd.to_datetime(["2024-12-30", "2025-01-03"]))))


def test_ohrana_mask1_na_realnom_ishodnike():
    from fastbt.approach_a import ApproachA
    src = V.year_mask1_src(inspect.getsource(ApproachA.year))
    assert "train_end > u.n" in src and "train_end >= u.n" not in src
    compile(src, "year_mask1", "exec")
    with pytest.raises(ValueError):
        V.year_mask1_src("def year(self, Y):\n    return None\n")


def test_okno_bez_vyhoda_ravno_staroy_formule():
    from fastbt import stats
    rng = np.random.default_rng(5)
    st = rng.normal(0, 1, (3, 4, stats.NF))
    st[..., stats.F_Q] = np.abs(st[..., stats.F_Q]) + 5
    st[..., stats.F_X] = 1.0
    nb = np.full(4, 50.0)
    S, Q, N = st[..., stats.F_S].sum(-1), st[..., stats.F_Q].sum(-1), 200.0
    old = (S / N) / np.sqrt(np.maximum(Q - S * S / N, 0) / (N - 1)) * np.sqrt(252)
    assert np.allclose(stats.window(st, nb, 3, 0.0)[0], old)                     # ОВ выключен = формула до #259
    assert not np.allclose(stats.window(st, nb, 3, 0.01)[0], old)               # ОВ включён — другой отбор
