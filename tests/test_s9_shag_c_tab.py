"""Шаги 1–3 главы 4 (v3), вторая часть: условия С8 v2 по отдельности, отбор при 0,04 и та же точка при 0,035,
устойчивость, разложение (сумма частей = Δ к A), комиссии, dry-run, проверка сдвига. Синтетика, данные не читаются."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from test_s9_shag_c import mini_grid, world  # noqa: E402

from fastbt import shag0  # noqa: E402
from fastbt import shag_c as C  # noqa: E402
from fastbt import shag_c_tab as T  # noqa: E402


@pytest.fixture(scope="module")
def tabs():
    import s9_shag_c
    return s9_shag_c.tables(world(seed=5), "hyb", mini_grid(), 200, n_shift=60, with_sverka=False)


def test_conditions_separate_and_itog(tabs):
    df, t = tabs["setka"], C.tag(0.0004)
    assert len(df) == 5
    c = lambda k: df[f"{k}_{t}"].astype(bool)          # noqa: E731
    assert (c("usl2_k3") == (df[f"sharpe_{t}"] > df[f"sharpe_k3_{t}"])).all()
    assert (c("usl3_k2") == (df[f"p_vs_k2_{t}"] < 0.05)).all()
    assert (c("proshla_c8v2") == (c("usl1_c8") & c("usl2_k3") & c("usl3_k2"))).all()


def _df(rows):
    out = []
    for i, (s40, ok40, s35, s50, ok50) in enumerate(rows):
        r = dict(sigma="hyb", h=1, family="C1", W=126, p1=60 + 10 * i, p2=0, point=f"C1:p{i}")
        for t, s, ok in (("040", s40, ok40), ("035", s35, ok40), ("050", s50, ok50)):
            r.update({f"sharpe_{t}": s, f"proshla_c8v2_{t}": ok, f"usl1_c8_{t}": True, f"u1_strogo_{t}": True,
                      f"u1_odno_isklyuchenie_{t}": True, f"usl2_k3_{t}": True,
                      f"usl3_k2_{t}": ok, f"p_vs_k2_{t}": 0.01 if ok else 0.3})
        out.append(r)
    return pd.DataFrame(out)


def test_otbor_max_sharpe_among_passed_at_040_and_same_point_at_035():
    df = _df([(0.9, False, 0.95, 0.5, False), (0.5, True, 0.55, 0.2, True), (0.3, True, 0.35, 0.4, True)])
    tab, best = T.otbor(df)
    assert best["point"] == "C1:p1"                         # 0,9 не прошла; среди прошедших максимум
    r = tab[tab["rol"] == "та же точка при 0,035"].iloc[0]
    assert r["point"] == "C1:p1" and r["sharpe"] == 0.55 and r["comm_pct"] == pytest.approx(0.035)
    ref = tab[tab["rol"].str.startswith("справочно")].iloc[0]
    assert ref["point"] == "C1:p2" and "ДРУГАЯ" in ref["rol"]   # при 0,05 лучше другая точка — отдельная строка
    assert T.otbor(_df([(0.9, False, 0.9, 0.9, False)]))[1] is None


def test_ustoychivost_edge_plateau_peak():
    df = _df([(0.5, True, 0, 0, True)] * 3)
    df["p1"] = [50, 60, 70]
    df["sharpe_040"] = [0.5, 0.52, 0.51]
    assert list(T.ustoychivost(df, None)["vyvod"]) == ["край", "плато", "плато"]
    df["sharpe_040"] = [0.5, 1.5, 0.2]
    assert T.ustoychivost(df, None)["vyvod"].iloc[1] == "пик"


def test_razlozhenie_sums_and_komissiya(tabs):
    raz = pd.DataFrame(tabs["razlozhenie"])
    d = raz[raz["rezhim"] == "разложение"]
    assert len(d) == 2 and (d["summa_chastey_minus_d"].abs() < 1e-12).all()
    assert (d["chast1_k3_minus_A"] + d["chast2_C_minus_k3"] - d["d_vs_A"]).abs().max() < 1e-12
    assert set(raz[raz["rezhim"] == "закрыто фильтром"]["semya"]) == set(shag0.SIDS)
    kom = pd.DataFrame(tabs["komissiya"])
    assert sorted(kom["comm_pct"].round(3).unique()) == [0.035, 0.04, 0.05, 0.06]
    assert kom[kom["comm_pct"] > 0.04]["prilozhenie"].all() and not kom[kom["comm_pct"] <= 0.04]["prilozhenie"].any()


def test_dry_run_reads_nothing_and_pair_check():
    cmd = [sys.executable, str(ROOT / "scripts" / "analysis" / "s9_shag_c.py")]
    r = subprocess.run(cmd + ["--sigma", "naive", "--shag", "1", "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0 and "36 точек" in r.stdout and "данные σ не читались" in r.stdout
    assert subprocess.run(cmd + ["--sigma", "naive", "--shag", "2", "--dry-run"], capture_output=True).returncode != 0


def test_sdvig_line_blocks_failed_rows():
    assert "НЕ ПРОЙДЕНА" in T.sdvig_line(pd.DataFrame(dict(ryad=["hyb_h1"], ok=[False])))
    assert "НЕ ПРОЙДЕНА" in T.sdvig_line(pd.DataFrame(dict(ryad=[], ok=[])))
    assert "допущены" in T.sdvig_line(pd.DataFrame(dict(ryad=["hyb_h1"], ok=["True"])))


DAY, WIN = "RV_{d+k} дневная", "y_{d+k} = среднее RV дней d+k+1…d+k+5 (h=5)"


def _sd(rows):
    return pd.DataFrame([dict(ryad="hyb_h5", ok=ok, argmax_prirascheniya=am, cel=cel) for ok, am, cel in rows])


def test_sdvig_line_day_blocks_window_does_not():
    assert "НЕ ПРОЙДЕНА" in T.sdvig_line(_sd([(True, 0.0, DAY), (False, 0.0, DAY)]))      # дневная ok=False
    assert "НЕ ПРОЙДЕНА" in T.sdvig_line(_sd([(True, -1.0, DAY)]))                          # дневная argmax != 0
    w = T.sdvig_line(_sd([(True, 0.0, DAY), (False, -1.0, WIN)]))                          # оконная ok=False
    assert "НЕ ПРОЙДЕНА" not in w and "допущены" in w and "справочно" in w.splitlines()[1]
    n = T.sdvig_line(_sd([(False, -1.0, WIN)]))                                            # только оконная
    assert "НЕ ПРОЙДЕНА" not in n and "отсутствует" in n


def test_sdvig_reads_comma_and_semicolon(tmp_path):
    """Формат ch4_dev_sigma (to_csv: «,», с текстом с «;» и «,» в примечании) и формат таблиц шагов («;», «,» десятичная)."""
    df = pd.DataFrame(dict(ryad=["hyb_h1", "hyb_h5", "kontrol:hyb_h1_sdvig-1", "naive"], rol=["a", "a", "контроль: x", "a"],
                           tip=["model"] * 4, corr=[0.87, 0.82, 0.5, 0.7], ok=[True, True, False, False],
                           primechanie=["", "", "уровень: отстаёт; a, b", "x; y, z"]))
    df.to_csv(tmp_path / "c.csv", index=False)
    df.to_csv(tmp_path / "s.csv", index=False, sep=";", decimal=",")
    a, b = T.sdvig(tmp_path / "c.csv", "hyb"), T.sdvig(tmp_path / "s.csv", "hyb")
    assert list(a["ryad"]) == list(b["ryad"]) == ["hyb_h1", "hyb_h5"]
    assert a["corr"].tolist() == b["corr"].tolist() == [0.87, 0.82]
    assert "допущены" in T.sdvig_line(a)
