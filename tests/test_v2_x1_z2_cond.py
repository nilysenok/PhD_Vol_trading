"""Z2: реестр (H1, Hc, H2 до счёта; H2 — отказ всегда), условие L1–L3 (известный ответ на 4 сочетаниях, границы), HD (PIT-терцили, бутстрап), бегун отказывает H1 при ложном условии."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_reg as R, x1_z2_cond as CD, x1_z2_hd as HD, x1_z2_reg as ZR, x1_z2_stage as STG, x1_z_reg as ZZ


def test_registry_rows_before_any_count_and_h2_always_refused():
    reg = ZR.R.read().set_index("id")
    for i in ("H1", "Hc"):
        assert reg.at[i, "status"] == "зарегистрирована; условная (L1–L3); после просмотра Y1–Y4, ZD"
        ZR.guard(i)
    assert reg.at["H2", "status"] == "не считается без решения владельца"
    with pytest.raises(SystemExit):
        ZR.guard("H2")
    with pytest.raises(SystemExit):
        ZR.guard("H2", computed_at="31.12.2099 00:00:00")
    with pytest.raises(SystemExit):
        ZR.guard("H3")                                       # незарегистрированная
    with pytest.raises(SystemExit):
        ZR.guard("H1", computed_at="01.01.2000 00:00:00")    # реестр позже прогноза
    assert list(reg.index).index("H1") > list(reg.index).index("ZCp")
    assert all(i in reg.index and reg.at[i, "status"] == s for i, s in ((r[0], r[4]) for r in list(R.ROWS) + list(ZZ.ROWS)))


def test_register_appends_only_missing_rows_bytes_intact(tmp_path):
    p = tmp_path / "t.csv"
    lines = R.TRIALS.read_text(encoding="utf-8").splitlines(keepends=True)
    p.write_text("".join(lines[:-3]), encoding="utf-8")
    before = p.read_bytes()
    assert ZR.register("10.11.2026 10:00:00", p) == 3 and p.read_bytes().startswith(before)
    assert ZR.register("10.11.2026 11:00:00", p) == 0


def tables(l1=True, l2a=True, nev=150, p=1e-9, corr=0.5):
    crit = pd.DataFrame([dict(y="ZB", crit=k, passed=l1) for k in ("K1", "K2", "K5")])
    pairs = pd.DataFrame([dict(y="ZB", base="ZB0", window="all", delta_q=-1.0 if l2a else 1.0)])
    states = pd.DataFrame([dict(y="ZB", state="E", window="all", y_qlike=1.0), dict(y="ZB0", state="E", window="all", y_qlike=2.0 if l2a else 0.5)])
    mo = pd.DataFrame(dict(nev=[nev, 300]))
    dup = pd.DataFrame(dict(var=["N_d", "s_d"], abs_corr=[corr, 0.1]))
    return crit, pairs, states, mo, dict(p_strict=p, p_mix=p / 2), dup


@pytest.mark.parametrize("l1,l2,l3,value,reasons", [(True, True, True, True, []), (False, True, True, False, ["L1"]), (True, False, True, False, ["L2"]), (True, True, False, False, ["L3"])])
def test_condition_known_answer_on_four_combinations(l1, l2, l3, value, reasons):
    crit, pairs, states, mo, lr, dup = tables(l1, l2, 150 if l3 else 40)
    c = CD.condition(crit, pairs, states, False, mo, lr, dup)
    assert c["value"] is value and c["reasons"] == reasons and (c["L1"], c["L2"], c["L3"]) == (l1, l2, l3)


def test_condition_boundaries_and_l2b_alternative():
    assert CD.condition(*tables(nev=100)[:3], False, *tables(nev=100)[3:])["L3"]
    assert not CD.condition(*tables(nev=99)[:3], False, *tables(nev=99)[3:])["L3"]
    assert not CD.condition(*tables(p=0.01)[:3], False, *tables(p=0.01)[3:])["L3"]                    # p < 0,01 строго
    assert CD.condition(*tables(corr=0.8)[:3], False, *tables(corr=0.8)[3:])["L3"]                    # |corr| ≤ 0,8
    assert not CD.condition(*tables(corr=0.81)[:3], False, *tables(corr=0.81)[3:])["L3"]
    c = CD.condition(*tables(l2a=False)[:3], True, *tables(l2a=False)[3:])                            # L2(б) вместо (а)
    assert c["L2"] and not c["L2a"] and c["L2b"]


def test_hd_terciles_use_only_past_e_days_and_bootstrap_known_answer():
    tau = np.r_[np.arange(40, dtype=float), np.arange(40, dtype=float)]
    d = pd.date_range("2018-01-01", periods=80).to_numpy()
    lab = HD.tercile_labels(tau, d)
    assert (lab[:30] == -1).all() and set(lab[30:]) <= {0, 1, 2}
    t2 = tau.copy()
    t2[70:] = 1e6                                                      # будущее не меняет метку дня 60
    assert HD.tercile_labels(t2, d)[60] == lab[60]
    x, g = np.r_[np.full(60, -1.0), np.zeros(60)] + np.random.default_rng(0).normal(0, 0.1, 120), np.r_[np.ones(60), np.zeros(60)].astype(int)
    pt, lo, hi, nv = HD.boot_diff(x, g, 1, 0)
    assert pt < -0.9 and hi < 0 and 1900 < nv <= 2000                  # повторы с пустой группой пропускаются
    assert HD.hd_negative(pd.DataFrame([dict(contrast="c", window="all", diff=-1.0, hi90=-0.1), dict(contrast="c", window="2018_2021", diff=-0.5, hi90=0.2)]))
    assert not HD.hd_negative(pd.DataFrame([dict(contrast="c", window="all", diff=-1.0, hi90=-0.1), dict(contrast="c", window="2018_2021", diff=0.5, hi90=0.9)]))


def test_runner_refuses_h1_on_false_condition_never_computes_and_h2_always(monkeypatch):
    called = []
    monkeypatch.setattr(STG.RUN, "compute_h1", lambda *a: called.append(1))
    c = CD.condition(*tables(l1=False)[:3], False, *tables(l1=False)[3:])
    with pytest.raises(SystemExit, match="условие ложно"):
        STG.run_h1(c, {}, {})
    assert not called
    with pytest.raises(SystemExit):
        STG.main(["--stage", "H2"])
    monkeypatch.setattr(STG, "h1_inputs", lambda: (None,))
    ok = CD.condition(*tables()[:3], False, *tables()[3:])
    monkeypatch.setattr(STG.RUN, "compute_h1", lambda *a: (_ for _ in ()).throw(RuntimeError("считается")))
    with pytest.raises(RuntimeError):                                  # истинное условие — счёт доходит до расчёта
        STG.run_h1(ok, {}, {})
    monkeypatch.setattr(STG.ZR, "guard", lambda cfg, **k: (_ for _ in ()).throw(SystemExit("не зарегистрирована")))
    with pytest.raises(SystemExit, match="не зарегистрирована"):       # без реестра счёта нет даже при истинном условии
        STG.run_h1(ok, {}, {})
