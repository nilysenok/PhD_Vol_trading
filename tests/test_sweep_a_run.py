"""Э1-г, PR 2: драйвер 36 сборок через `sweep_drivers.factorial_sweep`, ledger `A`, топ-2, экспорт A′. Синтетика."""
import json

import pandas as pd
import pytest
import yaml

from fastbt import sweep_a as sa
from fastbt.sweep_a import SynthPanelSource


@pytest.fixture(scope="module")
def prog(tmp_path_factory):
    out = tmp_path_factory.mktemp("ledgerA")
    sw = sa.make_sweep(SynthPanelSource(3, ("S1", "S2"), seed=2), out=str(out))
    res = sa.run_sborki(sw)
    sw.save()
    return sw, res, out


def ok_n36(res, sw) -> bool:
    return bool(len(res["table"]) == 36 and res["n"] == 36 and len(sw.trials_sr()) == 36 and res["n_modul"] == 36)


def test_36_sborok_N_v_ledger_A(prog):
    sw, res, out = prog
    assert ok_n36(res, sw)
    assert len(json.loads((out / "ledger.json").read_text("utf-8"))) == 36
    assert len(json.loads((out / "ledger_excess.json").read_text("utf-8"))) == 36


def test_grid_iz_yaml_3x4x3():
    g = sa.load_grid()
    assert [len(v) for v in g.values()] == [3, 4, 3] and list(g) == list(sa.BASE)


def test_top2_po_tseli_setup_i_stroki_dvuh_komissiy(prog):
    _, res, _ = prog
    t, top = res["table"], res["top2"]
    assert len(top) == 2 and top.objective.is_monotonic_decreasing and top.objective.iloc[0] == t.objective.max()
    for col in ("tune_sharpe_c0.04", "tune_sharpe_c0.035", "ctrl_sharpe_c0.04", "ctrl_sharpe_c0.035", "dsr"):
        assert t[col].notna().all()
    assert (t["tune_sharpe_c0.035"] > t["tune_sharpe_c0.04"] - 1e-9).all()      # дешевле комиссия — не хуже


def test_ishodnaya_A_est_v_tablice(prog):
    t = prog[1]["table"]
    assert len(t[(t.vesa == "ravnye") & (t.limit_pozitsiy == "net") & (t.limit_na_strategiyu == "net")]) == 1


def test_eksport_A_prime_format_main_yaml(tmp_path):
    pts = {"S1": {"ma_window": 20, "z_entry": 1.5}, "S2": {"x": 3}}
    sb = {"vesa": "obratnaya_vol", "limit_pozitsiy": 5, "limit_na_strategiyu": "net"}
    f = tmp_path / "a_prime.yaml"
    sa.export_a_prime(pts, sb, f)
    d, m = yaml.safe_load(f.read_text("utf-8")), yaml.safe_load(sa.MAIN_YAML.read_text("utf-8"))
    assert all(d[k] == v for k, v in m.items()) and d["a_prime"]["sborka"] == sb and d["a_prime"]["tochki"] == pts


def test_load_point_luchshaya_po_plato_ne_na_krayu(tmp_path):
    f = tmp_path / "S9_top3.csv"
    f.write_text("# шапка\n" + pd.DataFrame(dict(comm_pct=[0.04, 0.04, 0.04, 0.035], edge=[False, False, True, False],
                                                  plateau=[0.5, 0.9, 5.0, 7.0], **{"p.a": [3, 4, 5, 6], "p.b": [1.5, None, 2.5, 3.5]})).to_csv(index=False), "utf-8")
    assert sa.load_point(f) == {"a": 4}
    with pytest.raises(ValueError):
        sa.load_point(f, 0.05)
