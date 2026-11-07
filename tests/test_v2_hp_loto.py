"""P3, G12: LOTO (R11, У-13) через боевой путь `hp_loto.loto_ticker` на синтетической панели (сеть и данные проекта не читаются).
Бумага i не входит ни в обучение, ни в выбор года Y: подмена её целей не меняет LOTO-прогноз на i (побитно). Мутация «бумага i в обучении LOTO» (`E.kept` ≡ всё) — тест падает."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_engine as E, hp_hyb_core as C, hp_hyb_out as O, hp_loto as L

HP = dict(lr=0.2, rounds=15, patience=5, min_leaf=10, l2=1.0, es_dates=20, min_train=100)


def make_panel(seed: int = 0, n: int = 800, tickers=("A", "B", "C", "D")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    parts = []
    for t in tickers:
        z = {c: rng.normal(size=n - 1) for c in C.Z28}
        k1 = np.exp(rng.normal(0, 0.2, n - 1))
        sign = -1.0 if t == "D" else 1.0                                  # D устроена иначе: её цели сильно зависят от признака с обратным знаком
        rv = k1 * np.exp(0.8 * sign * np.sign(z["z_x_d_m"]) + 0.3 * rng.normal(size=n - 1))
        parts.append(pd.DataFrame({"ticker": t, "date": dates[:-1], "t_plus": dates[1:], "rv_tplus": rv, "sigma2_k1": k1, **z}))
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


def loto(panel, tk="D"):
    return L.loto_ticker("xgb", "A1", panel, panel["sigma2_k1"].to_numpy(float), tk, HP, 2)


def swap_targets(panel, tk="D"):
    q = panel.copy()
    m = (q["ticker"] == tk).to_numpy()
    q.loc[m, "rv_tplus"] = q.loc[m, "rv_tplus"].to_numpy() * np.exp(np.random.default_rng(5).normal(1.0, 0.8, m.sum()))
    return q


def test_loto_ne_zavisit_ot_celej_bumagi_i(panel):
    a, b = loto(panel), loto(swap_targets(panel))
    assert len(a) > 100 and (a["ticker"] == "D").all()
    assert np.array_equal(a["sigma2"].to_numpy(), b["sigma2"].to_numpy())


def test_pul_zavisit_ot_celej_bumagi_i_chuvstvitelnost(panel):
    """Тест чувствителен: пул (бумага в обучении) на подмене её целей меняется."""
    k1 = panel["sigma2_k1"].to_numpy(float)

    def pool(p):
        runs = {d: E.walk_forward(p, C.Z28, k1, "qlike", "xgb", d, hp=HP, nthread=2) for d in E.DEPTHS}
        fc, _ = E.finalize(p, k1, runs, "qlike")
        return fc[fc["ticker"] == "D"]["sigma2"].to_numpy()
    assert not np.array_equal(pool(panel), pool(swap_targets(panel)))


def test_mutaciya_bumaga_i_v_obuchenii_loto_lovitsya(panel, monkeypatch):
    monkeypatch.setattr(E, "kept", lambda tk, exclude: np.ones(len(tk), bool))        # мутация: исключаемая бумага остаётся в обучении
    a, b = loto(panel), loto(swap_targets(panel))
    assert not np.array_equal(a["sigma2"].to_numpy(), b["sigma2"].to_numpy())


def test_loto_vybor_goda_po_ostalnym_bumagam(panel, monkeypatch):
    """choose получает панель БЕЗ бумаги i (27 прочих)."""
    seen = []
    orig = E.choose

    def spy(p, base, runs, Y, loss, *a, **k):
        seen.append(set(p["ticker"].unique()))
        return orig(p, base, runs, Y, loss, *a, **k)
    monkeypatch.setattr(E, "choose", spy)
    loto(panel)
    assert seen and all("D" not in s and len(s) == 3 for s in seen)


def test_reestr_loto_odna_stroka_i_used_ne_menyaetsya(tmp_path, monkeypatch):
    reg = tmp_path / "ALL_TRIALS.csv"
    reg.write_text("key,program,source,direction,config,kind\nЖ-217:xgb:A1:abc,Ж-217,x,y,{},trial\n", encoding="utf-8")
    monkeypatch.setattr(O, "REG", reg)
    monkeypatch.setattr(O, "LOCK", tmp_path / "x.lock")
    assert L.register("xgb", "A1") is True and L.register("xgb", "A1") is False
    rows = reg.read_text(encoding="utf-8").splitlines()
    assert sum(r.startswith("Ж-217-LOTO:xgb:A1:") for r in rows) == 1
    assert sum(r.startswith("Ж-217:") for r in rows) == 1             # строки бюджета (used) не тронуты
