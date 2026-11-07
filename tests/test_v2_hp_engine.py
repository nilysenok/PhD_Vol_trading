"""P1b, G8: движок панельной поправки (hp_engine) через боевой путь walk_forward / finalize на синтетической панели (в сеть не ходит, данных проекта не читает).
Каждое правило — тест на правке и тест на мутации (подмена крючка правила в hp_engine): λ = 0 ≡ база побитно; блок ранней остановки не в обучении (№ 99); пара s⁺ > d не в обучении;
выбор года Y только по закрытым прогнозам (№ 102); исключённая бумага LOTO не в обучении; чужой день t+1 (R4); обрезка будущего; QLIKE grad/hess численно; offset действует."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_engine as E

HP = dict(lr=0.2, rounds=25, patience=5, min_leaf=10, l2=1.0, es_dates=20, min_train=100)
FEATS = ["f1", "f2", "id_date", "id_tp", "id_tk"]
DAY = 86_400_000_000_000


def make_panel(seed: int = 0, n: int = 780, tickers=("A", "B", "C", "D")) -> pd.DataFrame:
    """Строка (бумага, дата d): t⁺ — следующая дата, RV_{t⁺} = base·exp(0.4·sign(f1) + шум), id_* — для проверки состава обучения."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    parts = []
    for k, t in enumerate(tickers):
        f1, f2, base = rng.normal(size=n), rng.normal(size=n), np.exp(rng.normal(0, 0.2, n))
        rv_next = base * np.exp(0.4 * np.sign(f1) + 0.3 * rng.normal(size=n))      # значение на дату i+1 заготовлено заранее
        parts.append(pd.DataFrame({"ticker": t, "date": dates[:-1], "t_plus": dates[1:], "rv_tplus": rv_next[:-1], "base": base[:-1], "f1": f1[:-1], "f2": f2[:-1],
                                   "id_date": dates[:-1].asi8 // DAY, "id_tp": dates[1:].asi8 // DAY, "id_tk": float(k)}))
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def wf(panel, module="xgb", depth=2, **kw):
    return E.walk_forward(panel, FEATS, panel["base"], kw.pop("loss", "qlike"), module, depth, hp=HP, nthread=2, **kw)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


@pytest.fixture
def spy(monkeypatch):
    calls = []
    orig = E.fit_model

    def wrap(module, depth, loss, tr, es, hp, nthread):
        calls.append((tr[0].copy(), es[0].copy()))
        return orig(module, depth, loss, tr, es, hp, nthread)
    monkeypatch.setattr(E, "fit_model", wrap)
    return calls


def audit(calls, r, excluded=()):
    """Нарушения состава обучения: ES в обучении / пересечение, s⁺ > d, исключённая бумага, ES не последние es_dates дат."""
    bad, ds = [], np.sort(r["retrain"].dropna().unique()).astype("datetime64[D]").astype(int)
    assert len(calls) == len(ds) > 12
    for (tr, es), d in zip(calls, ds):
        key = lambda a: set(map(tuple, a[:, [4, 2]]))
        if key(tr) & key(es) or tr[:, 3].max() >= es[:, 3].min():
            bad.append("ES в обучении")
        if max(tr[:, 3].max(), es[:, 3].max()) > d:
            bad.append("s⁺ > d")
        if set(tr[:, 4]) & set(excluded) or set(es[:, 4]) & set(excluded):
            bad.append("исключённая бумага")
        if len(np.unique(es[:, 3])) > HP["es_dates"]:
            bad.append("ES длиннее es_dates")
    return set(bad)


def test_pravilnyj_sostav_obucheniya(panel, spy):
    assert audit(spy, wf(panel)) == set()


def test_mutaciya_es_v_obuchenii_lovitsya(panel, spy, monkeypatch):               # № 99
    monkeypatch.setattr(E, "split_es", lambda tp, n: (np.ones(len(tp), bool), np.isin(tp, np.unique(tp)[-n:])))
    assert "ES в обучении" in audit(spy, wf(panel))


def test_mutaciya_para_s_plus_posle_d_lovitsya(panel, spy, monkeypatch):
    monkeypatch.setattr(E, "pair_closed", lambda tp, d: tp <= d + np.timedelta64(30, "D"))
    assert "s⁺ > d" in audit(spy, wf(panel))


def test_loto_isklyuchennaya_bumaga_ne_v_obuchenii(panel, spy):
    ex = ("C",)
    r = wf(panel, exclude=ex)
    spy_ids = set(np.concatenate([np.unique(t[:, 4]) for t, _ in spy]))
    assert 2.0 not in spy_ids and {0.0, 1.0, 3.0} <= spy_ids                  # id_tk: C = 2
    assert r.loc[panel["ticker"] == "C", "g"].notna().any()                    # прогноз на исключённую бумагу есть


def test_mutaciya_loto_bumaga_v_obuchenii_lovitsya(panel, spy, monkeypatch):
    monkeypatch.setattr(E, "kept", lambda tk, ex: np.ones(len(tk), bool))
    wf(panel, exclude=("C",))
    assert 2.0 in set(np.concatenate([np.unique(t[:, 4]) for t, _ in spy]))


def test_lambda_nol_ravna_baze_pobajtno(panel):
    runs = {d: wf(panel, depth=d) for d in E.DEPTHS}
    out, ch = E.finalize(panel, panel["base"], runs, "qlike", lams=(0.0,), default=(2, 1.0, 0.0))
    b = panel.set_index(["ticker", "date"])["base"].reindex(pd.MultiIndex.from_frame(out[["ticker", "date"]])).to_numpy()
    assert len(out) > 1000 and np.array_equal(out["sigma2"].to_numpy(), b) and (out["lam"] == 0.0).all()
    assert out["g"].notna().sum() > 500                                        # поправка g существует, но λ = 0 её выключает


def test_mutaciya_lambda_nol_cherez_log_exp_lovitsya(panel, monkeypatch):
    monkeypatch.setattr(E, "apply_corr", lambda base, g, s2g, c, lam, loss: np.exp(np.log(base) + lam * np.clip(np.nan_to_num(g), -c, c) + 1e-9))
    out, _ = E.finalize(panel, panel["base"], {2: wf(panel)}, "qlike", lams=(0.0,), default=(2, 1.0, 0.0))
    b = panel.set_index(["ticker", "date"])["base"].reindex(pd.MultiIndex.from_frame(out[["ticker", "date"]])).to_numpy()
    assert not np.array_equal(out["sigma2"].to_numpy(), b)


def sel_inputs():
    """Выбор на 2017: закрытые строки (d и t⁺ в 2016) точны при g = 0; у строки с t⁺ в 2017 (последний день 2016) выброс RV = 1e4·base."""
    p = make_panel(n=780, tickers=("A", "B", "C"))
    p["base"], p["rv_tplus"] = 1.0, 1.0
    p.loc[p["t_plus"] >= "2017-01-01", "rv_tplus"] = 1e4
    g0, g4 = pd.DataFrame({"g": 0.0, "s2g": 0.0}, index=p.index), pd.DataFrame({"g": 0.1, "s2g": 0.0}, index=p.index)
    g4.loc[(p["date"].dt.year == 2016) & (p["t_plus"] >= "2017-01-01"), "g"] = 2.0
    return p, {2: g0, 4: g4}


def test_vybor_goda_tolko_po_zakrytym_prognozam():                              # № 102
    p, runs = sel_inputs()
    s = E.choose(p, p["base"], runs, 2017, "qlike")
    assert s["depth"] == 2 and s["crit"] == 0.0 and s["source"] == "closed 2016" and s["n"] > 500
    assert E.choose(p, p["base"], runs, 2015, "qlike")["source"] == "default"            # нет закрытых прогнозов — умолчания R8


def test_mutaciya_vybor_po_zakrytym_plus_odin_lovitsya(monkeypatch):
    p, runs = sel_inputs()
    monkeypatch.setattr(E, "closed_mask", lambda date, tp, Y: np.asarray(pd.DatetimeIndex(date).year == Y - 1))
    assert E.choose(p, p["base"], runs, 2017, "qlike")["depth"] == 4


def corrupt_after(p: pd.DataFrame, d0: pd.Timestamp, rows_of=None) -> pd.DataFrame:
    """Всё после d0 испорчено (признаки, база, RV), а также RV, известный только после d0 (строки с t⁺ > d0)."""
    rng, q = np.random.default_rng(5), p.copy()
    late = (q["date"] > d0).to_numpy()
    for c in ("f1", "f2", "base"):
        q.loc[late, c] = q.loc[late, c] * np.exp(rng.normal(0, 1.0, late.sum())) + rng.normal(size=late.sum())
    fut = (q["t_plus"] > d0).to_numpy()
    q.loc[fut, "rv_tplus"] = q.loc[fut, "rv_tplus"] * np.exp(rng.normal(0, 1.0, fut.sum()))
    return q


@pytest.mark.parametrize("module", ["xgb", "lgbm"])
def test_obrezka_budushchego_vsej_paneli(panel, module):
    d0 = pd.Timestamp("2017-06-15")
    q = corrupt_after(panel, d0)
    fa, fb = (E.finalize(x, x["base"], {d: wf(x, module, d) for d in E.DEPTHS}, "qlike")[0] for x in (panel, q))
    a, b = (f[f["date"] <= d0].drop(columns="fit_sec").reset_index(drop=True) for f in (fa, fb))     # fit_sec — время, не прогноз
    assert len(a) > 500 and a["g"].notna().sum() > 300
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_mutaciya_obrezka_budushchee_v_obuchenii_lovitsya(panel, monkeypatch):
    d0 = pd.Timestamp("2017-06-15")
    monkeypatch.setattr(E, "pair_closed", lambda tp, d: tp <= d + np.timedelta64(30, "D"))
    fa, fb = (E.walk_forward(x, FEATS, x["base"], "qlike", "xgb", 2, hp=HP, nthread=2) for x in (panel, corrupt_after(panel, d0)))
    m = (panel["date"] <= d0).to_numpy()
    assert not np.array_equal(fa["g"].to_numpy()[m], fb["g"].to_numpy()[m], equal_nan=True)


def test_chuzhoj_den_t_plus_1_ne_menyaet_prognoz(panel, monkeypatch):            # R4
    """Подмена дня t+1 другой бумаги (её строки с d = t+1 и RV дня t+1 в строке d = t) не меняет прогноз на t по строкам бумаги A."""
    t0 = E.month_starts(panel["date"].to_numpy(), 2017)[5]
    dates = np.sort(panel["date"].unique())
    t1 = dates[np.searchsorted(dates, t0) + 1]
    q = panel.copy()
    m = ((q["ticker"] == "B") & ((q["date"] == t1) | (q["date"] == t0))).to_numpy()
    q.loc[m, "rv_tplus"] *= 7.0
    q.loc[(q["ticker"] == "B") & (q["date"] == t1), ["f1", "f2", "base"]] += 3.0
    a = wf(panel)
    sel = ((panel["ticker"] == "A") & (panel["date"] == t0)).to_numpy()
    assert np.array_equal(a["g"].to_numpy()[sel], wf(q)["g"].to_numpy()[sel]) and np.isfinite(a["g"].to_numpy()[sel]).all()
    monkeypatch.setattr(E, "pair_closed", lambda tp, d: tp <= d + np.timedelta64(1, "D") * 3)
    assert not np.array_equal(wf(panel)["g"].to_numpy()[sel], wf(q)["g"].to_numpy()[sel])


def num_check(gh) -> bool:
    rng = np.random.default_rng(1)
    rv, m, h = np.exp(rng.normal(size=60)), rng.normal(size=60) * 0.5, 1e-5
    grad, hess = gh(m, rv)
    g_num = ((m + h) + rv * np.exp(-(m + h)) - (m - h) - rv * np.exp(-(m - h))) / (2 * h)     # L = ln f + RV/f, f = exp(m)
    h_num = (gh(m + h, rv)[0] - gh(m - h, rv)[0]) / (2 * h)
    return bool(np.allclose(grad, g_num, rtol=1e-6, atol=1e-8) and np.allclose(hess, h_num, rtol=1e-5, atol=1e-7))


def test_qlike_grad_hess_chislenno():
    assert num_check(E.qlike_gh)


@pytest.mark.parametrize("mut", [lambda m, rv: (np.exp(-m) * rv - 1.0, rv * np.exp(-m)), lambda m, rv: (1.0 - rv * np.exp(-m), 1.0 + rv * np.exp(-m)), lambda m, rv: (1.0 - rv * np.exp(m), rv * np.exp(m))])
def test_mutaciya_grad_hess_lovitsya(mut):
    assert not num_check(mut)


@pytest.mark.parametrize("module", ["xgb", "lgbm"])
def test_offset_deistvuet(module):
    """RV = exp(0.4·sign(x0) + шум): при offset = 1 оптимум g ≈ 0.4·sign(x0) + 0.045 − 1 (среднее ≈ −0.955), при нулевом offset среднее g ≈ 0.045 — offset доходит до обучения (base_margin / init_score)."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 3))
    y = np.exp(0.4 * np.sign(X[:, 0]) + 0.3 * rng.normal(size=2000))
    hp = {**E.HP, **HP, "rounds": 80, "patience": 10, "min_leaf": 50}
    for off, want in ((1.0, -0.955), (0.0, 0.045)):
        o = np.full(2000, off)
        pred, _, _ = E.fit_model(module, 2, "qlike", (X[:1600], y[:1600], o[:1600]), (X[1600:], y[1600:], o[1600:]), hp, 2)
        g = pred(X[1600:])
        assert abs(g.mean() - want) < 0.15 and np.corrcoef(g, np.sign(X[1600:, 0]))[0, 1] > 0.9


def test_l2log_s2g_i_sigma_polozhitelny(panel):
    r = wf(panel, loss="l2log")
    ok = r["g"].notna()
    assert ok.sum() > 1000 and (r.loc[ok, "s2g"] > 0).all()
    s = E.apply_corr(panel["base"], r["g"], r["s2g"], 1.0, 1.0, "l2log")
    assert np.isfinite(s).all() and (s > 0).all() and np.array_equal(E.apply_corr(panel["base"], r["g"], r["s2g"], 1.0, 0.0, "l2log"), panel["base"].to_numpy())
