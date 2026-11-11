"""S-Z-EVAL (Ж-258, ZE.G3), тестировщик: `z_eval_tab`, `z_eval_cfg` — таблицы, статистика, Холм, вердикты. Синтетика (2198 дат D_M, B = 200); независимые формулы QLIKE / Δ / Холма — здесь.
Мутант = подмена настоящей функции через monkeypatch (обёртка портит вход или выход); оракул обязан падать (`caught`). Боевые выходы не пишутся."""
import json

import numpy as np
import pandas as pd
import pytest

import test_s18_z_eval_fx as FX
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import z_eval_cfg as C
from models_v2.s18 import z_eval_tab as T

WEAK = {"S-RCKZ": 0.02, "S-RCKZs": 0.015, "S-RZ": 0.01}
WORSE = {"S-RCKZ": -0.06, "S-RCKZs": -0.02, "S-RZ": -0.02}
V = T.verdict
HYP = ["S-RCKZ|S-RCK", "S-RCKZ|S-RCKZs", "S-RCKZ|W5-G", "X1_R(S-RCKZ)|X1", "S-RCKZ|K2", "S-RCKZ|K3"]


def caught(oracle, *a) -> bool:
    try:
        oracle(*a)
    except AssertionError:
        return True
    return False


def holm(p, N):
    o, out, run = np.argsort(p, kind="stable"), np.zeros(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (N - r) * p[i]))
        out[i] = run
    return out


def build(gain):
    rv, tp, F = FX.world(gain=gain)
    ls = T.losses(rv, F)
    spa = T.spa_blocks(ls)
    return dict(rv=rv, tp=tp, F=F, ls=ls, spa=spa, tb=T.table(rv, tp, F, ls), st=T.stats(rv, tp, ls, spa, 855, 3415))


@pytest.fixture(scope="module")
def weak():
    with pytest.MonkeyPatch.context() as mp:
        FX.patch_boot(mp)
        yield build(WEAK)
    S.idx_for.cache_clear()


@pytest.fixture(scope="module")
def worse():
    with pytest.MonkeyPatch.context() as mp:
        FX.patch_boot(mp)
        yield build(WORSE)
    S.idx_for.cache_clear()


# ───────── константы спецификации и N ─────────
def test_spec_constants_and_registered_n(tmp_path):
    assert (C.THR, C.KTRIM, C.N_HOLM, C.N_WIDE, C.M_FAMILY, C.N_D) == ({"У1-S": 0.415, "У2-S": 0.50, "У3-S": 1.00}, 22, 855, 3415, 6, 2198)
    assert (S.B, S.MEAN_BLOCK, S.SEED, S.nw_lag(C.N_D)) == (10000, 20, 0, 8) and C.KEY == "S.Z.eval"
    assert C.SHOWN == ("K1", "K2", "K3", "B_mix", "X1", "W5-G", "S-RCK", "S-RL1", "S-RCKZ", "S-RZ", "S-RCKZs") and len(C.MCS_SET) == 12 and set(C.MCS_SET) >= {"S-RCKm", "S-RCKc", "B_mix"}
    c = tmp_path / "CHECKPOINT.json"
    cp = {"S.R.registered": dict(n=852, n_wide=3409), "S.A.registered": dict(n=831, n_wide=3367), "S.Z.registered": dict(n=855, n_wide=3415), "x": 1}
    c.write_text(json.dumps(cp), encoding="utf-8")
    assert C.registered_n(c) == (855, 3415)
    cp.pop("S.Z.registered")
    c.write_text(json.dumps(cp), encoding="utf-8")
    with pytest.raises(SystemExit):
        C.registered_n(c)                                      # без S.Z.registered оценка отказана (№ 30)
    cp["S.Z.registered"] = dict(n=855)
    c.write_text(json.dumps(cp), encoding="utf-8")
    with pytest.raises(SystemExit):
        C.registered_n(c)                                      # без n_wide — отказ


# ───────── таблица: QLIKE на всех датах; база S-RCK; обрезка дат ─────────
def oracle_table(tb, rv, tp, F):
    assert list(tb["series"][:len(C.SHOWN)]) == list(C.SHOWN)
    for _, r in tb.iterrows():
        q = float(np.mean(FX.ql(rv, F[r["series"]])))
        assert r["n"] == len(tp) == C.N_D and r["first_tplus"] == C.FIRST and r["last_tplus"] == C.LAST
        assert abs(r["qlike"] / q - 1) < 1e-12
        for b in C.BASES:
            qb = float(np.mean(FX.ql(rv, F[b])))
            assert abs(r["d_%s_pct" % b] - 100 * (q - qb) / qb) < 1e-9


def test_table_is_qlike_on_all_dates_and_delta_in_percent_of_base(weak):
    oracle_table(weak["tb"], weak["rv"], weak["tp"], weak["F"])
    r = weak["tb"].set_index("series").loc["S-RCK"]
    assert r["d_S-RCK_pct"] == 0.0 and set(weak["tb"]["role"]) == {"main", "layer X1"} and len(weak["tb"]) == 11 + 4


def test_mut_trimmed_dates_in_main_delta_caught(weak):
    d = weak["ls"][FX.Z] - weak["ls"]["S-RCK"]
    o = np.argsort(d, kind="stable")
    keep = np.ones(len(d), bool)
    keep[o[:22]] = False
    keep[o[-22:]] = False
    t2 = T.table(weak["rv"][keep], weak["tp"][keep], {k: v[keep] for k, v in weak["F"].items()}, {k: v[keep] for k, v in weak["ls"].items()})
    assert t2["n"].iloc[0] == C.N_D - 44                       # мутант действительно обрезан
    assert caught(oracle_table, t2, weak["rv"], weak["tp"], weak["F"])


def test_mut_base_not_srck_caught(weak):
    t2 = weak["tb"].copy()
    t2["d_S-RCK_pct"] = t2["d_W5-G_pct"]                       # «база процентов — не S-RCK»
    assert caught(oracle_table, t2, weak["rv"], weak["tp"], weak["F"])


def test_mut_qlike_as_mean_of_annual_means_caught_n160(weak):
    y = weak["tp"].year.to_numpy()
    ann = {k: float(np.mean([v[y == g].mean() for g in np.unique(y)])) for k, v in weak["ls"].items()}
    day = {k: float(v.mean()) for k, v in weak["ls"].items()}
    assert abs(ann[FX.Z] / day[FX.Z] - 1) > 1e-6               # синтетика различает определения (2026 — неполный год)
    t2 = weak["tb"].copy()
    t2["qlike"] = [ann[s] for s in t2["series"]]
    assert caught(oracle_table, t2, weak["rv"], weak["tp"], weak["F"])


# ───────── семейство, Холм, N ─────────
def test_family_is_six_hypotheses_and_hz2_base_is_srckzs(weak):
    m = weak["st"][weak["st"]["kind"] == "main"]
    assert list(m["hypothesis"]) == HYP and list(m["base"])[1] == "S-RCKZs" and set(m["m_family"]) == {6.0}
    assert set(weak["st"][weak["st"]["kind"] == "window"]["window"]) == {"h1_2018-2021", "h2_2022-2026"}


def test_mut_hz2_against_w5g_instead_of_srckzs_caught(weak, monkeypatch):
    monkeypatch.setattr(T, "FAM", tuple((h, "W5-G") if b == "S-RCKZs" else (h, b) for h, b in T.FAM))
    st = T.stats(weak["rv"], weak["tp"], weak["ls"], weak["spa"], 855, 3415)
    got = list(st[st["kind"] == "main"]["hypothesis"])
    assert got != HYP and "S-RCKZ|S-RCKZs" not in got         # H-Z2 уехала на другую базу


def test_holm_known_answer_and_stats_use_program_n(weak):
    assert np.allclose(S.holm_n([0.001, 0.01, 0.04], 855), [0.855, 1.0, 1.0]) and np.allclose(S.holm_n([0.001, 0.01, 0.04], 3), [0.003, 0.02, 0.04])
    m = weak["st"][weak["st"]["kind"] == "main"]
    p = m["p_dm"].to_numpy()
    assert not np.allclose(holm(p, 855), holm(p, 6))           # синтетика различает N = 855 и N = числу строк
    for col, N in (("p_holm_N", 855), ("p_holm_Nwide", 3415), ("p_holm_m", 6)):
        assert np.allclose(m[col], holm(p, N), rtol=1e-12, atol=0)
    assert (weak["st"]["n_holm"] == 855).all() and (weak["st"]["n_holm_wide"] == 3415).all()


def test_mut_holm_n_equals_row_count_caught_n103(weak, monkeypatch):
    orig = S.holm_n
    monkeypatch.setattr(S, "holm_n", lambda p, N: orig(p, len(p)))
    st = T.stats(weak["rv"], weak["tp"], weak["ls"], weak["spa"], 855, 3415)
    m = st[st["kind"] == "main"]
    assert not np.allclose(m["p_holm_N"], holm(m["p_dm"].to_numpy(), 855), rtol=1e-12, atol=0)


# ───────── вердикты: знак, «лучше / хуже», «эквивалентно», «подтверждено» ─────────
def oracle_verdict(fn):
    assert fn(-0.1, 0.01, 0.01, "S-RCK") == "лучше S-RCK; подтверждено (p Холма N < 0,05 и p SPA < 0,05)"
    for d, ph, ps in ((-0.1, 0.05, 0.01), (-0.1, 0.01, 0.05), (-0.1, 0.5, 0.5), (-0.1, 0.01, float("nan")), (-0.1, 0.0499, 1.0)):
        assert fn(d, ph, ps, "W5-G") == "численно лучше W5-G; не подтверждено (p Холма N ≥ 0,05 или p SPA ≥ 0,05)"      # граница 0,05 — строго «<»
    for d in (0.0, 0.2):
        for ph, ps in ((0.001, 0.001), (0.9, 0.9)):
            assert fn(d, ph, ps, "X1") == "численно не лучше X1; не подтверждено"
    for d, ph, ps in ((-0.1, 0.9, 0.9), (0.1, 0.001, 0.001), (-0.1, 0.001, 0.9)):
        assert "эквивал" not in fn(d, ph, ps, "K2")           # отсутствие значимости ≠ эквивалентность (разд. 7.5)


def test_verdict_known_answers():
    oracle_verdict(T.verdict)


OK_TXT = "лучше %s; подтверждено (p Холма N < 0,05 и p SPA < 0,05)"


@pytest.mark.parametrize("name,mut", [
    ("знак Δ перевёрнут", lambda d, ph, ps, b: V(-d if d != 0 else 1.0, ph, ps, b)),
    ("«численно не лучше» → «хуже»", lambda d, ph, ps, b: V(d, ph, ps, b).replace("численно не лучше", "хуже")),
    ("подтверждено без SPA", lambda d, ph, ps, b: (OK_TXT % b) if d < 0 and ph < 0.05 else V(d, ph, ps, b)),
    ("эквивалентно при p ≥ 0,05", lambda d, ph, ps, b: "эквивалентно %s" % b if ph >= 0.05 else V(d, ph, ps, b)),
    ("нестрогое p ≤ 0,05", lambda d, ph, ps, b: (OK_TXT % b) if d < 0 and ph <= 0.05 and ps <= 0.05 else V(d, ph, ps, b)),
    ("Δ ≥ 0 названо «лучше» (№ 125)", lambda d, ph, ps, b: V(-abs(d) - 1e-9, ph, ps, b)),
])
def test_mut_verdict_caught(name, mut):
    assert caught(oracle_verdict, mut), name


def oracle_stats_verdicts(st):
    for _, r in st[st["kind"] == "main"].iterrows():
        conf = bool(r["delta_pct"] < 0 and r["p_holm_N"] < 0.05 and r["p_spa_c"] < 0.05)
        assert r["verdict"].startswith("лучше") == conf
        assert (r["delta_pct"] >= 0) == r["verdict"].startswith("численно не лучше")
        assert "эквивал" not in r["verdict"]


def test_stats_verdicts_follow_rule_good_and_worse_world(weak, worse):
    oracle_stats_verdicts(weak["st"])
    oracle_stats_verdicts(worse["st"])
    m = worse["st"][worse["st"]["kind"] == "main"]
    assert (m["delta_pct"] > 0).any() and m["verdict"].str.startswith("численно не лучше").any()      # Δ ≥ 0 в мире «хуже» есть


def test_mut_sign_of_delta_flipped_in_stats_caught(weak, monkeypatch):
    orig = T.AT._row
    monkeypatch.setattr(T.AT, "_row", lambda kind, win, hid, h, b, ls, tp, m: orig(kind, win, hid, b, h, ls, tp, m))     # Δ = (b − h)/h
    st = T.stats(weak["rv"], weak["tp"], weak["ls"], weak["spa"], 855, 3415)

    def oracle(st_):
        for _, r in st_[(st_["kind"] == "main")].iterrows():
            h, b = r["hypothesis"].split("|")
            qh, qb = weak["ls"][h].mean(), weak["ls"][b].mean()
            assert abs(r["delta_pct"] - 100 * (qh - qb) / qb) < 1e-9
    oracle(weak["st"])
    assert caught(oracle, st)


def test_spa_blocks_a_b_c_follow_spec_and_feed_stats(weak):
    assert T.SPA_BLOCKS == {"a": ("S-RCK", ("S-RCKZ",)), "b": ("W5-G", ("S-RCKZ", "S-RZ")), "c": ("X1", ("X1_R(S-RCKZ)", "X1_R(S-RZ)"))}
    sp = T.spa_table(weak["spa"])
    fam = sp[sp["kind"] == "family"]
    assert list(fam["benchmark"]) == ["S-RCK", "W5-G", "X1"] and list(fam["candidates"]) == ["S-RCKZ", "S-RCKZ|S-RZ", "X1_R(S-RCKZ)|X1_R(S-RZ)"] and (fam["B"] == 10000).all()
    m = weak["st"][weak["st"]["kind"] == "main"].set_index("hypothesis")
    for h, k in (("S-RCKZ|S-RCK", "a"), ("S-RCKZ|W5-G", "b"), ("X1_R(S-RCKZ)|X1", "c")):
        assert m.loc[h, "p_spa_c"] == weak["spa"][k]["p_c"] and m.loc[h, "p_spa_block"] == k


def test_mcs_twelve_series_and_power_table_shape(weak):
    mc = T.mcs(weak["rv"], weak["F"])
    assert list(mc["series"]) == list(C.MCS_SET) and (mc["n_used"] == C.N_D).all() and mc["in_mcs"].dtype == bool
    pw = T.power_table(weak["tp"], weak["ls"])
    y = weak["tp"].year.to_numpy()
    assert len(pw) == 8 and set(pw["window"]) == {"D_M", "без 2022"} and sorted(set(pw["n"])) == [int((y != 2022).sum()), C.N_D]
    assert (pw[pw["window"] == "D_M"]["lag"] == 8).all()
