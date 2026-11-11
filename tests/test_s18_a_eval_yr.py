"""S-A-EVAL, годы / концентрация / защита / решение (`a_eval_yr`; S_A_SPEC разд. 6.2, 6.4, 6.5; mutations.md № 125, 140, 160): Σ вкладов лет = Δ (1e-12), выброшенная дата ломает тождество;
пороги У1-S…У3-S = 0,415 / 0,50 / 1,00 — константы файла, границы на выдуманных таблицах, мутации «порог из результата» и «пороги другого этапа»; М1 и М2 — строго < 0; У4-S строго лучше контролей (б) и (в); У5-S по периоду с наименьшим вкладом;
решение выведено из строк GUARD (в том числе после круга через CSV), порядок A2 → A1, а не по минимуму QLIKE; мощность — известный ответ. Только monkeypatch / копия исходника (№ 43); реальные файлы только читаются."""
from __future__ import annotations

import io
import math
import types
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import a_eval as E
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_tab as T
from models_v2.s18 import a_eval_yr as Y

H1, H2 = Y.H1, Y.H2


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        ps = [E.CKPT, E.R.TRIALS] + [E.REP / n for n in E.OUTS] + sorted(E.A_DIR.glob("*.parquet"))
        return {str(p): p.read_bytes() for p in ps if p.exists() and p.is_file()}
    before = st()
    yield
    assert st() == before


def mutant(old: str, new: str) -> types.ModuleType:
    src = Path(Y.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    m = types.ModuleType("a_eval_yr_mutant")
    m.__file__ = Y.__file__
    exec(compile(src.replace(old, new), Y.__file__, "exec"), m.__dict__)
    return m


def synth(n: int = 2198):
    rng = np.random.default_rng(4)
    tp = pd.bdate_range("2018-01-04", periods=n)
    Lv = np.exp(rng.normal(-8, 0.3, n))
    rv = Lv * np.exp(rng.normal(0, 0.45, n) + 0.1)

    def q(mult, s):
        return np.exp(np.random.default_rng(s).normal(0, 0.1, n)) * Lv * mult
    F = {"X1": q(1.45, 1), "W5-G": q(1.22, 2), "K2": q(1.7, 3), "K3": q(1.8, 4)}
    for i, c in enumerate(T.CANDS + ("S-A2p",)):
        F[c] = q(1.26 + 0.01 * i, 10 + i)
        F["X1_A(%s)" % c] = q(1.24 + 0.01 * i, 20 + i)
    return rv, tp, F


RV, TP, F = synth()
LS = T.losses(RV, F)


def test_years_identity_known_answer_and_dropped_date_mutation(monkeypatch) -> None:
    y = Y.years(TP, LS)
    assert Y.identity_gap(y) < 1e-12
    h, b = "X1_A(S-A2)", "X1"
    w = y[(y["series"] == h) & (y["base"] == b) & (y["kind"] == "period")].set_index("period")
    m20 = TP.year == 2020
    want = 100.0 * (LS[h] - LS[b])[m20].sum() / (len(RV) * LS[b].mean())                  # вклад года = 100 Σ(l_h − l_b) / (n · Q_b)
    assert w.at["2020", "contrib_pct"] == pytest.approx(want, rel=1e-9) and abs(want) > 0.1
    assert w["contrib_pct"].sum() == pytest.approx(w["delta_total_pct"].iloc[0], abs=1e-12) and w["delta_total_pct"].iloc[0] == pytest.approx(100 * (LS[h].mean() / LS[b].mean() - 1), rel=1e-12)
    assert set(y[y["base"] != "-"]["base"]) == set(T.BASES) and len({(s, bb) for s, bb in zip(y["series"], y["base"]) if bb != "-"}) == len(Y.COMPS)
    half = y[(y["series"] == h) & (y["base"] == b) & (y["kind"] == "half")].set_index("period")
    assert set(half.index) == {H1, H2} and half.at[H1, "n"] == int((TP.year <= 2021).sum()) and half.at[H2, "n"] == int((TP.year >= 2022).sum())
    assert half.at[H1, "delta_pct"] == pytest.approx(100 * (LS[h][TP.year <= 2021].mean() / LS[b][TP.year <= 2021].mean() - 1), rel=1e-12)      # Δ в h1 — от QLIKE базы в h1
    orig = Y.T5.masks

    def dropped(tp):                                                                      # мутация: одна дата выброшена из маски 2019
        mk = orig(tp)
        k = np.flatnonzero(mk["2019"])[3]
        mk["2019"] = mk["2019"].copy()
        mk["2019"][k] = False
        return mk
    monkeypatch.setattr(Y.T5, "masks", dropped)
    assert Y.identity_gap(Y.years(TP, LS)) > 1e-9


def test_conc_slices_known_answers() -> None:
    c = Y.conc(TP, LS)
    h, b = "S-A2", "W5-G"
    d, lb = LS[h] - LS[b], LS[b]
    g = c[(c["series"] == h) & (c["base"] == b)].set_index(["item", "key"])["value"]
    day22 = TP == pd.Timestamp("2022-02-24")
    assert day22.sum() == 1
    assert g[("no_day", "2022-02-24")] == pytest.approx(100 * d[~day22].sum() / lb[~day22].sum(), rel=1e-12)
    assert g[("no_year", "2022")] == pytest.approx(100 * d[TP.year != 2022].sum() / lb[TP.year != 2022].sum(), rel=1e-12)
    top = int(np.argmax(np.abs(d)))
    keep = np.arange(len(d)) != top
    assert g[("no_top_day", str(TP[top].date()))] == pytest.approx(100 * d[keep].sum() / lb[keep].sum(), rel=1e-12)
    e = (d - d.mean()) ** 2
    assert g[("var_share_top_day", str(TP[int(np.argmax(e))].date()))] == pytest.approx(e.max() / e.sum(), rel=1e-12) and g[("var_share_day", "2022-02-24")] == pytest.approx(e[day22].sum() / e.sum(), rel=1e-12)
    loyo = c[(c["series"] == h) & (c["base"] == b) & (c["item"] == "loyo")].set_index("key")["value"]
    m19 = np.asarray(TP.year == 2019)
    assert loyo["2019"] == pytest.approx(100 * d[~m19].sum() / lb[~m19].sum(), rel=1e-12)
    assert {(s, bb) for s, bb in zip(c["series"], c["base"])} == set(Y.COMPS)


def tables(over: dict | None = None):
    """Выдуманные таблицы для guard: все 10 условий каждого кандидата выполнены; over[(кандидат, поле)] меняет одно поле."""
    over = over or {}
    q = {"S-A2": 0.20, "S-A1": 0.205, "S-A0": 0.21, "S-A2p": 0.21} | {k[0]: v for k, v in over.items() if k[1] == "q"}
    yrs, cnc = [], []
    for c in Y.ORDER:
        for pid, b in Y.PAIRS:
            h = T.hkey(c, b)
            tot = over.get((c, "tot" + pid), -0.1)
            for p, k, dp, ct in (("2018", "period", over.get((c, "d18" + pid), 0.5), 0.1), ("2019", "period", -1.0, -0.3), (H1, "half", over.get((c, "h1" + pid), 0.1), np.nan), (H2, "half", over.get((c, "h2" + pid), 0.1), np.nan)):
                yrs.append(dict(series=h, base=b, kind=k, period=p, delta_pct=dp, contrib_pct=ct, delta_total_pct=tot))
        cnc += [dict(series=c, base="W5-G", item="loyo", key="2018", value=over.get((c, "loyo18"), 0.7)), dict(series=c, base="W5-G", item="loyo", key="2019", value=over.get((c, "loyo"), -0.05))]
    return pd.DataFrame(dict(series=list(q), qlike=list(q.values()))), pd.DataFrame(yrs), pd.DataFrame(cnc)


def passed(over=None, c="S-A2", i="У1-S", pair="М1") -> bool:
    g = Y.guard(*tables(over))
    r = g[(g["candidate"] == c) & (g["id"] == i) & (g["pair"] == pair)]
    assert len(r) == 1
    return bool(r["passed"].iloc[0])


def check_boundaries() -> None:
    for c in Y.ORDER:
        for pair in ("М1", "М2"):
            k = pair
            assert passed({(c, "h1" + k): 0.415}, c, "У1-S", pair) and not passed({(c, "h1" + k): 0.4151}, c, "У1-S", pair)          # на границе — проходит (≤)
            assert passed({(c, "h2" + k): 0.50}, c, "У2-S", pair) and not passed({(c, "h2" + k): 0.5001}, c, "У2-S", pair)
            assert passed({(c, "d18" + k): 1.00}, c, "У3-S", pair) and not passed({(c, "d18" + k): 1.0001}, c, "У3-S", pair)
            assert passed({(c, "d18" + k): 0.50}, c, pair, pair) and not passed({(c, "tot" + k): 0.0}, c, pair, pair) and not passed({(c, "tot" + k): 1e-9}, c, pair, pair) and passed({(c, "tot" + k): -1e-9}, c, pair, pair)      # М строго < 0
    assert passed({("S-A2", "d18М1"): 0.30}, "S-A2", "У3-S", "М1") and passed({("S-A2", "d18М1"): 0.5}, "S-A2", "У3-S", "М1")      # порог 1,00, не 0,26 (W5)


def test_thresholds_are_spec_constants_and_boundaries_known_answer() -> None:
    assert Y.THR == {"У1-S": 0.415, "У2-S": 0.50, "У3-S": 1.00} and Y.ORDER == ("S-A2", "S-A1")
    check_boundaries()
    g = Y.guard(*tables())
    assert set(g["id"]) == {"М1", "М2", "У1-S", "У2-S", "У3-S", "У4-S", "У5-S", "кандидат", "решение"} and len(g[g["candidate"] == "S-A2"]) == 11
    assert not passed({("S-A2", "h1М2"): 0.5}, "S-A2", "У1-S", "М2") and passed({("S-A2", "h1М2"): 0.5}, "S-A2", "У1-S", "М1")      # пары независимы: М2 не влияет на М1


def test_threshold_from_result_and_wrong_constants_mutations_are_caught(monkeypatch) -> None:
    monkeypatch.setattr(Y, "THR", {"У1-S": 0.26, "У2-S": 0.50, "У3-S": 0.26})            # мутация: пороги W5 вместо S (0,26 вместо 1,00)
    with pytest.raises(AssertionError):
        check_boundaries()
    monkeypatch.undo()
    monkeypatch.setattr(Y, "THR", {"У1-S": 0.60, "У2-S": 0.80, "У3-S": 1.50})            # пороги подогнаны под результат
    with pytest.raises(AssertionError):
        check_boundaries()
    monkeypatch.undo()
    for old, new in (("v <= THR[i]", "v <= v"), ("v <= THR[i]", "v <= abs(v) + 1.0")):    # мутация: порог вычислен из самого результата
        monkeypatch.setattr(Y, "guard", mutant(old, new).guard)
        with pytest.raises(AssertionError):
            check_boundaries()
        monkeypatch.undo()


def test_u4_u5_known_answers() -> None:
    assert passed({}, "S-A2", "У4-S", "—") and not passed({("S-A2", "q"): 0.21}, "S-A2", "У4-S", "—")                  # строго лучше A2p и A0: равенство не проходит
    assert not passed({("S-A0", "q"): 0.19}, "S-A2", "У4-S", "—") and not passed({("S-A2p", "q"): 0.19}, "S-A2", "У4-S", "—")      # контроль (в) A0 и (б) A2p — оба обязательны
    assert passed({}, "S-A1", "У4-S", "—") and not passed({("S-A0", "q"): 0.20}, "S-A1", "У4-S", "—")                 # для A1 — только против A0 (A2p вне условия)
    assert passed({("S-A2p", "q"): 0.19}, "S-A1", "У4-S", "—")
    assert passed({}, "S-A2", "У5-S", "М2") and passed({("S-A2", "loyo"): 0.0}, "S-A2", "У5-S", "М2") and not passed({("S-A2", "loyo"): 1e-9}, "S-A2", "У5-S", "М2")      # Δ без лучшего периода ≤ 0
    assert passed({("S-A2", "loyo18"): 5.0}, "S-A2", "У5-S", "М2")                       # берётся период с наименьшим вкладом (2019), а не любой


def row_ok(over) -> pd.DataFrame:
    return Y.guard(*tables(over))


def test_decision_order_a2_then_a1_and_derived_from_rows() -> None:
    g = row_ok({})
    d = Y.decide(g)
    assert d.startswith("новая контрольная версия: S-A2") and "не «подтверждено»" in d and "X1 и W5-G остаются контрольными версиями; W8-G — экспериментальная" in d
    assert g[g["id"] == "решение"]["value"].iloc[0] == d and bool(g[g["id"] == "решение"]["passed"].iloc[0])
    both = row_ok({("S-A1", "q"): 0.19})                                                 # A1 по QLIKE лучше A2 и тоже проходит: берётся A2 (порядок, не минимум QLIKE)
    assert Y.decide(both).startswith("новая контрольная версия: S-A2")
    a2_fails = row_ok({("S-A2", "d18М1"): 1.2})
    assert Y.decide(a2_fails).startswith("новая контрольная версия: S-A1") and bool(a2_fails[a2_fails["id"] == "решение"]["passed"].iloc[0])      # решение «новая версия» (A1) — passed = True
    bad = row_ok({("S-A2", "d18М1"): 1.2, ("S-A1", "h2М2"): 0.7})
    s = Y.decide(bad)
    assert s.startswith("не прошёл — ") and "S-A2: У3-S|М1 (+1.2000)" in s and "S-A1: У2-S|М2 (+0.7000)" in s and "У1-S" not in s and "W8-G — экспериментальная" in s
    flipped = bad.copy()
    flipped.loc[~flipped["passed"] & ~flipped["id"].isin(["кандидат", "решение"]), "passed"] = True            # подмена passed в таблице → решение меняется
    assert Y.decide(flipped).startswith("новая контрольная версия: S-A2")
    buf = io.StringIO()
    bad.to_csv(buf, index=False)
    back = pd.read_csv(io.StringIO(buf.getvalue()))
    assert Y.decide(back) == s                                                           # то же решение после круга через CSV (passed — не только bool)
    with pytest.raises(ValueError):
        Y.decide(bad[~((bad["candidate"] == "S-A1") & (bad["id"] == "У5-S"))])


def test_selection_by_minimal_qlike_or_reversed_order_mutations_are_caught(monkeypatch) -> None:
    both = row_ok({("S-A1", "q"): 0.19})
    assert Y.decide(both).startswith("новая контрольная версия: S-A2")
    monkeypatch.setattr(Y, "ORDER", ("S-A1", "S-A2"))                                    # мутация: перебор в порядке минимума QLIKE (здесь A1 первым)
    mut_g = Y.guard(*tables({("S-A1", "q"): 0.19}))
    assert not Y.decide(mut_g).startswith("новая контрольная версия: S-A2")
    monkeypatch.undo()
    m = mutant("        sub = g[(g[\"candidate\"] == c) & ~g[\"id\"].isin(SKIP)]", "        sub = g[(g[\"candidate\"] == c) & ~g[\"id\"].isin(SKIP)]\n        c = min(ORDER, key=lambda k: g[(g['candidate'] == k) & (g['id'] == 'У4-S')]['value'].iloc[0])")
    assert not m.decide(both).startswith("новая контрольная версия: S-A2")


def test_power_known_answer_and_wrong_sign_is_nan() -> None:
    rng = np.random.default_rng(2)
    d = rng.normal(-0.004, 0.05, 3000)
    p = S.power(d, 0.2, 8)
    z05, z08 = NormalDist().inv_cdf(0.95), NormalDist().inv_cdf(0.80)
    sig = S.sigma_lr(d, 8)
    assert p["effect_abs"] == pytest.approx(d.mean(), rel=1e-12) and p["effect_pct"] == pytest.approx(100 * d.mean() / 0.2, rel=1e-12)
    assert p["n_p05"] == pytest.approx((z05 * sig / abs(d.mean())) ** 2, rel=1e-9) and p["n_p08"] == pytest.approx(((z05 + z08) * sig / abs(d.mean())) ** 2, rel=1e-9) and p["n_p08"] > p["n_p05"]
    assert p["n_p08_hyp1pct"] == pytest.approx(((z05 + z08) * sig / (0.01 * 0.2)) ** 2, rel=1e-9) and sig == pytest.approx(0.05, rel=0.15)
    w = S.power(-d, 0.2, 8)
    assert math.isnan(w["n_p05"]) and math.isnan(w["n_p08"]) and not math.isnan(w["n_p08_hyp1pct"])      # эффект не в пользу кандидата — NaN, не «достаточно»
    t = Y.power_table(TP, LS)
    assert set(t["pair"]) == {"М1", "М2"} and set(t["window"]) == {"D_M", "без 2022"} and set(t["candidate"]) == set(T.CANDS) and len(t) == 12
    r = t[(t["candidate"] == "S-A2") & (t["pair"] == "М1") & (t["window"] == "без 2022")].iloc[0]
    assert r["series"] == "X1_A(S-A2)" and r["base"] == "X1" and r["n"] == int((TP.year != 2022).sum()) and r["lag"] == math.ceil(4 * (r["n"] / 100) ** (2 / 9))
