"""W8-B, оценка на синтетике, чистые функции `x1_w8_eval_t` (W8_SPEC 5, 6, 9; mutations.md № 103, 125, 140, 158): Σ вкладов лет = Δ (1e-12), выброшенная дата / тип ломают тождество; Холм N = 814 / N_wide = 3333
(независимая реализация), мутация «N = числу строк»; знак Δ, база процентов, «лучше/хуже»; «подтверждено» — по своей строке и только при p Холма < 0,05; пороги У1–У6 равны константам спецификации
(0,415 / 0,50 / 0,26 / 0,10), границы проверяются на выдуманных таблицах, мутация «порог из результата» ловится; решение выводится из таблицы GUARD; p по сдвигам (1 + #{Q_j ≤ Q}) / 20.
Только monkeypatch / копия исходника (№ 43). Реальные файлы только читаются (autouse-хэши, сравнение содержимого до и после)."""
from __future__ import annotations

import hashlib
import io
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_q as Q
from models_v2.s13 import v6_1_eval as EVX
from models_v2.s16 import x1_w1_stat as SS
from models_v2.s16 import x1_w8_eval as E
from models_v2.s16 import x1_w8_eval_t as T


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (E.REG.TRIALS, E.CKPT) if p.exists()}
        s.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.REP.glob("W8_*")) if p.is_file()})
        s["W8"] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.OUT.rglob("*")) if p.is_file()} if E.OUT.exists() else {}
        return s
    before = st()
    yield
    assert st() == before


def mutant(old: str, new: str) -> types.ModuleType:
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    m = types.ModuleType("x1_w8_eval_t_mutant")
    m.__file__ = T.__file__
    exec(compile(src.replace(old, new), T.__file__, "exec"), m.__dict__)
    return m


def synth(n: int = 2300) -> tuple:
    rng = np.random.default_rng(5)
    tp = pd.bdate_range("2018-01-04", periods=n)
    L = np.exp(rng.normal(-8, 0.4, n))
    rv = L * np.exp(rng.normal(0.0, 0.5, n) + 0.125)
    r2 = np.random.default_rng(2)                                                       # W8-G и W5-G — независимый шум вокруг одного уровня: строка к W5-G сырым p значима, Холмом — нет
    F5, F8 = L * np.exp(0.10 + 0.2 * r2.normal(size=n)), L * np.exp(0.10 + 0.2 * r2.normal(size=n))
    return rv, tp, {"X1": L, "K1": L * 1.5, "K2": L * 1.9, "K3": L * 2.1, "M": L * 1.8, "ZB": L, "W8-Gb": L * np.exp(0.005), "W5-G": F5, "W8-G": F8}


RV, TP, F = synth()


def ql(rv, f):
    r = rv / f
    return r - np.log(r) - 1.0


def holm(p: np.ndarray, n: int) -> np.ndarray:
    o = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate(p[o] * (n - np.arange(len(p)))))
    out = np.empty(len(p))
    out[o] = adj
    return out


def check_stats() -> None:
    s = T.stats(RV, TP, F, [0.2] * 19)
    m = s[s["kind"] == "main"]
    assert m["base"].tolist() == ["W5-G", "X1", "K2", "K3"] and set(m["hybrid"]) == {"W8-G"}      # порядок показа: W5-G, X1, K2, K3
    for r in m.itertuples():                                                          # знак и база: Δ = 100 (Q_W8 / Q_base − 1); минус — W8-G лучше
        assert r.delta_pct == pytest.approx(100.0 * (ql(RV, F["W8-G"]).mean() / ql(RV, F[r.base]).mean() - 1.0), rel=1e-12)
    assert (m["n_holm"] == 814).all() and (m["n_holm_wide"] == 3333).all()
    assert m["p_holm_N"].to_numpy() == pytest.approx(holm(m["p_one"].to_numpy(), 814), rel=1e-9)
    assert m["p_holm_Nwide"].to_numpy() == pytest.approx(holm(m["p_one"].to_numpy(), 3333), rel=1e-9)
    for r in m.itertuples():                                                          # № 158: «подтверждено» только при Δ < 0 и p Холма < 0,05
        assert ("подтверждено (p Холма < 0,05)" in r.verdict and "не подтверждено" not in r.verdict) == (r.delta_pct < 0 and r.p_holm_N < 0.05)
        assert ("численно лучше базы" in r.verdict) == (r.delta_pct < 0)              # № 125/140: слово «лучше» ↔ знак Δ
    ref = s[s["kind"] == "reference"]
    assert ref["base"].tolist() == ["W8-Gb", "M", "ZB"] and set(ref["hybrid"]) == {"W8-G"} and ref["p_holm_N"].isna().all()
    assert (ref["verdict"] == "справочно, критериев не получает").all()               # W8-Gb, M, ZB справочно


def test_stats_known_answer_and_precondition_is_not_vacuous() -> None:
    check_stats()
    m = T.stats(RV, TP, F, [0.2] * 19).query("kind == 'main'")
    w5 = m[m["base"] == "W5-G"].iloc[0]
    assert w5["delta_pct"] < 0 and w5["p_one"] < 0.05 <= w5["p_holm_N"]               # сырой p значим, Холм — нет: «подтверждено» тут быть не должно
    assert "не подтверждено (p Холма ≥ 0,05)" in w5["verdict"]
    assert "подтверждено (p Холма < 0,05)" in m[m["base"] == "K3"].iloc[0]["verdict"] and "не подтверждено" not in m[m["base"] == "K3"].iloc[0]["verdict"]
    assert T.N_HOLM == 814 and T.N_WIDE == 3333                                        # 798 + 4 + 10 + 2; 3301 + 8 + 24


def test_holm_n_equals_row_count_mutation_is_caught(monkeypatch) -> None:
    monkeypatch.setattr(T, "N_HOLM", 4)                                               # мутация № 103: N = числу строк
    with pytest.raises(AssertionError):
        check_stats()
    monkeypatch.setattr(T, "N_HOLM", 814)
    monkeypatch.setattr(T, "N_WIDE", 12)
    with pytest.raises(AssertionError):
        check_stats()


def test_sign_base_wording_and_confirmed_mutations_are_caught(monkeypatch) -> None:
    orig_ci, orig_v = EVX.delta_ci, SS.verdict
    monkeypatch.setattr(EVX, "delta_ci", lambda x, y, base, **k: {**orig_ci(x, y, base, **k), "delta_pct": -orig_ci(x, y, base, **k)["delta_pct"]})          # знак Δ (№ 125)
    with pytest.raises(AssertionError):
        check_stats()
    monkeypatch.setattr(EVX, "delta_ci", lambda x, y, base, **k: orig_ci(x, y, float(x.mean()), **k))           # база процентов — QLIKE W8-G, а не базы
    with pytest.raises(AssertionError):
        check_stats()
    monkeypatch.setattr(EVX, "delta_ci", orig_ci)
    monkeypatch.setattr(SS, "verdict", lambda d, p: orig_v(-d, p))                    # «лучше/хуже» перевёрнуты
    with pytest.raises(AssertionError):
        check_stats()
    monkeypatch.setattr(SS, "verdict", lambda d, p: orig_v(d, 0.0))                   # «подтверждено» при любом p
    with pytest.raises(AssertionError):
        check_stats()
    monkeypatch.setattr(SS, "verdict", orig_v)
    monkeypatch.setattr(EVX, "holm_both", lambda p, cp0: (np.asarray(p), np.asarray(p)))                       # Холм подменён сырым p
    with pytest.raises(AssertionError):
        check_stats()


def test_years_identity_known_answer_and_dropped_date_mutation(monkeypatch) -> None:
    F2 = dict(F, **{"W8-G": F["X1"] * np.where(TP.year == 2020, np.exp(-0.05), 1.0)})
    y = T.years(RV, TP, F2)
    assert T.identity_gap(y) < 1e-12 and set(y[y["base"] != "-"]["base"]) == {"W5-G", "X1", "K2", "K3"}
    w = y[(y["series"] == "W8-G") & (y["base"] == "X1") & (y["kind"] == "period")].set_index("period")
    lg, lx = ql(RV, F2["W8-G"]), ql(RV, F2["X1"])
    want = 100.0 * (lg - lx)[TP.year == 2020].sum() / (len(RV) * lx.mean())            # вклад года = 100 Σ(l_W8 − l_X1) / (n · Q_X1)
    assert w.at["2020", "contrib_pct"] == pytest.approx(want, rel=1e-9) and abs(want) > 0.1
    assert w.drop(index=["2020"])["contrib_pct"].abs().max() < 1e-12 and w["contrib_pct"].sum() == pytest.approx(w["delta_total_pct"].iloc[0], abs=1e-12)
    orig = T.masks

    def dropped(tp):                                                                  # мутация: одна дата выброшена из маски года
        mk = orig(tp)
        k = np.flatnonzero(mk["2019"])[3]
        mk["2019"] = mk["2019"].copy()
        mk["2019"][k] = False
        return mk
    monkeypatch.setattr(T, "masks", dropped)
    F3 = dict(F, **{"W8-G": F["X1"] * np.exp(np.random.default_rng(1).normal(0, 0.05, len(RV)))})
    assert T.identity_gap(T.years(RV, TP, F3)) > 1e-9


def tables(d1=0.0, d2=0.0, d18=0.0, qg=0.20, qb=0.21, d_all=-0.1, dx=-0.2, loyo=-0.05) -> tuple:
    tbl = pd.DataFrame(dict(series=["W8-G", "W8-Gb"], qlike=[qg, qb]))
    rows = [("2018", "period", d18, 0.1), ("2019", "period", -1.0, -0.3), (T.H1, "half", d1, np.nan), (T.H2, "half", d2, np.nan)]
    yrs = pd.DataFrame([dict(series="W8-G", base="W5-G", kind=k, period=p, delta_pct=dp, contrib_pct=c, delta_total_pct=d_all) for p, k, dp, c in rows]
                       + [dict(series="W8-G", base="X1", kind="period", period="2018", delta_pct=0.0, contrib_pct=0.0, delta_total_pct=dx)])
    cnc = pd.DataFrame([dict(base="W5-G", item="loyo", key="2019", value=loyo)])
    return tbl, yrs, cnc


def passed(pp=0.05, **kw) -> dict:
    g = T.guard(*tables(**kw), pp)
    return dict(zip(g["id"], g["passed"]))


def check_boundaries() -> None:
    assert all(passed(d1=0.415, d2=0.50, d18=0.26, pp=0.10).values())                 # на границе — проходит (≤)
    assert passed(d1=0.4151)["У1"] is False and passed(d2=0.5001)["У2"] is False and passed(d18=0.2601)["У3"] is False and passed(pp=0.1001)["У6"] is False
    assert passed(d1=0.4149)["У1"] and passed(d2=0.4999)["У2"] and passed(d18=0.2599)["У3"] and passed(pp=0.0999)["У6"]
    assert passed(d1=0.30)["У1"] and passed(d18=0.30)["У3"] is False                 # 0,30 между 0,26 и 0,415
    assert passed(d_all=0.0)["основной"] is False and passed(dx=0.0)["основной"] is False and passed(d_all=-1e-9, dx=-1e-9)["основной"]      # обе строго меньше нуля
    assert passed(qg=0.21)["У4"] is False and passed(qb=0.19)["У4"] is False         # строго лучше сдвинутого ряда
    assert passed(loyo=0.0)["У5"] and passed(loyo=1e-9)["У5"] is False                # Δ без лучшего периода ≤ 0


def test_guard_thresholds_are_spec_constants_and_boundaries_known_answer() -> None:
    assert T.THR == {"У1": 0.415, "У2": 0.50, "У3": 0.26, "У6": 0.10}                 # разд. 5: константы спецификации, не из данных
    assert "0,2525" in T.FORMULA["У1"] and "0,1567" in T.FORMULA["У3"] and "0,50" in T.FORMULA["У2"] and "0,10" in T.FORMULA["У6"]
    check_boundaries()
    assert set(T.guard(*tables(), 0.05)["id"]) == {"основной", "У1", "У2", "У3", "У4", "У5", "У6"}


def test_threshold_computed_from_result_mutation_is_caught(monkeypatch) -> None:
    monkeypatch.setattr(T, "THR", {"У1": 0.60, "У2": 0.80, "У3": 0.40, "У6": 0.20})   # мутация: пороги подогнаны под результат
    with pytest.raises(AssertionError):
        check_boundaries()
    for old, new in (("v <= THR[i]", "v <= v + 0.0"), ("pp <= THR[\"У6\"]", "pp <= pp")):          # пороги вычислены из самого результата
        with pytest.raises(AssertionError):
            monkeypatch.setattr(T, "guard", mutant(old, new).guard)
            check_boundaries()


def test_decision_is_derived_from_guard_table_roundtrip_and_names_failed_condition() -> None:
    ok = T.guard(*tables(), 0.05)
    assert T.decide(ok).startswith("третья контрольная версия") and "не «подтверждено»" in T.decide(ok) and "X1 и W5-G остаются контрольными версиями" in T.decide(ok)
    bad = T.guard(*tables(d18=0.30), 0.12)
    s = T.decide(bad)
    assert s.startswith("W8-G не прошёл: ") and "У3 (+0.3000)" in s and "У6 (+0.1200)" in s and "У1" not in s and "X1 и W5-G остаются контрольными версиями" in s
    flipped = bad.copy()
    flipped.loc[flipped["id"].isin(["У3", "У6"]), "passed"] = True                    # подмена в таблице → решение меняется
    assert T.decide(flipped).startswith("третья контрольная версия")
    buf = io.StringIO()
    pd.concat([bad, pd.DataFrame([dict(id="решение", condition="x", value=s, threshold="—", formula="f", passed=False)])]).to_csv(buf, index=False)
    back = pd.read_csv(io.StringIO(buf.getvalue()))
    assert T.decide(back[back["id"] != "решение"]) == s                               # то же решение после круга через CSV (passed — bool, не строка)


def test_perm_p_known_answer_ties_and_direction() -> None:
    assert T.perm_p(0.25, [0.1, 0.2, 0.3]) == (1 + 2) / 4 and T.perm_p(0.05, [0.1] * 19) == 1 / 20 and T.perm_p(0.5, [0.1] * 19) == 1.0
    assert T.perm_p(0.2, [0.2] * 19) == 1.0                                           # ≤: равный QLIKE сдвига считается
    assert T.perm_p(0.2, [0.2 + 1e-12] * 19) == 1 / 20
    q = float(ql(RV, F["W8-G"]).mean())
    s = T.stats(RV, TP, F, [q + 0.01] * 18 + [q - 0.01])                              # один сдвиг лучше (QLIKE ниже) W8-G, 18 хуже
    assert s[s["kind"] == "perm_p"]["p_one"].iloc[0] == (1 + 1) / 20 and (s["kind"] == "perm").sum() == 19      # (1 + число сдвигов с QLIKE_j ≤ QLIKE(W8-G)) / 20
    assert s[s["kind"] == "perm"]["hybrid"].iloc[0] == "сдвиг 252" and s[s["kind"] == "perm"]["hybrid"].iloc[18] == "сдвиг 2052"      # 252 + 100·(j − 1)
    assert T.stats(RV, TP, F, [q - 0.01] * 19)[lambda t: t["kind"] == "perm_p"]["p_one"].iloc[0] == 1.0        # все сдвиги лучше → p = 1 (признак не привязан ко времени)
