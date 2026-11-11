"""S-Z-EVAL (Ж-258, ZE.G3), тестировщик: `z_eval_par.mech` / `share` (разд. 7.2: доля выигрыша, которую даёт контроль; разложение Δ) и `z_eval_null` (нулевое распределение процедуры; `--check` только читает, № 123).
Синтетика n = 300; `S_Z_NULL.csv` в тестах — tmp-копия (боевой читается только в одном тесте, без записи)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import test_s18_z_eval_fx as FX
from models_v2.s18 import z_eval_null as ZN
from models_v2.s18 import z_eval_par as PAR
from test_s18_z_eval_tab import caught


def null_rows(vals=None, share=0.5) -> pd.DataFrame:
    v = np.arange(1.0, 11.0) if vals is None else np.asarray(vals, float)
    return pd.DataFrame([dict(kind="seed", seed=s, n_active=75, qlike_rel_pct=float(x), share_theta_nonzero=share) for s, x in zip(ZN.SEEDS, v)])


def test_null_summary_known_answer_ddof1_and_linear_quantiles():
    s = ZN.summary(null_rows()).set_index(["item", "stat"])["value"]
    assert abs(s[("qlike_rel_pct", "mean")] - 5.5) < 1e-12 and abs(s[("qlike_rel_pct", "std")] - 3.0276503540974917) < 1e-12      # ddof = 1
    assert abs(s[("qlike_rel_pct", "q05")] - 1.45) < 1e-12 and abs(s[("qlike_rel_pct", "q95")] - 9.55) < 1e-12 and s[("share_theta_nonzero", "std")] == 0.0
    assert ZN.SEEDS == tuple(range(10))


def test_null_rule_boundaries_inside_and_sign():
    assert ZN.rule(-0.3, -0.3, 0.4) == "не отличим от шума процедуры" and ZN.rule(0.4, -0.3, 0.4) == "не отличим от шума процедуры"
    assert ZN.rule(0.4001, -0.3, 0.4) == "вне 5–95 % нулевого распределения" and ZN.rule(-0.3001, -0.3, 0.4) == "вне 5–95 % нулевого распределения"
    assert ZN.rule(-0.05, -0.3, 0.4) == "не отличим от шума процедуры"                                                          # шум несимметричен — знак важен


def test_null_abs_bounds_and_rule_abs_known_answers():
    """Правило по |Δ| (журнал 01:34, версия кода 01:57): границы — квантили 5 и 95 % распределения |Δ_null|; знак Δ не важен."""
    lo, hi = ZN.abs_bounds(null_rows(np.arange(-5.0, 5.0)))                       # |x| = 5,4,3,2,1,0,1,2,3,4
    assert abs(lo - 0.45) < 1e-12 and abs(hi - 4.55) < 1e-12
    for d in (0.4501, -0.4501, 4.5499, -4.5499, 2.0, -2.0):
        assert ZN.rule_abs(d, lo, hi) == "не отличим от шума процедуры"
    assert ZN.rule_abs(lo, lo, hi) == ZN.rule_abs(hi, lo, hi) == "не отличим от шума процедуры"                          # границы входят
    for d in (0.44, -0.44, 4.56, -4.56, 10.0, -10.0):
        assert ZN.rule_abs(d, lo, hi) == "вне 5–95 % распределения |Δ_null|"


def setup_null(tmp_path, monkeypatch, rows=None):
    p = tmp_path / "S_Z_NULL.csv"
    monkeypatch.setattr(ZN, "OUT", p)
    monkeypatch.setattr(ZN.read, "__defaults__", (p,))                  # `read(p=OUT)` связывает путь при определении — без этого читается боевой файл
    p.write_text(ZN.texts(null_rows() if rows is None else rows), encoding="utf-8")
    return p


def test_null_check_good_mutations_and_read_only(tmp_path, monkeypatch):
    p = setup_null(tmp_path, monkeypatch)
    h0 = hashlib.sha256(p.read_bytes()).hexdigest()
    assert ZN.check() == [] and hashlib.sha256(p.read_bytes()).hexdigest() == h0
    full = pd.concat([null_rows(), ZN.summary(null_rows())], ignore_index=True)
    sub = full.copy()
    sub.loc[(sub["kind"] == "summary") & (sub["item"] == "qlike_rel_pct") & (sub["stat"] == "mean"), "value"] = 5.6          # сводка подменена
    p.write_text(ZN.HEAD.replace("%%", "%") + sub.to_csv(index=False, lineterminator="\n", float_format="%.12g"), encoding="utf-8")
    assert ZN.check() == ["сводка ≠ пересчёту из строк seeds"]
    p.write_text(ZN.texts(null_rows().iloc[:9]), encoding="utf-8")                                                           # 9 шумов вместо 10
    assert ZN.check() == ["seeds ≠ 0…9"] or "seeds ≠ 0…9" in ZN.check()
    std0 = null_rows()
    bad = pd.concat([std0, ZN.summary(std0).assign(value=lambda d: np.where(d["stat"] == "std", np.std(np.arange(1.0, 11.0), ddof=0), d["value"]))], ignore_index=True)
    p.write_text(ZN.HEAD.replace("%%", "%") + bad.to_csv(index=False, lineterminator="\n", float_format="%.12g"), encoding="utf-8")   # СКО с ddof = 0 в сводке
    assert "сводка ≠ пересчёту из строк seeds" in ZN.check()
    p.unlink()
    assert ZN.check() == ["нет S_Z_NULL.csv"]


def test_real_null_file_check_is_clean_and_unchanged():
    p = Path(ZN.REP) / "S_Z_NULL.csv"
    h0 = p.read_bytes()
    assert ZN.check() == [] and p.read_bytes() == h0


# ───────── разд. 7.2: доля контроля и разложение ─────────
def test_share_known_answers_and_mutation_swapped_ratio_caught():
    def oracle(fn):
        assert abs(fn(-0.2, -0.12) - 0.6) < 1e-12 and abs(fn(-0.4, -0.1) - 0.25) < 1e-12
        assert np.isnan(fn(0.1, -0.05)) and np.isnan(fn(0.0, -0.05))                                                         # выигрыша кандидата нет — доли нет
    oracle(PAR.share)
    assert caught(oracle, lambda a, b: a / b if b < 0 else float("nan"))
    assert PAR.SHARE_HALF == 0.5


def frames(n, act_n=100):
    rv, tp, F = FX.world(n=n, gain={"S-RCKZ": 0.10, "S-RCKZs": 0.07})
    ls = FX.losses(rv, F)
    act = np.zeros(n, bool)
    act[:act_n] = True
    rng = np.random.default_rng(1)

    def fr(name):
        return pd.DataFrame(dict(active=act, F_fr=F[name] * np.exp(0.01 * rng.normal(size=n)), C=F[name] * np.exp(0.02 * rng.normal(size=n))))
    fm = {"S-RCKZ": fr("S-RCKZ"), "S-RCK": fr("S-RCK"), "S-RCKZs": fr("S-RCKZs")}
    return dict(rv=rv, F=F, frames=fm), ls, act


class Mech:
    """Доступ по (секция, пункт) к строке `mech`: .v — значение, .k — ключ (дубликатов пары нет — проверяется)."""

    def __init__(self, df):
        self.df = df

    def row(self, sec, item):
        r = self.df[(self.df["section"] == sec) & (self.df["item"] == item)]
        assert len(r) == 1, (sec, item)
        return r.iloc[0]

    def v(self, sec, item):
        return float(self.row(sec, item)["value"])

    def k(self, sec, item):
        return self.row(sec, item)["key"]


def mech_of(inp, ls):
    s = null_rows(np.linspace(-0.3, 0.4, 10), share=0.6)
    return Mech(PAR.mech(inp, ls, (s, ZN.summary(s))))


def test_mech_share_decomposition_identity_and_text():
    inp, ls, act = frames(300)
    m = mech_of(inp, ls)
    dp = lambda a, b: 100 * (ls[a].sum() - ls[b].sum()) / ls[b].sum()
    d_z, d_zs = dp("S-RCKZ", "S-RCK"), dp("S-RCKZs", "S-RCK")
    assert abs(m.v("hz2", "delta_Z_vs_RCK_pct") - d_z) < 1e-9 and abs(m.v("hz2", "delta_Zs_vs_RCK_pct") - d_zs) < 1e-9
    assert abs(m.v("hz2", "delta_Z_vs_Zs_pct") - dp("S-RCKZ", "S-RCKZs")) < 1e-9
    sh = m.v("hz2", "share_control")
    assert abs(sh - d_zs / d_z) < 1e-12 and sh >= 0.5 and m.k("hz2", "share_control") == "выигрыш в основном не связан с датой z"
    tot, fr, lay = (m.v("decomp", i) for i in ("total_vs_RCK", "frozen_pi", "layer_reaction"))
    assert abs(tot - d_z) < 1e-12 and abs((fr + lay) - tot) < 1e-12                                                         # Δ = эффект при замороженном π + реакция слоя
    froz = 100 * (FX.ql(inp["rv"], inp["frames"]["S-RCKZ"]["F_fr"]).sum() - FX.ql(inp["rv"], inp["frames"]["S-RCK"]["F_fr"]).sum()) / ls["S-RCK"].sum()
    assert abs(fr - froz) < 1e-9
    assert m.v("lnratio", "n_active_dates") == 100
    s = null_rows(np.linspace(-0.3, 0.4, 10), share=0.6)
    lo, hi = ZN.abs_bounds(s)
    for k, d in (("Z_vs_RCK|D_M", d_z), ("Z_vs_RCK|active", None), ("Z_vs_Zs|D_M", dp("S-RCKZ", "S-RCKZs")), ("Z_vs_Zs|active", None)):
        assert m.k("noise_rule", k) in ("не отличим от шума процедуры", "вне 5–95 % распределения |Δ_null|")
        if d is not None:
            assert m.k("noise_rule", k) == ZN.rule_abs(d, lo, hi) and abs(m.v("noise_real", k) - d) < 1e-9


def test_mech_share_below_half_text_and_no_gain_text():
    inp, ls, act = frames(300)
    inp["F"]["S-RCKZs"] = inp["F"]["S-RCK"] * np.exp(0.0)
    inp2, ls2, _ = frames(300)
    ls2["S-RCKZs"] = ls2["S-RCK"] * 0.999 + ls2["S-RCKZ"] * 0.001                                                           # контроль почти не даёт выигрыша
    m = mech_of(inp2, ls2)
    assert m.v("hz2", "share_control") < 0.5 and m.k("hz2", "share_control") == "доля контроля < 0,5"
    ls3 = dict(ls2)
    ls3["S-RCKZ"] = ls2["S-RCK"] * 1.01                                                                                     # кандидат хуже S-RCK: доли нет
    m3 = mech_of(inp2, ls3)
    assert np.isnan(m3.v("hz2", "share_control")) and m3.k("hz2", "share_control").startswith("доли нет")
