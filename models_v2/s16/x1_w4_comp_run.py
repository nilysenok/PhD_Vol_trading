"""W4-B2: запуск ступени 1, `--check`, `--check-decision`. Счёт: 2 набора × (основной + 19 сдвигов 252…2052; сдвиг 252 = K_b, j = 1) + K_a — 39 проходов, пул ≤ 4 процессов.
Выходы: `data/regen/harj_x1/W4/{mplus_S1,mplus_S2,ka,kb1,kb2}.parquet`, `W4/perm/S{1,2}_{jj}.parquet`, `docs/otchety/harj_x1/W4_STAGE1.csv`, `W4_STAGE1_REPORT.md`, ключ `W4.stage1`.
Существующие выходы не перезаписываются (SystemExit). `--check` только читает (sha256 всех выходов до и после)."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as W3
from models_v2.s16 import x1_w4_comp as C
from models_v2.s16 import x1_w4_comp_stat as ST
from models_v2.s16 import x1_w4_feat as FEAT
from models_v2.s16 import x1_w4_reg as REG

W4, PERM, REP = REG.W4, REG.W4 / "perm", REG.REP
CSV, MD, CKPT = REP / "W4_STAGE1.csv", REP / "W4_STAGE1_REPORT.md", REG.CKPT
IDS = ("W4-S1", "W4-S2", "W4-Ka", "W4-Kb1", "W4-Kb2")
MAIN = {("S1", 0): "mplus_S1", ("S2", 0): "mplus_S2", ("Ka", 0): "ka", ("S1", C.SHIFT): "kb1", ("S2", C.SHIFT): "kb2"}
PROC = 4


def require_all() -> None:
    for i in IDS:
        REG.require(i)


def path_of(spec: tuple, j: int | None = None):
    return PERM / ("%s_%02d.parquet" % (spec[0], j)) if j else W4 / (MAIN[spec] + ".parquet")


def specs() -> list:
    return [("S1", 0), ("S2", 0), ("Ka", 0)] + [(n, s) for s in C.PERM_SHIFTS for n in ("S1", "S2")]


def compute(sp: list) -> dict:
    """-> {spec: кадр окна}; пул из 4 процессов (spawn)."""
    with mp.get_context("spawn").Pool(PROC) as pool:
        return dict(pool.imap_unordered(C.job, sp))


def outputs() -> list:
    return [W4 / (v + ".parquet") for v in MAIN.values()] + [PERM / ("%s_%02d.parquet" % (n, j)) for n in ("S1", "S2") for j in range(1, 20)] + [CSV, MD, CKPT]


def sha(paths) -> dict:
    return FEAT.sha(paths)


def load_series() -> tuple:
    rd = lambda p: pd.read_parquet(p)  # noqa: E731
    ser = {"S1": rd(path_of(("S1", 0))), "S2": rd(path_of(("S2", 0))), "Ka": rd(path_of(("Ka", 0))), "Kb1": rd(path_of(("S1", C.SHIFT))), "Kb2": rd(path_of(("S2", C.SHIFT)))}
    ser["M"] = ser["S1"].assign(F=ser["S1"]["M"])
    perms = {n: {j: float(ST.W3.qlike(ser[n]["rv"], rd(path_of((n, 0), j))["F"]).mean()) for j in range(1, 20)} for n in ("S1", "S2")}
    return ser, perms


def stage_state() -> np.ndarray:
    x = W3.load().set_index("d")["s"]
    return x.reindex(pd.DatetimeIndex(pd.read_parquet(path_of(("S1", 0)))["d"])).to_numpy(int)


def build_outputs() -> tuple:
    ser, perms = load_series()
    df = ST.table(ser, perms, stage_state())
    dec = ST.decide(df)
    return df, dec, ST.render(df, dec)


def run() -> int:
    require_all()
    if any(p.exists() for p in outputs() if p != CKPT):
        raise SystemExit("выходы ступени 1 уже есть — перезапись запрещена")
    res = compute(specs())
    for sp, df in res.items():
        if sp[1] in C.PERM_SHIFTS:
            FEAT.atomic(df, path_of(sp, C.PERM_SHIFTS.index(sp[1]) + 1))
        if sp in MAIN:
            FEAT.atomic(df, path_of(sp))
    df, dec, md = build_outputs()
    text = ST.csv_text(df)
    CSV.write_text(text, encoding="utf-8")
    MD.write_text(md, encoding="utf-8")
    ST.write_stage1(CKPT, dec, hashlib.sha256(text.encode("utf-8")).hexdigest(), dict(n_window=C.N_WINDOW, registered=list(IDS), nofit_note="переоценки 2018-04-02, 2018-05-02: g = 0 (обучатель не строится)"))
    print("решение: прошли %s; набор ступени 2: %s; ступень 2: %s" % (dec["passed"] or "ни один", dec["chosen"], dec["stage2"]))
    return 0


def check_rho_zero(ser: dict) -> list:
    bad = []
    for k in ("S1", "S2"):
        s = ser[k]
        if not np.array_equal(C.mplus(s["M"], s["g"], np.zeros(len(s))), s["M"].to_numpy(float)):
            bad.append("%s: ρ ≡ 0 даёт M⁺ ≠ M побитно" % k)
    return bad


def check_dates(ser: dict) -> list:
    x = W3.load()
    bad = [] if len(x) == C.N_WINDOW else ["окно x1: %d дат" % len(x)]
    for k, s in ser.items():
        if len(s) != C.N_WINDOW or not (pd.DatetimeIndex(s["d"]) == pd.DatetimeIndex(x["d"])).all():
            bad.append("%s: даты окна ≠ 2198 дат x1" % k)
    if not np.allclose(ser["S1"]["M"], x["M"], rtol=1e-10, atol=0) or not np.allclose(ser["S1"]["rv"], x["rv"], rtol=1e-12, atol=0):
        bad.append("M или rv ≠ x1")
    return bad


def check_first_refit(ser: dict) -> list:
    tab = C.load_table()
    first = C.first_refit(tab)
    tp = tab["t_plus"].to_numpy()
    months = [tab.index[a] for a in C.WF.refit_positions(tab.index)]
    first_ind = next(m for m in months if int((tp <= np.datetime64(m)).sum()) >= C.FT.MIN_TRAIN)
    bad = [] if first == first_ind else ["first_refit %s ≠ независимый счёт %s" % (first, first_ind)]
    for k in ("S1", "S2", "Ka"):
        s = ser[k]
        pre = s["d"] < first
        if (s.loc[pre, "g"] != 0).any() or (s.loc[pre, "F"] != s.loc[pre, "M"]).any() or s.loc[pre, "refit_date"].notna().any():
            bad.append("%s: до первой переоценки g ≠ 0 или есть refit_date" % k)
        fr = s["refit_date"].dropna()
        if len(fr) and fr.min() < first:
            bad.append("%s: переоценка раньше %s" % (k, first))
    return bad


def check_vs_fit_gbm() -> list:
    """Выбранное число деревьев и финальная модель = `fit_gbm` побитно на одной переоценке."""
    tab, (S, _) = C.load_table(), C.feature_source()
    X, e = S.reindex(tab.index).to_numpy(float), np.log(tab["rv"].to_numpy(float) / tab["M"].to_numpy(float))
    tr = C.closed(tab["t_plus"].to_numpy(), tab.index[len(tab) // 2])
    k, _, pred = C.gbm_learner(X[tr], e[tr])
    m = C.FT.fit_gbm(X[tr], e[tr])
    return [] if k == m.n_trees and np.array_equal(pred(X[tr]), m.g(X[tr])) else ["обучатель ≠ fit_gbm"]


def check() -> int:
    require_all()
    before = sha(outputs())
    if None in before.values():
        raise SystemExit("нет выходов: %s" % [k for k, v in before.items() if v is None])
    ser, _ = load_series()
    bad = check_rho_zero(ser) + check_dates(ser) + check_first_refit(ser) + check_vs_fit_gbm()
    res = compute(specs())
    for sp, df in res.items():
        p = path_of(sp, C.PERM_SHIFTS.index(sp[1]) + 1) if sp[1] in C.PERM_SHIFTS else path_of(sp)
        if not df.equals(pd.read_parquet(p)):
            bad.append("пересчёт ≠ файл: %s" % p.name)
        if sp in MAIN and not df.equals(pd.read_parquet(path_of(sp))):
            bad.append("пересчёт ≠ файл: %s" % path_of(sp).name)
    df, dec, md = build_outputs()
    bad += ([] if ST.csv_text(df) == CSV.read_text(encoding="utf-8") else ["CSV ≠ пересчёт из рядов"]) + ([] if md == MD.read_text(encoding="utf-8") else ["отчёт ≠ перерисовка из CSV"])
    bad += check_decision_keys()
    ok = sha(outputs()) == before
    print("check %s; пересчитано проходов %d; хэши выходов до = после %s" % ("ok" if not bad and ok else "FAIL " + "; ".join(bad), len(res), ok))
    return 0 if not bad and ok else 1


def check_decision_keys() -> list:
    import json
    df = pd.read_csv(CSV, float_precision="round_trip")
    dec, key = ST.decide(df), json.loads(CKPT.read_text(encoding="utf-8")).get("W4.stage1")
    if key is None:
        return ["нет ключа W4.stage1"]
    bad = [] if json.loads(json.dumps(dec)) == key["decision"] else ["решение ≠ вывод из CSV"]
    return bad + ([] if key["csv_sha256"] == hashlib.sha256(CSV.read_bytes()).hexdigest() else ["sha256 CSV ≠ ключ"])


def main(argv: list) -> int:
    if "--check" in argv:
        return check()
    if "--check-decision" in argv:
        bad = check_decision_keys()
        d = ST.decide(pd.read_csv(CSV, float_precision="round_trip"))
        print("check-decision %s; прошли %s; набор ступени 2 %s; ступень 2: %s" % ("ok" if not bad else "FAIL " + "; ".join(bad), d["passed"] or "ни один", d["chosen"], d["stage2"]))
        return 1 if bad else 0
    return run()
