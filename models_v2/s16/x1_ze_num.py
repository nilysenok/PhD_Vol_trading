"""Лист ZE: числа для ключей CHECKPOINT (`ZE.eval` — без p; `ZE.rules` — p и вывод по правилу 6.2) и хэш чужих выходов (до/после `--check`). Чистые функции."""
from __future__ import annotations

import hashlib
import json

from models_v2.s16 import x1_ya_stage as SA
from models_v2.s16.x1_ze_inp import REP, ROOT, sha_file


def eval_numbers(t: dict) -> dict:
    """QLIKE по окнам, Δ с интервалами, годы, состояния, диагностика, K1–K6 — БЕЗ p-значений."""
    m, pr, st, dg = t["METRICS"], t["PAIRS"], t["STATES"], t["DIAG"]
    ql = {s: {w: float(g.loc[g["window"] == w, "qlike"].iloc[0]) for w in g["window"]} for s, g in m.groupby("series")}
    pair = {"%s|%s|%s" % (r.y, r.base, r.window): [float(r.delta_pct), float(r.lo90), float(r.hi90), float(r.delta_q), int(r.years_neg), int(r.n_years)] for r in pr.itertuples()}
    states = {"%s|%s|%s|%s" % (r.y, r.state, r.window, r.base): [float(r.delta_pct), float(r.lo90), float(r.hi90), int(r.n), float(r.mean_g)] for r in st.itertuples()}
    diag = {"%s|%s|%s" % (r.y, r.base, r.window): [float(r.y_qlike), float(r.base_qlike), float(r.delta_pct), float(r.lo90), float(r.hi90), int(r.n)] for r in dg.itertuples()}
    crit = {"%s|%s" % (r.y, r.crit): [r.name, float(r.value), bool(r.passed)] for r in t["CRIT"].itertuples()}
    yrs = {"%s|%s|%d" % (r.y, r.base, r.year): float(r.delta_pct) for r in t["YEARS"].itertuples()}
    return dict(qlike=ql, pairs=pair, states=states, diag=diag, crit=crit, years=yrs, rule62={r.y: r.rule62 for r in t["RULE62"].itertuples()}, p_values="не входят: записаны в ZE.rules после этого ключа")


def rules(t: dict) -> dict:
    return dict(p={"%s|%s|%s" % (r.y, r.base, r.window): dict(p_one=float(r.p_one), p_holm_N=float(r.p_holm_N), p_holm_Nwide=float(r.p_holm_Nwide)) for r in t["PAIRS"].itertuples()},
                rule62={r.y: dict(rule62=r.rule62, p_dm_one=float(r.p_dm_one), p_holm_N=float(r.p_holm_N), confirmed=bool(r.rule62 == "да" and r.p_holm_N < 0.05)) for r in t["RULE62"].itertuples()})


def rep_digest() -> str:
    """Хэш docs/otchety/harj_x1 без файлов ZE_* и ключей ZE* CHECKPOINT (их пишет этот лист); ALL_TRIALS.csv и чужие ключи входят."""
    h = hashlib.sha256()
    for f in sorted(p for p in REP.rglob("*") if p.is_file() and not p.name.startswith("ZE_")):
        if f.name == "CHECKPOINT.json":
            cp = {k: v for k, v in json.loads(f.read_text(encoding="utf-8")).items() if not k.startswith("ZE")}
            h.update(str(f).encode() + json.dumps(cp, sort_keys=True).encode())
        else:
            h.update(str(f).encode() + sha_file(f).encode())
    return h.hexdigest()


def outputs_hash() -> str:
    """Чужие выходы: V6/V7/DE (WATCH), data/regen/harj_x1 без ZE/ (YA, YB, Z), docs/otchety/harj_x1 (P0, T0, YA, YB, Z, ZD, Z2). Каталог data/fwd не входит."""
    base = ROOT / "data/regen/harj_x1"
    return SA.digest(SA.WATCH + sorted(p for p in base.iterdir() if p.name != "ZE")) + rep_digest()
