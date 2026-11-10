"""Y-A, стадия: compute() (всё в памяти), write() (ПИШЕТ y1/y2.parquet, CHECKPOINT `YA` → `YA.eval` (БЕЗ p) → таблицы YA_*.csv с p → отчёт → `YA.rules`), check() (ТОЛЬКО ЧИТАЕТ: пересчёт, сверка, хэш выходов до и после).
Входы (X1, K1–K3, NTP2, A2, cfg6, cfg8, склад M10 V7) только читаются; файлы контролей сверяются по sha256 (загрузчик T0). Расходуется 2 конфигурации (Y1, Y2). Реестр — `x1_reg` (регистрация ОТДЕЛЬНОЙ командой до прогноза)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch первым, до models_v2.s14)
from models_v2.s16 import x1_reg as R, x1_t0 as T0, x1_ya_doc as DOC, x1_ya_eval as EV, x1_ya_feat as FE, x1_ya_walk as W

ROOT, REP = R.ROOT, R.REP
OUT = ROOT / "data/regen/harj_x1/YA"
X1_FILE = ROOT / "data/regen/harj_deep/E1/x1.parquet"
CP_FILE, REPORT = REP / "CHECKPOINT.json", REP / "YA_REPORT.md"
TABLES = {"YA_METRICS": "METRICS", "YA_PAIRS": "PAIRS", "YA_RULE62": "RULE62", "YA_YEARS": "YEARS", "YA_GO": "GO", "YA_MCS": "MCS", "YA_STRESS": "STRESS", "YA_CELLS": "CELLS", "YA_PARAMS": "PARAMS"}
WATCH = [ROOT / p for p in ("data/regen/harj_hybrid_v6", "data/regen/harj_hybrid_v7", "data/regen/harj_deep", "docs/otchety/harj_hybrid_v6", "docs/otchety/harj_hybrid_v7", "docs/otchety/harj_deep", "state/harj_hybrid_v6.md", "state/harj_hybrid_v7.md", "state/harj_deep.md")]
HEAD = "# диагностика/оценка, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-243, лист Y-A (Y1, Y2; SBER, h = 1); код ya %s\n"


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("x1_ya_*.py")) + [Path(R.__file__)]:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def digest(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
            if f.is_file():
                h.update(str(f).encode() + sha_file(f).encode())
    return h.hexdigest()


def read_cp() -> dict:
    return json.loads(CP_FILE.read_text(encoding="utf-8")) if CP_FILE.exists() else {}


def merge_cp(key: str, val: dict) -> None:
    """Слияние: чужие ключи (P0, T0, …) сохраняются."""
    cp = read_cp()
    cp[key] = val
    CP_FILE.write_text(json.dumps(cp, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def csv_text(df: pd.DataFrame) -> str:
    return HEAD % code_hash() + df.to_csv(index=False, lineterminator="\n")


def load() -> dict:
    """Ряды T0 (по sha256) на общих датах; x1.parquet — те же даты и прогноз X1; e_d = de_e1_state.states; склад M10 V7."""
    from models_v2.s15 import de0_data as D, de_e1_state as S
    from models_v2.s14 import v7_1m_stat as ST
    fr, _ = T0.load_all()
    al = T0.common(fr)
    x1 = pd.read_parquet(X1_FILE)
    d, tp, rv = pd.DatetimeIndex(al["K1"]["d"]), pd.DatetimeIndex(al["K1"]["t_plus"]), al["K1"]["rv"].to_numpy(float)
    if not (pd.DatetimeIndex(x1["d"]).equals(d) and np.array_equal(x1["F"].to_numpy(float), al["X1"]["F"].to_numpy(float)) and pd.DatetimeIndex(x1["t_plus"]).equals(tp)):
        raise SystemExit("x1.parquet ≠ ряд X1 общих дат T0")
    ser = {k: v["F"].to_numpy(float) for k, v in al.items()}
    ser.update({k: v for k, v in ST.refs(ser["K1"], ser["K2"], ser["K3"], ser["V4:NTP2"], ser["V6:A2"]).items() if k in ("M", "K8")})
    if not np.array_equal(x1["s"].to_numpy(int), S.states(D.load_store(), d)):
        raise SystemExit("e_d (колонка s x1.parquet) ≠ de_e1_state.states")
    store = FE.load_store()
    shas = {"x1": sha_file(X1_FILE), "store_raw": sha_file(D.OUT7 / "store_raw.npz"), **T0.expected_sha()}
    return dict(x1=x1, d=d, tp=tp, rv=rv, ser=ser, ft=FE.features(store, d, tp), shas=shas)


def compute(inp: dict | None = None) -> tuple:
    """-> (прогнозы {Y1, Y2}, таблицы, числа)."""
    inp = inp or load()
    ys = {y: W.walk(inp["x1"], inp["ft"], y) for y in EV.YS}
    s = {**inp["ser"], **{y: f["F"].to_numpy(float) for y, f in ys.items()}}
    t = EV.main_tables(inp["rv"], s, inp["tp"], inp["d"], inp["x1"]["s"].to_numpy(int), ys)
    ft = inp["ft"]
    num = dict(n_dates=int(len(inp["d"])), first=str(inp["d"][0].date()), last=str(inp["d"][-1].date()), pairs_at_activation=W.check_activation(inp["d"], inp["tp"]), min_pairs=W.MIN_PAIRS, n_inactive=int((ys["Y1"]["active"] == 0).sum()),
               close_from=FE.CLOSE_FROM, close_to=FE.CLOSE_TO, close_slots="%d…%d" % (FE.CLOSE_SLOTS[0], FE.CLOSE_SLOTS[-1]), n_zero_days=int(ft["zero"].sum()), n_extreme_days=int(inp["x1"]["s"].sum()),
               input_sha256=inp["shas"], configs_spent=2, n_holm=EV.N_HOLM, n_wide=EV.N_WIDE, lams=list(W.FT.LAMS))
    return ys, t, num


def _json(x):
    return json.loads(json.dumps(x))


def _close(a, b, rtol: float = 1e-9) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close(a[k], b[k], rtol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_close(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= rtol * max(1.0, abs(a), abs(b))
    return a == b


def eval_numbers(t: dict) -> dict:
    """`YA.eval`: QLIKE по окнам, Δ с интервалами (точка), правило 6.2, «go», клетки — БЕЗ p-значений."""
    m, pr, c = t["METRICS"], t["PAIRS"], t["CELLS"]
    ql = {s: {w: float(g.loc[g["window"] == w, "qlike"].iloc[0]) for w in g["window"]} for s, g in m.groupby("series")}
    pair = {"%s|%s|%s" % (r.y, r.base, r.window): [float(r.delta_pct), float(r.lo90), float(r.hi90), float(r.delta_q), int(r.years_neg), int(r.n_years)] for r in pr.itertuples()}
    go = {"%s|%d" % (r.y, r.cond): [float(r.value), bool(r.passed)] for r in t["GO"].itertuples()}
    cells = {"%s|%s|%s" % (r.cell, r.y, r.base): [float(r.delta_q), float(r.delta_pct), float(r.lo90), float(r.hi90), int(r.n)] for r in c.itertuples()}
    return dict(qlike=ql, pairs=pair, rule62={r.y: r.rule62 for r in t["RULE62"].itertuples()}, go=go, cells=cells, p_values="не входят: записаны в YA.rules после этого ключа")


def rules(t: dict) -> dict:
    pr = t["PAIRS"]
    return dict(p={"%s|%s|%s" % (r.y, r.base, r.window): dict(p_one=float(r.p_one), p_holm_N=float(r.p_holm_N), p_holm_Nwide=float(r.p_holm_Nwide)) for r in pr.itertuples()},
                rule62={r.y: dict(rule62=r.rule62, p_dm_one=float(r.p_dm_one), p_holm_N=float(r.p_holm_N)) for r in t["RULE62"].itertuples()})


def outputs_hash() -> str:
    return digest(WATCH + [REP, OUT])


def write() -> None:
    R.guard("Y1"), R.guard("Y2")
    at = R.now_text()
    ys, t, num = compute()
    OUT.mkdir(parents=True, exist_ok=True)
    for y, f in ys.items():
        f.to_parquet(OUT / (y.lower() + ".parquet"), index=False)
    merge_cp("YA", dict(sha256={y: sha_file(OUT / (y.lower() + ".parquet")) for y in ys}, computed_at=at, n=int(len(ys["Y1"])), code=code_hash(), columns=W.OUT_COLS, configs=list(ys)))
    R.guard("Y1", computed_at=at), R.guard("Y2", computed_at=at)
    merge_cp("YA.eval", dict(numbers=_json(eval_numbers(t)), run=_json(num), written_at=R.now_text()))
    for k, v in TABLES.items():
        (REP / (k + ".csv")).write_text(csv_text(t[v]), encoding="utf-8")
    REPORT.write_text(DOC.build(t, num), encoding="utf-8")
    merge_cp("YA.rules", dict(rules=_json(rules(t)), sha256={k: sha_file(REP / (k + ".csv")) for k in TABLES}, sha256_report=sha_file(REPORT), written_at=R.now_text(), code=code_hash()))


def check() -> list:
    """Только читает. -> список расхождений (пусто = 0)."""
    before, bad, cp = outputs_hash(), [], read_cp()
    ck, ev, ru = cp.get("YA"), cp.get("YA.eval"), cp.get("YA.rules")
    if not (ck and ev and ru):
        return ["нет ключей YA / YA.eval / YA.rules в CHECKPOINT"]
    for y in ("Y1", "Y2"):
        try:
            R.guard(y, computed_at=ck["computed_at"])
        except SystemExit as e:
            bad.append(str(e))
    ts = [R._parse(x) for x in (ck["computed_at"], ev["written_at"], ru["written_at"])]
    if ts != sorted(ts) or "p_holm" in json.dumps(ev["numbers"]) or "p_one" in json.dumps(ev["numbers"]):
        bad.append("порядок/состав ключей: YA.eval должен быть записан раньше YA.rules и без p")
    ys, t, num = compute()
    for y, f in ys.items():
        p = OUT / (y.lower() + ".parquet")
        if not p.exists() or sha_file(p) != ck["sha256"][y] or not _same(pd.read_parquet(p), f):
            bad.append("%s.parquet ≠ пересчёту / sha256 ≠ CHECKPOINT YA" % y.lower())
    if not _close(_json(eval_numbers(t)), ev["numbers"]) or not _close(_json(num), ev["run"]):
        bad.append("YA.eval ≠ пересчёту")
    if not _close(_json(rules(t)), ru["rules"]):
        bad.append("YA.rules ≠ пересчёту")
    for k, v in TABLES.items():
        p = REP / (k + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != csv_text(t[v]):
            bad.append("%s.csv ≠ пересчёту" % k)
        elif ru["sha256"].get(k) != sha_file(p):
            bad.append("%s.csv: sha256 ≠ CHECKPOINT" % k)
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != DOC.build(t, num):
        bad.append("YA_REPORT.md ≠ генератору")
    if outputs_hash() != before:
        bad.append("--check изменил выходы")
    return bad


def run(do_check: bool) -> int:
    """Вход из `run_x1 --stage YA [--check]`: код возврата 0/1."""
    if not do_check:
        write()
        print("YA: y1/y2.parquet, YA_*.csv и YA_REPORT.md записаны; CHECKPOINT ключи YA, YA.eval, YA.rules")
        return 0
    before, bad = outputs_hash(), check()
    print("\n".join(bad) if bad else "OK: Y1/Y2 пересчитаны и равны файлам, YA_*.csv и отчёт = генератор, реестр до прогноза, ключ YA.eval раньше таблиц с p; выходы V6/V7/DE/P0/T0 и YA не изменены (хэш %s)" % before[:16])
    return 1 if bad else 0


def _same(a: pd.DataFrame, b: pd.DataFrame, rtol: float = 1e-9) -> bool:
    return list(a.columns) == list(b.columns) and len(a) == len(b) and all(
        np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=rtol, atol=1e-12, equal_nan=True) if a[c].dtype.kind in "fi" else bool((a[c] == b[c]).all()) for c in a.columns)
