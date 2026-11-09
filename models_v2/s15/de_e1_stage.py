"""E1: стадия X1 — compute() (всё в памяти), write() (ПИШЕТ x1.parquet, CHECKPOINT `E1.x1` → `E1.eval` (БЕЗ p) → таблицы E1_*.csv с p, отчёт, `E1.rules`), check() (ТОЛЬКО ЧИТАЕТ: пересчёт, сверка, хэш выходов до и после).
Прогнозы V4–V7 и K1–K3 только читаются (загрузчик V7 сверяет sha256 по CHECKPOINT V7); обучения нет. Расходуется 1 конфигурация (X1)."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s15 import de0_groups as G, de0_io as IO, de_e1_doc as DOC, de_e1_eval as EV, de_e1_state as S, de_reg as R

OUT = IO.ROOT / "data/regen/harj_deep/E1"
X1_FILE = OUT / "x1.parquet"
TABLES = {"E1_METRICS": "METRICS", "E1_TESTS": "TESTS", "E1_PAIRS": "PAIRS", "E1_GO": "GO", "E1_YEARS": "YEARS", "E1_STRESS": "STRESS", "E1_MCS": "MCS"}
EXTRA = ["E1_STATE_QLIKE", "E1_PI"]
REPORT = IO.REP / "E1_REPORT.md"
HEAD = "# диагностика/оценка, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-240, лист E1 (X1; SBER, h = 1); код e1 %s\n"
COLS = ["d", "t_plus", "rv", "M", "C", "s", "pi", "F", "pi0", "pi1", "a0", "a1"]


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("de_e1_*.py")) + [Path(__file__).parent / "de_reg.py"]:      # только код E1 (листы E2… и run_de.py хэш E1 не меняют)
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def csv_text(df: pd.DataFrame) -> str:
    return HEAD % code_hash() + df.to_csv(index=False, lineterminator="\n")


def _emit_csv(name: str, df: pd.DataFrame) -> None:
    IO.REP.mkdir(parents=True, exist_ok=True)
    (IO.REP / (name + ".csv")).write_text(csv_text(df), encoding="utf-8")


def _cp(key: str, val: dict) -> None:
    IO.merge_checkpoint(key, val)


def load() -> dict:
    importlib.import_module("torch")      # первым: конфликт OpenMP при последующем импорте models_v2.s14 (через importlib — см. de_e1_eval)
    from models_v2.s15 import de0_data as D
    d, store = D.load(), D.load_store()
    ins = {n: IO.sha_file(p) for n, p in (("cfg8", D.OUT7 / "3/cfg8.parquet"), ("k23_SBER", IO.ROOT / "data/regen/harj_panel_hybrid/k23_SBER.parquet"), ("NTP2", IO.ROOT / "data/regen/harj_hybrid_v4/NTP2.parquet"), ("store_raw", D.OUT7 / "store_raw.npz"))}
    return dict(d=d, store=store, shas=ins)


def compute(inp: dict | None = None) -> tuple:
    """-> (x1 DataFrame, tables, state_table, pi_table, numbers)."""
    inp = inp or load()
    d, store = inp["d"], inp["store"]
    ser, idx, tp, rv = d["all_series"], d["idx"], d["tp"], d["rv"]
    M, C = np.asarray(ser["M"], float), np.asarray(ser["cfg8"], float)
    if not np.allclose(M, np.sqrt(np.asarray(ser["K2"], float) * np.asarray(ser["K3"], float)), rtol=1e-12, atol=0):
        raise SystemExit("M ≠ √(K2·K3)")
    s = S.states(store, idx)
    r = S.run(rv, M, C, s, idx.to_numpy(), tp.to_numpy())
    x1 = pd.DataFrame(dict(d=idx, t_plus=tp, rv=rv, M=M, C=C, s=s, pi=r["pi"], F=r["F"], pi0=r["pis"][:, 0], pi1=r["pis"][:, 1], a0=r["a0"], a1=r["a1"]))[COLS]
    if not (np.isfinite(r["F"]).all() and (r["F"] > 0).all()):
        raise SystemExit("X1: F не положителен / не конечен")
    series = {k: np.asarray(ser[k], float) for k in EV.SERIES if k != "X1"}
    series["X1"] = r["F"]
    t = EV.main_tables(rv, series, tp)
    st, pit = EV.state_qlike(rv, series, tp, s), EV.pi_table(x1)
    nobar = int((~np.isfinite(G.day_stats(store, idx)["maxabs"])).sum())
    num = dict(n_dates=int(len(idx)), first=str(idx[0].date()), last=str(idx[-1].date()), n_extreme_days=int(s.sum()), n_no_bar_days=nobar, input_sha256=inp["shas"], configs_spent=1, grid=list(S.GRID), val_pairs=S.VAL, n_holm=EV.N_HOLM, n_wide=EV.N_WIDE)
    return x1, t, st, pit, num


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


def emit(x1, t, st, pit, num) -> None:
    """Порядок записи: x1 → `E1.x1` → `E1.eval` (БЕЗ p) → таблицы с p и отчёт → `E1.rules`. Ключ `E1.eval` раньше любой таблицы с p."""
    R.guard("X1")
    at = R.now_text()
    OUT.mkdir(parents=True, exist_ok=True)
    x1.to_parquet(X1_FILE, index=False)
    _cp("E1.x1", dict(sha256=IO.sha_file(X1_FILE), computed_at=at, n=int(len(x1)), code=code_hash(), columns=COLS))
    R.guard("X1", computed_at=at)
    _cp("E1.eval", dict(numbers=_json(EV.eval_numbers(t, st, pit)), run=num, written_at=R.now_text()))
    for k, v in TABLES.items():
        _emit_csv(k, t[v])
    _emit_csv("E1_STATE_QLIKE", st)
    _emit_csv("E1_PI", pit)
    sb = EV.state_benefit(t["PAIRS"])
    REPORT.write_text(DOC.build(t, sb, st, pit, num), encoding="utf-8")
    _cp("E1.rules", dict(rules=_json(EV.rules(t)), state_benefit=_json(sb), sha256={k: IO.sha_file(IO.REP / (k + ".csv")) for k in list(TABLES) + EXTRA}, sha256_report=IO.sha_file(REPORT), written_at=R.now_text(), code=code_hash()))


def write() -> None:
    R.guard("X1")
    x1, t, st, pit, num = compute()
    emit(x1, t, st, pit, num)


def outputs_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(IO.REP.glob("E1_*")) + [IO.REP / "ALL_TRIALS.csv", X1_FILE]:
        h.update(f.name.encode() + (IO.sha_file(f).encode() if f.exists() else b"-"))
    return h.hexdigest()


def check() -> list:
    """Только читает. -> список расхождений (пусто = 0 расхождений)."""
    before, bad, cp = outputs_hash(), [], IO.read_cp()
    ck, ev, ru = cp.get("E1.x1"), cp.get("E1.eval"), cp.get("E1.rules")
    if not (ck and ev and ru):
        return ["нет ключей E1.x1 / E1.eval / E1.rules в CHECKPOINT"]
    try:
        R.guard("X1", cp=cp, computed_at=ck["computed_at"])
    except SystemExit as e:
        bad.append(str(e))
    if ev["written_at"] > ru["written_at"] or ck["computed_at"] > ev["written_at"] or "p_holm" in json.dumps(ev["numbers"]) or "p_one" in json.dumps(ev["numbers"]):
        bad.append("порядок/состав ключей: E1.eval должен быть записан раньше E1.rules и без p")
    x1, t, st, pit, num = compute()
    if not X1_FILE.exists() or not pd.read_parquet(X1_FILE).equals(x1) or IO.sha_file(X1_FILE) != ck["sha256"]:
        bad.append("x1.parquet ≠ пересчёту / sha256 ≠ CHECKPOINT E1.x1")
    if not _close(_json(EV.eval_numbers(t, st, pit)), ev["numbers"]) or not _close(_json(num), ev["run"]):
        bad.append("E1.eval ≠ пересчёту")
    if not _close(_json(EV.rules(t)), ru["rules"]):
        bad.append("E1.rules ≠ пересчёту")
    for name, df in [(k, t[v]) for k, v in TABLES.items()] + [("E1_STATE_QLIKE", st), ("E1_PI", pit)]:
        p = IO.REP / (name + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != csv_text(df):
            bad.append("%s.csv ≠ пересчёту" % name)
        elif ru["sha256"].get(name) != IO.sha_file(p):
            bad.append("%s.csv: sha256 ≠ CHECKPOINT" % name)
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != DOC.build(t, EV.state_benefit(t["PAIRS"]), st, pit, num):
        bad.append("E1_REPORT.md ≠ генератору")
    if outputs_hash() != before:
        bad.append("--check изменил выходы")
    return bad
