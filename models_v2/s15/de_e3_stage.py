"""E3: стадия X3 — compute() (всё в памяти), write() (ПИШЕТ x3.parquet, CHECKPOINT `E3.x3` → `E3.eval` (БЕЗ p) → таблицы E3_*.csv с p, отчёт, `E3.rules`, строку результата в реестр), check() (ТОЛЬКО ЧИТАЕТ: пересчёт, сверка, хэш выходов до и после).
Расходуется 1 конфигурация (X3). A0g (g_s), K1–K3, X1 только читаются; sha256 файла A0g — в `E3.x3` и сверяется; обучается только g_a (numpy + L-BFGS)."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s15 import de0_io as IO, de_e1_stage as S1, de_e1_state as ST1, de_e2_feat as FE, de_e2_stage as S2, de_e3_core as C, de_e3_doc as DOC, de_e3_eval as EV, de_reg as R

OUT = IO.ROOT / "data/regen/harj_deep/E3"
X3_FILE = OUT / "x3.parquet"
A0G_FILE = IO.ROOT / "data/regen/harj_hybrid_v6/A0g.parquet"
TABLES = {"E3_METRICS": "METRICS", "E3_TESTS": "TESTS", "E3_PAIRS": "PAIRS", "E3_GO": "GO", "E3_YEARS": "YEARS", "E3_STRESS": "STRESS", "E3_MCS": "MCS"}
EXTRA = ["E3_STATE_QLIKE", "E3_A_YEARS"]
REPORT = IO.REP / "E3_REPORT.md"
HEAD = "# диагностика/оценка, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-240, лист E3 (X3; SBER, h = 1); код e3 %s\n"
COLS = ["d", "t_plus", "rv", "H", "F0", "gs", "ga", "a", "lam", "n_rows", "F"]


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("de_e3_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def csv_text(df: pd.DataFrame) -> str:
    return HEAD % code_hash() + df.to_csv(index=False, lineterminator="\n")


def _emit_csv(name: str, df: pd.DataFrame) -> None:
    IO.REP.mkdir(parents=True, exist_ok=True)
    (IO.REP / (name + ".csv")).write_text(csv_text(df), encoding="utf-8")


def load_a0g(tab_index: pd.DatetimeIndex, H: np.ndarray, expected_sha: str | None = None) -> tuple:
    """Сохранённая A0g (g_s не переобучается): -> (F_A0g на датах таблицы, sha256 файла). Файл должен совпасть с датами и H таблицы; sha256 — с ожидаемым (если задан)."""
    sha = IO.sha_file(A0G_FILE)
    if expected_sha is not None and sha != expected_sha:
        raise SystemExit("A0g.parquet: sha256 ≠ записанному (g_s переобучена или файл изменён)")
    a = pd.read_parquet(A0G_FILE)
    if not (pd.DatetimeIndex(a["d"]).equals(pd.DatetimeIndex(tab_index)) and np.allclose(a["H"].to_numpy(float), H, rtol=1e-12, atol=0)):
        raise SystemExit("A0g.parquet: даты/H ≠ таблице дней")
    return a["F"].to_numpy(float), sha


def compute(inp: dict | None = None, expected_sha: str | None = None) -> tuple:
    """-> (x3 DataFrame, tables, state_table, a_table, numbers)."""
    inp = inp or S1.load()
    d = inp["d"]
    ser, idx, tp, rv, tab = d["all_series"], d["idx"], d["tp"], d["rv"], inp["d"]["tab"]
    H = tab["H"].to_numpy(float)
    F0all, sha = load_a0g(tab.index, H, expected_sha)
    Z = importlib.import_module("models_v2.s14.v7_1m_feat").build_Z(tab)[FE.ZNAMES].to_numpy(float)
    dates, tpa = tab.index, tab["t_plus"].to_numpy("datetime64[ns]")
    w = C.walk_forward_ga(F0all, tab["rv_next"].to_numpy(float), tpa, dates, Z)
    a = C.adaptive_weights(tab["rv_next"].to_numpy(float), F0all, w["ga"], w["fitted"], tpa, dates)
    F = C.forecast(F0all, w["ga"], a)
    pos = dates.get_indexer(idx)
    if (pos < 0).any() or not np.array_equal(tab["rv_next"].to_numpy(float)[pos], rv) or not np.allclose(F0all[pos], np.asarray(ser["V6:A0g"], float), rtol=1e-12, atol=0):
        raise SystemExit("таблица дней ≠ датам/цели/A0g загрузчика V7")
    if not (np.isfinite(F[pos]).all() and (F[pos] > 0).all() and ((a >= 0) & (a <= 1)).all()):
        raise SystemExit("X3: F не положителен / не конечен или a_t вне [0; 1]")
    x1 = pd.read_parquet(S1.X1_FILE)
    if not (pd.DatetimeIndex(x1["d"]) == idx).all():
        raise SystemExit("X1: даты ≠ датам листа")
    x3 = pd.DataFrame(dict(d=idx, t_plus=tp, rv=rv, H=H[pos], F0=F0all[pos], gs=np.log(F0all[pos] / H[pos]), ga=w["ga"][pos], a=a[pos], lam=w["lam"][pos], n_rows=w["n_rows"][pos], F=F[pos]))[COLS]
    series = {k: np.asarray(ser[k], float) for k in EV.SERIES if k not in ("X1", "X3")}
    series["X1"], series["X3"] = x1["F"].to_numpy(float), x3["F"].to_numpy(float)
    t = EV.main_tables(rv, series, tp)
    st, at = EV.state_qlike(rv, series, tp, ST1.states(inp["store"], idx)), EV.a_table(x3)
    fits = [f for f in w["fits"] if f["fitted"]]
    num = dict(n_dates=int(len(idx)), first=str(idx[0].date()), last=str(idx[-1].date()), input_sha256=dict(inp["shas"], x1=IO.sha_file(S1.X1_FILE), a0g=sha), configs_spent=3, lams=list(C.LAMS), win=C.WIN, hold=C.HOLD,
               min_train=C.MIN_TRAIN, h_mult=C.H_MULT, slope=C.SLOPE, min_delta=C.MIN_DELTA, n_refits=len(fits), n_lam_10=int(sum(f["lam"] == 10.0 for f in fits)), n_holm=EV.E1V.N_HOLM, n_wide=EV.E1V.N_WIDE, a0g_sha256=sha)
    return x3, t, st, at, num


def _json(x):
    return json.loads(json.dumps(x))


def _docs(t, st, at, num) -> str:
    return DOC.build(t, EV.own_base(t["PAIRS"], "V6:A0g"), st, at, num)


def add_result_row(sb: dict, go: bool) -> None:
    """Реестр только дополняется: строка результата X3 (если ещё нет)."""
    cur = R.read()
    if ((cur["id"] == "X3") & (cur["status"] == "посчитана")).any():
        return
    text = "X3 посчитана: 6.2 против A0g %s (Δ %.2f %% К1, p %.3f), «go» %s" % (sb["rule62"], sb["delta_pct"], sb["p_one"], "да" if go else "нет")
    row = pd.DataFrame([dict(id="X3", name="результат X3", formula=text, condition="", status="посчитана", registered_at=R.now_text())])
    R.TRIALS.write_text(R.TRIALS.read_text(encoding="utf-8") + row.to_csv(index=False, header=False, lineterminator="\n"), encoding="utf-8")


def emit(x3, t, st, at, num) -> None:
    """Порядок записи: x3 → `E3.x3` → `E3.eval` (БЕЗ p) → таблицы с p и отчёт → `E3.rules`. Ключ `E3.eval` раньше любой таблицы с p."""
    R.guard("X3")
    stamp = R.now_text()
    OUT.mkdir(parents=True, exist_ok=True)
    x3.to_parquet(X3_FILE, index=False)
    IO.merge_checkpoint("E3.x3", dict(sha256=IO.sha_file(X3_FILE), computed_at=stamp, n=int(len(x3)), code=code_hash(), columns=COLS, a0g_sha256=num["a0g_sha256"]))
    R.guard("X3", computed_at=stamp)
    IO.merge_checkpoint("E3.eval", dict(numbers=_json(EV.eval_numbers(t, st, at)), run=num, written_at=R.now_text()))
    for k, v in TABLES.items():
        _emit_csv(k, t[v])
    _emit_csv("E3_STATE_QLIKE", st)
    _emit_csv("E3_A_YEARS", at)
    REPORT.write_text(_docs(t, st, at, num), encoding="utf-8")
    ru = EV.rules(t)
    IO.merge_checkpoint("E3.rules", dict(rules=_json(ru), sha256={k: IO.sha_file(IO.REP / (k + ".csv")) for k in list(TABLES) + EXTRA}, sha256_report=IO.sha_file(REPORT), written_at=R.now_text(), code=code_hash()))
    add_result_row(EV.own_base(t["PAIRS"], "V6:A0g"), ru["go"])


def write() -> None:
    R.guard("X3")
    emit(*compute())


def outputs_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(IO.REP.glob("E3_*")) + [IO.REP / "ALL_TRIALS.csv", X3_FILE]:
        h.update(f.name.encode() + (IO.sha_file(f).encode() if f.exists() else b"-"))
    return h.hexdigest()


def check() -> list:
    """Только читает. -> список расхождений (пусто = 0 расхождений)."""
    before, bad, cp = outputs_hash(), [], IO.read_cp()
    ck, ev, ru = cp.get("E3.x3"), cp.get("E3.eval"), cp.get("E3.rules")
    if not (ck and ev and ru):
        return ["нет ключей E3.x3 / E3.eval / E3.rules в CHECKPOINT"]
    try:
        R.guard("X3", cp=cp, computed_at=ck["computed_at"])
    except SystemExit as e:
        bad.append(str(e))
    if ev["written_at"] > ru["written_at"] or ck["computed_at"] > ev["written_at"] or "p_holm" in json.dumps(ev["numbers"]) or "p_one" in json.dumps(ev["numbers"]):
        bad.append("порядок/состав ключей: E3.eval должен быть записан раньше E3.rules и без p")
    try:
        x3, t, st, at, num = compute(expected_sha=ck["a0g_sha256"])
    except SystemExit as e:
        return bad + [str(e)]
    if not X3_FILE.exists() or not pd.read_parquet(X3_FILE).equals(x3) or IO.sha_file(X3_FILE) != ck["sha256"]:
        bad.append("x3.parquet ≠ пересчёту / sha256 ≠ CHECKPOINT E3.x3")
    if not S2._close(_json(EV.eval_numbers(t, st, at)), ev["numbers"]) or not S2._close(_json(num), ev["run"]):
        bad.append("E3.eval ≠ пересчёту")
    if not S2._close(_json(EV.rules(t)), ru["rules"]):
        bad.append("E3.rules ≠ пересчёту")
    for name, df in [(k, t[v]) for k, v in TABLES.items()] + [("E3_STATE_QLIKE", st), ("E3_A_YEARS", at)]:
        p = IO.REP / (name + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != csv_text(df):
            bad.append("%s.csv ≠ пересчёту" % name)
        elif ru["sha256"].get(name) != IO.sha_file(p):
            bad.append("%s.csv: sha256 ≠ CHECKPOINT" % name)
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != _docs(t, st, at, num):
        bad.append("E3_REPORT.md ≠ генератору")
    if outputs_hash() != before:
        bad.append("--check изменил выходы")
    return bad
