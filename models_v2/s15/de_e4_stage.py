"""E4: стадия X4 — compute() (всё в памяти), write() (ПИШЕТ x4.parquet, CHECKPOINT `E4.x4` → `E4.eval` (БЕЗ p) → таблицы E4_*.csv с p, отчёт, `E4.rules`, строку результата в реестр), check() (ТОЛЬКО ЧИТАЕТ: пересчёт, сверка, хэш выходов до и после).
Расходуется 1 конфигурация (X4). Обучаются 5 GBM (`v6_1_fit.fit_gbm`, через importlib); A0g, K1–K3, X1 только читаются; расширяющаяся модель сверяется с A0g."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s15 import de0_io as IO, de_e1_stage as S1, de_e1_state as ST1, de_e2_stage as S2, de_e4_core as C, de_e4_doc as DOC, de_e4_eval as EV, de_reg as R

OUT = IO.ROOT / "data/regen/harj_deep/E4"
X4_FILE = OUT / "x4.parquet"
A0G_FILE = IO.ROOT / "data/regen/harj_hybrid_v6/A0g.parquet"
TABLES = {"E4_METRICS": "METRICS", "E4_TESTS": "TESTS", "E4_PAIRS": "PAIRS", "E4_GO": "GO", "E4_YEARS": "YEARS", "E4_STRESS": "STRESS", "E4_MCS": "MCS"}
EXTRA = ["E4_STATE_QLIKE", "E4_ALPHA_YEARS"]
REPORT = IO.REP / "E4_REPORT.md"
HEAD = "# диагностика/оценка, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-240, лист E4 (X4; SBER, h = 1); код e4 %s\n"
GN = ["g_exp", "g_750", "g_1000", "g_1250", "g_1500"]
COLS = ["d", "t_plus", "rv", "H", *GN, "gbar", "sd", "sbar", "act", "nval", "kappa", "alpha", "F", "F_ens"]


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("de_e4_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def csv_text(df: pd.DataFrame) -> str:
    return HEAD % code_hash() + df.to_csv(index=False, lineterminator="\n")


def _emit_csv(name: str, df: pd.DataFrame) -> None:
    IO.REP.mkdir(parents=True, exist_ok=True)
    (IO.REP / (name + ".csv")).write_text(csv_text(df), encoding="utf-8")


def panel() -> tuple:
    """(tab V6 с d, t_plus, rv_next, h; S — 9 признаков) с диска; импорт через importlib (AST-тест DE0)."""
    return importlib.import_module("models_v2.s13.v6_4_stage").panel()


def compute(inp: dict | None = None, fit=C.default_fit) -> tuple:
    """-> (x4 DataFrame, tables, state_table, alpha_table, numbers)."""
    inp = inp or S1.load()
    d = inp["d"]
    ser, idx, tp, rv = d["all_series"], d["idx"], d["tp"], d["rv"]
    tab, S = panel()
    H, tpa, dates = tab["h"].to_numpy(float), tab["t_plus"].to_numpy("datetime64[ns]"), tab.index
    e = np.log(tab["rv_next"].to_numpy(float) / H)
    r = C.run(S.to_numpy(float), e, tab["rv_next"].to_numpy(float), H, tpa, dates, fit)
    pos = dates.get_indexer(idx)
    if (pos < 0).any() or not np.array_equal(tab["rv_next"].to_numpy(float)[pos], rv) or not np.allclose(H[pos], np.asarray(ser["K1"], float), rtol=1e-12, atol=0):
        raise SystemExit("таблица дней V6 ≠ датам/цели/K1 загрузчика V7")
    if not np.allclose(H[pos] * np.exp(r["G"][pos, 0]), np.asarray(ser["V6:A0g"], float), rtol=1e-9, atol=0):
        raise SystemExit("расширяющаяся модель ≠ A0g")
    F = r["F"]
    if not (r["fitted"][pos].all() and np.isfinite(F[pos]).all() and (F[pos] > 0).all() and ((r["alpha"] > 0) & (r["alpha"] <= 1)).all()):
        raise SystemExit("X4: нет модели / F не положителен / α вне (0; 1]")
    x1 = pd.read_parquet(S1.X1_FILE)
    if not (pd.DatetimeIndex(x1["d"]) == idx).all():
        raise SystemExit("X1: даты ≠ датам листа")
    cols = dict(d=idx, t_plus=tp, rv=rv, H=H[pos], **{n: r["G"][pos, k] for k, n in enumerate(GN)}, **{k: r[k][pos] for k in ("gbar", "sd", "sbar", "act", "nval", "kappa", "alpha", "F", "F_ens")})
    x4 = pd.DataFrame(cols)[COLS]
    series = {k: np.asarray(ser[k], float) for k in EV.SERIES if k not in ("X1", "ENS", "X4")}
    series["X1"], series["ENS"], series["X4"] = x1["F"].to_numpy(float), x4["F_ens"].to_numpy(float), x4["F"].to_numpy(float)
    t = EV.main_tables(rv, series, tp)
    st, at = EV.state_qlike(rv, series, tp, ST1.states(inp["store"], idx)), EV.alpha_table(x4)
    num = dict(n_dates=int(len(idx)), first=str(idx[0].date()), last=str(idx[-1].date()), input_sha256=dict(inp["shas"], x1=IO.sha_file(S1.X1_FILE), a0g=IO.sha_file(A0G_FILE)), configs_spent=4, windows=list(C.WINDOWS), kappas=list(C.KAPPAS), val=C.VAL, min_val=C.MIN_VAL,
               min_s=C.MIN_S, n_refits=len(r["rows"]), n_act_days=int(x4["act"].sum()), n_holm=EV.E1V.N_HOLM, n_wide=EV.E1V.N_WIDE)
    return x4, t, st, at, num


def _json(x):
    return json.loads(json.dumps(x))


def _docs(t, st, at, num) -> str:
    return DOC.build(t, EV.own_base(t["PAIRS"], "ENS"), EV.own_base(t["PAIRS"], "V6:A0g"), st, at, num)


def add_result_row(sb_ens: dict, sb_a: dict, go: bool) -> None:
    """Реестр только дополняется: строка результата X4 (если ещё нет)."""
    cur = R.read()
    if ((cur["id"] == "X4") & (cur["status"] == "посчитана")).any():
        return
    text = "X4 посчитана: 6.2 против ENS %s (Δ %.2f %% К1, p %.3f), против A0g %s (Δ %.2f %% К1, p %.3f), «go» %s" % (sb_ens["rule62"], sb_ens["delta_pct"], sb_ens["p_one"], sb_a["rule62"], sb_a["delta_pct"], sb_a["p_one"], "да" if go else "нет")
    row = pd.DataFrame([dict(id="X4", name="результат X4", formula=text, condition="", status="посчитана", registered_at=R.now_text())])
    R.TRIALS.write_text(R.TRIALS.read_text(encoding="utf-8") + row.to_csv(index=False, header=False, lineterminator="\n"), encoding="utf-8")


def emit(x4, t, st, at, num) -> None:
    """Порядок записи: x4 → `E4.x4` → `E4.eval` (БЕЗ p) → таблицы с p и отчёт → `E4.rules`."""
    R.guard("X4")
    stamp = R.now_text()
    OUT.mkdir(parents=True, exist_ok=True)
    x4.to_parquet(X4_FILE, index=False)
    IO.merge_checkpoint("E4.x4", dict(sha256=IO.sha_file(X4_FILE), computed_at=stamp, n=int(len(x4)), code=code_hash(), columns=COLS, a0g_sha256=num["input_sha256"]["a0g"]))
    R.guard("X4", computed_at=stamp)
    IO.merge_checkpoint("E4.eval", dict(numbers=_json(EV.eval_numbers(t, st, at)), run=num, written_at=R.now_text()))
    for k, v in TABLES.items():
        _emit_csv(k, t[v])
    _emit_csv("E4_STATE_QLIKE", st)
    _emit_csv("E4_ALPHA_YEARS", at)
    REPORT.write_text(_docs(t, st, at, num), encoding="utf-8")
    ru = EV.rules(t)
    IO.merge_checkpoint("E4.rules", dict(rules=_json(ru), sha256={k: IO.sha_file(IO.REP / (k + ".csv")) for k in list(TABLES) + EXTRA}, sha256_report=IO.sha_file(REPORT), written_at=R.now_text(), code=code_hash()))
    add_result_row(EV.own_base(t["PAIRS"], "ENS"), EV.own_base(t["PAIRS"], "V6:A0g"), ru["go"])


def write() -> None:
    R.guard("X4")
    emit(*compute())


def outputs_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(IO.REP.glob("E4_*")) + [IO.REP / "ALL_TRIALS.csv", X4_FILE]:
        h.update(f.name.encode() + (IO.sha_file(f).encode() if f.exists() else b"-"))
    return h.hexdigest()


def _order_bad(ck: dict, ev: dict, ru: dict) -> bool:
    return ev["written_at"] > ru["written_at"] or ck["computed_at"] > ev["written_at"] or "p_holm" in json.dumps(ev["numbers"]) or "p_one" in json.dumps(ev["numbers"])


def check() -> list:
    """Только читает. -> список расхождений (пусто = 0 расхождений)."""
    before, bad, cp = outputs_hash(), [], IO.read_cp()
    ck, ev, ru = cp.get("E4.x4"), cp.get("E4.eval"), cp.get("E4.rules")
    if not (ck and ev and ru):
        return ["нет ключей E4.x4 / E4.eval / E4.rules в CHECKPOINT"]
    try:
        R.guard("X4", cp=cp, computed_at=ck["computed_at"])
    except SystemExit as e:
        bad.append(str(e))
    if _order_bad(ck, ev, ru):
        bad.append("порядок/состав ключей: E4.eval должен быть записан раньше E4.rules и без p")
    try:
        x4, t, st, at, num = compute()
    except SystemExit as e:
        return bad + [str(e)]
    if not X4_FILE.exists() or not pd.read_parquet(X4_FILE).equals(x4) or IO.sha_file(X4_FILE) != ck["sha256"] or num["input_sha256"]["a0g"] != ck["a0g_sha256"]:
        bad.append("x4.parquet ≠ пересчёту / sha256 ≠ CHECKPOINT E4.x4")
    if not S2._close(_json(EV.eval_numbers(t, st, at)), ev["numbers"]) or not S2._close(_json(num), ev["run"]):
        bad.append("E4.eval ≠ пересчёту")
    if not S2._close(_json(EV.rules(t)), ru["rules"]):
        bad.append("E4.rules ≠ пересчёту")
    for name, df in [(k, t[v]) for k, v in TABLES.items()] + [("E4_STATE_QLIKE", st), ("E4_ALPHA_YEARS", at)]:
        p = IO.REP / (name + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != csv_text(df):
            bad.append("%s.csv ≠ пересчёту" % name)
        elif ru["sha256"].get(name) != IO.sha_file(p):
            bad.append("%s.csv: sha256 ≠ CHECKPOINT" % name)
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != _docs(t, st, at, num):
        bad.append("E4_REPORT.md ≠ генератору")
    if outputs_hash() != before:
        bad.append("--check изменил выходы")
    return bad
