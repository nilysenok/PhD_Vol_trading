"""E2: стадия X2 — compute() (всё в памяти), write() (ПИШЕТ x2.parquet, CHECKPOINT `E2.x2` → `E2.eval` (БЕЗ p) → таблицы E2_*.csv с p, отчёт, `E2.rules`, строку результата в реестр), check() (ТОЛЬКО ЧИТАЕТ: пересчёт, сверка, хэш выходов до и после).
Расходуется 1 конфигурация (X2). Прогнозы V4–V7, K1–K3, X1 только читаются; обучается только сама X2 (numpy + L-BFGS)."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s15 import de0_io as IO, de_e1_stage as S1, de_e1_state as ST1, de_e2_doc as DOC, de_e2_eval as EV, de_e2_feat as FE, de_e2_wf as WF, de_reg as R

OUT = IO.ROOT / "data/regen/harj_deep/E2"
X2_FILE = OUT / "x2.parquet"
TABLES = {"E2_METRICS": "METRICS", "E2_TESTS": "TESTS", "E2_PAIRS": "PAIRS", "E2_GO": "GO", "E2_YEARS": "YEARS", "E2_STRESS": "STRESS", "E2_MCS": "MCS"}
EXTRA = ["E2_STATE_QLIKE", "E2_CALIB", "E2_YEARS_P"]
REPORT = IO.REP / "E2_REPORT.md"
HEAD = "# диагностика/оценка, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-240, лист E2 (X2; SBER, h = 1); код e2 %s\n"
COLS = ["d", "t_plus", "rv", "H", "e", "rs_neg_share", "p", "cn", "cs", "F", "beta", "thr"]


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("de_e2_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def csv_text(df: pd.DataFrame) -> str:
    return HEAD % code_hash() + df.to_csv(index=False, lineterminator="\n")


def _emit_csv(name: str, df: pd.DataFrame) -> None:
    IO.REP.mkdir(parents=True, exist_ok=True)
    (IO.REP / (name + ".csv")).write_text(csv_text(df), encoding="utf-8")


def _inputs(inp: dict) -> tuple:
    """-> (tab, Xraw (n, 11), e (n,)) на ВСЕХ датах таблицы дней (2016-12-30 …)."""
    tab, store = inp["d"]["tab"], inp["store"]
    feat = importlib.import_module("models_v2.s14.v7_1m_feat")          # построитель признаков Z V7 (не обучатель); torch уже импортирован в `load`
    Z = feat.build_Z(tab)[FE.ZNAMES].to_numpy(float)
    e = ST1.states(store, tab.index)
    return tab, FE.day_inputs(Z, e, FE.rs_neg_share(store, tab.index)), e


def compute(inp: dict | None = None) -> tuple:
    """-> (x2 DataFrame, tables, state_table, calibration, years_table, numbers)."""
    inp = inp or S1.load()
    d = inp["d"]
    ser, idx, tp, rv = d["all_series"], d["idx"], d["tp"], d["rv"]
    tab, X, e = _inputs(inp)
    w = WF.walk_forward(tab["H"].to_numpy(float), tab["rv_next"].to_numpy(float), tab["t_plus"].to_numpy("datetime64[ns]"), tab.index, X, str(idx[0].date()))
    pos = tab.index.get_indexer(idx)
    if (pos < 0).any() or not np.array_equal(tab["rv_next"].to_numpy(float)[pos], rv) or not np.allclose(tab["H"].to_numpy(float)[pos], np.asarray(ser["K1"], float), rtol=1e-12, atol=0):
        raise SystemExit("таблица дней ≠ датам/цели/K1 загрузчика V7")
    if not (w["fitted"][pos].all() and np.isfinite(w["F"][pos]).all() and (w["F"][pos] > 0).all() and np.isfinite(w["p"][pos]).all()):
        raise SystemExit("X2: на оцениваемых датах нет модели / F не положителен / p не конечен")
    x1 = pd.read_parquet(S1.X1_FILE)
    if not (pd.DatetimeIndex(x1["d"]) == idx).all():
        raise SystemExit("X1: даты ≠ датам листа")
    x2 = pd.DataFrame(dict(d=idx, t_plus=tp, rv=rv, H=tab["H"].to_numpy(float)[pos], e=e[pos], rs_neg_share=X[pos, -1], **{k: w[k][pos] for k in ("p", "cn", "cs", "F", "beta", "thr")}))[COLS]
    series = {k: np.asarray(ser[k], float) for k in EV.SERIES if k not in ("X1", "X2")}
    series["X1"], series["X2"] = x1["F"].to_numpy(float), x2["F"].to_numpy(float)
    t = EV.main_tables(rv, series, tp)
    st, cal, bt = EV.state_qlike(rv, series, tp, x2["e"].to_numpy()), EV.calibration(tp, x2["p"], rv, x2["thr"]), EV.beta_table(x2)
    fits = [f for f in w["fits"] if f["fitted"]]
    num = dict(n_dates=int(len(idx)), first=str(idx[0].date()), last=str(idx[-1].date()), input_sha256=dict(inp["shas"], x1=IO.sha_file(S1.X1_FILE)), configs_spent=2, betas=list(WF.M.BETAS), hold=WF.HOLD, seeds=list(WF.M.SEEDS),
               maxiter=WF.M.MAXITER, n_refits=len(fits), n_beta_1=int(sum(f["beta"] == 1.0 for f in fits)), n_holm=EV.E1V.N_HOLM, n_wide=EV.E1V.N_WIDE)
    return x2, t, st, cal, bt, num


def _json(x):
    return json.loads(json.dumps(x))


def _docs(t, st, cal, bt, num) -> str:
    return DOC.build(t, EV.own_base(t["PAIRS"], "V6:A0g"), EV.own_base(t["PAIRS"], "cfg4"), st, cal, bt, num)


def add_result_row(sb: dict, go: bool) -> None:
    """Реестр только дополняется: строка результата X2 (если ещё нет)."""
    cur = R.read()
    if ((cur["id"] == "X2") & (cur["status"] == "посчитана")).any():
        return
    text = "X2 посчитана: 6.2 против A0g %s (Δ %.2f %% К1, p %.3f), «go» %s" % (sb["rule62"], sb["delta_pct"], sb["p_one"], "да" if go else "нет")
    row = pd.DataFrame([dict(id="X2", name="результат X2", formula=text, condition="", status="посчитана", registered_at=R.now_text())])
    R.TRIALS.write_text(R.TRIALS.read_text(encoding="utf-8") + row.to_csv(index=False, header=False, lineterminator="\n"), encoding="utf-8")


def emit(x2, t, st, cal, bt, num) -> None:
    """Порядок записи: x2 → `E2.x2` → `E2.eval` (БЕЗ p) → таблицы с p и отчёт → `E2.rules`. Ключ `E2.eval` раньше любой таблицы с p."""
    R.guard("X2")
    at = R.now_text()
    OUT.mkdir(parents=True, exist_ok=True)
    x2.to_parquet(X2_FILE, index=False)
    IO.merge_checkpoint("E2.x2", dict(sha256=IO.sha_file(X2_FILE), computed_at=at, n=int(len(x2)), code=code_hash(), columns=COLS))
    R.guard("X2", computed_at=at)
    IO.merge_checkpoint("E2.eval", dict(numbers=_json(EV.eval_numbers(t, st, cal, bt)), run=num, written_at=R.now_text()))
    for k, v in TABLES.items():
        _emit_csv(k, t[v])
    _emit_csv("E2_STATE_QLIKE", st)
    _emit_csv("E2_CALIB", cal)
    _emit_csv("E2_YEARS_P", bt)
    REPORT.write_text(_docs(t, st, cal, bt, num), encoding="utf-8")
    ru = EV.rules(t)
    IO.merge_checkpoint("E2.rules", dict(rules=_json(ru), sha256={k: IO.sha_file(IO.REP / (k + ".csv")) for k in list(TABLES) + EXTRA}, sha256_report=IO.sha_file(REPORT), written_at=R.now_text(), code=code_hash()))
    add_result_row(EV.own_base(t["PAIRS"], "V6:A0g"), ru["go"])


def write() -> None:
    R.guard("X2")
    emit(*compute())


def outputs_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(IO.REP.glob("E2_*")) + [IO.REP / "ALL_TRIALS.csv", X2_FILE]:
        h.update(f.name.encode() + (IO.sha_file(f).encode() if f.exists() else b"-"))
    return h.hexdigest()


def _close(a, b, rtol: float = 1e-9) -> bool:
    """Как `de_e1_stage._close`, плюс равные бесконечности (условие go 5 при нулевом выигрыше даёт inf)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close(a[k], b[k], rtol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_close(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        if not (np.isfinite(a) and np.isfinite(b)):
            return a == b or (np.isnan(a) and np.isnan(b))
        return abs(a - b) <= rtol * max(1.0, abs(a), abs(b))
    return a == b


def check() -> list:
    """Только читает. -> список расхождений (пусто = 0 расхождений)."""
    before, bad, cp = outputs_hash(), [], IO.read_cp()
    ck, ev, ru = cp.get("E2.x2"), cp.get("E2.eval"), cp.get("E2.rules")
    if not (ck and ev and ru):
        return ["нет ключей E2.x2 / E2.eval / E2.rules в CHECKPOINT"]
    try:
        R.guard("X2", cp=cp, computed_at=ck["computed_at"])
    except SystemExit as e:
        bad.append(str(e))
    if ev["written_at"] > ru["written_at"] or ck["computed_at"] > ev["written_at"] or "p_holm" in json.dumps(ev["numbers"]) or "p_one" in json.dumps(ev["numbers"]):
        bad.append("порядок/состав ключей: E2.eval должен быть записан раньше E2.rules и без p")
    x2, t, st, cal, bt, num = compute()
    if not X2_FILE.exists() or not pd.read_parquet(X2_FILE).equals(x2) or IO.sha_file(X2_FILE) != ck["sha256"]:
        bad.append("x2.parquet ≠ пересчёту / sha256 ≠ CHECKPOINT E2.x2")
    if not _close(_json(EV.eval_numbers(t, st, cal, bt)), ev["numbers"]) or not _close(_json(num), ev["run"]):
        bad.append("E2.eval ≠ пересчёту")
    if not _close(_json(EV.rules(t)), ru["rules"]):
        bad.append("E2.rules ≠ пересчёту")
    for name, df in [(k, t[v]) for k, v in TABLES.items()] + [("E2_STATE_QLIKE", st), ("E2_CALIB", cal), ("E2_YEARS_P", bt)]:
        p = IO.REP / (name + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != csv_text(df):
            bad.append("%s.csv ≠ пересчёту" % name)
        elif ru["sha256"].get(name) != IO.sha_file(p):
            bad.append("%s.csv: sha256 ≠ CHECKPOINT" % name)
    if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != _docs(t, st, cal, bt, num):
        bad.append("E2_REPORT.md ≠ генератору")
    if outputs_hash() != before:
        bad.append("--check изменил выходы")
    return bad
