"""FIN: стадия итога пилота — compute() (всё в памяти; обучения нет), write() (ПИШЕТ `FIN_*.csv`, строку X1 в реестр, документ, CHECKPOINT `FIN`), check() (ТОЛЬКО ЧИТАЕТ: пересчёт, сверка таблиц, документа, реестра, листов E1–E4, хэш выходов до и после).
Требует CHECKPOINT[E5] с condition = false (X5 не считается). Прогнозы X1–X4 и контроли только читаются."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s15 import de0_io as IO, de_e1_stage as S1, de_e1_state as ST1, de_e2_stage as S2, de_e3_stage as S3, de_e5_stage as S5, de_fin_leaf as LEAF, de_fin_tabs as T, de_reg as R
from models_v2.s15 import de_fin_doc as D, de_fin_doc2 as D2

DOC_NAME = "HARJ_HYBRID_SBER_DEEP_EXTENSION_FINAL.md"
NAMES = {"FIN_METRICS": "METRICS", "FIN_TESTS": "TESTS", "FIN_PAIRS": "PAIRS", "FIN_GO": "GO", "FIN_YEARS": "YEARS", "FIN_STRESS": "STRESS", "FIN_MCS": "MCS", "FIN_STATE": "STATE", "FIN_STAB": "STAB", "FIN_ADV": "ADV"}
HEAD = "# итог пилота, не отбор; DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-240, лист FIN (SBER, h = 1; обучения нет); код fin %s\n"


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("de_fin_*.py")) + [Path(__file__).parent / "de_e5_stage.py"]:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def csv_text(df: pd.DataFrame) -> str:
    return HEAD % code_hash() + df.to_csv(index=False, lineterminator="\n")


def doc_path() -> Path:
    return IO.REP.parent / DOC_NAME


def x_files() -> dict:
    return {"X1": S1.X1_FILE, "X2": S2.X2_FILE, "X3": S3.X3_FILE, "X4": importlib.import_module("models_v2.s15.de_e4_stage").X4_FILE}


def forecasts(idx, tp, rv) -> dict:
    """F рядов X1–X4 и ENS из parquet листов; даты, t⁺ и RV — те же, что у загрузчика."""
    out = {}
    for k, p in x_files().items():
        x = pd.read_parquet(p)
        if not ((pd.DatetimeIndex(x["d"]) == idx).all() and (pd.DatetimeIndex(x["t_plus"]) == tp).all() and np.array_equal(x["rv"].to_numpy(float), rv)):
            raise SystemExit("%s: даты / t⁺ / RV ≠ загрузчику V7 (одинаковых дат нет)" % k)
        out[k] = x["F"].to_numpy(float)
        if k == "X4":
            out["ENS"] = x["F_ens"].to_numpy(float)
    return out


def compute(inp: dict | None = None) -> tuple:
    """-> (tables, num, leaf=(сверено, расхождений), список расхождений с листами)."""
    inp = inp or S1.load()
    d = inp["d"]
    s = T.assemble(d["all_series"], forecasts(d["idx"], d["tp"], d["rv"]))
    t = T.all_tables(d["rv"], s, d["tp"], ST1.states(inp["store"], d["idx"]))
    n, bad = LEAF.mismatches(t)
    num = dict(n_dates=int(len(d["idx"])), first=str(d["idx"][0].date()), last=str(d["idx"][-1].date()))
    return t, num, (n, len(bad)), bad


def add_x1_row(t: dict) -> None:
    """Реестр только дополняется: строка результата X1 (если ещё нет)."""
    cur = R.read()
    if ((cur["id"] == "X1") & (cur["status"] == "посчитана")).any():
        return
    r = D.pair(t, "X1", "V4:NTP2", "2018_2021")
    text = "X1 посчитана: 6.2 против NTP2 %s (Δ %.2f %% К1, p %.3f), «go» нет" % (r["rule62"], r["delta_pct"], r["p_one"])
    row = pd.DataFrame([dict(id="X1", name="результат X1", formula=text, condition="", status="посчитана", registered_at=R.now_text())])
    R.TRIALS.write_text(R.TRIALS.read_text(encoding="utf-8") + row.to_csv(index=False, header=False, lineterminator="\n"), encoding="utf-8")


def context(t: dict, num: dict, leaf: tuple) -> dict:
    cp = IO.read_cp()
    e5 = cp.get("E5")
    if not e5 or e5.get("condition") is not False:
        raise SystemExit("FIN: нет CHECKPOINT[E5] с condition = false (сначала run_de --stage E5)")
    return dict(t=t, num=num, leaf=leaf, e5={k: v for k, v in e5.items() if k not in ("computed_at", "x5_computed", "sha256")}, reg=R.read(), causes=IO.read_csv("DE0_CAUSES"), xsha={p.name: IO.sha_file(p) for p in x_files().values()},
                tab_sha={k: hashlib.sha256(csv_text(t[v]).encode("utf-8")).hexdigest() for k, v in NAMES.items()})


def render(ctx: dict) -> str:
    return D2.build(ctx)


def write() -> None:
    t, num, leaf, bad = compute()
    if bad:
        raise SystemExit("FIN: таблицы ≠ таблицам листов: " + "; ".join(bad))
    IO.REP.mkdir(parents=True, exist_ok=True)
    for k, v in NAMES.items():
        (IO.REP / (k + ".csv")).write_text(csv_text(t[v]), encoding="utf-8")
    add_x1_row(t)
    doc = render(context(t, num, leaf))
    if D.doc_decimals(doc) != D.recorded():
        raise SystemExit("FIN: в документе десятичные числа вне таблиц")
    doc_path().write_text(doc, encoding="utf-8")
    IO.merge_checkpoint("FIN", dict(sha256={k: IO.sha_file(IO.REP / (k + ".csv")) for k in NAMES}, sha256_doc=IO.sha_file(doc_path()), numbers_in_doc=int(sum(D.recorded().values())), leaf_checked=leaf[0], leaf_mismatch=leaf[1], written_at=R.now_text(), code=code_hash()))


def outputs_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(IO.REP.glob("FIN_*")) + [doc_path(), R.TRIALS, IO.CP_FILE]:
        h.update(f.name.encode() + (IO.sha_file(f).encode() if f.exists() else b"-"))
    return h.hexdigest()


def registry_bad(reg: pd.DataFrame) -> list:
    reg_ids = set(reg.loc[reg["status"] == "зарегистрирована", "id"])
    done, not_c = set(reg.loc[reg["status"] == "посчитана", "id"]), set(reg.loc[reg["status"] == "не считается", "id"])
    return [] if reg_ids == {"X1", "X2", "X3", "X4", "X5"} and done == {"X1", "X2", "X3", "X4"} and not_c == {"X5"} else ["реестр: ожидалось 5 мест, 4 посчитаны, X5 «не считается»"]


def check() -> list:
    """Только читает. -> список расхождений (пусто = 0 расхождений)."""
    before, bad, cp = outputs_hash(), [], IO.read_cp()
    fin = cp.get("FIN")
    if not fin:
        return ["нет ключа FIN в CHECKPOINT"]
    bad += S5.check()
    t, num, leaf, lbad = compute()
    bad += lbad + registry_bad(R.read())
    for k, v in NAMES.items():
        p = IO.REP / (k + ".csv")
        if not p.exists() or p.read_text(encoding="utf-8") != csv_text(t[v]):
            bad.append("%s.csv ≠ пересчёту" % k)
        elif fin["sha256"].get(k) != IO.sha_file(p):
            bad.append("%s.csv: sha256 ≠ CHECKPOINT FIN" % k)
    doc = render(context(t, num, leaf))
    if not doc_path().exists() or doc_path().read_text(encoding="utf-8") != doc:
        bad.append("документ ≠ генератору")
    elif fin["sha256_doc"] != IO.sha_file(doc_path()):
        bad.append("документ: sha256 ≠ CHECKPOINT FIN")
    if D.doc_decimals(doc) != D.recorded():
        bad.append("в документе десятичные числа вне таблиц")
    lines = doc.splitlines()
    if lines[0] != D.first_line(t, num) or lines[4] != D.STATUS or "bps" in doc.lower():
        bad.append("первая строка / шапка статуса / bps")
    heads = [ln for ln in lines if ln.startswith("## ")]
    bad += ["нет раздела %d «%s»" % (i + 1, p) for i, p in enumerate(D2.SECTIONS) if len(heads) <= i or p not in heads[i]]
    if outputs_hash() != before:
        bad.append("--check изменил выходы")
    return bad
