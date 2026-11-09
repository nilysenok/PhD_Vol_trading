"""V7-DX: стадия `python3 -W ignore -m models_v2.s14.run_v7 --stage DX [--check]` (G37–G41). Диагностика: конфигураций 0, прогнозов вне выборки и выборов по данным 2022+ нет; сохранённые прогнозы 1, 4, 6, 1–7 и ключи CHECKPOINT `1b`, `2` не меняются.
Без `--check`: пишет `DX_*.csv` (статус «диагностика, не отбор» в шапке), `DX_REPORT.md` (текст из таблиц), ключ `DX` в CHECKPOINT.json (числа, выводы, хэш кода, хэш ключей `1b` и `2` на момент записи) и дописывает строки результата 2b в `ALL_TRIALS.csv`.
С `--check`: пересчитывает всё в памяти и сверяет с файлами и CHECKPOINT; ничего не пишет (хэш выходов до и после). Также здесь `write_2b` — запись стадии 2 (`v7_2_evrun.write_eval` + строки 2b), которую вызывает `run_v7 --stage 2` без `--check`."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_io as IO, v7_1b_cfgrun as CR, v7_1m_reg as RG
from models_v2.s14 import v7_dx_core as K, v7_dx_dup as DUP, v7_dx_enc as ENC, v7_dx_reg as RGU, v7_dx_rep as REP, v7_dx_sig as SIG

KEYS = ("ENC", "ENC_STAB", "SIG", "SIG_EXPOST", "LAM", "G", "LAMSWEEP", "PCORR", "PIT", "GAP")
NOTES = {"ENC": "G37: задача-предлог на отложенных 10 %% против наивных, ранг z; по kind, году, seed", "ENC_STAB": "G37: стабильность z между seeds одного года (выравнивание перестановкой)",
         "SIG": "G38: снижение QLIKE к H в выборке (последняя переоценка года) против вне выборки (следующий год)", "SIG_EXPOST": "G38: верхняя оценка ЗАДНИМ ЧИСЛОМ, ex post, не для отбора: гребень, 5 блоков",
         "LAM": "G39: частоты выбранных λ_g, λ_LB", "G": "G39: распределение |G| по годам, граница tanh 0,9", "LAMSWEEP": "G39: в выборке, снижение QLIKE при λ_g ∈ {0; 0,01; 0,1; 1}, 3 даты, конфигурации 1 и 4",
         "PCORR": "G40: частные корреляции z и G с ln H, ℓ, последними ln(RV/K2), ln(RV/H); по годам и seeds", "PIT": "G40: R² вне выборки PIT-регрессии ln(RV/K2) на z по годам t⁺, по seeds",
         "GAP": "G40: разрыв конфигураций к K2 по годам, вклад года и доля 2022"}
REPORT = "DX_REPORT.md"


def name(k: str) -> str:
    return "DX_%s.csv" % k


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("v7_dx_*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def protected_hash(cp: dict) -> str:
    """Хэш ключей `1b`, `2` CHECKPOINT (их эта стадия не меняет)."""
    return hashlib.sha256(json.dumps({k: cp[k] for k in ("1b", "2")}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def compute() -> dict:
    inp, cp = K.inputs(), CR.read_cp()
    reps = {c: K.replay(c, inp) for c in K.CFGS}
    bad = [c for c in K.CFGS if not K.same_as_saved(c, reps[c])]
    if bad:
        raise SystemExit("повтор конфигураций %s не равен сохранённому прогнозу побитно: диагностика не принимается" % bad)
    rts = {c: K.refit_table(c, reps[c]) for c in K.CFGS}
    t = {}
    t.update(ENC.tables(ENC.load_pack()))
    t.update(SIG.tables(inp, rts))
    t.update(RGU.tables(reps, rts, cp["2"]))
    t.update(DUP.tables(inp))
    return {k: roundtrip(v) for k, v in t.items()}                      # отчёт и числа — из табличного вида (как в CSV), не из объектов памяти


def roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(df.to_csv(index=False, lineterminator="\n")), dtype={"cfg": str, "var": str, "kind": str})


def same(a: pd.DataFrame, b: pd.DataFrame, rtol: float = 1e-9) -> bool:
    a, b = roundtrip(a), b.reset_index(drop=True)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        if a[c].dtype.kind in "fi" and b[c].dtype.kind in "fi":
            if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=rtol, atol=1e-12, equal_nan=True):
                return False
        elif not a[c].astype(str).equals(b[c].astype(str)):
            return False
    return True


def record(t: dict, cp: dict) -> dict:
    n = REP.numbers(t)
    return dict(numbers=n, verdicts=REP.verdicts(n), code=code_hash(), protected_1b_2_sha256=protected_hash(cp), status=K.STATUS, tables=list(KEYS), n_configurations=0, new_forecasts=0)


def add_2b_rows(cp: dict, path: Path | None = None) -> list:
    """Строки результата 2b (суффикс `RG.R2B_SUFFIX`) для конфигураций 1–7 из CHECKPOINT `2.eval`/`2.rules`; повтор — «уже есть»; строки регистрации не трогаются."""
    ev, ru, out = cp["2"]["eval"], cp["2"]["rules"], []
    for c in "1234567":
        k = "cfg" + c
        out.append(RG.add_result(c, dict(stage="2b", qlike=ev["qlike"][k], n_dates=ev["n_dates"], first=ev["first"], last=ev["last"], go=ru[k]["go"], status=ru[k]["status"]), path=path, suffix=RG.R2B_SUFFIX))
    return out


def write_2b() -> int:
    """Запись стадии 2: оценка 1–7 (`v7_2_evrun.write_eval`), затем строки результата 2b в реестр."""
    from models_v2.s14 import v7_2_evrun as EVR
    rc = EVR.write_eval()
    if rc == 0:
        print("реестр, строки 2b: %s" % add_2b_rows(CR.read_cp()))
    return rc


def write() -> int:
    t = compute()
    cp = CR.read_cp()
    if "eval" not in cp.get("2", {}):
        raise SystemExit("CHECKPOINT 2.eval нет: стадия 2 не записана")
    for k in KEYS:
        IO.write_csv(t[k], name(k), NOTES[k].replace("%%", "%"), status=K.STATUS)
    (IO.REP / REPORT).write_text(REP.render(t, K.STATUS), encoding="utf-8")
    IO.merge_checkpoint("DX", record(t, cp))
    print("DX: записано %d таблиц, %s, ключ DX; строки 2b: %s" % (len(KEYS), REPORT, add_2b_rows(CR.read_cp())))
    print(REP.head_line(REP.verdicts(REP.numbers(t))))
    return 0


def check_rows_2b(cp: dict) -> list:
    rows = {r[0] for r in RG.read()[1:]}
    return ["ALL_TRIALS: нет строки 2b %s" % c for c in "1234567" if "Ж-233:%s%s" % (c, RG.R2B_SUFFIX) not in rows] + RG.bad()


def check_dx(cp: dict) -> list:
    """-> замечания (пусто = ок). Ничего не пишет."""
    if "DX" not in cp:
        return ["CHECKPOINT: нет ключа DX"]
    t, bad = compute(), []
    rec = record(t, cp)
    for f in ("code", "protected_1b_2_sha256"):
        if rec[f] != cp["DX"][f]:
            bad.append("DX.%s ≠ записанному (код или ключи 1b/2 изменились)" % f)
    if json.loads(json.dumps(rec["verdicts"], ensure_ascii=False)) != cp["DX"]["verdicts"]:
        bad.append("DX.verdicts: пересчёт ≠ CHECKPOINT")
    if not np.allclose(*[np.array([v for _, v in sorted(_flat(d))]) for d in (rec["numbers"], cp["DX"]["numbers"])], rtol=1e-9, atol=1e-12, equal_nan=True):
        bad.append("DX.numbers: пересчёт ≠ CHECKPOINT")
    for k in KEYS:
        p = IO.REP / name(k)
        if not p.exists() or not same(t[k], IO.read_csv(name(k))):
            bad.append("%s: нет или не совпал с пересчётом" % name(k))
    p = IO.REP / REPORT
    if not p.exists() or p.read_text(encoding="utf-8") != REP.render(t, K.STATUS):
        bad.append("%s: нет или не равен тексту из таблиц" % REPORT)
    return bad + check_rows_2b(cp)


def _flat(d: dict, pre: str = ""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, pre + k + ".")
        else:
            yield pre + k, float(v)


def run(check: bool) -> int:
    if not check:
        return write()
    before = IO.outputs_digest()
    bad = check_dx(CR.read_cp())
    if IO.outputs_digest() != before:
        bad.append("--check изменил выходы (хэш до ≠ после)")
    print("\n".join(bad) if bad else "OK: DX таблицы, отчёт и ключ DX равны пересчёту; повтор 1, 4, 6 побитно равен сохранённым; ключи 1b/2 не изменены; строки 2b в реестре; выходы не изменены (хэш %s)" % before[:16])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run("--check" in sys.argv))
