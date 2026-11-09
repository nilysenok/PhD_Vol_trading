"""HAR-J Hybrid V6, лист Z2 (Ж-230): сравнение SBER (h = 1) по программам V1–V6 и точный лучший результат. Запуск из корня репозитория, процесс один:
  python3 -W ignore -m models_v2.s13.run_v6_cmp            # пересчитать из СОХРАНЁННЫХ прогнозов и записать cmp/*.csv, CHECKPOINT_CMP.json, HARJ_SBER_V1_V6_SRAVNENIE.md
  python3 -W ignore -m models_v2.s13.run_v6_cmp --check    # ТОЛЬКО ЧТЕНИЕ: пересчёт в памяти, сверка с диском (1e-9), документ против пересборки, хэш выходов до и после
Обучения нет, конфигурации не расходуются (0)."""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys

import pandas as pd

from models_v2.s13 import v6_0_io as IO, v6_cmp_best as B, v6_cmp_doc2 as D2, v6_cmp_io as C, v6_cmp_repro as P, v6_cmp_reg as R, v6_cmp_tab as T

NAMES = {"CMP_SERIES": "SERIES", "CMP_REPRO": "REPRO", "CMP_YEARS": "YEARS", "CMP_WINDOWS": "WINDOWS", "CMP_TESTS": "TESTS", "CMP_MCS": "MCS", "CMP_BEST": "BEST"}
RULES = "правила Z2 п. 1–7 (docs/zadaniya/2026-11-09_harj_v6_Z2.md): общие даты; класс как в PRIOR_SBER; (а) финалист, (б) лучший PIT по QLIKE 2017–2021; лучший по окнам среди PIT; статистика V6 FIN; годы; воспроизведение 1e-9"


def compute() -> dict:
    u = R.universe()
    reg = R.build_registry(u)
    labs = T.labels(reg)
    mc = T.mcs(u, reg)
    return dict(SERIES=reg, REPRO=P.repro(u, reg), YEARS=T.years(u, labs), WINDOWS=T.windows(u, labs), TESTS=T.tests(u, reg), MCS=mc, BEST=B.best(u, reg, mc), COV=C.coverage())


def roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(df.to_csv(index=False, lineterminator="\n")))


def checkpoint(t: dict) -> dict:
    q = t["REPRO"]
    q = q[q["kind"] == "qlike"]
    return {"program": "HAR-J Hybrid V6 (Ж-227/Ж-230), SBER, h=1, лист Z2", "status": C.STATUS, "rules": RULES, "N": 722, "N_wide": 3122, "used": 0, "coverage": t["COV"].to_dict("records"), "n_series_rows": len(t["SERIES"]),
            "repro_rows": int((q["status"] != "нет числа в таблицах").sum()), "repro_unexplained": len(P.unexplained(t["REPRO"])), "code_hash": C.code_hash(), "inputs_hash": IO.inputs_hash(),
            "tables_sha": {n: IO.sha_file(C.CMP / (n + ".csv")) for n in C.TABLES}, "note": "лист Z2: конфигурации не расходуются; обучения нет"}


def stable(d: dict) -> dict:
    return json.loads(json.dumps({k: v for k, v in d.items() if k != "written_at"}))


def write(t: dict) -> None:
    for n, k in NAMES.items():
        C.write_csv(t[k], n)
    cp = checkpoint(t)
    old = json.loads(C.CP.read_text()) if C.CP.exists() else None
    cp["written_at"] = old["written_at"] if old and stable(old) == stable(cp) else dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    C.CP.write_text(json.dumps(cp, ensure_ascii=False, indent=1), encoding="utf-8")
    C.DOC.write_text(D2.build(t), encoding="utf-8")


def load_disk() -> dict:
    t = {k: C.read_csv(n) for n, k in NAMES.items()}
    t["COV"] = pd.DataFrame(json.loads(C.CP.read_text())["coverage"])
    return t


def compare_tables(fresh: dict) -> list:
    bad = []
    for n, k in NAMES.items():
        try:
            pd.testing.assert_frame_equal(roundtrip(fresh[k]), C.read_csv(n), check_dtype=False, atol=1e-9, rtol=0)
        except (AssertionError, FileNotFoundError) as e:
            bad.append("%s: диск ≠ пересчёту (%s)" % (n, str(e).splitlines()[0][:120]))
        head = (C.CMP / (n + ".csv")).read_text(encoding="utf-8").splitlines()[0] + "\n" if (C.CMP / (n + ".csv")).exists() else ""
        if head != C.header():
            bad.append("%s: шапка (хэши кода/входов) ≠ текущим" % n)
    return bad


def check_all(fresh: dict) -> list:
    bad = compare_tables(fresh)
    if not C.CP.exists() or stable(json.loads(C.CP.read_text())) != stable(checkpoint(fresh)):
        bad.append("CHECKPOINT_CMP.json ≠ пересчёту")
    doc = C.DOC.read_text(encoding="utf-8") if C.DOC.exists() else ""
    n, nbad = D2.compare(doc, D2.build(fresh))
    if nbad or doc != D2.build(fresh):
        bad.append("документ ≠ пересборке из таблиц (чисел %d, расхождений %d)" % (n, nbad))
    un = P.unexplained(fresh["REPRO"])
    if len(un):
        bad.append("G61: необъяснённых расхождений %d: %s" % (len(un), un["series"].tolist()[:5]))
    return bad


def evidence(t: dict) -> list:
    b, r = t["BEST"], t["REPRO"]
    b = b[b["set"].str.startswith("основное (") & (b["ref_kind"] == "второй")]
    q = r[r["kind"] == "qlike"]
    out = ["G60: рядов в реестре %d строк / %d уникальных; классы %s" % (len(t["SERIES"]), t["SERIES"]["series"].nunique(), t["SERIES"].drop_duplicates("series")["cls"].value_counts().to_dict()),
           "G61: чисел %d, совпало %d, необъяснённых %d, нет числа в таблицах %d, max|Δ| %.2e" % (len(q), (q["status"] == "совпало").sum(), len(P.unexplained(r)), (q["status"] == "нет числа в таблицах").sum(), q["abs_diff"].max())]
    out += ["G63: окно %s: %s QLIKE %.6f; второй %s %.6f; Δ %+.3f [%+.3f; %+.3f] p %.3f; MCS %d из %d (%s)" % (x["window"], x["best"], x["best_qlike"], x["ref"], x["ref_qlike"], x["delta_pct"], x["lo90"], x["hi90"], x["p_one"], x["mcs_size"], x["n_series"], x["verdict"][:70]) for _, x in b.iterrows()]
    return out


def main_check() -> int:
    before = C.digest()
    fresh = compute()
    bad = check_all(fresh)
    after = C.digest()
    if before != after:
        bad.append("--check изменил файлы: хэш до %s, после %s" % (before[:16], after[:16]))
    print("\n".join(evidence(fresh)))
    n, nbad = D2.compare(C.DOC.read_text(encoding="utf-8") if C.DOC.exists() else "", D2.build(fresh))
    print("G64: документ = пересборка из таблиц: сверено чисел %d, расхождений %d; разделов %d" % (n, nbad, len(D2.PARTS)))
    print("хэш выходов до = после: %s (%s)" % (before == after, after[:16]))
    print("\n".join(bad) if bad else "G60–G64: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return main_check()
    t = compute()
    write(t)
    print("\n".join(evidence(t)))
    print("записано; хэш выходов %s" % C.digest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
