"""HAR-J Hybrid V6, лист Z3 (Ж-230): причины, почему гибриды не обгоняют K2/K3 устойчиво. Запуск из корня репозитория, процесс один, обучения нет:
  python3 -W ignore -m models_v2.s13.run_v6_why           # пересчитать и записать WHY_*.csv, CHECKPOINT_WHY.json, HARJ_SBER_V6_PRICHINY.md
  python3 -W ignore -m models_v2.s13.run_v6_why --check   # ТОЛЬКО ЧТЕНИЕ: пересчёт в памяти, сверка с диском и с FIN, хэш выходов до и после"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from models_v2.s13 import v6_0_io as IO, v6_why_check as CK, v6_why_data as DT, v6_why_doc as W0, v6_why_doc2 as W, v6_why_docheck as DC, v6_why_stage as ST


def evidence(r: dict, own_before: dict, own_after: dict) -> list:
    t, d = r["t"], r["d"]
    y, g, z = t["WHY_YEARS"], t["WHY_REGIME"], t["WHY_TAILS"]
    mx = lambda df, a, b: float(np.abs(df[a] - df[b]).max())
    ident = max(mx(y[y.year == "total"], "delta", "contrib"), mx(g[g.group == "total"], "delta", "contrib"), float(np.abs(z.delta - z.contrib_top - z.contrib_rest).max()))
    doc = W0.load_tables()
    n = DC.count_numbers(doc)
    n_ok, bad = DC.compare(DT.DOC.read_text(encoding="utf-8"), W.build(doc)) if DT.DOC.exists() else (0, -1)
    return ["G65: семь диагностик: %s; дат %d; конфигураций 0; импорт обучателей в v6_why_*: %d нарушений" % ({k: len(v) for k, v in t.items()}, len(d["dates"]), len(CK.no_trainers())),
            "G66: терцили K1 PIT — пороги по датам ≤ d, первые %d дат без группы; усечение ряда меток не меняет: %s" % (CK.A.MIN_HIST, "да" if not CK.pit(d) else "НЕТ"),
            "G67: макс. |Δ − сумма вкладов| по годам, топ-дням и группам = %.2e (допуск %.0e); пар %d" % (ident, CK.ID_TOL, len(DT.PAIRS)),
            "G68: документ %s, десятичных чисел сверено %d, расхождений %d; статических чисел вне таблиц: %d; собственные выходы до = после: %s" % (DT.DOC.name, n, bad, len(DC.static_decimals(doc)), own_before == own_after)]


def main(check: bool) -> int:
    before, own_before = IO.outputs_digest(), ST.own_digest()
    r = ST.compute()
    if not check:
        ST.write(r)
        print("лист Z3 записан; собственные выходы %s; счёт %.1f с" % (ST.own_digest(), r["seconds"]))
        return 0
    bad = CK.all_bad(r)
    own_after, after = ST.own_digest(), IO.outputs_digest()
    if own_before != own_after:
        bad.append("--check изменил собственные файлы: %s → %s" % (own_before, own_after))
    try:
        print("\n".join(evidence(r, own_before, own_after)))
    except (KeyError, ValueError) as e:                                 # таблицы на диске испорчены: причина уже в списке bad
        print("evidence недоступна: %s" % e)
    print("счёт одного полного прогона: %.1f с" % r["seconds"])
    print("хэш всех выходов V6 до = после: %s (%s); при расхождении без изменения собственных файлов — запись параллельных листов (Z1/Z2)" % (before == after, after[:16]))
    print("\n".join(bad) if bad else "G65–G68: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    sys.exit(main(ap.parse_args().check))
