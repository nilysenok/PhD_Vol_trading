"""V5 лист FIN: запуск стадии FIN (из `run_v5 --stage FIN`). Процесс один.
  (без флагов)  замерить время стадий, собрать FIN_*.csv, ключ FIN в CHECKPOINT, HARJ_HYBRID_V5_FINAL.md, REPRODUCTION_V5.md (конфигурации не расходуются)
  --check       ТОЛЬКО ЧТЕНИЕ: пересчёт в памяти, сверка с диском, с таблицами листов 0/1a/2, итогового документа; суммарный хэш выходов до и после"""
from __future__ import annotations

from models_v2.s12 import v5_0_io as IO, v5_fin_check as CK, v5_fin_data as D, v5_fin_doc as W, v5_fin_docheck as DC, v5_fin_stage as ST


def evidence(r: dict, t: dict) -> list:
    g, a = W.go_row(t), W.m(t)
    best = [(x, g.loc[x, "crit_panel"], g.loc[x, "crit_no2017"], g.loc[x, "crit_trim1"], g.loc[x, "go"]) for x in (t["gs"], t["sp"])]
    n, bad = DC.compare(DC.B.FINAL.read_text(encoding="utf-8"), DC.A.build(t))
    cp = t["cp"]["FIN"]
    return ["G23: таблиц FIN %d; строк FIN_ALL %d, FIN_TICKERS %d, FIN_JOINT %d, FIN_ABLATION %d; кэш прогнозов 9 из 9: %s" % (len(D.FIN_TABLES), len(r["t"]["FIN_ALL"]), len(r["t"]["FIN_TICKERS"]), len(r["t"]["FIN_JOINT"]), len(r["t"]["FIN_ABLATION"]), all(r["d"]["cached"].values())),
            "G23: лучшие %s; K8 crit_panel %.6f; EqY K8 %.6f" % ("; ".join("%s %.6f / %.6f / %.6f go %s" % x for x in best), a.loc[("K8", "all"), "crit_panel"], a.loc[("K8", "all"), "EqY"]),
            "G24: чисел в документе сверено %d, расхождений %d; go прошли %d из 9" % (n, bad, W.n_go(t)),
            "G25: used_total %s из %s; N_declared %s, N_wide %s; закрытых направлений %d" % (cp["used_total"], cp["budget"], cp["N_declared_before_stats"], cp["N_wide"], len(cp["closed_directions"]))]


def main(a) -> int:
    before = IO.outputs_digest()
    if not a.check:
        r = ST.compute(measure=True)
        ST.write(r)
        print("стадия FIN записана; хэш выходов %s; конфигураций израсходовано 0" % IO.outputs_digest())
        return 0
    r = ST.compute(measure=False)
    bad = CK.disk_vs_memory(r) + CK.cross(r) + CK.registry(r)
    if not all(r["d"]["cached"].values()):
        bad.append("кэш прогнозов GS/SP отсутствует или устарел")
    t = W.load()
    bad += DC.check_doc(t) + DC.check_static(t)
    after = IO.outputs_digest()
    if before != after:
        bad.append("--check изменил файлы: хэш до %s, после %s" % (before, after))
    print("\n".join(evidence(r, t)))
    print("хэш выходов до = после: %s (%s)" % (before == after, after[:16]))
    print("\n".join(bad) if bad else "G23–G25: все проверки пройдены")
    print("ИТОГ:", "FAIL" if bad else "OK")
    return 1 if bad else 0
