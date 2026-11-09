"""V6 лист FIN: строки для EVIDENCE G50–G54 (числа из пересчёта и таблиц на диске)."""
from __future__ import annotations

from models_v2.s13 import v6_0_cp as CP0, v6_fin_asm as A, v6_fin_data as D, v6_fin_doc as W, v6_fin_docheck as DC


def evidence(r: dict) -> list:
    t, d = W.load(), r["d"]
    b17, ba = W.best(t, "2017_2021"), W.best(t, "all")
    cell = lambda c, w, b: W.iv(W.tt(t, c, b, w))
    mc = t["mcs"]
    mcs = {w: sorted(mc[(mc["window"] == w) & mc["in_mcs90"].astype(str).str.lower().eq("true")]["model"]) for w in ("2017_2021", "all")}
    n, bad = DC.compare(A.FINAL.read_text(encoding="utf-8"), A.build(t))
    cp = CP0.read()["FIN"]
    return ["G50: рядов %d (контролей %d, конфигураций %d); строк FIN_ALL %d, FIN_TESTS %d, FIN_STRESS %d, FIN_ENCOMPASS %d, FIN_INFO_ALL %d, FIN_ABLATION %d; дат %d" % (
                len(D.SER), len(D.CTRL), len(D.CFG), len(r["t"]["FIN_ALL"]), len(r["t"]["FIN_TESTS"]), len(r["t"]["FIN_STRESS"]), len(r["t"]["FIN_ENCOMPASS"]), len(r["t"]["FIN_INFO_ALL"]), len(r["t"]["FIN_ABLATION"]), len(d["w"])),
            "G50: лучшая 2017–2021 %s (QLIKE %.6f): к K3 %s, к K8 %s, к NTP2 %s; лучшая вся история %s (QLIKE %.6f): к K3 %s, к K8 %s, к NTP2 %s" % (
                b17, W.q(t, b17, "2017_2021"), cell(b17, "2017_2021", "K3"), cell(b17, "2017_2021", "K8"), cell(b17, "2017_2021", "V4:NTP2"), ba, W.q(t, ba, "all"), cell(ba, "all", "K3"), cell(ba, "all", "K8"), cell(ba, "all", "V4:NTP2")),
            "G51: MCS 90 %%: 2017–2021 — %d рядов %s; вся история — %d рядов %s" % (len(mcs["2017_2021"]), mcs["2017_2021"], len(mcs["all"]), mcs["all"]),
            "G52: разделов документа %d; чисел в документе сверено %d, расхождений %d; первая строка: %s" % (len(DC.PARTS), n, bad, W.first_line(t)[:160]),
            "G53: ключ FIN: израсходовано 0, used_total %s из %s, tables_sha %s; ALL_TRIALS — 18 строк" % (cp["used_total"], cp["budget"], cp["tables_sha"]),
            "G54: уровни 1–5: %d строк; уровень 5 «нет» у %d из %d" % (len(t["levels"]), int((t["levels"][t["levels"]["level"] == 5]["passed"] == "нет").sum()), len(D.CFG))]
