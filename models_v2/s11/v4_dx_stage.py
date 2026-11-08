"""V4-DX: стадия `DX` (G26). Диагностика недостатка информации по годам (Ж-225 п. 7) — ЗАДНИМ ЧИСЛОМ, описательно: DX_YEARS, DX_FEATURES, DX_2018, DX_REPORT. Конфигураций 0; прогнозы не создаются и не меняются (только чтение K1–K3, Z, C1, C8)."""
from __future__ import annotations

import datetime as dt
import json
import time

from models_v2.s11 import v4_b_cache as BC, v4_b_decl as DECL, v4_dx_2018 as D18, v4_dx_feat as DF, v4_dx_report as RP, v4_dx_years as DY, v4_io as IO

DX_RULE = ("лист DX (Ж-225 п. 7): диагностика по году цели 2017–2026 × группе (M16, SBER) × M ∈ {√(K2·K3), K2, K3}, H = K1; пункты 1–6 — DX_YEARS, пункт 7 — DX_FEATURES (квинтили по прошлому окну), разбор 2018 — DX_2018; "
           "задним числом, описательно, 0 конфигураций, в выбор не входит")


def declare() -> dict:
    cp = json.loads(DECL.CP.read_text())
    if "DX_declared_at" not in cp:
        DECL.update(DX_declared_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), DX_rule=DX_RULE, DX_configs_spent=0)
        cp = json.loads(DECL.CP.read_text())
    return cp


def stage_dx() -> dict:
    t0 = time.time()
    cp = declare()
    P = DY.panel_rows(IO.T17)
    Y = DY.years_table(P)
    F = DF.features_table(DF.frame(P, DF.load_z(IO.T17)))
    T = D18.table(P)
    ins = BC.digest(BC.input_files() + [BC.OUT / ("Z_%s.parquet" % t) for t in IO.T17] + [BC.OUT / "C8.parquet", BC.OUT / "C1.parquet"])
    code = BC.digest(sorted(BC.CODE_DIR.glob("v4_dx_*.py")))
    note = "ЗАДНИМ ЧИСЛОМ, описательно, не в выбор; входы sha256=%s…, код sha256=%s…" % (ins[:12], code[:12])
    IO.write_csv(Y, "DX_YEARS.csv", "пункты 1–6 по году цели × группе × M (H = K1, M = √(K2·K3) | K2 | K3); D = QLIKE(H) − QLIKE(M); вес HAR-J и 90 %% интервал — блочный бутстрап дат (300 выборок); %s" % note)
    IO.write_csv(F, "DX_FEATURES.csv", "пункт 7: условные средние D по квинтилям признаков Z (границы — прошлое окно до 1 января года; 2017 без квинтилей), post_* — 28 дней после излома; M16, M = √(K2·K3); " + note)
    IO.write_csv(T, "DX_2018.csv", "разбор 2018: вклад месяцев и бумаг в проигрыш C8 и C1 к K1, путь веса против лучшего постоянного веса задним числом (M16 пулом); " + note)
    r = {"Y": Y, "F": F, "T": T, "cp": cp, "ins": ins, "code": code}
    (IO.REP / "DX_REPORT.md").write_text(RP.build(Y, F, T))
    DECL.update(DX_runtime_sec=round(time.time() - t0, 1), DX_done=True)
    return r
