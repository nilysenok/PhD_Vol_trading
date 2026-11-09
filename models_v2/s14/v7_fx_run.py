"""V7-FX: пересчёт версии 2 (согласованные координаты z) на тех же местах бюджета 1–7: `python3 -W ignore -m models_v2.s14.v7_fx_run`. Порядок:
 архив версии 1 (`v7_fx_arch`) → E0 (v2: по z того же энкодера года Y; в версии 1 E0 УЖЕ считался так — сверка с архивом) → конфигурации 1 и 2 → выбор энкодера для 3 тем же правилом (`1b.choice3` v2, рядом `1b.choice3_v1`, ДО счёта 3)
 → конфигурация 3 (при другом энкодере — окно 1 обучается протоколом Stage A) → конфигурации 4–7 (`v7_2_run.write_cfgs`) → строки версии 2 в реестре и CHECKPOINT `fx`. Только QLIKE 2018–2021: контролей, p и окон 2022+ нет (лист 2b).
Возобновляемо: посчитанное и сошедшееся по sha256 пропускается. Решётки, seeds, λ, первая оцениваемая дата — без изменений."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_io as IO, v7_1b_cfg as CFG, v7_1b_cfg3 as C3, v7_1b_cfgrun as CR, v7_1b_e0 as EZ, v7_1b_stage as S1B, v7_1m_reg as RG, v7_2_run as S2, v7_fx_arch as AR, v7_fx_diag as DG, v7_fx_warm as WM


def e0_step() -> dict:
    r = EZ.run()
    v1 = CR.read_cp()["1b_v1"]["E0"]
    diff = {k: float(np.max(np.abs(np.array(r[k]["r2_mean"]) - np.array(v1[k]["r2_mean"])))) for k in CFG.KINDS}
    S1B.update_1b("E0", S1B.e0_record(r))
    IO.write_csv(r["table"], "1b_E0.csv", "G23 (версия 2, лист FX): частный R² компонент z на (S, Σ, ln H, ℓ), окно 2018–2021 по годам энкодера, потолок")
    return dict(r2_mean={k: r[k]["r2_mean"] for k in CFG.KINDS}, both_no_new_info=r["both_no_new_info"], max_abs_diff_vs_v1=diff)


def cfg12(tab: pd.DataFrame, warm: pd.DataFrame) -> dict:
    dfs = {}
    for k in CFG.KINDS:
        t0 = time.time()
        dfs[k] = CFG.run_kind(k, tab, warm=warm)
        CR.update_1b("cfg" + CFG.CFG_ID[k], CR.save(k, dfs[k], time.time() - t0))
        print("конфигурация %s (%s, v2): %d строк, %.0f с" % (CFG.CFG_ID[k], k, len(dfs[k]), time.time() - t0))
    ch = CFG.choose(dfs)
    CR.update_1b("choice3_v1", CR.read_cp()["1b_v1"]["choice3"])
    CR.update_1b("choice3", dict(ch, rule=RG.ENC_RULE, written_before_cfg3=True, cfg3_computed=False, version=2))     # ДО счёта конфигурации 3
    tbl = pd.DataFrame([dict(config=CFG.CFG_ID[k], encoder=k, qlike_2018_2021=ch["qlike"][k], n_dates=ch["n_dates"], chosen_for_3=(k == ch["choice"])) for k in CFG.KINDS])
    IO.write_csv(tbl, "1b_B_QLIKE.csv", "1b-B (версия 2, лист FX): средняя QLIKE на датах целей 2018–2021 (общие даты), выбор энкодера для конфигурации 3", status=CR.STATUS_1B)
    return ch


def cfg3_step(tab: pd.DataFrame, warm: pd.DataFrame) -> None:
    import shutil
    cp = CR.read_cp()
    kind, t0 = C3.kind3(cp), time.time()
    if shutil.disk_usage(IO.ROOT).free / 1e9 < 40:
        raise SystemExit("свободно < 40 ГБ: запись отменена")
    C3.train_missing3(kind)
    CR.update_1b("stageA_w1", dict(encoder=kind, window=C3.W3, years=list(C3.STA.YEARS), seeds=list(C3.STA.SEEDS), hashes=C3.hashes3(kind), note="конфигурация 3: энкодер окна 1 день, бары < 01.01.Y"))
    df = CFG.run_kind(kind, tab, window=C3.W3, warm=warm)
    CR.update_1b("cfg3", C3.save3(df, kind, time.time() - t0))
    print("конфигурация 3 (%s, w1, v2): %d строк, %.0f с" % (kind, len(df), time.time() - t0))


def register_v2() -> dict:
    cp, q = CR.read_cp(), {}
    for cfg in "1234567":
        rec = (cp["1b"] if cfg in "123" else cp["2"])["cfg" + cfg]
        RG.add_result(cfg, dict(stage="FX-v2", qlike_2018_2021=rec["qlike_2018_2021"], sha256_content=rec["sha256_content"], n_rows=rec["n_rows"]), suffix=RG.V2_SUFFIX)
        old = (cp["1b_v1"] if cfg in "123" else cp["2_v1"])["cfg" + cfg]
        q[cfg] = dict(v1=old["qlike_2018_2021"], v2=rec["qlike_2018_2021"], sha256_v2=rec["sha256_content"])
    return q


def main() -> int:
    import shutil
    from models_v2.s14 import v7_1m_data as DATA
    if shutil.disk_usage(IO.ROOT).free / 1e9 < 40:
        raise SystemExit("свободно < 40 ГБ: запись отменена")
    print("архив v1:", AR.archive())
    tab, warm = DATA.load_tab(), WM.load_warm()
    e0 = e0_step()
    print("E0 v2 против v1: макс. |Δ R²| %s" % e0["max_abs_diff_vs_v1"])
    ch = cfg12(tab, warm)
    print("выбор для 3: v1 %s, v2 %s" % (CR.read_cp()["1b_v1"]["choice3"]["choice"], ch["choice"]))
    cfg3_step(tab, warm)
    S2.write_cfgs()
    q = register_v2()
    d = DG.rows()
    IO.merge_checkpoint("fx", dict(e0=e0, qlike_2018_2021=q, choice3_v1=CR.read_cp()["1b_v1"]["choice3"]["choice"], choice3_v2=ch["choice"], diag_median_corr_same={"%s_%d" % (k, c): float(g["corr_same"].median()) for (k, c), g in d.groupby(["kind", "comp"])},
                                  warm_rows=WM.N_WARM, written_at=RG.now(), note="лист FX: версия 2 посчитана; сравнений с контролями, p и окон 2022+ нет (лист 2b)"))
    print("QLIKE 2018–2021 v1 → v2: %s" % {c: (round(v["v1"], 6), round(v["v2"], 6)) for c, v in q.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
