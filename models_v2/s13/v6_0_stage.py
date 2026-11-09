"""V6-0: стадия 0 = пересчёт всех таблиц листа в памяти (`compute`, чистая функция от входов), запись (`write`, не в `--check`) и сверка с диском (`check`, только чтение). Моделей не обучает, конфигурации не расходуются."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from models_v2.s11 import v4_fin_check as FC
from models_v2.s13 import v6_0_best as BE, v6_0_cp as CP, v6_0_crit as CR, v6_0_ctrl as CT, v6_0_io as IO, v6_0_load as LD, v6_0_map as MP, v6_0_pass as PA, v6_0_prior as PR, v6_0_resid as RE

TABLES = {"CONTROLS_SBER.csv": "controls", "PRIOR_SBER.csv": "prior", "BEST_SBER.csv": "best", "PASS_M10_SBER.csv": "pass", "RESID_SBER.csv": "resid", "CRIT_SBER.csv": "crit"}
NOTES = {"controls": "G1: контроли K1,K2,K3,M,K8,D2 SBER побитно прежние", "prior": "G2: прежние гибриды SBER по датам, класс сопоставимости с доказательством", "best": "G3: лучший результат SBER на общих датах",
         "pass": "G5: паспорт M10 SBER и тождество цели", "resid": "G6: остаток HAR-J e = ln(RV/H), только описательная статистика", "crit": "G7: crit_SBER, шум меры, пороги go"}
MAP = IO.REP / "MAP_DONE_V6.md"
CAUSAL_PRIOR = ("J210", "J211", "J212", "V2")


def crit_extra(res: dict) -> pd.DataFrame:
    th = res["thresholds"]
    rows = [dict(section="noise", item=k, value=v) for k, v in res["noise"].items()] + [dict(section="formula", item="max|crit_wf − формула весов V5|", value=res["formula_max_diff"])]
    for k, v in th.items():
        if isinstance(v, dict) and "threshold" in v:
            rows.append(dict(section="threshold", item=k, value=v["threshold"], note="min по go-рядам %.6f (%s); только K1,K2,K3,K8,D2: %.6f (%s)" % (v["min_over_go"], v["attained_by"], v["base5_threshold"], v["base5_attained_by"])))
    rows += [dict(section="threshold", item="cond5_best_k23_%s" % y, value=v) for y, v in th["cond5_annual_best_k23"].items()]
    return pd.DataFrame(rows)


def alt_thresholds(t: pd.DataFrame, prior: pd.DataFrame, pit: list) -> dict:
    """Чувствительность: если считать сопоставимыми и все причинные ряды Ж-210/211/212/214 (отобранные на SBER), планка ниже; в «go» не используется."""
    extra = [s for s, p, c in zip(prior["series"], prior["program"], prior["class"]) if p in CAUSAL_PRIOR and c == "потолок"]
    th = CR.thresholds(t, pit + extra)
    return {k: {"threshold": v["threshold"], "attained_by": v["attained_by"]} for k, v in th.items() if isinstance(v, dict) and "threshold" in v}


def compute() -> dict:
    w, meta = LD.load()
    ctrl, cres = CT.controls_table(w)
    prior = PR.prior_table(w, meta)
    best, bres = BE.best_table(w, prior)
    passp, pres = PA.passport(w)
    resid, rres = RE.resid_table(w)
    crit, cr = CR.crit_table(w, prior, bres)
    alt = alt_thresholds(cr["table"], prior, bres["pit"])
    crit_all = pd.concat([crit.assign(section="crit"), crit_extra(cr)], ignore_index=True).rename(columns={"class_": "class"})
    return dict(w=w, controls=ctrl, cres=cres, prior=prior, best=best, bres=bres, **{"pass": passp}, pres=pres, resid=resid, rres=rres, crit=crit_all, cr=cr, alt=alt, checkpoint=CP.build(cr, alt),
                map_text=MP.build_text(bres["met"], prior))


def write(r: dict) -> list:
    out = [IO.write_csv(r[v], f, NOTES[v]) for f, v in TABLES.items()]
    CP.write(r["checkpoint"])
    MAP.write_text(r["map_text"], encoding="utf-8")
    return out


def map_refs(text: str) -> list:
    """G4: MAP_DONE_V6.md существует, совпадает с пересчётом из BEST_SBER и все пути в обратных кавычках (docs/, models_v2/, tests/, data/) существуют."""
    if not MAP.exists():
        return ["G4: нет MAP_DONE_V6.md"]
    disk = MAP.read_text(encoding="utf-8")
    paths = re.findall(r"`((?:docs|models_v2|tests|data|state)/[^`\s]+?)(?:::[^`]*|:\d[\d,\-–]*)?`", disk)
    miss = sorted({p for p in paths if not (IO.ROOT / p.split("*")[0].rstrip("/")).exists()})
    bad = ["G4: нет файла %s" % p for p in miss] + ([] if paths else ["G4: нет ссылок"])
    return bad + ([] if disk == text else ["G4: MAP_DONE_V6.md ≠ пересчёт из метрик G3"])


def gate_bad(r: dict) -> list:
    c, bad = r["cres"], []
    if not (c["files_ok"] and c["series_ok"] and c["k8_bitwise"] and c["shift_ok"] and c["ref_ok"]):
        bad.append("G1: %s" % {k: c[k] for k in ("files_ok", "series_ok", "k8_bitwise", "shift_ok", "ref_ok")})
    p = r["prior"]
    if (p["class"].isna()).any() or (p["diff_reported"] > PR.TOL_REP).any():
        bad.append("G2: класс не задан или расхождение с отчётом > 1e-4: %s" % p.loc[p["diff_reported"] > PR.TOL_REP, "series"].tolist())
    if not (r["bres"]["common"]["n_dates"] > 0).all():
        bad.append("G3: нет общих дат")
    if r["pres"]["max_daily"] > PA.TOL_ID or r["pres"]["max_target"] > PA.TOL_ID:
        bad.append("G5: тождество Σr² = RV: %.2e / %.2e" % (r["pres"]["max_daily"], r["pres"]["max_target"]))
    wu = r["rres"]["warm"].set_index("item")["value"]
    if float(wu["max|forecast_at − сохранённый sigma2| на двух первых датах"]) > 1e-12:
        bad.append("G6: воспроизведение K1 не побитно")
    if r["cr"]["formula_max_diff"] > 1e-12:
        bad.append("G7: формула crit_wf расходится %.2e" % r["cr"]["formula_max_diff"])
    return bad


def check(r: dict) -> list:
    bad = gate_bad(r) + CP.check(r["checkpoint"]) + map_refs(r["map_text"])
    for f, v in TABLES.items():
        if not (IO.REP / f).exists():
            bad.append("%s: нет файла" % f)
        elif not FC.frames_equal(IO.read_csv(f), r[v], 1e-9):
            bad.append("%s: диск ≠ пересчёт в памяти" % f)
    return bad
