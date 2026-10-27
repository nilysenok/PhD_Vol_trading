"""Конвейер Ж-043, итог: ворота для кандидатов пула плато + дозапись в реестр отбора A (только дописывать).

Ворота (Ж-044, правка 27.10 22:20): G1 PBO по CSCV < 0,2 (по стратегии); G2 сделок ≥ 30 в каждом году (стороны/2, 17 тикеров);
G3 соседей с min mSR > 0 не меньше 3 и активность ≥ 0,5 базы; G4 s9 (`ekviv_s9.run_sk`: позиции побайтно, PnL ≤ 1e-12).
DSR (`sweep_val_stat.dsr_n`: var — по пробам реестра, N накопленное, SR/T/skew/kurt — 2018) — справочно, не ворота; min mSR > 0 — не ворота,
целевая min mSR — критерий выбора (ничья по `obj`, Ж-043 п. 4). Цель mSR ≥ 1 в каждом году — уровень. Validation не читается.
"""
from __future__ import annotations

import csv
import math
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "scripts"]
import zh043_lib as L                                   # noqa: E402
from fastbt import reestr_otbora as rs                  # noqa: E402
from fastbt.ekviv_s9 import PNL_TOL                     # noqa: E402
from fastbt.sweep_spec import pkey                      # noqa: E402
from fastbt.sweep_val_stat import dsr_n                 # noqa: E402

MIN_TRADES, MIN_NB, PBO_MAX, TOP = 30, 3, 0.2, 3
ST_OK, ST_NO = "kandidat_A_prim_zh043", "ne_doshla_zh043"
OLD_SRC = ("sweep_e1", "sweep_e1v2", "sweep_e1v2_dostroyka")
NAMES = {"G1": "PBO ≥ 0,2", "G2": "сделок < 30 в год", "G3": "соседей с min mSR > 0 меньше 3 или мала активность",
         "G4": "не воспроизводится s9"}


def key_of(exp, r) -> str:
    return rs.klyuch_str(rs.kanon_ledger(pkey(L.point_from_row(exp, r))))


def trials_n(sid: str, exp: dict, d: pd.DataFrame) -> tuple:
    """(SR дневные всех проб стратегии, N накопленное). Реестр (sweep_e1, e1v2, доводка, Ж-042) + пробы Ж-043; N = N_vsego kand_a
    + уникальные ключи Ж-042/Ж-043 сверх прежних источников (в kand_a их нет)."""
    old, new, tr = set(), set(), {}
    for r in rs.chitat(rs.VYHOD):
        if r["strategiya"] == sid and r["tune_004"] not in ("", None):
            tr[r["klyuch"]] = float(r["tune_004"]) / math.sqrt(252)
            (old if r["istochnik"] in OLD_SRC else new).add(r["klyuch"])
    for _, r in d.iterrows():
        k = key_of(exp, r)
        tr[k] = r.s2018 / math.sqrt(252)
        new.add(k)
    k0 = pd.read_csv("docs/tables/sweep_e1v2/kand_a.csv", comment="#")
    n0 = int(k0[k0.strategiya == sid].N_vsego.iloc[0])
    return list(tr.values()), n0 + len(new - old)


def gates(c: dict, ref: list, dsr: float, pbo: float | None = None) -> dict:
    """Ворота train по Ж-044: G1 PBO < 0,2 (по стратегии; None — не применяется), G2 сделок, G3 плато, G4 s9. DSR — справочно, не ворота;
    min mSR > 0 — не ворота (критерий выбора). `dsr` оставлен в сигнатуре для вызова Ж-043, на ворота не влияет."""
    g = {"G1": pbo is None or (L.fin(pbo) and pbo < PBO_MAX), "G2": all(r["sdelok_god"] >= MIN_TRADES for r in ref),
         "G3": c["sosedey_pos"] >= MIN_NB and bool(c["aktivnost_ok"]),
         "G4": all(r["sverka_s9"] == "sovpalo" and r["max_dpnl"] <= PNL_TOL and r["bar_poz_razl"] == 0 for r in ref)}
    why = {"G1": f"{NAMES['G1']}: {pbo}", "G2": f"{NAMES['G2']}: " + " / ".join(f"{r['sdelok_god']:.1f}" for r in ref),
           "G3": f"{NAMES['G3']}: {c['sosedey_pos']} из {c['sosedey']}"}
    return dict({k: int(v) for k, v in g.items()}, ok=all(g.values()), prichina="; ".join(why.get(k, NAMES[k]) for k, v in g.items() if not v) or "все ворота пройдены")


def itog(sid: str, pbo: float | None = None):
    sw, exp = L.open_sweep(sid)
    d = pd.read_csv(L.OUT / f"{sid}_dovodka.csv", comment="#")
    trials, n = trials_n(sid, exp, d)
    cands = []
    for i, (_, r) in enumerate(d[d.top10 == 1].sort_values("obj", ascending=False).iterrows(), 1):
        p = L.point_from_row(exp, r)
        ref = L.ref_rows(sid, sw.ev, p, f"{sid}_pul{i}", "zh043")
        ds = dsr_n(float(r["sr_d"]), trials, n, int(r["T"]), float(r["skew"]), float(r["kurt"]))
        cands.append(dict(r.to_dict(), p=p, ref=ref, dsr=ds["dsr"], var_sr=ds["var_sr"], N=n, **gates(r, ref, ds["dsr"], pbo)))
    full = [c for c in cands if c["ok"] and not c["kray"]]
    first = L.pick_best(full, "obj") or L.pick_best([c for c in cands if not c["kray"]] or cands, "obj")
    order = [first] + sorted((c for c in cands if c is not first), key=lambda c: (c["kray"], -c["plato"]))[:TOP - 1]
    out = []
    for i, c in enumerate(order, 1):
        ref, st = c["ref"], (ST_OK if c["ok"] else ST_NO) if i == 1 else "informatsionno_ne_luchshaya"
        out.append(dict(rang=i, status=st, **{k: c[k] for k in ("G1", "G2", "G3", "G4")}, prichina=c["prichina"], rol=c["rol"],
                        msr_2018=c["msr_2018"], msr_2019=c["msr_2019"], obj_min_msr=c["obj"], cel_msr_ge1_oba_goda=int(c["cel_ge1"]),
                        **{f"{m}_{g}": c[f"{m}_{g}"] for g in (2018, 2019) for m in ("sharpe", "sortino", "kalmar", "mdd")},
                        sdelok_2018=ref[0]["sdelok_god"], sdelok_2019=ref[1]["sdelok_god"], plato=c["plato"], sosedey=c["sosedey"],
                        sosedey_min_msr_pos=c["sosedey_pos"], sigma126=c["sigma126"], dsr=c["dsr"], N=c["N"], var_sr=c["var_sr"],
                        max_dpnl=max(x["max_dpnl"] for x in ref), **{f"p.{k}": v for k, v in c["p"].items()}))
    t = pd.DataFrame(out)
    L.write_csv(t, f"{sid}_itog.csv", "# строка ранга 1 — вердикт; ранги 2–3 — справка; ворота G1–G4 (Ж-044), цель mSR ≥ 1 — уровень (Ж-043 п. 3)\n")
    print(t.drop(columns=[c for c in t if c.startswith("p.")]).T.to_string())
    dopisat_reestr(sid, exp, d, t.iloc[0], rs.klyuch_str(rs.kanon_ledger(pkey(first["p"]))))


S3_KEY = ("adx_min=20;consec_entry=1;cooldown_bars=0;dc_window=40;exec_cooldown=3;exec_min_hold=0;initial_sl_mult=2.5;min_strength=0;"
          "obyom_mnozh_sma20=1;q_sily=0;rm_edinitsa=atr14;rm_parabolic.parabolic_max=0.15;rm_parabolic.parabolic_step=0.03;"
          "rm_sqrt_rv.k_sl=3;rm_sqrt_rv.k_trail=3;rm_sqrt_rv.okno_rv=20;tf_min_hold=0;tf_reentry=0;trail_atr_mult=3;trail_n=15;"
          "trail_type=parabolic_step;vf_porog=0.5")


def ispravlenie_s3_zh042() -> None:
    """Строка-исправление S3 по Ж-042 (старую `ne_doshla_zh042` не трогаем). Точку проверили заново (27.10): Шарп +0,0858 / +0,3077,
    сделок 93 / 76,5, соседей с min Шарпа > 0 не меньше 3, s9 совпал (все четыре ворота Ж-042)."""
    if ("S3", S3_KEY, "zh043_ispravlenie") in {(r["strategiya"], r["klyuch"], r["istochnik"]) for r in rs.chitat(rs.VYHOD)}:
        return
    row = dict.fromkeys(rs.POLYA, "")
    row.update(strategiya="S3", klyuch=S3_KEY, rezhim="zh043_ispravlenie", istochnik="zh043_ispravlenie", status="ispravlenie_zh042",
               tune_004=rs.fmt(0.08581652980603346), ctrl_004=rs.fmt(0.3076811042899724), ekviv_s9="da",
               prichina="ошибка пула плато: точка (сосед лучшей по плато, exec_cooldown=3) проходила ворота Ж-042; статус ne_doshla_zh042 для S3 ошибочен")
    with open(rs.VYHOD, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=rs.POLYA).writerow(row)


def dopisat_reestr(sid: str, exp: dict, d: pd.DataFrame, best, fk: str) -> None:
    """Дописать пробы этапа и итог; строки с тем же ключом и источником уже есть — не дублируются."""
    F = lambda x: rs.fmt(float(x))                      # noqa: E731
    seen = {(r["strategiya"], r["klyuch"], r["istochnik"]) for r in rs.chitat(rs.VYHOD)}
    new = []
    for _, r in d.iterrows():
        k, top = key_of(exp, r), r.top10 == 1
        base = dict(strategiya=sid, klyuch=k, rezhim=f"zh043_{r['rol']}", istochnik="zh043_konveyer", tune_004=F(r.s2018), ctrl_004=F(r.s2019),
                    tune_0035=F(r.s2018_0035), ctrl_0035=F(r.s2019_0035), storon_god=F(r.storon_god_2018),
                    plato=F(r.plato) if top else "", sosedey=F(r.sosedey) if top else "", aktivnost_ok=rs.flag(r.aktivnost_ok),
                    plato_ok=rs.flag(r.sosedey_pos >= MIN_NB) if top else "", vyrozhdena="", status="proba_zh043", ekviv_s9="da",
                    prichina=f"проба Ж-043; mSR 2018 {r.msr_2018:.4f} / 2019 {r.msr_2019:.4f}; Шарп в колонках tune/ctrl")
        if (sid, k, "zh043_konveyer") not in seen:
            new.append(base)
    if (sid, fk, "zh043_vorota") not in seen:             # вердикт — отдельной строкой с источником zh043_vorota
        row = dict.fromkeys(rs.POLYA, "")
        row.update(strategiya=sid, klyuch=fk, rezhim="zh043_itog", istochnik="zh043_vorota", status=best.status, prichina=best.prichina,
                   tune_004=F(best.sharpe_2018), ctrl_004=F(best.sharpe_2019), plato=F(best.plato), sosedey=F(best.sosedey), ekviv_s9="da")
        new.append(row)
    with open(rs.VYHOD, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=rs.POLYA).writerows(new)
    print(f"в реестр дописано строк: {len(new)}")
