"""Фильтр «есть боевой эталон» (вариант (а), Ж-039/Ж-040): воспроизводит ли точку Sk боевой путь s9 (`ekviv_s9.boevoy_pos`).

Без данных. Правило — ровно те условия, при которых `ekviv_s9.rm_row` или `Unit._build_rm` отказывают точке:
выходы в √RV; маски/квантили/выход S6 после кэша сигналов, не равные зашитым порогам (`MASK_BASE`); `trail_n` вне `TREND_TN`,
`trail_type` вне боевого списка; комбо пивотов вне `PIV_COMBOS`; `sl_dev` VWAP вне `VWAP_DEVS`; `rm_edinitsa` ≠ atr14.
Тест сверяет фильтр с `rm_row` (нет рассогласования) и с полем `etalon` в yaml.
"""
from __future__ import annotations

from fastbt import ekviv_s9 as ek
from fastbt.unit import PIV_COMBOS, TREND_TN, VWAP_DEVS

TRAIL_TYPES = ("fixed_atr", "chandelier", "parabolic_step")


def prichina_net(sid: str, p: dict) -> str | None:
    """None — боевой эталон есть; иначе причина (строка)."""
    try:
        ek.rm_row(sid, p)
    except ek.NoRef as e:
        return str(e)
    except KeyError as e:
        return f"нет параметра {e}"
    edin = p.get("rm_edinitsa")
    if sid in ("S1", "S2", "S3", "S4") and edin is not None and edin != "atr14":
        return f"rm_edinitsa={edin}: боевой RM только atr14"
    if sid in ("S3", "S4"):
        if int(p["trail_n"]) not in TREND_TN:
            return f"trail_n={p['trail_n']} вне TREND_TN"
        if p["trail_type"] not in TRAIL_TYPES:
            return f"trail_type={p['trail_type']}: Unit не принимает"
    if sid == "S5" and p["rm_tip"] == "pivot":
        c = tuple(p.get(f"rm_pivot.{k}") for k in ("sl_target", "tp_target", "buffer"))
        if c not in PIV_COMBOS:
            return f"комбо пивотов {c} вне PIV_COMBOS"
    if sid == "S6" and p["rm_tip"] == "vwap" and p.get("rm_vwap.sl_dev") not in VWAP_DEVS:
        return f"sl_dev={p.get('rm_vwap.sl_dev')} вне VWAP_DEVS"
    return None


def voproizvodima(sid: str, p: dict) -> bool:
    return prichina_net(sid, p) is None


def _podpis(r) -> tuple:
    """Подпись результата на train: одинаковые tune/ctrl/стороны — одна торговля (параметр на данных не действует)."""
    return tuple(round(float(r[c]), 9) for c in ("tune_sharpe", "ctrl_sharpe", "sdelok_y"))


def izbrat(t, k: int = 3):
    """Топ-k по плато среди строк `sweep_plato.table`: край (соседей < 3) не берётся, дубли по подписи сводятся."""
    seen, rows = set(), []
    for i, r in t.sort_values("plateau", ascending=False).iterrows():
        if r["edge"] or _podpis(r) in seen:
            continue
        seen.add(_podpis(r))
        rows.append(i)
    return t.loc[rows[:k]]


def plato_vosp(sid: str, scratch, root=".", n_top: int = 10) -> dict:
    """Плато на воспроизводимых точках: топ-n_top допустимых по активности воспроизводимых проб ledger + соседи ±1 (как `sweep_plato`).

    Ledger копируется в `scratch` (в репозитории не пишется); число проб после — в `n_after` (идёт в N). Данные train 2018/2019."""
    import json
    import shutil
    from pathlib import Path
    import pandas as pd
    from fastbt import reestr_otbora as rs
    from fastbt import sweep_plato as pl
    from fastbt.sweep import Sweep, make_evaluator
    from fastbt.sweep_pts import TOR_MIN, effect_params, make_fix, point_cols
    from fastbt.sweep_run import load, periods
    from fastbt.sweep_spec import Spec, pkey
    from fastbt.sweep_val_stat import dsr_n
    tab = Path(root) / "docs" / "tables" / "sweep_e1v2"
    spec_d, exp, raw = load(Path(root) / "fastbt" / "designs" / "sweep" / f"{sid}_setka_e1b.yaml")
    led = Path(scratch) / "_ledger" / sid
    shutil.copytree(tab / "_ledger" / sid, led, dirs_exist_ok=True)
    spec = Spec(**spec_d, out=str(led))
    sw = Sweep(spec, make_evaluator(spec))
    per, comms, fix = periods(raw), sw.spec.comm_pct, make_fix(exp)
    c0, n0 = comms[0], len(sw.ledger)
    keys = effect_params(exp, tab, sid, c0, TOR_MIN)
    min_sd = 0.5 * sw.evaluate(exp["base"], per[0], c0)["sdelok_y"]
    allc = pl.candidates(sw, exp, per, c0, min_sd, fix, n=10 ** 9)
    cands = [c for c in allc if voproizvodima(sid, c["p"])][:n_top]
    n_pts = pl.add_plateau(sw, exp, per, c0, keys, fix, cands)
    sw.save()
    t = pl.table(sw, cands, per, comms, sw.trials_sr())
    k0 = pd.read_csv(tab / f"{sid}_kandidaty_plato.csv", comment="#").iloc[0]               # N_старых проб: N_vsego − N_novye (до достройки)
    n_starye = int(k0["N_vsego"] - k0["N_novye"])
    n_all, trials0 = len(sw.ledger) + n_starye, list(json.loads((tab / "_ledger" / sid / "ledger.json").read_text("utf-8")).values())
    ms = [point_cols(sw, c["p_led"], per, c0) for c in sorted(cands, key=lambda c: -c["plateau"])]   # порядок как в table()
    t["N_novye"], t["N_starye"], t["N_vsego"] = len(sw.ledger), n_starye, n_all
    t["dsr_vsego"] = [dsr_n(m["sr_d"], trials0, n_all, m["T"], m["skew"], m["kurt"])["dsr"] for m in ms]    # var_sr — по ledger в репозитории
    t["klyuch"] = [rs.klyuch_str(rs.kanon_ledger(pkey(c["p_led"]))) for c in sorted(cands, key=lambda c: -c["plateau"])]
    return dict(table=t, n_before=n0, n_after=len(sw.ledger), n_neighbors=n_pts, n_dopustimyh=len(allc),
                n_vosp=sum(voproizvodima(sid, c["p"]) for c in allc), min_sd=min_sd, ledger=sw.ledger)
