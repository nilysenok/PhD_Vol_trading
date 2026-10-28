"""Разбор прохода 1 (ledger v2, train4, sobol1 × 5000 × S1–S6). Разработка 2016–2019, потолок. Выбор из справочных строк запрещён.
  --sid S1 --tmp DIR    одна Sk: шаги 1–5, 7, 8 -> DIR/S1.json (cwd = корень копии; кэши — через FASTBT_CACHE и др.)
  --svod --tmp DIR      свод DIR/S*.json -> docs/tables/progon_e1/prohod1_*.csv
  --reestr              все строки ledger -> docs/otbor/reestr_otbora_progon_v2.csv.gz + указатель в reestr_otbora_A_v1.csv
Запуск: FASTBT_CACHE=… FASTBT_E1_CACHE=… NUMBA_CACHE_DIR=… python3 scripts/analysis/prohod1_razbor.py --sid S1 --tmp …"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prohod1_lib as L  # noqa: E402
import prohod1_otbor as O  # noqa: E402
import prohod1_rezhimy as R  # noqa: E402

TAB = L.ROOT / "docs/tables/progon_e1"
TABLICY = ("stroki", "raspredelenie", "region", "wf", "pbo_vorota", "rezhimy", "prichiny", "aktivnost")


def obekty(d: dict, o: dict) -> list:
    """Объекты для режимов и причин: центр региона; при отсутствии региона — справочно лучшая устойчивая основная строка
    (не отбор: Sk всё равно «в»), чтобы режим и причины были видны."""
    if o["ic"] is not None:
        return [("tsentr_regiona", o["ic"])]
    m, _, ust, _ = O.msr_i_gody(o["M5"], d["daty"], O.GODA_TRAIN4)
    k = np.where(ust & np.isfinite(m), m, -np.inf)
    return [("sprav_luchshaya_ustojchivaya_NE_tsentr", int(np.argmax(k)))] if np.isfinite(k.max()) else []


def aktivnost(sid: str, nazv: str, row: dict, x: np.ndarray, daty: np.ndarray) -> dict:
    """Справочно (не отбор): сделок по годам и доля дней с ненулевым рядом (приближение «времени в позиции»)."""
    sd = row.get("sdelok_y") or {}
    god = np.asarray(daty).astype("datetime64[Y]").astype(int) + 1970
    return dict(S=sid, obekt=nazv, klyuch_sha=row["klyuch_sha"], sdelok_v_god_sred=float(np.mean(list(sd.values()))) if sd else float("nan"),
                **{f"sdelok_{y}": sd.get(str(y), sd.get(y)) for y in L.GODY}, dolya_dnej_s_dvizheniem=float((np.asarray(x) != 0).mean()),
                **{f"dolya_dnej_{y}": float((np.asarray(x)[god == y] != 0).mean()) for y in L.GODY})


def rezhimy_prichiny(sid: str, d: dict, o: dict) -> tuple:
    rez, pr, akt = [], [], []
    for nazv, i in obekty(d, o):
        row = d["rows"][i]
        akt.append(aktivnost(sid, nazv, row, d["M"][:, i], d["daty"]))
        ryady = [(nazv, o["M5"][:, i])] + ([("ansambl_regiona", o["res"]["ryad"])] if o["ic"] is not None else [])
        for ob, x in ryady:
            rez += [dict(S=sid, obekt=ob, klyuch_sha=row["klyuch_sha"], razrez=a, gruppa=g, dney=n, mSR=v) for a, g, n, v in R.razrezy(x, d["daty"])]
        dol, nd, no, dx = R.prichiny_vyhoda(sid, row, d["M"][:, i])
        pr.append(dict(S=sid, obekt=nazv, klyuch_sha=row["klyuch_sha"], zakryto_sdelok=nd, otkryto_na_konce=no, maxabs_port_vs_ledger=dx,
                       **{f"dolya_{k}": v for k, v in dol.items()}, ledger_prichiny=json.dumps(row.get("prichiny"), sort_keys=True)))
    return rez, pr, akt


def odna_sk(sid: str, tmp: Path) -> None:
    d = L.zagruzit(sid)
    o = O.otbor_sid(d)
    rez, pr, akt = rezhimy_prichiny(sid, d, o)
    sch = L.schetchiki(d["vse"], d["ch"])
    ras = [dict(S=sid, **r, sverka_tselevoj_rashozhdenij=L.sverka_tselevoj(d["M"], d["Z"], d["daty"])) for r in L.raspredelenie(d["M"], d["Z"], d["daty"])]
    pv = dict(S=sid, **o["pbo"], pbo_wf=o["pbo_wf"], **o["vorota"], iskhod=o["iskhod"])
    out = dict(stroki=[dict(S=sid, **sch)], raspredelenie=ras, region=[dict(S=sid, **o["region"])],
               wf=[dict(S=sid, **w) for w in o["wf"]], pbo_vorota=[pv], rezhimy=rez, prichiny=pr, aktivnost=akt)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / f"{sid}.json").write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf-8")
    print(sid, "исход", o["iskhod"], "PBO", o["pbo"]["pbo"], "регион", o["region"].get("region"))


def svod(tmp: Path, prefix: str) -> None:
    for t in TABLICY:
        rows = [r for sid in L.SIDS if (tmp / f"{sid}.json").exists() for r in json.loads((tmp / f"{sid}.json").read_text(encoding="utf-8"))[t]]
        pd.DataFrame(rows).to_csv(TAB / f"{prefix}_{t}.csv", index=False)
        print(t, len(rows))


def reestr(prefix: str) -> None:
    from fastbt import reestr_progon as RP
    n = 0
    for sid in L.SIDS:
        n += RP.dopisat_iz_ledger(L.LEDGER, sid)
    print("записано строк", n)
    print(RP.ukazatel(prefix.replace("prohod", "progon")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid"), ap.add_argument("--tmp"), ap.add_argument("--svod", action="store_true"), ap.add_argument("--reestr", action="store_true")
    ap.add_argument("--prefix", default="prohod1", help="префикс таблиц и имя указателя (prohod12 → progon12); ledger читается целиком (все проходы)")
    a = ap.parse_args()
    os.chdir(L.ROOT)
    if a.reestr:
        reestr(a.prefix)
    elif a.svod:
        svod(Path(a.tmp), a.prefix)
    else:
        odna_sk(a.sid, Path(a.tmp))


if __name__ == "__main__":
    main()
