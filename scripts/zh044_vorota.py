"""Ж-044: вердикты S1–S6 по новым воротам train (PBO < 0,2 · плато ≥ 3 · сделок ≥ 30 в год · s9). Только train 2018–2019, 0,04 %.

Кандидаты — топ-10 пула Ж-043 (`docs/tables/zh043/S{k}_dovodka.csv`, плато и соседи уже посчитаны). Точка стратегии — лучшая по целевой
min(mSR 2018, mSR 2019) среди прошедших ворота (ничья в пределах 0,05 — меньшее σ126, `zh043_lib.pick_best`); прошедших нет — показана лучшая
по целевой среди прошедших ворота кроме PBO (иначе просто лучшая), вердикт «не дошла». DSR/PSR справочно: 2018–2019, T = дней (не сделок).
Запуск: FASTBT_MAIN=<своя копия> /usr/bin/python3 scripts/zh044_vorota.py [S1 ...]. Таблица: docs/tables/zh044/vorota_train.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path[:0] = [".", "scripts"]
import zh043_itog as I                                  # noqa: E402
import zh043_lib as L                                   # noqa: E402
import zh044_lib as Z                                   # noqa: E402
from fastbt import metriki_e as me                      # noqa: E402
from fastbt import reestr_otbora as rs                  # noqa: E402
from fastbt.sweep_spec import pkey                      # noqa: E402

OUT = Path("docs/tables/zh044")
SHAPKA = "# статус: разработка, train 2018–2019; validation не открыта; Ж-044\n"
ST_OK, ST_NO = "kandidat_val_zh044", "ne_doshla_zh044"
V_OK, V_NO = "кандидат на validation", "не дошла: ворота"


def evaluate_pool(sid: str, sw, exp: dict, m: dict, pbo: float, n_all: int) -> list:
    """Кандидаты (топ-10 пула Ж-043): ряды из матрицы, mSR/Шарп по годам, ворота, справочные PSR/DSR."""
    d = pd.read_csv(L.OUT / f"{sid}_dovodka.csv", comment="#")
    col, n18, out = {k: j for j, k in enumerate(m["keys"])}, m["n18"], []
    for i, (_, r) in enumerate(d[d.top10 == 1].sort_values("obj", ascending=False).iterrows(), 1):
        p = L.point_from_row(exp, r)
        key = I.key_of(exp, r)
        x = m["M"][:, col[key]] if key in col else np.r_[Z.series(sw, p)[0], Z.series(sw, p)[1]]
        x18, x19 = x[:n18], x[n18:]
        sh = (me.sharpe(x18), me.sharpe(x19))
        assert abs(sh[0] - r.s2018) < 1e-9 and abs(sh[1] - r.s2019) < 1e-9, f"{sid}: ряд матрицы не совпал с доводкой Ж-043"
        ref = L.ref_rows(sid, sw.ev, p, f"{sid}_zh044_{i}", "zh044")
        g = I.gates(r, ref, float("nan"), pbo)
        out.append(dict(r.to_dict(), p=p, key=key, ref=ref, x=x, x18=x18, x19=x19, sh=sh, g=g, sigma126=r.sigma126,
                        **{f"gate_{k}": g[k] for k in ("G1", "G2", "G3", "G4")},
                        ok_bez_pbo=bool(g["G2"] and g["G3"] and g["G4"]), **Z.psr_dsr(x, m["M"], n_all)))
    return out


def pick(cands: list) -> dict:
    ok = [c for c in cands if c["g"]["ok"]]
    return L.pick_best(ok, "obj") or L.pick_best([c for c in cands if c["ok_bez_pbo"]], "obj") or L.pick_best(cands, "obj")


def row_for(sid: str, m: dict, pb: dict, c: dict, rang1_key: str, n_uniq: int, pbo_u: float) -> dict:
    ref, g = c["ref"], c["g"]
    ms = (c["msr_2018"], c["msr_2019"])
    dop = [x for x in (f"PBO {pb['pbo']:.3f} ≥ 0,2" if not g["G1"] else "", "" if g["G2"] else "сделок < 30 в год",
                       "" if g["G3"] else f"плато {c['sosedey_pos']:.0f} из {c['sosedey']:.0f} < 3", "" if g["G4"] else "s9 не воспроизводит") if x]
    return dict(strategiya=sid, sharpe_2018=c["sh"][0], sharpe_2019=c["sh"][1], msr_2018=ms[0], msr_2019=ms[1], obj_min_msr=c["obj"],
                pbo=pb["pbo"], N=pb["N"], T=pb["T"], plato_sosedey_min_msr_pos=int(c["sosedey_pos"]), plato_sosedey=int(c["sosedey"]),
                sdelok_2018=ref[0]["sdelok_god"], sdelok_2019=ref[1]["sdelok_god"],
                s9="sovpalo" if g["G4"] else "RASHOZhDENIE", psr=c["psr"], dsr=c["dsr"], T_psr_dsr=c["T"], N_dsr=c["N_dsr"],
                cel_msr_ge04=int(Z.fin(c["obj"]) and c["obj"] >= 0.4), cel_msr_ge1=int(Z.fin(c["obj"]) and c["obj"] >= 1.0),
                G1_pbo=g["G1"], G2_sdelki=g["G2"], G3_plato=g["G3"], G4_s9=g["G4"], verdikt=V_OK if g["ok"] else V_NO,
                prichina="все ворота пройдены" if g["ok"] else "; ".join(dop), tochka_ta_zhe_chto_zh043=int(c["key"] == rang1_key),
                N_unikalnykh_ryadov=n_uniq, pbo_unikalnye_ryady=pbo_u, klyuch=c["key"])


def one(sid: str) -> dict:
    sw, exp = L.open_sweep(sid)
    m = Z.build_matrix(sid, sw, exp)
    pb = Z.pbo(m["M"])
    uniq = np.unique(np.round(m["M"], 12), axis=1)
    pbo_u = Z.pbo(uniq)["pbo"]
    itog = pd.read_csv(L.OUT / f"{sid}_itog.csv", comment="#")
    r1 = itog.iloc[0]
    rang1 = I.key_of(exp, r1)
    cands = evaluate_pool(sid, sw, exp, m, pb["pbo"], int(r1.N))
    best = pick(cands)
    print(sid, f"T={pb['T']} N={pb['N']} PBO={pb['pbo']:.3f} PBO_unik={pbo_u:.3f} отброшено {m['drop']}; "
          f"прошли ворота: {sum(c['g']['ok'] for c in cands)} из {len(cands)}; точка={best['key'] == rang1 and 'та же, что Ж-043' or 'другая'}")
    return row_for(sid, m, pb, best, rang1, uniq.shape[1], pbo_u)


def dopisat_reestr(rows: list) -> None:
    """Итог по стратегии — отдельной строкой источника zh044_vorota; реестр только дописывается."""
    seen = {(r["strategiya"], r["klyuch"], r["istochnik"]) for r in rs.chitat(rs.VYHOD)}
    new = []
    for r in rows:
        if (r["strategiya"], r["klyuch"], "zh044_vorota") in seen:
            continue
        row = dict.fromkeys(rs.POLYA, "")
        ok = r["verdikt"] == V_OK
        row.update(strategiya=r["strategiya"], klyuch=r["klyuch"], rezhim="zh044_itog", istochnik="zh044_vorota", status=ST_OK if ok else ST_NO,
                   prichina=f"{r['prichina']}; PBO {r['pbo']:.3f} (N {r['N']}, T {r['T']}), плато {r['plato_sosedey_min_msr_pos']} из {r['plato_sosedey']}, "
                            f"min mSR {r['obj_min_msr']:.3f}", tune_004=rs.fmt(r["sharpe_2018"]), ctrl_004=rs.fmt(r["sharpe_2019"]),
                   plato=rs.fmt(float(r["plato_sosedey_min_msr_pos"])), sosedey=rs.fmt(float(r["plato_sosedey"])), ekviv_s9="da")
        new.append(row)
    with open(rs.VYHOD, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=rs.POLYA).writerows(new)
    print(f"в реестр дописано строк: {len(new)}")


if __name__ == "__main__":
    sids = sys.argv[1:] or [f"S{k}" for k in range(1, 7)]
    rows = [one(s) for s in sids]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vorota_train.csv").write_text(SHAPKA + pd.DataFrame(rows).to_csv(index=False), "utf-8")
    dopisat_reestr(rows)
    print(pd.DataFrame(rows).drop(columns=["klyuch"]).T.to_string())
