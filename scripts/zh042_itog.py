"""Конвейер Ж-042, этап итога: ворота уровня для лучшей по плато точки + дозапись в реестр отбора A.

Ворота: (1) Шарп после издержек 0,04 % > 0 в 2018 и в 2019 (боевой путь s9); (2) сделок ≥ 30 в год в каждом году
(стороны/2, сумма по 17 тикерам); (3) плато: соседей ≥ 3 и активность ≥ 0,5 базы; (4) точка воспроизводится s9:
позиции побайтно, PnL ≤ 1e-12 (`ekviv_s9.run_sk`). Валидация не читается. Реестр только дополняется.
"""
from __future__ import annotations

import csv
import sys

import pandas as pd

sys.path[:0] = [".", "scripts"]
import zh042_lib as L                                   # noqa: E402
from fastbt import reestr_otbora as rs                  # noqa: E402
from fastbt.ekviv_s9 import PNL_TOL                     # noqa: E402
from fastbt.sweep_spec import pkey                      # noqa: E402

MIN_TRADES, MIN_NB, TOP = 30, 3, 3
STATUS_OK, STATUS_NO = "kandidat_A_prim_zh042", "ne_doshla_zh042"


def vorota(rows: list, pl: dict) -> dict:
    """rows — две строки ref_rows (2018, 2019); pl — плато и активность точки. Значения 0/1 и текст причины."""
    g = {"G1_sharpe_pos": all(r["sharpe_posle_004"] > 0 for r in rows),
         "G2_sdelok_30": all(r["sdelok_god"] >= MIN_TRADES for r in rows),
         "G3_plato": pl["sosedey"] >= MIN_NB and bool(pl["aktivnost_ok"]),
         "G4_s9": all(r["sverka_s9"] == "sovpalo" and r["max_dpnl"] <= PNL_TOL and r["bar_poz_razl"] == 0 for r in rows)}
    txt = {"G1_sharpe_pos": "Шарп после 0,04 % ≤ 0: " + " / ".join(f"{r['god']} {r['sharpe_posle_004']:.3f}" for r in rows),
           "G2_sdelok_30": "сделок < 30 в год: " + " / ".join(f"{r['god']} {r['sdelok_god']:.1f}" for r in rows),
           "G3_plato": f"плато: соседей {pl['sosedey']:g}, активность_ok {pl['aktivnost_ok']:g}",
           "G4_s9": "не воспроизводится s9"}
    return dict({k: int(v) for k, v in g.items()}, prichina="; ".join(txt[k] for k, v in g.items() if not v) or "все ворота пройдены")


def itog(sid: str):
    sw, exp = L.open_sweep(sid)
    d = pd.read_csv(L.OUT / f"{sid}_dovodka.csv", comment="#")
    top = d[d.top10 == 1].sort_values(["kray", "plato"], ascending=[True, False]).head(TOP)
    out = []
    for i, (_, r) in enumerate(top.iterrows(), 1):
        p = L.point_from_row(exp, r)
        rows = L.ref_rows(sid, sw.ev, p, f"{sid}_plato{i}")
        g = vorota(rows, r)
        ok = all(g[k] for k in g if k.startswith("G"))
        st = (STATUS_OK if ok else STATUS_NO) if i == 1 else "informatsionno_ne_luchshaya_po_plato"
        out.append(dict(rang_plato=i, status=st, **g, plato=r.plato, sosedey=r.sosedey,
                        sharpe_2018_004=rows[0]["sharpe_posle_004"], sharpe_2019_004=rows[1]["sharpe_posle_004"],
                        sharpe_2018_do_izd=rows[0]["sharpe_do_izderzhek"], sharpe_2019_do_izd=rows[1]["sharpe_do_izderzhek"],
                        sdelok_2018=rows[0]["sdelok_god"], sdelok_2019=rows[1]["sdelok_god"],
                        max_dpnl=max(x["max_dpnl"] for x in rows), **{f"p.{k}": v for k, v in p.items()}))
    t = pd.DataFrame(out)
    L.write_csv(t, f"{sid}_itog.csv", "# строка ранга 1 — вердикт; ранги 2–3 — справка (ворота для лучшей по плато точки, Ж-042 п. 4)\n")
    print(t.drop(columns=[c for c in t if c.startswith("p.")]).T.to_string())
    dopisat_reestr(sid, exp, d, t.iloc[0])


def dopisat_reestr(sid: str, exp: dict, d: pd.DataFrame, best) -> None:
    """Дописать в реестр строки проб этапа и итоговую точку; строки с таким же ключом и источником уже есть — не дублируются."""
    F = lambda x: rs.fmt(float(x))                       # noqa: E731  (repr numpy-числа в CSV не пишем)
    seen = {(r["strategiya"], r["klyuch"], r["istochnik"]) for r in rs.chitat(rs.VYHOD)}
    new = {}
    for _, r in d.iterrows():
        p = L.point_from_row(exp, r)
        k = rs.klyuch_str(rs.kanon_ledger(pkey(p)))
        top = r.get("top10") == 1
        base = dict(strategiya=sid, klyuch=k, rezhim=f"zh042_{r['rol']}", istochnik="zh042_konveyer",
                    tune_004=F(r.s2018), ctrl_004=F(r.s2019), tune_0035=F(r.s2018_0035), ctrl_0035=F(r.s2019_0035),
                    storon_god=F(r.storon_god_2018), plato=F(r.plato) if top else "", sosedey=F(r.sosedey) if top else "",
                    aktivnost_ok=rs.flag(r.aktivnost_ok) if r["rol"] == "faktor" else "", plato_ok=rs.flag(r.sosedey >= MIN_NB) if top else "",
                    vyrozhdena="", status="proba_zh042", prichina="проба этапа доводки Ж-042 (факторный 3³ или сосед плато)", ekviv_s9="da")
        if r.get("rang_plato") == 1:                     # итоговая точка — та же строка ключа, со статусом вердикта
            base.update(rezhim="zh042_itog", status=best.status, prichina=best.prichina)
        new[k] = base if (k not in new or base["status"] != "proba_zh042") else new[k]
    new = [x for k, x in new.items() if (sid, k, "zh042_konveyer") not in seen]
    with open(rs.VYHOD, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=rs.POLYA).writerows(new)
    print(f"в реестр дописано строк: {len(new)}")


if __name__ == "__main__":
    itog(sys.argv[1])
