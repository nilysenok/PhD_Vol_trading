"""К-2 (σ̂ C/D на wf, шаги 9–10): RRV (d172) и блок Д7 (d7_block) на базовой матрице без выходных сессий MOEX (ДСВД).

Справка: docs/wf_sigma_910_dsvd.md, п. 1–3. Формулы генераторов не меняются: scripts/analysis/d172_rrv_features.py и d7_block.py
вызываются как библиотеки. RRV: из ряда rrv_1850 убираются даты ДСВД и всё позже --konec (окна d/w/m по строкам ряда не
захватывают воскресенья). Д7: окна считаются по датам матрицы, поэтому матрица без ДСВД (выход К-1) подаётся как есть, а
наличие в ней дат ДСВД — отказ. Вход матрицы — <out>/data/regen/features_pit_v3.parquet; выход — рядом (в data/ репозитория не пишет).

Запуск: FASTBT_MAIN=<корень данных> python3 scripts/dannye/wf_rrv_d7_bez_dsvd.py --out <каталог> [--konec 2026-02-03]
Гейты (печатает): G1 строки < 2025 побайтно равны текущим файлам; G2 дат ДСВД 0, 01.11.2025 есть; G3 сдвиг −2…+1, максимум на k=0.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/analysis"), str(REPO / "scripts/dannye")]
import d172_rrv_features as D172  # noqa: E402
import d7_block as D7  # noqa: E402
import wf_baza_bez_dsvd as K1  # noqa: E402

MAIN = Path(os.environ.get("FASTBT_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))
GRANICA = pd.Timestamp("2025-01-01")  # строки раньше — побайтно прежние
SUBBOTA = pd.Timestamp("2025-11-01")  # рабочая суббота основной сессии: остаётся
KS = (-2, -1, 0, 1)
SDVIG_COLS = ["rrv_pk_d", "rrv_gk_d", "d7_rvi", "d7_rvi_pctl252", "d7_gvol_pc1"]


def proverit_bez_dsvd(base, dsvd):
    """Отказ, если в матрице есть даты ДСВД (окна Д7 и ключи RRV пошли бы через них)."""
    est = sorted(set(base.date) & set(dsvd))
    if est:
        raise ValueError(f"в матрице {len(est)} дат ДСВД, первая {est[0].date()}")


def rrv_bez_dsvd(rrv, dsvd, konec):
    """Ряд rrv_1850 без дат ДСВД и без дат позже konec."""
    return rrv[(rrv.date <= konec) & ~rrv.date.isin(dsvd)].reset_index(drop=True)


def sobrat_rrv(base, rrv):
    out = base.copy()
    nc = D172.new_columns(base[["date", "ticker"]], rrv)
    for c in D172.NEW:
        out[c] = nc[c].values
    return out


def sobrat_d7(base, h, cal_dir):
    return D7.assemble(base, h, cal_dir=str(cal_dir))


def razlichiya(new, old, granica=GRANICA):
    """Столбцы, в которых строки date < granica различаются побайтно (число строк — '__len__')."""
    a = new[new.date < granica].sort_values(["date", "ticker"]).reset_index(drop=True)
    b = old[old.date < granica].sort_values(["date", "ticker"]).reset_index(drop=True)
    if len(a) != len(b):
        return ["__len__"]
    bad = [c for c in b.columns if c not in a.columns]
    for c in [c for c in b.columns if c in a.columns]:
        x, y = a[c], b[c]
        same = x.dtype == y.dtype and (x.equals(y) if x.dtype == object else x.to_numpy().tobytes() == y.to_numpy().tobytes())
        if not same:
            bad.append(c)
    return bad


def profil(frame, col, ot, do, diff=False):
    """Медиана по тикерам corr(x_d, ln rv_{d+k}), k = −2…+1 (ряд на строках матрицы); diff — по приращениям обоих рядов."""
    f = frame[(frame.date >= ot) & (frame.date <= do)]
    out = {}
    for k in KS:
        cs = []
        for _, g in f.groupby("ticker"):
            x, y = g[col], np.log(g["rv"])
            cs.append((x.diff() if diff else x).corr((y.diff() if diff else y).shift(-k)))
        out[k] = float(np.nanmedian(cs)) if len(cs) else np.nan
    return out


def maks_na_nule(prof):
    v = {k: abs(c) for k, c in prof.items()}
    return not any(np.isnan(c) for c in v.values()) and max(v, key=v.get) == 0


def gate_g2(m, dsvd, nazv):
    n = len(set(m.date) & set(dsvd))
    ok = n == 0 and SUBBOTA in set(m.date)
    print(f"G2 {'PASS' if ok else 'FAIL'} {nazv}: дат ДСВД {n}, 01.11.2025 {'есть' if SUBBOTA in set(m.date) else 'нет'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="корень выхода; <out>/data/regen/features_pit_v3.parquet — выход К-1")
    ap.add_argument("--konec", default="2026-02-03")
    a = ap.parse_args()
    konec = pd.Timestamp(a.konec)
    dst = Path(a.out) / "data/regen"
    base = pd.read_parquet(dst / "features_pit_v3.parquet")
    dsvd = K1.daty_dsvd()
    proverit_bez_dsvd(base, dsvd)
    assert base.date.max() <= konec, base.date.max()
    res = {"features_pit_v3_rrv": sobrat_rrv(base, rrv_bez_dsvd(D172.load_rrv(MAIN / "data/iss_check/rrv_1850.parquet"), dsvd, konec))}
    for h in (5, 22):
        res[f"features_pit_v3_d7_h{h}"] = sobrat_d7(base, h, MAIN / "data/calendar")
    old_base = pd.read_parquet(MAIN / "data/regen/features_pit_v3.parquet")
    old_base = old_base[old_base.date < GRANICA]
    ok = True
    for nazv, m in res.items():
        m.to_parquet(dst / f"{nazv}.parquet", index=False)
        bad = razlichiya(m, pd.read_parquet(MAIN / f"data/regen/{nazv}.parquet"))
        print(f"G1 {'PASS' if not bad else 'FAIL'} {nazv}: строк {len(m)}, до 2025 различаются {bad[:8]}")
        ok = gate_g2(m, dsvd, nazv) and not bad and ok
        if bad and "_d7_h" in nazv:  # календарь экспираций обновлён после сборки старых файлов: сверка со старой базой, текущим календарём
            h = int(nazv.rsplit("_h", 1)[1])
            bad_b = razlichiya(m, sobrat_d7(old_base, h, MAIN / "data/calendar"))
            print(f"G1b {'PASS' if not bad_b else 'FAIL'} {nazv}: старая база (<2025) + текущий календарь, различаются {bad_b[:8]}")
        for c in [c for c in SDVIG_COLS if c in m.columns]:
            for diff in (False, True):
                p = profil(m, c, GRANICA, konec, diff)
                g3 = maks_na_nule(p)
                ok = ok and g3
                print(f"G3 {'PASS' if g3 else 'FAIL'} {nazv} {c}{' (прирост)' if diff else ''}: "
                      + " ".join(f"k{k:+d}={p[k]:.3f}" for k in KS))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
