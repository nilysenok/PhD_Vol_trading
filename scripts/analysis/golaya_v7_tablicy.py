"""Л1.6 (Ж-146): контроли к2/плацебо, шапка и таблицы драйвера голых идей S7–S12 (ядро — golaya_v7.py).

Прогон: python3 scripts/analysis/golaya_v7_tablicy.py --period train4|v2 (отдельный лист Р1; здесь — только синтетика в тестах).
Файлы: svod_<period>.csv, k2_<period>.csv, placebo_<period>.csv — первая строка «# {json шапки}».
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golaya_v7 as gv  # noqa: E402
from fastbt import progon_kontrol as pk  # noqa: E402
from fastbt.period_v2 import PERIODY  # noqa: E402


def kontrol(D: dict, sid: str, tip: str, n: int = gv.N_POVT) -> list:
    """mSR n контролей на центре: k2 (n сдвигов pk.k2_lags) или плацебо (n seed)."""
    par = pk.k2_lags(n) if tip == "k2" else [gv.SEED + i for i in range(n)]
    return [gv.mera(gv.schet(D, gv.signaly(D, sid, (tip, p)), **gv.shag(sid)), D) for p in par]


def shapka(D: dict, period: str) -> dict:
    kod = subprocess.run(["git", "-C", str(gv.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "?"
    h = hashlib.sha256(b"".join(np.ascontiguousarray(D[k].to_numpy()).tobytes() for k in ("c", "o", "cr", "div"))).hexdigest()
    return {"v": "7.0-gol", "status": "разработка, до аудита, потолок, голая идея", "okno_s": f"{PERIODY[period].y0}-01-01",
            "okno_po": str(PERIODY[period].konec.date()), "zaderzhka": "d+1 (центры); T=17:00|18:00|18:40 — справочно", "chestnaya": "net",
            "kod": kod, "vhody": h, "ceny_i_D": "исполнение: нескорректированные open 10:00/close 18:50; D после НДФЛ ×0,87 на день отсечки "
            "(свод Д8); сигналы S7–S12 на скорректированном ряду (adjust_ohlc, 10-минутки × множитель дня)"}


def tercil(sl: np.ndarray, d: int, v: float, okno: int = 252, minimum: int = 30):
    """Терциль силы v по окну |скор| дней d−okno…d−1 (строго до дня решения d); мало наблюдений — пусто."""
    w = sl[max(0, d - okno):d]
    w = w[np.isfinite(w) & (w >= gv.CENTR["S8"]["porog"])]
    if len(w) < minimum or not np.isfinite(v):
        return ""
    return int(1 + (v > np.quantile(w, 1 / 3)) + (v > np.quantile(w, 2 / 3)))


def zhurnal(D: dict, sid: str, o: dict, variant: str, tochka: dict) -> pd.DataFrame:
    """Журнал сделок (Ж-150); PnL в долях капитала (вес тикера 1/N): сумма «pnl_posle» = сумма excess исполнителя."""
    N, sl, dt = len(D["tk"]), gv.sila(D, sid), D["daty"]
    tf = {"S10": "1ч", "S11": "2ч", "S12": "согласие 1ч+2ч"}.get(sid, "")
    vh = "d1" if tochka["vhod"] == "d1" else f"T={tochka['T']}"
    rows = []
    for j, d, e, x, s, br, kom, zm, st in o["tr"]:
        v = float(sl[d, j]) if sl is not None else float("nan")
        rows.append(dict(sid=sid, variant=variant, tiker=D["tk"][j], data_vhoda=dt[e].date(), data_vyhoda=dt[x].date(),
                         uderzhanie_dn=(dt[x] - dt[e]).days, storona=s, tip_vhoda="рынок", tf=tf, vhod=vh, sila=v,
                         tercil_sily=tercil(sl, d, v) if sl is not None else "", pnl_do_izderzhek=br / N, komissiya=kom / N,
                         plata_za_short=zm / N, stavka=st / N, pnl_posle=(br - kom - zm - st) / N))
    return pd.DataFrame(rows)


def tablicy(D: dict, period: str, out: Path, n: int = gv.N_POVT) -> None:
    svod, k2, pl, zh = [], [], [], []
    for sid in gv.SECH + gv.IMPULS:
        o = gv.schet(D, gv.signaly(D, sid), **gv.shag(sid))
        zh.append(zhurnal(D, sid, o, "centr", gv.shag(sid)))
        svod.append({"strategiya": sid, "variant": "centr", **gv.stroka(o, D)})
        if sid in gv.SECH:
            svod.append({"strategiya": sid, "variant": "spravka_kniga", **gv.stroka(gv.kniga_spravka(D, sid), D)})
        for T in (gv.T_DOP if sid in gv.IMPULS else ()):
            tk = gv.shag(sid, "T", T)
            oT = gv.schet(D, gv.signaly(D, sid, None, tk), **tk)
            zh.append(zhurnal(D, sid, oT, f"vhod_T:{T}", tk))
            svod.append({"strategiya": sid, "variant": f"vhod_T:{T}", **gv.stroka(oT, D)})
        fakt = gv.mera(o, D)
        for tip, rows in (("k2", k2), ("placebo", pl)):
            v = np.array(kontrol(D, sid, tip, n))
            rows.append(dict(strategiya=sid, centr=fakt, mediana=float(np.nanmedian(v)), min=float(np.nanmin(v)), max=float(np.nanmax(v)),
                             dolya_ne_huzhe_centra=float(np.mean(v >= fakt)), p_perm=pk.p_perm(fakt, v), n=n))
    out.mkdir(parents=True, exist_ok=True)
    h = "# " + json.dumps(shapka(D, period), ensure_ascii=False) + "\n"
    for nm, rows in (("svod", svod), ("k2", k2), ("placebo", pl)):
        (out / f"{nm}_{period}.csv").write_text(h + pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
    (out / f"sdelki_{period}.csv").write_text(h + pd.concat(zh, ignore_index=True).to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True, choices=gv.PERIODY_OK)
    ap.add_argument("--out", default=str(gv.ROOT / "docs" / "tables" / "golaya_v7"))
    a = ap.parse_args()
    tablicy(gv.zagruzit(a.period), a.period, Path(a.out))
