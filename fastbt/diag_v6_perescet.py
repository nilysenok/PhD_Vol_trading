"""Диагностика v6 train4 (Ж-145), лист Д-4: пересчёт точки куба v6 на Маке по её номеру Соболя. Данных ≥ 2020 не читает.
Запуск: FASTBT_MAIN=<основная копия> python3 -m fastbt.diag_v6_perescet <точки.csv sid,seed,i> <ledger запуска> <каталог Морриса> [кэш].
Точка — `tochki_polnye(sid, zam, i, i+1, seed, kub)` + якоря v5, как в progon_v6.tochki; `klyuch` обязан совпасть со строкой ledger
(иначе ValueError до счёта). Считает тем же слоем (fast/slow), что строка. Журнал сделок — из Trade оценщика: типа входа, ранга и силы
на входе в Trade нет (тип входа — свойство точки, `vhod_tip`); плата за шорт — рядом, рядами `zaem` по ставкам. Вклады в x делятся на cnt."""
from __future__ import annotations

import csv
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from fastbt import cache, ledger_v2 as lm, metriki_e as me, progon_sobol as ps, progon_morris_v6_polnyj as pm, progon_v6 as v6
from fastbt.diag_v6_chtenie import ChitatelPort, DOPUSK

KESH = Path("/private/tmp/claude-501/-Users-nilysenok-Desktop--------PhD-Vol-trading/bd772d77-32ea-43cd-b4a4-016ac48df19f/scratchpad/diag_v6/kesh")
PERIOD, Y0 = "train4", 2016


@lru_cache(maxsize=8)
def _chitatel(koren: str, sid: str) -> ChitatelPort:
    return ChitatelPort(Path(koren), sid)


def _zam(zam) -> tuple:
    """Каталог Морриса (<каталог>/<Sk>/zamorozka.yaml) или набор осей → tuple осей; разбор sid — снаружи."""
    return tuple(sorted(zam))


def tochka(sid: str, seed: str, i: int, zam) -> tuple:
    """(pt, kub, rz): точка № i Соболя v6 с якорями v5; zam — оси, вынутые Моррисом."""
    rz = ps.Rezhim("v6", PERIOD, seed, False, None, "sobol", None, ())
    kub = ps.load_kub(sid, rz)
    rz = rz._replace(fdr=kub.get("fdr"), zam=((sid, None, _zam(zam)),))
    pts = pm.tochki_polnye(sid, _zam(zam), i, i + 1, seed, kub)[0]
    if len(pts) != 1:
        raise ValueError(f"{sid} i={i}: точка не получена (выброшена каноном)")
    return {**kub["v5"].get("yakorya_progon", {}), **pts[0][1]}, kub, rz


def stroka(koren, sid: str, i: int) -> dict:
    rs = [r for r in _chitatel(str(koren), sid).rows() if r.get("sobol_i") == i]
    if len(rs) != 1:
        raise ValueError(f"{sid} i={i}: в ledger {len(rs)} строк sobol_i (ждали 1)")
    return rs[0]


def proverit_klyuch(sid: str, seed: str, i: int, koren, zam) -> tuple:
    """G4.1: klyuch пересобранной точки = klyuch строки; иначе ValueError. → (pt, rz, строка)."""
    pt, kub, rz = tochka(sid, seed, i, zam)
    r = stroka(koren, sid, i)
    kl = lm.klyuch(sid, pt, PERIOD, ps.KOMISSIYA, rz.fdr)
    if kl != r["klyuch"] or lm.klyuch_n(sid, kl) != r["klyuch_n"]:
        raz = sorted(k for k in set(kl) | set(r["klyuch"]) if kl.get(k) != r["klyuch"].get(k))
        raise ValueError(f"{sid} i={i}: klyuch не совпал со строкой ledger, оси: {raz}")
    return pt, rz, r


def _sovp(a: np.ndarray, b: np.ndarray) -> float:
    """max|Δ| с nan на тех же местах; nan в разных местах — inf."""
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    return 0.0 if np.isnan(a).all() else float(np.nanmax(np.abs(a - b)))


def _seti(sr):
    """(df, inv, eng, rs, keep, col): строки движка после evaluate_v2 — как etap1_eval_v2._series."""
    df, inv, eng, _, rs = sr.evr._last
    keep = np.isin(df.index.year, lm.gody_perioda(PERIOD))
    return df, inv, eng, rs, keep, np.cumsum(keep) - 1


def zhurnal(sr, comm: float) -> tuple:
    """(сделки, tik_poz [17, T] знаковые позиции, ne_otneseno [T]). Строка сделки — вклад в x = (g − комиссия)/cnt по строкам ТОЛЬКО
    своего интервала [вход; выход]; g вне интервалов сделок и комиссия строк без события — ne_otneseno (по датам, в x-единицах).
    Комиссия строки делится поровну между событиями в ней: сумма точна, разбивка по сделкам — приближённая (ограничение)."""
    df, inv, eng, rs, keep, col = _seti(sr)
    out, tp, ne, lo = [], np.zeros((len(rs), int(keep.sum()))), np.zeros(int(keep.sum())), 0
    for ti, r in enumerate(rs):
        t = sr.tikery[ti]
        dt, b = sr.evr.inp.meta[t]["dates"], sr.evr._b(t, PERIOD)
        a = int(np.searchsorted(dt[:b + 1], np.datetime64(f"{Y0}-01-01")))
        sl = slice(lo, lo + len(r.g))
        lo = sl.stop
        di, cn = inv[sl], eng.cnt[inv[sl]]
        ok = keep[di]
        g_ost, k_ost = r.g / cn, r.sides * comm / cn * ok          # остатки строки после ног (в x-единицах)
        for leg in _NOGI.get(id(r), [r]):                          # лестница: ноги по w/2 считаются отдельно (иначе g двух ног на одном интервале)
            sg, g_ost, k_ost = _noga(out, leg, (t, dt, a, b), cn, ok, comm, (g_ost, k_ost))
            tp[ti, col[di][ok]] += (leg.expo * sg)[ok]
        np.add.at(ne, col[di[ok]], (g_ost - k_ost)[ok])            # g вне ног и комиссия строк без события
    return out, tp, ne


def _noga(out, leg, tdab, cn, ok, comm, ost) -> tuple:
    """Сделки ноги (родители — g и комиссия; частичные выходы и добавки — отдельными строками со ссылкой `roditel`). → (знак, остатки)."""
    t, dt, a, b = tdab
    g_ost, k_ost = ost
    rk, nsob = leg.sides * comm / cn * ok, np.zeros(len(leg.g))      # комиссия строки делится поровну между событиями (входы/выходы) в ней
    sg, own, sob, idx, vl = np.zeros(len(leg.g)), np.zeros(len(leg.g), bool), [], {}, np.full(len(leg.g), -1)
    for n, tr in enumerate(leg.trades):
        if tr.parent is not None:
            continue
        k0, kg, ke = max(tr.row0, a) - a, _k_dohoda(tr, b) - a, _k_vyhoda(tr, leg, a, b)
        if kg < k0:
            continue
        ks = np.arange(k0, kg + 1)
        sg[ks], own[ks] = tr.side, True
        nsob[k0] += 1
        nsob[ke] += 1
        sob.append((k0, ke, len(out)))
        idx[n], vl[ks] = len(out), len(out)               # строка-владелец: позднейшая сделка перекрывает (как wr в run_e1)
        out.append(_zapis(t, dt, tr, b, float(leg.sides[k0]) or 1.0, 0.0, 0.0))
    for k0, ke, j in sob:
        out[j]["komissiya"] = float(rk[k0] / nsob[k0] + rk[ke] / nsob[ke])
        out[j]["pnl_do_izderzhek"] = float(np.sum((leg.g / cn)[(vl == j) & ok]))
    for tr in (x for x in leg.trades if x.parent is not None and x.parent in idx):   # ноги-дочерние: g/комиссия уже в родителе (gross × вес)
        z = _zapis(t, dt, tr, b, out[idx[tr.parent]]["ves"], 0.0, 0.0)
        out.append({**z, "roditel": idx[tr.parent], "frac": tr.frac, "vhodit_v_roditelya": True})
    g_ost = g_ost - np.where(own, leg.g / cn, 0.0)
    return sg, g_ost, k_ost - np.where(nsob > 0, rk, 0.0)


def _k_dohoda(tr, b: int) -> int:
    """Последняя строка дохода сделки: строка перед днём выхода (строка t — интервал [close t; close t+1))."""
    return b - 1 if tr.exit_day is None else min(tr.exit_day - 1, b - 1)


def _k_vyhoda(tr, r, a: int, b: int) -> int:
    """Строка комиссии выхода (локальная): день выхода (по close) или предыдущий — тот, где у ряда sides есть стороны."""
    if tr.exit_day is None:
        return b - 1 - a
    k1, k0 = min(tr.exit_day, b - 1) - a, min(tr.exit_day - 1, b - 1) - a
    return k1 if r.sides[k1] > 0 and not r.sides[k0] > 0 else k0 if r.sides[k0] > 0 else k1


def _zapis(t, dt, tr, b: int, w: float, g: float, kom: float) -> dict:
    d = lambda k: None if k is None else str(dt[k])[:10]   # noqa: E731
    return dict(tiker=t, d_sig=d(tr.d_sig), vhod=d(tr.entry_day), vyhod=d(tr.exit_day), storona=tr.side, prichina=tr.reason,
                uderzhanie_dn=(b if tr.exit_day is None else tr.exit_day) - tr.entry_day, px_in=tr.entry_px, px_out=tr.exit_px, ves=w,
                pnl_do_izderzhek=g, komissiya=kom, fee_in=tr.fee_in, fee_out=tr.fee_out)


_NOGI: dict = {}     # id(строки ряда тикера) → ноги лестницы (E1Rows), пока идёт пересчёт точки


def _zahvat(mod, imya: str):
    """Подмена mod.sum_rows: запоминает ноги, из которых сложена строка (оригинал вызывается как есть)."""
    orig = getattr(mod, imya)

    def f(rs):
        res = orig(rs)
        _NOGI[id(res)] = list(rs)
        return res
    return orig, f


def perescet(sid: str, seed: str, i: int, koren, zam, kesh=KESH) -> dict:
    """Пересчёт точки i: G4.1 → счёт → суммы port, x, tik_poz, журнал, метрики, кэш .npz. → dict."""
    pt, rz, r0 = proverit_klyuch(sid, seed, i, koren, zam)
    ps.fabrika, sr = None, ps.sreda(sid, rz)
    from fastbt import etap1_eval_v2, eval_v6_ispoln
    _NOGI.clear()
    mods = [(m, *_zahvat(m, "sum_rows")) for m in (etap1_eval_v2, eval_v6_ispoln)]
    for m, _, f in mods:
        m.sum_rows = f
    try:
        row, x, _ = ps.schitat_tochku(sr, sid, i, pt, "perescet", False, r0.get("sloy") == "fast", lm.V22, rz)[0:3]
    finally:
        for m, orig, _ in mods:
            m.sum_rows = orig
    if row["status"] != "ok":
        raise ValueError(f"{sid} i={i}: статус {row['status']}: {row.get('prichina')}")
    pp, _, sd, _ = sr.evr.poz
    port = lm.summy(sr.daty, x, pp, sd, lm.gody_perioda(PERIOD))
    serv = _chitatel(str(koren), sid).summy([r0])[0]
    jr, tp, ne = zhurnal(sr, ps.KOMISSIYA)
    df, inv, eng, rs, keep, col = _seti(sr)
    kap = (df["excess"] - df["r"]).to_numpy()[keep]
    res = dict(sid=sid, seed=seed, i=i, klyuch_ok=True, klyuch_n=r0["klyuch_n"], sloy=row["sloy"] if "sloy" in row else r0.get("sloy"),
               x=x, daty=sr.daty, tik_poz=tp, zhurnal=jr, ne_otneseno=ne, kapital=kap, port=port,
               d_port=_sovp(port, serv), kod_mac=cache.code_hash(), kod_server=_chitatel(str(koren), sid).manifest.get("kod"),
               sum_zhurnal=sum(t["pnl_do_izderzhek"] - t["komissiya"] for t in jr) + float(ne.sum() + kap.sum()), sum_x=float(x.sum()),
               vhod_tip=sr.evr.v6.get("vhod_tip"), msr_dn=me.msr(x), **me.metriki_ned(x, sr.daty), zaem_baza=np.asarray(sr.evr.zaem, float))
    _kesh(res, Path(kesh))
    return res


def _kesh(res: dict, kesh: Path) -> None:
    import json
    kesh.mkdir(parents=True, exist_ok=True)
    nm = f"{res['sid']}_{res['seed']}_{res['i']:06d}"
    np.savez_compressed(kesh / f"{nm}.npz", x=res["x"], daty=res["daty"], tik_poz=res["tik_poz"], port=res["port"], ne_otneseno=res["ne_otneseno"],
                        kapital=res["kapital"], zaem_baza=res["zaem_baza"])
    meta = {k: v for k, v in res.items() if isinstance(v, (str, int, float, bool, type(None)))}
    (kesh / f"{nm}.json").write_text(json.dumps({**meta, "zhurnal": res["zhurnal"]}, ensure_ascii=False, default=float), encoding="utf-8")


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) < 3:
        print(__doc__)
        return 2
    for sid, seed, i in csv.reader(open(a[0], encoding="utf-8")):
        zam = v6.zamorozka(a[2], [sid], lambda s: ps.kub_put(s, ps.Rezhim("v6", PERIOD, seed, False, None)))[0][2]
        r = perescet(sid, seed, int(i), a[1], zam, a[3] if len(a) > 3 else KESH)
        print(sid, seed, i, "d_port", r["d_port"], "Σжурнал−Σx", r["sum_zhurnal"] - r["sum_x"], "mSR дн/нед", r["msr_dn"], r["msr_ned"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
