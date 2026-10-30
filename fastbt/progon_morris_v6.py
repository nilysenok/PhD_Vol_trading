"""М-драйвер (Ж-137 п. 2): Моррис по кубу v6 (~5 000 точек на Sk) → заморозка → точки полного прохода. Данных сам не читает (оценщик — ps.sreda).
Траектории и статистика — fastbt.morris_v6, правило — fastbt.morris_v6_zamorozka, куб и канон — canon_v6; счёт — механизм progon_server
(оценщик строится ОДИН раз в родителе, fork, flock на каталог, checkpoint.json, --prodolzhit). Y = дневной Шарп портфеля (metriki_e.sr_dnevnoy)
в трёх окнах ≤2016, ≤2017, ≤2018 из excess точки; ошибка оценщика / точка вне canon_v5 → Y = NaN (шаг не считается, число ошибок — в итоге).
Только готовые оси куба (canon_v6.GOTOVY): неподключённая ось в траектории дала бы ValueError на каждом шаге. Выход в <out>/<Sk>/:
morris_y.jsonl (Y по уникальным точкам), morris_<Sk>.csv, zamorozka.yaml. Полный проход — progon_morris_v6_polnyj.tochki_polnye(sid, zam). Запуск: python3 -m fastbt.progon_morris_v6 --kub v6 --period train4 --out <каталог> --procs N."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from fastbt import canon_v5 as c5, canon_v6, morris_v6 as mv, morris_v6_zamorozka as mz, progon_server as srv, progon_sobol as ps
from fastbt.progon_morris_v6_polnyj import zamknut
from fastbt.progon_morris_v6_status import DO, NAN3, STATUSY, baza_bez_sdelok, itog_nol, nol_sdelok, status_okna, vnimanie, y_okna  # noqa: F401 — re-export

CHUNK, YF = 50, "morris_y.jsonl"


def setka(sid: str):
    kub = canon_v6.load_kub(sid)
    canon_v6.proverit(kub)
    return kub, canon_v6.osi(kub)


def plan(sid: str, kub: dict, ox: list, r: int, mnozh: int, seed: int):
    """(I точек (r, k+1) → номер уникальной точки, порядок осей, [канон или None]); точки с равным каноном считаются один раз.
    Траектории — только по каноническим точкам (mv.traektorii_kanon: None-шаг заменяется другим допустимым уровнем оси или ось
    в траектории пропускается); правило не зависит от набора None-правил canon_v6."""
    L, memo = [len(g) for _, g, _ in ox], {}

    def kanon(idx):
        if idx not in memo:
            c = canon_v6.canon_v6(sid, {n: g[i] for (n, g, _), i in zip(ox, idx)}, kub)
            memo[idx] = (None if c is None else json.dumps(c, sort_keys=True, default=str), c)
        return memo[idx]

    baza = [i for i, (n, _, _) in enumerate(ox) if n in mv.OSI_BAZA]          # старт траектории: допуски v6 и фильтры v5 «выкл.»
    ind, por = mv.traektorii_kanon(L, r, lambda idx: kanon(idx)[0], seed, mnozh, baza=baza)
    uniq, pts, nom = {}, [], np.empty(ind.shape[:2], dtype=np.int64)
    for m, t in np.ndindex(nom.shape):
        key, c = kanon(tuple(int(i) for i in ind[m, t]))
        key = "None" if key is None else key
        if key not in uniq:
            uniq[key] = len(pts)
            pts.append(c)
        nom[m, t] = uniq[key]
    return nom, por, pts


def _kusok(zad: tuple) -> tuple:
    sid, a, b = zad
    sr, ys, ob, st = ps.sreda(sid, ps._W["rz"]), [], {}, dict.fromkeys(STATUSY, 0)
    for pt in ps._W["pts"][sid][a:b]:
        try:
            y, s = status_okna(sr, pt, ps._W["rz"].period, ps._W.get("fast", False))
            ys.append(y)
        except Exception as e:  # noqa: BLE001 — сбой точки не роняет прогон: Y = NaN, причины с числом — в итоге
            ys.append(list(NAN3))
            s, m = "oshibka", f"{type(e).__name__}: {e}"[:120]
            ob[m] = ob.get(m, 0) + 1
        st[s] += 1
    return a, b, ys, ob, st


def _prochitat(f: Path, start: int) -> list:
    """Закрытые куски jsonl до start (хвост после обрыва отбрасывается; файл переписывается чистым)."""
    rows = []
    for ln in f.read_text("utf-8").splitlines() if f.exists() else []:
        try:
            e = json.loads(ln)
        except ValueError:
            break
        if e["b"] <= start:
            rows.append(e)
    f.write_text("".join(json.dumps(e) + "\n" for e in rows), "utf-8")
    return rows


def schitat(sid: str, pts: list, sr, rz, out: Path, procs: int, otp: dict, prodolzhit: bool, fast: bool = False) -> tuple:
    """Y (n, 3) по уникальным точкам, причины исключений и счётчик статусов уникальных точек. Ложится в <out>/<sid>/: flock, checkpoint, morris_y.jsonl."""
    d, n = out / sid, len(pts)
    zm = srv.zamok(d)
    try:
        start = srv.nachalo(d, otp, prodolzhit)
        rows, ob = _prochitat(d / YF, start), 0
        zad = [(sid, a, min(a + CHUNK, n)) for a in range(start, n, CHUNK)]
        if zad:
            ps._W.clear()
            staryj, ps.fabrika = ps.fabrika, {sid: sr}.__getitem__
            gc.collect(), gc.freeze()
            fk = mp.get_context("fork")
            try:
                with cf.ProcessPoolExecutor(procs, mp_context=fk, initializer=ps._init,
                                            initargs=(dict(rz=rz, pts={sid: pts}, fast=fast), fk.Value("i", 0))) as ex, open(d / YF, "a") as f:
                    for a, b, ys, o, s in ex.map(_kusok, zad):          # по порядку: чекпоинт — непрерывный префикс
                        f.write(json.dumps({"a": a, "b": b, "y": ys, "ob": o, "st": s}) + "\n"), f.flush()
                        srv.pisat_chekpoint(d, otp, b - 1, n)
                        rows.append({"a": a, "b": b, "y": ys, "ob": o, "st": s})
            finally:
                gc.unfreeze()
                ps.fabrika = staryj
        Y = np.full((n, 3), np.nan)
        for e in rows:
            Y[e["a"]:e["b"]] = np.array(e["y"], dtype=np.float64)
        ob, st = {}, dict.fromkeys(STATUSY, 0)
        for e in rows:
            for m, k in e.get("ob", {}).items():
                ob[m] = ob.get(m, 0) + k
            for m, k in e.get("st", {}).items():
                st[m] += k
        return Y, ob, st
    finally:
        zm.close()


def spisok_osej(kub: dict, ox: list) -> list:
    nov = {n: sec for sec in canon_v6.NOVYE for n in kub.get(sec, {})}
    gr = {n: s.get("gruppa", sec) for sec in c5.SEKCII for n, s in kub["v5"].get(sec, {}).items() if isinstance(s, dict)}
    return [{"os": n, "gruppa": nov.get(n) or gr.get(n, "v5"), "novaya": n in nov} for n, _, _ in ox]


def zamorozka(kub: dict, ox: list, nom, por, Y, nol=None) -> tuple:
    """(stat по окнам, решение {ось: (заморозить, причина)}) по Y (уникальные точки, 3); nol — точки без сделок (Д4): пары из них не эффект."""
    stat = {}
    for w, okno in enumerate(mz.OKNA):
        stat[okno] = mv.statistika(mv.effekty(Y[nom, w], nom, por, None if nol is None else nol[nom]))
        stat[okno]["n_nol"] = np.zeros(por.shape[1], int) if nol is None else mv.pary_nol(nom, por, nol[nom])
    res = mz.reshenie(spisok_osej(kub, ox), stat)
    return stat, zamknut(kub, res)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def strategiya(sid: str, a, out: Path, rz, r: int | None, mnozh: int = mv.MNOZH) -> dict:
    kub, ox = setka(sid)
    r = r or 5000 // (len(ox) + 1)
    seed = int(kub["sobol"]["morris_seed"])
    nom, por, pts = plan(sid, kub, ox, r, mnozh, seed)
    ps._W.clear()
    sr = ps.sreda(sid, rz)
    kp = canon_v6.KUB6 / f"kub_{sid}.yaml"
    otp = dict(kod=sr.kod, vhody=sr.vhody, kub_sha=sha(kp), sobol_seed=seed, period=rz.period, k3=False,
               setka="v6:" + hashlib.sha256(json.dumps([n for n, _, _ in ox]).encode()).hexdigest()[:16], sobol_klyuch=f"morris_r{r}_m{mnozh}")
    Y, ob, st = schitat(sid, pts, sr, rz, out, a.procs, otp, a.prodolzhit, getattr(a, "fast", True))
    nol, nl = nol_sdelok(Y, st)
    bz = baza_bez_sdelok(nol, nom)
    stat, res = zamorozka(kub, ox, nom, por, Y, nol)
    akt = mz.pometki(spisok_osej(kub, ox), stat)
    d = out / sid
    sh = dict(sid=sid, period=rz.period, r=r, k=len(ox), mnozh=mnozh, seed=seed, tochek=len(pts), oshibok=sum(ob.values()), statusy=st,
              nol_sdelok=nl, baza_bez_sdelok=bz, kod=sr.kod, kub_sha=otp["kub_sha"])
    mz.zapisat_csv(d / f"morris_{sid}.csv", sh, spisok_osej(kub, ox), stat, res)
    zam = mz.zapisat_yaml(d / "zamorozka.yaml", time.strftime("%Y-%m-%d"), dict(kub_sha=otp["kub_sha"], kod=sr.kod, csv=sha(d / f"morris_{sid}.csv")),
                          res, st, akt, nl, bz)
    return dict(tochek=len(pts), oshibok=sum(ob.values()), prichiny=ob, zamorozheno=zam, statusy=st, osi_akt=akt, nol_sdelok=nl, baza_bez_sdelok=bz)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Моррис v6 → заморозка (zamorozka.yaml); полный проход берёт tochki_polnye")
    ap.add_argument("--kub", required=True, choices=["v6"])
    ap.add_argument("--period", required=True, choices=["train4", "v2", "wf"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--procs", type=int, default=srv.procs_po_umolchaniyu())
    ap.add_argument("--strategii", default=",".join(ps.SIDS))
    ap.add_argument("--r", type=int, help="траекторий (по умолчанию 5000 // (k + 1)); 1 — малая проверка")
    ap.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                    help="быстрый слой (fast_etap1.patched(), как в progon_sobol) — по умолчанию; --no-fast — эталонный путь")
    ap.add_argument("--prodolzhit", action="store_true")
    a = ap.parse_args(argv)
    from fastbt import sobytiya_e1
    sobytiya_e1.trebovat_svod()
    rz = ps.Rezhim("v6", a.period, "seed", False, "t0d")
    ps.OCENSHCHIK_V5 = ("fastbt.eval_v6", "EvaluatorV6")           # setka != progon → ps.ocenshchik берёт этот класс
    for sid in a.strategii.split(","):
        i = strategiya(sid, a, a.out, rz, a.r)
        print(f"{sid}: точек {i['tochek']}, статусы {i['statusy']}, исключений {i['oshibok']}, заморожено {len(i['zamorozheno'])}: {i['zamorozheno']}")
        for p in (mz.MERTVAYA, mz.NET_SDELOK, mz.MALO):
            print(f"  {p}: {[k for k, v in i['osi_akt'].items() if v['pometka'] == p]}")
        if i.get("nol_sdelok"):
            print(itog_nol(sid, i["nol_sdelok"], i["baza_bez_sdelok"]))
        [print(f"  ошибка ×{k}: {m}") for m, k in i["prichiny"].items()]
    return 0


if __name__ == "__main__":
    sys.exit(main())
