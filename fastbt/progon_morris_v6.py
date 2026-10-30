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
from fastbt.metriki_e import sr_dnevnoy
from fastbt.progon_morris_v6_polnyj import zamknut

DO = dict(zip(mz.OKNA, ("2016-12-31", "2017-12-31", "2018-12-31")))
CHUNK, YF, NAN3 = 50, "morris_y.jsonl", [float("nan")] * 3


def setka(sid: str):
    kub = canon_v6.load_kub(sid)
    canon_v6.proverit(kub)
    return kub, canon_v6.osi(kub)


def plan(sid: str, kub: dict, ox: list, r: int, mnozh: int, seed: int):
    """(I точек (r, k+1) → номер уникальной точки, порядок осей, [канон или None]); точки с равным каноном считаются один раз."""
    L = [len(g) for _, g, _ in ox]
    ind, por = mv.traektorii(L, r, seed, mnozh)
    uniq, pts, nom = {}, [], np.empty(ind.shape[:2], dtype=np.int64)
    for m, t in np.ndindex(nom.shape):
        c = canon_v6.canon_v6(sid, {n: g[i] for (n, g, _), i in zip(ox, ind[m, t])}, kub)
        key = "None" if c is None else json.dumps(c, sort_keys=True, default=str)
        if key not in uniq:
            uniq[key] = len(pts)
            pts.append(c)
        nom[m, t] = uniq[key]
    return nom, por, pts


def y_okna(sr, pt, period: str) -> list:
    """Y трёх окон для точки; None-канон → NaN; excess None (нет сделок) → 0 (план §3.3: эффект оси, не пропуск)."""
    if pt is None:
        return list(NAN3)
    r, ex_p, _ = sr.evr.evaluate_v2(pt, period)
    if r.get("status", "ok") != "ok":
        return list(NAN3)
    if ex_p is None:
        return [0.0] * 3
    x = np.asarray(ex_p, dtype=np.float64)
    if len(x) != len(sr.daty):
        raise ValueError(f"excess {len(x)} дней, дат {len(sr.daty)}")
    out = []
    for d in DO.values():
        w = x[sr.daty <= np.datetime64(d)]
        out.append(sr_dnevnoy(w) if w.size > 1 and np.isfinite(w).all() else float("nan"))
    return out


def _kusok(zad: tuple) -> tuple:
    sid, a, b = zad
    sr, ys, ob = ps.sreda(sid, ps._W["rz"]), [], {}
    for pt in ps._W["pts"][sid][a:b]:
        try:
            ys.append(y_okna(sr, pt, ps._W["rz"].period))
        except Exception as e:  # noqa: BLE001 — сбой точки не роняет прогон: Y = NaN, причины с числом — в итоге
            ys.append(list(NAN3))
            m = f"{type(e).__name__}: {e}"[:120]
            ob[m] = ob.get(m, 0) + 1
    return a, b, ys, ob


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


def schitat(sid: str, pts: list, sr, rz, out: Path, procs: int, otp: dict, prodolzhit: bool) -> tuple:
    """Y (n, 3) по уникальным точкам и число ошибок. Ложится в <out>/<sid>/: flock, checkpoint, morris_y.jsonl."""
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
                                            initargs=(dict(rz=rz, pts={sid: pts}), fk.Value("i", 0))) as ex, open(d / YF, "a") as f:
                    for a, b, ys, o in ex.map(_kusok, zad):          # по порядку: чекпоинт — непрерывный префикс
                        f.write(json.dumps({"a": a, "b": b, "y": ys, "ob": o}) + "\n"), f.flush()
                        srv.pisat_chekpoint(d, otp, b - 1, n)
                        rows.append({"a": a, "b": b, "y": ys, "ob": o})
            finally:
                gc.unfreeze()
                ps.fabrika = staryj
        Y = np.full((n, 3), np.nan)
        for e in rows:
            Y[e["a"]:e["b"]] = np.array(e["y"], dtype=np.float64)
        ob = {}
        for e in rows:
            for m, k in e.get("ob", {}).items():
                ob[m] = ob.get(m, 0) + k
        return Y, ob
    finally:
        zm.close()


def spisok_osej(kub: dict, ox: list) -> list:
    nov = {n: sec for sec in canon_v6.NOVYE for n in kub.get(sec, {})}
    gr = {n: s.get("gruppa", sec) for sec in c5.SEKCII for n, s in kub["v5"].get(sec, {}).items() if isinstance(s, dict)}
    return [{"os": n, "gruppa": nov.get(n) or gr.get(n, "v5"), "novaya": n in nov} for n, _, _ in ox]


def zamorozka(kub: dict, ox: list, nom, por, Y) -> tuple:
    """(stat по окнам, решение {ось: (заморозить, причина)}) по Y (уникальные точки, 3)."""
    stat = {}
    for w, okno in enumerate(mz.OKNA):
        stat[okno] = mv.statistika(mv.effekty(Y[nom, w], nom, por))
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
    Y, ob = schitat(sid, pts, sr, rz, out, a.procs, otp, a.prodolzhit)
    stat, res = zamorozka(kub, ox, nom, por, Y)
    d = out / sid
    sh = dict(sid=sid, period=rz.period, r=r, k=len(ox), mnozh=mnozh, seed=seed, tochek=len(pts), oshibok=sum(ob.values()), kod=sr.kod, kub_sha=otp["kub_sha"])
    mz.zapisat_csv(d / f"morris_{sid}.csv", sh, spisok_osej(kub, ox), stat, res)
    zam = mz.zapisat_yaml(d / "zamorozka.yaml", time.strftime("%Y-%m-%d"), dict(kub_sha=otp["kub_sha"], kod=sr.kod, csv=sha(d / f"morris_{sid}.csv")), res)
    return dict(tochek=len(pts), oshibok=sum(ob.values()), prichiny=ob, zamorozheno=zam)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Моррис v6 → заморозка (zamorozka.yaml); полный проход берёт tochki_polnye")
    ap.add_argument("--kub", required=True, choices=["v6"])
    ap.add_argument("--period", required=True, choices=["train4", "v2", "wf"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--procs", type=int, default=srv.procs_po_umolchaniyu())
    ap.add_argument("--strategii", default=",".join(ps.SIDS))
    ap.add_argument("--r", type=int, help="траекторий (по умолчанию 5000 // (k + 1)); 1 — малая проверка")
    ap.add_argument("--prodolzhit", action="store_true")
    a = ap.parse_args(argv)
    from fastbt import sobytiya_e1
    sobytiya_e1.trebovat_svod()
    rz = ps.Rezhim("v6", a.period, "seed", False, "t0d")
    ps.OCENSHCHIK_V5 = ("fastbt.eval_v6", "EvaluatorV6")           # setka != progon → ps.ocenshchik берёт этот класс
    for sid in a.strategii.split(","):
        i = strategiya(sid, a, a.out, rz, a.r)
        print(f"{sid}: точек {i['tochek']}, ошибок {i['oshibok']}, заморожено {len(i['zamorozheno'])}: {i['zamorozheno']}")
        [print(f"  ошибка ×{k}: {m}") for m, k in i["prichiny"].items()]
    return 0


if __name__ == "__main__":
    sys.exit(main())
