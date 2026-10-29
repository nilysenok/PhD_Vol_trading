"""Эквивалентность Мак ↔ сервер (Ж-131 п. 3, Ж-133 п. 4): что считается и как сравнивается.

Набор: якорь A_fix каждой Sk на train4 (EvaluatorV2 в базовой точке этапа 1: `evr.evaluate_v2(evr.full({}), 'train4')`) и
N точек Соболя (seed 20261029, равномерно по Sk и индексу куба) медленным путём `progon_sobol.schitat_tochku` (fast = False).
Сравнение: якорь — max |Δ| ≤ 1e-12 по ряду портфеля и ряду тиков, маски NaN совпадают; точка — побайтно (ряд портфеля, ряд
тиков, строка ledger без t_sek/kod/kod_slit/klyuch_sha/excess/sloy — как IG-8 `progon_sobol.sverka`). Плюс: хэш кода fastbt и
хэш входов (`inp.digest()`) каждой Sk совпадают. Допуск не расширяется (Ж-131, толкование 3). Метрики не печатаются."""
import concurrent.futures as cf
import json
import multiprocessing as mp
import platform
import random

import numpy as np

SEED, N_TOCHEK, TOL_YAKOR = 20261029, 200, 1e-12


def _ps():
    from fastbt import progon_sobol as ps
    return ps


def vybor(n: int = N_TOCHEK, seed: int = SEED) -> list:
    """[(sid, i)] — n разных точек; индекс i < последнего индекса куба (prohod2 / prohod1 / N1)."""
    ps, rnd, out = _ps(), random.Random(seed), set()
    top = {}
    for sid in ps.SIDS:
        sb = ps.load_kub(sid)["sobol"]
        top[sid] = int(sb.get("prohod2") or sb.get("prohod1") or sb["N1"])
    while len(out) < n:
        sid = rnd.choice(ps.SIDS)
        out.add((sid, rnd.randrange(top[sid])))
    return sorted(out)


def _sig(r: dict) -> str:
    propusk = _ps().SVERKA_PROPUSK
    return json.dumps({k: v for k, v in r.items() if k not in propusk}, sort_keys=True, default=repr)


def _ryad(x) -> np.ndarray:
    return np.empty(0) if x is None else np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def _zadacha(z: tuple) -> tuple:
    """(sid, [i], yakor) -> ({imya: (sig, p, t)}, kod, vhody) в одном процессе (один оценщик на Sk)."""
    sid, idx, yakor = z
    ps = _ps()
    from fastbt import ledger_v2 as lm
    sr, c, out = ps.sreda(sid), ps.load_kub(sid), {}
    if yakor:
        r, p, t = sr.evr.evaluate_v2(sr.evr.full({}), ps.PERIOD)
        if p is None:
            raise RuntimeError(f"якорь A_fix {sid}: оценщик не дал ряда ({r})")
        out[f"yakor__{sid}"] = (_sig(r), _ryad(p), _ryad(t))
    for i in idx:
        _, pt = ps.tochki(c, i, i + 1)[0]
        r, p, t = ps.schitat_tochku(sr, sid, i, pt, "ekviv", False, False, lm.V21)
        out[f"tochka__{sid}__{i}"] = (_sig(r), _ryad(p), _ryad(t))
    return sid, out, sr.kod, sr.vhody


def poschitat(tochki: list, procs: int) -> tuple:
    """-> (rez {imya: (sig, p, t)}, meta {kod, vhody{sid}}). Задача — одна Sk (якорь + её точки)."""
    ps = _ps()
    zad = [(sid, [i for s, i in tochki if s == sid], True) for sid in ps.SIDS]
    rez, kod, vhody = {}, set(), {}
    if procs <= 1:
        it = map(_zadacha, zad)
    else:
        ex = cf.ProcessPoolExecutor(min(procs, len(zad)), mp_context=mp.get_context("fork"))
        it = ex.map(_zadacha, zad)
    for sid, out, k, v in it:
        rez.update(out)
        kod.add(k)
        vhody[sid] = v
    if procs > 1:
        ex.shutdown()
    if len(kod) != 1:
        raise RuntimeError(f"разный хэш кода в процессах: {sorted(kod)}")
    return rez, dict(kod=kod.pop(), vhody=vhody)


def sreda_meta() -> dict:
    import numba
    import pandas
    return dict(python=platform.python_version(), machine=platform.machine(), system=platform.system(),
                numpy=np.__version__, numba=numba.__version__, pandas=pandas.__version__)


def sohranit(path: str, rez: dict, meta: dict) -> None:
    arr = {"__meta__": np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)}
    for imya, (sig, p, t) in rez.items():
        arr[f"{imya}__sig"] = np.frombuffer(sig.encode(), dtype=np.uint8)
        arr[f"{imya}__p"], arr[f"{imya}__t"] = p, t
    with open(path, "wb") as f:
        np.savez(f, **arr)


def zagruzit(path: str) -> tuple:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z["__meta__"].tobytes().decode())
        imena = sorted({k.rsplit("__", 1)[0] for k in z.files if k != "__meta__"})
        rez = {m: (z[f"{m}__sig"].tobytes().decode(), z[f"{m}__p"], z[f"{m}__t"]) for m in imena}
    return rez, meta


def tochki_iz(rez: dict) -> list:
    return sorted((m.split("__")[1], int(m.split("__")[2])) for m in rez if m.startswith("tochka__"))


def _yakor_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    ok = ~np.isnan(a)
    return float(np.abs(a[ok] - b[ok]).max()) if ok.any() else 0.0


def _pervyj_indeks(a: np.ndarray, b: np.ndarray) -> str:
    if a.shape != b.shape or a.dtype != b.dtype or a.dtype != np.float64:
        return f"форма/тип {a.shape} {a.dtype} против {b.shape} {b.dtype}"
    k = int(np.flatnonzero(a.view(np.uint64) != b.view(np.uint64))[0])
    return f"элемент {k}: Мак {a[k]!r}, здесь {b[k]!r}"


def sravnit(eta: dict, meta_eta: dict, moi: dict, meta_moi: dict) -> tuple:
    """-> (ok, [расхождения по порядку], max |Δ| якоря). Первое расхождение — первым в списке."""
    bad, dmax = [], 0.0
    if meta_eta.get("kod") != meta_moi.get("kod"):
        bad.append(f"хэш кода fastbt: Мак {meta_eta.get('kod')}, здесь {meta_moi.get('kod')}")
    for sid in sorted(set(meta_eta.get("vhody", {})) | set(meta_moi.get("vhody", {}))):
        if meta_eta["vhody"].get(sid) != meta_moi["vhody"].get(sid):
            bad.append(f"хэш входов {sid}: Мак {meta_eta['vhody'].get(sid)}, здесь {meta_moi['vhody'].get(sid)}")
    if set(eta) != set(moi):
        bad.append(f"разный набор: нет здесь {sorted(set(eta) - set(moi))[:5]}, лишние {sorted(set(moi) - set(eta))[:5]}")
    for m in sorted(set(eta) & set(moi), key=lambda x: (not x.startswith("yakor"), x)):
        (s1, p1, t1), (s2, p2, t2) = eta[m], moi[m]
        if m.startswith("yakor__"):
            d = max(_yakor_delta(p1, p2), _yakor_delta(t1, t2))
            dmax = max(dmax, d)
            if not d <= TOL_YAKOR:
                bad.append(f"{m}: max |Δ| = {d:.3e} > {TOL_YAKOR:g}")
            continue
        for nazv, a, b in (("ряд портфеля", p1, p2), ("ряд тиков", t1, t2)):
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                bad.append(f"{m}: {nazv} не побайтно — {_pervyj_indeks(a, b)}")
        if s1 != s2:
            k = next((k for k in sorted(set(json.loads(s1)) | set(json.loads(s2)))
                      if json.loads(s1).get(k) != json.loads(s2).get(k)), "?")
            bad.append(f"{m}: строка ledger, поле {k}: Мак {json.loads(s1).get(k)!r}, здесь {json.loads(s2).get(k)!r}")
    return not bad, bad, dmax
