"""Эквивалентность Мак ↔ сервер (Ж-131 п. 3, Ж-133 п. 4): что считается. Хранение — ekviv_io.py, сравнение — ekviv_sravn.py.

Набор: якорь A_fix каждой Sk на train4 (`evr.evaluate_v2(evr.full({}), 'train4')`) и N точек Соболя (seed 20261029, фиксированные
индексы) медленным путём `progon_sobol.schitat_tochku` (fast = False). По каждой записи (dict):
  sig  — строка ledger без t_sek/kod/kod_slit/klyuch_sha/excess/sloy (json, для разбора);
  disk — она же с вещественными, заменёнными на null: дискретная часть (ключи, статусы, целые, строки);
  hex  — {путь: float.hex()} всех вещественных полей строки (Шарп, доходности, по годам и т. д.);
  p, t — ряд портфеля и ряд тиков (float64);
  pz   — |позиция| портфеля по дням (float64, непрерывная); sd — сделки портфеля по дням (int64);
  vi   — индексы входа/выхода по тикерам [(тикер, день, +1 вход / −1 выход)], выведены из |позиции| тикера (int64 [k, 3]).
Метрик не печатает."""
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import platform
import random

import numpy as np

SEED, N_TOCHEK = 20261029, 200
PERIOD_PO_DO = {"": "train4", "2021-12-31": "train4", "2026-02-28": "wf"}      # perenos_do из VERSIYA -> период эталона


def _ps():
    from fastbt import progon_sobol as ps
    return ps


def period_po_do(do) -> str:
    """Период эталона по `perenos_do` (VERSIYA / PERENOS_DO): пусто и 2021-12-31 — train4, 2026-02-28 — wf; иное — отказ."""
    do = (do or "").strip()
    if do not in PERIOD_PO_DO:
        raise ValueError(f"perenos_do = {do!r}: эталоны считаются для {sorted(k for k in PERIOD_PO_DO if k)} (или пусто = train4) — отказ")
    return PERIOD_PO_DO[do]


def rz_perioda(period: str):
    """train4 — прежний R0 (ключ среды и пути те же); wf — тот же прежний путь (setka progon, EvaluatorV2) с периодом wf."""
    ps = _ps()
    return ps.R0 if period == ps.PERIOD else ps.R0._replace(period=period)


def vybor(n: int = N_TOCHEK, seed: int = SEED) -> list:
    """[(sid, i)] — n разных точек; индекс i < последнего индекса куба (prohod2 / prohod1 / N1). Индексы фиксированы seed."""
    ps, rnd, out = _ps(), random.Random(seed), set()
    top = {}
    for sid in ps.SIDS:
        sb = ps.load_kub(sid)["sobol"]
        top[sid] = int(sb.get("prohod2") or sb.get("prohod1") or sb["N1"])
    while len(out) < n:
        sid = rnd.choice(ps.SIDS)
        out.add((sid, rnd.randrange(top[sid])))
    return sorted(out)


def _bez_vesh(x):
    if isinstance(x, float):
        return None
    if isinstance(x, dict):
        return {k: _bez_vesh(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_bez_vesh(v) for v in x]
    return x


def _hexy(x, put="", out=None) -> dict:
    out = {} if out is None else out
    if isinstance(x, float):
        out[put] = x.hex() if np.isfinite(x) else repr(x)
    elif isinstance(x, dict):
        [_hexy(v, f"{put}/{k}", out) for k, v in x.items()]
    elif isinstance(x, (list, tuple)):
        [_hexy(v, f"{put}/{i}", out) for i, v in enumerate(x)]
    return out


def _ryad(x) -> np.ndarray:
    return np.empty(0) if x is None else np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def _poz(evr) -> tuple:
    """(|pos| портфеля, сделки портфеля, [тикер, день, ±1]) последнего evaluate_v2; нет позиций → пустые массивы."""
    z = getattr(evr, "poz", None)
    if z is None:
        return np.empty(0), np.empty(0, np.int64), np.empty((0, 3), np.int64)
    pz, tp, sd, _ = z
    a = (np.asarray(tp, float) > 0).astype(np.int8)
    d = np.diff(np.concatenate([np.zeros((a.shape[0], 1), np.int8), a], axis=1), axis=1)
    ti, di = np.nonzero(d)
    vi = np.stack([ti, di, d[ti, di]], axis=1).astype(np.int64) if ti.size else np.empty((0, 3), np.int64)
    return _ryad(pz), np.rint(np.asarray(sd, float)).astype(np.int64), vi


def _zapis(r: dict, p, t, evr) -> dict:
    propusk = _ps().SVERKA_PROPUSK | {"vhody"}      # vhody — хэш float-массивов, зависит от libm/SVML; сверка Мак ↔ сервер его не берёт
    r = {k: v for k, v in r.items() if k not in propusk}
    pz, sd, vi = _poz(evr) if p is not None else _poz(None)
    return dict(sig=json.dumps(r, sort_keys=True, default=repr), disk=json.dumps(_bez_vesh(r), sort_keys=True, default=repr),
                hex=_hexy(r), p=_ryad(p), t=_ryad(t), pz=pz, sd=sd, vi=vi)


def _zadacha(z: tuple) -> tuple:
    """(sid, [i], yakor, period) -> (sid, {imya: запись}, kod, vhody, {kub_sha, daty_sha}) в одном процессе (один оценщик на Sk)."""
    sid, idx, yakor, period = z
    ps, rz = _ps(), rz_perioda(period)
    from fastbt import ledger_v2 as lm
    sr, c, out = ps.sreda(sid, rz), ps.load_kub(sid, rz), {}
    if yakor:
        r, p, t = sr.evr.evaluate_v2(sr.evr.full({}), period)
        if p is None:
            raise RuntimeError(f"якорь A_fix {sid}: оценщик не дал ряда ({r})")
        out[f"yakor__{sid}"] = _zapis(r, p, t, sr.evr)
    for i in idx:
        _, pt = ps.tochki(c, i, i + 1)[0]
        r, p, t = ps.schitat_tochku(sr, sid, i, pt, "ekviv", False, False, lm.V21)
        out[f"tochka__{sid}__{i}"] = _zapis(r, p, t, sr.evr)
    mf = ps.manifest(sid, sr, c, False, rz)
    return sid, out, sr.kod, sr.vhody, dict(kub_sha=mf["kub_sha"], daty_sha=mf["daty_sha"])


def poschitat(tochki: list, procs: int, period: str = "train4") -> tuple:
    """-> (rez {imya: запись}, meta {kod, vhody{sid}, kub{sid}, daty{sid}[, period]}). Задача — одна Sk (якорь + её точки).
    wf: только якорь A_fix (точки Соболя v5 на wf не считаются: tochki должен быть пуст); в meta period = wf (train4 — ключа нет, побайтно как раньше)."""
    ps = _ps()
    if period != ps.PERIOD and tochki:
        raise ValueError(f"период {period}: точки Соболя v5 считаются только на train4, на wf — якорь A_fix × 6 Sk")
    zad = [(sid, [i for s, i in tochki if s == sid], True, period) for sid in ps.SIDS]
    rez, kod, vhody, kub, daty = {}, set(), {}, {}, {}
    if procs <= 1:
        it = map(_zadacha, zad)
    else:
        ex = cf.ProcessPoolExecutor(min(procs, len(zad)), mp_context=mp.get_context("fork"))
        it = ex.map(_zadacha, zad)
    for sid, out, k, v, sh in it:
        rez.update(out)
        kod.add(k)
        vhody[sid], kub[sid], daty[sid] = v, sh["kub_sha"], sh["daty_sha"]
    if procs > 1:
        ex.shutdown()
    if len(kod) != 1:
        raise RuntimeError(f"разный хэш кода в процессах: {sorted(kod)}")
    meta = dict(kod=kod.pop(), vhody=vhody, kub=kub, daty=daty)
    return rez, meta if period == ps.PERIOD else dict(meta, period=period)


def net_fajla(rez: dict) -> list:
    """Записи, где оценщик поймал отсутствие файла (status = oshibka, FileNotFoundError): данных не хватает — не эквивалентность."""
    return sorted(m for m, r in rez.items() if "FileNotFoundError" in r["sig"] or "No such file" in r["sig"])


def sreda_meta() -> dict:
    import numba
    import pandas
    try:
        cpu = sorted(k for k, v in np._core._multiarray_umath.__cpu_features__.items() if v)
    except AttributeError:
        cpu = []
    return dict(python=platform.python_version(), machine=platform.machine(), system=platform.system(),
                numpy=np.__version__, numba=numba.__version__, pandas=pandas.__version__,
                numba_threads=os.environ.get("NUMBA_NUM_THREADS"), numba_layer=os.environ.get("NUMBA_THREADING_LAYER"),
                npy_disable=os.environ.get("NPY_DISABLE_CPU_FEATURES"), cpu=cpu)
