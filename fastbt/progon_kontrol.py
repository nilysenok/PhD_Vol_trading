"""Л-9: плацебо, контроли и возмущения для центров регионов (docs/otchety/2026-10-28_progon_etap1.md §1.2, §6; Ж-078).

Строки идут в отдельный каталог `<ledger>/kontrol/` (istochnik = kontrol | vozm) и не входят ни в N, ни в PBO.
Оценщик — любой объект с `evaluate_v2(pt, period, vozm)`, `sid`, `comm`, `fdr`, `inp.sigma/meta` (EvaluatorV2, Л-5).
Все цифры — разработка 2016–2019, потолок: контроли считаются на тех же данных, что и отбор.
↺ `klyuch` строк контроля дополнен полями `kontrol` и `vozm` (иначе плацебо/к2 одной точки дают один klyuch_sha).
"""
from __future__ import annotations

import json
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import ledger_v2

SEED, N_PLACEBO, N_K2, MIN_LAG, MAX_LAG = 0, 10, 20, 63, 1000
VOZM = ({"sigma_mult": 0.9}, {"sigma_mult": 1.1}, {"bar_plus": 1}, {"stop_tik": 1}, {"komissiya": 0.0004}, {"komissiya": 0.0005})
ZNACHENIE = "mediana_kv"


def runs_of(sig) -> list:
    """Сделки Sk: максимальные отрезки постоянного ненулевого значения → (старт, длина, значение)."""
    v = np.asarray(sig, float)
    if not len(v):
        return []
    st = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
    return [(int(s), int(e - s), float(v[s])) for s, e in zip(st, np.r_[st[1:], len(v)]) if v[s] != 0]


def _place(lens, size, rng):
    """Старты без пересечений, с зазором ≥ 1 и отступом от краёв года; None — не помещаются."""
    n = len(lens)
    free = size - 2 - sum(lens) - (n - 1)
    if free < 0:
        return None
    cuts, order = np.sort(rng.integers(0, free + 1, n)), rng.permutation(n)
    out, acc = [0] * n, 0
    for k, i in enumerate(order):
        out[i] = 1 + int(cuts[k]) + acc + k
        acc += lens[i]
    return out


def placebo_sig_fn(seed: int):
    """sig_fn плацебо: в каждом тикеро-годе старты сделок — случайные дни того же года; число, стороны, длительности те же."""
    def fn(t, sig, dates):
        sig, yr, out = np.asarray(sig, float), pd.DatetimeIndex(dates).year.to_numpy(), np.zeros(len(sig))
        for y in np.unique(yr):
            lo = int(np.flatnonzero(yr == y)[0])
            size = int((yr == y).sum())
            rs = [r for r in runs_of(sig) if yr[r[0]] == y]
            pos = _place([r[1] for r in rs], size, np.random.default_rng([seed, zlib.crc32(t.encode()), int(y)])) if rs else []
            for (s, ln, val), p in zip(rs, pos if pos is not None else [r[0] - lo for r in rs]):
                out[lo + p:lo + p + ln] = val
        return out
    return fn


def k2_lags(n: int = N_K2, seed: int = SEED, lo: int = MIN_LAG, hi: int = MAX_LAG) -> list:
    return sorted(int(x) for x in np.random.default_rng(seed).integers(lo, hi + 1, n))


def k2_sig_fn(lag: int):
    """к2: циклический сдвиг позиций тикера на lag дней (lag = 0 — исходный ряд, mutations.md № 9); lag ≥ 63 от краёв."""
    def fn(t, sig, dates):
        sig = np.asarray(sig, float)
        n = len(sig)
        k = lag if lag == 0 or n - 2 * MIN_LAG <= 0 else MIN_LAG + (lag - MIN_LAG) % max(n - 2 * MIN_LAG + 1, 1)
        out = sig[(np.arange(n) - k) % n] if n else sig.copy()
        if k:
            out[-1] = 0.0                                                    # день b — только close для строки b−1
        return out
    return fn


@contextmanager
def sigma_const(ev, key: str, period: str = "train4"):
    """к1: σ тикера = безусловное среднее σ̂ за прогрев (год до окна); ряд подменяется на время счёта."""
    from fastbt.etap1_eval import PERIODS
    store, old = ev.inp.sigma[key], dict(ev.inp.sigma[key])
    try:
        for t, arr in old.items():
            yr = pd.DatetimeIndex(ev.inp.meta[t]["dates"]).year.to_numpy()
            w = np.asarray(arr, float)[:len(yr)][yr == PERIODS[period][0] - 1]
            store[t] = np.full(len(arr), np.nanmean(w) if np.isfinite(w).any() else np.nan)
        yield
    finally:
        store.update(old)


@dataclass
class Zadacha:
    istochnik: str
    kontrol: str | None
    vozm: dict | None          # объявленное возмущение (в строку)
    pt: dict
    ev_vozm: dict              # то, что уходит в evaluate_v2 (там же sig_fn)
    k1: bool = False
    zaglyad: bool = False


def plan(pt: dict, seeds=range(N_PLACEBO), lags=None) -> list:
    from fastbt.etap1_eval_v2 import NOVYE
    z = [Zadacha("kontrol", f"placebo:{s}", None, pt, {"sig_fn": placebo_sig_fn(s)}) for s in seeds]
    z.append(Zadacha("kontrol", "k1", None, pt, {}, k1=True))
    z += [Zadacha("kontrol", f"k2:lag{g}", None, pt, {"sig_fn": k2_sig_fn(g)}) for g in (k2_lags() if lags is None else lags)]
    z.append(Zadacha("kontrol", "k3", None, {**pt, **NOVYE}, {}))
    z.append(Zadacha("kontrol", "sigma_mediana", None, pt, {"sigma_lim_mediana": True}, zaglyad=True))
    return z + [Zadacha("vozm", None, dict(v), pt, dict(v)) for v in VOZM]


def p_perm(fakt: float, vals) -> float:
    """p = (1 + #{контроль ≥ факт}) / (n + 1); nan контроля не ≥. Плацебо при n = 10: минимум 1/11 ≈ 0,09 — только в отчёт."""
    v = np.asarray(list(vals), float)
    return float("nan") if not np.isfinite(fakt) else (1 + int((v >= fakt).sum())) / (len(v) + 1)


def _row(ev, row, z, kod, vhody, kod_slit):
    kl = ledger_v2.klyuch(ev.sid, z.pt, row["period"], row["komissiya"], ev.fdr)
    kl.update(kontrol=z.kontrol or "-", vozm=json.dumps(z.vozm, sort_keys=True) if z.vozm else "-")
    row = {**row, "strategiya": ev.sid, "klyuch": kl, "kod": row.get("kod", kod), "vhody": row.get("vhody", vhody),
           "kod_slit": kod_slit, "istochnik": z.istochnik, "sobol_i": None, "kontrol": z.kontrol, "vozm": z.vozm, "sloy": "slow"}
    row["klyuch_sha"] = ledger_v2.klyuch_sha(kl, row["kod"], row["vhody"])
    if z.zaglyad:
        row["flagi"] = {**row.get("flagi", {}), "zaglyadyvanie": True}
    return row


def run_kandidat(ev, pt: dict, pisatel, period: str = "train4", kod_slit: bool = False, **kw) -> list:
    """Базовая точка (не пишется) + все контроли и возмущения (пишутся); [(Zadacha, row)]."""
    from fastbt import cache
    kod, vhody = cache.code_hash(), ev.inp.digest()
    out = [(Zadacha("baza", None, None, pt, {}), ev.evaluate_v2(pt, period)[0])]
    for z in plan(pt, **kw):
        if z.k1:
            with sigma_const(ev, pt["sigma"], period):
                row, x, tik = ev.evaluate_v2(z.pt, period, z.ev_vozm)
        else:
            row, x, tik = ev.evaluate_v2(z.pt, period, z.ev_vozm)
        row = _row(ev, row, z, kod, vhody, kod_slit)
        pisatel.add(row, x, tik)
        from fastbt.tejk_signal import pisat_diag
        pisat_diag(pisatel, row, getattr(ev, "diag", None) if row["status"] == "ok" else None)                 # Ж-087
        out.append((z, row))
    return out


def kontrol_pisatel(root, sid: str, potok: int, manifest: dict):
    """Pisatel только в подкаталоге `kontrol/` (контроль в основной каталог не пишется — иначе попадёт в N и PBO)."""
    if Path(root).name != "kontrol":
        raise ValueError("контроли пишутся только в каталог kontrol/ (не в основной ledger)")
    return ledger_v2.Pisatel(root, sid, potok, manifest)


def tablica(sid: str, rezultat: dict, path) -> None:
    """kontrol_S{k}.csv: по кандидату — строка на контроль/возмущение; p плацебо и к2 — по группе."""
    rows = []
    for name, res in rezultat.items():
        fakt = next(r[ZNACHENIE] for z, r in res if z.istochnik == "baza")
        grp = {g: [r[ZNACHENIE] for z, r in res if (z.kontrol or "").startswith(g)] for g in ("placebo", "k2")}
        for z, r in res:
            g = (z.kontrol or "").split(":")[0]
            rows.append(dict(kandidat=name, strategiya=sid, istochnik=z.istochnik, kontrol=z.kontrol or "", vozm=json.dumps(z.vozm) if z.vozm else "",
                             status=r["status"], sharpe=r.get("sharpe"), mediana_kv=r.get(ZNACHENIE), dolya_pol_kv=r.get("dolya_pol_kv"),
                             sdelok=sum((r.get("sdelok_y") or {}).values()), p=p_perm(fakt, grp[g]) if g in grp else "",
                             zaglyadyvanie=bool(z.zaglyad)))
    head = ("# разработка 2016–2019, потолок: контроли на тех же данных, что и отбор; в N и PBO не входят\n"
            "# p плацебо при 10 повторах не меньше 0,09 — только в отчёт; sigma_mediana — с заглядыванием (медиана года)\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(head + pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")


def run_strategiya(ev, centers: dict, root, csv, period: str = "train4", potok: int = 0, T: int = 1010, kod_slit: bool = False, **kw) -> dict:
    """centers: {имя: точка} (≤ 20, центры регионов Л-7) → ledger `root` (…/kontrol) и таблица csv. Счёт — по команде дирижёра."""
    from fastbt import cache
    if len(centers) > 20:
        raise ValueError("кандидатов на стратегию ≤ 20")
    man = dict(T=T, period=period, komissiya=ev.comm, fdr=ev.fdr, kod=cache.code_hash(), vhody=ev.inp.digest(), kod_slit=kod_slit)
    pis = kontrol_pisatel(root, ev.sid, potok, man)
    try:
        res = {n: run_kandidat(ev, pt, pis, period, kod_slit, **kw) for n, pt in centers.items()}
    finally:
        pis.close()
    tablica(ev.sid, res, csv)
    return res
