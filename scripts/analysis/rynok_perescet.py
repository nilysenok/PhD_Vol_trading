# -*- coding: utf-8 -*-
"""Пересчёт строк vhod = rynok (t_vyhod ≠ close, status ok) новым кодом базы: rynok по open 10:00 d+1 (Ж-086 п. 6, #474).

Старый ledger (проходы 1–2, kod d4711aa5, rynok по close d) — только чтение. Новый — docs/tables/progon_e1/_ledger_v2_rynok/,
istochnik = rynok_open. Запуск с FASTBT_BARS1M=…/data/regen/D25/bars1m. Данных ≥ 2020 нет (train4 2016–2019).
  rynok_perescet.py count | probe | schet [--processov 4] | sverka | otchet
Целевая Ж-090: fastbt.tselevaya.tselevaya_train4 (mSR train4, плата за шорт 5 %, устойчивость = mSR каждого года > 0).
Отчёт — потолок: выбор и счёт на одних данных train4.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import ledger_v2 as lm  # noqa: E402
from fastbt import progon_sobol as ps  # noqa: E402
from fastbt.tselevaya import tselevaya_train4  # noqa: E402
from fastbt.zaem_v2 import zaem_ryad  # noqa: E402

STARYJ = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/docs/tables/progon_e1/_ledger_v2")
NOVYJ = ROOT / "docs/tables/progon_e1/_ledger_v2_rynok"
OTCHET = ROOT / "docs/otchety/2026-10-28_rynok_perescet.md"
IST, KUSOK, N_PROBA, N_SVERKA, SEED = "rynok_open", 40, 20, 20, 20261028
PROPUSK = ps.SVERKA_PROPUSK | {"istochnik", "excess", "vhody"}   # vhody = cache.key включает хэш кода (cache.py:38), как и kod


def rynok(r: dict) -> bool:
    return r["klyuch"]["vhod"] == "rynok" and r["klyuch"]["t_vyhod"] != "close"


def tablica(sid: str, idx) -> dict:
    """{i: pt} для индексов куба (один проход Соболя, чтобы не делать fast_forward на каждую точку); ключ ledger сверяется."""
    c, nuzhny = ps.load_kub(sid), set(idx)
    return {i: pt for i, pt in ps.tochki(c, 0, max(nuzhny) + 1) if i in nuzhny}


def vybrat(root: Path, pred) -> dict:
    """{sid: [(строка, ch)]} из старого ledger по предикату (status ok, оба прохода — один каталог)."""
    out = {}
    for sid in ps.SIDS:
        ch = lm.Chitatel(root, sid)
        out[sid] = [(r, ch) for r in ch.rows() if pred(r)]
    return out


def tochki_zadaniya(vyb: dict) -> dict:
    zad = {}
    for sid, rs in vyb.items():
        tb = tablica(sid, [r["sobol_i"] for r, _ in rs]) if rs else {}
        for r, _ in rs:
            if lm.klyuch(sid, tb[r["sobol_i"]], ps.PERIOD, ps.KOMISSIYA, None) != r["klyuch"]:
                raise ValueError(f"{sid} i={r['sobol_i']}: ключ точки Соболя ≠ ключу ledger")
        zad[sid] = [(r["sobol_i"], tb[r["sobol_i"]]) for r, _ in rs]
    return zad


def rabotnik(zad: tuple) -> tuple:
    sid, tochki = zad
    c, sr = ps._W["kubs"][sid], ps.sreda(sid)
    w = lm.Pisatel(ps._W["root"], sid, ps._W["nn"], ps.manifest(sid, sr, c, True))
    zak, n, ob = w.done(), 0, 0
    for i, pt in tochki:
        if ps.sha_tochki(sid, sr, pt) in zak:
            continue
        r, ex_p, ex_t = ps.schitat_tochku(sr, sid, i, pt, IST, True, False)
        ps.importlib.import_module("fastbt.zaem_v2").pisat(w, r, ex_p, ex_t, getattr(sr.evr, "zaem", None))
        ps.importlib.import_module("fastbt.tejk_signal").pisat_diag(w.pd, r, getattr(sr.evr, "diag", None) if r["status"] == "ok" else None)
        n, ob = n + 1, ob + (r["status"] == "oshibka")
    ps.importlib.import_module("fastbt.zaem_v2").zakryt(w)
    w.close()
    return sid, n, ob


def schet(zad: dict, root: Path, processov: int) -> int:
    if not 1 <= processov <= 4:
        raise ValueError("процессов 1…4")
    ps.proverit_disk(sum(map(len, zad.values())), root)
    ctx = dict(kubs={s: ps.load_kub(s) for s in zad}, root=root)
    zadachi = [(s, t[a:a + KUSOK]) for s, t in zad.items() for a in range(0, len(t), KUSOK)]
    fk, oshibok = mp.get_context("fork"), 0
    with cf.ProcessPoolExecutor(processov, mp_context=fk, initializer=ps._init, initargs=(ctx, fk.Value("i", 0))) as ex:
        for sid, n, ob in ex.map(rabotnik, zadachi):
            oshibok += ob
            print(f"{sid}: +{n}, ошибок {ob}", flush=True)
    return oshibok


def sverka() -> int:
    """20 случайных точек t_vyhod = close, vhod = лимит (не rynok): новый код = старая строка (кроме kod и служебных полей) и порт побайтно."""
    vyb = vybrat(STARYJ, lambda r: r["klyuch"]["t_vyhod"] == "close" and r["klyuch"]["vhod"] != "rynok")   # vhod = лимит: доли 0.25…1.0
    pool = [(s, r, ch) for s, rs in vyb.items() for r, ch in rs]
    vz = random.Random(SEED).sample(pool, N_SVERKA)
    tb = {s: tablica(s, [r["sobol_i"] for s2, r, _ in vz if s2 == s]) for s in {s for s, _, _ in vz}}
    bad = []
    for sid, r, ch in vz:
        pt = tb[sid][r["sobol_i"]]
        r2, ex_p, _ = ps.schitat_tochku(ps.sreda(sid), sid, r["sobol_i"], pt, r["istochnik"], r["kod_slit"], False)
        sig = lambda x: json.dumps({k: v for k, v in x.items() if k not in PROPUSK}, sort_keys=True)  # noqa: E731
        ok = sig(r) == sig(r2) and np.asarray(ch.matrica([r])[:, 0], float).tobytes() == np.asarray(ex_p, float).tobytes()
        if not ok:
            raz = sorted(k for k in r if k not in PROPUSK and r[k] != r2.get(k))
            bad.append((sid, r["sobol_i"], raz, np.asarray(ch.matrica([r])[:, 0], float).tobytes() == np.asarray(ex_p, float).tobytes()))
    print(f"сверка close/limit: {len(vz)} точек, расхождений {len(bad)} {bad}")
    return 1 if bad else 0


def cel(root: Path, ch, r: dict, daty) -> tuple:
    m, _, ust = tselevaya_train4(ch.matrica([r])[:, 0], daty, zaem_ryad(root, r["strategiya"], r))
    return m, ust


def otchet() -> int:
    from collections import Counter
    st, nv = [], []
    sha, razm = {}, 0
    for sid in ps.SIDS:
        old, new = lm.Chitatel(STARYJ, sid), lm.Chitatel(NOVYJ, sid, sverka_kod=True)
        daty = np.load(STARYJ / sid / "daty.npy")
        if old.manifest["daty_sha"] != new.manifest["daty_sha"] or old.manifest["T"] != new.manifest["T"]:
            raise ValueError(f"{sid}: даты старого и нового ledger различаются")
        stary = {r["sobol_i"]: r for r in old.rows() if rynok(r)}
        for r in new.rows(None):
            o = stary.get(r["sobol_i"])
            a = cel(STARYJ, old, o, daty) if o else (np.nan, False)
            b = cel(NOVYJ, new, r, daty) if r["status"] == "ok" else (np.nan, False)
            st.append((sid, r["status"], *a, *b))
        for f in sorted((NOVYJ / sid).rglob("*")):
            if f.is_file():
                razm += f.stat().st_size
                if f.name in ("manifest.json", "rows.jsonl") or f.name.startswith("SHA256SUMS"):
                    sha[str(f.relative_to(NOVYJ))] = hashlib.sha256(f.read_bytes()).hexdigest()
    A = np.array([(x[2], x[3], x[4], x[5]) for x in st], float)
    ok = np.array([x[1] == "ok" for x in st])
    q = lambda v: " / ".join(f"{x:.2f}" for x in np.nanpercentile(v, [5, 25, 50, 75, 95]))  # noqa: E731
    L = ["# Пересчёт строк vhod = rynok новым кодом (rynok по open d+1)", "",
         "Разработка train4 (2016–2019), после аудита, **потолок**: выбор и счёт на одних данных; в отбор не идёт. Ж-086 п. 6, #474.", "",
         f"- число строк vhod = rynok, t_vyhod ≠ close, status ok, проходы 1–2: **{len(st)}** "
         f"(по Sk: {dict(Counter(x[0] for x in st))}); пересчитано ok: {int(ok.sum())}, не ok: {int((~ok).sum())} "
         f"({dict(Counter(x[1] for x in st if x[1] != 'ok'))})",
         f"- целевая Ж-090: mSR train4 за весь период, плата за шорт 5 %, устойчивость = mSR каждого года > 0",
         "", "| | старые (rynok по close d) | новые (rynok по open d+1) |", "|---|---|---|",
         f"| mSR, квантили 5/25/50/75/95 % | {q(A[ok, 0])} | {q(A[ok, 2])} |",
         f"| устойчивых | {int(A[ok, 1].sum())} | {int(A[ok, 3].sum())} |",
         f"| устойчивых и mSR ≥ 1,5 | {int(((A[:, 0] >= 1.5) & (A[:, 1] > 0))[ok].sum())} | {int(((A[:, 2] >= 1.5) & (A[:, 3] > 0))[ok].sum())} |",
         f"| медиана Δ mSR (новая − старая) | | {np.nanmedian(A[ok, 2] - A[ok, 0]):.3f} |",
         f"| mSR не определён (nan) | {int(np.isnan(A[ok, 0]).sum())} | {int(np.isnan(A[ok, 2]).sum())} |",
         f"| строк, где mSR изменился | | {int((np.nan_to_num(A[ok, 0], nan=9e9) != np.nan_to_num(A[ok, 2], nan=9e9)).sum())} из {int(ok.sum())} |", "",
         f"Ledger не в git: `{NOVYJ}`, {razm / 1e6:.0f} МБ (старый: `{STARYJ}`). sha256:", ""]
    L += [f"- `{k}` {v}" for k, v in sha.items() if "/" not in k or k.endswith("manifest.json") or "SHA256SUMS" in k]
    OTCHET.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:14]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["count", "probe", "schet", "sverka", "otchet"])
    ap.add_argument("--processov", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "sverka":
        return sverka()
    if a.cmd == "otchet":
        return otchet()
    zad = tochki_zadaniya(vybrat(STARYJ, rynok))
    if a.cmd == "count":
        print({s: len(t) for s, t in zad.items()}, "всего", sum(map(len, zad.values())))
        return 0
    if a.cmd == "probe":
        rnd = random.Random(SEED).sample([(s, t) for s, ts in zad.items() for t in ts], N_PROBA)
        zp = {s: [t for s2, t in rnd if s2 == s] for s in zad}
        return 2 if schet(zp, NOVYJ / "_probe", 1) else 0
    return 2 if schet(zad, NOVYJ, a.processov) else 0


if __name__ == "__main__":
    sys.exit(main())
