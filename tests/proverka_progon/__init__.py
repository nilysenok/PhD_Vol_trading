"""П-1 (проверяющий): общие помощники тестов формата ledger v2.0 (контракт etap1 §1). Синтетика, без данных."""
import json

import numpy as np

T = 8
KOD, VHODY = "kod-A", "vhody-A"
TIKERY = [f"T{i:02d}" for i in range(17)]
KVARTALY = [f"{y}Q{q}" for y in range(2016, 2020) for q in range(1, 5)]

# Точка, не схлопываемая canon_v2 (k_s != net, vhod != rynok, sid S1)
BAZA = {
    "ma_window": 20, "z_entry": 2.0, "z_exit": 0.5,
    "q_sily": 0.0, "sigma": "rv22", "vhod": 0.5, "k_s": 2.0, "R": 2.0, "k_tr": "net",
    "bezubytok": "net", "tau": 5, "proverka": "zakrytie", "razmer": "ravnyy", "rezhim": "net",
    "ed_lim": "sigma", "zhizn": "den", "lestnica": "net", "gap": "vhod", "t_vyhod": "close",
    "k_x": "net", "storona": "obe", "filtr": "net",
}


def manifest(kod=KOD, sid="S1", t=T):
    return {"v": "2.0", "format": "etap1-ledger", "strategiya": sid, "period": "train4",
            "okno_s": "2016-01-01", "okno_po": "2019-12-31", "progrev_s": "2015-01-01",
            "komissiya": 0.0004, "fdr": None, "kod": kod, "vhody": VHODY, "kod_slit": False,
            "daty_sha": "x", "tikery": TIKERY, "T": t, "kub_sha": "x", "sobol_seed": 20261029,
            "sozdan": "2026-10-28"}


def kv(v=0.1):
    return {k: v for k in KVARTALY}


def row(lm, i, kod=KOD, status="ok", istochnik="sobol1", kontrol=None, **izm):
    """Строка §1.2; точка i отличается осью tau, ключ — через ledger_v2.klyuch."""
    pt = dict(BAZA, tau=float(i + 1))
    pt.update(izm)
    kl = lm.klyuch("S1", lm.canon_v2("S1", pt), "train4", 0.0004, "null")
    ok = status == "ok"
    nan = float("nan")
    return {"v": "2.0", "format": "etap1-ledger", "strategiya": "S1", "klyuch": kl,
            "klyuch_sha": lm.klyuch_sha(kl, kod, VHODY), "kod": kod, "vhody": VHODY, "kod_slit": False,
            "period": "train4", "okno_s": "2016-01-01", "okno_po": "2019-12-31",
            "progrev_s": "2015-01-01", "zaderzhka": "d+1", "chestnaya": "net", "komissiya": 0.0004,
            "fdr": None, "status": status, "prichina": None if ok else "test", "istochnik": istochnik,
            "sobol_i": i if istochnik.startswith("sobol") else None, "kontrol": kontrol, "vozm": None,
            "by_year": {str(y): 0.5 for y in range(2016, 2020)} if ok else {},
            "msr_by_year": {str(y): 0.4 for y in range(2016, 2020)} if ok else {},
            "by_quarter": kv(0.1 if ok else nan),
            "mediana_kv": 0.1 if ok else nan, "dolya_pol_kv": 1.0 if ok else nan,
            "by_quarter_zaem": {"5": kv(), "10": kv()},
            "sdelok_y": {str(y): 3 for y in range(2016, 2020)}, "storony_y": {}, "sdelok_kv": {},
            "dolya_ispolneniya_limita": 0.5, "prichiny": {}, "sharpe": 0.5, "msr": 0.4, "sortino": 1.0,
            "kalmar": 1.0, "mdd": 0.1, "T": T, "skew": 0.0, "kurt": 3.0,
            "flagi": {"vyrozhdena": False, "kray": [], "short_shag1_kray": False},
            "excess": {"potok": "p00", "shard": 0, "stroka": i} if ok else None, "t_sek": 0.1,
            "sloy": "slow"}


def ryady(i, t=T):
    """Ряды точки i: tik float32 [17, T]; port = nanmean по тикерам (float64) — тождество §1.4."""
    rng = np.random.default_rng(i)
    tik = rng.normal(0, 0.01, (17, t)).astype(np.float32)
    return np.nanmean(tik.astype(np.float64), axis=0), tik


def pisat(lm, root, nrows, shard=4, kod=KOD, potok=0, start=0, zakryt=True):
    from pathlib import Path
    root = Path(root)
    (root / "S1").mkdir(parents=True, exist_ok=True)
    np.save(root / "S1" / "daty.npy", np.arange(np.datetime64("2016-01-04"), np.datetime64("2016-01-04") + T))
    w = lm.Pisatel(root, "S1", potok, manifest(kod), shard=shard)
    for i in range(start, start + nrows):
        w.add(row(lm, i, kod), *ryady(i))
    if zakryt:
        w.close()
    return w


def stroki_jsonl(root, potok=0):
    p = root / "S1" / f"p{potok:02d}" / "rows.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
