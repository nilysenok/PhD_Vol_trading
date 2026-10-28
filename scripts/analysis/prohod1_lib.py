"""Разбор прохода 1 (Ж-078/082/090): чтение ledger v2 (только чтение), счётчики строк, распределение целевой.
Разработка 2016–2019 (train4), потолок. Данных ≥ 2020 не читает. Основная копия — только на чтение."""
from __future__ import annotations

import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get("FASTBT_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))
os.environ.setdefault("FASTBT_MAIN", str(MAIN))
sys.path.insert(0, str(ROOT))

from fastbt.ledger_v2 import Chitatel  # noqa: E402
from fastbt.progon_osnovnaya import chisla_N, maska  # noqa: E402
from fastbt.tselevaya import GODA_TRAIN4, STAVKA_SHORT, msr_i_gody, tselevaya_train4  # noqa: E402
from fastbt.zaem_v2 import SUB, SUMS  # noqa: E402

LEDGER = MAIN / "docs/tables/progon_e1/_ledger_v2"
GODY = GODA_TRAIN4
SIDS = [f"S{k}" for k in range(1, 7)]


def daty_sid(sid: str, ch: Chitatel) -> np.ndarray:
    d = np.load(LEDGER / sid / "daty.npy").astype("datetime64[D]")
    if hashlib.sha256(d.tobytes()).hexdigest() != ch.manifest["daty_sha"]:
        raise ValueError(f"{sid}: daty.npy не совпал с manifest.daty_sha")
    if d.max() >= np.datetime64("2020-01-01"):
        raise ValueError(f"{sid}: даты ≥ 2020")
    return d


def _proverit_zaem(sid: str) -> None:
    """SHA каждого шарда zaem/ по SHA256SUMS_zaem.txt (читатель ledger их не проверяет)."""
    d = LEDGER / sid
    for ln in (d / "SHA256SUMS_zaem.txt").read_text(encoding="utf-8").splitlines():
        h, rel = ln.split()
        if hashlib.sha256((d / rel).read_bytes()).hexdigest() != h:
            raise ValueError(f"{sid}/{rel}: sha256 не совпал")


def zaem_matrica(sid: str, rows: list, T: int) -> np.ndarray:
    """T × N: база платы за заём каждой строки (zaem_v2.zaem_ryad по ссылке excess), float64."""
    _proverit_zaem(sid)
    out, mm = np.empty((T, len(rows))), {}
    for j, r in enumerate(rows):
        e = r["excess"]
        k = (e["potok"], int(e["shard"]))
        if k not in mm:
            mm[k] = np.load(LEDGER / sid / k[0] / SUB / f"{k[1]:05d}.npy", mmap_mode="r")
        out[:, j] = mm[k][int(e["stroka"])]
    if not np.isfinite(out).all():
        raise ValueError(f"{sid}: в базе платы есть nan у строк ok")
    return out


def zagruzit(sid: str) -> dict:
    """Все строки ledger, основные ok-строки, матрицы M (excess после 0,04 %) и Z (база платы), даты."""
    ch = Chitatel(LEDGER, sid)
    vse, ok = ch.rows(None), ch.rows("ok")
    osn = [r for r, m in zip(ok, maska(ok)) if m]
    daty = daty_sid(sid, ch)
    M = np.asarray(ch.matrica(osn), float)
    return dict(ch=ch, vse=vse, ok=ok, rows=osn, daty=daty, M=M, Z=zaem_matrica(sid, osn, len(daty)))


def schetchiki(vse: list, ch: Chitatel) -> dict:
    """Шаг 1: строки по статусам, причины ne_schitana, основные/справочные, уникальные klyuch_sha (= N)."""
    st = Counter(r["status"] for r in vse)
    osn = maska(vse)
    ok = [r for r in vse if r["status"] == "ok"]
    mok = maska(ok)
    pr = Counter(str(r.get("prichina"))[:90] for r in vse if r["status"] == "ne_schitana")
    return dict(strok=len(vse), ok=st["ok"], ne_schitana=st["ne_schitana"], oshibka=st["oshibka"],
                osnovnyh=int(osn.sum()), spravochnyh=int((~osn).sum()), osnovnyh_ok=int(mok.sum()),
                spravochnyh_ok=int((~mok).sum()), N_klyuch_sha=len({r["klyuch_sha"] for r in vse}),
                istochniki=";".join(f"{k}:{v}" for k, v in Counter(r["istochnik"] for r in vse).items()),
                kod=ch.manifest["kod"], kod_slit=ch.manifest["kod_slit"],
                prichiny_ne_schitana=" | ".join(f"{k} ×{v}" for k, v in pr.most_common()), **chisla_N(vse))


def _kv(x: np.ndarray) -> dict:
    return {"p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)), "max": float(x.max())} if len(x) else \
        {"p50": float("nan"), "p90": float("nan"), "max": float("nan")}


def diagnostika_nan(X: np.ndarray, py: np.ndarray, daty: np.ndarray) -> dict:
    """Справочно, не отбор: почему мало устойчивых. ploskij — год без движения (σ = 0, mSR не определён);
    nan_pri_sigma — mSR года не определён при σ > 0 (mVaR ≤ 0); ust_esli_nan_pri_pol_srednem — устойчивость, если год с
    неопределённым mSR считать «>0» при положительном среднем (допущение, в правилах Ж-090 его нет)."""
    g = np.asarray(daty).astype("datetime64[Y]").astype(int) + 1970
    mu = np.column_stack([X[g == y].mean(axis=0) for y in GODY])
    flat = np.column_stack([X[g == y].std(axis=0, ddof=1) == 0 for y in GODY])
    return dict(hotya_by_ploskij_god=int(flat.any(axis=1).sum()), nan_pri_sigma=int((np.isnan(py) & ~flat).any(axis=1).sum()),
                srednee_pol_vse_gody=int((mu > 0).all(axis=1).sum()),
                ust_esli_nan_pri_pol_srednem=int(((py > 0) | (np.isnan(py) & (mu > 0))).all(axis=1).sum()))


def raspredelenie(M: np.ndarray, Z: np.ndarray, daty: np.ndarray) -> list:
    """Шаг 2: целевая основных строк — с платой 5 % (основная) и без неё (справочно)."""
    out = []
    for nazv, st in (("s_platoy_5", STAVKA_SHORT), ("bez_platy", 0.0)):
        X = M - st * Z
        m, py, ust, _ = msr_i_gody(X, daty, GODY)
        ok = ust & np.isfinite(m)
        q, qa = _kv(m[ok]), _kv(m[np.isfinite(m)])
        out.append(dict(variant=nazv, strok=len(m), mSR_nan=int((~np.isfinite(m)).sum()),
                        nan_hotya_by_god=int(np.isnan(py).any(axis=1).sum()), srednee_pol=int((X.mean(axis=0) > 0).sum()),
                        **diagnostika_nan(X, py, daty), ustojchivyh=int(ok.sum()),
                        dolya_ustojchivyh=float(ok.mean()), ust_p50=q["p50"], ust_p90=q["p90"], ust_max=q["max"],
                        vse_konech_p50=qa["p50"], vse_konech_p90=qa["p90"], vse_konech_max=qa["max"]))
    return out


def sverka_tselevoj(M: np.ndarray, Z: np.ndarray, daty: np.ndarray, n: int = 20) -> int:
    """Сверка векторной целевой с tselevaya_train4 на n равномерных столбцах; число расхождений (ожидается 0)."""
    m, _, ust, _ = msr_i_gody(M - STAVKA_SHORT * Z, daty, GODY)
    bad = 0
    for j in np.linspace(0, M.shape[1] - 1, n).astype(int):
        v, _, u = tselevaya_train4(M[:, j], daty, Z[:, j])
        bad += int(not (np.isclose(v, m[j], equal_nan=True) and u == bool(ust[j])))
    return bad
