"""Сравнение эталона Мака с пересчётом, два уровня (Ж-131 п. 3; разбор arm64 → x86: docs/otchety/2026-10-29_kross_arh_ekviv.md).

Дискретное — строго, без допуска: хэш кода, хэши входов/куба/дат по Sk, набор записей, `disk` (ключи, статусы, целые, строки),
число сделок по дням, индексы входа/выхода, формы массивов. Любое расхождение — FAIL (счёт не начинать).
Непрерывное — с допуском: вещественные поля строки (по hex), ряд портфеля, ряд тиков, |позиция| портфеля. Допуск: якорь A_fix
TOL_YAKOR = 1e-12 (Ж-131), точки — `dopusk` (переменная DOPUSK, по умолчанию 1e-12, Ж-137 п. 3; расширяет только владелец).
Всегда считается max |Δ| по непрерывным величинам — отдельно якорь и точки, даже когда всё в допуске."""
import numpy as np

TOL_YAKOR = 1e-12
YAKOR = ("yakor", "v6yakor")
SHAPKA = ("kod", "vhody", "kub", "daty", "v6_vhody", "v6_kub", "v6_daty")      # v6_* — набор v6 (ekviv_v6.py); у старого эталона None == None


def _x(s: str) -> float:
    return float.fromhex(s) if s.startswith(("0x", "-0x")) else float(s)


def delta(a: float, b: float) -> float:
    if np.isnan(a) and np.isnan(b) or a == b:
        return 0.0
    return float("inf") if not (np.isfinite(a) and np.isfinite(b)) else abs(a - b)


def delta_ryada(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    return float(np.abs(a[ok] - b[ok]).max()) if ok.any() else 0.0


def _diskretnoe(eta, meta_eta, moi, meta_moi) -> list:
    bad = []
    for k in SHAPKA:
        if meta_eta.get(k) != meta_moi.get(k):
            bad.append(f"{k}: Мак {meta_eta.get(k)}, здесь {meta_moi.get(k)}")
    if set(eta) != set(moi):
        bad.append(f"набор записей: нет здесь {sorted(set(eta) - set(moi))[:5]}, лишние {sorted(set(moi) - set(eta))[:5]}")
    for m in sorted(set(eta) & set(moi)):
        a, b = eta[m], moi[m]
        if a["disk"] != b["disk"]:
            import json
            x, y = json.loads(a["disk"]), json.loads(b["disk"])
            k = next((k for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)), "?")
            bad.append(f"{m}: поле строки ledger {k}: Мак {x.get(k)!r}, здесь {y.get(k)!r}")
        for nazv, k in (("сделки по дням", "sd"), ("индексы входа/выхода", "vi")):
            if a[k].shape != b[k].shape or not np.array_equal(a[k], b[k]):
                i = int(np.flatnonzero(a[k].ravel() != b[k].ravel())[0]) if a[k].shape == b[k].shape else -1
                bad.append(f"{m}: {nazv} различаются (форма {a[k].shape}/{b[k].shape}, первый элемент {i})")
        for k in ("p", "t", "pz"):
            if a[k].shape != b[k].shape:
                bad.append(f"{m}: форма ряда {k} {a[k].shape} против {b[k].shape}")
        if set(a["hex"]) != set(b["hex"]):
            bad.append(f"{m}: набор вещественных полей строки различается")
    return bad


def _nepreryvnoe(eta, moi, dopusk: float) -> tuple:
    """-> (первое превышение (имя, что, Δ, допуск) или None, число превышений, max Δ якоря, max Δ точек)."""
    pervoe, chislo, dm = None, 0, {True: 0.0, False: 0.0}
    for m in sorted(set(eta) & set(moi), key=lambda x: (not x.startswith(YAKOR), x)):
        a, b, yak = eta[m], moi[m], m.startswith(YAKOR)
        tol = TOL_YAKOR if yak else dopusk
        ds = [(f"поле {p}", delta(_x(h), _x(b["hex"][p]))) for p, h in sorted(a["hex"].items()) if p in b["hex"]]
        ds += [(f"ряд {k}", delta_ryada(a[k], b[k])) for k in ("p", "t", "pz") if a[k].shape == b[k].shape]
        for chto, d in ds:
            dm[yak] = max(dm[yak], d)
            if not d <= tol:
                chislo += 1
                pervoe = pervoe or (m, chto, d, tol)
    return pervoe, chislo, dm[True], dm[False]


def sravnit(eta: dict, meta_eta: dict, moi: dict, meta_moi: dict, dopusk: float = 0.0) -> dict:
    disk = _diskretnoe(eta, meta_eta, moi, meta_moi)
    pervoe, chislo, dy, dt = _nepreryvnoe(eta, moi, dopusk)
    return dict(diskretnoe=disk, nepreryvnoe_pervoe=pervoe, nepreryvnoe_chislo=chislo, max_yakor=dy, max_tochki=dt,
                kod=3 if (pervoe and not disk) else (1 if disk else 0))
