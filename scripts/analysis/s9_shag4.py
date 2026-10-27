"""Шаг 4 разработки главы 4 (v3): сравнение прогнозов (H2), 2018–2021 → docs/tables/strat_dev_v3/shag4_*.csv.

    python3 scripts/analysis/s9_shag4.py --tochki tochki.yaml [--reps 10000] [--dry-run]
tochki.yaml — три точки выбора шагов 1–3 (ключ s9: [семья:метка, источник σ]), по одной на σ:
    C_naive: ["C1:W252_p70", "naive_h1"]
    C_harj:  ["C1:W252_p70", "harj_h1"]
    C_hyb:   ["C1:W504_p70", "hyb_h5"]
Комиссии 0,035 и 0,04 % — обе основные колонки (Ъ 12:05). Портфель точки — fastbt.shag4.portfel (заглушка до
слияния s9_shag_c.py). σ — только data/regen/ch4_dev (s9.use_v2 сверяет SHA); цель QLIKE — rv_target из
features_pit_v3 с фильтром date ≤ 2021-12-31 при чтении; цель с окном t+h за 2021-12-31 — NaN (g3.cut_target).
--dry-run: разбор ключей и план, данные не читаются. Реальный запуск — только с --razreshayu-shag4 и при наличии
fastbt/shag_c.py (PR #270), иначе отказ; затвор — fastbt.shag_c.
Самопроверка без аргументов.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastbt import shag0, shag4, shag4_gl3 as g3  # noqa: E402
from fastbt.fast_kontrol import k3_sids  # noqa: E402
from s9_shag0 import write  # noqa: E402

TABLES = ("shag4_luchshiy.csv", "shag4_odna_tochka.csv", "shag4_lestnitsa.csv", "shag4_svyazka_gl3.csv")
LAST = pd.Timestamp("2021-12-31")


def load_tochki(path) -> dict:
    """{имя: (ключ, источник)}; источник σ обязан быть из своей группы (C_naive → naive_h*)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = {}
    for nm, sg in zip(shag4.NAMES, shag4.SIGMAS):
        ckey, src = raw[nm]
        shag4.parse_label(ckey.split(":", 1)[1])
        if not src.startswith(sg + "_h") or src[-2:] not in ("h1", "h5"):
            raise ValueError(f"{nm}: источник σ {src!r} не из группы {sg}")
        out[nm] = (ckey, src)
    return out


def years_matrix(eng) -> np.ndarray:
    Y = np.zeros((len(eng.tickers), eng.nmax), int)
    for ti, t in enumerate(eng.tickers):
        Y[ti, :eng.n_t[ti]] = pd.DatetimeIndex(eng.cache[t]["dates"]).year
    return Y


def target_matrix(eng, h: str) -> np.ndarray:
    """Цель QLIKE (дисперсия) на баре d: rv_target_h1 / rv_target_mean_h5."""
    return feature_matrix(eng, "rv_target_h1" if h == "h1" else "rv_target_mean_h5", 1 if h == "h1" else 5)


def feature_matrix(eng, col: str, h: int = 0, path=None) -> np.ndarray:
    """Столбец features_pit_v3 → [тикер, бар]; чтение с фильтром date ≤ 2021-12-31. Цель (h > 0, окно t+1…t+h)
    сразу при чтении обрезается: окно за 2021-12-31 → NaN (g3.cut_target)."""
    from fastbt import s9
    d = pd.read_parquet(path or s9.REGEN / "features_pit_v3.parquet", columns=["date", "ticker", col],
                        filters=[("date", "<=", LAST)])
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= LAST].rename(columns={col: "sigma"}).sort_values(["ticker", "date"])
    if h:
        d = g3.cut_target(d, h, LAST)
    assert d["date"].max() <= LAST
    Y = np.full((len(eng.tickers), eng.nmax), np.nan)
    for ti, t in enumerate(eng.tickers):
        Y[ti, :eng.n_t[ti]] = s9.align(eng.cache[t]["dates"], d[d["ticker"] == t])
    return Y


def svyazka(eng, T: dict, razr: bool = False) -> list:
    """4.D: корреляция σ по тикерам, доля расхождений решений фильтра, QLIKE открыт/закрыт (точка гибрида)."""
    valid = g3.dev_mask(years_matrix(eng))
    S = {nm: eng.sigma(src) for nm, (_, src) in T.items()}
    rows = g3.corr_sigma(S, valid, eng.tickers)
    for pn, (pk, psrc) in T.items():                    # W, p — точки pn; h — её же
        h = psrc[-2:]
        G = {}
        for qn, sg in zip(shag4.NAMES, shag4.SIGMAS):
            try:
                G[qn] = shag4.gate(eng, pk, shag4.src_of(sg, h), razr)
            except ValueError:
                G[qn] = None
        names = list(G)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                v = np.nan if G[a] is None or G[b] is None else g3.share_diff(G[a], G[b], valid)
                rows.append(dict(pometka=shag4.POMETKA, rezhim="решения фильтра", para=f"{a} – {b}",
                                 tochka=f"{pn}: {pk} ({h})", dolya_raznykh=v))
    ck, src = T["C_hyb"]
    h = src[-2:]
    y = target_matrix(eng, h)
    op = g3.open_mask(shag4.gate(eng, ck, src, razr))
    f = lambda sg: eng.sigma(shag4.src_of(sg, h)) ** 2                    # noqa: E731
    for r in g3.qlike_split(y, f("harj"), f("hyb"), op, valid):
        rows.append({**r, "tochka": f"C_hyb: {ck} ({h})"})
    return rows


def sdvig(eng, T: dict) -> None:
    """Проверка сдвига входов (первая строка журнала шага): профиль k = −2…+1, максимум на k = 0, иначе стоп."""
    valid, rv = g3.dev_mask(years_matrix(eng)), feature_matrix(eng, "rv")
    for nm, (_, src) in T.items():
        prof = g3.shift_profile(eng.sigma(src), rv, valid, diff=src.startswith("naive"))
        print(f"сдвиг {src}:", {k: round(v, 3) for k, v in prof.items()})
        g3.check_shift(prof)


def run(T: dict, reps: int, razr: bool = False) -> None:
    from fastbt import s9
    shag4.shag_c(razr)                                  # отказ до чтения данных
    s9.use_v2(shag4.COMMS[1])
    s9.load_c_dop()
    eng = s9.Engine(s9.load_cache())
    A = np.asarray(eng.A_row, float)
    k3 = shag0.zero_sids(A, shag0.sid_row(eng), k3_sids())
    P_l, P_o, P_s, A_by = {}, {}, {}, {}
    for comm in shag4.COMMS:
        pts = {nm: shag4.portfel(eng, ck, src, comm, razr) for nm, (ck, src) in T.items()}
        A_by[comm] = eng.port(A, comm=comm)
        P_l[comm] = pts
        P_s[comm] = {"A": A_by[comm], "к3": eng.port(k3, comm=comm), **pts}
        P_o[comm] = {}
        for nm, (ck, src) in T.items():
            for sg in shag4.SIGMAS:
                try:
                    P_o[comm][(nm, sg)] = shag4.portfel(eng, ck, shag4.src_of(sg, src[-2:]), comm, razr)
                except ValueError:
                    P_o[comm][(nm, sg)] = None
    shag4.check_odna_tochka(P_o, P_l)
    sdvig(eng, T)
    lest = shag4.lestnitsa(P_s, reps)
    shag4.check_lestnitsa(lest)
    write(shag4.luchshiy(P_l, reps), TABLES[0])
    write(shag4.odna_tochka(P_o, A_by, reps), TABLES[1])
    write(lest, TABLES[2])
    write(svyazka(eng, T, razr), TABLES[3])
    print("записано:", ", ".join(TABLES))


def dry_run(T: dict, reps: int) -> None:
    for nm, (ck, src) in T.items():
        W, par = shag4.parse_label(ck.split(":", 1)[1])
        print(f"{nm}: {ck} на {src}: W={W}, p={par}; в 4.B — σ {shag4.SIGMAS} с h={src[-2:]}")
    print(f"комиссии {[round(c * 100, 4) for c in shag4.COMMS]} %, ЛВ: блок 5, {reps} повторов; таблицы → docs/tables/strat_dev_v3/:")
    print("  " + "\n  ".join(TABLES))
    print("dry-run: данные не читались")


def selftest() -> None:
    assert shag4.parse_label("W252_p70") == (252, 0.70) and shag4.parse_label("W504_p70_90") == (504, (0.7, 0.9))
    rows = [dict(comm_pct=4.0, d_k_A=0.0), dict(comm_pct=4.0, d_sharpe=0.2, d_k_A=0.2),
            dict(comm_pct=4.0, d_sharpe=0.1, d_k_A=0.3)]
    shag4.check_lestnitsa(rows)
    rows[2]["d_k_A"] = 0.5
    try:
        shag4.check_lestnitsa(rows)
        raise SystemExit("check_lestnitsa не поймала разрыв")
    except AssertionError:
        pass
    print("OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tochki")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--razreshayu-shag4", action="store_true", help="реальный запуск; нужен fastbt/shag_c.py (PR #270)")
    a = ap.parse_args()
    if not a.tochki:
        selftest()
    else:
        t = load_tochki(a.tochki)
        dry_run(t, a.reps) if a.dry_run else run(t, a.reps, a.razreshayu_shag4)
