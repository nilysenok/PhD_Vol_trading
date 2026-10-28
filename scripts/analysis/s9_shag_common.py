"""Общие куски шага 3′ (D и B) главы 4 (v3): метрики §0.3, С8 v2 с флагами по условиям, к2/к3 с проверками,
устойчивость, разложение, итог B. Чистые функции над движком `s9.Engine` (проверяются на синтетическом).
Свой модуль: если s9_shag_c.py получит общий — сверить при слиянии (отмечено в PR). Данные не читает.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fastbt import fast_kontrol as fk
from fastbt import s9, shag0
from fastbt.shag_c import USLOVIE1, usloviya

ROOT = Path(__file__).resolve().parents[2]
DESIGNS = ROOT / "fastbt" / "designs" / "s9"
OUT = ROOT / "docs" / "tables" / "strat_dev_v3"
COMM_MAIN, COMM_ALT, COMM_APP = 0.0004, 0.00035, (0.0005, 0.0006)
ALPHA, TOL = 0.05, 0.1
COMM_REF = 0.0005                                   # справочный независимый выбор (§0.5)
SIGMAS = {"naive": ("naive_h1", "naive_h5"), "harj": ("harj_h1", "harj_h5"), "hyb": ("hyb_h1", "hyb_h5")}
B_NEGATIVE = "B не фиксируется — отрицательный результат"


def plan(podhod: str, sigma: str | None = None) -> dict:
    """План и число точек — из yaml сетки; σ и RVI не читаются."""
    y = yaml.safe_load((DESIGNS / f"setka_{podhod}_v3.yaml").read_text(encoding="utf-8"))
    ryady = [r for s, rs in SIGMAS.items() if sigma in (None, s) for r in rs]
    return dict(podhod=podhod, ryady=ryady, na_ryad=y["tochek_na_ryad_sigma"],
                vsego=y["tochek_na_ryad_sigma"] * len(ryady), komissii=y["komissii"])


def check_exit_fee(eng, pos, comm: float, tol: float = 1e-12) -> None:
    """Комиссия портфеля = comm × (входы/выходы по изменению позиции + выход на последнем баре пары)."""
    pos = np.asarray(pos, float)
    dp = np.diff(pos, prepend=0.0)
    dp[eng.first] = pos[eng.first]
    last = np.r_[eng.first[1:], True]
    sides = np.abs(dp) + np.where(last, np.abs(pos), 0.0)
    p = eng.port(pos, comm=comm)
    paid = float(((p["gross"] - p["r"]).values * eng.cnt).sum())
    if abs(paid - comm * sides.sum()) > tol * max(1.0, sides.sum()):
        raise AssertionError("комиссия выхода на границе окна не уплачена (Н5)")


def check_shift(K, g=None, sharpe_point: float | None = None, tol: float = 1e-9) -> None:
    """Все проверки сдвига к2 в одном месте: лаги каждого тикера ∈ [21, T−21] (лаг 0 в повторах — ошибка);
    если даны g и Шарп точки — сдвиг 0 воспроизводит точку (A × g). g — маска или множитель масштаба."""
    ml = fk.MIN_LAG
    for ti, T in enumerate(np.asarray(K.t_tick)):
        if T >= 2 * ml and (K.LAG[:, ti].min() < ml or K.LAG[:, ti].max() > T - ml):
            raise AssertionError(f"к2: лаг вне [{ml}, T−{ml}] у тикера {ti}")
    if g is not None and abs(K.identity_sharpe(g) - sharpe_point) > tol:
        raise AssertionError("к2: сдвиг 0 не даёт исходную маску")


def check_k3(eng, k3_pos: np.ndarray, sids=None) -> None:
    """к3: S5/S6 (из designs/s9/k3.yaml) обнулены на всех днях, прочие семьи — как в A."""
    sidr, A = shag0.sid_row(eng), np.asarray(eng.A_row, float)
    drop = np.isin(sidr, list(sids or fk.k3_sids()))
    if np.any(k3_pos[drop] != 0) or not np.array_equal(k3_pos[~drop], A[~drop]):
        raise AssertionError("к3: S5/S6 не выключены или тронуты прочие семьи")


def uslovie1_yaml(podhod: str) -> str:
    """Условие 1 С8 из сетки (Ъ 27.10, 13:05: strogo); нет ключа — умолчание `fastbt.shag_c.USLOVIE1`."""
    y = yaml.safe_load((DESIGNS / f"setka_{podhod}_v3.yaml").read_text(encoding="utf-8"))
    return y.get("uslovie1", USLOVIE1)


def make_base(eng, comm: float, reps: int, uslovie1: str = USLOVIE1) -> dict:
    A = eng.port(eng.A_row, comm=comm)
    k3 = fk.k3_positions(np.asarray(eng.A_row, float), eng.r["pair"], eng.pairs)
    check_k3(eng, k3)
    p3 = eng.port(k3, comm=comm)
    return dict(A=A, k3=p3, mA=s9.metrics(A), m3=s9.metrics(p3), comm=comm, reps=reps, uslovie1=uslovie1)


def eval_point(eng, K, base: dict, pos, g) -> dict:
    """Строка точки: §0.3, Δ к A (ЛВ, знак по годам), Δ к к3 (ЛВ), к2 и флаги условий 1–3 отдельно.
    Условие 3 — для всех (D-маски, D7, B; Ъ 27.10): к2 — круговой сдвиг ряда экспозиции g (маски или
    множителя масштаба) по тикерам на лаг ∈ [21, T−21]; D7 — масштаб min(1,(med/σ)^γ), не бинарный затвор."""
    p = eng.port(pos, comm=base["comm"])
    m = s9.metrics(p)
    row = shag0.stats(p, np.ones(len(p), bool))
    row.update({f"sharpe_{y}": m[str(y)] for y in s9.YEARS})
    row.update(shag0.delta(p, base["A"], base["reps"]))
    d3, p3 = shag0.delta_p(p, base["k3"], base["reps"])
    row.update(sharpe_k3=base["m3"]["sharpe"], d_vs_k3=d3, p_vs_k3=p3)
    check_shift(K, g, m["sharpe"])                      # обязательно для каждой точки
    sh = K.k2_sharpes(g)
    pk2 = fk.p_perm(row["sharpe"], sh)
    row["sharpe_k2_med"], row["n_shift_k2"] = float(np.median(sh)), len(sh)
    row["p_vs_k2"] = pk2
    u = usloviya(m, base["mA"], base["m3"], pk2, base.get("uslovie1", USLOVIE1))
    row.update(usl1=u["usl1_c8"], usl2=u["usl2_k3"], usl3=u["usl3_k2"], passed_v2=u["proshla_c8v2"],
               u1_strogo=u["u1_strogo"], u1_odno_isklyuchenie=u["u1_odno_isklyuchenie"])  # u1_odno_* — справочно
    return row


def select_ref(d_ref: pd.DataFrame, sel):
    """Справочный выбор при 0,05 % — независимый отбор (С8 v2 при той комиссии). Строка: точка, Шарп, флаг
    sovpal (совпал с основным выбором). Другой выбор — обе точки идут на тест (флаг obe_na_test)."""
    r = select(d_ref)
    if r is None:
        return dict(vybor_005="нет прошедших", sovpal=sel is None, obe_na_test=False)
    same = sel is not None and (r["family"], r["point"], r["sigma"]) == (sel["family"], sel["point"], sel["sigma"])
    return dict(vybor_005=f'{r["family"]}:{r["point"]}:{r["sigma"]}', sharpe_005=r["sharpe"],
                sovpal=bool(same), obe_na_test=not same)


def select(df: pd.DataFrame):
    """Среди прошедших С8 v2 — максимум Шарпа при основной комиссии; None, если никто."""
    ok = df[df["passed_v2"].astype(bool)]
    return None if ok.empty else ok.sort_values("sharpe", ascending=False).iloc[0]


def axes(label: str) -> dict:
    return {k: float(v) for k, v in re.findall(r"([A-Za-z]+?)(\d+(?:\.\d+)?)", label)}


def ustoychivost(df: pd.DataFrame, sel) -> pd.DataFrame:
    """Соседи ±1 по каждому параметру семьи (остальные те же; семья, σ-ряд те же), число точек в пределах
    0,1 Шарпа, вывод: край (выбранная на границе оси) / плато (все ближайшие соседи в 0,1 и прошли) /
    пик (ни один) / частичное плато. Правило — как ustoychivost_c8v2 (двусторонние соседи)."""
    d = df[(df["family"] == sel["family"]) & (df["sigma"] == sel["sigma"])]
    ax0 = axes(sel["point"])
    allax = {k: sorted({axes(x).get(k) for x in d["point"] if set(axes(x)) == set(ax0)}) for k in ax0}
    rows, edge = [], False
    for _, r in d.iterrows():
        a = axes(r["point"])
        if set(a) != set(ax0):
            continue
        diff = [k for k in ax0 if a[k] != ax0[k]]
        if len(diff) == 1 and abs(allax[diff[0]].index(a[diff[0]]) - allax[diff[0]].index(ax0[diff[0]])) == 1:
            rows.append(dict(rol=f"сосед по {diff[0]}", point=r["point"], sharpe=r["sharpe"],
                             d_k_vybrannoy=r["sharpe"] - sel["sharpe"], passed_v2=bool(r["passed_v2"])))
    edge = any(len(v) > 1 and ax0[k] in (v[0], v[-1]) for k, v in allax.items())
    n = len(rows)
    near = sum(abs(r["d_k_vybrannoy"]) <= TOL for r in rows)
    full = sum(r["passed_v2"] and abs(r["d_k_vybrannoy"]) <= TOL for r in rows)
    verdict = ("край" if edge else "плато" if n and full == n else "пик" if full == 0 else "частичное плато")
    n_tol = int((np.abs(d["sharpe"] - sel["sharpe"]) <= TOL).sum()) - 1
    head = dict(rol="выбранная", point=sel["point"], sharpe=sel["sharpe"], d_k_vybrannoy=0.0,
                passed_v2=True, sosedei=n, sosedei_v_0_1=near, tochek_v_0_1_ryada=n_tol, vyvod=verdict)
    return pd.DataFrame([head] + rows)


def closed_by_sid(eng, g: np.ndarray) -> list:
    """По семьям S1–S6: доля дней с позицией A, где g < 1, и доля объёма |A|, снятая затвором."""
    sidr, A = shag0.sid_row(eng), np.abs(np.asarray(eng.A_row, float))
    rows = []
    for s in shag0.SIDS:
        m = (sidr == s) & (A > 0)
        rows.append(dict(semya=s, dnei_s_poziciey=int(m.sum()),
                         zakryto_dnei_pct=float(np.mean(g[m] < 1) * 100) if m.any() else np.nan,
                         snyato_obyema_pct=float((1 - (A[m] * g[m]).sum() / A[m].sum()) * 100) if m.any() else np.nan))
    return rows


def razlozhenie(base: dict, row_p: dict, p_pt: pd.DataFrame, reps: int) -> list:
    """Δ к A = (к3 − A) + (точка − к3); обе части с p (ЛВ, блок 5)."""
    d1, p1 = shag0.delta_p(base["k3"], base["A"], reps)
    d2, p2 = shag0.delta_p(p_pt, base["k3"], reps)
    return [dict(chast="к3 − A", d_sharpe=d1, p=p1), dict(chast="точка − к3", d_sharpe=d2, p=p2),
            dict(chast="точка − A", d_sharpe=row_p["d_vs_A"], p=row_p["p_vs_A"])]


def b_itog(passed: dict) -> pd.DataFrame:
    """Правило §3′: B не проходит ни на одной σ (наивная → HAR-J → гибрид) — строка отрицательного результата.
    passed: {σ: число точек, прошедших С8 v2 (условия 1–3)} или None, если не считалась."""
    rows = [dict(sigma=s, proshlo_tochek=("не считалась" if passed.get(s) is None else passed[s]))
            for s in SIGMAS]
    done = [v for v in passed.values() if v is not None]
    if len(done) == len(SIGMAS):
        rows.append(dict(sigma="итог", proshlo_tochek=sum(done),
                         vyvod=B_NEGATIVE if sum(done) == 0 else "B проходит; точка — в otbor"))
    return pd.DataFrame(rows)
