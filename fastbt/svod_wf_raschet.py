"""Свод wf, расчёт (предрегистрация v2 §4–§7): walk-forward, контроли, DSR, PBO по суммам кварталов ledger 2.1/2.2 (рядов нет). Данных не читает.
Окно калибровки года t — `kalibr_v5_21.god` (сегменты до Q4 года t−1, год t в окно не входит). Шарп года t ансамбля — по усреднённым по парам
(тикер, член) суммам S и Q: Q̄ ≥ Q ряда-среднего (Йенсен), sd завышен, Шарп — НИЖНЯЯ оценка (ковариаций из сумм нет, как в `pbo_wf`).
↺ класс B: к2 — циклический сдвиг выбора по годам (n−1 сдвигов, p ≥ 1/n), не 199 сдвигов по дням (дневных рядов в 2.1/2.2 нет). Откат: ledger 2.0."""
from __future__ import annotations

import math
from statistics import median

import numpy as np

from fastbt import kalibr_v5 as K, kalibr_v5_21 as K21, kontroli_v5 as kv, ledger_v2 as lm, metriki_e as ME, pbo_wf as PW, stats
from fastbt import sweep_drivers as SD, sweep_plato as SP
from fastbt.kalibr_v5_21 import Pul21

GOD0 = 2015
GODY = K.PERIODY["wf"][1]                      # 2017…2026 (2026 — янв.–фев.)
POLNYE = tuple(range(2017, 2026))               # 9 полных лет (У-14)
N_PREFIKS = 4096


def pul_wf(ch, osi: dict) -> Pul21:
    """Пул одного прохода: строки без роли kontrol; комиссия единая; нужна |позиция| на конце кварталов."""
    if not ch.manifest.get("poz_konec"):
        raise ValueError(f"{ch.d.name}: нужен ledger с poz_konec")
    rows = [r for r in ch.rows() if r.get("rol") != lm.KONTROL]
    kom = {float(r["klyuch"]["komissiya"]) for r in rows}
    if len(kom) != 1:
        raise ValueError(f"{ch.d.name}: комиссия строк не единая: {sorted(kom)}")
    tochki = [{a: v for a, v in r["klyuch"].items() if a not in K.SLUZH} for r in rows]
    pul = Pul21(ch.d.name, {a: [lm.tok(v) for v in vs] for a, vs in osi.items()}, tochki, ch.summy(rows, "tik"),
                np.asarray(ch.manifest["nbars"], float), GOD0, kom.pop())
    pul.sobol_i = [r.get("sobol_i") for r in rows]
    return pul


def sharpe_s(S: float, Q: float, n: float) -> float:
    return float(PW.sharpe_iz_summ(np.array([S]), np.array([Q]), n)[0]) if n >= 2 else float("nan")


def summy_goda(pul: Pul21, t: int, pary: list):
    """(S̄, Q̄, баров) года t по парам (точка, тикер); пар нет — None."""
    a, b = 4 * (t - pul.god0), min(4 * (t - pul.god0) + 4, len(pul.nbars))
    if not pary:
        return None
    blok = np.array([pul.sm[i, k, a:b] for i, k in pary])
    return float(blok[..., stats.F_S].sum(-1).mean()), float(blok[..., stats.F_Q].sum(-1).mean()), float(pul.nbars[a:b].sum())


def pary_iz_vybora(sel: dict) -> list:
    return [(i, k) for k, idx in sel.items() for i in idx]


def itog(sums: dict) -> dict:
    """sums {t: (S, Q, n) | None} → Шарп по годам, итоговый, медиана 2017–2025, §7: результат есть при обоих > 0."""
    ok = {t: v for t, v in sums.items() if v}
    god = {t: sharpe_s(*v) for t, v in ok.items()}
    sh = sharpe_s(sum(v[0] for v in ok.values()), sum(v[1] for v in ok.values()), sum(v[2] for v in ok.values())) if ok else float("nan")
    pol = [god[t] for t in POLNYE if t in god]
    med = median(pol) if len(pol) == len(POLNYE) else float("nan")
    return dict(god=god, sharpe=sh, mediana=med, dni=sum(v[2] for v in ok.values()), rezultat=bool(sh > 0 and med > 0))


def wf(pul: Pul21) -> dict:
    """10 окон: {t: {sel: {тикер: [индексы членов]}, n_proby, okno}} и итог по OOS-годам."""
    sel, sums = {}, {}
    for t in GODY:
        r = K21.god(pul, t)
        sel[t] = dict(sel={k: x["i"] for k, x in r["tikery"].items() if x}, n_proby=r["n_proby"], okno=r["okno"])
        sums[t] = summy_goda(pul, t, pary_iz_vybora(sel[t]["sel"]))
    return dict(vybor=sel, **itog(sums))


def k1(pul: Pul21, n: int = N_PREFIKS) -> dict:
    """Безусловное среднее: равный вес точек префикса Соболя (sobol_i < n) × все тикеры, каждый год. Пусто — ошибка, не 0."""
    idx = [i for i, s in enumerate(pul.sobol_i) if s is not None and s < n]
    if not idx:
        raise ValueError(f"в пуле нет точек Соболя с индексом < {n}")
    return itog({t: summy_goda(pul, t, [(i, k) for i in idx for k in range(pul.sm.shape[1])]) for t in GODY})


def k2(pul: Pul21, vybor: dict) -> dict:
    """Циклический сдвиг выбора по годам: год t торгует выбором года t+s (s = 1…n−1); сдвиг 0 — исходный ряд. p = (1 + #≥)/(n_сдвигов + 1)."""
    gody = list(vybor)

    def sh(s):
        return itog({t: summy_goda(pul, t, pary_iz_vybora(vybor[gody[(j + s) % len(gody)]]["sel"])) for j, t in enumerate(gody)})["sharpe"]
    obs, perm = sh(0), [sh(s) for s in range(1, len(gody))]
    return dict(sharpe=obs, perm=perm, p=(1 + sum(1 for x in perm if x >= obs)) / (len(perm) + 1))


def odna_tochka(pul: Pul21) -> dict:
    """Одна точка на стратегию для всех тикеров: argmax плато (Шарп окна ≤ t−1, среднее по тикерам), заново каждый год."""
    sums, tochki = {}, {}
    for t in GODY:
        pl = kv.plato(pul, pul.okno(t)[0].mean(axis=1))
        if np.isnan(pl).all():
            sums[t] = None
            continue
        tochki[t] = i = int(np.nanargmax(pl))
        sums[t] = summy_goda(pul, t, [(i, k) for k in range(pul.sm.shape[1])])
    return dict(itog(sums), tochki=tochki)


def seed2(w1: dict, w2: dict) -> dict:
    return {t: (w1["god"].get(t, float("nan")), w2["god"].get(t, float("nan"))) for t in GODY}


def dsr_wf(ob, nbars: np.ndarray, sr_oos: float, dni: float) -> dict:
    """DSR(N = ob.n_k()) по общей матрице: SR* из дисперсии дневных Шарпов всех точек (полный период, из сумм), PSR с нормальными skew/kurt
    (рядов нет — ↺: асимметрия 0, эксцесс 3). dsr_n1 — тот же SR с N = 1 (справочно: обязан быть ≥ dsr, мутация № 14)."""
    S = ob.summy("port")
    sr_d = PW.sharpe_iz_summ(S[..., stats.F_S].sum(1), S[..., stats.F_Q].sum(1), float(nbars.sum())) / math.sqrt(252.0)
    n, v = ob.n_k(), float(np.var(sr_d, ddof=1)) if len(sr_d) > 1 else 0.0
    zv = ME.sr_zvezda
    return dict(N=n, V=v, sr_zvezda=zv(n, v), dsr=SD.psr(sr_oos / math.sqrt(252.0), zv(n, v), int(dni)),
                dsr_n1=SD.psr(sr_oos / math.sqrt(252.0), zv(1, v), int(dni)))


def pbo_ob(ob, chitateli: list, osi: dict, sid: str) -> dict:
    """PBO по кварталам (`pbo_wf`, правило процедуры): суммы 45 кварталов общей матрицы; соседи — по осям с эффектом из ledger."""
    nb = np.asarray(chitateli[0].manifest["nbars"], float)
    if nb.size < PW.NQ or nb[PW.NQ:].any():
        raise ValueError(f"nbars: ожидалось {PW.NQ} кварталов и пусто далее")
    S = ob.summy("port")
    tochki = [{a: v for a, v in r["klyuch"].items() if a not in K.SLUZH} for _, _, r in ob._poz]      # noqa: SLF001 — точки в порядке матрицы
    pul = Pul21(sid, {a: [lm.tok(v) for v in vs] for a, vs in osi.items()}, tochki, S[:, None], nb, GOD0, 0.0)
    ef, exp = K.osi_s_effektom(pul, pul.okno(2026)[0][:, 0]), {"grid": pul.osi}
    fix = lambda q: lm.canon_v2(sid, q)        # noqa: E731
    sosedi = [[pul.idx[K.klyuch(sid, q)] for q in SP.neighbours(exp, fix(pt), ef, fix) if K.klyuch(sid, q) in pul.idx] for pt in tochki]
    r = PW.pbo_wf(S[:, :PW.NQ, stats.F_S], S[:, :PW.NQ, stats.F_Q], nb[:PW.NQ], PW.pravilo_procedury(sosedi))
    return dict(pbo=r["pbo"], n=r["n"], n_propusk=r["n_propusk"])
