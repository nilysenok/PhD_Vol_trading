#!/usr/bin/env python3
"""R2, агент O: уровень 2 — последовательное отсечение точек сетки остаточного ученика (решение владельца 26.10, 16:25; R2 на Маке).

Протокол v4 § 5: LHS-60 → ступень 1: складки 2020, 2021 × 1 seed → 30 → ступень 2: 2018–2021 × 1 seed → 15 → ступень 3: все 5 × 3 seed → 8.
- Годы ступеней — yaml `halving.stupeni` (заданы протоколом до счёта); наборы вложены.
- Критерий ступени — средняя t панельного DM против E_h на отложенных годах ступени (r2_gates.t_skladok — код DM теста); при равенстве —
  меньший средний QLIKE, затем меньший номер точки. Вложенного выбора и c3 на ступенях нет (у точки один вариант, ряды только нужных лет).
- 8 выживших идут в ворота кластера 17 (r2_gates.ocenka: все 5 складок, c3, вход, выбор со сложностью).
Встраивание в r2_driver.py агента S: driver даёт функцию ryad(tochka, seeds, gody) -> кадр val (date, ticker, year, step, rv_actual,
rv_pred — итоговый прогноз; при нескольких seed — уже агрегированный), O возвращает выживших и трассу. Driver сам решает, какие шаги
считать (год Y = шаг Y − 2015; калибровке нужны и прошлые шаги).

python3 scripts/analysis/r2_halving.py --selftest"""
import argparse, sys, zlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r2_gates as RG  # noqa: E402
import r2_loyo as O  # noqa: E402

H = O.CFG['halving']


def stupeni():
    """[(номер, годы складок, seeds, оставить)] — протокол v4 § 5: {2020, 2021} → {2018–2021} → все 5; наборы вложены."""
    S = [(i + 1, sorted(s['gody']), list(s['seeds']), int(s['ostavit'])) for i, s in enumerate(H['stupeni'])]
    assert all(set(a[1]) < set(b[1]) for a, b in zip(S, S[1:])) and S[-1][1] == [O.Y(s) for s in O.T], 'ступени вложены, последняя — все 5'
    return S


def otsechenie(tochki, ryad, E, h):
    """tochki — список имён точек (LHS-60 + контроль); ryad(tochka, seeds, gody) -> кадр val. -> (выжившие, трасса DataFrame)."""
    zhivye = list(tochki)
    trace = []
    for k, gody, seeds, ost in stupeni():
        res = {}
        for t in zhivye:
            r = RG.t_skladok(ryad(t, seeds, gody), E, h, gody)
            res[t] = r
            trace.append(dict(stupen=k, tochka=t, gody=','.join(map(str, gody)), seeds=','.join(map(str, seeds)), t_mean=r['t_mean'],
                              q_mean=r['q_mean'], vyigrysh_pct=r['vyigrysh_pct'], **{f't_{y}': r['t'][y] for y in gody}))
        order = sorted(zhivye, key=lambda t: (-res[t]['t_mean'], res[t]['q_mean'], tochki.index(t)))  # v4: равенство — средний QLIKE
        zhivye = order[:ost]
        for row in trace:
            if row['stupen'] == k:
                row['proshla'] = row['tochka'] in zhivye
    return zhivye, pd.DataFrame(trace)


def zapisat(h, kletka, zhivye, trace, inputs=()):
    """Трасса -> R2/O/halving_<клетка>_h{h}.csv; решение -> в gate_D (через r2_loyo.write_gate)."""
    O.OUT.mkdir(parents=True, exist_ok=True)
    f = O.OUT / f'halving_{kletka}_h{h}.csv'
    trace.to_csv(f, index=False, float_format='%.6g')
    return f


def selftest():
    rng = np.random.default_rng(60)
    E, _ = RG._sint(rng, 0.0)
    kach = {f'p{i:02d}': q for i, q in enumerate(np.linspace(-0.05, 0.30, 60))}  # истинное качество точки (доля уменьшения лог-шума)
    kach['p_hot'] = 0.30
    calls = []

    def ryad(t, seeds, gody):
        calls.append((t, tuple(seeds), tuple(gody)))
        g = E[E.year.isin(gody)]
        r = np.random.default_rng(zlib.crc32(f'{t}|{seeds}'.encode()))  # воспроизводимо между процессами
        lnE = np.log(g.rv_pred.values / g.rv_actual.values)  # ошибка E; точка уменьшает её на долю kach[t]
        return g.assign(rv_pred=g.rv_actual.values * np.exp(lnE * (1 - kach[t]) + r.normal(0, 0.02, len(g)) / np.sqrt(len(seeds))))
    tochki = list(kach)
    zh, tr = otsechenie(tochki, ryad, E, 1)
    st = stupeni()
    assert [s[3] for s in st] == [30, 15, 8] and len(zh) == 8
    assert st[0][1] == [2020, 2021] and st[1][1] == [2018, 2019, 2020, 2021] and len(st[2][1]) == 5  # v4 § 5
    n = tr.groupby('stupen').tochka.count()
    assert list(n) == [61, 30, 15] and list(tr.groupby('stupen').proshla.sum()) == [30, 15, 8]
    best = sorted(kach, key=lambda t: -kach[t])[:8]
    assert len(set(zh) & set(best)) >= 6, (zh, best)
    # ступени 1–2 — один seed и только годы ступени; ступень 3 — 3 seed и все 5 лет
    assert {c[1] for c in calls if len(c[2]) < 5} == {(42,)} and all(c[1] == (42, 43, 44) for c in calls if len(c[2]) == 5)
    # трасса — t по годам ступени = r2_gates.t_skladok
    r = RG.t_skladok(ryad(zh[0], [42, 43, 44], st[2][1]), E, 1, st[2][1])
    row = tr[(tr.stupen == 3) & (tr.tochka == zh[0])].iloc[0]
    assert abs(row.t_mean - r['t_mean']) < 1e-12
    print('selftest r2_halving: OK (60 → 30 → 15 → 8; вложенные складки 2 → 4 → 5; seeds 1 → 1 → 3; лучшие точки выживают; трасса)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    ap.print_help()


if __name__ == '__main__':
    main()
