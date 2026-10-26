#!/usr/bin/env python3
"""R2, агент O: ворота по кластеру 17 и расчёт мощности (кластер 20) — решение владельца 26.10, 16:25 (R2 на Маке; протокол v4 пишет
архитектор). Панельный DM — тем же кодом, что в тесте: ftp_stats.qloss, daily (дневной ряд средней по тикерам), dm_series (NW, лаг h−1).

Для дизайна u (варианты — значения-списки, выбор внутри складки по c3, как r2_loyo п. 2.1) и эталона В2 (HAR-J + s² sm, d0 = E_h):
- складка Y: d_t = средняя по тикерам разность потерь QLIKE (E − u) за день t отложенного года Y; t_Y = dm_series(d_t, h).t;
- вход: t_Y ≥ 1,0 не меньше чем в 3 складках из 5 и средняя t ≥ 1,5; hit rate (доля дней d_t > 0 по 5 отложенным годам) ≥ 55%;
  выигрыш (средняя разность потерь тикера > 0) не меньше чем на 12 тикерах из 17; концентрация — на 2% дней с наибольшим d_t
  приходится не больше 40% суммарного выигрыша Σ d_t; c3 на всех складках (r2_loyo.crit, против E_h);
- выбор среди прошедших: максимум средней t; при Δt ≤ 0,2 — больший средний выигрыш (в % к средней потере E); при Δt ≤ 0,2 и разнице
  выигрыша ≤ 1 п. п. — более простой дизайн (сложность по schema_v2, r2_loyo.slozhnost = r2_core.complexity);
- мощность (кластер 20): по пулу 5 отложенных лет — среднее d̄ и долгосрочная дисперсия NW на день; на N_test днях 2022–2026
  ожидаемая t = d̄ / √(σ²_LR / N_test), P(p < 0,05) = Φ(t − z_0,95) (одностороннее); P < 70% — пометка «не хуже».
Читаются только val 2017–2021 (ряды подаются кадрами; r2_loyo.read_val / r2_core.load_series). N_test — календарь будних дней, тест не читается.

python3 scripts/analysis/r2_gates.py --selftest
python3 scripts/analysis/r2_gates.py dym --h 1 5 22      # дым: F_h(d0) seed 42 против E_h на реальных val-рядах (3 тикера или все)"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ftp_stats as FS  # noqa: E402  панельный DM теста — без изменений
import r2_loyo as O  # noqa: E402  вложенный выбор по c3, критерии, сложность, чтение val

G = O.CFG['gates17']
KEY = ['date', 'ticker']


def _q_steps_bez(df):
    return O.q_steps(df, 'bez')


def etalon_df(h):
    """E_h = эталон (б) ВЫР-1: HAR-J d0 + s² sm (столбец har_corr; = har_raw с OOS sm, сверено в r2_loyo)."""
    e = O.read_val(O.R / O.CFG['etalon']['file'].format(h=h))
    return e[['date', 'ticker', 'year', 'step', 'rv_actual']].assign(rv_pred=e.har_corr.values)


def tablica_c3(variants, E, ref_q):
    """Таблица шагов для c3 (r2_loyo.crit): q варианта по шагам T, q_har = q_har_c1 = q E_h, q_xgb/lgb_single — ML-эталоны (ref_q)."""
    qE = _q_steps_bez(E)
    rows = []
    for v, df in variants.items():
        q = _q_steps_bez(df)
        for st in O.T:
            rows.append(dict(unit='u', variant=v, step=st, q_hyb=q[st], q_har=qE[st], q_har_c1=qE[st],
                             q_xgb_single=ref_q['xgb_v0'][st] if ref_q else qE[st], q_lgb_single=ref_q['lgb_v0'][st] if ref_q else qE[st],
                             scheme='cell', v1_ml='', w_har=0.0, w_xgb=0.0, w_lgb=0.0))
    return pd.DataFrame(rows)


def d_panel(u, E):
    """Строки (дата, тикер) обоих рядов: d = QLIKE(E) − QLIKE(u), фильтр rv_actual > 1e-12 (как final_test)."""
    m = E[KEY + ['year', 'rv_actual', 'rv_pred']].rename(columns={'rv_pred': 'pE'}).merge(u[KEY + ['rv_pred']], on=KEY, how='inner')
    m = m[m.rv_actual > 1e-12]
    lE, lu = FS.qloss(m.rv_actual.values, m.pE.values), FS.qloss(m.rv_actual.values, m.rv_pred.values)
    return m.assign(d=lE - lu, lE=lE)


def t_skladok(u, E, h, folds):
    """Быстрая оценка для отсечения (r2_halving): t панельного DM на отложенных годах folds, без вложенного выбора и c3
    (у точки сетки один вариант; ряды — только нужных шагов). -> {год: t}, средняя t, средний выигрыш %."""
    P = d_panel(u, E)
    out = {}
    for y in folds:
        g = P[P.year == y]
        out[y] = FS.dm_series(FS.daily(g.date.values, g.d.values), h)['t']
    g = P[P.year.isin(folds)]
    lu = FS.qloss(g.rv_actual.values, g.rv_pred.values)
    return dict(t=out, t_mean=float(np.mean(list(out.values()))),
                q_mean=float(np.mean([lu[(g.year == y).values].mean() for y in folds])),  # средний QLIKE лет ступени (равенство t)
                vyigrysh_pct=float(100 * FS.daily(g.date.values, g.d.values).mean() / FS.daily(g.date.values, g.lE.values).mean()))


def koncentraciya(d_t, lE_t):
    """v4 § 2 п. 4 (❓ 2): 2% дней с наибольшей потерей эталона (средней по тикерам) в объединении отложенных лет — их доля в Σ d_t.
    Σ d_t ≤ 0 — inf (вход не пройден)."""
    total = float(d_t.sum())
    if total <= 0:
        return np.inf
    k = max(1, int(np.ceil(G['konc_dolya_dney'] * len(d_t))))
    hot = lE_t.sort_values(ascending=False).index[:k]
    return float(d_t.loc[hot].sum() / total)


def ocenka(variants, E, h, ref_q=None):
    """variants — {имя варианта: кадр val (date, ticker, year, step, rv_actual, rv_pred — итоговый прогноз дизайна), шаги 1–6}.
    Вложенный выбор (v4 § 2): в складке Y среди вариантов с c3 на T\\{Y} — максимум средней панельной t на обучающих годах, при равенстве —
    меньший средний QLIKE; если c3 нет ни у одного — складка без c3 (вариант — тем же правилом среди всех).
    -> dict: по складкам (вариант, c3, t, n дней, выигрыш), t̄, hit rate, тикеры, концентрация, мощность, вход."""
    folds = [O.Y(s) for s in O.T]
    tab = tablica_c3(variants, E, ref_q)
    Gv = {v: g.sort_values('step') for v, g in tab.groupby('variant', sort=False)}
    vs = list(Gv)
    P = {v: d_panel(df, E) for v, df in variants.items()}
    ty = {v: {y: FS.dm_series(FS.daily(P[v][P[v].year == y].date.values, P[v][P[v].year == y].d.values), h)['t'] for y in folds} for v in vs}
    qy = {v: dict(zip(Gv[v].step, Gv[v].q_hyb)) for v in vs}
    po, dd, ee, rows_all = [], [], [], []
    for y in folds:
        st = y - 2015
        tr = tuple(s for s in O.T if s != st)
        key = lambda v: (-np.mean([ty[v][O.Y(s)] for s in tr]), np.mean([qy[v][s] for s in tr]), vs.index(v))
        ok3 = [v for v in vs if O.crit(Gv[v], tr, c1_E=True, c3_E=True)['c3']]
        c3 = bool(ok3)
        vv = sorted(ok3 or vs, key=key)[0]
        g = P[vv][P[vv].year == y]
        dt = FS.daily(g.date.values, g.d.values)
        r = FS.dm_series(dt, h)
        days = np.sort(g.date.unique())
        po.append(dict(Y=y, variant=vv, c3=c3, t=r['t'], t_obuch=-key(vv)[0], n_dney=r['n'], mean_d=r['mean_d'],
                       vyigrysh_pct=100 * r['mean_d'] / FS.daily(g.date.values, g.lE.values).mean()))
        dd.append(pd.Series(dt, index=days))
        ee.append(pd.Series(FS.daily(g.date.values, g.lE.values), index=days))
        rows_all.append(g)
    F = pd.DataFrame(po)
    d_all, lE_all = pd.concat(dd).sort_index(), pd.concat(ee).sort_index()
    A = pd.concat(rows_all)
    tick = A.groupby('ticker').d.mean()
    res = dict(folds=F, t_mean=float(F.t.mean()), n_t_ge=int((F.t >= G['t_skladka']).sum()), hit=float((d_all > 0).mean()),
               n_tikerov_vyigrysh=int((tick > 0).sum()), n_tikerov=int(len(tick)), konc=koncentraciya(d_all, lE_all),
               c3_vse=bool(F.c3.all()), mean_d=float(d_all.mean()), mean_lE=float(lE_all.mean()),
               vyigrysh_pct=float(100 * d_all.mean() / lE_all.mean()), moshchnost=moshchnost(d_all.values, h))
    res['vhod'] = vhod(res)
    return res


def vhod(r):
    ok = dict(t_skladki=r['n_t_ge'] >= G['skladok_min'], t_srednyaya=r['t_mean'] >= G['t_srednyaya'], hit=r['hit'] >= G['hit_min'],
              tikery=r['n_tikerov_vyigrysh'] >= G['tikerov_min'], konc=r['konc'] <= G['konc_max'], c3=r['c3_vse'])
    return dict(**ok, proshyol=all(ok.values()), net=','.join(k for k, v in ok.items() if not v))


def n_test_dney(h=None):
    """N_test по h — число дат контрольного периода 2022–2026 (дирижёр 26.10: 1088 / 1084 / 1067; только счёт дней, без метрик).
    Если h нет в yaml — будние дни календаря 2022-01-01 … 2026-02-28 (тест не читается)."""
    nt = {int(k): int(v) for k, v in G.get('n_test', {}).items()}
    if h is not None and int(h) in nt:
        return nt[int(h)]
    return int(len(pd.bdate_range(G['test_s'], G['test_po'])))


def moshchnost(d, h, n_test=None):
    """Кластер 20: ожидаемая t на тесте по LOYO-оценкам среднего и долгосрочной дисперсии d; P(p < 0,05) в нормальной аппроксимации."""
    n_test = n_test or n_test_dney(h)
    d = np.asarray(d, float)
    v_day = FS.nw_var(d, h - 1) * len(d)  # долгосрочная дисперсия на одно наблюдение (nw_var — дисперсия среднего)
    t_exp = float(d.mean() / np.sqrt(v_day / n_test)) if v_day > 0 else (0.0 if d.mean() == 0 else float(np.inf * np.sign(d.mean())))
    z = stats.norm.ppf(1 - G['alpha'])
    p = float(stats.norm.cdf(t_exp - z))
    return dict(n_test=n_test, d_mean=float(d.mean()), sigma2_LR=float(v_day), t_ozhid=t_exp, P_p_lt_005=p,
                pometka='не хуже' if p < G['moshchnost_min'] else 'лучше')


def vybor(R, slozhnost):
    """R — {unit: результат ocenka}; slozhnost — {unit: c(u)}. v4 § 2: среди прошедших вход — макс. t̄; при Δt̄ ≤ 0,2 — больший средний
    выигрыш mean d; если и выигрыш различается не больше чем на 1% (относительно: Δ mean d / средняя потеря E ≤ 0,01; vyigrysh_pct —
    mean d в % к потере E, поэтому это Δ vyigrysh_pct ≤ 1) — меньшая сложность (затем большая t̄). -> (победитель | None, таблица)."""
    T = pd.DataFrame([dict(unit=u, t_mean=r['t_mean'], vyigrysh_pct=r['vyigrysh_pct'], slozhnost=slozhnost.get(u, 0),
                           proshyol=r['vhod']['proshyol'], net=r['vhod']['net'], hit=r['hit'], tikery=r['n_tikerov_vyigrysh'],
                           konc=r['konc'], t_ozhid=r['moshchnost']['t_ozhid'], P_moshchnost=r['moshchnost']['P_p_lt_005'],
                           pometka=r['moshchnost']['pometka']) for u, r in R.items()])
    c = T[T.proshyol]
    if c.empty:
        return None, T
    best = c.sort_values('t_mean', ascending=False).iloc[0]
    A = c[best.t_mean - c.t_mean <= G['dt']]
    bg = A.sort_values(['vyigrysh_pct', 't_mean'], ascending=False).iloc[0]
    B = A[bg.vyigrysh_pct - A.vyigrysh_pct <= 100 * G['dvyigrysh_otn']]
    w = B.sort_values(['slozhnost', 't_mean'], ascending=[True, False]).iloc[0]
    T['pobeditel'] = T.unit == w.unit
    return w.unit, T


# ---------------------------------------------------------------- дым и самопроверка
def dym(hs, tickers=None):
    """Реальные val-ряды: F_h(d0) seed 42 как «дизайн» против E_h (только 2017–2021)."""
    for h in hs:
        E = etalon_df(h)
        F = O.series([O.R / O.CFG['F_h_d0']['file'].format(h=h, seed=42)])
        if tickers:
            E, F = E[E.ticker.isin(tickers)], F[F.ticker.isin(tickers)]
        r = ocenka({'F_d0_s42': F}, E, h)
        print(f'h={h}: t по складкам {[round(x, 2) for x in r["folds"].t]}, средняя t {r["t_mean"]:.2f}, hit {r["hit"]:.3f}, '
              f'тикеров {r["n_tikerov_vyigrysh"]}/{r["n_tikerov"]}, конц. {r["konc"]:.2f}, c3 {r["c3_vse"]}, выигрыш {r["vyigrysh_pct"]:.2f}%, '
              f'вход {r["vhod"]["proshyol"]} ({r["vhod"]["net"] or "—"}); мощность: t_ож {r["moshchnost"]["t_ozhid"]:.2f}, '
              f'P {r["moshchnost"]["P_p_lt_005"]:.2f} -> {r["moshchnost"]["pometka"]}', flush=True)


def _sint(rng, eff, n_t=17, noise=0.4, years=range(2016, 2022), drift=None):
    """Синтетика: E и вариант u с относительной точностью eff (лог-шум у u меньше на eff) по тикерам и дням."""
    dates = pd.bdate_range(f'{min(years)}-01-04', f'{max(years)}-12-30')
    g = pd.DataFrame([(d, f'T{k}') for d in dates for k in range(n_t)], columns=KEY)
    g['year'] = g.date.dt.year
    g['step'] = g.year - 2015
    s = np.exp(rng.normal(-8, 0.6, len(g)))
    g['rv_actual'] = s * np.exp(rng.normal(0, 0.5, len(g)))
    E = g.assign(rv_pred=s * np.exp(rng.normal(0, noise, len(g))))
    ne = noise * (1 - eff) if drift is None else noise * (1 - eff * drift(g))
    u = g.assign(rv_pred=s * np.exp(rng.normal(0, 1, len(g)) * ne))
    return E, u


def selftest():
    rng = np.random.default_rng(17)
    # 1. DM — ровно код теста: равные прогнозы -> t = 0; u = E -> d ≡ 0
    E, u = _sint(rng, 0.0)
    r0 = ocenka({'a': E}, E, 1)
    assert abs(r0['t_mean']) < 1e-9 and r0['hit'] == 0 and not r0['vhod']['proshyol']
    # 2. явно лучший дизайн проходит вход; t по складкам совпадает с FS.dm_series на дневном ряде
    E, u = _sint(rng, 0.25)
    r = ocenka({'a': u}, E, 5)
    assert r['vhod']['proshyol'], r['vhod']
    g = d_panel(u, E)
    g = g[g.year == 2019]
    assert abs(r['folds'].set_index('Y').t[2019] - FS.dm_series(FS.daily(g.date.values, g.d.values), 5)['t']) < 1e-12
    # 3. концентрация (v4, ❓ 2): выигрыш только в 1% самых турбулентных дней (наибольшая потеря E) -> konc > 0,40, вход не пройден;
    #    такой же по размеру выигрыш в 1% спокойных дней -> эти дни не «худшие», konc мала
    lEd = d_panel(E.assign(rv_pred=E.rv_actual), E).groupby('date').lE.mean().sort_values(ascending=False)
    k1 = int(0.01 * len(lEd))
    for dni, bolshe in ((set(lEd.index[:k1]), True), (set(lEd.index[-k1:]), False)):
        uk = E.copy()
        m = uk.date.isin(dni).values
        uk.loc[m, 'rv_pred'] = uk.loc[m, 'rv_actual'] * np.exp(rng.normal(0, 0.05, m.sum()))
        rk = ocenka({'a': uk}, E, 1)
        assert (rk['konc'] > 0.40) == bolshe and rk['vhod']['konc'] != bolshe, (bolshe, rk['konc'])
    dd_ = pd.Series([1.0, -2.0, 0.5]); ee_ = pd.Series([3.0, 1.0, 2.0])
    assert koncentraciya(dd_, ee_) == np.inf  # Σ d ≤ 0 — вход не пройден
    # 4. мощность: t_ож растёт как √N; P = Φ(t − 1,645); маленький эффект -> «не хуже»
    d = rng.normal(0.01, 0.2, 1250)
    m1, m4 = moshchnost(d, 1, 250), moshchnost(d, 1, 1000)
    assert abs(m4['t_ozhid'] / m1['t_ozhid'] - 2) < 1e-9 and abs(m1['P_p_lt_005'] - stats.norm.cdf(m1['t_ozhid'] - stats.norm.ppf(0.95))) < 1e-12
    assert moshchnost(rng.normal(0.001, 0.2, 1250), 1, 1000)['pometka'] == 'не хуже'
    assert moshchnost(rng.normal(0.05, 0.2, 1250), 1, 1000)['pometka'] == 'лучше'
    assert (n_test_dney(1), n_test_dney(5), n_test_dney(22)) == (1088, 1084, 1067) and 1020 < n_test_dney() < 1100
    assert moshchnost(d, 22)['n_test'] == 1067
    # 5. выбор: макс. t̄; Δt̄ ≤ 0,2 -> больший выигрыш; Δt̄ ≤ 0,2 и разница выигрыша ≤ 1% потери E -> проще
    mk = lambda t, g_, ok=True: dict(t_mean=t, vyigrysh_pct=g_, vhod=dict(proshyol=ok, net=''), hit=0.6, n_tikerov_vyigrysh=15, konc=0.1,
                                    moshchnost=dict(t_ozhid=3, P_p_lt_005=0.9, pometka='лучше'))
    w, _ = vybor({'A': mk(3.0, 5.0), 'B': mk(2.9, 7.0), 'C': mk(2.5, 9.0)}, {'A': 3, 'B': 3, 'C': 1})
    assert w == 'B'
    w, _ = vybor({'A': mk(3.0, 5.0), 'B': mk(2.9, 5.5), 'C': mk(2.85, 5.2)}, {'A': 3, 'B': 4, 'C': 1})
    assert w == 'C'
    w, _ = vybor({'A': mk(3.0, 5.0, ok=False), 'B': mk(1.9, 4.0)}, {'A': 1, 'B': 2})
    assert w == 'B' and vybor({'A': mk(3.0, 5.0, ok=False)}, {})[0] is None
    # 6. вложенный выбор по c3 и без утечки: шум на отложенном году не меняет вариант складки
    E, u1 = _sint(rng, 0.2)
    _, u2 = _sint(rng, 0.1)
    r1 = ocenka({'v1': u1, 'v2': u2}, E, 1)
    u1b = u1.copy()
    mm = u1b.year == 2018
    u1b.loc[mm, 'rv_pred'] *= np.exp(rng.normal(0, 1.0, mm.sum()))
    r2 = ocenka({'v1': u1b, 'v2': u2}, E, 1)
    assert r1['folds'].set_index('Y').variant[2018] == r2['folds'].set_index('Y').variant[2018]
    assert (r1['folds'].variant == 'v1').all()  # вложенный выбор по t на обучающих годах берёт лучший вариант
    F1 = r1['folds'].set_index('Y')
    assert abs(F1.t_obuch[2019] - np.mean([F1.t[y] for y in (2017, 2018, 2020, 2021)])) < 1e-12  # t обучения = t тех же лет
    print('selftest r2_gates: OK (v4: DM = код теста; вход кластера 17: t, hit, тикеры, концентрация по дням наибольшей потери эталона, c3; '
          'вложенный выбор по t на обучающих годах; выбор с Δt̄ и относительным выигрышем; мощность кластера 20; без утечки)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', nargs='?', choices=['dym'])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--h', type=int, nargs='*', default=[1, 5, 22])
    ap.add_argument('--tickers', nargs='*', default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == 'dym':
        return dym(a.h, a.tickers)
    ap.print_help()


if __name__ == '__main__':
    main()
