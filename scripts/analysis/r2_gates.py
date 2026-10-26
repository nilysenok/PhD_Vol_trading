#!/usr/bin/env python3
"""R2, агент O: ворота по кластеру 17 и расчёт мощности (кластер 20) — решение владельца 26.10, 16:25 (R2 на Маке; протокол v4 пишет
архитектор). Панельный DM — тем же кодом, что в тесте: ftp_stats.qloss, daily (дневной ряд средней по тикерам), dm_series (NW, лаг h−1).

Для дизайна u (варианты — значения-списки, выбор внутри складки по c3, как r2_loyo п. 2.1) и эталона В2 (HAR-J + s² sm, d0 = E_h):
- складка Y: d_t = средняя по тикерам разность потерь QLIKE (E − u) за день t отложенного года Y; t_Y = dm_series(d_t, h).t;
- вход: t_Y ≥ 1,0 не меньше чем в 3 складках из 5 и средняя t ≥ 1,5; hit rate (доля дней d_t > 0 по 5 отложенным годам) ≥ 55%;
  выигрыш (средняя разность потерь тикера > 0) не меньше чем на 12 тикерах из 17; концентрация — на 2% самых турбулентных
  дней (наибольшая потеря эталона, ❓ 2) приходится не больше 40% суммарного выигрыша Σ d_t; c3 на всех складках (r2_loyo.crit, против E_h);
- выбор среди прошедших: максимум средней t; при Δt ≤ 0,2 — больший средний выигрыш (в % к средней потере E); при Δt ≤ 0,2 и разнице
  выигрыша ≤ 1 п. п. — более простой дизайн (сложность по schema_v2, r2_loyo.slozhnost = r2_core.complexity);
- мощность (кластер 20): по пулу 5 отложенных лет — среднее d̄ и долгосрочная дисперсия NW на день; на N_test днях 2022–2026
  ожидаемая t = d̄ / √(σ²_LR / N_test), P(p < 0,05) = Φ(t − z_0,95) (одностороннее); решение владельца 16:45 (вместо 0,70): P ≥ 0,8 — «значимый»,
  0,5 ≤ P < 0,8 — «не хуже, ожидание на границе», P < 0,5 — на h остаётся нога (финал = эконометрическая компонента, V0); itog_E;
- ворота C, D, E (v4 §§ 2, 4–6): gate_C_v4 / gate_D_v4 / gate_E_v4 — payload файла ворот (пишет драйвер S); 2σ по 10 seed — sigma2_F.
Читаются только val 2017–2021 (ряды подаются кадрами; r2_loyo.read_val / r2_core.load_series). N_test — календарь будних дней, тест не читается.

python3 scripts/analysis/r2_gates.py --selftest
python3 scripts/analysis/r2_gates.py dym --h 1 5 22      # дым: F_h(d0) seed 42 против E_h на реальных val-рядах (3 тикера или все)"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
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
    kat = kategoriya_moshchnosti(p)
    return dict(n_test=n_test, d_mean=float(d.mean()), sigma2_LR=float(v_day), t_ozhid=t_exp, P_p_lt_005=p, **kat)


def kategoriya_moshchnosti(p):
    """Решение владельца 16:45 (ворота E, кластер 20): P ≥ 0,8 — «значимый»; 0,5 ≤ P < 0,8 — «не хуже, ожидание на границе»;
    P < 0,5 — на h остаётся нога (финал = эконометрическая компонента, V0). -> {pometka, kategoriya, deystvie}."""
    pr = G['moshchnost_porogi']
    if p >= pr['znachimyy']:
        return dict(pometka='значимый', kategoriya='znachimyy', deystvie='na_otkrytie')
    if p >= pr['granica']:
        return dict(pometka='не хуже, ожидание на границе', kategoriya='granica', deystvie='na_otkrytie')
    return dict(pometka='остаётся нога (V0)', kategoriya='noga', deystvie='final_V0')


def itog_E(pobeditel, moshch, noga_v0=None):
    """Итог ворот E с учётом мощности победителя: при P < 0,5 на h остаётся нога (V0-кандидат победителя — его эконометрическая
    компонента, noga_v0), иначе победитель идёт на открытие с формулировкой категории. -> dict для gate_E."""
    k = kategoriya_moshchnosti(moshch['P_p_lt_005'])
    final = pobeditel if k['deystvie'] == 'na_otkrytie' else (noga_v0 or 'noga (V0)')
    return dict(pobeditel_otbora=pobeditel, final=final, P_p_lt_005=moshch['P_p_lt_005'], t_ozhid=moshch['t_ozhid'],
                n_test=moshch['n_test'], **k)


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


# ---------------------------------------------------------------- ворота C, D, E (v4 §§ 2, 4–6): содержимое файла ворот
# Функции возвращают payload; файл пишет драйвер S своим write_gate (= r2_loyo.write_gate вне дыма). Ключи, которые читает драйвер
# (C: yacheyki, v0_finalisty, dizainy; D: luchshiy, dizayn; E: pobeditel, design_sha, dizayn, sigma2), сохранены.
GV4 = O.CFG['gates_v4']


def rol(d):
    """boost.role дизайна (schema: none = V0)."""
    return str(((d or {}).get('boost') or {}).get('role', 'none'))


def noga_tip(d):
    return str(((d or {}).get('leg') or {}).get('type', 'HARJ'))


def poryadok(R, S, k=None):
    """Порядок выбора кластера 17: vybor повторно (победитель снимается, выбор заново) среди прошедших вход. -> (список, таблица vybor
    по всем с колонкой poryadok)."""
    left, out = dict(R), []
    while left and (k is None or len(out) < k):
        w, _ = vybor(left, S)
        if w is None:
            break
        out.append(w)
        left.pop(w)
    _, T = vybor(R, S) if R else (None, pd.DataFrame(columns=['unit']))
    T['poryadok'] = [out.index(u) + 1 if u in out else np.nan for u in T.unit]
    return out, T


def chislo(R):
    """(перебрано, прошли вход) — для r2_loyo.lozhnye (п. 2.7 + доля прошедших вход на уровне)."""
    return len(R), int(sum(bool(r['vhod']['proshyol']) for r in R.values()))


def gate_C_v4(R, S, dizainy, nepr=None, k=None):
    """v4 § 4. R — {unit: ocenka} всех ячеек C (нога × роль, V0); dizainy — {unit: дизайн dict}; nepr — {unit: [факторы]} (неприменимые).
    Ячейки D — первые k (2) с ролью ≠ none в порядке выбора; V0 (роль none, нога из gates_v4.v0_nogi), прошедшие вход, — в финалисты."""
    k = k or GV4['k_C']
    boost = {u: r for u, r in R.items() if rol(dizainy.get(u)) != 'none'}
    v0 = {u: r for u, r in R.items() if rol(dizainy.get(u)) == 'none' and noga_tip(dizainy.get(u)) in GV4['v0_nogi']}
    two, Tb = poryadok(boost, S, k)
    v0p, _ = poryadok(v0, S)
    _, T = vybor(R, S) if R else (None, pd.DataFrame(columns=['unit']))
    T['rol'] = [rol(dizainy.get(u)) for u in T.unit]
    T['noga'] = [noga_tip(dizainy.get(u)) for u in T.unit]
    T['poryadok_yacheyki'] = T.unit.map(dict(zip(Tb.unit, Tb.poryadok)))
    T['v0_finalist'] = T.unit.isin(v0p)
    T['faktor_neprimenim'] = [', '.join((nepr or {}).get(u, [])) for u in T.unit]
    N = {'C_boost': chislo(boost), 'C_V0': chislo(v0)}
    return dict(yacheyki=two, v0_finalisty=v0p, pobeditel_vybora=two[0] if two else None, net_yacheek=not two,
                pravilo='v4 § 4: вход кластера 17 + c3; порядок — макс. t̄, Δt̄ ≤ 0,2 → выигрыш, Δвыигрыш ≤ 1% → проще; роль ≠ none',
                chislo=O.lozhnye(N, 0)['urovni'], tablica=T.to_dict('records'),
                faktor_neprimenim={u: v for u, v in (nepr or {}).items() if u in R},
                dizainy={u: dizainy[u] for u in two + v0p if u in dizainy})


def gate_D_v4(R, S, dizainy, zhivye, n_tochek=None):
    """v4 § 5, итог 8 → 1 на ячейку. R — {unit: ocenka} выживших точек (3 seed, все 5 складок); zhivye — {ячейка: [unit выживших]};
    n_tochek — {ячейка: число точек LHS на входе} (для числа дизайнов). Лучшая на ячейку — выбор кластера 17 среди её выживших;
    лучший уровня (luchshiy -> E) — выбор среди лучших ячеек. Никто не прошёл вход — luchshiy None (на h остаётся F_h(d0))."""
    luch = {}
    for c, us in zhivye.items():
        w, _ = vybor({u: R[u] for u in us if u in R}, S) if any(u in R for u in us) else (None, None)
        luch[c] = w
    best = {u: R[u] for u in luch.values() if u}
    w, _ = vybor(best, S) if best else (None, None)
    _, T = vybor(R, S) if R else (None, pd.DataFrame(columns=['unit']))
    T['yacheyka'] = T.unit.map({u: c for c, us in zhivye.items() for u in us})
    T['luchshaya_v_yacheyke'] = T.unit.isin([u for u in luch.values() if u])
    N = {f'D_{c}': (int((n_tochek or {}).get(c, len(us))), chislo({u: R[u] for u in us if u in R})[1]) for c, us in zhivye.items()}
    return dict(luchshiy=w, dizayn=dizainy.get(w) if w else None, luchshie=luch, zhivye=zhivye,
                pravilo='v4 § 5: 60 → 30 → 15 → 8 (r2_halving); 8 → 1 на ячейку и лучший уровня — вход и выбор кластера 17',
                chislo=O.lozhnye(N, 0)['urovni'], tablica=T.to_dict('records'))


def sigma2_F(cand, h, seeds=None, F=None):
    """2σ (Д24.3) по 10 seed парно с F_h(d0): cand — {seed: кадр val итогового прогноза} или кадр (детерминированный ряд, V0);
    F — {seed: кадр} (по умолчанию ряды ВЫР-1 R2/cache/VYR/F_d0_h{h}_s{seed}_val.parquet). Q̄_T — средний QLIKE шагов 2–6 (bez:
    поправка уже в ряде). -> r2_loyo.sigma2."""
    seeds = list(seeds or O.SEEDS)
    fc = O.CFG['F_h_d0']
    F = F or {s: O.series([O.R / fc['file'].format(h=h, seed=s)]) for s in seeds}
    qF = {s: O.qbar_T(O.q_steps(F[s], fc['s2'])) for s in seeds}
    qc = {s: O.qbar_T(O.q_steps(cand[s], 'bez')) for s in seeds} if isinstance(cand, dict) else O.qbar_T(O.q_steps(cand, 'bez'))
    return O.sigma2(qc, qF, seeds)


def _noga(pob, dizainy, v0_all, lestnica):
    """Нога победителя для P < 0,5: сам V0; иначе V0 с той же ногой (leg целиком, затем тип); иначе ступень лестницы «нога + калибровка»."""
    if pob in v0_all:
        return dict(unit=pob, istochnik='V0 — победитель')
    lg = (dizainy.get(pob) or {}).get('leg') or {}
    for same in (lambda d: ((d or {}).get('leg') or {}) == lg, lambda d: noga_tip(d) == noga_tip(dizainy.get(pob))):
        m = [u for u in v0_all if same(dizainy.get(u))]
        if m:
            return dict(unit=m[0], istochnik='V0 с той же ногой')
    st = next((r for r in (lestnica or []) if r.get('stupen') == GV4['noga_lestnica']), None)
    if st:
        return dict(unit=f'noga_{pob}', istochnik=f'лестница {GV4["noga_lestnica"]}', design_sha=st['design_sha'],
                    design_yaml=st['design_yaml'], nuzhen_schet=True)
    return dict(unit=None, istochnik='нет V0 и лестницы — ❓ дирижёру (умолчание: F_h(d0))')


def gate_E_v4(R, S, dizainy, sig, h, v0_units=(), N=None, k=None, lestnica=True, out_dir=None):
    """v4 § 2, § 6. R — {unit: ocenka на 10 seed (LightGBM + XGBoost), вход заново}; sig — {unit: sigma2_F}; v0_units — V0 из ворот C.
    Финалисты — первые k (3) не-V0 в порядке выбора + V0, прошедшие вход; победитель — первый финалист, прошедший 2σ (никто — F_h(d0)).
    Мощность победителя (кластер 20) и itog_E (владелец 16:45): P < 0,5 — финал = нога (V0). В `pobeditel` — то, что идёт на открытие
    (агент A собирает pobeditel.pobeditel), выбор отбора — pobeditel.pobeditel_otbora. Хэши финалистов, лестница и абляции — r2_loyo."""
    k = k or GV4['k_E']
    out_dir = Path(out_dir or O.OUT / f'gate_E_h{h}_dizainy')
    v0_all = [u for u in v0_units]
    top, _ = poryadok({u: r for u, r in R.items() if u not in v0_all}, S, k)
    v0p, _ = poryadok({u: r for u, r in R.items() if u in v0_all}, S)
    fin = top + v0p
    pob = next((u for u in fin if (sig.get(u) or {}).get('passed')), None)
    hs = {}
    for u in fin:
        if u in dizainy:
            p = out_dir / f'finalist_{u}.yaml'
            O._zapisat_dizain(dizainy[u], p)
            hs[u] = O.dizain_hashi(p)
    L = A = None
    if pob and lestnica and pob in dizainy:
        L, A = O.lestnica_ablyacii(dizainy[pob], h, out_dir)
    if pob:
        mosh = R[pob]['moshchnost']
        noga = _noga(pob, dizainy, v0_all, L)
        it = itog_E(pob, mosh, noga_v0=noga['unit'])
        final = it['final'] if (it['deystvie'] == 'na_otkrytie' or noga['unit']) else 'F_h(d0)'
        it['final'] = final
        V0 = final != 'F_h(d0)' and (pob in v0_all or it['deystvie'] == 'final_V0')
        it['gipoteza'] = GV4['gipoteza'][it['kategoriya']]
    else:
        mosh, noga, final, V0 = None, None, 'F_h(d0)', False
        it = dict(pobeditel_otbora=None, final='F_h(d0)', kategoriya=None, pometka='2σ не прошёл никто — на h остаётся F_h(d0)')
    fh = hs.get(final) or ({'design_sha': noga.get('design_sha'), 'design_yaml': noga.get('design_yaml')} if noga and final == noga['unit'] else {})
    diz = dizainy.get(final) or (yaml.safe_load(Path(fh['design_yaml']).read_text()) if fh.get('design_yaml') else None)
    rows = [dict(unit=u, tip='V0' if u in v0_all else 'top', poryadok=i + 1, t_mean=R[u]['t_mean'], vyigrysh_pct=R[u]['vyigrysh_pct'],
                 slozhnost=S.get(u, 0), vhod=bool(R[u]['vhod']['proshyol']), P_moshchnost=R[u]['moshchnost']['P_p_lt_005'],
                 **{a: b for a, b in hs.get(u, {}).items() if a != 'canonical_yaml'},
                 **{f's2_{a}': b for a, b in (sig.get(u) or {}).items()}) for i, u in enumerate(fin)]
    _, T = vybor(R, S) if R else (None, pd.DataFrame(columns=['unit']))
    N = dict(N or {})
    N['E_10seed'] = chislo(R)
    return dict(finalisty=rows, sigma2={u: sig[u] for u in fin if u in sig},
                pobeditel=dict(pobeditel=final, V0=bool(V0), pobeditel_otbora=pob or 'F_h(d0)', poryadok=fin, noga=noga, itog=it),
                design_sha=fh.get('design_sha'), design_yaml=fh.get('design_yaml'), dizayn=diz,
                moshchnost=mosh, lestnica=L, ablyacii=A, primechanie_Fd0=O.CFG['F_h_d0'].get('otstuplenie', ''),
                lozhnye=O.lozhnye(N, len(fin)), tablica=T.to_dict('records'),
                ogovorka_moshchnosti='расчёт мощности по LOYO оптимистичен: выбор на тех же складках (v4 § 2)')


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
    # 4. мощность: t_ож растёт как √N; P = Φ(t − 1,645); категории владельца 16:45
    d = rng.normal(0.01, 0.2, 1250)
    m1, m4 = moshchnost(d, 1, 250), moshchnost(d, 1, 1000)
    assert abs(m4['t_ozhid'] / m1['t_ozhid'] - 2) < 1e-9 and abs(m1['P_p_lt_005'] - stats.norm.cdf(m1['t_ozhid'] - stats.norm.ppf(0.95))) < 1e-12
    # пороги владельца 16:45: ≥ 0,8 — значимый; [0,5; 0,8) — граница; < 0,5 — нога (V0)
    assert moshchnost(rng.normal(0.001, 0.2, 1250), 1, 1000)['kategoriya'] == 'noga'
    assert moshchnost(rng.normal(0.05, 0.2, 1250), 1, 1000)['kategoriya'] == 'znachimyy'
    for p_, k_ in ((0.95, 'znachimyy'), (0.8, 'znachimyy'), (0.79, 'granica'), (0.5, 'granica'), (0.49, 'noga'), (0.0, 'noga')):
        assert kategoriya_moshchnosti(p_)['kategoriya'] == k_, (p_, k_)
    it = itog_E('D1', dict(P_p_lt_005=0.3, t_ozhid=1.1, n_test=1084), noga_v0='HARX_s2')
    assert it['final'] == 'HARX_s2' and it['deystvie'] == 'final_V0'
    it = itog_E('D1', dict(P_p_lt_005=0.6, t_ozhid=1.9, n_test=1084))
    assert it['final'] == 'D1' and it['pometka'] == 'не хуже, ожидание на границе'
    assert (n_test_dney(1), n_test_dney(5), n_test_dney(22)) == (1088, 1084, 1067) and 1020 < n_test_dney() < 1100
    assert moshchnost(d, 22)['n_test'] == 1067
    # 5. выбор: макс. t̄; Δt̄ ≤ 0,2 -> больший выигрыш; Δt̄ ≤ 0,2 и разница выигрыша ≤ 1% потери E -> проще
    mk = lambda t, g_, ok=True: dict(t_mean=t, vyigrysh_pct=g_, vhod=dict(proshyol=ok, net=''), hit=0.6, n_tikerov_vyigrysh=15, konc=0.1,
                                    moshchnost=dict(t_ozhid=3, P_p_lt_005=0.9, pometka='значимый'))
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
    selftest_vorota(rng)
    print('selftest r2_gates: OK (v4: DM = код теста; вход кластера 17: t, hit, тикеры, концентрация по дням наибольшей потери эталона, c3; '
          'вложенный выбор по t на обучающих годах; выбор с Δt̄ и относительным выигрышем; мощность кластера 20; без утечки)')


def selftest_vorota(rng):
    """Ворота C, D, E v4 на синтетических результатах ocenka (вход, порядок выбора, роли, V0, 2σ, мощность, нога)."""
    import tempfile
    mk = lambda t, g_, ok=True, P=0.9: dict(t_mean=t, vyigrysh_pct=g_, vhod=dict(proshyol=ok, net='' if ok else 'konc'), hit=0.6,
                                           n_tikerov_vyigrysh=15, konc=0.1,
                                           moshchnost=dict(t_ozhid=3, P_p_lt_005=P, pometka='', n_test=1084))
    dz = lambda name, leg, role, **kw: dict(design=name, h=5, leg=dict(type=leg), boost=dict(role=role), **kw)
    # --- C: две ячейки с ролью ≠ none в порядке выбора; V0 только из v0_nogi и только прошедшие вход
    D = {'HARX_res': dz('HARX_res', 'HARX', 'residual'), 'HARX_feat': dz('HARX_feat', 'HARX', 'feature'),
         'HARJ_off': dz('HARJ_off', 'HARJ', 'offset'), 'HEAVY_res': dz('HEAVY_res', 'HEAVY', 'residual'),
         'HARX_V0': dz('HARX_V0', 'HARX', 'none'), 'RVI_anchor_V0': dz('RVI_anchor_V0', 'RVI_anchor', 'none'),
         'HEAVY_V0': dz('HEAVY_V0', 'HEAVY', 'none')}
    R = {'HARX_res': mk(3.0, 5.0), 'HARX_feat': mk(2.9, 7.0), 'HARJ_off': mk(2.0, 9.0), 'HEAVY_res': mk(4.0, 9.0, ok=False),
         'HARX_V0': mk(5.0, 9.0), 'RVI_anchor_V0': mk(1.0, 1.0, ok=False), 'HEAVY_V0': mk(2.0, 2.0)}
    S = {u: (0 if rol(d) == 'none' else 2) for u, d in D.items()}
    gC = gate_C_v4(R, S, D, nepr={'HARJ_off': ['J']})
    assert gC['yacheyki'] == ['HARX_feat', 'HARX_res'], gC['yacheyki']  # V0 с t̄ 5 — не ячейка; HEAVY_res не прошёл вход
    assert gC['v0_finalisty'] == ['HARX_V0'] and set(gC['dizainy']) == {'HARX_feat', 'HARX_res', 'HARX_V0'}
    assert gC['chislo']['C_boost'] == dict(perebrano=4, proshli_vhod=3, dolya_vhoda=0.75) and gC['faktor_neprimenim'] == {'HARJ_off': ['J']}
    assert gate_C_v4({'HEAVY_res': R['HEAVY_res']}, S, D)['net_yacheek']
    # --- D: лучшая точка на ячейку и лучший уровня
    RD = {'a_p1': mk(2.0, 3.0), 'a_p2': mk(2.5, 3.0), 'b_p1': mk(2.4, 6.0), 'b_p2': mk(9.0, 9.0, ok=False)}
    DD = {u: dz(u, 'HARX', 'residual') for u in RD}
    gD = gate_D_v4(RD, {u: 1 for u in RD}, DD, {'a': ['a_p1', 'a_p2'], 'b': ['b_p1', 'b_p2']}, {'a': 60, 'b': 60})
    assert gD['luchshie'] == {'a': 'a_p2', 'b': 'b_p1'} and gD['luchshiy'] == 'b_p1' and gD['dizayn'] == DD['b_p1']  # Δt̄ 0,1 → выигрыш
    assert gD['chislo']['D_b'] == dict(perebrano=60, proshli_vhod=1, dolya_vhoda=1 / 60)
    assert gate_D_v4({'x': mk(3, 3, ok=False)}, {}, {'x': {}}, {'a': ['x']})['luchshiy'] is None
    # --- 2σ к F_h(d0): парно по seed; детерминированный V0 — число
    E, u = _sint(rng, 0.2, years=range(2016, 2022))
    seeds = list(range(42, 52))
    F = {s: E.assign(rv_pred=E.rv_pred * np.exp(rng.normal(0, 0.01, len(E)))) for s in seeds}
    good = {s: u.assign(rv_pred=u.rv_pred * np.exp(rng.normal(0, 0.01, len(u)))) for s in seeds}
    sg = sigma2_F(good, 5, seeds, F)
    assert sg['passed'] and sg['n'] == 10 and sg['n_better'] == 10
    assert not sigma2_F(F, 5, seeds, F)['passed'] and sigma2_F(u, 5, seeds, F)['passed']
    # --- E: финалисты = top-3 по выбору + V0; победитель — первый, прошедший 2σ; мощность и нога
    RE = {'E1': mk(3.0, 5.0, P=0.95), 'E2': mk(2.7, 4.0, P=0.6), 'E3': mk(2.0, 3.0, P=0.3), 'E4': mk(1.9, 2.0), 'E5': mk(5.0, 9.0, ok=False),
          'HARX_V0': mk(1.6, 2.0, P=0.2)}
    DE = {u: dz(u, 'HARX', 'residual', features=['G0', 'G1']) for u in RE if u != 'HARX_V0'} | {'HARX_V0': D['HARX_V0']}
    SE = {u: 1 for u in RE}
    yes, no = dict(passed=True, d_mean=-1e-3, sigma=1e-4, thr=-6e-5), dict(passed=False, d_mean=1e-3, sigma=1e-4, thr=-6e-5)
    with tempfile.TemporaryDirectory() as td:
        gE = gate_E_v4(RE, SE, DE, {'E1': yes, 'E2': yes, 'E3': yes, 'HARX_V0': yes}, 5, v0_units=['HARX_V0'], out_dir=td)
        pb = gE['pobeditel']
        assert pb['poryadok'] == ['E1', 'E2', 'E3', 'HARX_V0'] and pb['pobeditel'] == 'E1' and not pb['V0']
        assert pb['itog']['kategoriya'] == 'znachimyy' and pb['itog']['gipoteza'] == 'лучше эталона'
        assert gE['design_sha'] == gE['finalisty'][0]['design_sha'] and len(gE['design_sha']) == 64 and gE['lestnica'][-1]['stupen'] == 'финал'
        assert gE['lozhnye']['urovni']['E_10seed'] == dict(perebrano=6, proshli_vhod=5, dolya_vhoda=5 / 6)
        # E1 не прошёл 2σ -> E2 (P 0,6: граница, «не хуже»)
        gE = gate_E_v4(RE, SE, DE, {'E1': no, 'E2': yes}, 5, v0_units=['HARX_V0'], out_dir=td, lestnica=False)
        assert gE['pobeditel']['pobeditel'] == 'E2' and gE['pobeditel']['itog']['kategoriya'] == 'granica'
        assert gE['pobeditel']['itog']['gipoteza'].startswith('не хуже')
        # победитель E3 с P 0,3 -> финал = V0 с той же ногой (HARX_V0), V0 = true; design_sha — у ноги
        gE = gate_E_v4(RE, SE, DE, {'E3': yes}, 5, v0_units=['HARX_V0'], out_dir=td)
        pb = gE['pobeditel']
        assert pb['pobeditel_otbora'] == 'E3' and pb['pobeditel'] == 'HARX_V0' and pb['V0'] and pb['itog']['deystvie'] == 'final_V0'
        assert gE['design_sha'] == next(r for r in gE['finalisty'] if r['unit'] == 'HARX_V0')['design_sha'] and gE['dizayn'] == D['HARX_V0']
        # без V0 той же ноги -> ступень лестницы «нога + калибровка», нужен счёт
        gE = gate_E_v4({u: RE[u] for u in ('E1', 'E3')}, SE, DE, {'E3': yes}, 5, out_dir=td)
        pb = gE['pobeditel']
        assert pb['noga']['istochnik'].startswith('лестница') and pb['noga']['nuzhen_schet'] and pb['V0']
        assert gE['design_sha'] == next(r for r in gE['lestnica'] if r['stupen'] == GV4['noga_lestnica'])['design_sha']
        assert gE['dizayn']['leg']['type'] == 'HARX' and 'boost' not in gE['dizayn']
        # никто не прошёл 2σ -> F_h(d0)
        gE = gate_E_v4(RE, SE, DE, {}, 5, v0_units=['HARX_V0'], out_dir=td, lestnica=False)
        assert gE['pobeditel']['pobeditel'] == 'F_h(d0)' and gE['design_sha'] is None and gE['moshchnost'] is None
        # победил V0 сам (P 0,2): финал — он же
        gE = gate_E_v4(RE, SE, DE, {'HARX_V0': yes}, 5, v0_units=['HARX_V0'], out_dir=td, lestnica=False)
        assert gE['pobeditel']['pobeditel'] == 'HARX_V0' and gE['pobeditel']['V0']
        O.json.dumps(gE, default=O._js)  # сериализуется для write_gate


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
