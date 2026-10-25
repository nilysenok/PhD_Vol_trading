"""Поправка Б0.1: F=exp(s2/2), одна и та же оценка s2 для HAR-J, XGB, LGB (и, значит, гибрида). Только валидационные остатки.
Оценки s2 (решение 25.10): ord - дисперсия остатков val-года шага; rob - (1,4826*MAD)^2 остатков val-года; sm - дисперсия по объединённым
остаткам трёх последних val-лет (на первых шагах - сколько есть); smrob - (1,4826*MAD)^2 по объединённым остаткам трёх лет.
Для сравнения на валидации - вне выборки: на val шага k берётся оценка по годам до val-года k (шаг k-1; на шаге 1 F=1); на тесте шага k - оценка шага k.
ord_in - справка: оценка по тому же val-году (in-sample)."""
import numpy as np

CODES = ('ord', 'rob', 'sm', 'smrob')
COMPS = ('har', 'xgb', 'lgb', 'hcj')  # hcj - HAR-CJ (Д8.6), та же оценка s2, свои остатки


def resid(val):
    """Остатки ln(rv)-ln(прогноз) по компонентам и шагам; считать ДО поправки."""
    return {c: {st: np.log(g['rv_actual'].values + 1e-10) - np.log(g[c].values) for st, g in val.groupby('step')} for c in COMPS}


def s2_est(code, E, c, st):
    code = code.replace('_in', '')
    e = np.concatenate([E[c][j] for j in range(max(1, st - 2) if code in ('sm', 'smrob') else st, st + 1)])
    if code in ('rob', 'smrob'):
        return float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2)
    return float(np.var(e))


def correct(val, test, E, code, h, log):
    insample = code.endswith('_in')
    for st in sorted(val['step'].unique()):
        vm, tm = val['step'] == st, test['step'] == st
        for c in COMPS:
            s2 = s2_est(code, E, c, st)
            fo = np.exp(s2 / 2)
            fp = np.exp(s2_est(code, E, c, st - 1) / 2) if st > 1 else 1.0
            val.loc[vm, c] *= fo if insample else fp
            test.loc[tm, c] *= fo
            log.append(dict(h=h, step=st, comp=c, code=code, s2=s2, F_own=fo, F_oos=fp))
