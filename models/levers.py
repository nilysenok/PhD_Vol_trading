"""Рычаги уровня 2 (владелец 25.10, 18:35) поверх обучения по протоколу B v2 (models/v2.py):
V9     harj_residual — цель ln RV - ln F_HARJ_corr, прогноз F_HARJ_corr * exp(прогноз остатка);
Д8.12  harj_feature  — ln F_HARJ_corr как признак;
Д8.6   h1_feature    — прогноз модели h=1 (та же модель, тот же шаг, параметры дизайна для h=1) как признак на h=5/22;
Д8.4   avg_prev_step — среднее прогнозов моделей шагов t и t-1 (постобработка в pipeline.avg_prev).
HAR-J: оценка etap2_okno.fit_predict (как walk_forward_full.py:193-204) на train_pure шага; F_corr = exp(yhat + s2/2), s2 — дисперсия
остатков обучающего окна (известна на дату прогноза). Прогноз h=1 на строках обучения — внутривыборочный (модель h=1 видела эти строки):
оговорка рычага Д8.6, проверяется только на валидации."""
import numpy as np, pandas as pd
from . import cache as K, config as C, data as D

_a = {}


def _all(dk):
    if dk not in _a:
        _a.clear()
        _a[dk] = D.read_all(dk)
    return _a[dk]


def _idx(fk, nm):
    return np.load(K.stage_dir('features', fk) / f'idx_{nm}.npy')


def harj_lnf(t):
    """ln F_HARJ_corr для строк tp, vi, td, nx шага и tp1 (строки шага 1 — для бинов reference)."""
    import etap2_okno as E2
    a, harj = _all(t['dk'])
    tg = f'rv_target_h{t["h"]}'
    tp = a.loc[_idx(t['fk'], 'tp')]
    m, s2 = E2.fit_predict(tp, harj, tg, None, None)
    f = lambda idx: m.predict(E2.clean(np.log(a.loc[idx, harj].values.clip(1e-10)))) + s2 / 2 if len(idx) else np.zeros(0)
    out = {nm: f(_idx(t['fk'], nm)) for nm in ('tp', 'vi', 'td', 'nx')}
    out['tp1'] = f(_idx(t['fk1'], 'tp'))
    return out


def h1_pred(t):
    from . import train as TR
    r = TR.run_task(dict(t['h1task'], save_all=True))
    allp = np.load(K.stage_dir('allpred', r['key']) / 'pred.npy')
    out = {nm: allp[_idx(t['fk'], nm)] for nm in ('tp', 'vi', 'td', 'nx')}
    out['tp1'] = allp[_idx(t['fk1'], 'tp')]
    return out


def prepare(t):
    """-> (extra: {nm: 2D} | None, yoff: {nm: 1D} | None)."""
    lv, cols, yoff = t.get('levers') or {}, [], None
    if lv.get('harj_residual') or lv.get('harj_feature'):
        lnf = harj_lnf(t)
        if lv.get('harj_residual'):
            yoff = lnf
        if lv.get('harj_feature'):
            cols.append(lnf)
    if lv.get('h1_feature') and t['h'] != 1:
        cols.append(h1_pred(t))
    extra = {nm: np.column_stack([c[nm] for c in cols]) for nm in ('tp', 'vi', 'td', 'nx', 'tp1')} if cols else None
    return extra, yoff


def avg_prev(keys, steps):
    """Д8.4: val шага t = среднее(val_t, test_{t-1}); test шага t = среднее(test_t, next_{t-1}) по (date, ticker). Шаг 1 — без изменений."""
    from .cache import pred_path, write_atomic
    out = []
    for i, (k, s) in enumerate(zip(keys, steps)):
        if i == 0 or steps[i - 1] != s - 1:
            out.append(k); continue
        kp, nk = keys[i - 1], K.key('avg_prev', a=k, b=keys[i - 1])
        if not pred_path('test', 'boost', nk).exists():
            for scope, prev in (('val', pred_path('test', 'boost', kp)), ('test', pred_path('test', 'boost_next', kp))):
                cur = pd.read_parquet(pred_path(scope, 'boost', k))
                pv = pd.read_parquet(prev)[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': 'p_prev'})
                m = cur.merge(pv, on=['date', 'ticker'], how='left')
                m['rv_pred'] = np.where(m.p_prev.notna(), (m.rv_pred + m.p_prev) / 2, m.rv_pred)
                write_atomic(m.drop(columns='p_prev'), pred_path(scope, 'boost', nk))
        out.append(nk)
    return out
