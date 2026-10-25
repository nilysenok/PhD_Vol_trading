"""σ_pred(h=1) дизайном, зафиксированным владельцем 25.10 18:50: top5/V0_XGB/B-sm (XGBoost, среднее топ-5 точек, поправка Б0.1 с s² sm).
Только прогнозы: метрики на 2017-2026 не считаются, rv_actual тестовых файлов не читается.
Источник: общий data/regen/Bv2b_* (val) и data/regen/test_locked/Bv2b/ (test, только прогнозы).
Ноги по шагам: топ-5 (seed 42, ранги 1-5) есть только для шагов 1-6 (tune_Bv2.py: отбор на разработочном наборе); на шагах 7-9
конвейер замораживает одну точку (лучшая на шаге 6) - берётся она (seed 42). Шага 10 (тест 2026) в файлах нет.
Поправка (как для теста в gibrid_bias, оценка sm): тест шага k *= exp(s²_k/2), s²_k - дисперсия объединённых остатков ln(RV) - ln(прогноз)
val-лет шагов max(1, k-2)..k по прогнозам той же ноги, что на шаге k (val-строки с RV > 1e-12, как в gibrid_io.merged).
Единицы: sigma_pred_h1 = sqrt(RV-прогноза), дневная, без годовой нормировки - как sigma_h1 в vpred_aligned.parquet
(там sigma_h1 = sqrt(rv_pred) гибрида hybrid_adaptive_h1_annual, проверено: отношение 1,0000).
python3 scripts/analysis/sigma_pred_h1_v2.py [--regen <общий data/regen>]"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

KEY = ['date', 'ticker', 'step']


def top5(path, cols):
    d = pd.read_parquet(path, columns=cols + ['point', 'seed', 'rank'])
    d = d[(d.seed == 42) & (d['rank'] <= 5) & (d.step <= 6)]
    assert (d.groupby(KEY).size() == 5).all(), 'топ-5: не по 5 точек на строку'
    return d.groupby(KEY, as_index=False)[[c for c in cols if c not in KEY]].mean()


def single(path, cols):
    d = pd.read_parquet(path, columns=cols + ['seed'])
    return d[d.seed == 42].drop(columns='seed') if 'seed' in d else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regen', default='/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
    a = ap.parse_args()
    R = Path(a.regen); T = R / 'test_locked/Bv2b'
    vc, tc = KEY + ['rv_actual', 'rv_pred'], KEY + ['rv_pred']  # тест - без rv_actual
    val = {'top5': top5(R / 'Bv2b_xgb_h1_top5_val.parquet', vc), 'single': single(R / 'Bv2b_xgb_h1_val.parquet', vc)}
    test = {'top5': top5(T / 'Bv2b_xgb_h1_top5_test.parquet', tc), 'single': single(T / 'Bv2b_xgb_h1_test.parquet', tc)}
    out, log = [], []
    for k in range(1, 10):
        leg = 'top5' if k <= 6 else 'single'
        v = val[leg]; v = v[(v.step >= max(1, k - 2)) & (v.step <= k) & (v.rv_actual > 1e-12)]
        e = np.log(v.rv_actual.values + 1e-10) - np.log(v.rv_pred.values)
        s2 = float(np.var(e)); F = float(np.exp(s2 / 2))
        t = test[leg]; t = t[t.step == k]
        assert len(t) and t[['date', 'ticker']].duplicated().sum() == 0
        out.append(t[['date', 'ticker']].assign(sigma_pred_h1=np.sqrt(np.clip(t.rv_pred.values, 1e-10, None) * F)))
        log.append(dict(step=k, test_year=int(pd.to_datetime(t.date).dt.year.iloc[0]), leg=leg, val_steps=f'{max(1, k - 2)}-{k}', n_val=len(e), s2_sm=s2, F=F, n_test=len(t)))
    S = pd.concat(out).sort_values(['date', 'ticker']).reset_index(drop=True)
    assert not S.duplicated(['date', 'ticker']).any()
    S.to_parquet(R / 'sigma_pred_h1_v2.parquet', index=False)
    L = pd.DataFrame(log)
    L.to_csv(R / 'sigma_pred_h1_v2_steps.csv', index=False)
    print(L.to_string(index=False)); print(len(S), S.date.min(), S.date.max())


if __name__ == '__main__':
    main()
